#include <filesystem>
#include <vector>

#include <fmt/core.h>
#include <spdlog/spdlog.h>

#include "vehicle.hpp"

namespace plt = matplotlibcpp;

std::filesystem::path source_file_path(__FILE__);
std::filesystem::path vehicle_img_path =
    source_file_path.parent_path().parent_path() / "img" / "vehicle";
const std::vector<std::pair<std::string, std::string>> vehicle_show_config = {
    // {"#30A9DE", vehicle_img_path / "blue.png"},
    {"#0000FF", vehicle_img_path / "blue.mat.txt"},
    {"#E53A40", vehicle_img_path / "red.mat.txt"},
    {"#4CAF50", vehicle_img_path / "green.mat.txt"},
    {"#000000", vehicle_img_path / "black.mat.txt"},
    {"#FFFF00", vehicle_img_path / "yellow.mat.txt"},
    {"#00FFFF", vehicle_img_path / "cyan.mat.txt"},
    {"#FA58F4", vehicle_img_path / "purple.mat.txt"},
};

int Vehicle::global_vehicle_idx = 0;
PyObject *Vehicle::imshow_func = nullptr;

Vehicle::Vehicle(std::string _name, const YAML::Node &cfg)
    : VehicleBase(_name), planner(KLevelPlanner::get_instance(cfg))
{
  YAML::Node vehicle_info = cfg["vehicle_list"][_name];
  level = vehicle_info["level"].as<int>();
  init_x_min = vehicle_info["init"]["x"]["min"].as<double>();
  init_x_max = vehicle_info["init"]["x"]["max"].as<double>();
  init_y_min = vehicle_info["init"]["y"]["min"].as<double>();
  init_y_max = vehicle_info["init"]["y"]["max"].as<double>();
  init_v_min = vehicle_info["init"]["v"]["min"].as<double>();
  init_v_max = vehicle_info["init"]["v"]["max"].as<double>();
  init_yaw = vehicle_info["init"]["yaw"].as<double>();
  target.x = vehicle_info["target"]["x"].as<double>();
  target.y = vehicle_info["target"]["y"].as<double>();
  target.yaw = vehicle_info["target"]["yaw"].as<double>();
  vis_text_pos.x = vehicle_info["text"]["x"].as<double>();
  vis_text_pos.y = vehicle_info["text"]["y"].as<double>();
  is_ego = vehicle_info["is_ego"].as<bool>();

  spdlog::info(fmt::format("state:{},level:{}", is_ego, level));

  int local_loop_idx = Vehicle::global_vehicle_idx % vehicle_show_config.size();
  color = vehicle_show_config[local_loop_idx].first;
  std::string vehicle_pic_path = vehicle_show_config[local_loop_idx].second;
  outlook.data = utils::imread(vehicle_pic_path, outlook.rows, outlook.cols,
                               outlook.colors);

  vehicle_box2d = VehicleBase::get_box2d(state);
  safezone = VehicleBase::get_safezone(state);
  vehicle_tail_box2d = VehicleBase::get_tail_box2d(state);
  dt = cfg["delta_t"].as<double>();

  if (imshow_func == nullptr && Vehicle::global_vehicle_idx == 0)
  {
    Py_Initialize();
    std::filesystem::path source_file_path(__FILE__);
    std::filesystem::path project_path =
        source_file_path.parent_path().parent_path();
    std::string script_path = project_path / "scripts";
    PyRun_SimpleString("import sys");
    PyRun_SimpleString(
        fmt::format("sys.path.append('{}')", script_path).c_str());

    PyObject *py_name = PyUnicode_DecodeFSDefault("imshow");
    PyObject *py_module = PyImport_Import(py_name);
    Py_DECREF(py_name);
    if (py_module != nullptr)
    {
      imshow_func = PyObject_GetAttrString(py_module, "imshow");
    }
    if (imshow_func == nullptr || !PyCallable_Check(imshow_func))
    {
      spdlog::error("py.imshow call failed and the vehicle drawing will only "
                    "support linestyle");
      imshow_func = nullptr;
    }
  }
  ++Vehicle::global_vehicle_idx;

  reset();
}

void Vehicle::imshow(const Outlook &out, const State &state,
                     std::vector<double> para)
{
  std::vector<double> state_list{state.x, state.y, state.yaw};

  PyObject *vehicle_state = matplotlibcpp::detail::get_array(state_list);
  PyObject *vehicle_para = matplotlibcpp::detail::get_array(para);
  npy_intp dims[3] = {out.rows, out.cols, out.colors};

  const float *imptr = &(out.data[0]);

  PyObject *args = PyTuple_New(3);
  PyTuple_SetItem(args, 0,
                  PyArray_SimpleNewFromData(3, dims, NPY_FLOAT, (void *)imptr));
  PyTuple_SetItem(args, 1, vehicle_state);
  PyTuple_SetItem(args, 2, vehicle_para);

  PyObject *ret = PyObject_CallObject(imshow_func, args);

  Py_DECREF(args);
  if (ret)
  {
    Py_DECREF(ret);
  }
}

double GetAngleInInterval2PI(double angle)
{
  double angle2 = fmod(angle, 2 * M_PI);

  if (angle2 < 0)
  {
    angle2 += 2 * M_PI;
  }
  else if (std::signbit(angle2))
  {
    angle2 = 0;
  }

  return angle2;
}

void Vehicle::reset(void)
{
  footprint.clear();
  cur_action = Action::MAINTAIN;
  excepted_traj = StateList();
  have_got_target = false;
  episode_done = false;

  state.x = Random::uniform(init_x_min, init_x_max);
  state.y = Random::uniform(init_y_min, init_y_max);
  state.v = Random::uniform(init_v_min, init_v_max);
  state.is_ego = is_ego;
  state.yaw = init_yaw;
  if (state.is_ego)
  {
    // 牵引车头钩子车身系向量
    // SE_Vector v0(state.hitch_x, 0.0);
    // // 牵引车头钩子世界系向量
    // v0 = v0.Rotate(state.yaw) + SE_Vector(state.x, state.y);
    // // 计算车后方挂钩的向量 到 后方车厢的位置 向量v1
    // SE_Vector v1 = SE_Vector(state.tail_x, state.tail_y) - v0;
    // // 修改v1的长度 为 后方车厢后轴到挂接点的距离
    // // 计算后挂车中心位置在世界系的向量
    // v1.SetLength(state.couple_x);
    // // std::shared_ptr<State> new_trail_state =
    // //     std::make_shared<State>(*state.trail_state);
    // state.tail_x = v0.x() + v1.x();
    // state.tail_y = v0.y() + v1.y();
    // state.tail_yaw = GetAngleInInterval2PI(atan2(v1.y(), v1.x()) + M_PI);
    // spdlog::info(fmt::format("state.tail_x:{},state:{},level:{},name{}",
    //                          state.tail_x, state.is_ego, level, name));

    SE_Vector hitch_position(state.hitch_x, 0.0); // 相对于牵引车的挂钩位置
    hitch_position =
        hitch_position.Rotate(state.yaw);          // 将挂钩位置旋转到世界坐标系
    hitch_position += SE_Vector(state.x, state.y); // 加上牵引车的世界坐标

    // 计算挂车后轴的位置
    SE_Vector tail_position(state.couple_x, 0.0); // 相对于挂钩的挂车后轴位置
    tail_position =
        tail_position.Rotate(state.yaw + M_PI); // 按当前 yaw 旋转180度
    tail_position += hitch_position;            // 挂钩位置加上挂车后轴位置

    // 更新 state.tail_x 和 state.tail_y
    state.tail_x = tail_position.x();
    // spdlog::info(fmt::format(" vehicle:{} ,tail_x:{}", name, state.tail_x));
    state.tail_y = tail_position.y();
    SE_Vector v1 = SE_Vector(state.tail_x, state.tail_y) - hitch_position;
    state.tail_yaw = GetAngleInInterval2PI(atan2(v1.y(), v1.x()) + M_PI);
  }
  footprint.push_back(state);
}

void Vehicle::excute(void)
{
  // always remember old state for RL training
  State old_state = state;

  // vehicle dynamics: ego uses planner unless already reached target
  if (is_get_target())
  {
    have_got_target = true;
    state.v = 0;
    cur_action = Action::MAINTAIN;
    excepted_traj = StateList();
  }
  else if (is_ego)
  {
    std::pair<Action, StateList> act_and_traj = planner.planning(*this);
    cur_action = act_and_traj.first;
    excepted_traj = act_and_traj.second;
    state = utils::kinematic_propagate(state,
                                       utils::get_action_value(cur_action), dt);
  }
  else
  {
    // other vehicles move straight
    Eigen::Vector2d other_act;
    other_act << 0.0, 0.0;
    state = utils::kinematic_propagate(
        state,
        other_act,
        dt);
  }

  // ===== 4. RL 训练部分 =====
  if (planner.is_rl_mode() && is_ego)
  {
    auto state_vec =
        planner.encode_state_from_vehicle(old_state, tracked_objects);
    auto next_state_vec =
        planner.encode_state_from_vehicle(state, tracked_objects);
    double reward =
        planner.compute_reward(old_state, state, tracked_objects, target);
    bool done = planner.check_terminal(state, tracked_objects, target);
    bool returned_done = planner.train_agent(
        state_vec,
        static_cast<int>(cur_action),
        reward,
        next_state_vec,
        done);
    // communicate episode termination to the main loop
    this->episode_done = returned_done;
  }

  footprint.push_back(state);
}

std::string vectorToString(const std::vector<std::vector<double>> &vec)
{
  std::string result = "[";
  for (const auto &innerVec : vec)
  {
    result += "[";
    for (const auto &elem : innerVec)
    {
      result += std::to_string(elem) + ", ";
    }
    // 删除最后一个多余的 ", "
    if (!innerVec.empty())
    {
      result.pop_back();
      result.pop_back();
    }
    result += "], ";
  }
  // 删除最后一个多余的 ", "
  if (!vec.empty())
  {
    result.pop_back();
    result.pop_back();
  }
  result += "]";
  return result;
}

void Vehicle::draw_vehicle(std::string draw_style /* = "realistic"*/,
                           bool fill_mode /* = false */)
{
  if (draw_style == "realistic" && imshow_func != nullptr)
  {
    imshow(outlook, state, {length, width});
  }
  else
  {
    Eigen::Matrix<double, 2, 2, Eigen::RowMajor> head;
    Eigen::Matrix2d rot;
    head << 0.3 * VehicleBase::length, 0.3 * VehicleBase::length,
        VehicleBase::width / 2, -VehicleBase::width / 2;
    rot << cos(state.yaw), -sin(state.yaw), sin(state.yaw), cos(state.yaw);

    head = rot * head;
    head += Eigen::Vector2d(state.x, state.y).replicate(1, 2);

    // vehicle_box2d = VehicleBase::get_box2d(state);

    std::vector<std::vector<double>> tail_vec(2);
    std::vector<std::vector<double>> tractor_vec(2);
    std::vector<std::vector<double>> box2d_vec(2);
    if (state.is_ego)
    {
      vehicle_tail_box2d = VehicleBase::get_tail_box2d(state);
      vehicle_tractor_box2d = VehicleBase::get_tractor_box2d(state);
      tail_vec[0].assign(vehicle_tail_box2d.row(0).data(),
                         vehicle_tail_box2d.row(0).data() +
                             vehicle_tail_box2d.cols());
      tail_vec[1].assign(vehicle_tail_box2d.row(1).data(),
                         vehicle_tail_box2d.row(1).data() +
                             vehicle_tail_box2d.cols());
      tractor_vec[0].assign(vehicle_tractor_box2d.row(0).data(),
                            vehicle_tractor_box2d.row(0).data() +
                                vehicle_tractor_box2d.cols());
      tractor_vec[1].assign(vehicle_tractor_box2d.row(1).data(),
                            vehicle_tractor_box2d.row(1).data() +
                                vehicle_tractor_box2d.cols());
    }
    else
    {
      vehicle_box2d = VehicleBase::get_box2d(state);
      box2d_vec[0].assign(vehicle_box2d.row(0).data(),
                          vehicle_box2d.row(0).data() + vehicle_box2d.cols());
      box2d_vec[1].assign(vehicle_box2d.row(1).data(),
                          vehicle_box2d.row(1).data() + vehicle_box2d.cols());
    }
    // spdlog::info(
    //     fmt::format(" vehicle:{} is drawing,tail_x:{}", name, state.tail_x));

    std::vector<std::vector<double>> head_vec(2);

    // spdlog::info("Vector content: {}", vectorToString(tail_vec));

    head_vec[0].assign(head.row(0).data(), head.row(0).data() + head.cols());
    head_vec[1].assign(head.row(1).data(), head.row(1).data() + head.cols());

    if ((!fill_mode) && (state.is_ego))
    {
      plt::plot(tractor_vec[0], tractor_vec[1], color);
      // plt::plot(head_vec[0], head_vec[1], color);
      plt::plot(tail_vec[0], tail_vec[1], color);
    }
    else if (!(fill_mode && state.is_ego))
    {
      plt::plot(box2d_vec[0], box2d_vec[1], color);
      plt::plot(head_vec[0], head_vec[1], color);
    }
    else
    {
      plt::fill(box2d_vec[0], box2d_vec[1],
                {{"color", color}, {"alpha", "0.5"}});
      // plt::fill(tail_vec[0], tail_vec[1], {{"color", color}, {"alpha",
      // "0.5"}});
    }
  }
}

void VehicleList::reset(void)
{
  for (std::shared_ptr<Vehicle> &vehicle : vehicle_list)
  {
    spdlog::info(fmt::format("11state:{},name:{}", vehicle->state.is_ego,
                             vehicle->name));
    vehicle->reset();
  }
  update_track_objects();
}

bool VehicleList::is_all_get_target(void)
{
  bool all_get_target = std::all_of(vehicle_list.begin(), vehicle_list.end(),
                                    [](const std::shared_ptr<Vehicle> vehicle)
                                    {
                                      return vehicle->is_get_target();
                                    });

  return all_get_target;
}

bool VehicleList::is_any_collision(void)
{
  for (int i = 0; i < vehicle_list.size() - 1; ++i)
  {
    for (int j = i + 1; j < vehicle_list.size(); ++j)
    {
      if (utils::has_overlap(VehicleBase::get_box2d(vehicle_list[i]->state),
                             VehicleBase::get_box2d(vehicle_list[j]->state)))
      {
        return true;
      }
    }
  }

  return false;
}

void VehicleList::push_back(std::shared_ptr<Vehicle> vehicle)
{
  if (vehicle_names.count(vehicle->name) > 0)
  {
    spdlog::error(fmt::format(
        "vehicle name [{}] duplication is not acceptable !", vehicle->name));
    std::exit(EXIT_FAILURE);
  }
  else
  {
    vehicle_list.push_back(vehicle);
    vehicle_names.insert(vehicle->name);
  }
}

void VehicleList::pop_back(void)
{
  vehicle_names.erase(vehicle_list.back()->name);
  vehicle_list.pop_back();
}

std::shared_ptr<Vehicle> VehicleList::operator[](std::string name)
{
  for (std::shared_ptr<Vehicle> vehicle : vehicle_list)
  {
    if (vehicle->name == name)
    {
      return vehicle;
    }
  }

  return nullptr;
}

void VehicleList::set_track_objects(void)
{
  for (size_t i = 0; i < vehicle_list.size(); ++i)
  {
    for (size_t j = 0; j < vehicle_list.size(); ++j)
    {
      if (j != i)
      {
        TrackedObject track_object(vehicle_list[j]->name);
        track_object.state = vehicle_list[j]->state;
        track_object.target = vehicle_list[j]->target;
        vehicle_list[i]->tracked_objects.emplace_back(track_object);
      }
    }
  }
}

void VehicleList::update_track_objects(void)
{
  for (std::shared_ptr<Vehicle> &vehicle : vehicle_list)
  {
    for (TrackedObject &object : vehicle->tracked_objects)
    {
      object.state = (*this)[object.name]->state;
    }
  }
}

std::vector<VehicleBase> VehicleList::exclude(int ego_idx)
{
  std::shared_ptr<Vehicle> ego_vehicle = vehicle_list[ego_idx];
  return exclude(ego_vehicle);
}

std::vector<VehicleBase> VehicleList::exclude(std::shared_ptr<Vehicle> ego)
{
  std::vector<VehicleBase> exclude_list;
  for (std::shared_ptr<Vehicle> vehicle : vehicle_list)
  {
    if (vehicle->name != ego->name)
    {
      exclude_list.push_back(*vehicle);
    }
  }

  return exclude_list;
}
