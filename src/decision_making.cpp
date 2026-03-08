#include <cmath>
#include <filesystem>
#include <getopt.h>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>

#include <fmt/core.h>
#include <fmt/format.h>
#include <matplotlib-cpp/matplotlibcpp.h>
#include <spdlog/spdlog.h>
#include <sstream>
#include <yaml-cpp/yaml.h>
// #include "env.hpp"
#include "double_lane_env.hpp"
#include "planner.hpp"
#include "utils.hpp"
#include "vehicle.hpp"
#include "vehicle_base.hpp"

using std::string;
namespace plt = matplotlibcpp;
// 自定义 std::thread::id 格式化器
namespace fmt
{
  template <>
  struct formatter<std::thread::id>
  {
    // 解析格式化选项
    constexpr auto parse(format_parse_context &ctx)
    {
      // 这里只处理简单的格式化，没有格式选项
      auto it = ctx.begin();
      if (it != ctx.end() && *it != '}')
      {
        throw format_error("Invalid format");
      }
      return it;
    }

    // 格式化 std::thread::id
    template <typename FormatContext>
    auto format(const std::thread::id &id, FormatContext &ctx)
    {
      // 将 std::thread::id 转换为字符串
      std::ostringstream oss;
      oss << id;
      // 将字符串写入到格式化上下文
      return format_to(ctx.out(), "{}", oss.str());
    }
  };
} // namespace fmt

static struct option long_options[] = {
    {"rounds", required_argument, 0, 'r'},
    {"output_path", required_argument, 0, 'o'},
    {"log_level", required_argument, 0, 'l'},
    {"config", required_argument, 0, 'c'},
    {"no_animation", no_argument, 0, 'n'},
    {"save_fig", no_argument, 0, 'f'},
};

std::unordered_map<std::string, spdlog::level::level_enum> LOG_LEVEL_DICT = {
    {"trace", spdlog::level::trace}, {"debug", spdlog::level::debug}, {"info", spdlog::level::info}, {"warn", spdlog::level::warn}, {"err", spdlog::level::err}, {"critical", spdlog::level::critical}};

void run(int rounds_num, std::filesystem::path config_path,
         std::filesystem::path save_path, bool show_animation, bool save_fig)
{
  YAML::Node config;
  spdlog::info(fmt::format("config path: {}", config_path.string()));
  try
  {
    config = YAML::LoadFile(config_path.string());
    // spdlog::info(fmt::format("config parameters:\n{}", YAML::Dump(config)));
  }
  catch (const YAML::Exception &e)
  {
    spdlog::error(fmt::format("Error parsing YAML file: {}", e.what()));
    return;
  }

  // initialize
  double delta_t = config["delta_t"].as<double>();
  double max_simulation_time = config["max_simulation_time"].as<double>();
  double map_size = config["map_size"].as<double>();
  double lane_width = config["lane_width"].as<double>();
  bool is_show_predict_traj = config["is_show_predict_traj"].as<bool>();
  std::string vehicle_draw_style =
      config["vehicle_display_style"].as<std::string>();
  std::string ego_vehicle_name;
  if (config["ego_vehicle"])
  {
    ego_vehicle_name = config["ego_vehicle"].as<std::string>();
  }

  std::shared_ptr<EnvCrossroads> env =
      std::make_shared<EnvCrossroads>(map_size, lane_width);
  VehicleBase::initialize(env, 5, 2, 8, 2.4);
  MonteCarloTreeSearch::initialize(config);
  // Node::initialize(config["max_step"].as<int>(),
  //                  MonteCarloTreeSearch::calc_cur_value);

  VehicleList vehicles;
  for (const auto &yaml_node : config["vehicle_list"])
  {
    std::string vehicle_name = yaml_node.first.as<std::string>();
    spdlog::info(fmt::format(
        "================== vehicle_name{} ==================", vehicle_name));
    std::shared_ptr<Vehicle> vehicle =
        std::make_shared<Vehicle>(vehicle_name, config);
    // spdlog::info(
    //     fmt::format("state:{},name:{}", vehicle->state.is_ego,
    //     vehicle->name));
    vehicles.push_back(vehicle);
  }
  if (vehicles.size() < 1)
  {
    spdlog::error(
        "Please set the vehicles parameters in the configuration file !");
    return;
  }
  vehicles.set_track_objects();

  uint64_t succeed_count = 0;
  for (uint64_t iter = 0; iter < rounds_num; ++iter)
  {
    vehicles.reset();

    spdlog::info(
        fmt::format("================== Round {} ==================", iter));
    for (auto vehicle : vehicles)
    {
      spdlog::info(fmt::format(
          "{} >>> init_x: {:.2f}, init_y: {:.2f}, init_v: {:.2f},is_ego:{}",
          vehicle->name, vehicle->state.x, vehicle->state.y, vehicle->state.v,
          vehicle->state.is_ego));
    }

    double timestamp = 0.0;
    TicToc total_cost_time;
    bool whilebreak = false;
    while (true)
    {
      // todo-> 加一个判断角度的条件
      for (auto vehicle : vehicles)
      {
        if (vehicle->state.is_ego)
        {
          // only perform the original immediate target check for non-RL (MCTS) mode
          if (!vehicle->is_rl_mode())
          {
            if ((abs(vehicle->state.x - vehicle->target.x) < 1 &&
                 abs(vehicle->state.y - vehicle->target.y) < 2) ||
                (abs(vehicle->state.x - vehicle->target.x) < 1 &&
                 abs(vehicle->state.tail_x - vehicle->target.x) < 1))
            {
              spdlog::info(fmt::format("Round {:d} successed, simulation time: "
                                       "{:.3f} s, actual timecost: {:.3f} s",
                                       iter, timestamp, total_cost_time.toc()));
              ++succeed_count;
              whilebreak = true;
              break;
            }
          }
          // In RL mode, episode termination is signaled from inside the per-vehicle
          // thread (via planner.train_agent -> agent_->observe). We check that after
          // the vehicle threads have executed (below) so we don't duplicate logic here.
        }
      }
      if (whilebreak)
      {
        break;
      }

      // if (vehicles.is_all_get_target()) {
      //   spdlog::info(fmt::format("Round {:d} successed, simulation time: "
      //                            "{:.3f} s, actual timecost: {:.3f} s",
      //                            iter, timestamp, total_cost_time.toc()));
      //   ++succeed_count;
      //   break;
      // }
      // Update planner with current simulation time so per-vehicle checks can use it
      KLevelPlanner::set_simulation_times(timestamp, max_simulation_time);

      TicToc iter_cost_time;
      std::vector<std::thread> threads;
      for (std::shared_ptr<Vehicle> &vehicle : vehicles)
      {
        // 这里输出的vehicle信息是什么

        std::thread thread([&vehicle]()
                           {
          std::thread::id this_id = std::this_thread::get_id();
          // spdlog::info(fmt::format("thread_idddd:{}, actually_ego:{}",
          // this_id,
          //                          vehicle->name));
          vehicle->excute(); });
        threads.emplace_back(std::move(thread));
      }

      for (auto &thread : threads)
      {
        if (thread.joinable())
        {
          thread.join();
        }
      }

      vehicles.update_track_objects();
      // After vehicle threads run, check if any RL-driven vehicle reported episode termination
      for (const std::shared_ptr<Vehicle> &vehicle : vehicles)
      {
        if (vehicle->episode_done)
        {
          // decide success vs non-success by distance-to-target
          double dx = vehicle->target.x - vehicle->state.x;
          double dy = vehicle->target.y - vehicle->state.y;
          bool success = (std::sqrt(dx * dx + dy * dy) < 1.0);
          if (success)
          {
            spdlog::info(fmt::format("Round {:d} successed (RL episode_end), simulation time: {:.3f} s, actual timecost: {:.3f} s",
                                     iter, timestamp, total_cost_time.toc()));
            ++succeed_count;
          }
          else
          {
            spdlog::info(fmt::format("Round {:d} ended (RL episode_end) without reaching target, simulation time: {:.3f} s, actual timecost: {:.3f} s",
                                     iter, timestamp, total_cost_time.toc()));
          }
          whilebreak = true;
          break;
        }
      }
      spdlog::debug(fmt::format("simulation time {:.3f} step cost {:.3f} sec",
                                timestamp, iter_cost_time.toc()));

      // If an RL-driven vehicle reported episode termination, exit the
      // outer simulation loop immediately to avoid treating the same
      // round as a failure due to later collision/timeout checks.
      if (whilebreak)
      {
        break;
      }

      // Check if this is RL mode by looking at ego vehicle
      bool is_rl_mode_round = false;
      for (const auto &vehicle : vehicles)
      {
        if (vehicle->state.is_ego && vehicle->is_rl_mode())
        {
          is_rl_mode_round = true;
          break;
        }
      }

      // In MCTS mode, collision/timeout ends the round immediately.
      // In RL mode, let the agent's check_terminal/observe(done) signal the end.
      if (!is_rl_mode_round)
      {
        if (vehicles.is_any_collision() || timestamp > max_simulation_time)
        {
          spdlog::info(fmt::format("Round {:d} failed, simulation time: {:.3f} "
                                   "s, actual timecost: {:.3f} s",
                                   iter, timestamp, total_cost_time.toc()));
          break;
        }
      }

      if (show_animation)
      {
        plt::cla();
        env->draw_env();
        for (const std::shared_ptr<Vehicle> &vehicle : vehicles)
        {
          // spdlog::info(fmt::format("state:{},name:{}", vehicle->state.is_ego,
          //                          vehicle->name));
          auto excepted_traj = vehicle->excepted_traj.to_vector();
          vehicle->draw_vehicle(vehicle_draw_style);
          plt::plot({vehicle->target.x}, {vehicle->target.y},
                    {{"marker", "x"}, {"color", vehicle->color}});
          plt::plot(excepted_traj[0], excepted_traj[1],
                    {{"color", vehicle->color}, {"linewidth", "1"}});
          plt::text(vehicle->vis_text_pos.x, vehicle->vis_text_pos.y + 8,
                    fmt::format("level {:d}", vehicle->level),
                    {{"color", vehicle->color}, {"fontsize", "22"}});
          plt::text(vehicle->vis_text_pos.x, vehicle->vis_text_pos.y,
                    fmt::format("v = {:.2f} m/s", vehicle->state.v),
                    {{"color", vehicle->color}, {"fontsize", "22"}});
          plt::text(
              vehicle->vis_text_pos.x, vehicle->vis_text_pos.y - 8,
              fmt::format("{}", utils::get_action_name(vehicle->cur_action)),
              {{"color", vehicle->color}, {"fontsize", "22"}});
        }
        if (is_show_predict_traj)
        {
          if (ego_vehicle_name.empty())
          {
            ego_vehicle_name = vehicles[0]->name;
            spdlog::warn("ego_vehicle parameter in yaml is none, defualt: " +
                         ego_vehicle_name);
          }

          for (const TrackedObject &obj :
               vehicles[ego_vehicle_name]->tracked_objects)
          {
            for (const PredictTraj &predict_traj : obj.predict_trajs)
            {
              double belief = predict_traj.confidence;
              std::vector<std::vector<double>> prediction =
                  predict_traj.traj.to_vector();
              plt::plot(prediction[0], prediction[1],
                        {{"color", "gray"}, {"linewidth", "1"}});
              plt::text(prediction[0].back(), prediction[1].back(),
                        fmt::format("{:.2f}", belief), {{"color", "gray"}});
            }
          }
        }
        plt::xlim(-map_size, map_size);
        plt::ylim(-map_size, map_size);
        plt::title(fmt::format("Round {} / {}", iter + 1, rounds_num));
        plt::set_aspect_equal();
        plt::pause(0.01);
      }
      timestamp += delta_t;
    }

    if (show_animation || save_fig)
    {
      plt::clf();
      env->draw_env();
      for (std::shared_ptr<Vehicle> &vehicle : vehicles)
      {
        for (const State &state : vehicle->footprint)
        {
          vehicle->state = state;
          vehicle->draw_vehicle(vehicle_draw_style, true);
        }
        plt::text(vehicle->vis_text_pos.x, vehicle->vis_text_pos.y + 3,
                  fmt::format("level {:d}", vehicle->level),
                  {{"color", vehicle->color}});
      }
      plt::xlim(-map_size, map_size);
      plt::ylim(-map_size, map_size);
      plt::title(fmt::format("Round {} / {}", iter + 1, rounds_num));
      plt::set_aspect_equal();
      if (show_animation)
      {
        plt::pause(1);
      }
      if (save_fig)
      {
        plt::save(
            (save_path / ("Round_" + std::to_string(iter) + ".svg")).string(),
            600);
      }
    }
  }

  double succeed_rate = 100 * succeed_count / rounds_num;
  spdlog::info("\n=========================================");

  spdlog::info(fmt::format("Experiment success {}/{}({:.2f}%) rounds.",
                           succeed_count, rounds_num, succeed_rate));
}

int main(int argc, char **argv)
{
  std::filesystem::path source_file_path(__FILE__);
  std::filesystem::path project_path =
      source_file_path.parent_path().parent_path();

  int rounds_num = 1000;
  std::filesystem::path output_path = project_path / "logs";
  std::filesystem::path config_path =
      project_path / "config" / "triple_interact.yaml";
  bool show_animation = false;
  bool save_flag = false;
  std::string log_level = "info"; // info

  int opt, option_index = 0;
  while ((opt = getopt_long(argc, argv, "r:o:l:c:n:f:", long_options,
                            &option_index)) != -1)
  {
    switch (opt)
    {
    case 'r':
      rounds_num = std::stoi(optarg);
      break;
    case 'o':
      output_path = utils::absolute_path(optarg);
      break;
    case 'l':
      log_level = std::string(optarg);
      break;
    case 'c':
      config_path = utils::absolute_path(optarg);
      break;
    case 'n':
      show_animation = false;
      break;
    case 'f':
      save_flag = true;
      break;
    default:
      exit(EXIT_FAILURE);
    }
  }

  spdlog::set_level(LOG_LEVEL_DICT[log_level]);
  spdlog::set_pattern("%Y-%m-%d %H:%M:%S.%e - %^%l%$ - %v");
  spdlog::info("log level : " + log_level);

  // output path
  if (save_flag)
  {
    auto now = std::chrono::system_clock::now();
    std::time_t now_time_t = std::chrono::system_clock::to_time_t(now);
    struct tm now_tm;
    localtime_r(&now_time_t, &now_tm);
    std::stringstream ss;
    ss << std::put_time(&now_tm, "%Y-%m-%d-%H-%M-%S");
    output_path = output_path / ss.str();
    if (!std::filesystem::exists(output_path))
    {
      std::filesystem::create_directories(output_path);
    }
  }

  run(rounds_num, config_path, output_path, show_animation, save_flag);

  return 0;
}
