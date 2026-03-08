#include "planner.hpp"
#include "action_evaluator.hpp"
#include "DQNEvaluator.hpp"
#include "MCTSEvaluator.hpp"
#include "dqn_agent.hpp"
#include <fmt/core.h>
#include <spdlog/spdlog.h>
#include <thread>
#include <unordered_set>

// initialize static time variables
double KLevelPlanner::current_simulation_time = 0.0;
double KLevelPlanner::max_simulation_time = 0.0;

#include <fmt/format.h>
#include <sstream>

constexpr double TWO_PI = M_PI * 2;

// std::string actually_ego = " ";
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

double MonteCarloTreeSearch::EXPLORATE_RATE = 1 / (2 * sqrt(2.0));
double MonteCarloTreeSearch::LAMDA = 0.9;
double MonteCarloTreeSearch::WEIGHT_AVOID = 10;
// 原0.2，暂时改为0
double MonteCarloTreeSearch::WEIGHT_SAFE = 0;
// 老会出现上马路牙子的情况，原2，增加为10，
double MonteCarloTreeSearch::WEIGHT_OFFROAD = 10;
// 原-1，现在改为0
double MonteCarloTreeSearch::WEIGHT_DIRECTION = 0;
// 原0.1，改为10，它应该是权重大小仅次于碰撞
double MonteCarloTreeSearch::WEIGHT_DISTANCE = 10;
double MonteCarloTreeSearch::WEIGHT_VELOCITY = 0.05;

MonteCarloTreeSearch::MonteCarloTreeSearch(
    const std::vector<StateList> &other_traj, const YAML::Node &cfg)
{
  computation_budget = cfg["computation_budget"].as<uint64_t>();
  dt = cfg["delta_t"].as<double>();

  for (int i = 0; i < Node::MAX_LEVEL + 1; ++i)
  {
    StateList other_states;
    for (const StateList states : other_traj)
    {
      other_states.push_back(states[i]);
    }
    other_predict_traj.emplace_back(other_states);
  }
}

double MonteCarloTreeSearch::calc_cur_value(std::shared_ptr<Node> node,
                                            double last_node_value)
{
  double x = node->state.x;
  double y = node->state.y;
  double yaw = node->state.yaw;
  double velocity = node->state.v;
  bool is_ego = node->state.is_ego;
  int step = node->cur_level;

  Eigen::Matrix<double, 2, 5> ego_box2d = VehicleBase::get_box2d(node->state);
  Eigen::Matrix<double, 2, 5> ego_tail_box2d;
  Eigen::Matrix<double, 2, 5> ego_tractor_box2d;
  if (is_ego)
  {
    ego_tractor_box2d = VehicleBase::get_tractor_box2d(node->state);
    ego_tail_box2d = VehicleBase::get_tail_box2d(node->state);
  }
  double avoid = 1.0;
  // double avoid_trailer = 1.0;

  for (auto &cur_other_state : node->other_agent_state)
  {
    if (!(is_ego && cur_other_state.is_ego))
    {
      if (utils::has_overlap(ego_box2d,
                             VehicleBase::get_box2d(cur_other_state)))
      {
        avoid = 0;
        break;
      }
    }
    if ((!is_ego) && cur_other_state.is_ego)
    {
      if (utils::has_overlap(ego_box2d,
                             VehicleBase::get_tail_box2d(cur_other_state)) ||
          utils::has_overlap(ego_box2d,
                             VehicleBase::get_tractor_box2d(cur_other_state)))
      {
        avoid = 0;
        break;
      }
    }
    else if (is_ego && (!cur_other_state.is_ego))
    {
      if (utils::has_overlap(ego_tail_box2d,
                             VehicleBase::get_box2d(cur_other_state)) ||
          utils::has_overlap(ego_tractor_box2d,
                             VehicleBase::get_box2d(cur_other_state)))
      {
        avoid = 0;
        break;
      }
    }
  }
  // double avoid = avoid_tractor * avoid_trailer;
  // int avoid = 1;
  // int safe = 0;
  // for (auto &cur_other_state : node->other_agent_state) {
  //   if (utils::has_overlap(ego_box2d,
  //                          VehicleBase::get_box2d(cur_other_state))) {
  //     avoid = 0;
  //   }
  //   if (utils::has_overlap(ego_safezone,
  //                          VehicleBase::get_safezone(cur_other_state))) {
  //     safe = -1;
  //   }
  // }
  Eigen::Matrix<double, 2, 5> ego_safezone =
      VehicleBase::get_safezone(node->state);
  Eigen::Matrix<double, 2, 5> safe_tail_box2d;
  Eigen::Matrix<double, 2, 5> safe_tractor_box2d;

  if (is_ego)
  {
    safe_tractor_box2d = VehicleBase::get_tractor_safe_box2d(node->state);
    safe_tail_box2d = VehicleBase::get_tail_safe_box2d(node->state);
  }

  double safe = 1.0;

  for (auto &cur_other_state : node->other_agent_state)
  {
    if (!(is_ego && cur_other_state.is_ego))
    {
      if (utils::has_overlap(ego_safezone,
                             VehicleBase::get_safezone(cur_other_state)))
      {
        safe = 0;
        break;
      }
    }
    if ((!is_ego) && cur_other_state.is_ego)
    {
      if (utils::has_overlap(ego_safezone, VehicleBase::get_tail_safe_box2d(
                                               cur_other_state)) ||
          utils::has_overlap(ego_safezone, VehicleBase::get_tractor_safe_box2d(
                                               cur_other_state)))
      {
        safe = 0;
        break;
      }
    }
    else if (is_ego && (!cur_other_state.is_ego))
    {
      if (utils::has_overlap(safe_tail_box2d,
                             VehicleBase::get_safezone(cur_other_state)) ||
          utils::has_overlap(safe_tractor_box2d,
                             VehicleBase::get_safezone(cur_other_state)))
      {
        safe = 0;
        break;
      }
    }
  }

  int offroad = 0;
  if (!is_ego)
  {
    for (auto &rect : VehicleBase::env->rect_mat)
    {
      if (utils::has_overlap(ego_box2d, rect))
      {
        offroad = -1;
        break;
      }
    }
  }
  else
  {
    for (auto &rect : VehicleBase::env->rect_mat)
    {
      if (utils::has_overlap(ego_tractor_box2d, rect) ||
          utils::has_overlap(ego_tail_box2d, rect))
      {
        offroad = -1;
        break;
      }
    }
  }

  // int direction = 0;
  // if (MonteCarloTreeSearch::is_opposite_direction(node->state,
  //                                                 ego_tractor_box2d)) {
  //   direction = -1;
  // }

  double delta_yaw = std::fmod(abs(yaw - node->goal_pose.yaw), TWO_PI);
  delta_yaw = std::min(delta_yaw, TWO_PI - delta_yaw);
  double d = (x - node->goal_pose.x) * (x - node->goal_pose.x) +
             (y - node->goal_pose.y) * (y - node->goal_pose.y);
  // double distance =
  //     1 - std::tanh((x - node->goal_pose.x) * (x - node->goal_pose.x) +
  //                   (y - node->goal_pose.y) * (y - node->goal_pose.y));
  double distance = -std::log1p(d) / std::log(10);
  // spdlog::info(fmt::format("distance: {:.7f}", distance));
  // double distance = -(abs(x - node->goal_pose.x) + abs(y - node->goal_pose.y)
  // +
  //                     1.5 * delta_yaw);
  // 不要到终点的距离了，只算横向距离
  double l = (x - node->goal_pose.x) * (x - node->goal_pose.x);
  // double lateral_distance =
  //     1 - std::tanh((x - node->goal_pose.x) * (x - node->goal_pose.x));
  double lateral_distance = -std::log1p(l) / std::log(10);
  // double distance = -(abs(x - node->goal_pose.x) + 1.5 * delta_yaw);
  // if (avoid == 1) {
  //   spdlog::info(fmt::format("avoid: {}", avoid));
  // }
  // if (!node || !node->parent) {
  //   spdlog::error("Node or parent node is null.");
  // }
  double acc_current = node->state.acc;
  double acc_last;
  if (!node->parent)
  {
    // spdlog::error("Node or parent node is null.");
    acc_last = acc_current;
  }
  else
  {
    std::shared_ptr<Node> last_node = node->parent;
    acc_last = last_node->state.acc;
  }
  double ride_exp =
      1 - std::tanh(acc_current * acc_current +
                    (acc_last - acc_current) * (acc_last - acc_current));
  double final_avoid = 1000 * avoid;
  double final_lateral = 700 * lateral_distance;
  double final_offroad = 300 * offroad;
  double final_distance = 600 * distance;
  double final_safe = 100 * safe;
  // double final_direction = MonteCarloTreeSearch::WEIGHT_DIRECTION *
  // direction;
  double final_velocity = MonteCarloTreeSearch::WEIGHT_VELOCITY * velocity;
  double final_ride = 1 * ride_exp;

  double cur_reward = final_avoid + final_lateral + final_offroad +
                      final_distance + final_ride + final_safe;

  spdlog::info(fmt::format("cur_reward:{},final_avoid: "
                           "{},final_lateral: {},final_offroad: "
                           "{},final_distance: {} "
                           ",final_velocity:{},ride_exp: {}, final_safe: {} ",
                           cur_reward, final_avoid, final_lateral,
                           final_offroad, final_distance, final_velocity,
                           ride_exp, final_safe));

  // spdlog::info(fmt::format("cur_reward:{:.2f},final_avoid: "
  //                          "{:.2f},final_lateral: {:.2f},final_offroad: "
  //                          "{:.2f},final_distance: {:.2f},final_direction: "
  //                          "{:.2f},final_velocity: {:.2f}",
  //                          cur_reward, final_avoid, final_lateral,
  //                          final_offroad, final_distance, final_direction,
  //                          final_velocity));
  double total_reward =
      last_node_value +
      pow(MonteCarloTreeSearch::LAMDA, (step - 1)) * cur_reward;
  node->value = total_reward;
  // spdlog::info(fmt::format("current was used"));
  return total_reward;
}

double MonteCarloTreeSearch::other_calc_cur_value(std::shared_ptr<Node> node,
                                                  double last_node_value)
{
  double x = node->state.x;
  double y = node->state.y;
  double yaw = node->state.yaw;
  double velocity = node->state.v;
  bool is_ego = node->state.is_ego;
  int step = node->cur_level;

  Eigen::Matrix<double, 2, 5> ego_box2d = VehicleBase::get_box2d(node->state);
  Eigen::Matrix<double, 2, 5> ego_tail_box2d;
  Eigen::Matrix<double, 2, 5> ego_tractor_box2d;
  if (is_ego)
  {
    ego_tractor_box2d = VehicleBase::get_tractor_box2d(node->state);
    ego_tail_box2d = VehicleBase::get_tail_box2d(node->state);
  }
  double avoid = 1.0;
  // double avoid_trailer = 1.0;

  for (auto &cur_other_state : node->other_agent_state)
  {
    if (!(is_ego && cur_other_state.is_ego))
    {
      if (utils::has_overlap(ego_box2d,
                             VehicleBase::get_box2d(cur_other_state)))
      {
        avoid = 0;
        break;
      }
    }
    if ((!is_ego) && cur_other_state.is_ego)
    {
      if (utils::has_overlap(ego_box2d,
                             VehicleBase::get_tail_box2d(cur_other_state)) ||
          utils::has_overlap(ego_box2d,
                             VehicleBase::get_tractor_box2d(cur_other_state)))
      {
        avoid = 0;
        break;
      }
    }
    else if (is_ego && (!cur_other_state.is_ego))
    {
      if (utils::has_overlap(ego_tail_box2d,
                             VehicleBase::get_box2d(cur_other_state)) ||
          utils::has_overlap(ego_tractor_box2d,
                             VehicleBase::get_box2d(cur_other_state)))
      {
        avoid = 0;
        break;
      }
    }
  }
  // int avoid = 1;
  // int safe = 0;
  // for (auto &cur_other_state : node->other_agent_state) {
  //   if (utils::has_overlap(ego_box2d,
  //                          VehicleBase::get_box2d(cur_other_state))) {
  //     avoid = 0;
  //   }
  //   if (utils::has_overlap(ego_safezone,
  //                          VehicleBase::get_safezone(cur_other_state))) {
  //     safe = -1;
  //   }
  // }
  Eigen::Matrix<double, 2, 5> ego_safezone =
      VehicleBase::get_safezone(node->state);
  Eigen::Matrix<double, 2, 5> safe_tail_box2d;
  Eigen::Matrix<double, 2, 5> safe_tractor_box2d;

  if (is_ego)
  {
    safe_tractor_box2d = VehicleBase::get_tractor_safe_box2d(node->state);
    safe_tail_box2d = VehicleBase::get_tail_safe_box2d(node->state);
  }

  double safe = 1.0;

  for (auto &cur_other_state : node->other_agent_state)
  {
    if (!(is_ego && cur_other_state.is_ego))
    {
      if (utils::has_overlap(ego_safezone,
                             VehicleBase::get_safezone(cur_other_state)))
      {
        safe = 0;
        break;
      }
    }
    if ((!is_ego) && cur_other_state.is_ego)
    {
      if (utils::has_overlap(ego_safezone, VehicleBase::get_tail_safe_box2d(
                                               cur_other_state)) ||
          utils::has_overlap(ego_safezone, VehicleBase::get_tractor_safe_box2d(
                                               cur_other_state)))
      {
        safe = 0;
        break;
      }
    }
    else if (is_ego && (!cur_other_state.is_ego))
    {
      if (utils::has_overlap(safe_tail_box2d,
                             VehicleBase::get_safezone(cur_other_state)) ||
          utils::has_overlap(safe_tractor_box2d,
                             VehicleBase::get_safezone(cur_other_state)))
      {
        safe = 0;
        break;
      }
    }
  }

  int offroad = 0;
  if (!is_ego)
  {
    for (auto &rect : VehicleBase::env->rect_mat)
    {
      if (utils::has_overlap(ego_box2d, rect))
      {
        offroad = -1;
        break;
      }
    }
  }
  else
  {
    for (auto &rect : VehicleBase::env->rect_mat)
    {
      if (utils::has_overlap(ego_tractor_box2d, rect) ||
          utils::has_overlap(ego_tail_box2d, rect))
      {
        offroad = -1;
        break;
      }
    }
  }

  // int direction = 0;
  // if (MonteCarloTreeSearch::is_opposite_direction(node->state,
  //                                                 ego_tractor_box2d)) {
  //   direction = -1;
  // }

  double delta_yaw = std::fmod(abs(yaw - node->goal_pose.yaw), TWO_PI);
  delta_yaw = std::min(delta_yaw, TWO_PI - delta_yaw);
  // double distance = -(abs(x - node->goal_pose.x) + abs(y - node->goal_pose.y)
  // +
  //                     1.5 * delta_yaw);
  // 不要到终点的距离了，只算横向距离
  // double distance = -(abs(x - node->goal_pose.x) + 1.5 * delta_yaw);

  double l = (x - node->goal_pose.x) * (x - node->goal_pose.x);
  // double lateral_distance =
  //     1 - std::tanh((x - node->goal_pose.x) * (x - node->goal_pose.x));
  double lateral_distance = -std::log1p(l) / std::log(10);
  double final_avoid = 1000 * avoid;
  double final_lateral = 700 * lateral_distance;
  double final_offroad = 300 * offroad;
  double final_safe = 100 * safe;
  // double final_distance = 100 * distance;
  // double final_direction = MonteCarloTreeSearch::WEIGHT_DIRECTION *
  // direction;
  double final_velocity = MonteCarloTreeSearch::WEIGHT_VELOCITY * velocity;
  // double final_ride = 1 * ride_exp;

  double cur_reward = final_avoid + final_lateral + final_offroad + final_safe;

  // double cur_reward = MonteCarloTreeSearch::WEIGHT_AVOID * avoid +
  //                     MonteCarloTreeSearch::WEIGHT_SAFE * safe +
  //                     MonteCarloTreeSearch::WEIGHT_OFFROAD * offroad +
  //                     1 * lateral_distance;
  double total_reward =
      last_node_value +
      pow(MonteCarloTreeSearch::LAMDA, (step - 1)) * cur_reward;
  node->value = total_reward;
  // spdlog::info(fmt::format("other was used"));
  return total_reward;
}

bool MonteCarloTreeSearch::is_opposite_direction(State pos,
                                                 Eigen::MatrixXd ego_box2d)
{
  double x = pos.x;
  double y = pos.y;
  double yaw = pos.yaw;

  for (auto laneline : VehicleBase::env->laneline_mat)
  {
    if (utils::has_overlap(ego_box2d, laneline))
    {
      return true;
    }
  }

  double lanewidth = VehicleBase::env->lanewidth;
  if (x > -lanewidth && x < 0 && (y < -lanewidth || y > lanewidth))
  {
    // down lane
    if (yaw > 0 && yaw < M_PI)
    {
      return true;
    }
  }
  else if (x > 0 && x < lanewidth && (y < -lanewidth || y > lanewidth))
  {
    // up lane
    if (!(yaw > 0 && yaw < M_PI))
    {
      return true;
    }
  }
  else if (y > -lanewidth && y < 0 && (x < -lanewidth || x > lanewidth))
  {
    // right lane
    if (yaw > M_PI_2 && yaw < 3 * M_PI_2)
    {
      return true;
    }
  }
  else if (y > 0 && y < lanewidth && (x < -lanewidth || x > lanewidth))
  {
    // left lane
    if (!(yaw > M_PI_2 && yaw < 3 * M_PI_2))
    {
      return true;
    }
  }

  return false;
}

std::shared_ptr<Node> MonteCarloTreeSearch::excute(std::shared_ptr<Node> root)
{
  for (uint64_t iter = 0; iter < computation_budget; ++iter)
  {
    // 1. Find the best node to expand
    std::shared_ptr<Node> expand_node = tree_policy(root);
    // 2. Random run to add node and get reward
    double reward = default_policy(expand_node);
    // 3. Update all passing nodes with reward
    update(expand_node, reward);
  }

  return get_best_child(root, 0);
}

std::shared_ptr<Node>
MonteCarloTreeSearch::tree_policy(std::shared_ptr<Node> node)
{
  while (node->is_terminal() == false)
  {
    if (node->children.empty())
    {
      return expand(node);
    }
    else if (Random::uniform(0.0, 1.0) < 0.5)
    {
      node = get_best_child(node, MonteCarloTreeSearch::EXPLORATE_RATE);
    }
    else
    {
      if (node->is_fully_expanded() == false)
      {
        return expand(node);
      }
      else
      {
        node = get_best_child(node, MonteCarloTreeSearch::EXPLORATE_RATE);
      }
    }
  }
  return node;
}

std::shared_ptr<Node> MonteCarloTreeSearch::expand(std::shared_ptr<Node> node)
{
  std::unordered_set<Action> tried_actions;
  for (auto child : node->children)
  {
    tried_actions.insert(child->action);
  }

  Action next_action = Random::choice(ACTION_LIST);
  while (!node->is_terminal() && tried_actions.count(next_action))
  {
    next_action = Random::choice(ACTION_LIST);
  }
  StateList other_states = other_predict_traj[node->cur_level + 1];
  node->add_child(next_action, dt, other_states);

  return node->children.back();
}

std::shared_ptr<Node>
MonteCarloTreeSearch::get_best_child(std::shared_ptr<Node> node,
                                     double scalar)
{
  double best_score = -INFINITY;
  std::vector<std::shared_ptr<Node>> best_children;

  for (auto child : node->children)
  {
    double exploit = child->reward / child->visits;
    double explore = sqrt(2 * log(node->visits) / child->visits);
    double score = exploit + scalar + explore;
    if (score == best_score)
    {
      best_children.push_back(child);
    }
    else if (score > best_score)
    {
      best_children = {child};
      best_score = score;
    }
  }
  if (best_children.empty())
  {
    return node;
  }

  return Random::choice(best_children);
}

double MonteCarloTreeSearch::default_policy(std::shared_ptr<Node> node)
{
  while (!node->is_terminal())
  {
    StateList other_states = other_predict_traj[node->cur_level + 1];
    std::shared_ptr<Node> next_node = node->next_node(dt, other_states);
    node = next_node;
  }

  return node->value;
}

void MonteCarloTreeSearch::update(std::shared_ptr<Node> node, double r)
{
  while (node != nullptr)
  {
    node->visits += 1;
    node->reward += r;
    node = node->parent;
  }
}

std::vector<double> KLevelPlanner::encode_state(
    const VehicleBase &ego,
    const std::vector<TrackedObject> &others)
{
  std::vector<double> s;

  // ego
  s.push_back(ego.state.x);
  s.push_back(ego.state.y);
  s.push_back(ego.state.v);

  s.push_back(ego.target.x - ego.state.x);
  s.push_back(ego.target.y - ego.state.y);

  // others（取最近 N 个）
  for (int i = 0; i < max_vehicle; ++i)
  {
    if (i < others.size())
    {
      s.push_back(others[i].state.x - ego.state.x);
      s.push_back(others[i].state.y - ego.state.y);
      s.push_back(others[i].state.v - ego.state.v);
    }
    else
    {
      s.push_back(0);
      s.push_back(0);
      s.push_back(0);
    }
  }

  return s;
}

std::pair<Action, StateList> KLevelPlanner::planning(VehicleBase &ego)
{
  bool use_mcts = false;
  if (use_mcts)
  {
    std::vector<VehicleBase> others;
    // spdlog::info(fmt::format("actually_ego:{}", ego.name));
    // actually_ego = ego.name;
    // std::thread::id this_id = std::this_thread::get_id();
    std::string initical_ego_name = ego.name;
    for (const TrackedObject &obj : ego.tracked_objects)
    {
      VehicleBase other(obj.name);
      other.state = obj.state;
      other.target = obj.target;
      other.have_got_target = other.is_get_target();
      others.emplace_back(other);
      // spdlog::info(fmt::format("pid:{},actually_ego:{},trackedcar:{}", this_id,
      //                          ego.name, other.name));
    }

    std::vector<StateList> other_prediction =
        get_prediction(ego, others, initical_ego_name);

    for (size_t i = 0; i < others.size(); ++i)
    {
      PredictTraj predict_traj{1.0, other_prediction[i]};
      ego.tracked_objects[i].predict_trajs.clear();
      ego.tracked_objects[i].predict_trajs.emplace_back(predict_traj);
    }

    std::pair<std::vector<Action>, StateList> ret =
        forward_simulate(ego, other_prediction, initical_ego_name);

    return std::make_pair(ret.first[0], ret.second);
  }
  else
  {

    // std::shared_ptr<DQNAgent> agent = std::make_shared<DQNAgent>(state_dim, action_dim);
    double dt = 0.5;
    // 这里是RL的state，和物理状态的state不一样，物理状态是(x,y,v,acc,...)，RL的state是经过编码的状态向量，它包含了ego和其他车的状态信息，经过归一化和差分处理，以便于RL模型的输入
    auto state_vec = encode_state(ego, ego.tracked_objects);
    //  推理 Q(s, a)
    int action_id = agent_->select_action(state_vec);
    Action action = static_cast<Action>(action_id);

    // 这里输入用的sate是物理状态的state
    // ego.state = utils::kinematic_propagate(
    //     ego.state,
    //     utils::get_action_value(action),
    //     dt);
    // for (auto &obj : ego.tracked_objects)
    // {
    //   Eigen::Vector2d other_act;
    //   other_act << 0.0, 0.0; // 匀速直行

    //   obj.state = utils::kinematic_propagate(
    //       obj.state,
    //       other_act,
    //       dt);
    // }
    // auto next_state_vec = encode_state(ego, ego.tracked_objects);
    // double reward = compute_reward(ego);
    // bool done = check_terminal(ego);

    // agent_->observe(state_vec,
    //                 action_id,
    //                 reward,
    //                 next_state_vec,
    //                 done);
    // std::vector<State> next_states = {ego.state};
    // StateList next_state_list(next_states);
    StateList dummy_traj; // RL 暂时不需要轨迹预测
    return std::make_pair(action, dummy_traj);
  }
}

std::pair<std::vector<Action>, StateList>
KLevelPlanner::forward_simulate(const VehicleBase &ego,
                                const std::vector<StateList> &traj,
                                const std::string &initical_ego_name)
{
  MonteCarloTreeSearch mcts(traj, config);

  std::shared_ptr<Node> current_node = std::make_shared<Node>(
      ego.state, 0, nullptr, Action::MAINTAIN, StateList(), ego.target);
  if (ego.name == initical_ego_name)
  {
    current_node->initialize(Node::MAX_LEVEL,
                             MonteCarloTreeSearch::calc_cur_value);
    // spdlog::info(fmt::format("match, initical: {}, now:{}",
    // initical_ego_name,
    //                          ego.name));
    // std::unique_ptr<ActionEvaluator> evaluator;
    // bool use_mcts = true;
    // std::shared_ptr<DQNAgent> agent = std::make_shared<DQNAgent>();
    // double dt = 0.5;
    // if (use_mcts)
    // {
    //   evaluator_ = std::make_unique<MCTSEvaluator>(traj, config);
    // }
    // else
    // {
    //   evaluator_ = std::make_unique<DQNEvaluator>(agent, dt);
    // }

    // current_node = evaluator_->evaluate(current_node);
    current_node = mcts.excute(current_node);
  }
  else
  {
    current_node->initialize(Node::MAX_LEVEL,
                             MonteCarloTreeSearch::other_calc_cur_value);
    // spdlog::info(fmt::format("not match, initical: {}, now:{}",
    //                          initical_ego_name, ego.name));
    current_node = mcts.excute(current_node);
  }
  // current_node = mcts.excute(current_node);
  for (int i = 0; i < Node::MAX_LEVEL - 1; ++i)
  {
    current_node = mcts.get_best_child(current_node, 0);
  }

  std::vector<Action> actions = current_node->actions;
  StateList expected_traj;
  while (current_node != nullptr)
  {
    expected_traj.push_back(current_node->state);
    current_node = current_node->parent;
  }
  expected_traj.reverse();

  if (expected_traj.size() < steps + 1)
  {
    spdlog::debug(fmt::format("The max level of the node is not "
                              "enough({}),using the last value to complete it.",
                              expected_traj.size()));
    expected_traj.expand(steps + 1);
  }

  return std::make_pair(actions, expected_traj);
}

std::vector<StateList>
KLevelPlanner::get_prediction(const VehicleBase &ego,
                              const std::vector<VehicleBase> &others,
                              const std::string &initical_ego_name)
{
  std::vector<StateList> pred_trajectory;
  // std::thread::id this_id = std::this_thread::get_id();
  // spdlog::info(fmt::format("thread_id:{},actually_ego:{},pred_ego:{}",
  // this_id,
  //                          actually_ego, ego.name));

  // if (ego.name == initical_ego_name) {
  //   Node::initialize(8, MonteCarloTreeSearch::calc_cur_value);
  //   spdlog::info(fmt::format("match, initical: {}, now:{}",
  //   initical_ego_name,
  //                            ego.name));
  // } else {
  //   Node::initialize(8, MonteCarloTreeSearch::other_calc_cur_value);
  //   // spdlog::info(fmt::format("not match, initical: {}, now:{}",
  //   //  initical_ego_name, ego.name));
  // }
  if (ego.level == 0)
  {
    for (const VehicleBase &other : others)
    {
      StateList pred_traj;
      for (size_t i = 0; i < steps + 1; ++i)
      {
        pred_traj.push_back(other.state);
      }
      pred_trajectory.emplace_back(pred_traj);
    }
  }
  else if (ego.level > 0)
  {
    for (size_t idx = 0; idx < others.size(); ++idx)
    {
      if (others[idx].have_got_target)
      {
        StateList pred_traj;
        for (size_t i = 0; i < steps + 1; ++i)
        {
          pred_traj.push_back(others[idx].state);
        }
        pred_trajectory.emplace_back(pred_traj);
        continue;
      }
      VehicleBase exchanged_ego = others[idx];
      exchanged_ego.level = ego.level - 1;
      std::vector<VehicleBase> exchanged_others = {ego};
      for (size_t i = 0; i < others.size(); ++i)
      {
        if (i != idx)
        {
          exchanged_others.push_back(others[i]);
        }
      }
      std::vector<StateList> exchage_pred_others =
          get_prediction(exchanged_ego, exchanged_others, initical_ego_name);
      auto pred_idx_vechicle = forward_simulate(
          exchanged_ego, exchage_pred_others, initical_ego_name);
      pred_trajectory.emplace_back(pred_idx_vechicle.second);
    }
  }
  else
  {
    spdlog::error(
        "get_prediction() excute error, the level must be >= 0 and > 3 !");
  }

  return pred_trajectory;
}

double KLevelPlanner::compute_reward(const VehicleBase &ego)
{
  double reward = 0.0;

  // 1️⃣ 鼓励前进（速度奖励）
  reward += 0.1 * ego.state.v;

  // 2️⃣ 鼓励接近目标（距离减少奖励）
  double dx = ego.target.x - ego.state.x;
  double dy = ego.target.y - ego.state.y;
  double dist = std::sqrt(dx * dx + dy * dy);

  reward -= 0.01 * dist;

  // 3️⃣ 碰撞惩罚
  for (const auto &obj : ego.tracked_objects)
  {
    double dx = obj.state.x - ego.state.x;
    double dy = obj.state.y - ego.state.y;
    double d = std::sqrt(dx * dx + dy * dy);

    if (d < 2.0) // 简单碰撞阈值
    {
      reward -= 10.0;
    }
  }

  // 4️⃣ 到达目标奖励
  if (dist < 1.0)
  {
    reward += 20.0;
  }

  return reward;
}
bool KLevelPlanner::check_terminal(const VehicleBase &ego)
{
  double dx = ego.target.x - ego.state.x;
  double dy = ego.target.y - ego.state.y;
  double dist = std::sqrt(dx * dx + dy * dy);

  if (dist < 1.0)
    return true;

  for (const auto &obj : ego.tracked_objects)
  {
    double dx = obj.state.x - ego.state.x;
    double dy = obj.state.y - ego.state.y;
    double d = std::sqrt(dx * dx + dy * dy);

    if (d < 2.0)
      return true;
  }

  // timeout-based termination
  if (current_simulation_time > max_simulation_time)
  {
    // log once when timeout triggered
    spdlog::warn("terminal check: simulation time {} exceeded max {}", current_simulation_time, max_simulation_time);
    return true;
  }

  return false;
}

// ===== 新版本的 RL 函数 =====

std::vector<double> KLevelPlanner::encode_state_from_vehicle(
    const State &state,
    const std::vector<TrackedObject> &tracked_objects)
{
  std::vector<double> s;

  // ego 状态
  s.push_back(state.x / 50.0); // 归一化位置
  s.push_back(state.y / 50.0); // 归一化位置
  s.push_back(state.v / 20.0); // 归一化速度

  // 这里需要目标信息，但参数里没有，先用占位符
  s.push_back(0.0); // dx to target
  s.push_back(0.0); // dy to target

  // 其他车辆（相对位置和速度差）
  for (int i = 0; i < max_vehicle; ++i)
  {
    if (i < tracked_objects.size())
    {
      s.push_back((tracked_objects[i].state.x - state.x) / 50.0); // 归一化位置差
      s.push_back((tracked_objects[i].state.y - state.y) / 50.0); // 归一化位置差
      s.push_back((tracked_objects[i].state.v - state.v) / 20.0); // 归一化速度差
    }
    else
    {
      s.push_back(0.0);
      s.push_back(0.0);
      s.push_back(0.0);
    }
  }

  return s;
}

// double KLevelPlanner::compute_reward(
//     const State &old_state,
//     const State &new_state,
//     const std::vector<TrackedObject> &tracked_objects,
//     const State &target)
// {
//   // Reference MCTS calc_cur_value logic
//   double x = new_state.x;
//   double y = new_state.y;
//   double yaw = new_state.yaw;
//   double velocity = new_state.v;
//   bool is_ego = true; // Assuming ego vehicle
//   double acc_current = new_state.acc;
//   double acc_last = old_state.acc;

//   // Build other agent states
//   std::vector<State> other_agent_state;
//   for (const auto &obj : tracked_objects)
//   {
//     other_agent_state.push_back(obj.state);
//   }

//   Eigen::Matrix<double, 2, 5> ego_box2d = VehicleBase::get_box2d(new_state);
//   Eigen::Matrix<double, 2, 5> ego_tail_box2d = VehicleBase::get_tail_box2d(new_state);
//   Eigen::Matrix<double, 2, 5> ego_tractor_box2d = VehicleBase::get_tractor_box2d(new_state);

//   double avoid = 1.0;
//   for (auto &cur_other_state : other_agent_state)
//   {
//     if (!(is_ego && cur_other_state.is_ego))
//     {
//       if (utils::has_overlap(ego_box2d, VehicleBase::get_box2d(cur_other_state)))
//       {
//         avoid = 0;
//         break;
//       }
//     }
//     if ((!is_ego) && cur_other_state.is_ego)
//     {
//       if (utils::has_overlap(ego_box2d, VehicleBase::get_tail_box2d(cur_other_state)) ||
//           utils::has_overlap(ego_box2d, VehicleBase::get_tractor_box2d(cur_other_state)))
//       {
//         avoid = 0;
//         break;
//       }
//     }
//     else if (is_ego && (!cur_other_state.is_ego))
//     {
//       if (utils::has_overlap(ego_tail_box2d, VehicleBase::get_box2d(cur_other_state)) ||
//           utils::has_overlap(ego_tractor_box2d, VehicleBase::get_box2d(cur_other_state)))
//       {
//         avoid = 0;
//         break;
//       }
//     }
//   }

//   Eigen::Matrix<double, 2, 5> ego_safezone = VehicleBase::get_safezone(new_state);
//   Eigen::Matrix<double, 2, 5> safe_tail_box2d = VehicleBase::get_tail_safe_box2d(new_state);
//   Eigen::Matrix<double, 2, 5> safe_tractor_box2d = VehicleBase::get_tractor_safe_box2d(new_state);

//   double safe = 1.0;
//   for (auto &cur_other_state : other_agent_state)
//   {
//     if (!(is_ego && cur_other_state.is_ego))
//     {
//       if (utils::has_overlap(ego_safezone, VehicleBase::get_safezone(cur_other_state)))
//       {
//         safe = 0;
//         break;
//       }
//     }
//     if ((!is_ego) && cur_other_state.is_ego)
//     {
//       if (utils::has_overlap(ego_safezone, VehicleBase::get_tail_safe_box2d(cur_other_state)) ||
//           utils::has_overlap(ego_safezone, VehicleBase::get_tractor_safe_box2d(cur_other_state)))
//       {
//         safe = 0;
//         break;
//       }
//     }
//     else if (is_ego && (!cur_other_state.is_ego))
//     {
//       if (utils::has_overlap(safe_tail_box2d, VehicleBase::get_safezone(cur_other_state)) ||
//           utils::has_overlap(safe_tractor_box2d, VehicleBase::get_safezone(cur_other_state)))
//       {
//         safe = 0;
//         break;
//       }
//     }
//   }

//   int offroad = 0;
//   for (auto &rect : VehicleBase::env->rect_mat)
//   {
//     if (utils::has_overlap(ego_tractor_box2d, rect) || utils::has_overlap(ego_tail_box2d, rect))
//     {
//       offroad = -1;
//       break;
//     }
//   }

//   double delta_yaw = std::fmod(std::abs(yaw - target.yaw), TWO_PI);
//   delta_yaw = std::min(delta_yaw, TWO_PI - delta_yaw);
//   double d = (x - target.x) * (x - target.x) + (y - target.y) * (y - target.y);
//   double distance = -std::log1p(d) / std::log(10);
//   double l = (x - target.x) * (x - target.x);
//   double lateral_distance = -std::log1p(l) / std::log(10);

//   double ride_exp = 1 - std::tanh(acc_current * acc_current + (acc_last - acc_current) * (acc_last - acc_current));

//   double final_avoid = 1000 * avoid;
//   double final_lateral = 700 * lateral_distance;
//   double final_offroad = 300 * offroad;
//   double final_distance = 600 * distance;
//   double final_safe = 100 * safe;
//   double final_velocity = MonteCarloTreeSearch::WEIGHT_VELOCITY * velocity;
//   double final_ride = 1 * ride_exp;

//   double reward = final_avoid + final_lateral + final_offroad + final_distance + final_ride + final_safe + final_velocity;

//   spdlog::info("compute_reward components: avoid={} lateral={} offroad={} distance={} safe={} velocity={} ride={} total={}",
//                final_avoid, final_lateral, final_offroad, final_distance, final_safe, final_velocity, final_ride, reward);

//   return reward;
// }

double KLevelPlanner::compute_reward(
    const State &old_state,
    const State &new_state,
    const std::vector<TrackedObject> &tracked_objects,
    const State &target)
{
  double reward = 0.0;

  // ---------- progress reward ----------
  double old_d = hypot(old_state.x - target.x, old_state.y - target.y);
  double new_d = hypot(new_state.x - target.x, new_state.y - target.y);

  double progress = old_d - new_d;
  double progress_reward = 5.0 * progress;
  reward += progress_reward;

  // ---------- lateral alignment ----------
  double old_lat = fabs(old_state.x - target.x);
  double new_lat = fabs(new_state.x - target.x);

  double lateral_progress = old_lat - new_lat;
  double lateral_reward = 2.0 * lateral_progress;
  reward += lateral_reward;

  // ---------- velocity reward ----------
  double velocity_reward = 0.1 * new_state.v;
  reward += velocity_reward;

  // ---------- comfort (jerk penalty) ----------
  double jerk = new_state.acc - old_state.acc;
  double comfort_penalty = 0.2 * fabs(jerk);
  reward -= comfort_penalty;

  // ---------- collision ----------
  bool collision = false;
  double collision_penalty = 0.0;

  Eigen::Matrix<double, 2, 5> ego_box = VehicleBase::get_box2d(new_state);

  for (const auto &obj : tracked_objects)
  {
    if (utils::has_overlap(ego_box, VehicleBase::get_box2d(obj.state)))
    {
      collision = true;
      break;
    }
  }

  if (collision)
  {
    collision_penalty = 50.0;
    reward -= collision_penalty;
  }

  // ---------- offroad ----------
  bool offroad = false;
  double offroad_penalty = 0.0;

  Eigen::Matrix<double, 2, 5> ego_tail = VehicleBase::get_tail_box2d(new_state);
  Eigen::Matrix<double, 2, 5> ego_tractor = VehicleBase::get_tractor_box2d(new_state);

  for (auto &rect : VehicleBase::env->rect_mat)
  {
    if (utils::has_overlap(ego_tractor, rect) || utils::has_overlap(ego_tail, rect))
    {
      offroad = true;
      break;
    }
  }

  if (offroad)
  {
    offroad_penalty = 30.0;
    reward -= offroad_penalty;
  }

  // ---------- unsafe distance ----------
  bool unsafe = false;
  double unsafe_penalty = 0.0;

  Eigen::Matrix<double, 2, 5> safezone = VehicleBase::get_safezone(new_state);

  for (const auto &obj : tracked_objects)
  {
    if (utils::has_overlap(safezone, VehicleBase::get_safezone(obj.state)))
    {
      unsafe = true;
      break;
    }
  }

  if (unsafe)
  {
    unsafe_penalty = 10.0;
    reward -= unsafe_penalty;
  }

  // ---------- step penalty ----------
  double step_penalty = 0.01;
  reward -= step_penalty;

  // ---------- goal reward ----------
  double goal_reward = 0.0;
  if (new_d < 1.0)
  {
    goal_reward = 100.0;
    reward += goal_reward;
  }

  spdlog::info("compute_reward breakdown: progress={} lateral={} velocity={} comfort={} collision={} offroad={} unsafe={} step={} goal={} total={}",
               progress_reward, lateral_reward, velocity_reward, comfort_penalty,
               collision_penalty, offroad_penalty, unsafe_penalty,
               step_penalty, goal_reward, reward);

  return reward;
}
// RL-specific terminal check.  Includes reaching the target as a terminal
// condition.  Signature changed to accept target state.
bool KLevelPlanner::check_terminal(
    const State &state,
    const std::vector<TrackedObject> &tracked_objects,
    const State &target)
{
  // 碰撞终止条件
  for (const auto &obj : tracked_objects)
  {
    double d_x = obj.state.x - state.x;
    double d_y = obj.state.y - state.y;
    double dist = std::sqrt(d_x * d_x + d_y * d_y);

    if (dist < 1.5)
    {
      return true;
    }
  }

  // check timeout as additional terminal condition
  if (current_simulation_time > max_simulation_time)
  {
    spdlog::warn("terminal check (state): simulation time {} exceeded max {}", current_simulation_time, max_simulation_time);
    return true;
  }

  // check reaching target
  double dx = target.x - state.x;
  double dy = target.y - state.y;
  double td = std::sqrt(dx * dx + dy * dy);
  if (td < 1.0)
  {
    return true;
  }

  // 可以添加其他终止条件，如到达目标等
  return false;
}