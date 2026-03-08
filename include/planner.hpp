#pragma once
#ifndef __PLANNER_HPP
#define __PLANNER_HPP

#include <yaml-cpp/yaml.h>
#include "action_evaluator.hpp"
#include "utils.hpp"
#include "vehicle_base.hpp"
#include "dqn_agent.hpp"
#include <filesystem>
#include <spdlog/spdlog.h>

// extern std::string actually_ego;
// class ActionEvaluator;
class MonteCarloTreeSearch
{
private:
  /* data */
public:
  static double EXPLORATE_RATE;
  static double LAMDA;
  static double WEIGHT_AVOID;
  static double WEIGHT_SAFE;
  static double WEIGHT_OFFROAD;
  static double WEIGHT_DIRECTION;
  static double WEIGHT_DISTANCE;
  static double WEIGHT_VELOCITY;

  std::vector<StateList> other_predict_traj;
  uint64_t computation_budget;
  double dt;

  MonteCarloTreeSearch(const std::vector<StateList> &other_traj,
                       const YAML::Node &cfg);
  ~MonteCarloTreeSearch() {}

  static void initialize(const YAML::Node &cfg)
  {
    MonteCarloTreeSearch::LAMDA = cfg["lamda"].as<double>();
    MonteCarloTreeSearch::WEIGHT_AVOID = cfg["weight_avoid"].as<double>();
    MonteCarloTreeSearch::WEIGHT_SAFE = cfg["weight_safe"].as<double>();
    MonteCarloTreeSearch::WEIGHT_OFFROAD = cfg["weight_offroad"].as<double>();
    MonteCarloTreeSearch::WEIGHT_DIRECTION =
        cfg["weight_direction"].as<double>();
    MonteCarloTreeSearch::WEIGHT_DISTANCE = cfg["weight_distance"].as<double>();
    MonteCarloTreeSearch::WEIGHT_VELOCITY = cfg["weight_velocity"].as<double>();
  }
  static bool is_opposite_direction(State pos, Eigen::MatrixXd ego_box2d);
  static double calc_cur_value(std::shared_ptr<Node> node,
                               double last_node_value);
  static double other_calc_cur_value(std::shared_ptr<Node> node,
                                     double last_node_value);
  std::shared_ptr<Node> excute(std::shared_ptr<Node> root);
  std::shared_ptr<Node> tree_policy(std::shared_ptr<Node> node);
  std::shared_ptr<Node> expand(std::shared_ptr<Node> node);
  std::shared_ptr<Node> get_best_child(std::shared_ptr<Node> node,
                                       double scalar);
  double default_policy(std::shared_ptr<Node> node);
  void update(std::shared_ptr<Node> node, double r);
};

class KLevelPlanner
{
private:
  int steps;
  YAML::Node config;

  KLevelPlanner(const KLevelPlanner &single) = delete;
  const KLevelPlanner &operator=(const KLevelPlanner &single) = delete;
  KLevelPlanner(const YAML::Node &cfg) : config(cfg)
  {
    steps = cfg["max_step"].as<int>();
    max_vehicle = cfg["max_vehicle_num"].as<int>();
    int state_dim = 5 + 3 * max_vehicle;
    int action_dim = 5;
    agent_ = std::make_shared<DQNAgent>(state_dim, action_dim);
  }
  ~KLevelPlanner() {}

  std::unique_ptr<ActionEvaluator> evaluator_;
  std::shared_ptr<DQNAgent> agent_;
  int max_vehicle;

public:
  static KLevelPlanner &
  get_instance(const YAML::Node &cfg)
  {
    static KLevelPlanner planner_(cfg);
    return planner_;
  }

  std::pair<Action, StateList> planning(VehicleBase &ego);
  std::pair<std::vector<Action>, StateList>
  forward_simulate(const VehicleBase &ego, const std::vector<StateList> &traj,
                   const std::string &initical_ego_name);
  std::vector<StateList>
  get_prediction(const VehicleBase &ego, const std::vector<VehicleBase> &others,
                 const std::string &initical_ego_name);
  std::vector<double> encode_state(
      const VehicleBase &ego,
      const std::vector<TrackedObject> &others);
  double compute_reward(const VehicleBase &ego);
  bool check_terminal(const VehicleBase &ego);

  // global timing information for checking timeout in terminal condition
  // must be updated externally (e.g., in decision_making loop)
  static double current_simulation_time;
  static double max_simulation_time;
  static void set_simulation_times(double current, double maximum)
  {
    current_simulation_time = current;
    max_simulation_time = maximum;
  }

  // RL 版本的函数，用于逐步学习
  std::vector<double> encode_state_from_vehicle(
      const State &state,
      const std::vector<TrackedObject> &tracked_objects);
  double compute_reward(const State &old_state,
                        const State &new_state,
                        const std::vector<TrackedObject> &tracked_objects,
                        const State &target);
  // RL-specific terminal check; now includes target so that reaching goal
  // can properly terminate an episode.  This does not affect the original
  // version taking a VehicleBase reference, which is still used by MCTS.
  bool check_terminal(const State &state,
                        const std::vector<TrackedObject> &tracked_objects,
                        const State &target);

  // RL agent training interface: forwards to agent_->observe and returns the
  // 'done' value so callers (e.g., Vehicle) can react to episode termination.
  bool train_agent(const std::vector<double> &state_vec,
                   int action_id,
                   double reward,
                   const std::vector<double> &next_state_vec,
                   bool done)
  {
    if (agent_)
    {
      agent_->observe(state_vec, action_id, reward, next_state_vec, done);
    }
    return done;
  }

  bool is_rl_mode() const { return true; }
};

#endif
