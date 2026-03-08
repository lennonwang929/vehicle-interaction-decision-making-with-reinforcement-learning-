#pragma once
#ifndef __UTILS_HPP
#define __UTILS_HPP

#include <algorithm>
#include <chrono>
#include <functional>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>

enum class Action {
  MAINTAIN,
  TURNLEFT,
  TURNRIGHT,
  ACCELERATE,
  DECELERATE,
  BRAKE
};
const std::vector<Action> ACTION_LIST = {
    Action::MAINTAIN,   // (0, 0)
    Action::TURNLEFT,   // (0, pi/4)
    Action::TURNRIGHT,  // (0, -pi/4)
    Action::ACCELERATE, // (2.5, 0)
    Action::DECELERATE, // (-2.5, 0)
    Action::BRAKE       // (-5.0, 0)
};
#define SMALL_NUMBER (1E-6)
#define SMALL_NUMBERF (1E-6f)
#define LARGE_NUMBER (1E+10)
#define LARGE_NUMBERF (1E+10f)

class SE_Vector {
public:
  SE_Vector() : x_(0.0), y_(0.0) {}
  SE_Vector(double x, double y) : x_(x), y_(y) {}

  void SetX(double x) { x_ = x; }
  void SetY(double y) { y_ = y; }
  void Set(double x, double y) { x_ = x, y_ = y; }

  SE_Vector operator+(SE_Vector const &p) const {
    SE_Vector res;
    res.x_ = x_ + p.x_;
    res.y_ = y_ + p.y_;
    return res;
  }

  SE_Vector operator-(SE_Vector const &p) const {
    SE_Vector res;
    res.x_ = x_ - p.x_;
    res.y_ = y_ - p.y_;
    return res;
  }

  SE_Vector &operator+=(SE_Vector const &p) {
    this->x_ += p.x_;
    this->y_ += p.y_;
    return *this;
  }

  SE_Vector &operator-=(SE_Vector const &p) {
    this->x_ -= p.x_;
    this->y_ -= p.y_;
    return *this;
  }

  SE_Vector Rotate(double angle) {
    SE_Vector res;
    res.x_ = x_ * cos(angle) - y_ * sin(angle);
    res.y_ = x_ * sin(angle) + y_ * cos(angle);
    return res;
  }

  double Dot(const SE_Vector &corner) const {
    return x_ * corner.x_ + y_ * corner.y_;
  }

  double Cross(const SE_Vector &vec) const { return x_ * vec.y_ - vec.x_ * y_; }

  double GetLength() {
    double l2 = x_ * x_ + y_ * y_;
    if (l2 > SMALL_NUMBER) {
      return sqrt(l2);
    }

    return 0.0;
  }

  void SetLength(double length) {
    double current_length = GetLength();
    if (current_length > SMALL_NUMBER) {
      x_ *= length / current_length;
      y_ *= length / current_length;
    } else {
      x_ = 0.0;
      y_ = 0.0;
    }
  }

  void Normalize() {
    double len = GetLength();
    if (len < SMALL_NUMBER) {
      len = SMALL_NUMBER;
    }
    x_ = x_ / len;
    y_ = y_ / len;
  }

  double x() const { return x_; }
  double y() const { return y_; }

private:
  double x_;
  double y_;
};

class Random {
private:
  static std::default_random_engine engine;

  Random() = delete;
  Random(const Random &) = delete;
  Random &operator=(const Random &) = delete;
  Random(Random &&) = delete;
  Random &operator=(Random &&) = delete;

public:
  static int uniform(int _min, int _max);
  static double uniform(double _min, double _max);
  template <typename T> static T choice(const std::vector<T> &vec) {
    if (vec.empty()) {
      throw std::runtime_error(
          "Cannot select an element from an empty vector.");
    }

    int random_idx = Random::uniform(0, vec.size() - 1);
    return vec[random_idx];
  }
};

class TicToc {
public:
  TicToc(void) { tic(); }

  void tic(void) { start = std::chrono::system_clock::now(); }

  double toc(void) {
    end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    return elapsed_seconds.count();
  }

private:
  std::chrono::time_point<std::chrono::system_clock> start, end;
};

class State {
private:
  /* data */
public:
  double x;
  double y;
  double yaw;
  double v;
  double acc;
  bool is_ego;
  static double hitch_x;  // 前车头挂钩位置
  static double couple_x; // 挂斗挂钩位置
  double tail_x;
  double tail_y;
  double tail_yaw;
  // double tail_v;

  State()
      : x(0), y(0), yaw(0), v(0), acc(0), is_ego(false), tail_x(0), tail_y(0),
        tail_yaw(0) {}
  State(double _x, double _y, double _yaw, double _v, double _acc,
        double _tail_x, double _tail_y, double _tail_yaw)
      : x(_x), y(_y), yaw(_yaw), v(_v), acc(_acc), tail_x(_tail_x),
        tail_y(_tail_y), tail_yaw(_tail_yaw) {}
  ~State() {}

  std::vector<double> to_vector(void) {
    return std::vector<double>{x, y, yaw, v};
  }
};

class StateList {
private:
  std::vector<State> states;
  std::vector<std::vector<double>> state_list;
  std::vector<std::vector<double>> state_list_trans;
  void append(const State &state) {
    states.push_back(state);
    state_list_trans[0].push_back(state.x);
    state_list_trans[1].push_back(state.y);
    state_list_trans[2].push_back(state.yaw);
    state_list_trans[3].push_back(state.v);
    std::vector<double> state_vec_tmp = {state.x, state.y, state.yaw, state.v};
    state_list.emplace_back(state_vec_tmp);
  }

public:
  StateList() {
    states.clear();
    state_list.clear();
    state_list_trans.clear();
    state_list_trans.resize(4);
  }
  StateList(const std::vector<State> &st) {
    states.clear();
    state_list.clear();
    state_list_trans.clear();
    state_list_trans.resize(4);

    for (const State &s : st) {
      this->append(s);
    }
  }
  ~StateList() {}

  size_t size(void) { return states.size(); }

  void push_back(const State &state) { this->append(state); }

  void reverse(void) {
    std::reverse(states.begin(), states.end());
    std::reverse(state_list.begin(), state_list.end());
    for (size_t idx = 0; idx < state_list_trans.size(); ++idx) {
      std::reverse(state_list_trans[idx].begin(), state_list_trans[idx].end());
    }
  }

  void expand(int excepted_len) {
    if (states.size() < 1) {
      return;
    }
    State expand_state = states[-1];
    expand(excepted_len, expand_state);
  }

  void expand(int excepted_len, const State &expand_state) {
    size_t cur_size = states.size();
    if (cur_size >= excepted_len) {
      return;
    }

    for (size_t it = 0; it < excepted_len - cur_size; ++it) {
      this->append(expand_state);
    }
  }

  std::vector<std::vector<double>> to_vector(bool trans = true) const {
    if (trans) {
      return state_list_trans;
    } else {
      return state_list;
    }
  }

  State &operator[](int index) {
    if (index < 0 || index >= states.size()) {
      throw std::out_of_range("Index out of range");
    }
    return states[index];
  }

  const State &operator[](int index) const {
    if (index < 0 || index >= states.size()) {
      throw std::out_of_range("Index out of range");
    }
    return states[index];
  }

  auto begin() { return states.begin(); }

  auto end() { return states.end(); }

  auto begin() const { return states.begin(); }

  auto end() const { return states.end(); }
};

class Node : public std::enable_shared_from_this<Node> {
private:
  /* data */
public:
  static int MAX_LEVEL;
  double (*calc_value_callback)(std::shared_ptr<Node>, double);
  // static std::function<double(std::shared_ptr<Node>, double)>
  //     calc_value_callback;

  State state;
  double value;
  double reward;
  int visits;
  Action action;
  std::shared_ptr<Node> parent;
  int cur_level;
  State goal_pose;
  std::vector<std::shared_ptr<Node>> children;
  std::vector<Action> actions;
  StateList other_agent_state;

  Node() = delete;
  Node(State _state, int _level, std::shared_ptr<Node> p, Action act,
       StateList others, State goal);
  ~Node() {}

  void initialize(int max_level,
                  double (*callback)(std::shared_ptr<Node>, double)) {
    Node::MAX_LEVEL = max_level;
    calc_value_callback = callback;
  }

  bool is_terminal(void);
  bool is_fully_expanded(void);
  std::shared_ptr<Node> add_child(Action next_action, double delta_t,
                                  StateList others);
  std::shared_ptr<Node> next_node(double delta_t, StateList others);
};

namespace utils {

std::string get_action_name(Action action);
Eigen::Vector2d get_action_value(Action act);
bool has_overlap(Eigen::MatrixXd box2d_0, Eigen::MatrixXd box2d_1);
State kinematic_propagate(const State &state, Eigen::Vector2d act, double dt);
std::string absolute_path(std::string path);
std::vector<float> imread(std::string filename, int &rows, int &cols,
                          int &colors);
} // namespace utils

#endif
