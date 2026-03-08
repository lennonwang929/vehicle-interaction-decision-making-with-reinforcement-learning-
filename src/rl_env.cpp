#include "rl_env.hpp"
#include <spdlog/spdlog.h>
#include <fmt/core.h>
#include <cmath>

constexpr double TWO_PI = M_PI * 2;

// ==================== 构造与初始化 ====================

RLEnvironment::RLEnvironment(const YAML::Node &cfg)
    : config(cfg),
      dt(cfg["delta_t"].as<double>()),
      max_steps(cfg["max_step"].as<int>())
{
}

// ==================== 动作转换 ====================

Eigen::Vector2d RLEnvironment::action_to_acceleration(ActionType action)
{
    // 对应现有的 utils::get_action_value(Action)
    // 动作 → 加速度 [acc, omega]

    switch (action)
    {
    case ActionType::ACCELERATE:
        return Eigen::Vector2d(2.5, 0.0); // 加速：+2.5 m/s²
    case ActionType::DECELERATE:
        return Eigen::Vector2d(-2.5, 0.0); // 减速：-2.5 m/s²
    case ActionType::MAINTAIN:
        return Eigen::Vector2d(0.0, 0.0); // 保持：不加速
    case ActionType::LANE_CHANGE_LEFT:
        return Eigen::Vector2d(0.0, M_PI_4); // 左转：角速度 π/4
    case ActionType::LANE_CHANGE_RIGHT:
        return Eigen::Vector2d(0.0, -M_PI_4); // 右转：角速度 -π/4
    default:
        spdlog::error("Unknown action type: {}", static_cast<int>(action));
        return Eigen::Vector2d(0.0, 0.0);
    }
}

// ==================== 状态提取 ====================

RLEnvironment::StateVector RLEnvironment::extract_state_vector(
    const VehicleBase &ego,
    const std::vector<VehicleBase> &others,
    const State &goal_pose)
{

    StateVector state_vec;

    // ========== [0-2] 相对目标位置 ==========
    double dx = ego.state.x - goal_pose.x;
    double dy = ego.state.y - goal_pose.y;
    double d_yaw = ego.state.yaw - goal_pose.yaw;

    // 归一化角度差到 [-π, π]
    while (d_yaw > M_PI)
        d_yaw -= TWO_PI;
    while (d_yaw < -M_PI)
        d_yaw += TWO_PI;

    state_vec[0] = dx;
    state_vec[1] = dy;
    state_vec[2] = d_yaw;

    // ========== [3-5] 自车状态 ==========
    state_vec[3] = ego.state.v;                  // 速度
    state_vec[4] = ego.state.acc;                // 加速度
    state_vec[5] = ego.state.is_ego ? 1.0 : 0.0; // 是否为自车

    // ========== [6-8] 相对最近障碍物 ==========
    NearestObstacle nearest = find_nearest_obstacle(ego.state, others);
    if (nearest.found)
    {
        state_vec[6] = nearest.state.x - ego.state.x; // 相对 x
        state_vec[7] = nearest.state.y - ego.state.y; // 相对 y
        state_vec[8] = nearest.state.v - ego.state.v; // 相对速度
    }
    else
    {
        // 没有障碍物时置零
        state_vec[6] = 0.0;
        state_vec[7] = 0.0;
        state_vec[8] = 0.0;
    }

    // ========== [9-10] 车道信息 ==========
    // 横向偏离：相对于目标的 x 偏离
    state_vec[9] = dx;
    // 航向偏离：相对于目标的 yaw 偏离
    state_vec[10] = d_yaw;

    return state_vec;
}

RLEnvironment::NearestObstacle RLEnvironment::find_nearest_obstacle(
    const State &ego_state,
    const std::vector<VehicleBase> &others)
{

    NearestObstacle result;
    result.found = false;
    result.distance = std::numeric_limits<double>::max();

    for (const auto &other : others)
    {
        double dist = std::hypot(
            other.state.x - ego_state.x,
            other.state.y - ego_state.y);

        if (dist < result.distance && dist > 1e-6)
        { // 排除自己
            result.found = true;
            result.distance = dist;
            result.state = other.state;
            result.relative_velocity = other.state.v - ego_state.v;
        }
    }

    return result;
}

// ==================== 成本计算 ====================

RLEnvironment::CostBreakdown RLEnvironment::compute_cost(
    const State &state,
    const std::vector<State> &other_agent_states,
    const State &goal_pose,
    bool is_ego)
{

    CostBreakdown cost;

    // ========== 1. 碰撞检测 (avoid) ==========
    cost.cost_avoid = 1.0; // 默认安全

    Eigen::Matrix<double, 2, 5> ego_box2d = VehicleBase::get_box2d(state);
    Eigen::Matrix<double, 2, 5> ego_tail_box2d;
    Eigen::Matrix<double, 2, 5> ego_tractor_box2d;

    if (is_ego)
    {
        ego_tractor_box2d = VehicleBase::get_tractor_box2d(state);
        ego_tail_box2d = VehicleBase::get_tail_box2d(state);
    }

    for (const auto &other_state : other_agent_states)
    {
        // 自车 vs 其他车的车身
        if (is_ego && !other_state.is_ego)
        {
            if (utils::has_overlap(ego_tail_box2d, VehicleBase::get_box2d(other_state)) ||
                utils::has_overlap(ego_tractor_box2d, VehicleBase::get_box2d(other_state)))
            {
                cost.cost_avoid = 0.0; // 碰撞
                break;
            }
        }
        // 其他车 vs 自车的车身或挂车
        else if (!is_ego && other_state.is_ego)
        {
            if (utils::has_overlap(ego_box2d, VehicleBase::get_tail_box2d(other_state)) ||
                utils::has_overlap(ego_box2d, VehicleBase::get_tractor_box2d(other_state)))
            {
                cost.cost_avoid = 0.0;
                break;
            }
        }
        // 其他车 vs 其他车
        else
        {
            if (utils::has_overlap(ego_box2d, VehicleBase::get_box2d(other_state)))
            {
                cost.cost_avoid = 0.0;
                break;
            }
        }
    }

    // ========== 2. 横向距离 ==========
    double lateral_error = state.x - goal_pose.x;
    double lateral_distance_sq = lateral_error * lateral_error;
    cost.cost_lateral = -std::log1p(lateral_distance_sq) / std::log(10);
    cost.cost_lateral *= 700; // 权重

    // ========== 3. 离线检测 ==========
    cost.cost_offroad = 0.0; // 默认在道路内

    if (!is_ego)
    {
        for (const auto &rect : VehicleBase::env->rect_mat)
        {
            if (utils::has_overlap(ego_box2d, rect))
            {
                cost.cost_offroad = -300; // 离线惩罚
                break;
            }
        }
    }
    else
    {
        for (const auto &rect : VehicleBase::env->rect_mat)
        {
            if (utils::has_overlap(ego_tractor_box2d, rect) ||
                utils::has_overlap(ego_tail_box2d, rect))
            {
                cost.cost_offroad = -300;
                break;
            }
        }
    }

    // ========== 4. 纵向距离 ==========
    double d = (state.x - goal_pose.x) * (state.x - goal_pose.x) +
               (state.y - goal_pose.y) * (state.y - goal_pose.y);
    cost.cost_distance = -std::log1p(d) / std::log(10);
    cost.cost_distance *= 600; // 权重

    // ========== 5. 安全区 ==========
    cost.cost_safe = 1.0; // 默认安全

    Eigen::Matrix<double, 2, 5> ego_safezone = VehicleBase::get_safezone(state);
    Eigen::Matrix<double, 2, 5> safe_tail_box2d;
    Eigen::Matrix<double, 2, 5> safe_tractor_box2d;

    if (is_ego)
    {
        safe_tractor_box2d = VehicleBase::get_tractor_safe_box2d(state);
        safe_tail_box2d = VehicleBase::get_tail_safe_box2d(state);
    }

    for (const auto &other_state : other_agent_states)
    {
        if (is_ego && !other_state.is_ego)
        {
            if (utils::has_overlap(safe_tail_box2d, VehicleBase::get_safezone(other_state)) ||
                utils::has_overlap(safe_tractor_box2d, VehicleBase::get_safezone(other_state)))
            {
                cost.cost_safe = 0.0;
                break;
            }
        }
        else if (!is_ego && other_state.is_ego)
        {
            if (utils::has_overlap(ego_safezone, VehicleBase::get_tail_safe_box2d(other_state)) ||
                utils::has_overlap(ego_safezone, VehicleBase::get_tractor_safe_box2d(other_state)))
            {
                cost.cost_safe = 0.0;
                break;
            }
        }
    }

    cost.cost_safe *= 100; // 权重

    // ========== 6. 乘坐舒适度 ==========
    // 简化版：只看当前加速度的平方
    cost.cost_ride = 1.0 - std::tanh(state.acc * state.acc);
    cost.cost_ride *= 1.0; // 权重较小

    // ========== 7. 总成本 ==========
    cost.cost_total = cost.cost_avoid * 1000.0 // 碰撞权重最高
                      + cost.cost_lateral + cost.cost_offroad + cost.cost_distance + cost.cost_ride + cost.cost_safe;

    return cost;
}

// ==================== 碰撞检测 ====================

bool RLEnvironment::check_collision(
    const State &ego_state,
    const std::vector<VehicleBase> &others)
{

    bool is_ego = ego_state.is_ego;
    Eigen::Matrix<double, 2, 5> ego_box2d = VehicleBase::get_box2d(ego_state);

    Eigen::Matrix<double, 2, 5> ego_tail_box2d;
    Eigen::Matrix<double, 2, 5> ego_tractor_box2d;
    if (is_ego)
    {
        ego_tractor_box2d = VehicleBase::get_tractor_box2d(ego_state);
        ego_tail_box2d = VehicleBase::get_tail_box2d(ego_state);
    }

    for (const auto &other : others)
    {
        if (is_ego && !other.state.is_ego)
        {
            if (utils::has_overlap(ego_tail_box2d, VehicleBase::get_box2d(other.state)) ||
                utils::has_overlap(ego_tractor_box2d, VehicleBase::get_box2d(other.state)))
            {
                return true;
            }
        }
        else if (!is_ego && other.state.is_ego)
        {
            if (utils::has_overlap(ego_box2d, VehicleBase::get_tail_box2d(other.state)) ||
                utils::has_overlap(ego_box2d, VehicleBase::get_tractor_box2d(other.state)))
            {
                return true;
            }
        }
        else
        {
            if (utils::has_overlap(ego_box2d, VehicleBase::get_box2d(other.state)))
            {
                return true;
            }
        }
    }

    return false;
}

// ==================== 离线检测 ====================

bool RLEnvironment::check_offroad(const State &state, bool is_ego)
{
    Eigen::Matrix<double, 2, 5> ego_box2d = VehicleBase::get_box2d(state);

    if (!is_ego)
    {
        for (const auto &rect : VehicleBase::env->rect_mat)
        {
            if (utils::has_overlap(ego_box2d, rect))
            {
                return true;
            }
        }
    }
    else
    {
        Eigen::Matrix<double, 2, 5> ego_tail_box2d = VehicleBase::get_tail_box2d(state);
        Eigen::Matrix<double, 2, 5> ego_tractor_box2d = VehicleBase::get_tractor_box2d(state);

        for (const auto &rect : VehicleBase::env->rect_mat)
        {
            if (utils::has_overlap(ego_tractor_box2d, rect) ||
                utils::has_overlap(ego_tail_box2d, rect))
            {
                return true;
            }
        }
    }

    return false;
}

// ==================== 目标检测 ====================

bool RLEnvironment::check_goal_reached(
    const State &ego_state,
    const State &target,
    double threshold)
{

    double dist = std::hypot(ego_state.x - target.x, ego_state.y - target.y);
    return dist < threshold;
}

// ==================== 重置环境 ====================

RLEnvironment::StateVector RLEnvironment::reset(VehicleBase &ego)
{
    // 重置时，提取初始状态向量
    return extract_state_vector(ego, {}, ego.target);
}

// ==================== 核心 Step 函数 ====================

RLEnvironment::StepResult RLEnvironment::step(
    VehicleBase &ego,
    const std::vector<VehicleBase> &others,
    ActionType action)
{

    StepResult result;

    // ========== 1. 提取执行前的状态 s ==========
    result.s = extract_state_vector(ego, others, ego.target);
    result.a = action;

    // ========== 2. 将动作转换为加速度向量 ==========
    Eigen::Vector2d acceleration = action_to_acceleration(action);

    // ========== 3. 使用动力学模型传播状态 s' ==========
    State next_state = utils::kinematic_propagate(ego.state, acceleration, dt);

    // ========== 4. 收集其他车的状态 ==========
    std::vector<State> other_states;
    for (const auto &other : others)
    {
        other_states.push_back(other.state);
    }

    // ========== 5. 计算成本和奖励 ==========
    CostBreakdown cost = compute_cost(
        next_state,
        other_states,
        ego.target,
        ego.state.is_ego);

    // reward = -cost (越小的成本 → 越大的奖励)
    result.r = -cost.cost_total;

    // ========== 6. 提取执行后的状态 s' ==========
    ego.state = next_state; // 更新自车状态
    result.s_next = extract_state_vector(ego, others, ego.target);

    // ========== 7. 检查终止条件 ==========
    result.done = false;
    result.done_reason = "ongoing";

    // 7.1: 碰撞
    if (check_collision(next_state, others))
    {
        result.done = true;
        result.done_reason = "collision";
        result.r = -1000.0; // 碰撞惩罚
        spdlog::warn("Episode terminated: COLLISION");
    }
    // 7.2: 离线
    else if (check_offroad(next_state, ego.state.is_ego))
    {
        result.done = true;
        result.done_reason = "offroad";
        result.r = -500.0; // 离线惩罚
        spdlog::warn("Episode terminated: OFFROAD");
    }
    // 7.3: 到达目标
    else if (check_goal_reached(next_state, ego.target))
    {
        result.done = true;
        result.done_reason = "goal_reached";
        result.r = 100.0; // 目标奖励
        spdlog::info("Episode terminated: GOAL REACHED");
    }
    // 7.4: 超时（可选）
    // else if (step_count >= max_steps) {
    //     result.done = true;
    //     result.done_reason = "max_steps";
    //     spdlog::warn("Episode terminated: MAX STEPS");
    // }

    spdlog::debug(fmt::format(
        "RL Step | s=[{:.2f},{:.2f},{:.2f}] | "
        "a={} | r={:.3f} | s'=[{:.2f},{:.2f},{:.2f}] | "
        "done={} ({})",
        result.s[0], result.s[1], result.s[2],
        static_cast<int>(action),
        result.r,
        result.s_next[0], result.s_next[1], result.s_next[2],
        result.done, result.done_reason));

    return result;
}
