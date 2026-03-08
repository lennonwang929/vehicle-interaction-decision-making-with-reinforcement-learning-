#pragma once
#ifndef __RL_ENV_HPP
#define __RL_ENV_HPP

#include <vector>
#include <memory>
#include <Eigen/Core>
#include <yaml-cpp/yaml.h>

#include "vehicle_base.hpp"
#include "utils.hpp"

/**
 * @brief 强化学习环境接口 - 标准 Markov Decision Process (MDP)
 *
 * 将现有规划器的"MCTS + cost function"替换为"DQN + Q 函数"
 *
 * 核心设计：
 * - 状态 (s): VehicleBase 当前的状态 (x, y, v, yaw, acc, ...)
 * - 动作 (a): 离散动作空间 (加速/减速/保持/变道)
 * - 奖励 (r): reward = -cost，使用现有的 cost function
 * - 下一状态 (s'): 动力学传播后的状态
 * - 终止条件 (done): 碰撞/离开道路/到达目标/超时
 */

class RLEnvironment
{
public:
    // ==================== 动作空间定义 ====================
    // 保持与现有 Action enum 一致
    enum class ActionType : int
    {
        ACCELERATE = 0, // 加速: acc = +2.0 m/s²
        DECELERATE = 1, // 减速: acc = -2.0 m/s²
        MAINTAIN = 2,   // 保持: acc = 0
        LANE_CHANGE_LEFT = 3,
        LANE_CHANGE_RIGHT = 4,
        NUM_ACTIONS = 5
    };

    // ==================== 状态向量定义 ====================
    /**
     * @brief 状态向量 (低维表示，便于神经网络学习)
     *
     * 第一版简化版本：
     * [0-2] 相对目标位置: [dx, dy, d_yaw]  (3维)
     * [3-5] 自车状态: [v, acc, is_ego]     (3维)
     * [6-8] 相对最近障碍物: [rel_x, rel_y, rel_v]  (3维)
     * [9-10] 车道信息: [lateral_error, yaw_error]  (2维)
     *
     * 总计: 11 维状态向量
     *
     * 优点：
     * - 充分利用规划器已有的计算（目标、障碍物、车道等）
     * - 低维→快速学习
     * - 可扩展：添加更多特征很简单
     */
    struct StateVector
    {
        static constexpr int DIM = 11;
        std::vector<double> data;

        StateVector() : data(DIM, 0.0) {}
        StateVector(const std::vector<double> &v) : data(v)
        {
            assert(v.size() == DIM);
        }

        double &operator[](int i) { return data[i]; }
        const double &operator[](int i) const { return data[i]; }
    };

    // ==================== Step 返回值 ====================
    /**
     * @brief 单步执行的返回值
     *
     * 标准 RL step() 的返回格式:
     * - s:     当前状态 (执行动作前)
     * - a:     执行的动作
     * - r:     获得的奖励
     * - s':    下一状态 (执行动作后)
     * - done:  是否终止
     */
    struct StepResult
    {
        StateVector s;           // 执行前的状态
        ActionType a;            // 执行的动作
        double r;                // 立即奖励
        StateVector s_next;      // 执行后的下一状态
        bool done;               // 是否终止
        std::string done_reason; // 终止原因 (便于调试)
    };

    // ==================== 成本项分解 ====================
    /**
     * @brief 成本详细分解 (便于调试和分析)
     *
     * 对应 planner.cpp 中的:
     * - final_avoid:     碰撞成本 (碰撞→0, 安全→1)
     * - final_lateral:   横向距离成本
     * - final_offroad:   离线成本
     * - final_distance:  纵向距离成本
     * - final_safe:      安全区成本
     * - final_ride:      乘坐舒适度 (加速度突变)
     */
    struct CostBreakdown
    {
        double cost_avoid;    // 碰撞代价
        double cost_lateral;  // 横向偏离
        double cost_offroad;  // 离线
        double cost_distance; // 纵向距离
        double cost_safe;     // 安全区侵入
        double cost_ride;     // 舒适性
        double cost_total;    // 总成本 = sum of above
    };

public:
    // ==================== 构造与初始化 ====================
    RLEnvironment(const YAML::Node &cfg);
    ~RLEnvironment() = default;

    /**
     * @brief 重置环境，返回初始状态 s_0
     */
    StateVector reset(VehicleBase &ego);

    // ==================== 核心 Step 函数 ====================
    /**
     * @brief 执行一步动作，返回 (s, a, r, s', done)
     *
     * @param ego        自车引用
     * @param others     其他车辆（用于预测和碰撞检测）
     * @param action     要执行的动作
     *
     * @return StepResult 包含 s, a, r, s', done, done_reason
     *
     * 流程：
     * 1. 提取当前状态 s (执行前)
     * 2. 将 action 转换为加速度向量 [acc, omega]
     * 3. 使用动力学模型传播状态: s' = kinematic_propagate(s, a, dt)
     * 4. 计算奖励 r = -cost(s')，使用现有 cost function
     * 5. 检查终止条件 done
     * 6. 返回 (s, a, r, s', done)
     */
    StepResult step(VehicleBase &ego,
                    const std::vector<VehicleBase> &others,
                    ActionType action);

    // ==================== 辅助函数 ====================
    /**
     * @brief 将 ActionType 转换为加速度向量 [acc, omega]
     *
     * 对应现有代码的 utils::get_action_value(Action)
     *
     * @param action 动作类型
     * @return Eigen::Vector2d [加速度, 角速度]
     */
    static Eigen::Vector2d action_to_acceleration(ActionType action);

    /**
     * @brief 从 State 提取低维状态向量
     *
     * @param ego          自车对象
     * @param others       其他车辆
     * @param goal_pose    目标位置
     * @return StateVector 低维状态向量 (11维)
     */
    static StateVector extract_state_vector(
        const VehicleBase &ego,
        const std::vector<VehicleBase> &others,
        const State &goal_pose);

    /**
     * @brief 计算成本 (对应 planner.cpp 中的 calc_cur_value)
     *
     * @param state              当前状态
     * @param other_agent_states 其他车辆状态
     * @param goal_pose          目标位置
     * @param is_ego             是否为自车
     * @return CostBreakdown     分解后的成本
     */
    CostBreakdown compute_cost(
        const State &state,
        const std::vector<State> &other_agent_states,
        const State &goal_pose,
        bool is_ego);

    /**
     * @brief 检查是否碰撞
     *
     * @param ego_state 自车状态
     * @param others    其他车辆
     * @return true 发生碰撞
     */
    static bool check_collision(
        const State &ego_state,
        const std::vector<VehicleBase> &others);

    /**
     * @brief 检查是否离线（驶出边界）
     *
     * @param state 车辆状态
     * @param is_ego 是否为自车（自车有挂车）
     * @return true 离线
     */
    static bool check_offroad(const State &state, bool is_ego);

    /**
     * @brief 检查是否到达目标
     *
     * @param ego_state 自车状态
     * @param target    目标位置
     * @param threshold 到达阈值
     * @return true 到达目标
     */
    static bool check_goal_reached(
        const State &ego_state,
        const State &target,
        double threshold = 1.7);

private:
    // ==================== 内部参数 ====================
    YAML::Node config;
    double dt;     // 时间步长
    int max_steps; // 最大步数

    // ==================== 内部辅助函数 ====================
    /**
     * @brief 从其他车辆列表中找到最近的障碍物
     */
    struct NearestObstacle
    {
        bool found;
        State state;
        double distance;
        double relative_velocity;
    };

    static NearestObstacle find_nearest_obstacle(
        const State &ego_state,
        const std::vector<VehicleBase> &others);
};

#endif // __RL_ENV_HPP
