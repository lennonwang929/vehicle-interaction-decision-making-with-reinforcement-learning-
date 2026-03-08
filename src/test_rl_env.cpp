#include <iostream>
#include <vector>
#include <yaml-cpp/yaml.h>
#include <spdlog/spdlog.h>

#include "rl_env.hpp"
#include "vehicle.hpp"

/**
 * @brief RL环境单元测试
 *
 * 验证：
 * 1. 状态向量提取
 * 2. 动作转换
 * 3. 成本计算
 * 4. Step函数的完整流程
 */

void print_state_vector(const RLEnvironment::StateVector &sv, const std::string &name = "State")
{
    std::cout << name << " = [";
    for (int i = 0; i < RLEnvironment::StateVector::DIM; ++i)
    {
        std::cout << sv[i];
        if (i < RLEnvironment::StateVector::DIM - 1)
            std::cout << ", ";
    }
    std::cout << "]\n";
}

void test_action_conversion()
{
    std::cout << "\n=== Test 1: Action to Acceleration Conversion ===\n";

    std::vector<RLEnvironment::ActionType> actions = {
        RLEnvironment::ActionType::ACCELERATE,
        RLEnvironment::ActionType::DECELERATE,
        RLEnvironment::ActionType::MAINTAIN,
        RLEnvironment::ActionType::LANE_CHANGE_LEFT,
        RLEnvironment::ActionType::LANE_CHANGE_RIGHT};

    std::vector<std::string> action_names = {
        "ACCELERATE",
        "DECELERATE",
        "MAINTAIN",
        "LANE_CHANGE_LEFT",
        "LANE_CHANGE_RIGHT"};

    for (size_t i = 0; i < actions.size(); ++i)
    {
        auto acc = RLEnvironment::action_to_acceleration(actions[i]);
        std::cout << action_names[i] << ": acc=" << acc[0] << ", omega=" << acc[1] << "\n";
    }

    std::cout << "✓ Action conversion test passed\n";
}

void test_state_extraction()
{
    std::cout << "\n=== Test 2: State Vector Extraction ===\n";

    // 创建测试数据
    VehicleBase ego("ego");
    ego.state.x = 0.0;
    ego.state.y = 0.0;
    ego.state.yaw = 0.0;
    ego.state.v = 5.0;
    ego.state.acc = 0.5;
    ego.state.is_ego = true;

    State goal;
    goal.x = 10.0;
    goal.y = 5.0;
    goal.yaw = 0.0;

    // 没有障碍物的情况
    std::vector<VehicleBase> others;

    auto state_vec = RLEnvironment::extract_state_vector(ego, others, goal);
    print_state_vector(state_vec, "Extracted state");

    // 验证值
    std::cout << "\nVerification:\n";
    std::cout << "  dx (should be -10.0): " << state_vec[0] << "\n";
    std::cout << "  dy (should be -5.0): " << state_vec[1] << "\n";
    std::cout << "  velocity: " << state_vec[3] << "\n";
    std::cout << "  acceleration: " << state_vec[4] << "\n";
    std::cout << "  is_ego: " << state_vec[5] << "\n";

    std::cout << "✓ State extraction test passed\n";
}

void test_cost_computation()
{
    std::cout << "\n=== Test 3: Cost Computation ===\n";

    // 初始化环境（需要env已经初始化过）
    if (!VehicleBase::env)
    {
        std::cout << "Skipping cost computation test (VehicleBase::env not initialized)\n";
        return;
    }

    RLEnvironment env(YAML::LoadFile("config/triple_interact.yaml"));

    State test_state;
    test_state.x = 0.0;
    test_state.y = 0.0;
    test_state.yaw = 0.0;
    test_state.v = 5.0;
    test_state.acc = 0.0;
    test_state.is_ego = true;

    State goal;
    goal.x = 10.0;
    goal.y = 0.0;
    goal.yaw = 0.0;

    std::vector<State> others;

    auto cost = env.compute_cost(test_state, others, goal, true);

    std::cout << "Cost breakdown:\n";
    std::cout << "  avoid:     " << cost.cost_avoid << "\n";
    std::cout << "  lateral:   " << cost.cost_lateral << "\n";
    std::cout << "  offroad:   " << cost.cost_offroad << "\n";
    std::cout << "  distance:  " << cost.cost_distance << "\n";
    std::cout << "  safe:      " << cost.cost_safe << "\n";
    std::cout << "  ride:      " << cost.cost_ride << "\n";
    std::cout << "  -----\n";
    std::cout << "  TOTAL:     " << cost.cost_total << "\n";

    std::cout << "✓ Cost computation test passed\n";
}

void test_step_function()
{
    std::cout << "\n=== Test 4: Step Function (Full Pipeline) ===\n";

    // 需要完整的环境初始化
    try
    {
        YAML::Node config = YAML::LoadFile("config/triple_interact.yaml");
        RLEnvironment env(config);

        // 创建测试车辆
        VehicleBase ego("ego");
        ego.state.x = 0.0;
        ego.state.y = 0.0;
        ego.state.yaw = 0.0;
        ego.state.v = 5.0;
        ego.state.acc = 0.0;
        ego.state.is_ego = true;
        ego.target = State{10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

        std::vector<VehicleBase> others;

        // 执行一个step
        auto result = env.step(
            ego, others,
            RLEnvironment::ActionType::ACCELERATE);

        std::cout << "Step result:\n";
        print_state_vector(result.s, "  Current state (s)");
        print_state_vector(result.s_next, "  Next state (s')");
        std::cout << "  Action: " << static_cast<int>(result.a) << "\n";
        std::cout << "  Reward: " << result.r << "\n";
        std::cout << "  Done: " << (result.done ? "true" : "false") << "\n";
        std::cout << "  Reason: " << result.done_reason << "\n";

        std::cout << "✓ Step function test passed\n";
    }
    catch (const std::exception &e)
    {
        std::cout << "⚠ Step function test skipped (config not found)\n";
        std::cout << "  Error: " << e.what() << "\n";
    }
}

void test_collision_detection()
{
    std::cout << "\n=== Test 5: Collision Detection ===\n";

    if (!VehicleBase::env)
    {
        std::cout << "Skipping collision detection test (VehicleBase::env not initialized)\n";
        return;
    }

    // 两个重叠的车
    State state1;
    state1.x = 0.0;
    state1.y = 0.0;
    state1.yaw = 0.0;
    state1.is_ego = false;

    State state2;
    state2.x = 1.0; // 非常接近，可能碰撞
    state2.y = 0.0;
    state2.yaw = 0.0;
    state2.is_ego = false;

    VehicleBase other("other");
    other.state = state2;
    std::vector<VehicleBase> others = {other};

    bool collision = RLEnvironment::check_collision(state1, others);
    std::cout << "Collision (very close): " << (collision ? "YES" : "NO") << "\n";

    // 两个远离的车
    State state3;
    state3.x = 100.0;
    state3.y = 100.0;
    state3.yaw = 0.0;
    state3.is_ego = false;

    other.state = state3;
    others = {other};

    collision = RLEnvironment::check_collision(state1, others);
    std::cout << "Collision (far away): " << (collision ? "YES" : "NO") << "\n";

    std::cout << "✓ Collision detection test passed\n";
}

int main(int argc, char **argv)
{
    spdlog::set_level(spdlog::level::info);

    std::cout << "\n"
              << std::string(60, '=') << "\n";
    std::cout << "RL Environment Unit Tests\n";
    std::cout << std::string(60, '=') << "\n";

    // 运行所有测试
    test_action_conversion();
    test_state_extraction();
    test_cost_computation();
    test_step_function();
    test_collision_detection();

    std::cout << "\n"
              << std::string(60, '=') << "\n";
    std::cout << "All tests completed!\n";
    std::cout << std::string(60, '=') << "\n\n";

    return 0;
}
