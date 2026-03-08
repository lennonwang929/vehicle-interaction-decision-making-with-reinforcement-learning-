# RLEnvironment 集成与对接指南

## 快速开始

### 1. 编译 C++ 环境

在 CMakeLists.txt 中添加 rl_env 的编译：

```cmake
# 添加到 CMakeLists.txt

# ... 现有代码 ...

# RL环境的源文件
add_library(rl_env src/rl_env.cpp)

# 链接依赖
target_link_libraries(rl_env 
    PRIVATE 
        vehicle_base
        Eigen3::Eigen
        yaml-cpp
        spdlog::spdlog
)

# 如果需要Python绑定，添加pybind11
# find_package(pybind11 REQUIRED)
# pybind11_add_module(rl_env_py src/rl_env_binding.cpp)
# target_link_libraries(rl_env_py PRIVATE rl_env)
```

编译：
```bash
cd /home/jing/dev/vehicle-interaction-decision-making-final/vehicle-interaction-decision-making
mkdir -p build
cd build
cmake ..
make -j4
```

### 2. 理解核心数据流

#### 旧方式（MCTS）
```
Vehicle.excute()
  └─> planner.planning(ego)
      └─> MCTS 树搜索 + 显式rollout
          └─> 为每个叶节点评估成本
              └─> 返回最优动作
  └─> kinematic_propagate(action)
  └─> state.update()
```

#### 新方式（DQN）
```
for step in episode:
  ├─> s = env.extract_state_vector(ego, others, goal)
  ├─> a = Q_network(s).argmax()  ← 神经网络替代MCTS
  ├─> (s, a, r, s', done) = env.step(ego, others, a)
  │   ├─> kinematic_propagate(a → acceleration)
  │   ├─> compute_cost(s') → r = -cost
  │   └─> check_collision/offroad/goal
  ├─> replay_buffer.append(s, a, r, s', done)
  ├─> batch = replay_buffer.sample()
  └─> Q_network.update(batch)  ← Bellman更新
```

### 3. 关键接口说明

#### 3.1 初始化环境

```cpp
#include "rl_env.hpp"

// 从YAML配置初始化
RLEnvironment env(config);

// 重置到初始状态
RLEnvironment::StateVector state = env.reset(ego_vehicle);
```

**StateVector 是 11 维的状态表示**
```
state[0-2]:   相对目标位置 (dx, dy, d_yaw)
state[3]:     速度 v
state[4]:     加速度 acc
state[5]:     车辆类型 (1=自车, 0=普通车)
state[6-8]:   相对最近障碍物 (rel_x, rel_y, rel_v)
state[9-10]:  车道偏差 (lateral_error, yaw_error)
```

#### 3.2 执行单步动作

```cpp
// 定义要执行的动作
RLEnvironment::ActionType action = RLEnvironment::ActionType::ACCELERATE;

// 执行步骤：输入 ego, others, action → 输出 (s, a, r, s', done)
RLEnvironment::StepResult result = env.step(ego_vehicle, other_vehicles, action);

// 提取返回值
RLEnvironment::StateVector s = result.s;           // 执行前的状态
RLEnvironment::ActionType a = result.a;            // 执行的动作
double r = result.r;                               // 立即奖励
RLEnvironment::StateVector s_next = result.s_next; // 执行后的状态
bool done = result.done;                           // 是否终止
std::string reason = result.done_reason;           // 终止原因
```

#### 3.3 成本分解（用于调试）

```cpp
std::vector<State> other_states;
for (const auto& other : other_vehicles) {
    other_states.push_back(other.state);
}

RLEnvironment::CostBreakdown cost = env.compute_cost(
    ego_vehicle.state,
    other_states,
    ego_vehicle.target,
    ego_vehicle.state.is_ego
);

// 查看各项成本
std::cout << "Avoid cost: " << cost.cost_avoid << "\n";
std::cout << "Lateral cost: " << cost.cost_lateral << "\n";
std::cout << "Offroad cost: " << cost.cost_offroad << "\n";
std::cout << "Distance cost: " << cost.cost_distance << "\n";
std::cout << "Safe cost: " << cost.cost_safe << "\n";
std::cout << "Ride cost: " << cost.cost_ride << "\n";
std::cout << "Total cost: " << cost.cost_total << "\n";
```

### 4. 集成到现有项目

#### 选项A：替换MCTS（激进）

```cpp
// src/decision_making.cpp (或你的主循环)

#include "rl_env.hpp"

// 在初始化处
RLEnvironment env(config);
DQNNetworkWrapper dqn_network("model.pth");  // 加载预训练模型

// 在主循环中
for (int step = 0; step < max_steps; ++step) {
    // 用DQN替代原来的planner.planning()
    
    // 提取状态
    auto state = RLEnvironment::extract_state_vector(
        ego_vehicle, other_vehicles, ego_vehicle.target
    );
    
    // 用DQN选择动作（替代MCTS）
    int action_id = dqn_network.forward(state);  // 返回 0-4
    RLEnvironment::ActionType action = static_cast<RLEnvironment::ActionType>(action_id);
    
    // 执行动作
    auto result = env.step(ego_vehicle, other_vehicles, action);
    
    // 更新状态
    ego_vehicle.state = result.s_next;
    
    if (result.done) break;
}
```

#### 选项B：并行测试（保守）

```cpp
// 同时运行MCTS和DQN，对比结果

std::pair<Action, StateList> mcts_result = planner.planning(ego);
RLEnvironment::StepResult dqn_result = env.step(ego, others, dqn_action);

// 对比奖励
std::cout << "MCTS cost: " << mcts_cost << "\n";
std::cout << "DQN reward: " << dqn_result.r << "\n";
```

### 5. Python 侧集成（DQN训练）

#### 5.1 使用 ctypes 调用 C++ 库（最简单）

```python
# scripts/dqn_train.py

import ctypes
import numpy as np
from pathlib import Path

# 加载C++编译的共享库
lib_path = Path(__file__).parent.parent / "build" / "librl_env.so"
lib = ctypes.CDLL(str(lib_path))

# 定义接口
class RLEnvWrapper:
    def __init__(self, config_path):
        # 初始化环境的C函数
        self.lib = lib
        self.env = lib.rl_env_create(config_path.encode())
    
    def step(self, action):
        # 调用C函数执行一步
        # ... 需要实现ctypes映射 ...
        pass
    
    def reset(self):
        # 重置环境
        pass

# 使用示例
# env = RLEnvWrapper("config.yaml")
# for episode in range(100):
#     s = env.reset()
#     for step in range(200):
#         a = agent.select_action(s)
#         s, a, r, s_next, done = env.step(a)
```

#### 5.2 使用 pybind11 调用（推荐，类型安全）

首先编写绑定层（src/rl_env_binding.cpp）：

```cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "rl_env.hpp"

namespace py = pybind11;

PYBIND11_MODULE(rl_env_py, m) {
    // 绑定 StateVector
    py::class_<RLEnvironment::StateVector>(m, "StateVector")
        .def(py::init<>())
        .def("__getitem__", [](RLEnvironment::StateVector& sv, int i) {
            return sv[i];
        })
        .def("__setitem__", [](RLEnvironment::StateVector& sv, int i, double v) {
            sv[i] = v;
        })
        .def_readonly("data", &RLEnvironment::StateVector::data);
    
    // 绑定 ActionType
    py::enum_<RLEnvironment::ActionType>(m, "ActionType")
        .value("ACCELERATE", RLEnvironment::ActionType::ACCELERATE)
        .value("DECELERATE", RLEnvironment::ActionType::DECELERATE)
        .value("MAINTAIN", RLEnvironment::ActionType::MAINTAIN)
        .value("LANE_CHANGE_LEFT", RLEnvironment::ActionType::LANE_CHANGE_LEFT)
        .value("LANE_CHANGE_RIGHT", RLEnvironment::ActionType::LANE_CHANGE_RIGHT)
        .export_values();
    
    // 绑定 StepResult
    py::class_<RLEnvironment::StepResult>(m, "StepResult")
        .def_readonly("s", &RLEnvironment::StepResult::s)
        .def_readonly("a", &RLEnvironment::StepResult::a)
        .def_readonly("r", &RLEnvironment::StepResult::r)
        .def_readonly("s_next", &RLEnvironment::StepResult::s_next)
        .def_readonly("done", &RLEnvironment::StepResult::done)
        .def_readonly("done_reason", &RLEnvironment::StepResult::done_reason);
    
    // 绑定 RLEnvironment
    py::class_<RLEnvironment>(m, "RLEnvironment")
        .def(py::init<const YAML::Node&>())
        .def("reset", &RLEnvironment::reset)
        .def("step", &RLEnvironment::step)
        .def_static("action_to_acceleration", &RLEnvironment::action_to_acceleration);
}
```

然后在CMakeLists.txt中：

```cmake
find_package(pybind11 REQUIRED)
pybind11_add_module(rl_env_py src/rl_env_binding.cpp)
target_link_libraries(rl_env_py PRIVATE rl_env vehicle_base)
```

编译后可直接在Python中使用：

```python
from rl_env_py import RLEnvironment, ActionType
import yaml

# 加载配置
with open('config/triple_interact.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 初始化环境
env = RLEnvironment(config)

# 使用
s = env.reset(ego_vehicle)
result = env.step(ego_vehicle, other_vehicles, ActionType.ACCELERATE)
print(f"Reward: {result.r}, Done: {result.done}")
```

### 6. 调试与验证

#### 6.1 验证状态向量提取

```python
import numpy as np

# 提取多个样本的状态向量
states = []
for i in range(10):
    s = RLEnvironment.extract_state_vector(ego, others, goal)
    states.append(s.data)

states = np.array(states)
print("State shapes:", states.shape)  # 应该是 (10, 11)
print("State ranges:")
for i in range(11):
    print(f"  Dim {i}: [{states[:, i].min():.2f}, {states[:, i].max():.2f}]")
```

#### 6.2 验证奖励信号

```python
# 运行几个episode，检查奖励变化

episode_rewards = []
for ep in range(5):
    s = env.reset(ego)
    total_r = 0
    for step in range(100):
        a = random.randint(0, 4)  # 随机动作
        result = env.step(ego, others, a)
        total_r += result.r
        if result.done:
            print(f"Episode {ep}: terminated with reason '{result.done_reason}', "
                  f"total_reward={total_r:.2f}, steps={step}")
            break
```

#### 6.3 对比成本

```python
# 验证成本计算是否合理

# MCTS方式
mcts_cost = planner.planning(ego)[2]  # 假设返回cost

# DQN方式
result = env.step(ego, others, action)
dqn_cost = -result.r

print(f"MCTS cost: {mcts_cost:.3f}")
print(f"DQN cost (from reward): {dqn_cost:.3f}")
```

### 7. 完整示例：从MCTS迁移到DQN

```cpp
// src/decision_making_dqn.cpp

#include "rl_env.hpp"
#include <iostream>

class DQNPlanner {
private:
    RLEnvironment env;
    // DQNNetwork* dqn_network;  // C++侧的网络或调用Python
    
public:
    DQNPlanner(const YAML::Node& cfg) : env(cfg) {}
    
    std::pair<Action, StateList> planning(VehicleBase& ego) const {
        // 1. 提取状态
        auto state = RLEnvironment::extract_state_vector(
            ego, {}, ego.target
        );
        
        // 2. DQN推理（需要调用Python或C++网络）
        int action_id = 0;  // inference_network(state)
        
        // 3. 执行动作
        auto result = env.step(ego, {}, 
                              static_cast<RLEnvironment::ActionType>(action_id));
        
        // 4. 返回兼容的Action和StateList
        // （需要转换回原来的接口）
        return {Action::MAINTAIN, StateList()};
    }
};
```

### 8. 性能与对比

预期性能提升：
| 指标 | MCTS | DQN |
|------|------|-----|
| 推理延迟 | 100-500ms | 1-10ms |
| 首次训练 | N/A | ~1小时 |
| 泛化能力 | 差 | 好 |
| 实时性 | 取决于budget | 实时 |

### 9. 故障排查

#### 问题1：编译错误 "undefined reference to RLEnvironment"
**原因**: CMakeLists.txt中没有链接rl_env库
**解决**: 在target_link_libraries中添加rl_env

#### 问题2：状态向量全0
**原因**: extract_state_vector中的计算有误
**解决**: 添加调试日志，检查ego状态和goal位置

#### 问题3：奖励全是负数
**原因**: cost计算有问题，或者初始配置不合理
**解决**: 检查CostBreakdown的各项值，调整权重

#### 问题4：碰撞检测不工作
**原因**: 通常是box2d计算或overlap检测的问题
**解决**: 打印box2d的坐标，验证位置是否正确

---

下一步建议：
1. 编译C++代码，运行单元测试
2. 实现Python绑定
3. 在Python中运行DQN训练循环
4. 对比MCTS vs DQN的性能
5. 优化超参数（学习率、隐藏层大小等）
