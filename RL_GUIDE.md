# RL环境重构指南：从MCTS到DQN

## 核心概念

### 1. 什么被替换了？
```
MCTS方式（原系统）:
  state → [MCTS Rollout] → evaluate cost → select best action
  
DQN方式（新系统）:
  state → [神经网络 Q(s,a)] → select argmax_a Q(s,a) → execute action
```

**关键区别：**
- **MCTS**: 在线显式模拟未来轨迹 (rollout)，评估每个动作的长期成本
- **DQN**: 离线学习，将未来价值编码进神经网络的参数里

### 2. Bellman方程
```
Q(s,a) = r + γ · max_a' Q(s', a')
```
- `r`: 执行 `a` 后的立即奖励
- `γ`: 折扣系数 (通常 0.9-0.99)
- `max_a' Q(s', a')`: 下一状态的最优价值

## RL环境接口详解

### 核心 `step()` 函数的含义

```cpp
StepResult step(VehicleBase& ego,
                const std::vector<VehicleBase>& others,
                ActionType action);
```

**返回值解析：**
```
s       : 执行动作前的状态向量 (11维)
a       : 执行的动作 (离散，5个选项)
r       : 立即奖励 = -cost
s'      : 执行动作后的下一状态 (11维)
done    : 是否终止 (碰撞/离线/到达目标)
reason  : 终止原因 (便于调试)
```

### 状态向量结构 (11维)

| 索引 | 含义 | 说明 |
|------|------|------|
| 0-2 | 相对目标位置 | [dx, dy, d_yaw] - 自车与目标的相对位置 |
| 3 | 速度 | 自车当前速度 |
| 4 | 加速度 | 自车当前加速度 |
| 5 | 车辆类型 | 1=自车(有挂车), 0=普通车 |
| 6-8 | 相对最近障碍物 | [rel_x, rel_y, rel_v] |
| 9 | 横向偏离 | 自车x相对目标x的偏离 |
| 10 | 航向偏离 | 自车yaw相对目标yaw的偏离 |

**为什么这样设计？**
- 充分利用规划器已有的物理信息
- 相对表示→与起点无关，便于泛化
- 低维→网络收敛快

### 动作空间

```cpp
enum class ActionType : int {
    ACCELERATE = 0,          // 加速: acc = +2.5 m/s²
    DECELERATE = 1,          // 减速: acc = -2.5 m/s²
    MAINTAIN = 2,            // 保持: acc = 0
    LANE_CHANGE_LEFT = 3,    // 左转: omega = π/4
    LANE_CHANGE_RIGHT = 4,   // 右转: omega = -π/4
    NUM_ACTIONS = 5
};
```

对应DQN的输出头：
```
Q网络输出: [Q(s,0), Q(s,1), Q(s,2), Q(s,3), Q(s,4)]
           其中每个值是该动作的价值估计

推理时:   action = argmax_i Q(s,i)
```

### 奖励设计

```
reward = -cost

cost 的组成：
├── collision      1000*avoid       (碰撞最严重)
├── lateral        700*lateral_dist (横向对齐)
├── offroad        300*offroad      (保持在道路内)
├── distance       600*distance     (到达目标)
├── ride           1*comfort        (平滑驾驶)
└── safe_zone      100*safe         (保持安全距离)
```

**终止条件的特殊奖励：**
```cpp
if (collision)    → reward = -1000  (额外惩罚)
if (offroad)      → reward = -500
if (goal_reached) → reward = +100   (额外奖励)
```

## 实现步骤

### Step 1: 集成 RLEnvironment

在你的 DQN 代码中：
```python
# Python 侧（对接C++ RLEnvironment）
from rl_env_wrapper import RLEnvCpp  # C++绑定

env = RLEnvCpp(config_path="config.yaml")
env.reset(ego_vehicle)

# 交互循环
for episode in range(num_episodes):
    s = env.reset(ego_vehicle)
    
    for step in range(max_steps):
        # DQN推理
        a = agent.select_action(s, epsilon=eps)  # epsilon-greedy
        
        # 环境执行
        result = env.step(ego_vehicle, other_vehicles, a)
        s, a, r, s_next, done = result.s, result.a, result.r, result.s_next, result.done
        
        # 存入经验缓冲区
        replay_buffer.push(s, a, r, s_next, done)
        
        # 训练
        if len(replay_buffer) > batch_size:
            batch = replay_buffer.sample(batch_size)
            loss = agent.train_step(batch)
        
        s = s_next
        if done:
            break
```

### Step 2: DQN 神经网络设计

**最小可行网络：**
```python
import torch.nn as nn

class DQNNetwork(nn.Module):
    def __init__(self, state_dim=11, action_dim=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, s):
        """输入: 状态向量 (batch_size, 11)
           输出: Q值 (batch_size, 5)"""
        return self.net(s)
```

### Step 3: Bellman更新

```python
def compute_loss(batch):
    """
    batch: (s, a, r, s_next, done)
    """
    s = batch['s']           # (batch_size, 11)
    a = batch['a']           # (batch_size,)
    r = batch['r']           # (batch_size,)
    s_next = batch['s_next'] # (batch_size, 11)
    done = batch['done']     # (batch_size,)
    
    # Current Q value: Q(s, a)
    q_values = q_network(s)  # (batch_size, 5)
    q_value = q_values.gather(1, a.unsqueeze(1)).squeeze(1)
    
    # Target Q value: r + γ·max_a'Q(s', a')
    with torch.no_grad():
        next_q_values = target_network(s_next)  # (batch_size, 5)
        next_q_value = next_q_values.max(1)[0]  # (batch_size,)
        target_q = r + 0.99 * next_q_value * (1 - done.float())
    
    # MSE Loss
    loss = nn.MSELoss()(q_value, target_q)
    return loss
```

### Step 4: 贪心与探索

```python
def select_action(state, epsilon):
    if random() < epsilon:
        # 探索：随机动作
        return random.randint(0, 4)
    else:
        # 利用：选最优动作
        with torch.no_grad():
            q_values = q_network(state)
            return q_values.argmax().item()

# 训练中衰减epsilon
epsilon = epsilon_start * epsilon_decay ** episode
```

## 物理模型和成本函数保持不变

**重要：不修改以下内容**
```cpp
// ✓ 保留
State kinematic_propagate(state, acc_vector, dt)
CostBreakdown compute_cost(state, obstacles, ...)
check_collision(state, obstacles)
check_offroad(state)

// ✓ 保留（物理参数）
vehicle.length, vehicle.width
tractor.length, tractor_width
tail.length, tail.width
...
```

## 数据流对比

### MCTS方式（原系统）
```
Vehicle.execute()
├─> planner.planning(ego_vehicle)
│   ├─> get_prediction(ego, others)
│   ├─> KLevelPlanner.forward_simulate()
│   │   └─> MCTS.execute()
│   │       ├─> 显式展开搜索树
│   │       ├─> 每个Node评估: calc_cur_value()
│   │       └─> 反向传播价值
│   └─> 返回最优动作
├─> kinematic_propagate(state, action, dt)
└─> 更新state
```

### DQN方式（新系统）
```
RL环境初始化
├─> 训练循环
│   └─> for episode:
│       └─> s = env.reset(ego)
│           for step:
│           ├─> a = Q-network(s).argmax()  ← 替代MCTS
│           ├─> result = env.step(ego, others, a)
│           │   ├─> action → acceleration
│           │   ├─> kinematic_propagate()
│           │   ├─> compute_cost()
│           │   ├─> r = -cost
│           │   └─> 提取 s_next
│           ├─> replay_buffer.push(s,a,r,s_next,done)
│           ├─> train_step(batch)  ← 更新Q-network
│           └─> s = s_next
```

## 调试建议

### 1. 验证环境的正确性
```cpp
// 在 rl_env.cpp 中启用日志
spdlog::set_level(spdlog::level::debug);

// 运行单步测试
RLEnvironment env(config);
auto result = env.step(ego, others, ActionType::ACCELERATE);

// 检查：
// - result.s 与 result.s_next 的差异
// - result.r 是否合理（通常 -10~0）
// - done 条件是否正确
```

### 2. 验证状态向量
```cpp
auto state_vec = RLEnvironment::extract_state_vector(ego, others, goal);
// 检查：
// - [0-2] 相对目标位置是否变化
// - [3-5] 自车状态是否更新
// - [6-8] 障碍物信息是否正确
```

### 3. 检查奖励信号
```python
# 训练过程中监控：
print(f"Episode {ep}: "
      f"cumulative_reward={sum_reward}, "
      f"avg_q={avg_q}, "
      f"loss={loss}")

# 期望：
# - 随着训练，cumulative_reward应该增长
# - loss应该下降
# - 碰撞/离线的频率应该减少
```

## 与原MCTS的优劣对比

| 方面 | MCTS | DQN |
|------|------|-----|
| 推理速度 | 慢（需要rollout） | 快（前向传播） |
| 泛化能力 | 差（每次重新规划） | 好（学到通用策略） |
| 分布外场景 | 差 | 可通过更多数据改进 |
| 样本效率 | 差（离线环境） | 中等（replay buffer） |
| 实现复杂度 | 高 | 中等 |
| 可解释性 | 中等（树结构可视化） | 差（黑盒） |

## 下一步工作

1. **实现DQN智能体**：完成Python侧的DQN实现
2. **C++绑定**：使用pybind11或ctypes将RLEnvironment暴露给Python
3. **训练循环**：运行episode收集数据并训练
4. **评估指标**：
   - 成功率 (到达目标的百分比)
   - 碰撞率
   - 平均step数
   - 平均奖励
5. **对比实验**：MCTS vs DQN的性能对比

## 文件对应关系

```
原MCTS系统:
├─ include/planner.hpp
├─ src/planner.cpp           (MCTS核心)
├─ src/utils.cpp             (kinematic_propagate)
└─ src/vehicle.cpp           (vehicle.execute())

新DQN系统:
├─ include/rl_env.hpp        ← 新增：标准RL接口
├─ src/rl_env.cpp            ← 新增：环境实现
├─ include/planner.hpp       ✓ 保留（可选用)
├─ src/utils.cpp             ✓ 保留（kinematic_propagate)
└─ scripts/dqn_agent.py      ← 新增：DQN算法
```
