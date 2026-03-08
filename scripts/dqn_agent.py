"""
DQN 智能体示例实现

配合 C++ RLEnvironment 使用：
- 状态维度: 11
- 动作数量: 5 (ACCELERATE, DECELERATE, MAINTAIN, LANE_LEFT, LANE_RIGHT)
- 奖励: -cost (通常在 -10 到 +100 之间)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from typing import Tuple, List


class DQNNetwork(nn.Module):
    """
    深度Q网络
    
    输入: 状态向量 (batch_size, 11)
    输出: Q值向量 (batch_size, 5)
    
    Q(s, a) 表示在状态s执行动作a的长期价值估计
    """
    
    def __init__(self, state_dim: int = 11, action_dim: int = 5, hidden_dim: int = 128):
        super(DQNNetwork, self).__init__()
        
        self.net = nn.Sequential(
            # 输入层: 11维状态向量
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            
            # 隐藏层1
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            
            # 隐藏层2 (可选，提升表示能力)
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            
            # 输出层: 5维Q值向量，每个维度对应一个动作
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            state: (batch_size, 11) 或 (11,)
        
        Returns:
            q_values: (batch_size, 5) 或 (5,)
        """
        return self.net(state)


class ReplayBuffer:
    """
    经验回放缓冲区
    
    存储转移: (s, a, r, s', done)
    DQN通过从缓冲区中采样来打破样本之间的相关性
    """
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool):
        """
        添加转移到缓冲区
        
        Args:
            state: 当前状态 (11,)
            action: 动作 (0-4)
            reward: 奖励 (标量)
            next_state: 下一状态 (11,)
            done: 是否终止 (bool)
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple:
        """
        从缓冲区中采样一个批次
        
        Returns:
            (states, actions, rewards, next_states, dones) - 都是numpy数组
        """
        batch = random.sample(self.buffer, batch_size)
        
        states = np.array([item[0] for item in batch])
        actions = np.array([item[1] for item in batch])
        rewards = np.array([item[2] for item in batch])
        next_states = np.array([item[3] for item in batch])
        dones = np.array([item[4] for item in batch])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    """
    DQN智能体
    
    核心算法:
    1. ε-greedy策略探索
    2. Bellman更新: Q(s,a) = r + γ·max_a'Q(s',a')
    3. 目标网络稳定训练
    """
    
    def __init__(self,
                 state_dim: int = 11,
                 action_dim: int = 5,
                 learning_rate: float = 1e-3,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.05,
                 epsilon_decay: float = 0.995,
                 device: str = 'cpu'):
        """
        初始化DQN智能体
        
        Args:
            state_dim: 状态维度 (11)
            action_dim: 动作数量 (5)
            learning_rate: 学习率
            gamma: 折扣因子，控制长期vs短期收益的权重
            epsilon_start: 初始探索率
            epsilon_end: 最小探索率
            epsilon_decay: 探索率衰减系数
            device: 计算设备 ('cpu' 或 'cuda')
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.device = torch.device(device)
        
        # ε-greedy参数
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # 主网络和目标网络
        self.q_network = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_network = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # 目标网络仅用于推理
        
        # 优化器
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        
        # 训练统计
        self.step_count = 0
        self.episode_count = 0
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        ε-greedy策略选择动作
        
        Args:
            state: 状态向量 (11,)
            training: 是否训练模式(使用epsilon-greedy)，测试模式(纯贪心)
        
        Returns:
            action: 0-4的整数
        """
        # 探索: 概率ε随机选择
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        # 利用: 选择Q值最大的动作
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax(dim=1).item()
    
    def train_step(self, batch: Tuple) -> float:
        """
        执行一步Bellman更新
        
        Args:
            batch: (states, actions, rewards, next_states, dones)
        
        Returns:
            loss: 本次更新的损失值
        
        Bellman更新公式:
        ┌─────────────────────────────────────────────┐
        │ Q(s,a) := r + γ·max_a' Q_target(s', a')   │
        │                                             │
        │ loss = (Q_main(s,a) - target)²             │
        └─────────────────────────────────────────────┘
        
        其中:
        - Q_main: 主网络，用于选择动作
        - Q_target: 目标网络，用于估计目标Q值
        - γ: 折扣因子
        """
        states, actions, rewards, next_states, dones = batch
        
        # 转换为tensor
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # ========== 计算Q(s,a) ==========
        # 主网络预测当前状态的Q值
        q_values = self.q_network(states)  # (batch_size, 5)
        # 选择执行的动作对应的Q值
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)  # (batch_size,)
        
        # ========== 计算目标Q值 ==========
        with torch.no_grad():
            # 目标网络预测下一状态的Q值
            next_q_values = self.target_network(next_states)  # (batch_size, 5)
            # 选择Q值最大的下一动作
            max_next_q = next_q_values.max(1)[0]  # (batch_size,)
            # Bellman: target = r + γ·max_a'Q(s',a')
            # 如果终止，则target = r (没有未来奖励)
            target_q = rewards + self.gamma * max_next_q * (1 - dones)
        
        # ========== 计算损失 ==========
        loss = self.loss_fn(q_value, target_q)
        
        # ========== 反向传播和优化 ==========
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        self.step_count += 1
        
        return loss.item()
    
    def update_target_network(self):
        """
        更新目标网络
        
        定期将主网络的参数复制到目标网络，稳定训练
        """
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def decay_epsilon(self):
        """
        衰减探索率ε
        
        随着训练进行，逐步减少随机探索，增加利用
        """
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'step_count': self.step_count,
            'episode_count': self.episode_count,
            'epsilon': self.epsilon
        }, path)
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.step_count = checkpoint['step_count']
        self.episode_count = checkpoint['episode_count']
        self.epsilon = checkpoint['epsilon']


# ========== 训练框架示例 ==========

def train_dqn(env, agent: DQNAgent, num_episodes: int = 1000, 
              batch_size: int = 32, update_target_interval: int = 100):
    """
    DQN训练主循环
    
    Args:
        env: RLEnvironment (C++侧)
        agent: DQNAgent实例
        num_episodes: 训练episode数
        batch_size: 批大小
        update_target_interval: 多少步后更新目标网络
    """
    replay_buffer = ReplayBuffer(capacity=10000)
    
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        # ========== 重置环境 ==========
        state = env.reset(ego_vehicle)
        episode_reward = 0
        episode_length = 0
        
        # ========== Episode循环 ==========
        done = False
        while not done and episode_length < 200:
            # 1. 选择动作 (ε-greedy)
            action = agent.select_action(state, training=True)
            
            # 2. 执行动作
            result = env.step(ego_vehicle, other_vehicles, action)
            next_state = result.s_next
            reward = result.r
            done = result.done
            
            # 3. 存储经验
            replay_buffer.push(state, action, reward, next_state, done)
            
            # 4. 训练 (如果缓冲区足够大)
            if len(replay_buffer) > batch_size:
                batch = replay_buffer.sample(batch_size)
                loss = agent.train_step(batch)
                
                # 定期更新目标网络
                if agent.step_count % update_target_interval == 0:
                    agent.update_target_network()
            
            # 5. 更新状态和奖励
            state = next_state
            episode_reward += reward
            episode_length += 1
        
        # ========== Episode结束 ==========
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # 衰减ε
        agent.decay_epsilon()
        
        # 定期输出日志
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_length = np.mean(episode_lengths[-10:])
            print(f"Episode {episode+1}/{num_episodes} | "
                  f"avg_reward={avg_reward:.2f} | "
                  f"avg_length={avg_length:.1f} | "
                  f"epsilon={agent.epsilon:.4f}")
        
        # 定期保存
        if (episode + 1) % 100 == 0:
            agent.save(f'dqn_checkpoint_ep{episode+1}.pt')
    
    return episode_rewards, episode_lengths


# ========== 推理/测试 ==========

def test_agent(env, agent: DQNAgent, num_episodes: int = 10):
    """
    测试训练好的DQN智能体
    
    Args:
        env: RLEnvironment
        agent: 训练好的DQNAgent
        num_episodes: 测试episode数
    """
    test_rewards = []
    collision_count = 0
    goal_count = 0
    
    for episode in range(num_episodes):
        state = env.reset(ego_vehicle)
        episode_reward = 0
        done = False
        
        while not done:
            # 纯贪心策略 (无探索)
            action = agent.select_action(state, training=False)
            result = env.step(ego_vehicle, other_vehicles, action)
            
            episode_reward += result.r
            
            # 统计终止原因
            if result.done_reason == 'collision':
                collision_count += 1
            elif result.done_reason == 'goal_reached':
                goal_count += 1
            
            state = result.s_next
            done = result.done
        
        test_rewards.append(episode_reward)
    
    success_rate = goal_count / num_episodes * 100
    collision_rate = collision_count / num_episodes * 100
    
    print(f"\n=== Test Results ===")
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"Collision Rate: {collision_rate:.1f}%")
    print(f"Avg Reward: {np.mean(test_rewards):.2f}")
    
    return test_rewards


# ========== 使用示例 ==========

if __name__ == "__main__":
    """
    注意：这是伪代码框架，实际使用需要：
    1. 实现C++ RLEnvironment 的Python绑定 (pybind11或ctypes)
    2. 初始化ego_vehicle和other_vehicles
    3. 加载YAML配置
    """
    
    # 初始化（需要补全）
    # env = RLEnvironment('config/triple_interact.yaml')
    # ego_vehicle = Vehicle(...)
    # other_vehicles = [Vehicle(...), ...]
    
    # 创建智能体
    # agent = DQNAgent(state_dim=11, action_dim=5, learning_rate=1e-3)
    
    # 训练
    # episode_rewards, episode_lengths = train_dqn(
    #     env, agent,
    #     num_episodes=500,
    #     batch_size=32,
    #     update_target_interval=100
    # )
    
    # 保存模型
    # agent.save('dqn_final.pt')
    
    # 测试
    # test_rewards = test_agent(env, agent, num_episodes=10)
    
    print("DQN Agent framework loaded successfully.")
    print("State dimension: 11")
    print("Action dimension: 5 (ACCEL, DECEL, MAINTAIN, LEFT, RIGHT)")
    print("See docstrings for detailed usage.")
