#pragma once
#include <vector>
#include <fstream>
#include <string>
#include "utils.hpp"

class DQNAgent
{
public:
    // DQNAgent() = default;
    // 为什么输入是状态维度和动作维度，思路是什么？构造函数输入变量定义的思路？
    // reward_log_path: path where episode rewards are recorded
    // append: if true, the file will be opened in append mode (old data preserved);
    //         if false (the default) the file is truncated/overwritten each time
    //         an agent is constructed. This avoids data accumulating over multiple
    //         program executions unless explicitly desired.
    DQNAgent(int state_dim, int action_dim,
             const std::string &reward_log_path = "episode_rewards.csv",
             bool append = false);

    // 这个就是调用时候用的接口，输入是当前状态，输出是动作的ID（整数），这个函数的实现就是根据当前状态选择一个动作，这个动作就是智能体在当前状态下应该采取的行动。
    int select_action(const std::vector<double> &state);

    // 训练用的接口，输入是当前状态、动作、奖励、下一个状态和是否结束，这个函数的实现就是根据这些输入来更新智能体的内部模型，使得智能体能够更好地选择动作。
    void observe(const std::vector<double> &state,
                 int action,
                 double reward,
                 const std::vector<double> &next_state,
                 bool done);
    bool save_weights(const std::string &path);
    bool load_weights(const std::string &path);
    bool save_epsilon_episodeCount(const std::string &path);
    bool load_epsilon_episodeCount(const std::string &path);
    void init_weights();
    void init_meta();

private:
    int state_dim_;
    int action_dim_;
    // 这三个参数是更新Q值函数的超参数，alpha是学习率，gamma是折扣因子，epsilon是探索率，这些参数的选择会影响智能体的学习效果。
    // 公式是：Q(s, a) = Q(s, a) + alpha * (reward + gamma * max(Q(next_state)) - Q(s, a))，其中max(Q(next_state))是下一个状态的最大Q值。
    // 公式含义是
    double alpha_;                             // learning rate
    double gamma_;                             // discount
    double epsilon_;                           // exploration
                                               // 这里为什么是一个二维的vector，根据线性qlearning，不是把它们要串联成一个一维的vector吗？
    std::vector<std::vector<double>> weights_; // [action][state_dim]

    // 统计当前 episode 的累计 reward
    double episode_reward_ = 0.0;
    int episode_count_ = 0;

    // 暂时用随机策略，占位
    std::mt19937 rng_;

    // Episode reward tracking
    std::string reward_log_path_;
    bool append_mode_ = false; // whether log file was opened in append mode
    std::ofstream reward_log_file_;
    void save_episode_reward();

    // reset counters (useful when you want to treat a new round/run as fresh)
    void reset_episode_counter()
    {
        episode_count_ = 0;
        episode_reward_ = 0.0;
    }

    double compute_q(const std::vector<double> &state, int action);
};
