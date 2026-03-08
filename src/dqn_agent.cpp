#include "dqn_agent.hpp"
#include <iostream>
#include <spdlog/spdlog.h>
#include <filesystem>
#include <sstream>

// rng_这是什么?这是一个随机数生成器，用于在select_action函数中生成随机动作。std::random_device{}()是一个随机数种子，确保每次运行程序时生成的随机数序列不同。

// By default append=false, so new log file is created/truncated on each agent instantiation
DQNAgent::DQNAgent(int state_dim, int action_dim, const std::string &reward_log_path,
                   bool append)
    : state_dim_(state_dim),
      action_dim_(action_dim), alpha_(0.05),
      gamma_(0.95),
      epsilon_(0.5),
      rng_(std::random_device{}()),
      reward_log_path_(reward_log_path),
      append_mode_(append)
{
    // 这一步是做了什么？

    init_weights();
    init_meta();

    // Open reward log file using either append or truncate depending on append_mode_
    std::ios_base::openmode mode = std::ios::out;
    if (append_mode_)
        mode |= std::ios::app;
    else
        mode |= std::ios::trunc;

    reward_log_file_.open(reward_log_path_, mode);
    if (reward_log_file_.is_open())
    {
        if (!append_mode_)
        {
            // fresh file: always write header
            reward_log_file_ << "episode,reward\n";
        }
        else
        {
            // append mode: only add header if file empty
            reward_log_file_.seekp(0, std::ios::end);
            if (reward_log_file_.tellp() == 0)
                reward_log_file_ << "episode,reward\n";
        }
        reward_log_file_.flush();
    }
    else
    {
        spdlog::warn("Failed to open reward log file: {}", reward_log_path_);
    }

    // Try to load saved checkpoints from current working directory
    try
    {
        namespace fs = std::filesystem;
        fs::path weights_file = fs::current_path() / "dqn_weights.txt";
        fs::path meta_file = fs::current_path() / "dqn_meta.txt";

        if (fs::exists(weights_file))
        {
            if (load_weights(weights_file.string()))
                spdlog::info("Loaded DQN weights from {}", weights_file.string());
            else
                spdlog::warn("Failed to load DQN weights from {}", weights_file.string());
        }

        if (fs::exists(meta_file))
        {
            if (load_epsilon_episodeCount(meta_file.string()))
                spdlog::info("Loaded DQN meta (epsilon, episode_count) from {}", meta_file.string());
            else
                spdlog::warn("Failed to load DQN meta from {}", meta_file.string());
        }
    }
    catch (const std::exception &e)
    {
        spdlog::error("Exception while loading DQN checkpoints in constructor: {}", e.what());
    }
}

int DQNAgent::select_action(const std::vector<double> &state)
{
    std::uniform_real_distribution<double> uni(0.0, 1.0);

    if (uni(rng_) < epsilon_)
    {
        std::uniform_int_distribution<int> dist(0, action_dim_ - 1);
        return dist(rng_);
    }

    // greedy
    double best_q = -1e9;
    int best_action = 0;

    for (int a = 0; a < action_dim_; ++a)
    {
        double q = compute_q(state, a);
        if (q > best_q)
        {
            best_q = q;
            best_action = a;
        }
    }

    return best_action;
}
// 这输入咋还空着？hpp文件不都写了吗？声明参数，但暂时不用它们，这样编译器就不会报错。等以后实现这个函数的时候再用这些参数。
// 线性qlearning的核心就是根据当前状态和动作计算Q值，然后根据奖励和下一个状态的最大Q值来更新权重。这个函数就是实现了这个更新过程。
void DQNAgent::observe(const std::vector<double> &state,
                       int action,
                       double reward,
                       const std::vector<double> &next_state,
                       bool done)
{
    // 先根据当前权重矩阵算出动作对应Q值
    double q_sa = compute_q(state, action);

    double max_next_q = 0.0;
    // 这里的done参数是用来判断是否是终止状态的，如果是终止状态，那么就没有下一个状态了，所以max_next_q应该是0，否则就要计算下一个状态的最大Q值。
    if (!done)
    {
        max_next_q = -1e9;
        // 这是根据下一个状态用当前的权重矩阵计算出所有动作的Q值，然后取最大值作为max_next_q，这样不需要知道下一个状态的具体动作，只需要知道下一个状态的最大Q值就可以了。
        for (int a = 0; a < action_dim_; ++a)
        {
            double q = compute_q(next_state, a);
            if (q > max_next_q)
                max_next_q = q;
        }
    }

    double target = reward + gamma_ * max_next_q;
    double td_error = target - q_sa;

    // 有求导
    for (int i = 0; i < state_dim_; ++i)
    {
        weights_[action][i] += alpha_ * td_error * state[i];
    }

    // accumulate reward for episode tracking
    episode_reward_ += reward;
    // print step info every call (so we see values even if 'done' never occurs)
    {
        double sample_weight = weights_[0][0];
        spdlog::info("step reward={} episode_reward={} epsilon={} w00={} done={}",
                     reward, episode_reward_, epsilon_, sample_weight, done);
    }
    if (done)
    {
        episode_count_++;
        // 衰减epsilon，逐渐减少探索率，让智能体更多地利用学到的知识来选择动作，而不是随机选择动作。这里设置了一个最小值0.05，确保智能体仍然有一定的探索能力。
        // epsilon_ = std::max(0.05, epsilon_ * 0.995);
        if (episode_count_ < 200)
        {
            epsilon_ = 0.2;
        }
        else if (episode_count_ < 600)
        {
            epsilon_ = 0.2;
        }
        else
        {
            epsilon_ = 0.3;
        }

        // Save episode reward before reset
        save_episode_reward();
        // checkpoint every 50 episodes: save weights and meta (epsilon + episode count)
        if (episode_count_ % 50 == 0)
        {
            try
            {
                std::filesystem::path rp(reward_log_path_);
                std::filesystem::path dir = rp.parent_path();
                if (dir.empty())
                    dir = std::filesystem::current_path();
                std::string weights_path = (dir / "dqn_weights.txt").string();
                std::string meta_path = (dir / "dqn_meta.txt").string();
                if (save_weights(weights_path))
                    spdlog::info("Saved weights to {}", weights_path);
                else
                    spdlog::warn("Failed to save weights to {}", weights_path);

                if (save_epsilon_episodeCount(meta_path))
                    spdlog::info("Saved meta to {}", meta_path);
                else
                    spdlog::warn("Failed to save meta to {}", meta_path);
            }
            catch (const std::exception &e)
            {
                spdlog::error("Checkpoint save exception: {}", e.what());
            }
        }
        // reset cumulative after episode completes
        episode_reward_ = 0.0;
    }
}

void DQNAgent::save_episode_reward()
{
    if (reward_log_file_.is_open())
    {
        reward_log_file_ << episode_count_ << "," << episode_reward_ << "\n";
        reward_log_file_.flush();
        spdlog::info("Episode {} completed with reward: {}", episode_count_, episode_reward_);
    }
}
// 计算Q的时候是没有reawrd参与的，reward放在了更新权重的过程中。
double DQNAgent::compute_q(const std::vector<double> &state, int action)
{

    double q = 0.0;
    // action所在行的权重向量和状态向量的点积，得到Q值
    for (int i = 0; i < state_dim_; ++i)
        q += weights_[action][i] * state[i];
    return q;
}
// 输入变量是个路径吗？要具体到文件吗？举例来说，如果你想保存权重到当前目录下的weights.txt文件，你就可以调用save_weights("weights.txt")，这样权重就会被保存到这个文件中。同样的，如果你想加载之前保存的权重，你也可以调用load_weights("weights.txt")，这样程序就会从这个文件中读取权重并加载到智能体中。
// 如果现在没有这个文件，save_weights会自动创建一个新的文件来保存权重；如果文件已经存在，save_weights会覆盖这个文件（除非你在调用构造函数时设置了append=true，这样它就会在文件末尾追加内容而不是覆盖）。load_weights则需要确保指定的文件存在并且格式正确，否则它会返回false表示加载失败。
bool DQNAgent::save_weights(const std::string &path)
{
    std::ofstream out(path);
    if (!out.is_open())
        return false;

    out << action_dim_ << " " << state_dim_ << "\n";

    for (int a = 0; a < action_dim_; ++a)
    {
        for (int i = 0; i < state_dim_; ++i)
        {
            out << weights_[a][i] << " ";
        }
        out << "\n";
    }

    out.close();
    return true;
}

bool DQNAgent::load_weights(const std::string &path)
{
    std::ifstream in(path);
    if (!in.is_open())
        return false;

    int action_dim, state_dim;
    in >> action_dim >> state_dim;

    if (action_dim != action_dim_ || state_dim != state_dim_)
    {
        spdlog::error("weight dimension mismatch!");
        return false;
    }

    for (int a = 0; a < action_dim_; ++a)
    {
        for (int i = 0; i < state_dim_; ++i)
        {
            in >> weights_[a][i];
        }
    }

    in.close();
    return true;
}

// save epsilon and episode count
bool DQNAgent::save_epsilon_episodeCount(const std::string &path)
{
    std::ofstream out(path);
    if (!out.is_open())
        return false;

    out << epsilon_ << " " << episode_count_ << "\n";
    out.close();
    return true;
}

bool DQNAgent::load_epsilon_episodeCount(const std::string &path)
{
    std::ifstream in(path);
    if (!in.is_open())
        return false;

    in >> epsilon_ >> episode_count_;
    in.close();
    return true;
}

void DQNAgent::init_weights()
{
    weights_.resize(action_dim_,
                    std::vector<double>(state_dim_, 0.0));
}

void DQNAgent::init_meta()
{
    episode_reward_ = 0.0;
    episode_count_ = 0;
}