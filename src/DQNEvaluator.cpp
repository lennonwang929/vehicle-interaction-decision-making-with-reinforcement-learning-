#include "DQNEvaluator.hpp"
#include "DQNagent.hpp"
#include "utils.hpp"

std::shared_ptr<Node>
DQNEvaluator::evaluate(std::shared_ptr<Node> root)
{
    // 1. 从 Node 提取 state vector
    // Eigen::VectorXd state_vec =
    //     agent_->extract_state(root);
    std::vector<double> dummy_state;
    // 2. 推理 Q(s, a)
    int action_id = agent_->select_action(dummy_state);

    Action action = static_cast<Action>(action_id);

    // 3. 用现有动力学 forward 一步
    State next_state = utils::kinematic_propagate(
        root->state,
        utils::get_action_value(action),
        dt_);

    // 4. 构造新 Node，这个node的构造为什么是这样？要看node的构造函数。
    // auto next_node = std::make_shared<Node>(
    //     next_state,
    //     root->level + 1,
    //     root,
    //     action,
    //     StateList(),
    //     root->target);
    // 这里要返回node，是为了能跟mcts替换掉吧，如果单纯用DQN,不需要返回node，直接返回action就行了，但是现在是要跟mcts替换掉，所以要返回node，这样mcts就能继续往下扩展了。
    return root;
    // return next_node;
}
