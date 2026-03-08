#pragma once
#include "action_evaluator.hpp"
#include "planner.hpp"

class MCTSEvaluator : public ActionEvaluator
{
public:
    MCTSEvaluator(const std::vector<StateList> &other_traj, const YAML::Node &cfg) : mcts_(other_traj, cfg) {}
    std::shared_ptr<Node> evaluate(std::shared_ptr<Node> node) override
    {
        return mcts_.excute(node);
    }

private:
    MonteCarloTreeSearch mcts_;
};