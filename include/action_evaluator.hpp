#pragma once
#include <memory>

// planner —— uses —— evaluator —— uses —— policy

class Node;
class ActionEvaluator
{
public:
    virtual std::shared_ptr<Node> evaluate(std::shared_ptr<Node> node) = 0;
    virtual ~ActionEvaluator() = default;
};
