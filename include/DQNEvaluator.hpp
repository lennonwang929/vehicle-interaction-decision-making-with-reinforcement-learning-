#pragma once

#include "action_evaluator.hpp"
// 智能指针需要用到这个头文件
#include <memory>

class DQNAgent; // 前向声明
class VehicleBase;
class Target;

class DQNEvaluator : public ActionEvaluator
{
public:
    // 为什么这个构造函数是这样的，思路是什么？这个肯定得知道DQN的实现原理，不需要知道，这里是调用DQNAgent的接口，DQNAgent是“被训练的长期主体”，DQNEvaluator是“短期调用它的工具”，所以DQNEvaluator需要持有一个DQNAgent的智能指针，这样就可以在evaluate函数中调用DQNAgent的接口来得到动作了。dt是时间步长，可能在evaluate函数中需要用到，所以也传进来。
    // agent 是“被训练的长期主体”，
    // evaluator 是“短期调用它的工具”
    // 我这个 evaluator 的时间步长是多少”
    DQNEvaluator(std::shared_ptr<DQNAgent> agent,
                 double dt);
    // 这个是继承自ActionEvaluator的evaluate函数，按它的输入来写就行，必须有这个函数
    std::shared_ptr<Node>
    evaluate(std::shared_ptr<Node> root) override;

private:
    std::shared_ptr<DQNAgent> agent_; // 共享是合理的
    double dt_;
};
