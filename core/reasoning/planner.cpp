#pragma once
#include <string>
#include <vector>
#include <functional>
#include <memory>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <random>

/*
=====================================================
  TITANCORE AGI: GOAL-DIRECTED PLANNER (MCTS)
=====================================================
  Enables the AGI to plan multi-step action sequences
  toward a defined goal using Monte Carlo Tree Search.

  Implements:
  - MCTS with UCB1 selection
  - Neural-guided rollout policy
  - Goal state evaluation via value network
  - Hierarchical goal decomposition
  - Plan extraction and execution
=====================================================
*/

struct Action {
    std::string name;
    std::string description;
    float       cost;         // computational / time cost
};

struct PlanState {
    std::string  description;
    float        value;        // estimated goal-proximity (0 = far, 1 = goal reached)
    bool         is_terminal;
    std::vector<std::string> achieved_subgoals;
};

struct MCTSNode {
    PlanState   state;
    Action      action_taken;  // action that led to this node
    int         visits  = 0;
    float       total_value = 0.0f;

    std::weak_ptr<MCTSNode>              parent;
    std::vector<std::shared_ptr<MCTSNode>> children;

    float ucb1(float exploration = 1.414f) const {
        if (visits == 0) return std::numeric_limits<float>::infinity();
        auto p = parent.lock();
        int parent_visits = p ? p->visits : 1;
        return (total_value / visits) +
               exploration * std::sqrt(std::log((float)parent_visits) / visits);
    }

    float mean_value() const {
        return visits > 0 ? total_value / visits : 0.0f;
    }
};

class TitanPlanner {
private:
    int   mcts_iterations;
    int   max_depth;
    float exploration_c;
    std::mt19937 rng;

    // Expand a node by generating candidate actions
    std::function<std::vector<Action>(const PlanState&)>  expand_fn;
    // Simulate applying an action to get the next state
    std::function<PlanState(const PlanState&, const Action&)> transition_fn;
    // Evaluate how close a state is to the goal
    std::function<float(const PlanState&, const std::string&)> value_fn;

public:
    TitanPlanner(
        std::function<std::vector<Action>(const PlanState&)>            expand,
        std::function<PlanState(const PlanState&, const Action&)>       transition,
        std::function<float(const PlanState&, const std::string&)>      value,
        int   mcts_iter   = 500,
        int   max_depth   = 10,
        float exploration  = 1.414f)
        : mcts_iterations(mcts_iter),
          max_depth(max_depth),
          exploration_c(exploration),
          rng(std::random_device{}()),
          expand_fn(expand),
          transition_fn(transition),
          value_fn(value) {
        std::cout << "[Planner] MCTS Goal-Directed Planner initialized."
                  << " Iterations: " << mcts_iter << std::endl;
    }

    // Plan a sequence of actions from start_state toward the goal
    std::vector<Action> plan(const PlanState& start_state,
                             const std::string& goal,
                             float goal_threshold = 0.85f) {
        auto root = std::make_shared<MCTSNode>();
        root->state = start_state;

        for (int iter = 0; iter < mcts_iterations; ++iter) {
            // 1. Selection: traverse tree using UCB1
            auto node = select(root);

            // 2. Expansion: add child nodes if not terminal or max depth
            if (!node->state.is_terminal && depth_of(node, root) < max_depth) {
                node = expand(node, goal);
            }

            // 3. Simulation: rollout to estimate value
            float val = rollout(node->state, goal, max_depth - depth_of(node, root));

            // 4. Backpropagation: update all nodes on the path
            backprop(node, val);
        }

        // Extract best plan from root to deepest best child
        return extract_plan(root, goal, goal_threshold);
    }

    // Decompose a high-level goal into ordered subgoals
    std::vector<std::string> decompose_goal(const std::string& goal,
        std::function<std::vector<std::string>(const std::string&)> decompose_fn) {
        auto subgoals = decompose_fn(goal);
        std::cout << "[Planner] Goal decomposed into " << subgoals.size()
                  << " subgoals." << std::endl;
        return subgoals;
    }

private:
    std::shared_ptr<MCTSNode> select(std::shared_ptr<MCTSNode> node) {
        while (!node->children.empty() && !node->state.is_terminal) {
            node = *std::max_element(
                node->children.begin(), node->children.end(),
                [](auto& a, auto& b) { return a->ucb1() < b->ucb1(); });
        }
        return node;
    }

    std::shared_ptr<MCTSNode> expand(std::shared_ptr<MCTSNode> node,
                                     const std::string& goal) {
        auto actions = expand_fn(node->state);
        for (auto& action : actions) {
            auto child_state = transition_fn(node->state, action);
            auto child       = std::make_shared<MCTSNode>();
            child->state       = child_state;
            child->action_taken = action;
            child->parent       = node;
            node->children.push_back(child);
        }
        if (!node->children.empty()) {
            // Return a random unexplored child
            std::uniform_int_distribution<int> dist(0, node->children.size() - 1);
            return node->children[dist(rng)];
        }
        return node;
    }

    float rollout(const PlanState& state, const std::string& goal, int steps_left) {
        auto cur = state;
        for (int s = 0; s < steps_left && !cur.is_terminal; ++s) {
            float v = value_fn(cur, goal);
            if (v >= 0.85f) return v;
            auto actions = expand_fn(cur);
            if (actions.empty()) break;
            std::uniform_int_distribution<int> dist(0, actions.size() - 1);
            cur = transition_fn(cur, actions[dist(rng)]);
        }
        return value_fn(cur, goal);
    }

    void backprop(std::shared_ptr<MCTSNode> node, float value) {
        while (node) {
            node->visits++;
            node->total_value += value;
            node = node->parent.lock();
        }
    }

    int depth_of(std::shared_ptr<MCTSNode> node,
                 std::shared_ptr<MCTSNode> root) {
        int d = 0;
        auto cur = node->parent.lock();
        while (cur && cur != root) { d++; cur = cur->parent.lock(); }
        return d;
    }

    std::vector<Action> extract_plan(std::shared_ptr<MCTSNode> root,
                                     const std::string& goal,
                                     float threshold) {
        std::vector<Action> plan;
        auto node = root;
        while (!node->children.empty()) {
            auto best = *std::max_element(
                node->children.begin(), node->children.end(),
                [](auto& a, auto& b) { return a->mean_value() < b->mean_value(); });
            if (best->mean_value() < threshold * 0.5f) break;
            plan.push_back(best->action_taken);
            if (best->state.is_terminal) break;
            node = best;
        }
        std::cout << "[Planner] Plan extracted: " << plan.size()
                  << " steps toward goal." << std::endl;
        return plan;
    }
};
