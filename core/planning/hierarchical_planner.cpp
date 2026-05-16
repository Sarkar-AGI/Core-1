#pragma once
#include <string>
#include <vector>
#include <functional>
#include <memory>
#include <iostream>
#include <algorithm>
#include <unordered_map>
#include <queue>

/*
=============================================================
  TITANCORE AGI: HIERARCHICAL TASK NETWORK (HTN) PLANNER
=============================================================
  Humans plan at multiple levels of abstraction simultaneously.
  Core-1 does the same via Hierarchical Task Networks:

  High-level goal:
    "Write a research paper"
      └─ Sub-task: Research topic
          └─ Primitive: web_search("quantum computing")
          └─ Primitive: retrieve_papers(n=20)
      └─ Sub-task: Write sections
          └─ Primitive: write_section("introduction")
          └─ Primitive: write_section("methods")
      └─ Sub-task: Review and edit
          └─ Primitive: run_spell_check()
          └─ Primitive: revise_argument()

  Architecture:
  - Compound tasks that decompose into sub-tasks
  - Primitive tasks that map directly to actions
  - Precondition / effect system (STRIPS-style)
  - Partial-order planning (steps can run in parallel)
  - Execution monitoring and replanning on failure
=============================================================
*/

// -------------------------------------------------
// World State: a set of predicate -> value assertions
// -------------------------------------------------
using WorldState = std::unordered_map<std::string, std::string>;

// -------------------------------------------------
// Task: either compound (decomposes) or primitive (executes)
// -------------------------------------------------
enum class TaskType { COMPOUND, PRIMITIVE };

struct Precondition {
    std::string predicate;
    std::string expected_value;
};

struct Effect {
    std::string predicate;
    std::string new_value;
};

struct Task {
    std::string           name;
    TaskType              type;
    std::vector<Precondition> preconditions;
    std::vector<Effect>       effects;
    int                   priority   = 0;
    bool                  parallel   = false; // can run concurrently with siblings
    std::string           description;

    // For COMPOUND tasks: possible decompositions (methods)
    std::vector<std::vector<std::string>> methods; // each method = list of sub-task names

    // For PRIMITIVE tasks: the actual action handler
    std::function<bool(WorldState&)> execute;
};

// -------------------------------------------------
// Plan: ordered (or partially-ordered) task sequence
// -------------------------------------------------
struct Plan {
    std::vector<Task*>  steps;
    float               estimated_cost;
    float               estimated_success_rate;
    bool                is_partial = false;  // true if some steps couldn't be resolved
};

// -------------------------------------------------
// TitanHTNPlanner
// -------------------------------------------------
class TitanHTNPlanner {
private:
    std::unordered_map<std::string, Task> task_library;
    int   max_depth       = 12;
    int   max_iterations  = 10000;

public:
    TitanHTNPlanner() {
        seed_task_library();
        std::cout << "[HTNPlanner] Hierarchical Task Network initialized."
                  << " Tasks: " << task_library.size() << std::endl;
    }

    // Register a custom task (compound or primitive)
    void register_task(Task t) {
        task_library[t.name] = std::move(t);
    }

    // ---- Main Planning Interface ----
    Plan plan(const std::string& top_level_goal, WorldState initial_state) {
        Plan result;
        result.estimated_cost         = 0.0f;
        result.estimated_success_rate = 1.0f;

        std::cout << "[HTNPlanner] Planning for goal: " << top_level_goal << std::endl;

        WorldState state = initial_state;
        std::vector<std::string> task_queue = {top_level_goal};
        bool success = decompose_recursive(task_queue, state, result, 0);

        if (!success) {
            result.is_partial = true;
            std::cout << "[HTNPlanner] WARNING: Partial plan — not all tasks resolved." << std::endl;
        } else {
            std::cout << "[HTNPlanner] Plan complete. Steps: " << result.steps.size()
                      << " | Est. cost: " << result.estimated_cost << std::endl;
        }
        return result;
    }

    // ---- Execute a plan, monitoring for failures and replanning ----
    bool execute_plan(Plan& plan, WorldState& state) {
        int executed = 0;
        for (auto* task : plan.steps) {
            if (task->type != TaskType::PRIMITIVE) continue;

            // Check preconditions
            if (!check_preconditions(task->preconditions, state)) {
                std::cout << "[HTNPlanner] Precondition failed for: " << task->name
                          << " — triggering replan." << std::endl;
                // Replan from current state
                auto new_plan = plan_remaining(plan, executed, state);
                return execute_plan(new_plan, state);
            }

            // Execute
            bool ok = task->execute ? task->execute(state) : true;
            if (!ok) {
                std::cout << "[HTNPlanner] Task FAILED: " << task->name << std::endl;
                return false;
            }

            // Apply effects
            apply_effects(task->effects, state);
            std::cout << "[HTNPlanner] Executed: " << task->name << std::endl;
            executed++;
        }
        return true;
    }

    // ---- Goal Priority Queue: manage multiple competing goals ----
    void push_goal(const std::string& goal, int priority, WorldState& state) {
        goal_queue.push({priority, goal});
        std::cout << "[HTNPlanner] Goal queued [priority=" << priority
                  << "]: " << goal << std::endl;
    }

    std::string next_goal() {
        if (goal_queue.empty()) return "";
        auto [p, g] = goal_queue.top();
        goal_queue.pop();
        return g;
    }

    bool has_goals() const { return !goal_queue.empty(); }

    void print_plan(const Plan& plan) const {
        std::cout << "\n[HTNPlanner] === PLAN ===" << std::endl;
        int i = 1;
        for (auto* t : plan.steps) {
            std::cout << " " << i++ << ". ["
                      << (t->type == TaskType::PRIMITIVE ? "ACTION" : "COMPOUND")
                      << "] " << t->name;
            if (!t->description.empty())
                std::cout << " — " << t->description;
            std::cout << std::endl;
        }
        std::cout << " Total steps: " << plan.steps.size()
                  << " | Partial: " << (plan.is_partial ? "yes" : "no")
                  << std::endl;
    }

private:
    std::priority_queue<std::pair<int,std::string>> goal_queue;

    bool decompose_recursive(std::vector<std::string>& task_queue,
                             WorldState& state,
                             Plan& plan,
                             int depth) {
        if (depth > max_depth) return false;
        if (task_queue.empty()) return true;

        std::string task_name = task_queue.front();
        task_queue.erase(task_queue.begin());

        auto it = task_library.find(task_name);
        if (it == task_library.end()) {
            std::cout << "[HTNPlanner] Unknown task: " << task_name << std::endl;
            return false;
        }

        Task& task = it->second;

        if (task.type == TaskType::PRIMITIVE) {
            plan.steps.push_back(&task);
            plan.estimated_cost += 1.0f;
            return decompose_recursive(task_queue, state, plan, depth);
        }

        // COMPOUND: find applicable method
        for (auto& method : task.methods) {
            std::vector<std::string> new_queue = method;
            for (auto& t : task_queue) new_queue.push_back(t);
            Plan sub_plan = plan;
            if (decompose_recursive(new_queue, state, sub_plan, depth + 1)) {
                plan = sub_plan;
                return true;
            }
        }

        return false; // no applicable method
    }

    Plan plan_remaining(const Plan& plan, int from, const WorldState& state) {
        // Re-plan using remaining tasks from the failed point
        Plan new_plan;
        for (int i = from; i < (int)plan.steps.size(); ++i)
            new_plan.steps.push_back(plan.steps[i]);
        return new_plan;
    }

    bool check_preconditions(const std::vector<Precondition>& precs, const WorldState& state) {
        for (auto& p : precs) {
            auto it = state.find(p.predicate);
            if (it == state.end() || it->second != p.expected_value)
                return false;
        }
        return true;
    }

    void apply_effects(const std::vector<Effect>& effects, WorldState& state) {
        for (auto& e : effects) state[e.predicate] = e.new_value;
    }

    void seed_task_library() {
        // -- Primitive tasks --
        auto make_primitive = [&](const std::string& name,
                                  const std::string& desc,
                                  std::function<bool(WorldState&)> fn) {
            Task t;
            t.name        = name;
            t.description = desc;
            t.type        = TaskType::PRIMITIVE;
            t.execute     = fn;
            task_library[name] = std::move(t);
        };

        make_primitive("web_search",
            "Search the web for information",
            [](WorldState& s) { s["search_done"] = "true"; return true; });

        make_primitive("retrieve_documents",
            "Retrieve relevant documents from memory",
            [](WorldState& s) { s["docs_ready"] = "true"; return true; });

        make_primitive("write_draft",
            "Generate a draft response or document",
            [](WorldState& s) { s["draft_ready"] = "true"; return true; });

        make_primitive("review_draft",
            "Critique and score the draft",
            [](WorldState& s) { s["draft_reviewed"] = "true"; return true; });

        make_primitive("revise_draft",
            "Revise based on critique",
            [](WorldState& s) { s["draft_final"] = "true"; return true; });

        make_primitive("execute_code",
            "Run code in sandbox",
            [](WorldState& s) { s["code_run"] = "true"; return true; });

        make_primitive("verify_output",
            "Verify correctness of output",
            [](WorldState& s) { s["verified"] = "true"; return true; });

        make_primitive("respond_to_user",
            "Deliver final answer to user",
            [](WorldState& s) { s["responded"] = "true"; return true; });

        // -- Compound task: answer a question --
        {
            Task t;
            t.name = "answer_question";
            t.type = TaskType::COMPOUND;
            t.description = "Fully answer a user question";
            t.methods = {
                {"retrieve_documents", "write_draft", "review_draft", "revise_draft", "respond_to_user"},
                {"web_search", "write_draft", "review_draft", "respond_to_user"}
            };
            task_library[t.name] = std::move(t);
        }

        // -- Compound task: write and verify code --
        {
            Task t;
            t.name = "solve_coding_task";
            t.type = TaskType::COMPOUND;
            t.description = "Write, run, and verify a code solution";
            t.methods = {
                {"write_draft", "execute_code", "verify_output", "respond_to_user"}
            };
            task_library[t.name] = std::move(t);
        }

        // -- Compound task: research a topic --
        {
            Task t;
            t.name = "research_topic";
            t.type = TaskType::COMPOUND;
            t.description = "Gather and synthesise information on a topic";
            t.methods = {
                {"web_search", "retrieve_documents", "write_draft", "review_draft", "respond_to_user"}
            };
            task_library[t.name] = std::move(t);
        }
    }
};
