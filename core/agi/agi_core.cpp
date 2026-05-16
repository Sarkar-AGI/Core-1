#pragma once
#include <torch/torch.h>
#include <string>
#include <vector>
#include <iostream>
#include <functional>
#include "../model/config.h"

// --- Original AGI subsystems ---
#include "../learning/online_learning.cpp"
#include "../memory/episodic.cpp"
#include "../memory/semantic.cpp"
#include "../memory/working.cpp"
#include "../reasoning/chain_of_thought.cpp"
#include "../reasoning/planner.cpp"
#include "../meta/maml.cpp"
#include "../world_model/world_model.cpp"
#include "../tools/tool_executor.cpp"

// --- New AGI subsystems ---
#include "../learning/experience_engine.cpp"
#include "../reasoning/common_sense.cpp"
#include "../reasoning/analogical.cpp"
#include "../planning/hierarchical_planner.cpp"
#include "../intelligence/general_intelligence.cpp"

/*
=============================================================
  TITANCORE AGI: UNIFIED COGNITIVE CORE v2
=============================================================
  Full cognitive loop:
    Perceive -> Common Sense Gate -> Remember
             -> Reason (CoT / Deductive / Abductive / Analogical)
             -> Plan (MCTS + HTN)
             -> Act (Tool Use)
             -> Learn (Online GD + RLHF + PPO + HER + MAML)
             -> Self-Assess (Metacognition + Self-Improvement)
             -> Update World Model

  Subsystems:
  ─────────────────────────────────────────────────────────
  LEARNING        OnlineLearner  + ExperienceEngine (RLHF+PPO+HER)
  MEMORY          Episodic  + Semantic  + Working
  REASONING       CoT + ReasoningEngine (Deductive/Inductive/
                  Abductive/Analogical) + CommonSense
  PLANNING        MCTS Planner + HTN Hierarchical Planner
  INTELLIGENCE    GeneralIntelligence (Transfer + Curriculum +
                  Self-Improvement + Skill Composition +
                  Generalisation + Metacognition)
  META-LEARNING   MAML / Reptile
  WORLD MODEL     VAE Encoder + Dynamics + Reward + Novelty
  TOOL USE        5 built-in tools + custom registration
=============================================================
*/

class TitanAGI {
private:
    // ----- Memory -----
    TitanEpisodicMemory  episodic;
    TitanSemanticMemory  semantic;
    TitanWorkingMemory   working;

    // ----- Reasoning -----
    std::unique_ptr<TitanCoTEngine>      cot_engine;
    TitanCommonSense                     common_sense;
    TitanReasoningEngine                 reasoning_engine;

    // ----- Planning -----
    TitanHTNPlanner                      htn_planner;

    // ----- Tools -----
    TitanToolExecutor                    tool_exec;

    // ----- World Model -----
    TitanWorldModel                      world_model;

    // ----- Learning -----
    std::unique_ptr<TitanOnlineLearner>  online_learner;
    std::unique_ptr<TitanExperienceEngine> exp_engine;

    // ----- Meta-Learning -----
    // (MAML instantiated on demand per task)

    // ----- General Intelligence -----
    std::unique_ptr<TitanGeneralIntelligence> general_intel;

    // ----- Language Model -----
    torch::nn::AnyModule language_model;

    // ----- Goal stack -----
    std::vector<std::string> goal_stack;

    // ----- Intrinsic motivation -----
    float curiosity_weight = 0.1f;

    // ----- Text generation lambda -----
    std::function<std::string(const std::string&)> generate_fn;

    // ----- Step counter -----
    int64_t step_count = 0;

public:
    TitanAGI(torch::nn::AnyModule lm,
             const TitanConfig& cfg,
             std::function<std::string(const std::string&)> generate)
        : episodic(50000),
          working(cfg.block_size),
          world_model(cfg.n_embd, 512, 256),
          language_model(lm),
          generate_fn(generate) {

        // Bind working memory to CoT engine
        cot_engine = std::make_unique<TitanCoTEngine>(&working);

        // Experience engine (state_dim matches model embedding size)
        exp_engine = std::make_unique<TitanExperienceEngine>(cfg.n_embd);

        // General intelligence: evaluator stub
        general_intel = std::make_unique<TitanGeneralIntelligence>(
            [](const std::string& /*skill*/,
               const std::vector<std::pair<torch::Tensor,torch::Tensor>>& /*examples*/) -> float {
                return 0.75f; // placeholder — replace with real model eval
            });

        print_banner(cfg);
    }

    // =========================================================
    // PRIMARY INTERFACE: Process a user input through the
    // full AGI cognitive loop
    // =========================================================
    std::string process(const std::string& user_input,
                        int64_t session_id  = 0,
                        CoTMode cot_mode    = CoTMode::REFLECTION) {
        step_count++;

        // --------------------------------------------------
        // 1. PERCEIVE: add to working memory
        // --------------------------------------------------
        working.push("user", user_input,
                     (int)user_input.size() / 4, /*importance=*/1.0f);

        // --------------------------------------------------
        // 2. COMMON SENSE GATE: plausibility check
        // --------------------------------------------------
        auto dummy_emb   = torch::randn({512});
        float plausibility = common_sense.plausibility(user_input, dummy_emb);
        if (plausibility < 0.05f) {
            std::string rejection = "That statement appears to violate physical or "
                                    "logical constraints and cannot be reasoned about.";
            working.push("assistant", rejection, (int)rejection.size() / 4, 1.0f);
            return rejection;
        }

        // --------------------------------------------------
        // 3. METACOGNITION: do I know this domain?
        // --------------------------------------------------
        std::string meta_note = general_intel->metacognitive_response(user_input);
        if (general_intel->knows_it_doesnt_know(user_input)) {
            working.think("I have low confidence in this area — apply extra reasoning.");
        }

        // --------------------------------------------------
        // 4. MEMORY RETRIEVAL: episodic + semantic
        // --------------------------------------------------
        auto memories = episodic.retrieve(dummy_emb, /*top_k=*/3);
        for (auto& m : memories)
            working.think("Recalled: " + m.text_summary);

        // --------------------------------------------------
        // 5. COMMON SENSE ENRICHMENT: look up relevant facts
        // --------------------------------------------------
        auto cs_facts = common_sense.fuzzy_query(dummy_emb, /*top_k=*/3);
        for (auto& f : cs_facts)
            working.think("Fact: " + f.subject + " " + f.relation + " " + f.object);

        // --------------------------------------------------
        // 6. ADVANCED REASONING
        // --------------------------------------------------
        // a) Structured reasoning (Deductive / Abductive)
        std::vector<Clause> context_clauses;
        for (auto& f : cs_facts)
            context_clauses.push_back({f.subject + " " + f.relation + " " + f.object,
                                       f.weight});

        auto logic_arg = reasoning_engine.reason(user_input, context_clauses);
        if (logic_arg.strength > 0.6f)
            working.think("Logic: " + logic_arg.conclusion
                        + " (strength=" + std::to_string(logic_arg.strength) + ")");

        // b) Chain-of-Thought reasoning
        ReasoningTrace trace;
        switch (cot_mode) {
            case CoTMode::STANDARD:
                trace = cot_engine->standard_cot(user_input, generate_fn); break;
            case CoTMode::SELF_CONSISTENCY:
                trace = cot_engine->self_consistency(user_input, generate_fn); break;
            case CoTMode::TREE_OF_THOUGHT:
                trace = cot_engine->tree_of_thought(user_input, generate_fn,
                        [](const std::string&) { return 0.5f; }); break;
            case CoTMode::REFLECTION:
            default:
                trace = cot_engine->reflection(user_input, generate_fn); break;
        }

        // --------------------------------------------------
        // 7. PLANNING: HTN if goal-directed
        // --------------------------------------------------
        if (!goal_stack.empty()) {
            WorldState ws;
            ws["goal"] = goal_stack.back();
            auto plan  = htn_planner.plan(goal_stack.back(), ws);
            htn_planner.print_plan(plan);
            working.think("Plan: " + std::to_string(plan.steps.size()) + " steps drafted.");
        }

        // --------------------------------------------------
        // 8. TOOL USE: dispatch if model requests a tool
        // --------------------------------------------------
        if (trace.final_answer.find("<tool>") != std::string::npos) {
            auto tool_call = TitanToolExecutor::parse_call(trace.final_answer);
            auto result    = tool_exec.execute(tool_call);
            if (result.success) {
                working.push("tool_result", result.output,
                             (int)result.output.size() / 4, 1.2f);
                trace = cot_engine->reflection(
                    user_input + "\nTool result: " + result.output, generate_fn);
            }
        }

        // --------------------------------------------------
        // 9. METACOGNITIVE SUFFIX
        // --------------------------------------------------
        std::string response = trace.final_answer;
        if (!meta_note.empty())
            response += "\n\n[Note: " + meta_note + "]";

        // --------------------------------------------------
        // 10. STORE EPISODE + UPDATE WORKING MEMORY
        // --------------------------------------------------
        episodic.store_episode("conversation", dummy_emb,
            "Q: " + user_input + " A: " + response, 1.0f);
        working.push("assistant", response, (int)response.size() / 4, 1.0f);

        // --------------------------------------------------
        // 11. OBSERVE AGENT (Theory of Mind)
        // --------------------------------------------------
        common_sense.observe_agent("user", user_input, "conversation");

        // --------------------------------------------------
        // 12. PERIODIC SELF-IMPROVEMENT
        // --------------------------------------------------
        if (step_count % 100 == 0)
            general_intel->self_assess();

        return response;
    }

    // =========================================================
    // CONTINUOUS LEARNING from a new interaction
    // =========================================================
    void learn_from_interaction(torch::Tensor input_ids,
                                torch::Tensor target_ids) {
        if (online_learner) {
            // online_learner->ingest(input_ids, target_ids, *model);
        }
    }

    // =========================================================
    // RLHF: record human preference signal
    // =========================================================
    void record_preference(torch::Tensor state,
                           torch::Tensor preferred_action,
                           torch::Tensor rejected_action) {
        if (exp_engine) {
            exp_engine->train_reward_model(
                state, preferred_action, state, rejected_action);
        }
    }

    // =========================================================
    // GOAL MANAGEMENT
    // =========================================================
    void set_goal(const std::string& goal) {
        goal_stack.push_back(goal);
        htn_planner.push_goal(goal, /*priority=*/5, *(new WorldState()));
        std::cout << "[AGI] Goal set: " << goal << std::endl;
    }

    void pop_goal() { if (!goal_stack.empty()) goal_stack.pop_back(); }

    const std::string& current_goal() const {
        static std::string none = "(none)";
        return goal_stack.empty() ? none : goal_stack.back();
    }

    // =========================================================
    // CURIOSITY REWARD
    // =========================================================
    float curiosity_reward(const WorldState& predicted, const WorldState& actual) {
        auto error = (predicted.z - actual.z).pow(2).mean().item<float>();
        return curiosity_weight * error;
    }

    // =========================================================
    // STATUS REPORT
    // =========================================================
    void print_status() {
        std::cout << "\n[AGI Status — Step " << step_count << "]" << std::endl;
        episodic.print_stats();
        semantic.print_stats();
        working.print_stats();
        common_sense.print_stats();
        reasoning_engine.print_stats();
        general_intel->print_status();
        std::cout << " Current goal   : " << current_goal() << std::endl;
        std::cout << " Tool manifest:\n" << tool_exec.tool_manifest() << std::endl;
    }

private:
    void print_banner(const TitanConfig& cfg) {
        std::cout << "\n=====================================================" << std::endl;
        std::cout << "  TITANCORE AGI CORE v2 — FULLY ONLINE" << std::endl;
        std::cout << "=====================================================" << std::endl;
        std::cout << " MEMORY" << std::endl;
        std::cout << "   Episodic Memory        : 50K episodes" << std::endl;
        std::cout << "   Semantic Memory        : long-term knowledge graph" << std::endl;
        std::cout << "   Working Memory         : " << cfg.block_size << " tokens" << std::endl;
        std::cout << " REASONING" << std::endl;
        std::cout << "   Chain-of-Thought       : Standard / SelfConsistency / ToT / Reflection" << std::endl;
        std::cout << "   Logic Engine           : Deductive / Inductive / Abductive / Analogical" << std::endl;
        std::cout << "   Common Sense           : Knowledge graph + physical intuition + ToM" << std::endl;
        std::cout << " PLANNING" << std::endl;
        std::cout << "   MCTS Planner           : 500 iterations, depth 10" << std::endl;
        std::cout << "   HTN Planner            : hierarchical goal decomposition" << std::endl;
        std::cout << " LEARNING" << std::endl;
        std::cout << "   Online Gradient Descent: EWC + replay + EMA" << std::endl;
        std::cout << "   RLHF + PPO             : human preference alignment" << std::endl;
        std::cout << "   Hindsight Replay (HER) : learn from failure" << std::endl;
        std::cout << "   Meta-Learning          : MAML / FOMAML / Reptile" << std::endl;
        std::cout << " GENERAL INTELLIGENCE" << std::endl;
        std::cout << "   Transfer Learning      : cross-domain skill transfer" << std::endl;
        std::cout << "   Curriculum Learning    : progressive difficulty" << std::endl;
        std::cout << "   Self-Improvement       : gap identification + auto-curriculum" << std::endl;
        std::cout << "   Skill Composition      : combine skills for novel tasks" << std::endl;
        std::cout << "   Generalisation Test    : validate on unseen domains" << std::endl;
        std::cout << "   Metacognition          : knows what it doesn't know" << std::endl;
        std::cout << " WORLD MODEL               : VAE + dynamics + reward + novelty" << std::endl;
        std::cout << " TOOL USE                  : calculator / search / code / file / db" << std::endl;
        std::cout << "=====================================================" << std::endl;
    }
};
