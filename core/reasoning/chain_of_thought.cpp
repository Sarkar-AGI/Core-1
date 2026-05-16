#pragma once
#include <torch/torch.h>
#include <string>
#include <vector>
#include <iostream>
#include "../memory/working.cpp"

/*
=====================================================
  TITANCORE AGI: CHAIN-OF-THOUGHT REASONING ENGINE
=====================================================
  Implements structured multi-step reasoning before
  producing a final answer — a key AGI capability.

  Modes:
  - Standard CoT   : "Let me think step by step…"
  - Self-Consistency: Sample N reasoning paths, vote
  - Tree-of-Thought : Branch + prune reasoning tree
  - Reflection      : Critique & revise own output
=====================================================
*/

enum class CoTMode {
    STANDARD,
    SELF_CONSISTENCY,
    TREE_OF_THOUGHT,
    REFLECTION
};

struct ReasoningStep {
    int         step_id;
    std::string thought;
    float       confidence; // model's self-assessed confidence
    bool        is_final;
};

struct ReasoningTrace {
    std::vector<ReasoningStep> steps;
    std::string final_answer;
    float       overall_confidence;
    CoTMode     mode;
};

// ToT node for branching search
struct ThoughtNode {
    std::string thought;
    float       value;      // estimated value of this reasoning path
    int         depth;
    std::vector<std::shared_ptr<ThoughtNode>> children;
};

class TitanCoTEngine {
private:
    int   max_steps;
    int   consistency_samples;
    int   tot_breadth;   // branching factor
    int   tot_depth;     // max tree depth
    float min_confidence;

    TitanWorkingMemory* working_mem;

public:
    TitanCoTEngine(TitanWorkingMemory* wm,
                   int   max_steps           = 8,
                   int   consistency_samples  = 5,
                   int   tot_breadth          = 3,
                   int   tot_depth            = 4,
                   float min_confidence       = 0.4f)
        : max_steps(max_steps),
          consistency_samples(consistency_samples),
          tot_breadth(tot_breadth),
          tot_depth(tot_depth),
          min_confidence(min_confidence),
          working_mem(wm) {
        std::cout << "[CoT] Reasoning Engine initialized. Max steps: "
                  << max_steps << std::endl;
    }

    // Standard Chain-of-Thought: linear step-by-step reasoning
    ReasoningTrace standard_cot(const std::string& question,
                                std::function<std::string(const std::string&)> generate) {
        ReasoningTrace trace;
        trace.mode = CoTMode::STANDARD;

        std::string prompt = "Question: " + question +
                             "\nLet me think step by step.\n";

        working_mem->think("Starting CoT for: " + question);

        for (int s = 0; s < max_steps; ++s) {
            std::string thought = generate(prompt);
            float conf = estimate_confidence(thought);

            ReasoningStep step{s, thought, conf, false};
            trace.steps.push_back(step);
            working_mem->think(thought);

            prompt += "Step " + std::to_string(s + 1) + ": " + thought + "\n";

            // Stop if model signals conclusion
            if (thought.find("Therefore") != std::string::npos ||
                thought.find("In conclusion") != std::string::npos ||
                conf > 0.9f) {
                break;
            }
        }

        // Final answer generation
        std::string answer_prompt = prompt + "Final Answer:";
        trace.final_answer        = generate(answer_prompt);
        trace.steps.back().is_final = true;
        trace.overall_confidence  = compute_trace_confidence(trace.steps);

        return trace;
    }

    // Self-Consistency: sample N reasoning paths, take majority vote
    ReasoningTrace self_consistency(const std::string& question,
                                    std::function<std::string(const std::string&)> generate) {
        ReasoningTrace best;
        best.mode = CoTMode::SELF_CONSISTENCY;

        std::unordered_map<std::string, int> answer_votes;
        std::vector<ReasoningTrace> traces;

        for (int i = 0; i < consistency_samples; ++i) {
            auto trace = standard_cot(question, generate);
            traces.push_back(trace);
            answer_votes[trace.final_answer]++;
        }

        // Select most common answer
        std::string majority_answer;
        int max_votes = 0;
        for (auto& [ans, count] : answer_votes) {
            if (count > max_votes) {
                max_votes       = count;
                majority_answer = ans;
            }
        }

        // Return the trace that produced the majority answer
        for (auto& t : traces) {
            if (t.final_answer == majority_answer) {
                best            = t;
                best.overall_confidence = (float)max_votes / consistency_samples;
                return best;
            }
        }

        return traces[0];
    }

    // Tree-of-Thought: BFS branching with value-guided pruning
    ReasoningTrace tree_of_thought(const std::string& question,
                                   std::function<std::string(const std::string&)> generate,
                                   std::function<float(const std::string&)> evaluate) {
        ReasoningTrace trace;
        trace.mode = CoTMode::TREE_OF_THOUGHT;

        auto root = std::make_shared<ThoughtNode>();
        root->thought = "Problem: " + question;
        root->value   = 1.0f;
        root->depth   = 0;

        // BFS across the thought tree
        std::vector<std::shared_ptr<ThoughtNode>> frontier = {root};
        std::shared_ptr<ThoughtNode> best_leaf = root;
        float best_val = 0.0f;

        for (int d = 0; d < tot_depth && !frontier.empty(); ++d) {
            std::vector<std::shared_ptr<ThoughtNode>> next_frontier;

            for (auto& node : frontier) {
                // Generate tot_breadth candidate next thoughts
                for (int b = 0; b < tot_breadth; ++b) {
                    std::string prompt = node->thought + "\nNext reasoning step:";
                    std::string thought = generate(prompt);
                    float val = evaluate(node->thought + " " + thought);

                    auto child = std::make_shared<ThoughtNode>();
                    child->thought = node->thought + "\n" + thought;
                    child->value   = val;
                    child->depth   = d + 1;
                    node->children.push_back(child);

                    if (val > best_val) {
                        best_val  = val;
                        best_leaf = child;
                    }

                    if (val >= min_confidence)
                        next_frontier.push_back(child);
                }
            }

            // Keep only top-breadth nodes
            std::sort(next_frontier.begin(), next_frontier.end(),
                      [](auto& a, auto& b) { return a->value > b->value; });
            if ((int)next_frontier.size() > tot_breadth)
                next_frontier.resize(tot_breadth);

            frontier = std::move(next_frontier);
        }

        trace.final_answer        = generate(best_leaf->thought + "\nFinal Answer:");
        trace.overall_confidence  = best_val;
        return trace;
    }

    // Reflection: generate answer, critique it, revise
    ReasoningTrace reflection(const std::string& question,
                              std::function<std::string(const std::string&)> generate) {
        ReasoningTrace trace;
        trace.mode = CoTMode::REFLECTION;

        // Step 1: initial answer
        std::string draft = generate("Question: " + question + "\nAnswer:");
        trace.steps.push_back({0, "Draft: " + draft, 0.5f, false});
        working_mem->think("Draft answer: " + draft);

        // Step 2: critique
        std::string critique_prompt = "Question: " + question +
            "\nDraft Answer: " + draft +
            "\nCritique the answer. What is wrong or incomplete?";
        std::string critique = generate(critique_prompt);
        trace.steps.push_back({1, "Critique: " + critique, 0.5f, false});
        working_mem->think("Critique: " + critique);

        // Step 3: revision
        std::string revise_prompt = "Question: " + question +
            "\nDraft: " + draft +
            "\nCritique: " + critique +
            "\nRevised final answer:";
        std::string revised = generate(revise_prompt);
        trace.steps.push_back({2, "Revised: " + revised, 0.9f, true});

        trace.final_answer       = revised;
        trace.overall_confidence = 0.85f;
        return trace;
    }

private:
    float estimate_confidence(const std::string& text) {
        // Simple heuristic: certainty words boost confidence
        float score = 0.5f;
        if (text.find("definitely") != std::string::npos) score += 0.2f;
        if (text.find("therefore")  != std::string::npos) score += 0.15f;
        if (text.find("uncertain")  != std::string::npos) score -= 0.2f;
        if (text.find("might")      != std::string::npos) score -= 0.1f;
        return std::max(0.0f, std::min(1.0f, score));
    }

    float compute_trace_confidence(const std::vector<ReasoningStep>& steps) {
        if (steps.empty()) return 0.0f;
        float sum = 0.0f;
        for (auto& s : steps) sum += s.confidence;
        return sum / steps.size();
    }
};
