#pragma once
#include <torch/torch.h>
#include <string>
#include <vector>
#include <functional>
#include <iostream>
#include <unordered_map>

/*
=============================================================
  TITANCORE AGI: ADVANCED REASONING ENGINE
=============================================================
  Four complementary reasoning modes that cover the full
  spectrum of human rational thought:

  1. DEDUCTIVE   — from general rules to specific conclusions
                   (All men are mortal; Socrates is a man;
                    therefore Socrates is mortal)

  2. INDUCTIVE   — from specific observations to general rules
                   (Every swan I've seen is white;
                    therefore all swans are probably white)

  3. ABDUCTIVE   — best explanation for observed evidence
                   (The ground is wet; best explanation: it rained)

  4. ANALOGICAL  — transfer structure from source to target
                   (The atom is like a solar system)
=============================================================
*/

// --- Logic clause ---
struct Clause {
    std::string premise;
    float       confidence;  // 0.0 – 1.0
};

struct Argument {
    std::vector<Clause> premises;
    std::string         conclusion;
    float               strength;   // overall argument strength
    std::string         type;       // "deductive" | "inductive" | "abductive" | "analogical"
};

// --- Analogy Mapping ---
struct AnalogyMapping {
    std::string source_domain;  // e.g. "solar system"
    std::string target_domain;  // e.g. "atom"
    std::vector<std::pair<std::string,std::string>> correspondences; // (source, target)
    float structural_similarity;  // 0.0 – 1.0
    std::string inferred_property; // what we infer about the target
};

class TitanReasoningEngine {
private:
    // Rule base for deductive reasoning
    std::vector<std::pair<std::vector<std::string>, std::string>> rules;
    // Observation history for induction
    std::vector<std::pair<std::string, bool>> observations;

public:
    TitanReasoningEngine() {
        seed_rules();
        std::cout << "[ReasoningEngine] Deductive/Inductive/Abductive/Analogical"
                     " reasoning loaded. Rules: " << rules.size() << std::endl;
    }

    // =============================================
    // DEDUCTIVE REASONING
    // Given premises and a rule base, derive conclusions
    // =============================================
    Argument deduce(const std::vector<Clause>& premises) {
        Argument arg;
        arg.type = "deductive";

        for (auto& [antecedents, consequent] : rules) {
            bool all_satisfied = true;
            float min_conf     = 1.0f;

            for (auto& ant : antecedents) {
                bool found = false;
                for (auto& p : premises) {
                    if (p.premise.find(ant) != std::string::npos) {
                        found    = true;
                        min_conf = std::min(min_conf, p.confidence);
                        break;
                    }
                }
                if (!found) { all_satisfied = false; break; }
            }

            if (all_satisfied) {
                arg.premises    = premises;
                arg.conclusion  = consequent;
                arg.strength    = min_conf;  // conclusion is as strong as weakest premise
                std::cout << "[Deductive] Derived: " << consequent
                          << " (conf=" << min_conf << ")" << std::endl;
                return arg;
            }
        }

        arg.conclusion = "(no deductive conclusion reachable)";
        arg.strength   = 0.0f;
        return arg;
    }

    // =============================================
    // INDUCTIVE REASONING
    // From repeated observations, infer a general rule
    // =============================================
    Argument induce(const std::string& subject, const std::string& property) {
        Argument arg;
        arg.type = "inductive";

        int positive = 0, total = 0;
        for (auto& [obs, val] : observations) {
            if (obs.find(subject) != std::string::npos &&
                obs.find(property) != std::string::npos) {
                total++;
                if (val) positive++;
            }
        }

        if (total < 3) {
            arg.conclusion = "(insufficient observations for induction)";
            arg.strength   = 0.0f;
            return arg;
        }

        float freq     = (float)positive / total;
        arg.conclusion = "All " + subject + " are " + property;
        arg.strength   = freq * (1.0f - 1.0f / std::sqrt((float)total)); // penalise small n
        std::cout << "[Inductive] Generalised: " << arg.conclusion
                  << " (strength=" << arg.strength << ", n=" << total << ")" << std::endl;
        return arg;
    }

    // Record a new observation
    void observe(const std::string& description, bool is_true) {
        observations.push_back({description, is_true});
    }

    // =============================================
    // ABDUCTIVE REASONING
    // Infer the best explanation for observed evidence
    // =============================================
    Argument abduce(const std::string& observation,
                    const std::vector<std::pair<std::string,float>>& candidate_hypotheses) {
        Argument arg;
        arg.type = "abductive";

        // Score each hypothesis: P(obs | hyp) * P(hyp)
        float     best_score = -1.0f;
        std::string best_hyp;
        std::vector<std::string> evidence_words = tokenise(observation);

        for (auto& [hyp, prior] : candidate_hypotheses) {
            float likelihood = 0.0f;
            std::vector<std::string> hyp_words = tokenise(hyp);
            for (auto& hw : hyp_words)
                for (auto& ew : evidence_words)
                    if (hw == ew) likelihood += 0.1f;
            likelihood = std::min(likelihood, 1.0f);

            float score = likelihood * prior;
            if (score > best_score) {
                best_score = score;
                best_hyp   = hyp;
            }
        }

        arg.conclusion = "Best explanation: " + best_hyp;
        arg.strength   = best_score;
        std::cout << "[Abductive] " << arg.conclusion
                  << " (score=" << best_score << ")" << std::endl;
        return arg;
    }

    // =============================================
    // ANALOGICAL REASONING
    // Transfer structure from source → target domain
    // =============================================
    AnalogyMapping analogise(const std::string& source_domain,
                             const std::string& target_domain,
                             const std::vector<std::pair<std::string,std::string>>& correspondences,
                             const std::string& source_property,
                             torch::Tensor src_emb, torch::Tensor tgt_emb) {
        AnalogyMapping mapping;
        mapping.source_domain = source_domain;
        mapping.target_domain = target_domain;
        mapping.correspondences = correspondences;

        // Structural similarity via embedding cosine distance
        float dot = torch::dot(src_emb.view(-1), tgt_emb.view(-1)).item<float>();
        float na  = src_emb.norm().item<float>();
        float nb  = tgt_emb.norm().item<float>();
        mapping.structural_similarity = (na * nb > 1e-8f) ? dot / (na * nb) : 0.0f;

        // Infer target property by analogy
        if (mapping.structural_similarity > 0.3f) {
            mapping.inferred_property = "By analogy with " + source_domain
                + ", " + target_domain + " likely also exhibits: " + source_property;
        } else {
            mapping.inferred_property = "(analogy too weak to infer)";
        }

        std::cout << "[Analogical] " << source_domain << " -> " << target_domain
                  << " sim=" << mapping.structural_similarity << std::endl;
        std::cout << "  Inference: " << mapping.inferred_property << std::endl;
        return mapping;
    }

    // =============================================
    // COMPOSITE: choose the best reasoning mode
    // =============================================
    Argument reason(const std::string& question,
                    const std::vector<Clause>& context,
                    const std::vector<std::pair<std::string,float>>& hypotheses = {}) {
        // Prefer deductive if rules apply
        auto ded = deduce(context);
        if (ded.strength > 0.7f) return ded;

        // Fall back to abductive if hypotheses provided
        if (!hypotheses.empty()) {
            std::string obs;
            for (auto& c : context) obs += c.premise + " ";
            return abduce(obs, hypotheses);
        }

        // Last resort: inductive (needs prior observations)
        if (!observations.empty() && !context.empty())
            return induce(context[0].premise, question);

        Argument fallback;
        fallback.type       = "unknown";
        fallback.conclusion = "(no reasoning chain reachable — insufficient knowledge)";
        fallback.strength   = 0.0f;
        return fallback;
    }

    void print_stats() const {
        std::cout << "[ReasoningEngine] Rules: " << rules.size()
                  << " | Observations: " << observations.size() << std::endl;
    }

private:
    std::vector<std::string> tokenise(const std::string& s) {
        std::vector<std::string> tokens;
        std::string tok;
        for (char c : s) {
            if (c == ' ' || c == ',' || c == '.') {
                if (!tok.empty()) { tokens.push_back(tok); tok.clear(); }
            } else tok += std::tolower(c);
        }
        if (!tok.empty()) tokens.push_back(tok);
        return tokens;
    }

    void seed_rules() {
        // Format: {premises}, conclusion
        rules = {
            {{"is a mammal", "mammals are warm-blooded"}, "is warm-blooded"},
            {{"is a bird", "birds have wings"},           "has wings"},
            {{"is on fire", "fire requires oxygen"},      "oxygen is being consumed"},
            {{"is raining", "rain causes wet ground"},    "ground is wet"},
            {{"is frozen", "frozen things are cold"},     "is cold"},
            {{"is a plant", "plants need sunlight"},      "needs sunlight"},
            {{"has a fever", "fever indicates infection"},"may have an infection"},
            {{"is heavier than air", "gravity acts on mass"}, "will fall if unsupported"},
            {{"is conductive", "conducts electricity"},   "allows electric current to flow"},
            {{"is acidic", "acids corrode metal"},        "may corrode metal surfaces"},
        };
    }
};
