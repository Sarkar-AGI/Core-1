#pragma once
#include <torch/torch.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <functional>
#include <iostream>
#include <algorithm>
#include <cmath>

/*
=============================================================
  TITANCORE AGI: COMMON SENSE REASONING ENGINE
=============================================================
  Humans understand the world without being told everything.
  This module gives Core-1 the same capability through:

  - Common Sense Knowledge Graph (ConceptNet-style triples)
  - Physical Intuition (object permanence, causality, time)
  - Social Reasoning (intent, beliefs, emotions of others)
  - Plausibility Scoring (is this claim reasonable?)
  - Analogy-Based Inference (A:B :: C:?)
  - Default Reasoning (typical/normal case assumptions)
=============================================================
*/

// -----------------------------------------------
// Knowledge Triple: (subject, relation, object)
// -----------------------------------------------
struct CSTriple {
    std::string subject;
    std::string relation;   // IsA, UsedFor, CapableOf, Causes, HasProperty, etc.
    std::string object;
    float       weight;     // confidence (0.0 – 1.0)
    torch::Tensor subject_emb; // [D]
    torch::Tensor object_emb;  // [D]
};

// -----------------------------------------------
// Physical World Schema (object affordances)
// -----------------------------------------------
struct PhysicalSchema {
    std::string object_name;
    bool        has_mass       = true;
    bool        can_fall       = true;
    bool        is_permanent   = true;  // object permanence
    bool        is_liquid      = false;
    float       typical_weight_kg = 0.0f;
    float       typical_size_m    = 0.0f;
    std::vector<std::string> typical_uses;
};

// -----------------------------------------------
// Theory of Mind: model another agent's mental state
// -----------------------------------------------
struct MentalState {
    std::string agent;
    std::unordered_map<std::string, float> beliefs;    // topic -> confidence
    std::unordered_map<std::string, float> desires;    // goal -> salience
    std::string perceived_emotion;                      // happy, angry, confused …
    float       cooperation_intent = 0.5f;              // 0 = hostile, 1 = cooperative
};

// -----------------------------------------------
// TitanCommonSense
// -----------------------------------------------
class TitanCommonSense {
private:
    std::vector<CSTriple>    knowledge_graph;
    std::vector<PhysicalSchema> physical_schemas;
    std::unordered_map<std::string, MentalState> agent_models;

    // Plausibility neural scorer
    torch::nn::Linear plaus_fc1{nullptr}, plaus_fc2{nullptr}, plaus_out{nullptr};
    bool net_initialized = false;

    float cosine_sim(const torch::Tensor& a, const torch::Tensor& b) {
        auto na = a.norm().item<float>();
        auto nb = b.norm().item<float>();
        if (na < 1e-8f || nb < 1e-8f) return 0.0f;
        return torch::dot(a.view(-1), b.view(-1)).item<float>() / (na * nb);
    }

public:
    TitanCommonSense() {
        // Pre-load a minimal built-in common sense knowledge base
        seed_knowledge_base();
        seed_physical_schemas();
        std::cout << "[CommonSense] Engine initialized." << std::endl;
        std::cout << " > Knowledge triples : " << knowledge_graph.size() << std::endl;
        std::cout << " > Physical schemas  : " << physical_schemas.size() << std::endl;
    }

    // ---- Query: find all triples matching subject + relation ----
    std::vector<CSTriple> query(const std::string& subject,
                                const std::string& relation = "",
                                float min_weight = 0.3f) {
        std::vector<CSTriple> results;
        for (auto& t : knowledge_graph) {
            if (t.subject == subject &&
                (relation.empty() || t.relation == relation) &&
                t.weight >= min_weight)
                results.push_back(t);
        }
        return results;
    }

    // ---- Embedding-based fuzzy lookup ----
    std::vector<CSTriple> fuzzy_query(torch::Tensor query_emb,
                                      int top_k = 5,
                                      float min_weight = 0.3f) {
        std::vector<std::pair<float, int>> scores;
        for (int i = 0; i < (int)knowledge_graph.size(); ++i) {
            if (!knowledge_graph[i].subject_emb.defined()) continue;
            if (knowledge_graph[i].weight < min_weight) continue;
            float s = cosine_sim(query_emb, knowledge_graph[i].subject_emb)
                    * knowledge_graph[i].weight;
            scores.push_back({s, i});
        }
        std::sort(scores.begin(), scores.end(),
                  [](auto& a, auto& b) { return a.first > b.first; });
        std::vector<CSTriple> out;
        for (int i = 0; i < std::min(top_k, (int)scores.size()); ++i)
            out.push_back(knowledge_graph[scores[i].second]);
        return out;
    }

    // ---- Add new knowledge from experience ----
    void learn_triple(const std::string& subj, const std::string& rel,
                      const std::string& obj, float weight = 0.8f) {
        // Check for contradiction
        for (auto& t : knowledge_graph) {
            if (t.subject == subj && t.relation == rel && t.object != obj) {
                // Conflicting knowledge: average weights
                t.weight = 0.5f * (t.weight + weight);
                std::cout << "[CommonSense] Conflicting triple resolved: "
                          << subj << " " << rel << " " << t.object
                          << " vs " << obj << std::endl;
            }
        }
        knowledge_graph.push_back({subj, rel, obj, weight});
    }

    // ---- Plausibility: is this statement reasonable? ----
    float plausibility(const std::string& statement,
                       torch::Tensor statement_emb) {
        // Simple heuristic: check if any supporting triple exists
        float score = 0.5f; // default: uncertain

        for (auto& t : knowledge_graph) {
            if (statement.find(t.subject) != std::string::npos &&
                statement.find(t.object) != std::string::npos) {
                score = std::max(score, t.weight);
            }
        }

        // Check for impossibility markers
        auto impossible_phrases = {"violates physics", "infinite energy",
                                   "faster than light", "perpetual motion"};
        for (auto& p : impossible_phrases)
            if (statement.find(p) != std::string::npos) score = 0.02f;

        return score;
    }

    // ---- Analogy: A is to B as C is to ? ----
    std::string analogy(const std::string& A, const std::string& B,
                        const std::string& C,
                        torch::Tensor emb_A, torch::Tensor emb_B, torch::Tensor emb_C) {
        // Compute the relation vector in embedding space
        auto relation_vec = emb_B - emb_A;  // [D]
        auto target_emb   = emb_C + relation_vec;

        // Find the closest concept in the knowledge graph
        float best_score = -1.0f;
        std::string best_answer = "(unknown)";

        for (auto& t : knowledge_graph) {
            if (!t.object_emb.defined()) continue;
            float s = cosine_sim(target_emb, t.object_emb);
            if (s > best_score) {
                best_score  = s;
                best_answer = t.object;
            }
        }

        std::cout << "[CommonSense] Analogy: " << A << ":" << B
                  << " :: " << C << ":" << best_answer
                  << " (score=" << best_score << ")" << std::endl;
        return best_answer;
    }

    // ---- Physical Intuition: what can this object do? ----
    PhysicalSchema* get_physical_schema(const std::string& object_name) {
        for (auto& s : physical_schemas)
            if (s.object_name == object_name) return &s;
        return nullptr;
    }

    bool will_fall(const std::string& object_name) {
        auto* s = get_physical_schema(object_name);
        return s ? (s->has_mass && s->can_fall) : true;
    }

    // ---- Theory of Mind: model what another agent believes ----
    void observe_agent(const std::string& agent_name,
                       const std::string& action,
                       const std::string& context) {
        auto& state = agent_models[agent_name];
        state.agent = agent_name;

        // Infer belief update from observed action
        if (action.find("ask") != std::string::npos)
            state.beliefs[context] = std::max(0.0f, state.beliefs[context] - 0.2f);
        if (action.find("assert") != std::string::npos || action.find("claim") != std::string::npos)
            state.beliefs[context] = std::min(1.0f, state.beliefs[context] + 0.3f);
        if (action.find("help") != std::string::npos)
            state.cooperation_intent = std::min(1.0f, state.cooperation_intent + 0.1f);
        if (action.find("refuse") != std::string::npos || action.find("block") != std::string::npos)
            state.cooperation_intent = std::max(0.0f, state.cooperation_intent - 0.15f);
    }

    MentalState get_agent_model(const std::string& agent_name) {
        auto it = agent_models.find(agent_name);
        if (it != agent_models.end()) return it->second;
        return {agent_name, {}, {}, "neutral", 0.5f};
    }

    // ---- Default Assumption: fill in missing context ----
    std::string default_assumption(const std::string& context) {
        // Built-in default rules
        std::unordered_map<std::string, std::string> defaults = {
            {"weather",      "typical weather for the region at this time of year"},
            {"intent",       "the agent intends to be helpful"},
            {"time",         "the present day and time"},
            {"location",     "the most recently mentioned location"},
            {"person_age",   "an adult (18+)"},
            {"object_owner", "the person who last mentioned it"},
        };
        auto it = defaults.find(context);
        return it != defaults.end() ? it->second : "unknown (no default available)";
    }

    void print_stats() const {
        std::cout << "[CommonSense] Triples: " << knowledge_graph.size()
                  << " | Schemas: " << physical_schemas.size()
                  << " | Agent models: " << agent_models.size() << std::endl;
    }

private:
    void seed_knowledge_base() {
        // Seed with foundational common sense triples (ConceptNet-style)
        std::vector<std::tuple<std::string,std::string,std::string,float>> seed = {
            {"knife",        "UsedFor",    "cutting food",          0.95f},
            {"fire",         "Causes",     "heat",                  0.99f},
            {"fire",         "Causes",     "burns",                 0.97f},
            {"rain",         "Causes",     "wet ground",            0.98f},
            {"water",        "HasProperty","liquid at room temp",   0.99f},
            {"ice",          "IsA",        "frozen water",          0.99f},
            {"bird",         "CapableOf",  "flying",                0.85f},
            {"penguin",      "IsA",        "bird",                  0.99f},
            {"penguin",      "CapableOf",  "swimming",              0.97f},
            {"penguin",      "CannotDo",   "flying",                0.95f},
            {"car",          "UsedFor",    "transportation",        0.99f},
            {"car",          "Requires",   "fuel or electricity",   0.97f},
            {"human",        "Requires",   "food and water",        0.99f},
            {"human",        "CapableOf",  "language",              0.99f},
            {"human",        "HasProperty","social",                0.90f},
            {"glass",        "HasProperty","fragile",               0.95f},
            {"ice",          "Causes",     "slipping hazard",       0.85f},
            {"sun",          "Causes",     "daytime",               0.99f},
            {"gravity",      "Causes",     "objects to fall",       0.99f},
            {"book",         "UsedFor",    "reading",               0.97f},
            {"chair",        "UsedFor",    "sitting",               0.98f},
            {"door",         "UsedFor",    "entering/exiting room", 0.98f},
            {"money",        "UsedFor",    "purchasing goods",      0.97f},
            {"medicine",     "UsedFor",    "treating illness",      0.96f},
        };
        for (auto& [s, r, o, w] : seed)
            knowledge_graph.push_back({s, r, o, w, {}, {}});
    }

    void seed_physical_schemas() {
        physical_schemas = {
            {"rock",   true,  true,  false, false, 2.0f,  0.15f, {"throwing", "building"}},
            {"water",  true,  true,  false, true,  0.0f,  0.0f,  {"drinking", "cleaning", "swimming"}},
            {"feather",true,  false, false, false, 0.001f,0.1f,  {"writing (quill)", "decoration"}},
            {"glass",  true,  true,  false, false, 0.5f,  0.1f,  {"drinking", "windows"}},
            {"car",    true,  true,  false, false, 1500.f,4.5f,  {"transportation"}},
            {"book",   true,  true,  false, false, 0.4f,  0.25f, {"reading", "reference"}},
        };
    }
};
