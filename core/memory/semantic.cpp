#pragma once
#include <torch/torch.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <iostream>
#include <mutex>

/*
=====================================================
  TITANCORE AGI: SEMANTIC MEMORY
=====================================================
  Long-term structured knowledge store.
  Stores factual knowledge as (key, embedding, value)
  triples — analogous to a neural knowledge graph.

  Implements:
  - Key-value knowledge insertion and lookup
  - Embedding-based fuzzy retrieval (cosine sim)
  - Conflict resolution (newer knowledge wins or merges)
  - Confidence scoring per knowledge entry
=====================================================
*/

struct KnowledgeEntry {
    std::string   key;          // Canonical concept key
    std::string   value;        // Textual fact / answer
    torch::Tensor embedding;    // [D] key embedding
    float         confidence;   // 0.0 - 1.0
    int           access_count; // How often retrieved
};

class TitanSemanticMemory {
private:
    std::vector<KnowledgeEntry> knowledge_base;
    std::mutex mtx;

    float cosine_sim(const torch::Tensor& a, const torch::Tensor& b) {
        auto na = a.norm().item<float>();
        auto nb = b.norm().item<float>();
        if (na < 1e-8f || nb < 1e-8f) return 0.0f;
        return torch::dot(a.view(-1), b.view(-1)).item<float>() / (na * nb);
    }

public:
    TitanSemanticMemory() {
        std::cout << "[SemanticMemory] Long-term knowledge store initialized." << std::endl;
    }

    // Insert or update a knowledge entry
    void store(const std::string& key,
               const std::string& value,
               torch::Tensor embedding,
               float confidence = 1.0f) {
        std::lock_guard<std::mutex> lock(mtx);

        // Check for existing entry with same key
        for (auto& entry : knowledge_base) {
            if (entry.key == key) {
                // Update if new confidence is higher
                if (confidence >= entry.confidence) {
                    entry.value      = value;
                    entry.embedding  = embedding.detach().clone();
                    entry.confidence = confidence;
                }
                return;
            }
        }

        knowledge_base.push_back({
            key, value, embedding.detach().clone(), confidence, 0
        });
    }

    // Retrieve top-K most relevant knowledge entries
    std::vector<KnowledgeEntry> query(torch::Tensor query_emb,
                                      int top_k = 5,
                                      float min_confidence = 0.3f) {
        std::lock_guard<std::mutex> lock(mtx);

        std::vector<std::pair<float, int>> scores;
        for (int i = 0; i < (int)knowledge_base.size(); ++i) {
            if (knowledge_base[i].confidence < min_confidence) continue;
            float sim = cosine_sim(query_emb, knowledge_base[i].embedding);
            scores.push_back({sim * knowledge_base[i].confidence, i});
        }

        std::sort(scores.begin(), scores.end(),
                  [](auto& a, auto& b) { return a.first > b.first; });

        std::vector<KnowledgeEntry> results;
        for (int i = 0; i < std::min(top_k, (int)scores.size()); ++i) {
            knowledge_base[scores[i].second].access_count++;
            results.push_back(knowledge_base[scores[i].second]);
        }
        return results;
    }

    // Exact key lookup
    std::string lookup(const std::string& key) {
        std::lock_guard<std::mutex> lock(mtx);
        for (auto& e : knowledge_base)
            if (e.key == key) return e.value;
        return "";
    }

    size_t size() const { return knowledge_base.size(); }

    void print_stats() const {
        std::cout << "[SemanticMemory] Knowledge entries: "
                  << knowledge_base.size() << std::endl;
    }
};
