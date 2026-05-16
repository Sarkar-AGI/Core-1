#pragma once
#include <torch/torch.h>
#include <vector>
#include <deque>
#include <string>
#include <unordered_map>
#include <iostream>
#include <chrono>
#include <mutex>

/*
=====================================================
  TITANCORE AGI: EPISODIC MEMORY
=====================================================
  Stores and retrieves past interaction episodes.
  Enables the model to recall specific events,
  context from prior conversations, and temporal
  sequences — a critical AGI capability.

  Implements:
  - Fixed-capacity circular episode store
  - Embedding-based associative retrieval (cosine sim)
  - Temporal decay weighting (recent = more relevant)
  - Tag-based episode filtering
=====================================================
*/

struct Episode {
    int64_t     id;
    std::string tag;                // e.g. "conversation", "task", "observation"
    torch::Tensor embedding;        // [D] — summary embedding of the episode
    std::string text_summary;       // human-readable summary
    std::chrono::system_clock::time_point timestamp;
    float importance = 1.0f;        // salience score (updated by consolidation)
};

class TitanEpisodicMemory {
private:
    std::deque<Episode>   store;
    size_t                capacity;
    int64_t               next_id = 0;
    std::mutex            mtx;

    // Cosine similarity between two embedding vectors
    float cosine_sim(const torch::Tensor& a, const torch::Tensor& b) {
        auto norm_a = a.norm().item<float>();
        auto norm_b = b.norm().item<float>();
        if (norm_a < 1e-8f || norm_b < 1e-8f) return 0.0f;
        return torch::dot(a.view(-1), b.view(-1)).item<float>() / (norm_a * norm_b);
    }

    // Temporal decay: episodes further in the past score lower
    float temporal_weight(const Episode& ep) const {
        auto now     = std::chrono::system_clock::now();
        auto age_sec = std::chrono::duration<float>(now - ep.timestamp).count();
        // Half-life of 24 hours
        return std::exp(-age_sec / 86400.0f);
    }

public:
    explicit TitanEpisodicMemory(size_t cap = 10000) : capacity(cap) {
        std::cout << "[EpisodicMemory] Initialized. Capacity: " << cap << " episodes." << std::endl;
    }

    // Store a new episode
    int64_t store_episode(const std::string& tag,
                          torch::Tensor embedding,
                          const std::string& summary,
                          float importance = 1.0f) {
        std::lock_guard<std::mutex> lock(mtx);

        Episode ep;
        ep.id           = next_id++;
        ep.tag          = tag;
        ep.embedding    = embedding.detach().clone();
        ep.text_summary = summary;
        ep.timestamp    = std::chrono::system_clock::now();
        ep.importance   = importance;

        if (store.size() >= capacity) {
            // Evict least important + oldest episode
            store.pop_front();
        }
        store.push_back(std::move(ep));
        return ep.id;
    }

    // Retrieve top-K most relevant episodes for a query embedding
    std::vector<Episode> retrieve(torch::Tensor query_emb,
                                  int top_k = 5,
                                  const std::string& tag_filter = "") {
        std::lock_guard<std::mutex> lock(mtx);

        std::vector<std::pair<float, int>> scores;
        for (int i = 0; i < (int)store.size(); ++i) {
            if (!tag_filter.empty() && store[i].tag != tag_filter) continue;
            float sim  = cosine_sim(query_emb, store[i].embedding);
            float tw   = temporal_weight(store[i]);
            float score = sim * tw * store[i].importance;
            scores.push_back({score, i});
        }

        std::sort(scores.begin(), scores.end(),
                  [](auto& a, auto& b) { return a.first > b.first; });

        std::vector<Episode> results;
        for (int i = 0; i < std::min(top_k, (int)scores.size()); ++i)
            results.push_back(store[scores[i].second]);

        return results;
    }

    // Boost importance of an episode (reinforce salient memories)
    void reinforce(int64_t episode_id, float boost = 0.1f) {
        std::lock_guard<std::mutex> lock(mtx);
        for (auto& ep : store) {
            if (ep.id == episode_id) {
                ep.importance = std::min(ep.importance + boost, 10.0f);
                break;
            }
        }
    }

    size_t size() const { return store.size(); }

    void print_stats() const {
        std::cout << "[EpisodicMemory] Episodes stored: " << store.size()
                  << " / " << capacity << std::endl;
    }
};
