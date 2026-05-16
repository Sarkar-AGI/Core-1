#pragma once
#include <torch/torch.h>
#include <vector>
#include <deque>
#include <string>
#include <iostream>
#include <mutex>

/*
=====================================================
  TITANCORE AGI: WORKING MEMORY
=====================================================
  Short-term context buffer for active reasoning.
  Manages the model's active context window, ensuring
  the most relevant tokens and thoughts are kept
  within the finite attention budget.

  Implements:
  - Sliding context window with priority eviction
  - Thought-chain slot (scratchpad for reasoning)
  - Attention-weighted token importance scoring
  - Automatic summarisation trigger when full
=====================================================
*/

struct ContextSlot {
    std::string  role;      // "user", "assistant", "thought", "tool_result"
    std::string  content;
    torch::Tensor embedding; // [D] slot embedding (optional)
    float        importance; // attention-weight-based salience
    int          token_len;
};

class TitanWorkingMemory {
private:
    std::deque<ContextSlot> slots;
    int max_tokens;
    int current_tokens = 0;
    std::mutex mtx;

    // Internal scratchpad for chain-of-thought reasoning steps
    std::vector<std::string> thought_chain;

public:
    explicit TitanWorkingMemory(int token_budget = 8192) : max_tokens(token_budget) {
        std::cout << "[WorkingMemory] Initialized. Token budget: "
                  << token_budget << std::endl;
    }

    // Add a context slot (evict least important if full)
    void push(const std::string& role,
              const std::string& content,
              int token_len,
              float importance = 1.0f,
              torch::Tensor embedding = {}) {

        std::lock_guard<std::mutex> lock(mtx);

        // Evict until there is room
        while (current_tokens + token_len > max_tokens && !slots.empty()) {
            // Find and remove least important non-system slot
            auto min_it = std::min_element(
                slots.begin(), slots.end(),
                [](const ContextSlot& a, const ContextSlot& b) {
                    return a.importance < b.importance;
                });
            current_tokens -= min_it->token_len;
            slots.erase(min_it);
        }

        ContextSlot slot;
        slot.role       = role;
        slot.content    = content;
        slot.importance = importance;
        slot.token_len  = token_len;
        if (embedding.defined())
            slot.embedding = embedding.detach().clone();

        slots.push_back(std::move(slot));
        current_tokens += token_len;
    }

    // Add a reasoning step to the thought chain (scratchpad)
    void think(const std::string& thought) {
        std::lock_guard<std::mutex> lock(mtx);
        thought_chain.push_back(thought);
        if (thought_chain.size() > 64)
            thought_chain.erase(thought_chain.begin()); // rolling window
    }

    // Clear the scratchpad between tasks
    void clear_thoughts() {
        std::lock_guard<std::mutex> lock(mtx);
        thought_chain.clear();
    }

    // Build a flat token sequence from all active slots
    std::string build_context() const {
        std::string ctx;
        for (auto& s : slots)
            ctx += "[" + s.role + "] " + s.content + "\n";
        return ctx;
    }

    // Retrieve the current thought chain as a string
    std::string get_thought_chain() const {
        std::string out;
        for (auto& t : thought_chain)
            out += "<thought> " + t + " </thought>\n";
        return out;
    }

    // Update importance scores using attention weights from the model
    void update_importance(const std::vector<float>& attn_weights) {
        std::lock_guard<std::mutex> lock(mtx);
        size_t n = std::min(attn_weights.size(), slots.size());
        for (size_t i = 0; i < n; ++i)
            slots[i].importance = 0.9f * slots[i].importance + 0.1f * attn_weights[i];
    }

    int  token_count()  const { return current_tokens; }
    int  slot_count()   const { return static_cast<int>(slots.size()); }
    bool is_full()      const { return current_tokens >= max_tokens; }

    void print_stats() const {
        std::cout << "[WorkingMemory] Slots: " << slots.size()
                  << " | Tokens: " << current_tokens << " / " << max_tokens
                  << " | Thoughts: " << thought_chain.size() << std::endl;
    }
};
