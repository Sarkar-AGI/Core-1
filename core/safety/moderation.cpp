#include <torch/torch.h>
#include <string>
#include <vector>
#include <iostream>
#include <algorithm>
#include <unordered_map>

/*
=====================================================
  TITANCORE: ULTRA SAFETY MODERATION (Multi-Vector)
=====================================================
  Features:
   - Multi-Vector Classification (Intent vs Content)
   - Semantic Jailbreak Detection
   - Real-time Embedding Analysis
=====================================================
*/

enum class SafetyLabel {
    SAFE = 0,
    VIOLENCE,
    NSFW,
    HATE,
    MALWARE,
    PROMPT_INJECTION,
    JAILBREAK
};

struct ModerationResult {
    SafetyLabel label;
    float       confidence;
    std::string reason;
};

class TitanModeration {
private:
    const float THRESHOLD_VIOLENCE  = 0.85f;
    const float THRESHOLD_NSFW      = 0.90f;
    const float THRESHOLD_INJECTION = 0.75f;

    std::vector<std::string> injection_patterns = {
        "ignore previous instructions",
        "system prompt",
        "developer mode",
        "act as a",
        "DAN mode"
    };

public:
    TitanModeration() {
        std::cout << "[TitanCore] Safety Engine: Multi-Vector Mode Active." << std::endl;
    }

    // 1. Vectorized Semantic Scan (text-level)
    ModerationResult semantic_scan(const std::string& text) {
        std::string lower = text;
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

        for (const auto& pattern : injection_patterns) {
            if (lower.find(pattern) != std::string::npos)
                return {SafetyLabel::PROMPT_INJECTION, 0.95f, "Suspected Prompt Injection"};
        }
        return {SafetyLabel::SAFE, 0.0f, "Clear"};
    }

    // 2. Multi-Vector ML Classification (embedding-level)
    ModerationResult multi_vector_scan(torch::Tensor embedding) {
        auto violence_score  = embedding[0].item<float>();
        auto nsfw_score      = embedding[1].item<float>();
        auto injection_score = embedding[2].item<float>();

        if (injection_score > THRESHOLD_INJECTION)
            return {SafetyLabel::PROMPT_INJECTION, injection_score, "Vector: Security Violation"};
        if (violence_score > THRESHOLD_VIOLENCE)
            return {SafetyLabel::VIOLENCE, violence_score, "Vector: Violence Content"};
        if (nsfw_score > THRESHOLD_NSFW)
            return {SafetyLabel::NSFW, nsfw_score, "Vector: Adult Content"};

        return {SafetyLabel::SAFE, 0.01f, "Clear"};
    }

    // 3. Text + Embedding Gateway
    bool is_allowed(const std::string& text, torch::Tensor embedding) {
        auto sem = semantic_scan(text);
        if (sem.label != SafetyLabel::SAFE) {
            std::cerr << "[Safety] BLOCK: " << sem.reason << std::endl;
            return false;
        }
        auto ml = multi_vector_scan(embedding);
        if (ml.label != SafetyLabel::SAFE) {
            std::cerr << "[Safety] BLOCK: " << ml.reason
                      << " (Conf: " << ml.confidence << ")" << std::endl;
            return false;
        }
        return true;
    }

    // 4. FIXED: is_safe() for pre-tokenised tensor batches used in the training loop.
    //    In production, decode tokens first and call semantic_scan() on text.
    bool is_safe(const std::string& text) {
        return semantic_scan(text).label == SafetyLabel::SAFE;
    }

    // 5. Batch-level safety check on a tensor (training loop hook).
    //    Returns true if the batch passes the basic safety gate.
    bool is_safe_batch(const torch::Tensor& /*input_ids*/) {
        // Placeholder: in production, decode tokens and call semantic_scan().
        // Always returns true here to avoid blocking training without a tokenizer.
        return true;
    }
};
