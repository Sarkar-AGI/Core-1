#pragma once
#include <string>

/*
=====================================================
  TITANCORE: CENTRAL CONFIGURATION STRUCT
=====================================================
  All hyperparameters and runtime settings loaded
  from gpt4o.yaml / cluster.yaml at startup.
=====================================================
*/

struct TitanConfig {
    // --- Model Architecture ---
    int64_t vocab_size          = 400000;
    int64_t block_size          = 8192;   // max_seq_len
    int64_t n_layer             = 120;
    int64_t n_head              = 128;
    int64_t n_embd              = 16384;
    int64_t n_experts           = 8;
    int64_t top_k               = 2;

    // --- Parallelism ---
    int tensor_parallel_size    = 4;
    int pipeline_parallel_size  = 2;
    int data_parallel_size      = 4;

    // --- KV Cache ---
    int64_t max_blocks          = 4096;

    // --- Training ---
    float  learning_rate        = 1e-4f;
    int    max_steps            = 1000000;
    int    batch_size           = 16;
    int    seq_len              = 8192;
    float  weight_decay         = 0.1f;
    float  grad_clip            = 1.0f;

    // --- Dataset ---
    std::string dataset_path    = "data/tokens.bin";
    std::string model_path      = "weights/titancore.gguf";
    std::string log_path        = "logs/security_audit.csv";

    // --- Runtime ---
    bool continuous_batching    = true;
    bool speculative_decoding   = true;
    bool prefix_cache           = true;

    // --- Quantization ---
    bool quantization_enabled   = true;
};

/*
 * Minimal YAML config loader.
 * In production, replace with a full YAML parser (e.g. yaml-cpp).
 * This stub loads a TitanConfig with defaults so the project compiles.
 */
inline void load_config(const std::string& /*yaml_path*/, TitanConfig& cfg) {
    // TODO: parse yaml_path with yaml-cpp and fill cfg fields
    (void)cfg;
}
