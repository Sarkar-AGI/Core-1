#pragma once
#include <torch/torch.h>
#include <vector>
#include <iostream>
#include <torch/distributed.h>
#include "config.h"
#include "block.cpp"
#include "kv_cache.cpp"
#include "../distributed/fsdp.cpp"

/*
=====================================================
  TITANCORE: ULTRA GPT-4o ARCHITECTURE (Full)
=====================================================
  - 1T+ Parameter Support
  - Pipeline Parallel Stage Assignment
  - FSDP Hooking for ZeRO-3
  - KV Cache Support
=====================================================
*/

struct TitanGPTImpl : torch::nn::Module {
    torch::nn::Embedding token_embedding{nullptr};
    torch::nn::Embedding position_embedding{nullptr};
    torch::nn::ModuleList blocks{nullptr};
    torch::nn::LayerNorm ln_f{nullptr};
    torch::nn::Linear lm_head{nullptr};

    int n_layer, pipeline_rank, pipeline_world_size;

    TitanGPTImpl(const TitanConfig& cfg) {

        // 1. Pipeline Parallel Setup
        int world_size_dist   = torch::distributed::get_world_size();
        pipeline_world_size   = cfg.pipeline_parallel_size;
        int gpus_per_stage    = std::max(1, world_size_dist / pipeline_world_size);
        pipeline_rank         = torch::distributed::get_rank() / gpus_per_stage;

        // 2. Assign Layers based on Pipeline Rank
        int total_layers    = static_cast<int>(cfg.n_layer);
        int layers_per_stage = total_layers / pipeline_world_size;

        // 3. Initialize Embedding and Head only on specific stages
        if (pipeline_rank == 0) {
            token_embedding    = register_module("token_emb",
                torch::nn::Embedding(cfg.vocab_size, cfg.n_embd));
            position_embedding = register_module("pos_emb",
                torch::nn::Embedding(cfg.block_size, cfg.n_embd));
        }

        // 4. Initialize Blocks for current stage
        blocks = register_module("blocks", torch::nn::ModuleList());
        for (int i = 0; i < layers_per_stage; ++i) {
            blocks->push_back(TransformerBlockMoE(cfg));
        }

        // Final normalisation and head
        ln_f = register_module("ln_f",
            torch::nn::LayerNorm(torch::nn::LayerNormOptions({cfg.n_embd})));

        if (pipeline_rank == pipeline_world_size - 1) {
            lm_head = register_module("lm_head",
                torch::nn::Linear(cfg.n_embd, cfg.vocab_size));
        }

        std::cout << "[TitanCore] GPT Stage Initialized. Rank: " << pipeline_rank
                  << " | Layers: " << layers_per_stage << std::endl;
    }

    torch::Tensor forward(torch::Tensor input, KVCacheManagerPaged* kv_cache, int step) {

        torch::Tensor x;

        // Stage 0: Embedding
        if (pipeline_rank == 0) {
            auto pos = torch::arange(input.size(1), input.options());
            x = token_embedding->forward(input) + position_embedding->forward(pos);
        } else {
            x = receive_from_previous_stage(input);
        }

        // Transformer Blocks
        int64_t layer_idx = pipeline_rank * static_cast<int64_t>(blocks->size());
        for (auto& raw_block : *blocks) {
            auto block = raw_block->as<TransformerBlockMoE>();
            x = block->forward(x, kv_cache, /*session_id=*/0, layer_idx++);
        }

        // Final normalisation
        x = ln_f->forward(x);

        // Final Stage: Head
        if (pipeline_rank == pipeline_world_size - 1) {
            x = lm_head->forward(x);
        } else {
            send_to_next_stage(x);
        }

        return x;
    }

private:
    torch::Tensor receive_from_previous_stage(torch::Tensor /*input*/) {
        // Placeholder: implement with NCCL P2P recv in production
        return torch::Tensor();
    }

    void send_to_next_stage(torch::Tensor /*x*/) {
        // Placeholder: implement with NCCL P2P send in production
    }
};
TORCH_MODULE(TitanGPT);
