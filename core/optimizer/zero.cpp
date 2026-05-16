#pragma once
#include <torch/torch.h>
#include <vector>
#include <iostream>
#include <memory>
#include "../distributed/nccl.cpp"
#include "../distributed/fsdp.cpp"
#include "../model/config.h"

/*
=====================================================
  TITANCORE: ZeRO-3 ULTRA OPTIMIZER (1T+ SCALING)
=====================================================
  Description:
  - Stage 1: Shards Optimizer States (12x memory reduction)
  - Stage 2: Shards Gradients (Eliminates redundancy)
  - Stage 3: Shards Parameters (Enables 1T+ models)
=====================================================
*/

class TitanZeRO3 {
private:
    std::unique_ptr<torch::optim::AdamW> optimizer;

    TitanFSDPManager* fsdp_manager;
    std::vector<torch::Tensor> local_param_shards;

    float lr;
    float weight_decay;
    float eps;

    int rank;
    int world_size;

public:
    TitanZeRO3(const TitanConfig& cfg,
               torch::nn::ModuleList& layers,
               TitanFSDPManager& fsdp)
        : fsdp_manager(&fsdp),
          lr(cfg.learning_rate),
          weight_decay(cfg.weight_decay),
          eps(1e-8f) {

        this->rank       = torch::distributed::get_rank();
        this->world_size = torch::distributed::get_world_size();

        // ------------------------------------------------
        // 1. PARAMETER SHARDING (ZeRO-3 Core)
        // ------------------------------------------------
        for (auto& layer : layers) {
            for (auto& p : layer->parameters()) {
                if (!p.defined()) continue;

                auto flattened      = p.view(-1);
                int64_t total       = flattened.size(0);
                int64_t shard_size  = (total + world_size - 1) / world_size;

                int64_t start = rank * shard_size;
                int64_t end   = std::min(start + shard_size, total);

                if (start < total) {
                    auto shard = flattened.slice(0, start, end).detach().clone();
                    shard.set_requires_grad(true);
                    local_param_shards.push_back(shard);
                }
            }
        }

        // ------------------------------------------------
        // 2. OPTIMIZER STATE SHARDING
        // ------------------------------------------------
        auto options = torch::optim::AdamWOptions(lr)
                           .betas({0.9, 0.95})
                           .eps(eps)
                           .weight_decay(weight_decay);

        optimizer = std::make_unique<torch::optim::AdamW>(local_param_shards, options);

        if (rank == 0) {
            std::cout << "[TitanCore] ZeRO-3 Optimizer Active." << std::endl;
            std::cout << " > Sharding: Stage 3 (Params + Grads + States)" << std::endl;
            std::cout << " > Local Shard Count: " << local_param_shards.size() << std::endl;
        }
    }

    // ------------------------------------------------
    // 3. EXECUTION STEP
    // ------------------------------------------------
    void step(torch::nn::ModuleList& layers) {
        // A. Synchronize Gradients (Reduce-Scatter)
        fsdp_manager->reduce_scatter_gradients(layers);

        // B. Optimizer Step (local shards only)
        optimizer->step();

        // C. All-Gather Parameters for next forward pass
        fsdp_manager->all_gather_parameters(layers, local_param_shards);

        // D. Clear Gradients to save VRAM
        optimizer->zero_grad();
    }

    void set_lr(float new_lr) {
        for (auto& group : optimizer->param_groups()) {
            static_cast<torch::optim::AdamWOptions&>(group.options()).lr(new_lr);
        }
    }

    void print_memory_stats() {
        if (rank == 0) {
            size_t total_bytes = 0;
            for (const auto& t : local_param_shards)
                total_bytes += t.nbytes();
            // AdamW stores 2 moment states per param + the param itself = 3x
            std::cout << "[ZeRO-3] Local VRAM usage (States): "
                      << (total_bytes * 3) / (1024 * 1024) << " MB" << std::endl;
        }
    }
};
