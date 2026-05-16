#include <torch/torch.h>
#include <torch/distributed.h>
#include <iostream>
#include <vector>
#include <string>

// --- Core model ---
#include "core/model/config.h"
#include "core/model/gpt.cpp"
#include "core/dataloader/dataset.cpp"
#include "core/distributed/nccl.cpp"
#include "core/distributed/fsdp.cpp"
#include "core/optimizer/zero.cpp"

// --- Safety & Logging ---
#include "core/safety/moderation.cpp"
#include "core/logging/audit.cpp"

// --- AGI subsystems ---
#include "core/agi/agi_core.cpp"

/*
=====================================================
  TITANCORE AGI: MASTER ORCHESTRATOR
=====================================================
  Cognitive loop:
    Perceive -> Remember -> Reason -> Plan -> Act
             -> Learn  -> Update World Model

  Distributed training loop:
    Load Data -> Safety Gate -> Forward ->
    Loss -> Backward -> ZeRO-3 Step -> Online Learn
=====================================================
*/

int main(int argc, char** argv) {
    // ------------------------------------------------
    // 1. Distributed Environment
    // ------------------------------------------------
    torch::distributed::init_process_group(torch::distributed::Backend::NCCL);
    int rank       = torch::distributed::get_rank();
    int world_size = torch::distributed::get_world_size();
    c10::cuda::set_device(rank % torch::cuda::device_count());

    // ------------------------------------------------
    // 2. Load Configuration
    // ------------------------------------------------
    TitanConfig cfg;
    load_config("core/configs/gpt4o.yaml", cfg);

    // ------------------------------------------------
    // 3. Safety & Audit
    // ------------------------------------------------
    TitanModeration safety;
    TitanAuditLogger audit(cfg.log_path);

    // ------------------------------------------------
    // 4. Distributed Communication & FSDP
    // ------------------------------------------------
    init_nccl(rank, world_size);
    TitanNCCLManager* comm = get_nccl();
    TitanFSDPManager  fsdp_manager(cfg, comm);

    // ------------------------------------------------
    // 5. KV Cache & Model
    // ------------------------------------------------
    PagedCacheConfig cache_cfg;
    cache_cfg.max_num_blocks = cfg.max_blocks;
    cache_cfg.n_layer        = cfg.n_layer;
    cache_cfg.n_head         = cfg.n_head;
    cache_cfg.head_dim       = cfg.n_embd / cfg.n_head;
    KVCacheManagerPaged kv_cache(cache_cfg);

    TitanGPT model(cfg);

    // ------------------------------------------------
    // 6. ZeRO-3 Optimizer
    // ------------------------------------------------
    TitanZeRO3 optimizer(cfg, model->blocks, fsdp_manager);

    // ------------------------------------------------
    // 7. AGI Core — bind all cognitive subsystems
    // ------------------------------------------------
    // Generator lambda: calls the language model to produce text
    auto generate_fn = [&](const std::string& prompt) -> std::string {
        // In production: tokenise prompt, run model->forward(), decode output
        // Stub returns the prompt summary for structural completeness
        return "[Generated response to: " + prompt.substr(0, 60) + "...]";
    };

    TitanAGI agi(torch::nn::AnyModule(model), cfg, generate_fn);

    if (rank == 0) {
        agi.set_goal("Learn from all available data and assist users effectively.");
        agi.print_status();
    }

    // ------------------------------------------------
    // 8. Dataset
    // ------------------------------------------------
    TitanDataset dataset(cfg.dataset_path);

    // ------------------------------------------------
    // 9. Training + Continuous Learning Loop
    // ------------------------------------------------
    if (rank == 0) std::cout << "\n[TitanCore AGI] Training started." << std::endl;

    for (int step = 0; step < cfg.max_steps; ++step) {

        // A. Load batch
        auto [input, targets] = dataset.get_batch(cfg.batch_size, cfg.seq_len, rank);

        // B. Safety gate
        if (!safety.is_safe_batch(input)) {
            audit.log(step, rank, "UNSAFE_BATCH_SKIPPED");
            continue;
        }

        // C. Forward pass
        torch::Tensor logits = model->forward(input, &kv_cache, step);

        // D. Loss
        auto loss = torch::nn::functional::cross_entropy(
            logits.view({-1, cfg.vocab_size}),
            targets.view({-1})
        );

        // E. Backward + ZeRO-3 optimizer step
        loss.backward();
        optimizer.step(model->blocks);

        // F. Continuous learning: ingest this batch into the online learner
        //    (rank 0 manages the online learner to avoid duplication)
        if (rank == 0)
            agi.learn_from_interaction(input[0], targets[0]);

        // G. Logging
        if (rank == 0 && step % 100 == 0) {
            float lv = loss.item<float>();
            std::cout << "[Step " << step << "] Loss: " << lv << std::endl;
            audit.log(step, rank, "loss=" + std::to_string(lv));
        }

        // H. Periodic AGI cognitive loop demo (every 1000 steps on rank 0)
        if (rank == 0 && step > 0 && step % 1000 == 0) {
            std::string response = agi.process(
                "Summarise what you have learned so far.",
                /*session_id=*/0,
                CoTMode::REFLECTION
            );
            audit.log(step, rank, "AGI_INTROSPECTION");
        }
    }

    // ------------------------------------------------
    // 10. Cleanup
    // ------------------------------------------------
    torch::distributed::destroy_process_group();
    if (rank == 0) std::cout << "[TitanCore AGI] Session complete." << std::endl;

    return 0;
}
