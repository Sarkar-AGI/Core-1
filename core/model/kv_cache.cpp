#pragma once
#include <torch/torch.h>
#include <vector>
#include <unordered_map>
#include <iostream>
#include <utility>

/*
=====================================================
  TITANCORE: ULTRA PAGED KV CACHE (Multi-Layer)
=====================================================
  Optimized for 1T+ models and 120+ layers.
  Implements Logical -> Physical block mapping.
=====================================================
*/

// Renamed from BLOCK_SIZE to avoid collision with attention.cu's BLOCK_SIZE
static constexpr int64_t KV_BLOCK_SIZE = 16;

struct PagedCacheConfig {
    int64_t max_num_blocks  = 4096;
    int64_t n_layer         = 120;
    int64_t n_head          = 128;
    int64_t head_dim        = 128;
    torch::Device device    = torch::kCUDA;
    torch::ScalarType dtype = torch::kHalf;
};

class KVCacheManagerPaged {
private:
    // Physical Memory Pool: [Layers, Blocks, KV_BLOCK_SIZE, Heads, Dim]
    torch::Tensor K_pool;
    torch::Tensor V_pool;

    std::vector<int64_t> free_blocks;
    std::unordered_map<int64_t, std::vector<int64_t>> block_tables;
    std::unordered_map<int64_t, int64_t> context_lens;
    PagedCacheConfig config;

public:
    explicit KVCacheManagerPaged(PagedCacheConfig cfg) : config(cfg) {
        auto options = torch::TensorOptions().dtype(cfg.dtype).device(cfg.device);

        K_pool = torch::empty(
            {cfg.n_layer, cfg.max_num_blocks, KV_BLOCK_SIZE, cfg.n_head, cfg.head_dim}, options);
        V_pool = torch::empty(
            {cfg.n_layer, cfg.max_num_blocks, KV_BLOCK_SIZE, cfg.n_head, cfg.head_dim}, options);

        free_blocks.reserve(cfg.max_num_blocks);
        for (int64_t i = cfg.max_num_blocks - 1; i >= 0; --i)
            free_blocks.push_back(i);

        std::cout << "[TitanCore] KV Cache Initialized: " << cfg.n_layer << " Layers | "
                  << cfg.max_num_blocks << " Blocks per Layer." << std::endl;
    }

    int64_t allocate_block() {
        if (free_blocks.empty())
            throw std::runtime_error("KV Cache OOM: No free blocks left!");
        int64_t idx = free_blocks.back();
        free_blocks.pop_back();
        return idx;
    }

    void append(int64_t session_id, int64_t layer_idx, torch::Tensor k, torch::Tensor v) {
        if (block_tables.find(session_id) == block_tables.end()) {
            block_tables[session_id] = {};
            context_lens[session_id] = 0;
        }

        int64_t cur_len           = context_lens[session_id];
        int64_t logical_block_idx = cur_len / KV_BLOCK_SIZE;
        int64_t offset_in_block   = cur_len % KV_BLOCK_SIZE;

        if (offset_in_block == 0) {
            block_tables[session_id].push_back(allocate_block());
        }

        int64_t physical_block_idx = block_tables[session_id][logical_block_idx];

        using namespace torch::indexing;
        auto k_sq = k.squeeze(0).squeeze(0);
        auto v_sq = v.squeeze(0).squeeze(0);

        K_pool.index_put_(
            {layer_idx, physical_block_idx, offset_in_block, Slice(), Slice()}, k_sq);
        V_pool.index_put_(
            {layer_idx, physical_block_idx, offset_in_block, Slice(), Slice()}, v_sq);

        if (layer_idx == config.n_layer - 1)
            context_lens[session_id]++;
    }

    // Return concatenated K and V for a session / layer
    std::pair<torch::Tensor, torch::Tensor> get_full_kv(int64_t session_id, int64_t layer_idx) {
        auto& blocks   = block_tables[session_id];
        int64_t ctx    = context_lens[session_id];
        int64_t n_blks = static_cast<int64_t>(blocks.size());

        using namespace torch::indexing;
        // Gather all physical blocks for this session
        std::vector<torch::Tensor> k_tensors, v_tensors;
        for (int64_t b = 0; b < n_blks; ++b) {
            int64_t phys = blocks[b];
            k_tensors.push_back(K_pool.index({layer_idx, phys, Slice(), Slice(), Slice()}));
            v_tensors.push_back(V_pool.index({layer_idx, phys, Slice(), Slice(), Slice()}));
        }

        auto K_full = torch::cat(k_tensors, 0).slice(0, 0, ctx); // [ctx, n_head, head_dim]
        auto V_full = torch::cat(v_tensors, 0).slice(0, 0, ctx);

        // Reshape to [1, n_head, ctx, head_dim] for attention
        K_full = K_full.transpose(0, 1).unsqueeze(0);
        V_full = V_full.transpose(0, 1).unsqueeze(0);

        return {K_full, V_full};
    }

    torch::Tensor get_block_table(int64_t session_id) {
        auto& blks   = block_tables[session_id];
        auto options = torch::TensorOptions().dtype(torch::kInt64).device(config.device);
        return torch::from_blob((void*)blks.data(), {(int64_t)blks.size()}, options).clone();
    }

    void free_session(int64_t session_id) {
        if (block_tables.count(session_id)) {
            for (auto idx : block_tables[session_id]) free_blocks.push_back(idx);
            block_tables.erase(session_id);
            context_lens.erase(session_id);
        }
    }

    int64_t seq_len(int64_t session_id) {
        return context_lens.count(session_id) ? context_lens[session_id] : 0;
    }
};
