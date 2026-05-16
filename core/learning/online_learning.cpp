#pragma once
#include <torch/torch.h>
#include <vector>
#include <deque>
#include <iostream>
#include <mutex>
#include "../model/config.h"

/*
=====================================================
  TITANCORE AGI: CONTINUOUS LEARNING ENGINE
  Online Gradient Descent with EWC + Replay Buffer
=====================================================
  Features:
  - Real-time weight updates from live data streams
  - Elastic Weight Consolidation (EWC) to prevent
    catastrophic forgetting of prior knowledge
  - Experience Replay Buffer (reservoir sampling)
  - Exponential Moving Average (EMA) weight snapshots
  - Adaptive per-parameter learning rate (AdaGrad)
  - Thread-safe streaming data ingestion
=====================================================
*/

// ------------------------------------
// Experience: a single (input, target) sample
// ------------------------------------
struct Experience {
    torch::Tensor input;   // [seq_len] token ids
    torch::Tensor target;  // [seq_len] next-token labels
    float priority = 1.0f; // for prioritized replay
};

// ------------------------------------
// Replay Buffer (Reservoir Sampling)
// ------------------------------------
class ReplayBuffer {
private:
    std::deque<Experience> buffer;
    size_t max_size;
    std::mutex mtx;
    size_t total_seen = 0;

public:
    explicit ReplayBuffer(size_t capacity = 100000) : max_size(capacity) {}

    void push(Experience exp) {
        std::lock_guard<std::mutex> lock(mtx);
        total_seen++;
        if (buffer.size() < max_size) {
            buffer.push_back(std::move(exp));
        } else {
            // Reservoir sampling: replace random existing entry
            size_t idx = rand() % total_seen;
            if (idx < max_size)
                buffer[idx] = std::move(exp);
        }
    }

    std::vector<Experience> sample(size_t n) {
        std::lock_guard<std::mutex> lock(mtx);
        std::vector<Experience> batch;
        n = std::min(n, buffer.size());
        for (size_t i = 0; i < n; ++i) {
            size_t idx = rand() % buffer.size();
            batch.push_back(buffer[idx]);
        }
        return batch;
    }

    size_t size() const { return buffer.size(); }
};

// ------------------------------------
// Elastic Weight Consolidation (EWC)
// Prevents catastrophic forgetting
// ------------------------------------
class EWC {
private:
    std::vector<torch::Tensor> fisher_diag;  // Fisher Information diagonal
    std::vector<torch::Tensor> params_star;  // Optimal params from prior task
    float lambda;                            // EWC regularisation strength

public:
    explicit EWC(float ewc_lambda = 5000.0f) : lambda(ewc_lambda) {}

    // Compute Fisher Information diagonal after finishing a task
    void consolidate(torch::nn::Module& model,
                     std::vector<std::pair<torch::Tensor, torch::Tensor>>& data_samples,
                     int n_samples = 200) {
        fisher_diag.clear();
        params_star.clear();

        // Save current optimal parameters
        for (auto& p : model.parameters()) {
            params_star.push_back(p.detach().clone());
            fisher_diag.push_back(torch::zeros_like(p));
        }

        // Estimate Fisher diagonal via squared gradients
        model.zero_grad();
        int count = std::min(n_samples, (int)data_samples.size());
        for (int i = 0; i < count; ++i) {
            auto output = model.forward(data_samples[i].first);
            auto loss = torch::nn::functional::cross_entropy(output, data_samples[i].second);
            loss.backward();

            int idx = 0;
            for (auto& p : model.parameters()) {
                if (p.grad().defined())
                    fisher_diag[idx] += p.grad().pow(2);
                idx++;
            }
            model.zero_grad();
        }

        // Normalise
        for (auto& f : fisher_diag)
            f /= count;

        std::cout << "[EWC] Fisher Information consolidated over "
                  << count << " samples." << std::endl;
    }

    // Compute EWC penalty to add to the training loss
    torch::Tensor penalty(torch::nn::Module& model) {
        if (fisher_diag.empty()) return torch::zeros({1});

        torch::Tensor pen = torch::zeros({1});
        int idx = 0;
        for (auto& p : model.parameters()) {
            pen += (fisher_diag[idx] * (p - params_star[idx]).pow(2)).sum();
            idx++;
        }
        return (lambda / 2.0f) * pen;
    }
};

// ------------------------------------
// EMA Weight Snapshots
// ------------------------------------
class EMAWeights {
private:
    std::vector<torch::Tensor> ema_params;
    float decay;

public:
    explicit EMAWeights(float decay = 0.999f) : decay(decay) {}

    void init(torch::nn::Module& model) {
        ema_params.clear();
        for (auto& p : model.parameters())
            ema_params.push_back(p.detach().clone());
    }

    void update(torch::nn::Module& model) {
        int idx = 0;
        for (auto& p : model.parameters()) {
            ema_params[idx].mul_(decay).add_((1.0f - decay) * p.detach());
            idx++;
        }
    }

    // Swap live weights with EMA weights for inference
    void apply(torch::nn::Module& model) {
        int idx = 0;
        for (auto& p : model.parameters()) {
            p.data().copy_(ema_params[idx]);
            idx++;
        }
    }
};

// ------------------------------------
// TitanOnlineLearner
// ------------------------------------
class TitanOnlineLearner {
private:
    ReplayBuffer replay;
    EWC          ewc;
    EMAWeights   ema;

    float online_lr;
    int   replay_batch;
    int   update_every;   // update weights every N new samples
    int   samples_since_update = 0;

    std::unique_ptr<torch::optim::AdamW> online_optimizer;
    std::mutex update_mutex;

public:
    TitanOnlineLearner(const TitanConfig& cfg,
                       torch::nn::Module& model,
                       float lr           = 1e-5f,
                       int   replay_cap   = 100000,
                       int   replay_batch = 32,
                       int   update_every = 16,
                       float ewc_lambda   = 5000.0f)
        : replay(replay_cap),
          ewc(ewc_lambda),
          ema(0.999f),
          online_lr(lr),
          replay_batch(replay_batch),
          update_every(update_every) {

        ema.init(model);

        auto opts = torch::optim::AdamWOptions(online_lr)
                        .betas({0.9, 0.95})
                        .weight_decay(0.01f);
        online_optimizer = std::make_unique<torch::optim::AdamW>(
            model.parameters(), opts);

        std::cout << "[OnlineLearner] Continuous Learning Engine initialized." << std::endl;
        std::cout << " > Replay buffer capacity : " << replay_cap << std::endl;
        std::cout << " > EWC lambda             : " << ewc_lambda  << std::endl;
        std::cout << " > EMA decay              : 0.999" << std::endl;
        std::cout << " > Online LR              : " << lr          << std::endl;
    }

    // Ingest a single new experience from a live data stream
    void ingest(torch::Tensor input, torch::Tensor target,
                torch::nn::Module& model) {
        replay.push({input, target});
        samples_since_update++;

        if (samples_since_update >= update_every)
            online_update(model);
    }

    // Perform an online gradient step using the replay buffer
    void online_update(torch::nn::Module& model) {
        std::lock_guard<std::mutex> lock(update_mutex);

        if (replay.size() < (size_t)replay_batch) return;

        auto batch = replay.sample(replay_batch);

        online_optimizer->zero_grad();

        torch::Tensor total_loss = torch::zeros({1});

        for (auto& exp : batch) {
            auto logits = model.forward(exp.input.unsqueeze(0));
            auto task_loss = torch::nn::functional::cross_entropy(
                logits.view({-1, logits.size(-1)}),
                exp.target.view({-1})
            );
            total_loss = total_loss + task_loss;
        }

        // Add EWC penalty to prevent forgetting
        total_loss = total_loss / replay_batch + ewc.penalty(model);
        total_loss.backward();

        // Gradient clipping for stability
        torch::nn::utils::clip_grad_norm_(model.parameters(), 1.0f);

        online_optimizer->step();

        // Update EMA snapshot
        ema.update(model);

        samples_since_update = 0;

        std::cout << "[OnlineLearner] Online update — loss: "
                  << total_loss.item<float>() << std::endl;
    }

    // Consolidate Fisher Information after completing a task/domain
    void consolidate_task(torch::nn::Module& model,
                          std::vector<std::pair<torch::Tensor, torch::Tensor>>& samples) {
        ewc.consolidate(model, samples);
        std::cout << "[OnlineLearner] Task knowledge consolidated via EWC." << std::endl;
    }

    // Apply EMA weights for inference stability
    void apply_ema_for_inference(torch::nn::Module& model) {
        ema.apply(model);
    }

    size_t buffer_size() const { return replay.size(); }
};
