#pragma once
#include <torch/torch.h>
#include <vector>
#include <string>
#include <iostream>
#include "../model/config.h"

/*
=====================================================
  TITANCORE AGI: WORLD MODEL
=====================================================
  An internal predictive model of the environment.
  Enables the AGI to:
  - Predict outcomes of actions before executing them
  - Simulate future states for planning
  - Build causal representations of the world
  - Detect anomalies (unexpected outcomes)

  Architecture:
  - Encoder  : maps observation -> latent state z
  - Dynamics : predicts next latent z' given z + action
  - Decoder  : maps z -> reconstructed observation
  - Reward   : predicts reward signal from z
=====================================================
*/

// ------------------------------------
// Latent World State
// ------------------------------------
struct WorldState {
    torch::Tensor z;            // [latent_dim] compressed world representation
    torch::Tensor uncertainty;  // [latent_dim] epistemic uncertainty estimate
    float         reward_pred;  // predicted reward
    bool          is_novel;     // true if state is unlike any seen before
};

// ------------------------------------
// Encoder: Observation -> Latent
// ------------------------------------
struct WorldEncoderImpl : torch::nn::Module {
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc_mu{nullptr}, fc_log_var{nullptr};

    WorldEncoderImpl(int obs_dim, int latent_dim) {
        fc1        = register_module("fc1",       torch::nn::Linear(obs_dim,    2048));
        fc2        = register_module("fc2",       torch::nn::Linear(2048,       1024));
        fc_mu      = register_module("fc_mu",     torch::nn::Linear(1024, latent_dim));
        fc_log_var = register_module("fc_log_var",torch::nn::Linear(1024, latent_dim));
    }

    // Returns (mu, log_var) — VAE-style for uncertainty estimation
    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor obs) {
        auto h = torch::relu(fc2(torch::relu(fc1(obs))));
        return {fc_mu(h), fc_log_var(h)};
    }
};
TORCH_MODULE(WorldEncoder);

// ------------------------------------
// Dynamics Model: (z, action_emb) -> z'
// ------------------------------------
struct DynamicsModelImpl : torch::nn::Module {
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc_out{nullptr};

    DynamicsModelImpl(int latent_dim, int action_dim) {
        fc1    = register_module("fc1",    torch::nn::Linear(latent_dim + action_dim, 2048));
        fc2    = register_module("fc2",    torch::nn::Linear(2048,                    1024));
        fc_out = register_module("fc_out", torch::nn::Linear(1024,                    latent_dim));
    }

    torch::Tensor forward(torch::Tensor z, torch::Tensor action_emb) {
        auto inp = torch::cat({z, action_emb}, -1);
        return fc_out(torch::relu(fc2(torch::relu(fc1(inp)))));
    }
};
TORCH_MODULE(DynamicsModel);

// ------------------------------------
// Reward Predictor: z -> reward
// ------------------------------------
struct RewardPredictorImpl : torch::nn::Module {
    torch::nn::Linear fc1{nullptr}, fc_out{nullptr};

    explicit RewardPredictorImpl(int latent_dim) {
        fc1    = register_module("fc1",    torch::nn::Linear(latent_dim, 512));
        fc_out = register_module("fc_out", torch::nn::Linear(512,        1));
    }

    torch::Tensor forward(torch::Tensor z) {
        return fc_out(torch::relu(fc1(z)));
    }
};
TORCH_MODULE(RewardPredictor);

// ------------------------------------
// TitanWorldModel — Unified Interface
// ------------------------------------
class TitanWorldModel {
private:
    WorldEncoder    encoder;
    DynamicsModel   dynamics;
    RewardPredictor reward_head;

    int     latent_dim;
    int     action_dim;
    float   novelty_threshold; // z-score threshold for anomaly detection

    std::vector<torch::Tensor> seen_states; // for novelty detection (rolling buffer)
    size_t novelty_buffer_size = 1000;

    std::unique_ptr<torch::optim::Adam> optimizer;

public:
    TitanWorldModel(int obs_dim    = 16384,
                    int latent_dim = 512,
                    int action_dim = 256,
                    float lr       = 1e-4f,
                    float novelty_thresh = 2.5f)
        : encoder(obs_dim,    latent_dim),
          dynamics(latent_dim, action_dim),
          reward_head(latent_dim),
          latent_dim(latent_dim),
          action_dim(action_dim),
          novelty_threshold(novelty_thresh) {

        // Collect all parameters for joint optimisation
        std::vector<torch::Tensor> params;
        for (auto& p : encoder->parameters())     params.push_back(p);
        for (auto& p : dynamics->parameters())    params.push_back(p);
        for (auto& p : reward_head->parameters()) params.push_back(p);

        optimizer = std::make_unique<torch::optim::Adam>(
            params, torch::optim::AdamOptions(lr));

        std::cout << "[WorldModel] Initialized."
                  << " Latent dim: " << latent_dim
                  << " | Action dim: " << action_dim << std::endl;
    }

    // Encode an observation into a latent world state
    WorldState encode(torch::Tensor obs) {
        encoder->eval();
        torch::NoGradGuard no_grad;

        auto [mu, log_var] = encoder->forward(obs);

        // Reparameterisation trick for sampling
        auto std    = (0.5f * log_var).exp();
        auto eps    = torch::randn_like(std);
        auto z      = mu + eps * std;

        float reward = reward_head->forward(z).item<float>();
        bool  novel  = is_novel_state(z);

        return {z, std, reward, novel};
    }

    // Predict the next world state given current state and action
    WorldState predict_next(const WorldState& current,
                            torch::Tensor action_embedding) {
        dynamics->eval();
        torch::NoGradGuard no_grad;

        auto z_next   = dynamics->forward(current.z, action_embedding);
        float reward  = reward_head->forward(z_next).item<float>();
        bool  novel   = is_novel_state(z_next);

        auto uncertainty = torch::ones_like(z_next) * 0.1f; // placeholder
        return {z_next, uncertainty, reward, novel};
    }

    // Simulate N steps into the future (for planning)
    std::vector<WorldState> simulate(const WorldState& start,
                                     std::vector<torch::Tensor>& action_sequence) {
        std::vector<WorldState> trajectory;
        WorldState cur = start;
        for (auto& action_emb : action_sequence) {
            cur = predict_next(cur, action_emb);
            trajectory.push_back(cur);
            if (cur.is_novel) {
                std::cout << "[WorldModel] Novel state encountered at step "
                          << trajectory.size() << " — high uncertainty." << std::endl;
            }
        }
        return trajectory;
    }

    // Train the world model on (obs, action, obs_next, reward) tuples
    float train_step(torch::Tensor obs,
                     torch::Tensor action_emb,
                     torch::Tensor obs_next,
                     float         reward_gt) {
        encoder->train(); dynamics->train(); reward_head->train();
        optimizer->zero_grad();

        auto [mu,     log_var]     = encoder->forward(obs);
        auto [mu_next,lv_next]     = encoder->forward(obs_next);

        // Reparameterisation
        auto z = mu + (0.5f * log_var).exp() * torch::randn_like(mu);

        // Dynamics prediction
        auto z_next_pred = dynamics->forward(z, action_emb);

        // Reward prediction
        auto reward_pred = reward_head->forward(z);

        // Losses
        auto dynamics_loss = torch::mse_loss(z_next_pred, mu_next.detach());
        auto reward_loss   = torch::mse_loss(reward_pred,
            torch::full_like(reward_pred, reward_gt));
        // KL divergence (VAE regularisation)
        auto kl_loss = -0.5f * (1 + log_var - mu.pow(2) - log_var.exp()).mean();

        auto total_loss = dynamics_loss + reward_loss + 0.001f * kl_loss;
        total_loss.backward();

        torch::nn::utils::clip_grad_norm_(encoder->parameters(),   1.0f);
        torch::nn::utils::clip_grad_norm_(dynamics->parameters(),   1.0f);
        torch::nn::utils::clip_grad_norm_(reward_head->parameters(),1.0f);

        optimizer->step();

        // Store state for novelty tracking
        if (seen_states.size() >= novelty_buffer_size) seen_states.erase(seen_states.begin());
        seen_states.push_back(mu.detach().clone());

        return total_loss.item<float>();
    }

private:
    bool is_novel_state(const torch::Tensor& z) {
        if (seen_states.empty()) return true;
        auto stacked = torch::stack(seen_states); // [N, D]
        auto mean    = stacked.mean(0);
        auto stddev  = stacked.std(0).clamp_min(1e-6f);
        auto z_score = ((z - mean) / stddev).abs().mean().item<float>();
        return z_score > novelty_threshold;
    }
};
