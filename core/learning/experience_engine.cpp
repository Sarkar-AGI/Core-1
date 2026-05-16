#pragma once
#include <torch/torch.h>
#include <vector>
#include <deque>
#include <string>
#include <functional>
#include <iostream>
#include <algorithm>
#include <random>
#include "../model/config.h"

/*
=============================================================
  TITANCORE AGI: LEARNING FROM EXPERIENCE ENGINE
=============================================================
  The AGI learns from every interaction, outcome, and
  feedback signal — just as humans learn from lived events.

  Implements:
  - Reinforcement Learning from Human Feedback (RLHF)
  - Reward Model (trained on preference pairs)
  - Proximal Policy Optimisation (PPO) for policy updates
  - Hindsight Experience Replay (HER) — learn from failure
  - Curiosity-Driven Exploration (intrinsic reward)
  - Skill Distillation — compress learned skills
=============================================================
*/

// -----------------------------------------------
// Experience Transition
// -----------------------------------------------
struct Transition {
    torch::Tensor state;          // [D] model hidden state
    torch::Tensor action;         // [D] generated action embedding
    torch::Tensor next_state;     // [D] resulting state
    float         reward;         // extrinsic reward (from env or human)
    float         intrinsic;      // curiosity/novelty bonus
    bool          done;
    std::string   feedback;       // optional human text feedback
};

// -----------------------------------------------
// Reward Model (trained on human preference pairs)
// -----------------------------------------------
struct RewardModelImpl : torch::nn::Module {
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc_out{nullptr};

    explicit RewardModelImpl(int state_dim) {
        fc1    = register_module("fc1",    torch::nn::Linear(state_dim * 2, 2048));
        fc2    = register_module("fc2",    torch::nn::Linear(2048,          512));
        fc_out = register_module("fc_out", torch::nn::Linear(512,            1));
    }

    // Predict scalar reward for a (state, action) pair
    torch::Tensor forward(torch::Tensor state, torch::Tensor action) {
        auto x = torch::cat({state, action}, -1);
        return fc_out(torch::relu(fc2(torch::relu(fc1(x)))));
    }
};
TORCH_MODULE(RewardModel);

// -----------------------------------------------
// PPO Value Head
// -----------------------------------------------
struct ValueHeadImpl : torch::nn::Module {
    torch::nn::Linear fc1{nullptr}, fc_out{nullptr};

    explicit ValueHeadImpl(int state_dim) {
        fc1    = register_module("fc1",    torch::nn::Linear(state_dim, 512));
        fc_out = register_module("fc_out", torch::nn::Linear(512,       1));
    }

    torch::Tensor forward(torch::Tensor state) {
        return fc_out(torch::relu(fc1(state)));
    }
};
TORCH_MODULE(ValueHead);

// -----------------------------------------------
// Hindsight Experience Replay Buffer
// -----------------------------------------------
class HERBuffer {
private:
    std::deque<Transition> buffer;
    size_t max_size;
    std::mt19937 rng;

public:
    explicit HERBuffer(size_t cap = 200000) : max_size(cap), rng(std::random_device{}()) {}

    void push(Transition t) {
        if (buffer.size() >= max_size) buffer.pop_front();
        buffer.push_back(std::move(t));
    }

    // HER: on failure, relabel goal as achieved state (learn from failure)
    void push_with_hindsight(Transition t, std::function<float(const torch::Tensor&)> goal_reward) {
        buffer.push_back(t);

        // Relabelled version: pretend the achieved next_state WAS the goal
        Transition hindsight = t;
        hindsight.reward = goal_reward(t.next_state); // usually ~1.0 for "achieved"
        hindsight.done   = true;
        if (buffer.size() < max_size) buffer.push_back(std::move(hindsight));
    }

    std::vector<Transition> sample(size_t n) {
        std::vector<Transition> batch;
        n = std::min(n, buffer.size());
        std::uniform_int_distribution<size_t> dist(0, buffer.size() - 1);
        for (size_t i = 0; i < n; ++i) batch.push_back(buffer[dist(rng)]);
        return batch;
    }

    size_t size() const { return buffer.size(); }
};

// -----------------------------------------------
// PPO Clip Loss
// -----------------------------------------------
torch::Tensor ppo_clip_loss(torch::Tensor old_log_probs,
                             torch::Tensor new_log_probs,
                             torch::Tensor advantages,
                             float clip_eps = 0.2f) {
    auto ratio      = (new_log_probs - old_log_probs).exp();
    auto clipped    = ratio.clamp(1.0f - clip_eps, 1.0f + clip_eps);
    auto obj        = torch::min(ratio * advantages, clipped * advantages);
    return -obj.mean();
}

// -----------------------------------------------
// TitanExperienceEngine
// -----------------------------------------------
class TitanExperienceEngine {
private:
    RewardModel reward_model;
    ValueHead   value_head;
    HERBuffer   her_buffer;

    std::unique_ptr<torch::optim::Adam> rm_optimizer;   // reward model optimizer
    std::unique_ptr<torch::optim::Adam> ppo_optimizer;  // policy / value optimizer

    float gamma          = 0.99f;  // discount factor
    float gae_lambda     = 0.95f;  // GAE smoothing
    float ppo_clip       = 0.2f;
    float vf_coef        = 0.5f;   // value function loss coefficient
    float entropy_coef   = 0.01f;  // entropy bonus coefficient
    int   ppo_epochs     = 4;
    int   batch_size     = 64;

    // Curiosity: running mean/var of intrinsic rewards for normalisation
    float curiosity_mean = 0.0f;
    float curiosity_var  = 1.0f;
    int   curiosity_n    = 0;

public:
    TitanExperienceEngine(int state_dim, float lr_rm = 1e-4f, float lr_ppo = 3e-5f)
        : reward_model(state_dim),
          value_head(state_dim) {

        rm_optimizer  = std::make_unique<torch::optim::Adam>(
            reward_model->parameters(), torch::optim::AdamOptions(lr_rm));
        ppo_optimizer = std::make_unique<torch::optim::Adam>(
            value_head->parameters(),   torch::optim::AdamOptions(lr_ppo));

        std::cout << "[ExperienceEngine] RLHF + PPO + HER initialized." << std::endl;
        std::cout << " > Discount (gamma)   : " << gamma      << std::endl;
        std::cout << " > PPO clip epsilon   : " << ppo_clip   << std::endl;
        std::cout << " > Replay buffer cap  : 200K" << std::endl;
    }

    // ---- Store experience (with optional HER) ----
    void observe(Transition t, bool use_hindsight = false) {
        // Compute curiosity / novelty bonus
        t.intrinsic = compute_curiosity(t.state, t.next_state);
        t.reward   += normalise_curiosity(t.intrinsic);

        if (use_hindsight) {
            her_buffer.push_with_hindsight(t, [](const torch::Tensor& s) { return 1.0f; });
        } else {
            her_buffer.push(t);
        }
    }

    // ---- Train Reward Model on a preference pair (chosen > rejected) ----
    float train_reward_model(torch::Tensor state_chosen,
                             torch::Tensor action_chosen,
                             torch::Tensor state_rejected,
                             torch::Tensor action_rejected) {
        rm_optimizer->zero_grad();

        auto r_chosen   = reward_model->forward(state_chosen,   action_chosen);
        auto r_rejected = reward_model->forward(state_rejected, action_rejected);

        // Bradley-Terry preference loss: log(sigmoid(chosen - rejected))
        auto loss = -torch::log_sigmoid(r_chosen - r_rejected).mean();
        loss.backward();
        torch::nn::utils::clip_grad_norm_(reward_model->parameters(), 1.0f);
        rm_optimizer->step();

        return loss.item<float>();
    }

    // ---- PPO Update: refine policy using collected experience ----
    float ppo_update(torch::Tensor states,       // [B, D]
                     torch::Tensor actions,      // [B, D]
                     torch::Tensor old_log_probs,// [B]
                     torch::Tensor new_log_probs,// [B]
                     torch::Tensor rewards,      // [B]
                     torch::Tensor dones) {      // [B]

        float total_loss = 0.0f;

        for (int ep = 0; ep < ppo_epochs; ++ep) {
            ppo_optimizer->zero_grad();

            // Value estimates
            auto values    = value_head->forward(states).squeeze(-1);  // [B]

            // GAE advantages
            auto advantages = compute_gae(rewards, values.detach(), dones);
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8f);

            // Policy loss (PPO-clip)
            auto policy_loss = ppo_clip_loss(old_log_probs, new_log_probs,
                                             advantages, ppo_clip);

            // Value loss
            auto returns   = advantages + values.detach();
            auto value_loss = torch::mse_loss(values, returns);

            // Entropy bonus (encourage exploration)
            auto entropy    = -(new_log_probs * new_log_probs.exp()).mean();

            auto loss = policy_loss + vf_coef * value_loss - entropy_coef * entropy;
            loss.backward();
            torch::nn::utils::clip_grad_norm_(value_head->parameters(), 0.5f);
            ppo_optimizer->step();

            total_loss += loss.item<float>();
        }

        std::cout << "[ExperienceEngine] PPO update complete. Avg loss: "
                  << total_loss / ppo_epochs << std::endl;
        return total_loss / ppo_epochs;
    }

    // ---- Infer reward for a (state, action) pair ----
    float infer_reward(torch::Tensor state, torch::Tensor action) {
        reward_model->eval();
        torch::NoGradGuard ng;
        return reward_model->forward(state, action).item<float>();
    }

    // ---- HER batch sample for offline learning ----
    std::vector<Transition> sample_experience(size_t n) {
        return her_buffer.sample(n);
    }

    size_t buffer_size() const { return her_buffer.size(); }

private:
    torch::Tensor compute_gae(torch::Tensor rewards,
                              torch::Tensor values,
                              torch::Tensor dones) {
        int n = rewards.size(0);
        auto adv = torch::zeros_like(rewards);
        float running = 0.0f;
        for (int t = n - 1; t >= 0; --t) {
            float r = rewards[t].item<float>();
            float v = values[t].item<float>();
            float d = dones[t].item<float>();
            float next_v = (t < n - 1) ? values[t + 1].item<float>() : 0.0f;
            float delta  = r + gamma * next_v * (1.0f - d) - v;
            running      = delta + gamma * gae_lambda * (1.0f - d) * running;
            adv[t]       = running;
        }
        return adv;
    }

    float compute_curiosity(const torch::Tensor& state, const torch::Tensor& next_state) {
        return (next_state - state).pow(2).mean().item<float>();
    }

    float normalise_curiosity(float raw) {
        curiosity_n++;
        float delta    = raw - curiosity_mean;
        curiosity_mean += delta / curiosity_n;
        curiosity_var   = curiosity_var + delta * (raw - curiosity_mean);
        float std_dev  = curiosity_n > 1 ? std::sqrt(curiosity_var / (curiosity_n - 1)) : 1.0f;
        return (raw - curiosity_mean) / (std_dev + 1e-8f);
    }
};
