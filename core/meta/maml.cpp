#pragma once
#include <torch/torch.h>
#include <vector>
#include <functional>
#include <iostream>
#include "../model/config.h"

/*
=====================================================
  TITANCORE AGI: META-LEARNING ENGINE (MAML)
=====================================================
  Model-Agnostic Meta-Learning — "Learn to Learn".
  Finds model initial weights that can be fine-tuned
  to any new task in just a few gradient steps.

  Implements:
  - MAML (Finn et al., 2017) — first-order variant
  - Reptile (Nichol et al.) — scalable alternative
  - Task sampling and inner-loop adaptation
  - Outer-loop meta-gradient accumulation
=====================================================
*/

struct MetaTask {
    std::string name;
    std::vector<std::pair<torch::Tensor, torch::Tensor>> support;  // few-shot examples
    std::vector<std::pair<torch::Tensor, torch::Tensor>> query;    // evaluation set
};

class TitanMAML {
private:
    float inner_lr;        // learning rate for task adaptation (inner loop)
    float outer_lr;        // meta-learning rate (outer loop)
    int   inner_steps;     // gradient steps per task in inner loop
    int   meta_batch;      // tasks per meta-update
    bool  first_order;     // use FOMAML (faster, slightly less accurate)

    std::unique_ptr<torch::optim::Adam> meta_optimizer;

public:
    TitanMAML(torch::nn::Module& model,
              float inner_lr    = 0.01f,
              float outer_lr    = 1e-3f,
              int   inner_steps = 5,
              int   meta_batch  = 8,
              bool  first_order = true)
        : inner_lr(inner_lr),
          outer_lr(outer_lr),
          inner_steps(inner_steps),
          meta_batch(meta_batch),
          first_order(first_order) {

        auto opts = torch::optim::AdamOptions(outer_lr).betas({0.9, 0.999});
        meta_optimizer = std::make_unique<torch::optim::Adam>(
            model.parameters(), opts);

        std::cout << "[MAML] Meta-Learning Engine initialized." << std::endl;
        std::cout << " > Inner LR    : " << inner_lr    << std::endl;
        std::cout << " > Outer LR    : " << outer_lr    << std::endl;
        std::cout << " > Inner steps : " << inner_steps << std::endl;
        std::cout << " > First-order : " << (first_order ? "yes (FOMAML)" : "no (MAML)") << std::endl;
    }

    // Full MAML meta-update over a batch of tasks
    float meta_update(torch::nn::Module& model,
                      std::vector<MetaTask>& tasks,
                      std::function<torch::Tensor(torch::nn::Module&,
                                                  torch::Tensor,
                                                  torch::Tensor)> loss_fn) {
        meta_optimizer->zero_grad();

        float total_meta_loss = 0.0f;

        for (int t = 0; t < std::min(meta_batch, (int)tasks.size()); ++t) {
            auto& task = tasks[t];

            // --- INNER LOOP: adapt a copy of the model to this task ---
            // Clone current parameters
            std::vector<torch::Tensor> adapted_params;
            for (auto& p : model.parameters())
                adapted_params.push_back(p.clone());

            // K gradient steps on the support set
            for (int s = 0; s < inner_steps; ++s) {
                torch::Tensor task_loss = torch::zeros({1});
                for (auto& [x, y] : task.support) {
                    task_loss = task_loss + loss_fn(model, x, y);
                }
                task_loss = task_loss / (float)task.support.size();

                // Compute gradients w.r.t. adapted_params
                auto grads = torch::autograd::grad(
                    {task_loss},
                    model.parameters(),
                    {},
                    /*retain_graph=*/true,
                    /*create_graph=*/!first_order
                );

                // Manual SGD update on adapted params
                for (size_t i = 0; i < adapted_params.size(); ++i) {
                    if (grads[i].defined())
                        adapted_params[i] = adapted_params[i] - inner_lr * grads[i];
                }
            }

            // --- OUTER LOOP: evaluate adapted model on the query set ---
            // Temporarily apply adapted params
            auto orig_params = snapshot(model);
            apply_params(model, adapted_params);

            torch::Tensor meta_loss = torch::zeros({1});
            for (auto& [x, y] : task.query)
                meta_loss = meta_loss + loss_fn(model, x, y);
            meta_loss = meta_loss / (float)task.query.size();

            meta_loss.backward();
            total_meta_loss += meta_loss.item<float>();

            // Restore original parameters
            apply_params(model, orig_params);
        }

        // Clip gradients and apply meta-update
        torch::nn::utils::clip_grad_norm_(model.parameters(), 1.0f);
        meta_optimizer->step();

        float avg_loss = total_meta_loss / meta_batch;
        std::cout << "[MAML] Meta-update complete. Avg query loss: "
                  << avg_loss << std::endl;
        return avg_loss;
    }

    // Reptile: simpler first-order meta-learning (scales better)
    float reptile_update(torch::nn::Module& model,
                         std::vector<MetaTask>& tasks,
                         std::function<torch::Tensor(torch::nn::Module&,
                                                     torch::Tensor,
                                                     torch::Tensor)> loss_fn,
                         float epsilon = 0.1f) {
        auto orig = snapshot(model);
        float total_loss = 0.0f;

        for (int t = 0; t < std::min(meta_batch, (int)tasks.size()); ++t) {
            auto& task = tasks[t];

            // Adapt model for inner_steps on support set
            auto inner_opt = torch::optim::SGD(model.parameters(),
                torch::optim::SGDOptions(inner_lr));

            for (int s = 0; s < inner_steps; ++s) {
                inner_opt.zero_grad();
                torch::Tensor loss = torch::zeros({1});
                for (auto& [x, y] : task.support)
                    loss = loss + loss_fn(model, x, y);
                loss = loss / (float)task.support.size();
                loss.backward();
                inner_opt.step();
                total_loss += loss.item<float>();
            }

            // Reptile: move original params toward adapted params
            auto adapted = snapshot(model);
            apply_params(model, orig); // restore first
            int idx = 0;
            for (auto& p : model.parameters()) {
                p.data().add_(epsilon * (adapted[idx] - orig[idx]));
                idx++;
            }
        }

        return total_loss / (meta_batch * inner_steps);
    }

    // Fast adaptation at inference time (few-shot fine-tune)
    void adapt(torch::nn::Module& model,
               MetaTask& task,
               std::function<torch::Tensor(torch::nn::Module&,
                                           torch::Tensor,
                                           torch::Tensor)> loss_fn) {
        auto opt = torch::optim::SGD(model.parameters(),
            torch::optim::SGDOptions(inner_lr));

        for (int s = 0; s < inner_steps; ++s) {
            opt.zero_grad();
            torch::Tensor loss = torch::zeros({1});
            for (auto& [x, y] : task.support)
                loss = loss + loss_fn(model, x, y);
            loss = loss / (float)task.support.size();
            loss.backward();
            opt.step();
        }
        std::cout << "[MAML] Fast adaptation complete for task: "
                  << task.name << std::endl;
    }

private:
    std::vector<torch::Tensor> snapshot(torch::nn::Module& model) {
        std::vector<torch::Tensor> snap;
        for (auto& p : model.parameters())
            snap.push_back(p.detach().clone());
        return snap;
    }

    void apply_params(torch::nn::Module& model,
                      const std::vector<torch::Tensor>& params) {
        int idx = 0;
        for (auto& p : model.parameters())
            p.data().copy_(params[idx++]);
    }
};
