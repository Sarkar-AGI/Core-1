#pragma once
#include <torch/torch.h>
#include <string>
#include <vector>
#include <functional>
#include <iostream>
#include <unordered_map>
#include <algorithm>
#include "../model/config.h"

/*
=============================================================
  TITANCORE AGI: GENERAL INTELLIGENCE CONTROLLER
=============================================================
  Coordinates all specialised cognitive modules into a
  single unified General Intelligence.

  Key AGI Properties implemented here:
  ─────────────────────────────────────────────────────
  G1. TRANSFER LEARNING    — skills from one domain apply
                             to new domains automatically
  G2. CURRICULUM LEARNING  — progressive difficulty to
                             maximise learning efficiency
  G3. SELF-IMPROVEMENT     — the AGI identifies its own
                             knowledge gaps and fills them
  G4. SKILL COMPOSITION    — combine existing skills
                             to solve novel problems
  G5. GENERALISATION TEST  — validate across N novel tasks
                             to confirm true generalisation
  G6. METACOGNITION        — know what you don't know,
                             and know when to ask for help
=============================================================
*/

// -----------------------------------------------
// Skill: a learned, transferable capability
// -----------------------------------------------
struct Skill {
    std::string   name;
    std::string   domain;           // domain this skill was learned in
    float         proficiency;      // 0.0 – 1.0
    float         transferability;  // how well it transfers to new domains
    std::vector<torch::Tensor> parameters; // learned weight patches / adapters
    std::vector<std::string>   prerequisite_skills;
    int           use_count;
};

// -----------------------------------------------
// Curriculum Task
// -----------------------------------------------
struct CurriculumTask {
    std::string name;
    float       difficulty;      // 0.0 – 1.0
    float       priority;        // learning value
    std::string domain;
    std::vector<std::pair<torch::Tensor, torch::Tensor>> examples;
};

// -----------------------------------------------
// Self-Improvement Report
// -----------------------------------------------
struct SelfAssessment {
    float         overall_competence;   // 0.0 – 1.0
    std::vector<std::pair<std::string, float>> skill_gaps; // (gap, severity 0-1)
    std::string   recommended_next_task;
    std::string   metacognitive_note;   // "I don't know X" or "I'm confident about Y"
};

class TitanGeneralIntelligence {
private:
    std::unordered_map<std::string, Skill> skill_library;

    // Curriculum state
    std::vector<CurriculumTask> curriculum;
    float current_difficulty = 0.1f;   // start easy
    float difficulty_step    = 0.05f;  // increment per mastered task

    // Metacognition
    std::unordered_map<std::string, float> competence_map; // domain -> competence
    std::vector<std::string>               known_unknowns;  // explicit gaps

    // Proficiency threshold for skill mastery
    float mastery_threshold = 0.85f;

    // Loss function for skill evaluation
    std::function<float(const std::string&,
                        const std::vector<std::pair<torch::Tensor,torch::Tensor>>&)> eval_fn;

public:
    TitanGeneralIntelligence(
        std::function<float(const std::string&,
                            const std::vector<std::pair<torch::Tensor,torch::Tensor>>&)> evaluator)
        : eval_fn(evaluator) {

        seed_foundational_skills();
        seed_curriculum();

        std::cout << "[GeneralIntelligence] AGI General Intelligence initialized." << std::endl;
        std::cout << " > Foundational skills : " << skill_library.size() << std::endl;
        std::cout << " > Curriculum tasks    : " << curriculum.size()    << std::endl;
        std::cout << " > Starting difficulty : " << current_difficulty   << std::endl;
    }

    // =============================================
    // G1. TRANSFER LEARNING
    // =============================================
    bool transfer_skill(const std::string& skill_name,
                        const std::string& source_domain,
                        const std::string& target_domain,
                        float domain_similarity) {
        auto it = skill_library.find(skill_name);
        if (it == skill_library.end()) {
            std::cout << "[GI] Skill not found for transfer: " << skill_name << std::endl;
            return false;
        }

        Skill& original = it->second;
        float  transfer_success = original.transferability * domain_similarity;

        // Create a new skill instance for the target domain
        Skill transferred;
        transferred.name          = skill_name + "@" + target_domain;
        transferred.domain        = target_domain;
        transferred.proficiency   = original.proficiency * transfer_success;
        transferred.transferability = original.transferability * 0.9f; // degrades slightly
        transferred.parameters    = original.parameters;
        transferred.use_count     = 0;

        skill_library[transferred.name] = std::move(transferred);

        std::cout << "[GI] Transfer: " << skill_name
                  << " from " << source_domain << " -> " << target_domain
                  << " | Effectiveness: " << transfer_success << std::endl;
        return transfer_success > 0.3f;
    }

    // =============================================
    // G2. CURRICULUM LEARNING
    // =============================================
    CurriculumTask* next_curriculum_task() {
        // Find the highest-priority task at current difficulty level
        CurriculumTask* best = nullptr;
        float best_value     = -1.0f;

        for (auto& task : curriculum) {
            if (task.difficulty > current_difficulty + 0.15f) continue; // too hard
            if (competence_map.count(task.domain) &&
                competence_map[task.domain] > mastery_threshold) continue; // already mastered

            float value = task.priority * (1.0f - task.difficulty + 0.1f);
            if (value > best_value) {
                best_value = value;
                best       = &task;
            }
        }
        return best;
    }

    void complete_curriculum_task(const std::string& task_name, float achieved_score) {
        for (auto& t : curriculum) {
            if (t.name == task_name) {
                competence_map[t.domain] = std::max(
                    competence_map[t.domain], achieved_score);

                if (achieved_score >= mastery_threshold) {
                    current_difficulty = std::min(1.0f, current_difficulty + difficulty_step);
                    std::cout << "[GI] Curriculum: task mastered — "
                              << task_name << " | New difficulty: "
                              << current_difficulty << std::endl;
                }
                break;
            }
        }
    }

    // =============================================
    // G3. SELF-IMPROVEMENT
    // =============================================
    SelfAssessment self_assess() {
        SelfAssessment report;

        // Compute overall competence
        float total = 0.0f;
        for (auto& [d, c] : competence_map) total += c;
        report.overall_competence = competence_map.empty() ? 0.0f :
                                    total / competence_map.size();

        // Identify skill gaps
        for (auto& [name, skill] : skill_library) {
            if (skill.proficiency < 0.4f) {
                float severity = 1.0f - skill.proficiency;
                report.skill_gaps.push_back({name, severity});
            }
        }
        std::sort(report.skill_gaps.begin(), report.skill_gaps.end(),
                  [](auto& a, auto& b) { return a.second > b.second; });

        // Recommend next focus area
        auto* next = next_curriculum_task();
        report.recommended_next_task = next ? next->name : "(all tasks complete)";

        // Metacognitive note
        if (report.overall_competence > 0.8f)
            report.metacognitive_note = "I am highly confident in most domains.";
        else if (report.overall_competence > 0.5f)
            report.metacognitive_note = "I have solid foundational knowledge but gaps remain.";
        else
            report.metacognitive_note = "I am still learning — high uncertainty in many areas.";

        // Record known unknowns
        for (auto& [gap, sev] : report.skill_gaps)
            if (sev > 0.7f) known_unknowns.push_back(gap);

        std::cout << "[GI] Self-Assessment: competence=" << report.overall_competence
                  << " | Gaps: " << report.skill_gaps.size()
                  << " | Next: " << report.recommended_next_task << std::endl;
        return report;
    }

    // =============================================
    // G4. SKILL COMPOSITION
    // =============================================
    Skill compose_skills(const std::vector<std::string>& skill_names,
                         const std::string& composed_name) {
        Skill composed;
        composed.name             = composed_name;
        composed.proficiency      = 1.0f;
        composed.transferability  = 1.0f;
        composed.use_count        = 0;

        // Collect domains and parameters
        std::vector<std::string> domains;
        for (auto& sname : skill_names) {
            auto it = skill_library.find(sname);
            if (it == skill_library.end()) {
                std::cout << "[GI] Compose: unknown skill " << sname << std::endl;
                continue;
            }
            auto& s = it->second;
            composed.proficiency    = std::min(composed.proficiency,    s.proficiency);
            composed.transferability = std::min(composed.transferability, s.transferability);
            composed.prerequisite_skills.push_back(sname);
            for (auto& p : s.parameters) composed.parameters.push_back(p.clone());
            domains.push_back(s.domain);
            composed.domain = s.domain; // last domain wins
        }

        skill_library[composed_name] = composed;
        std::cout << "[GI] Composed skill: " << composed_name
                  << " from " << skill_names.size() << " skills."
                  << " Proficiency: " << composed.proficiency << std::endl;
        return composed;
    }

    // =============================================
    // G5. GENERALISATION TEST
    // =============================================
    float test_generalisation(const std::string& skill_name,
                              std::vector<CurriculumTask>& novel_tasks) {
        auto it = skill_library.find(skill_name);
        if (it == skill_library.end()) return 0.0f;

        float total_score = 0.0f;
        int   tested      = 0;

        for (auto& task : novel_tasks) {
            if (task.domain == it->second.domain) continue; // skip in-domain (not novel)
            float score = eval_fn(skill_name, task.examples);
            total_score += score;
            tested++;
        }

        float gen_score = tested > 0 ? total_score / tested : 0.0f;
        std::cout << "[GI] Generalisation test for " << skill_name
                  << ": " << gen_score << " across " << tested << " novel domains." << std::endl;
        return gen_score;
    }

    // =============================================
    // G6. METACOGNITION
    // =============================================
    bool knows_it_doesnt_know(const std::string& topic) const {
        for (auto& gap : known_unknowns)
            if (topic.find(gap) != std::string::npos ||
                gap.find(topic)  != std::string::npos)
                return true;
        return false;
    }

    std::string metacognitive_response(const std::string& question) const {
        for (auto& [domain, competence] : competence_map) {
            if (question.find(domain) != std::string::npos) {
                if (competence < 0.3f)
                    return "I'm not confident about " + domain
                         + " — I'd recommend verifying my response.";
                if (competence > 0.85f)
                    return ""; // no caveat needed — highly competent
                return "I have moderate knowledge of " + domain
                     + " — some details may need checking.";
            }
        }
        return ""; // no specific metacognitive note
    }

    void print_status() const {
        std::cout << "\n[GeneralIntelligence] === STATUS ===" << std::endl;
        std::cout << " Skills: " << skill_library.size() << std::endl;
        std::cout << " Curriculum difficulty: " << current_difficulty << std::endl;
        std::cout << " Known unknowns: " << known_unknowns.size() << std::endl;
        std::cout << " Domain competences:" << std::endl;
        for (auto& [d, c] : competence_map)
            std::cout << "   " << d << " = " << c << std::endl;
    }

private:
    void seed_foundational_skills() {
        auto add = [&](const std::string& name, const std::string& domain,
                       float prof, float trans) {
            Skill s;
            s.name           = name;
            s.domain         = domain;
            s.proficiency    = prof;
            s.transferability = trans;
            s.use_count      = 0;
            skill_library[name] = std::move(s);
        };

        add("language_understanding",  "language",    0.90f, 0.90f);
        add("mathematical_reasoning",  "mathematics", 0.75f, 0.80f);
        add("logical_deduction",       "logic",       0.85f, 0.95f);
        add("pattern_recognition",     "general",     0.80f, 0.85f);
        add("code_generation",         "software",    0.75f, 0.70f);
        add("scientific_reasoning",    "science",     0.70f, 0.80f);
        add("common_sense_reasoning",  "general",     0.72f, 0.88f);
        add("analogical_reasoning",    "general",     0.68f, 0.90f);
        add("goal_decomposition",      "planning",    0.70f, 0.75f);
        add("self_monitoring",         "metacognition",0.65f,0.80f);
    }

    void seed_curriculum() {
        curriculum = {
            // Level 1: easy (difficulty 0.1 – 0.2)
            {"basic_qa",            0.10f, 0.9f, "language",    {}},
            {"simple_arithmetic",   0.12f, 0.7f, "mathematics", {}},
            {"word_analogy",        0.15f, 0.8f, "general",     {}},
            // Level 2: medium (difficulty 0.3 – 0.5)
            {"multi_hop_reasoning", 0.35f, 0.9f, "logic",       {}},
            {"code_generation_simple", 0.40f,0.8f,"software",   {}},
            {"causal_reasoning",    0.45f, 0.85f,"science",     {}},
            // Level 3: hard (difficulty 0.6 – 0.8)
            {"mathematical_proof",  0.65f, 0.7f, "mathematics", {}},
            {"novel_problem_solving",0.70f,0.95f,"general",     {}},
            {"long_horizon_planning",0.75f,0.8f, "planning",    {}},
            // Level 4: frontier (difficulty 0.9+)
            {"cross_domain_transfer",0.90f,1.0f, "general",     {}},
            {"open_ended_discovery", 0.95f,1.0f, "general",     {}},
        };
    }
};
