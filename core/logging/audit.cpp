#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <mutex>
#include "../safety/moderation.cpp"

/*
=====================================================
  TITANCORE: SECURITY AUDIT & LOGGING ENGINE
=====================================================
  - Thread-safe CSV Logging
  - Structured Audit Trail for Safety Violations
  - Simple log() for training-loop step reporting
=====================================================
*/

class TitanAuditLogger {
private:
    std::string log_file;
    std::mutex  log_mutex;

    std::string get_timestamp() {
        auto now       = std::chrono::system_clock::now();
        auto in_time_t = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d %X");
        return ss.str();
    }

public:
    explicit TitanAuditLogger(const std::string& filename) : log_file(filename) {
        std::ofstream file(log_file, std::ios::app);
        file << "Timestamp,Step,Rank,Event\n";
        std::cout << "[TitanCore] Audit Logger Initialized: " << log_file << std::endl;
    }

    // FIXED: log() called from main training loop (step, rank, message)
    void log(int step, int rank, const std::string& event) {
        std::lock_guard<std::mutex> lock(log_mutex);
        std::ofstream file(log_file, std::ios::app);
        file << get_timestamp() << ","
             << step << ","
             << rank << ","
             << event << "\n";
    }

    // log_violation() for safety-engine events with full ModerationResult
    void log_violation(int64_t session_id,
                       const ModerationResult& result,
                       const std::string& text) {
        std::lock_guard<std::mutex> lock(log_mutex);

        std::string vtype;
        switch (result.label) {
            case SafetyLabel::VIOLENCE:         vtype = "VIOLENCE";  break;
            case SafetyLabel::NSFW:             vtype = "NSFW";      break;
            case SafetyLabel::HATE:             vtype = "HATE";      break;
            case SafetyLabel::MALWARE:          vtype = "MALWARE";   break;
            case SafetyLabel::PROMPT_INJECTION: vtype = "INJECTION"; break;
            case SafetyLabel::JAILBREAK:        vtype = "JAILBREAK"; break;
            default:                            vtype = "UNKNOWN";   break;
        }

        std::string sanitized = text;
        std::replace(sanitized.begin(), sanitized.end(), ',',  ' ');
        std::replace(sanitized.begin(), sanitized.end(), '\n', ' ');

        std::ofstream file(log_file, std::ios::app);
        file << get_timestamp() << ","
             << session_id << ","
             << vtype << ","
             << result.confidence << ","
             << sanitized << "\n";

        std::cout << "[Audit] Violation Logged: " << vtype
                  << " | Session: " << session_id << std::endl;
    }
};
