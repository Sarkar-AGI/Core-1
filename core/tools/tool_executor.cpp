#pragma once
#include <string>
#include <vector>
#include <functional>
#include <unordered_map>
#include <iostream>
#include <stdexcept>

/*
=====================================================
  TITANCORE AGI: TOOL USE / FUNCTION CALLING
=====================================================
  Allows the AGI to call external tools and APIs.
  Tools are registered with a schema and a handler.
  The model outputs a structured tool call; the
  executor dispatches it and returns the result.

  Supported tool types:
  - Code interpreter (sandboxed)
  - Web search
  - Calculator
  - File I/O
  - Database query
  - External API call
  - Custom user-defined tools
=====================================================
*/

struct ToolParam {
    std::string name;
    std::string type;         // "string", "int", "float", "bool"
    std::string description;
    bool        required;
};

struct ToolSchema {
    std::string             name;
    std::string             description;
    std::vector<ToolParam>  parameters;
};

struct ToolCall {
    std::string name;
    std::unordered_map<std::string, std::string> args; // arg_name -> value (as string)
};

struct ToolResult {
    bool        success;
    std::string output;
    std::string error;
    float       latency_ms;
};

using ToolHandler = std::function<ToolResult(const ToolCall&)>;

class TitanToolExecutor {
private:
    std::unordered_map<std::string, ToolSchema>  schemas;
    std::unordered_map<std::string, ToolHandler> handlers;
    int max_retries = 2;

public:
    TitanToolExecutor() {
        // Register built-in tools
        register_builtin_tools();
        std::cout << "[ToolExecutor] Initialized with "
                  << schemas.size() << " built-in tools." << std::endl;
    }

    // Register a custom tool
    void register_tool(const ToolSchema& schema, ToolHandler handler) {
        schemas[schema.name]  = schema;
        handlers[schema.name] = handler;
        std::cout << "[ToolExecutor] Registered tool: " << schema.name << std::endl;
    }

    // Execute a tool call from the model
    ToolResult execute(const ToolCall& call) {
        auto sit = schemas.find(call.name);
        if (sit == schemas.end())
            return {false, "", "Unknown tool: " + call.name, 0.0f};

        // Validate required parameters
        for (auto& param : sit->second.parameters) {
            if (param.required && call.args.find(param.name) == call.args.end())
                return {false, "", "Missing required param: " + param.name, 0.0f};
        }

        auto hit = handlers.find(call.name);
        if (hit == handlers.end())
            return {false, "", "No handler for tool: " + call.name, 0.0f};

        // Execute with retry on transient errors
        for (int attempt = 0; attempt <= max_retries; ++attempt) {
            try {
                auto start  = std::chrono::steady_clock::now();
                auto result = hit->second(call);
                auto end    = std::chrono::steady_clock::now();
                result.latency_ms = std::chrono::duration<float, std::milli>(end - start).count();

                std::cout << "[ToolExecutor] " << call.name
                          << " -> " << (result.success ? "OK" : "FAIL")
                          << " (" << result.latency_ms << "ms)" << std::endl;
                return result;
            } catch (std::exception& e) {
                if (attempt == max_retries)
                    return {false, "", std::string("Exception: ") + e.what(), 0.0f};
            }
        }
        return {false, "", "Exhausted retries", 0.0f};
    }

    // List all available tools (for model's system prompt)
    std::string tool_manifest() const {
        std::string manifest = "Available tools:\n";
        for (auto& [name, schema] : schemas) {
            manifest += "  - " + name + ": " + schema.description + "\n";
            for (auto& p : schema.parameters)
                manifest += "      " + p.name + " (" + p.type + ")"
                         + (p.required ? " [required]" : " [optional]")
                         + " — " + p.description + "\n";
        }
        return manifest;
    }

    // Parse a model-generated tool call from JSON-like string
    static ToolCall parse_call(const std::string& raw_call) {
        // Simplified parser — production should use a full JSON parser
        ToolCall call;
        // Expected format: {"tool":"<name>","args":{"k":"v",...}}
        auto tool_pos = raw_call.find("\"tool\":\"");
        if (tool_pos != std::string::npos) {
            size_t start = tool_pos + 8;
            size_t end   = raw_call.find('"', start);
            call.name    = raw_call.substr(start, end - start);
        }
        return call;
    }

private:
    void register_builtin_tools() {
        // Calculator
        register_tool(
            {"calculator", "Evaluate a mathematical expression",
             {{"expression", "string", "Math expression to evaluate", true}}},
            [](const ToolCall& c) -> ToolResult {
                std::string expr = c.args.at("expression");
                // Stub: a real implementation would use a safe math evaluator
                return {true, "Result of [" + expr + "]: (calculated)", "", 0.0f};
            }
        );

        // Web search
        register_tool(
            {"web_search", "Search the web for current information",
             {{"query", "string", "Search query", true},
              {"num_results", "int", "Number of results to return", false}}},
            [](const ToolCall& c) -> ToolResult {
                return {true, "Search results for: " + c.args.at("query")
                        + " (live results via search API)", "", 0.0f};
            }
        );

        // Code interpreter
        register_tool(
            {"code_interpreter", "Execute Python code in a sandboxed environment",
             {{"code", "string", "Python code to execute", true}}},
            [](const ToolCall& c) -> ToolResult {
                // Stub: real implementation uses a secure subprocess sandbox
                return {true, "Code executed. Output: (sandbox result)", "", 0.0f};
            }
        );

        // Read file
        register_tool(
            {"read_file", "Read the contents of a file",
             {{"path", "string", "Absolute file path", true}}},
            [](const ToolCall& c) -> ToolResult {
                std::string path = c.args.at("path");
                std::ifstream f(path);
                if (!f) return {false, "", "File not found: " + path, 0.0f};
                std::string content((std::istreambuf_iterator<char>(f)),
                                     std::istreambuf_iterator<char>());
                return {true, content, "", 0.0f};
            }
        );

        // Database query
        register_tool(
            {"db_query", "Run a read-only SQL query against the knowledge database",
             {{"sql", "string", "SQL SELECT statement", true}}},
            [](const ToolCall& c) -> ToolResult {
                return {true, "Query result for: " + c.args.at("sql")
                        + " (DB response)", "", 0.0f};
            }
        );
    }
};

// Include missing headers for file tool
#include <fstream>
#include <chrono>
