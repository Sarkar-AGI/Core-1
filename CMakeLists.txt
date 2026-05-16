cmake_minimum_required(VERSION 3.20)
project(TitanCore LANGUAGES CXX CUDA)

# -----------------------------------------------
# 1. Standards & CUDA Architectures
# -----------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CUDA_ARCHITECTURES "80;86;90")   # A100 / H100

# -----------------------------------------------
# 2. Dependencies
# -----------------------------------------------
find_package(Torch       REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(MPI         REQUIRED)

find_library(NCCL_LIBRARY nccl REQUIRED HINTS
    /usr/local/lib /usr/lib ENV NCCL_HOME)

message(STATUS "Found LibTorch      : ${Torch_DIR}")
message(STATUS "CUDA Toolkit version: ${CUDAToolkit_VERSION}")
message(STATUS "Found NCCL          : ${NCCL_LIBRARY}")

# -----------------------------------------------
# 3. C++ Sources
# -----------------------------------------------
set(CPP_SOURCES
    main.cpp

    # --- Core Model ---
    core/model/gpt.cpp
    core/model/block.cpp
    core/model/embedding.cpp
    core/model/kv_cache.cpp
    core/model/sampler.cpp

    # --- Distributed ---
    core/distributed/nccl.cpp
    core/distributed/fsdp.cpp
    core/distributed/mpi.cpp

    # --- Optimizer ---
    core/optimizer/zero.cpp

    # --- Data ---
    core/dataloader/dataset.cpp

    # --- Safety ---
    core/safety/moderation.cpp
    core/safety/jailbreak.cpp
    core/safety/rate_limit.cpp

    # --- Logging ---
    core/logging/audit.cpp

    # --- Runtime ---
    core/runtime/engine.cpp
    core/runtime/memory.cpp
    core/runtime/scheduler.cpp

    # --- Tokenizer ---
    core/tokenizer/sentencepiece.cpp

    # --- API ---
    core/api/server.cpp
    core/api/routes.cpp
    core/api/auth.cpp

    # --- Quantization ---
    core/quant/int4.cpp
    core/quant/int8.cpp
    core/quant/fp8.cpp

    # --- Vision ---
    core/vision/vit.cpp
    core/vision/transformer/clip.cpp

    # --- Audio ---
    core/audio/whisper.cpp
    core/audio/mel.cpp

    # --- Retrieval ---
    core/retrieval/embedder.cpp
    core/retrieval/loader.cpp

    # --- Monitoring ---
    core/monitoring/metrics.cpp
    core/monitoring/prometheus.cpp

    # --- Built-in Tools ---
    core/tools/benchmark.cpp
    core/tools/profiler.cpp

    # ============================================
    # AGI SUBSYSTEMS — COGNITIVE ARCHITECTURE
    # ============================================

    # Learning
    core/learning/online_learning.cpp      # Online GD + EWC + Replay + EMA
    core/learning/experience_engine.cpp    # RLHF + PPO + HER + Curiosity

    # Memory
    core/memory/episodic.cpp               # Past episode store + retrieval
    core/memory/semantic.cpp               # Long-term knowledge graph
    core/memory/working.cpp                # Active context window

    # Reasoning
    core/reasoning/chain_of_thought.cpp    # CoT / Self-Consistency / ToT / Reflection
    core/reasoning/planner.cpp             # MCTS goal-directed planner
    core/reasoning/common_sense.cpp        # Common sense + physical intuition + ToM
    core/reasoning/analogical.cpp          # Deductive / Inductive / Abductive / Analogical

    # Planning
    core/planning/hierarchical_planner.cpp # HTN hierarchical task planner

    # Meta-Learning
    core/meta/maml.cpp                     # MAML / FOMAML / Reptile

    # World Model
    core/world_model/world_model.cpp       # VAE encoder + dynamics + reward + novelty

    # Tool Use
    core/tools/tool_executor.cpp           # Function calling + 5 built-in tools

    # General Intelligence
    core/intelligence/general_intelligence.cpp  # Transfer / Curriculum / Self-Improvement
                                                # Skill Composition / Generalisation / Metacognition

    # AGI Core Controller
    core/agi/agi_core.cpp                  # Unified cognitive loop
)

set(CUDA_SOURCES
    core/model/attention.cu
)

# -----------------------------------------------
# 4. Executable
# -----------------------------------------------
add_executable(titancore ${CPP_SOURCES} ${CUDA_SOURCES})

# -----------------------------------------------
# 5. Include Directories
# -----------------------------------------------
target_include_directories(titancore PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/core/model          # config.h visible everywhere
    ${TORCH_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${MPI_CXX_INCLUDE_DIRS}
)

# -----------------------------------------------
# 6. Link Libraries
# -----------------------------------------------
target_link_libraries(titancore PRIVATE
    ${TORCH_LIBRARIES}
    CUDA::cudart
    CUDA::cuda_driver
    ${MPI_CXX_LIBRARIES}
    ${NCCL_LIBRARY}
)

# -----------------------------------------------
# 7. Compile Properties
# -----------------------------------------------
set_target_properties(titancore PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE  ON
)

target_compile_options(titancore PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    $<$<COMPILE_LANGUAGE:CUDA>:--ftz=true>
    $<$<COMPILE_LANGUAGE:CXX>:-Wall>
    $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
)

target_compile_definitions(titancore PRIVATE
    _GLIBCXX_USE_CXX11_ABI=1
)
