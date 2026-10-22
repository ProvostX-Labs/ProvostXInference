cmake_minimum_required(VERSION 3.18)
project(ProvostXInference VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ── Options ──
option(PVX_BACKEND_METAL  "Build with Metal backend (macOS)" OFF)
option(PVX_BACKEND_CUDA   "Build with CUDA backend" OFF)
option(PVX_BACKEND_VULKAN "Build with Vulkan backend" OFF)
option(PVX_BACKEND_CPU   "Build CPU-only backend" ON)

# ── llama.cpp submodule ──
# Expects llama.cpp as a subdirectory at ${CMAKE_SOURCE_DIR}/llama.cpp
if(NOT EXISTS ${CMAKE_SOURCE_DIR}/llama.cpp/CMakeLists.txt)
    message(FATAL_ERROR "llama.cpp submodule not found. Run: git submodule update --init --recursive")
endif()

# Build llama.cpp as a static library (we link it into our shared lib)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_MTMD ON CACHE BOOL "" FORCE)

if(PVX_BACKEND_METAL)
    set(GGML_METAL ON CACHE BOOL "" FORCE)
endif()
if(PVX_BACKEND_CUDA)
    set(GGML_CUDA ON CACHE BOOL "" FORCE)
endif()
if(PVX_BACKEND_VULKAN)
    set(GGML_VULKAN ON CACHE BOOL "" FORCE)
endif()

# Spec-decode (INF BL-05) uses upstream common/speculative.cpp (vocab
# compatibility check); the vendor default skips common in subproject mode.
set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
add_subdirectory(llama.cpp)

# ── Our library ──
add_library(pvxinference SHARED
    src/pvxinference.cpp
    src/pvx_backend.cpp
    src/pvx_spec_engine.cpp
    src/pvx_ngram.cpp
    src/pvx_json_schema_grammar.cpp
    src/pvx_adapters.cpp
    src/pvx_tts.cpp
)

# Specialist module naming (BL-02 stage 1): the CI matrix builds ALL FIVE
# specialists — same sources/export table, one backend each — named after the
# per-OS specialist convention the managed resolver loads (names live in ONE
# place: csharp/Models/PvxEngineFileResolver.cs):
#   macOS : libpvx-metal.dylib       (metal specialist)
#   Linux : libpvx-cuda.so | libpvx-vulkan.so
#   Win   : pvx-cuda.dll    | pvx-vulkan.dll
# The target name stays pvxinference (logical DllImport name); the output is
# renamed per specialist so each runtimes/<rid>/native slice carries exactly
# the specialists for its OS.
if(PVX_BACKEND_METAL)
    set(PVX_SPECIALIST_NAME "libpvx-metal")
elseif(PVX_BACKEND_CUDA)
    set(PVX_SPECIALIST_NAME "pvx-cuda")
elseif(PVX_BACKEND_VULKAN)
    set(PVX_SPECIALIST_NAME "pvx-vulkan")
else()
    set(PVX_SPECIALIST_NAME "pvxinference")
endif()
set_target_properties(pvxinference PROPERTIES
    OUTPUT_NAME "${PVX_SPECIALIST_NAME}"
    CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON
)

target_include_directories(pvxinference
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/llama.cpp/src
        ${CMAKE_SOURCE_DIR}/llama.cpp/include
        ${CMAKE_SOURCE_DIR}/llama.cpp/ggml/include
        ${CMAKE_SOURCE_DIR}/llama.cpp/common
        ${CMAKE_SOURCE_DIR}/llama.cpp/tools/mtmd
)

target_link_libraries(pvxinference
    PRIVATE
        llama
        ggml
        ggml-base
        mtmd
        llama-common
        llama-common-base
)

# CUDA driver-API link (CI fix 2026-10-22): ggml-cuda references driver API
# entry points (cuGetErrorString et al.), which live in libcuda.so.1 — the
# GPU *driver*, not the toolkit. CI runners have no GPU driver, so the
# specialist .so failed to link its consumers with undefined references.
# Link the toolkit's stub libcuda (Linux only; Windows ships libcuda in
# System32-side CUDA dirs and links fine — win-x64-cuda was already green).
if(PVX_BACKEND_CUDA AND NOT WIN32 AND NOT APPLE)
    find_package(CUDAToolkit REQUIRED)
    find_library(PVX_CUDA_DRIVER_STUB libcuda.so
        PATHS "${CUDAToolkit_LIBRARY_DIR}/stubs" "/usr/local/cuda/lib64/stubs"
        NO_DEFAULT_PATH)
    if(PVX_CUDA_DRIVER_STUB)
        target_link_libraries(pvxinference PRIVATE "${PVX_CUDA_DRIVER_STUB}")
    else()
        message(WARNING "PVX CUDA: driver stub libcuda.so not found — link may fail on driverless runners")
    endif()
endif()

target_compile_definitions(pvxinference PRIVATE PVXINFERENCE_EXPORTS)

# Windows DLL export parity (2026-10-20, Windows ARM64 verify): the header is a
# plain extern "C" surface with no __declspec(dllexport) and there are C++ internal
# seams the native test executables exercise — on macOS/Linux the shared library
# exports all of them by default. CMake's auto-generated export table restores
# that behavior for the Windows DLL (no import-lib link failures for the tests,
# no per-symbol maintenance).
if(WIN32)
    set_target_properties(pvxinference PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

# ── Tests ──
enable_testing()

add_executable(pvxinference_tests
    tests/test_basic.cpp
)

add_executable(pvxinference_ngram_tests
    tests/test_ngram.cpp
)

add_executable(pvxinference_stress_tests
    tests/test_stress.cpp
)

target_link_libraries(pvxinference_stress_tests
    PRIVATE
        pvxinference
)

target_include_directories(pvxinference_stress_tests
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(pvxinference_tests
    PRIVATE
        pvxinference
)

# Windows test links (2026-10-20, Windows ARM64 verify): the shared library does
# not re-export transitive static-lib symbols on Windows (import libs carry only
# the DLL's own export table) — test binaries that call llama/ggml symbols
# directly must link those static libs themselves. No-op on macOS/Linux, where
# the default-visibility dylib/so re-exports the surface.
if(WIN32)
    target_link_libraries(pvxinference_tests PRIVATE llama ggml ggml-base)
    target_link_libraries(pvxinference_ngram_tests PRIVATE llama ggml ggml-base)
endif()

target_include_directories(pvxinference_tests
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/llama.cpp/ggml/include
        ${CMAKE_BINARY_DIR}/ggml/include
)

add_test(NAME pvx_tests COMMAND pvxinference_tests)
set_tests_properties(pvx_tests PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_test(NAME pvx_stress_tests COMMAND pvxinference_stress_tests)
# BL-45: stress needs the repo-root ./models fixtures (default model paths are
# cwd-relative) — without this it silently ran from the build dir and failed
# to find the GGUFs under ctest (the binary ran fine when invoked directly).
set_tests_properties(pvx_stress_tests PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# Pure n-gram + mirror-ring unit tests (ABI v6 wave 1) — no model files needed.
target_link_libraries(pvxinference_ngram_tests
    PRIVATE
        pvxinference
)

target_include_directories(pvxinference_ngram_tests
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/llama.cpp/include
        ${CMAKE_SOURCE_DIR}/llama.cpp/ggml/include
        ${CMAKE_SOURCE_DIR}/llama.cpp/tools/mtmd
)

add_test(NAME pvx_ngram_tests COMMAND pvxinference_ngram_tests)
