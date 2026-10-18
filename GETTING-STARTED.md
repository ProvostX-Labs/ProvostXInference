# ProvostXInference — Getting Started (LLM/AI-Agent Friendly)

**Summary:** Build the native C++ engine + C# bindings from a fresh clone and run all test suites. Exact commands, no fluff.

## Prerequisites
- .NET 8.0 SDK (`dotnet --version` → 8.x)
- Git (with submodule support)
- CMake ≥ 3.20 + a C++ toolchain (clang/gcc/MSVC)
- Backend SDK (optional, defaults exist): Metal is auto on macOS; CUDA or Vulkan SDKs for those backends
- No API keys needed — everything is local

## Fresh Clone → Native Build
```bash
git clone --recursive https://github.com/ProvostX-Labs/ProvostXInference.git
cd ProvostXInference
git submodule update --init --recursive   # pulls llama.cpp

# macOS (Metal):
cmake -B build-metal -DPVX_BACKEND_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-metal -j
```
Other platforms: `PVX_BACKEND_CUDA=ON` (Linux/Windows), `PVX_BACKEND_VULKAN=ON` (Linux/Windows), or omit for CPU fallback.

## Run Native (C++) Tests
```bash
cd build-metal
DYLD_LIBRARY_PATH=. ./pvxinference_tests         # basic suite
DYLD_LIBRARY_PATH=. ./pvxinference_stress_tests  # stress suite
# Linux: use LD_LIBRARY_PATH instead of DYLD_LIBRARY_PATH
```

## Build C# Bindings
```bash
cd csharp
dotnet build
```
- The bindings ship a packaged native library in `csharp/runtimes/<rid>/native/`; on first P/Invoke, `PvxNative` loads the dylib from `runtimes/` next to the assembly and fails fast with a descriptive `DllNotFoundException` on ABI mismatch (`pvx_abi_version()` handshake).

## Run C# Tests (51 xUnit tests)
```bash
cd csharp
DYLD_LIBRARY_PATH=../build-metal dotnet test Tests/   # macOS
# Linux: LD_LIBRARY_PATH=../build-metal dotnet test Tests/
```
- Without the env var, the packaged `runtimes/` copy is used.

## Debug vs Release (⚠ CRITICAL)
- `build-metal` = Debug (`-O0`, ~70 tok/s) — dev only.
- Release needs: `-DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DGGML_ACCELERATE=ON -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=Apple -DBUILD_SHARED_LIBS=OFF` (~89 tok/s).
- NEVER benchmark/ship/package a Debug dylib.
- PACKAGING RULE: refresh `csharp/runtimes/osx-arm64/native/libpvxinference.dylib` from the Release build dir before building a NuGet package.
- Reconfigure copied build dirs: they keep ABSOLUTE source paths — run `cmake -S . -B <dir>` again in the copy, or the original repo's sources get compiled silently.
- Never "fix" native-load issues by copying a dylib into an app root — that stale copy silently wins load order.

## Key Entry Points (for agents navigating the code)
- `include/pvxinference.h` — the flat C API (60+ functions); ABI version constant `PVX_ABI_VERSION`
- `src/pvxinference.cpp` — core (model, context, pool, executor, sampling, vision, KV)
- `src/pvx_backend.cpp` — backend probe + progressive fallback + Vulkan stale-KV flush
- `csharp/` — .NET 8 bindings (Interop / SafeHandles / Abstractions / Implementation / Tests)
- `tests/test_basic.cpp`, `tests/test_stress.cpp` — native suites
- `docs/CAPABILITY.md` — capability matrix + careful-handling areas for the C# layer
- `docs/CONFIGURATION.md` — embedder params + test env vars
- `ARCHITECTURE.md` — authoritative design doc (addenda at top = current truth)

## ABI Rule
Any native signature change: bump `PVX_ABI_VERSION` (header) and `PvxNative.ExpectedAbiVersion` (C#) in lockstep — mismatch must fail fast, not corrupt stacks.

**Status:** Current as of 2026-10-11.