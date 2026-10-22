# Third-Party Notices

ProvostXInference is MIT licensed (see LICENSE). It incorporates the following
third-party components.

## llama.cpp / ggml

- **Used for:** the entire native inference core — GGUF model loading, tokenization,
  batched decoding, KV cache management, vision (MTMD), embeddings, and the Metal/CUDA/
  Vulkan/CPU backend layer
- **Source:** https://github.com/ggml-org/llama.cpp (vendored as a git submodule under `llama.cpp/`)
- **License:** MIT — the full license text ships in `llama.cpp/LICENSE`
  (Copyright (c) 2023-2026 The ggml authors)
- **Linking:** statically linked into `libpvxinference` (`BUILD_SHARED_LIBS=OFF`).
  Upstream llama.cpp is used PRISTINE — no local patches, no fork.

## Platform acceleration libraries

- **Apple Metal / Accelerate / BLAS (macOS builds)** and **CUDA / Vulkan SDKs
  (Linux/Windows builds)** are platform libraries provided by the OS vendor and are
  not redistributed by this project; they are linked from the user's installed SDKs.

## Microsoft.ML.OnnxRuntime (C# Decision lane)

- **Used for:** the ONNX judge execution lane (`csharp/Decision/`) — tokenizer-driven
  decision sessions over ONNX models
- **Packages:** `Microsoft.ML.OnnxRuntime` 1.30.0 (+ `.Gpu` and `.Gpu.Linux` variants)
- **License:** MIT — verified against the package file
  `~/.nuget/packages/microsoft.ml.onnxruntime/1.30.0/LICENSE`
  (Copyright (c) Microsoft Corporation)
- **Note:** the GPU variants enable CUDA/TensorRT execution-provider chains; the
  decision lane runs CPU-only on hosts without them (typed, truthful provider report).

## C# bindings

Beyond the ONNX Runtime packages above, the `csharp/` project P/Invokes the native
library directly and takes no further third-party NuGet packages.

If you redistribute ProvostXInference (as a NuGet package or as part of a combined
installation), you must preserve the llama.cpp/ggml MIT license text as described above.
