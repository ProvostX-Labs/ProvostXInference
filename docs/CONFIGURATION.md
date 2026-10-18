# ProvostXInference — Configuration Reference

**Summary:** ProvostXInference (PVX) is a C/C++ inference library with C# bindings, wrapping llama.cpp. The product code reads **zero environment variables** — all configuration is passed programmatically via C API structs by the embedding application (e.g., the ProvostX LLM server). Test/benchmark code reads three `PVX_*` env vars for model paths.

## Product Code: Environment Variables

**None.** A grep of `src/*.cpp` and `csharp/**/*.cs` (excluding tests) for `getenv` / `GetEnvironmentVariable` returned zero matches. All PVX behavior is controlled by programmatic parameters set by the caller. This was confirmed by audit on 2026-09-26 and remains true as of 2026-09-27.

## Product Code: C API Programmatic Parameters

These structs are the "knobs" embedders set when calling the PVX C API. They are not env vars — they are passed directly to `pvx_load_model`, `pvx_create_context`, and `pvx_executor_sample` / `pvx_conversation_sample`.

### `pvx_model_params_t` (passed to `pvx_load_model`)

| Field | C Type | Default | What it does |
|-------|--------|---------|--------------|
| `model_path` | `const char*` | — (required) | Path to GGUF model file |
| `gpu_layers` | `int` | — | GPU layers to offload. `-1` = all, `0` = CPU only |
| `threads` | `int` | — | CPU threads. `-1` = auto-detect |
| `flash_attn` | `bool` | — | Enable Flash Attention |
| `kv_cache_type` | `pvx_kv_type_t` | — | KV cache quantization: `F16`, `Q8_0`, `Q4_0`, `Q4_1` |
| `reasoning_markers` | `const char**` | `NULL` → built-in `{" think"}` | Tag strings probed against vocab to detect thinking-capable models. First single-special-token match drives empty-block prefill when `enable_thinking=0` |
| `n_reasoning_markers` | `int` | `<=0` → built-in default | Count of reasoning marker strings |

### `pvx_context_params_t` (passed to `pvx_create_context`)

| Field | C Type | Default | What it does |
|-------|--------|---------|--------------|
| `context_size` | `uint32_t` | — | Context window size (`n_ctx`) |
| `batch_size` | `uint32_t` | — | Batch size for decode (`n_batch`) |
| `seq_max` | `uint32_t` | — | Max concurrent sequences (batch mode pool size) |
| `pooling_type` | `pvx_pooling_type_t` | — | Embedding pooling: `NONE`, `MEAN`, `CLS`, `LAST` |

### `pvx_sampling_params_t` (passed to sample functions)

| Field | C Type | Default (C# `SamplingConfig`) | What it does |
|-------|--------|-------------------------------|--------------|
| `temperature` | `float` | `0.3` | Sampling temperature |
| `top_p` | `float` | `0.95` | Nucleus sampling top-p |
| `top_k` | `int` | `40` | Top-k sampling. `0` = disabled |
| `min_p` | `float` | `0.0` | Min-p filtering threshold |
| `repeat_penalty` | `float` | `1.1` | Penalty for repeated tokens |
| `repeat_last_n` | `int` | `-1` | Window size for repeat penalty. `-1` = full context |
| `penalty_present` | `float` | `0.0` | Presence penalty |
| `max_tokens` | `int` | `256` | Max tokens to generate |
| `ignore_eos` | `bool` | `false` | If true, EOS token is not treated as end-of-generation |

### C# Managed Defaults (record defaults)

| Record | Field | Default |
|--------|-------|---------|
| `ModelConfig` | `GpuLayers` | `-1` (all) |
| `ModelConfig` | `Threads` | `0` (auto) |
| `ModelConfig` | `FlashAttention` | `true` |
| `ModelConfig` | `KvCacheType` | `F16` |
| `ContextConfig` | `ContextSize` | `4096` |
| `ContextConfig` | `BatchSize` | `512` |
| `ContextConfig` | `SeqMax` | `4` |
| `ContextConfig` | `PoolingType` | `None` |

### Logging (Programmatic, not env-var)

| Knob | API | Default | What it does |
|------|-----|---------|--------------|
| Log callback | `pvx_set_log_callback(NULL)` | `NULL` (zero-overhead) | Sets a global C callback for structured logs. NULL = no logging at all |
| Min log level | `pvx_set_log_level(level)` | `PVX_LOG_DEBUG` | Messages below this level are not dispatched. Levels: `DEBUG < INFO < WARN < ERROR < NONE` |

## Test-Only Environment Variables

These are read **only** by the C# test suite (`SmokeTests.cs`). They are never read by product code.

| Variable | Type | Default | What it does |
|----------|------|---------|--------------|
| `PVX_MODEL_PATH` | string (env) | `./models/Qwen3.5-4B-Q4_K_M.gguf` | Path to the main GGUF model used by smoke tests |
| `PVX_EMBED_MODEL_PATH` | string (env) | `./models/all-MiniLM-L6-v2-Q5_K_M.gguf` | Path to the embedding model for embedding tests |
| `PVX_MMPROJ_PATH` | string (env) | `./models/mmproj-Qwen3.5-4B-BF16.gguf` | Path to the multimodal projector for vision tests |

**Status:** Current as of 2026-09-27. Product code has zero env-var-driven behavior. All runtime configuration is programmatic via C API structs set by the embedding application.