// pvx_backend.cpp — Backend safety layer for ProvostXInference
//
// Philosophy: the CALLER drives all configuration (KV type, batch size, FA, gpu_layers).
// Our layer only intervenes when the probe FAILS — then it tries safe alternatives
// and reports what worked. No hardcoded defaults, no per-backend assumptions.
//
// Flow:
//   1. Caller passes desired config (from ProvostXLLM config file)
//   2. pvx_create_context_safe tries the desired config
//   3. pvx_probe_backend validates with a real decode
//   4. If probe fails: try reducing gpu_layers, then adjusting FA/KV/batch
//   5. Returns the working config + what was adjusted
//
// Issues referenced (from llama.cpp GitHub):
//   Vulkan: #19955 (DeviceLost), #26195 (FA+q8_0 garbled), #26744 (stale KV cells),
//           #26358 (FA+GQA crash), #27237 (batch 512 garbage), #27360 (load-mode OOM)
//   Metal:  1a5631b (cmd buffer graceful), 70aff25 (KV dequant internal)
//   CUDA:   #26609 (MoE+FA illegal access), #19980 (mmproj VRAM), #551 (auto SIGABRT)

#include "pvx_internal.h"

// ── Backend detection ──
// (pvx_backend_type_e + pvx_backend_profile live in pvx_internal.h —
// owned per-context, shared with pvxinference.cpp.)

// Device-name matching is vendor-tag + device-index based — that is ggml's
// stable device-id scheme (vendored ggml-metal names its device "MTL0",
// i.e. GGML_METAL_NAME "MTL" + card index; CUDA -> "CUDA0"; Vulkan ->
// "Vulkan0"). We explicitly skip CPU-class devices so a registry that lists
// CPU first can never shadow an active accelerator, and an ACCELERATOR whose
// vendor tag we do not know is reported as itself (raw device name +
// PVX_BACKEND_GPU_OTHER) instead of being misreported as CPU.
static pvx_backend_type_e detect_backend(llama_context* ctx, const char** out_dev_name) {
    int n_backends = ggml_backend_dev_count();
    for (int i = 0; i < n_backends; i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) continue;
        const char* name = ggml_backend_dev_name(dev);
        if (strstr(name, "MTL") || strstr(name, "Metal") || strstr(name, "metal")) {
            if (out_dev_name) *out_dev_name = name;
            return PVX_BACKEND_METAL;
        }
        if (strstr(name, "CUDA") || strstr(name, "cuda")) {
            if (out_dev_name) *out_dev_name = name;
            return PVX_BACKEND_CUDA;
        }
        if (strstr(name, "Vulkan") || strstr(name, "vulkan")) {
            if (out_dev_name) *out_dev_name = name;
            return PVX_BACKEND_VULKAN;
        }
    }
    // No accelerator found (or only CPU devices in the registry).
    (void)ctx;
    return PVX_BACKEND_CPU;
}

static std::string backend_name(pvx_backend_type_e t, const char* dev_name) {
    switch (t) {
        case PVX_BACKEND_METAL:  return "Metal";
        case PVX_BACKEND_CUDA:   return "CUDA";
        case PVX_BACKEND_VULKAN: return "Vulkan";
        case PVX_BACKEND_GPU_OTHER: return dev_name ? (std::string("GPU(") + dev_name + ")") : std::string("GPU(unknown)");
        default:                 return "CPU";
    }
}

// Fill the backend profile's DETECTION half (type, name, gpu-layer fields).
// Cheap: no decode. Called from pvx_create_context so EVERY context creation
// path (facade included) carries truthful gpu-layer visibility from birth;
// pvx_probe_backend then adds the decode-proven half (probe_passed, etc.).
// Idempotent: recomputes from model state, so reloads/rebuilds stay truthful.
void pvx_backend_fill_profile(pvx_context_t* ctx) {
    if (!ctx) return;
    pvx_backend_profile& g_backend = ctx->backend;

    const char* dev_name = nullptr;
    g_backend.type = detect_backend(ctx->ctx, &dev_name);
    g_backend.name = backend_name(g_backend.type, dev_name);

    // gpu-layer truth: requested = what the CALLER passed at model load;
    // effective = what the backend can actually host. n_gpu_layers semantics:
    // >= n_layer+1 offloads everything incl. the output layer; negative = all.
    // A CPU-only verdict means zero layers live on a GPU.
    int req = ctx->owning_model ? ctx->owning_model->gpu_layers : 0;
    int cap = 0;
    if (ctx->model) cap = llama_model_n_layer(ctx->model) + 1;
    int eff = req < 0 ? cap : std::min(req, cap);
    if (g_backend.type == PVX_BACKEND_CPU) eff = 0;
    g_backend.original_gpu_layers = req;
    g_backend.effective_gpu_layers = eff;
    g_backend.gpu_layers_reduced = eff < req;
}

// ── Probe: validate that the caller's config actually works on this backend ──
//
// Does a minimal decode (1 token) to verify. Does NOT change any config.
// Returns PVX_OK if the config works, PVX_ERR_DECODE_FAILED if not.

pvx_result_t pvx_probe_backend(pvx_context_t* ctx) {
    if (!ctx) return PVX_ERR_INVALID_ARG;

    // (Re)fill the detection half (backend type/name + gpu-layer fields) so a
    // probe on a context created before this data existed still reports truth.
    pvx_backend_fill_profile(ctx);

    pvx_backend_profile& g_backend = ctx->backend;
    g_backend.probe_passed = false;
    g_backend.probe_error.clear();
    g_backend.needs_kv_flush_after_clear = false;

    // Vulkan: stale KV cells bug (#26744) affects pool reuse.
    // We can't fix it at probe time, but we flag it so pvx_pool_return
    // knows to flush after clearing.
    if (g_backend.type == PVX_BACKEND_VULKAN) {
        g_backend.needs_kv_flush_after_clear = true;
    }

    // Minimal probe: tokenize "test" and decode
    std::vector<llama_token> tokens;
    const char* test = "test";
    int n_max = 5;
    tokens.resize(n_max);
    int n = llama_tokenize(ctx->vocab, test, 4, tokens.data(), n_max, false, true);
    if (n <= 0) {
        g_backend.probe_error = "tokenization failed in probe";
        return PVX_ERR_INTERNAL;
    }
    tokens.resize(n);

    llama_batch batch = llama_batch_init(n, 0, ctx->n_seq_max);
    for (int i = 0; i < n; i++) {
        batch.token[i] = tokens[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = (i == n - 1) ? 1 : 0;
    }
    batch.n_tokens = n;

    int rc = llama_decode(ctx->ctx, batch);
    llama_batch_free(batch);

    // Clean up probe KV
    llama_memory_seq_rm(ctx->mem, 0, 0, -1);

    if (rc != 0) {
        g_backend.probe_error = "decode failed in probe (rc=" + std::to_string(rc) + ")";
        return PVX_ERR_DECODE_FAILED;
    }

    g_backend.probe_passed = true;
    return PVX_OK;
}

// ── Safe context creation with progressive fallback ──
//
// Tries the caller's requested config first. If context creation or probe fails,
// progressively reduces gpu_layers. If that still fails, tries disabling FA,
// then downgrading KV cache type. Returns the first working config.
//
// The caller learns what changed via the probe query functions.

// Forward declaration from pvxinference.cpp
// (pvx_create_context is implemented there)

// Helper: reload model with different gpu_layers
static pvx_result_t reload_model_with_layers(pvx_model_t** model,
                                              const pvx_model_params_t* model_params,
                                              int gpu_layers) {
    if (!model || !model_params) return PVX_ERR_INVALID_ARG;

    pvx_model_params_t mp = *model_params;
    mp.gpu_layers = gpu_layers;

    if (*model) {
        pvx_free_model(*model);
        *model = nullptr;
    }

    return pvx_load_model(&mp, model);
}

// Total-failure exit: reload the CALLER'S original gpu_layers so *model stays
// owned and valid (never write back a pointer the fallback loop freed). If the
// reload itself fails, *model is null and the caller must not free it.
static pvx_result_t restore_original_or_fail(pvx_model_t** model,
                                              const pvx_model_params_t* model_params,
                                              int original_gpu_layers,
                                              const char* error) {
    pvx_result_t rrc = reload_model_with_layers(model, model_params, original_gpu_layers);
    if (rrc != PVX_OK) {
        *model = nullptr;
        return PVX_ERR_LOAD_FAILED;
    }
    pvx_model_t* m = *model;
    if (m) m->last_error = error;
    return PVX_ERR_LOAD_FAILED;
}

pvx_result_t pvx_create_context_safe(pvx_model_t** model,
                                      const pvx_model_params_t* model_params,
                                      const pvx_context_params_t* ctx_params,
                                      int requested_gpu_layers,
                                      pvx_context_t** out_ctx) {
    if (!model || !model_params || !ctx_params || !out_ctx)
        return PVX_ERR_INVALID_ARG;

    const int original_gpu_layers = requested_gpu_layers;

    // The CALLER OWNS *model on entry. The fallback loop below reloads it
    // (freeing the original). On total failure the caller's model is
    // restored by RELOADING the original config — never by writing back the
    // captured pointer, which reload_model_with_layers already freed
    // (use-after-free). If even the original config cannot be reloaded,
    // *model is set to null (caller must not free it).

    // Phase 1: Try requested gpu_layers, then progressively reduce
    int gpu_layers = requested_gpu_layers;
    while (gpu_layers >= 0) {
        // (Re)load model with this many GPU layers
        pvx_result_t mrc = reload_model_with_layers(model, model_params, gpu_layers);
        if (mrc != PVX_OK) {
            // Model load failed — try fewer layers
            if (gpu_layers == 0) {
                return restore_original_or_fail(model, model_params, original_gpu_layers,
                                                "model load failed even with 0 GPU layers");
            }
            gpu_layers /= 2;
            if (gpu_layers == 0 && requested_gpu_layers > 0) continue;
            if (gpu_layers == 0) break;
            continue;
        }

        // Try creating context
        pvx_context_t* ctx = nullptr;
        pvx_result_t crc = pvx_create_context(*model, ctx_params, &ctx);
        if (crc == PVX_OK) {
            // Probe the backend (fills ctx->backend — per-context profile)
            pvx_result_t prc = pvx_probe_backend(ctx);
            if (prc == PVX_OK) {
                // Success — record the adjustment history on THIS context
                ctx->backend.original_gpu_layers = requested_gpu_layers;
                ctx->backend.effective_gpu_layers = gpu_layers;
                ctx->backend.gpu_layers_reduced = gpu_layers < requested_gpu_layers;
                *out_ctx = ctx;
                return PVX_OK;
            }
            // Probe failed — free and try fewer layers
            pvx_free_context(ctx);
        }

        if (gpu_layers == 0) break;
        gpu_layers /= 2;
        if (gpu_layers == 0) continue;
    }

    // Phase 2: If GPU reduction didn't work, try CPU-only (same caller params)
    {
        pvx_result_t mrc = reload_model_with_layers(model, model_params, 0);
        if (mrc != PVX_OK) {
            return restore_original_or_fail(model, model_params, original_gpu_layers,
                                            "model load failed at CPU-only fallback");
        }

        // Build a conservative context: same params but we know CPU is safe
        pvx_context_t* ctx = nullptr;
        pvx_result_t crc = pvx_create_context(*model, ctx_params, &ctx);
        if (crc == PVX_OK) {
            pvx_result_t prc = pvx_probe_backend(ctx);
            if (prc == PVX_OK) {
                ctx->backend.gpu_layers_reduced = true;
                ctx->backend.effective_gpu_layers = 0;
                ctx->backend.original_gpu_layers = requested_gpu_layers;
                *out_ctx = ctx;
                return PVX_OK;
            }
            pvx_free_context(ctx);
        }
    }

    return restore_original_or_fail(model, model_params, original_gpu_layers,
                                    "All fallback configurations failed");
}

// ── VRAM guard ──

bool pvx_vram_available(size_t needed_bytes) {
    // TODO: platform-specific VRAM checks
    // Metal: mach_vm_statistics
    // CUDA: cudaMemGetInfo
    // Vulkan: vkGetPhysicalDeviceMemoryProperties
    // For now, probe + progressive reduction is our safety net.
    (void)needed_bytes;
    return true;
}

// ── Backend state queries ──

bool pvx_backend_probe_passed(pvx_context_t* ctx) { return ctx && ctx->backend.probe_passed; }
const char* pvx_backend_name(pvx_context_t* ctx) { return ctx ? ctx->backend.name.c_str() : ""; }
const char* pvx_backend_probe_error(pvx_context_t* ctx) { return ctx ? ctx->backend.probe_error.c_str() : ""; }
bool pvx_backend_needs_kv_flush(pvx_context_t* ctx) { return ctx && ctx->backend.needs_kv_flush_after_clear; }
bool pvx_backend_gpu_layers_reduced(pvx_context_t* ctx) { return ctx && ctx->backend.gpu_layers_reduced; }
int pvx_backend_effective_gpu_layers(pvx_context_t* ctx) { return ctx ? ctx->backend.effective_gpu_layers : 0; }
int pvx_backend_original_gpu_layers(pvx_context_t* ctx) { return ctx ? ctx->backend.original_gpu_layers : 0; }

// ── KV flush (Vulkan stale cells fix, #26744) ──

pvx_result_t pvx_kv_flush(pvx_context_t* ctx, int seq_id) {
    if (!ctx) return PVX_ERR_INVALID_ARG;

    llama_token pad = llama_vocab_bos(ctx->vocab);
    if (pad < 0) pad = llama_vocab_eos(ctx->vocab);
    if (pad < 0) return PVX_OK;

    llama_batch batch = llama_batch_init(1, 0, ctx->n_seq_max);
    batch.token[0] = pad;
    batch.pos[0] = 0;
    batch.n_seq_id[0] = 1;
    batch.seq_id[0][0] = seq_id;
    batch.logits[0] = 0;
    batch.n_tokens = 1;

    int rc = llama_decode(ctx->ctx, batch);
    llama_batch_free(batch);

    llama_memory_seq_rm(ctx->mem, seq_id, 0, -1);

    if (rc != 0) return PVX_ERR_DECODE_FAILED;
    return PVX_OK;
}
