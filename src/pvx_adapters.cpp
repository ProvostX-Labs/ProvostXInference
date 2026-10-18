// pvx_adapters.cpp — ABI v8 wave 3: LoRA adapter runtime attach/detach +
// control vectors (pattern-ports of the vendored llama.cpp adapter surface;
// common/common.cpp common_control_vector_load is reference reading only —
// never linked, never included).
//
// KV-neutrality (owner constraint, argued): both LoRA adapters and control
// vectors are WEIGHTS-surface modifiers (llama_set_adapters_lora /
// llama_set_adapter_cvec). They change future forward passes exactly like a
// sampling-parameter change; existing KV content, conversations, mirror
// rings and context bookkeeping are untouched. Attach/detach runs under the
// owning context's infer mutex (serialized against decodes). No context
// recreation is required (upstream semantics — llama-server hot-swaps them).

#include "pvx_internal.h"
#include "gguf.h"
#include <string>
#include <vector>

// ── LoRA adapters (plan feature 9) ──

struct pvx_adapter_s {
    llama_adapter_lora* lora = nullptr;
    pvx_model_t* owner = nullptr;   // adapters die with their model (upstream)
};

pvx_result_t pvx_adapter_load(pvx_model_t* model, const char* path,
                              pvx_adapter_t** out_adapter) {
    if (!model || !path || !out_adapter) return PVX_ERR_INVALID_ARG;
    *out_adapter = nullptr;

    // Capability-truthful base-arch precheck: the adapter's GGUF must declare
    // the same base architecture family as the target model (typed, named —
    // before any weights are read).
    char model_arch[64] = {0};
    char adapter_arch[64] = {0};

    llama_adapter_lora* probe = llama_adapter_lora_init(model->model, path);
    if (!probe) {
        model->last_error = "adapter/file: llama_adapter_lora_init failed for '" +
                           std::string(path) + "' (missing/unreadable, wrong base "
                           "model, or rank/shape mismatch)";
        return PVX_ERR_LOAD_FAILED;
    }
    // Re-read arch AFTER a successful init (metadata access needs a live adapter).
    if (llama_adapter_meta_val_str(probe, "general.architecture", adapter_arch, sizeof(adapter_arch)) > 0 &&
        llama_model_meta_val_str(model->model, "general.architecture", model_arch, sizeof(model_arch)) > 0 &&
        std::string(adapter_arch) != std::string(model_arch)) {
        llama_adapter_lora_free(probe);
        model->last_error = "adapter/arch: adapter base '" + std::string(adapter_arch) +
                            "' does not match model '" + std::string(model_arch) + "'";
        return PVX_ERR_LOAD_FAILED;
    }

    auto* adapter = new pvx_adapter_s();
    adapter->lora = probe;
    adapter->owner = model;
    *out_adapter = adapter;
    return PVX_OK;
}

void pvx_adapter_free(pvx_adapter_t* adapter) {
    if (!adapter) return;
    // NOTE: loaded adapters freed here are owned by us; the model also frees
    // any not manually freed at model destruction (upstream contract).
    if (adapter->lora) llama_adapter_lora_free(adapter->lora);
    delete adapter;
}

pvx_result_t pvx_context_set_lora(pvx_context_t* ctx,
                                  const pvx_adapter_t* const* adapters,
                                  const float* scales,
                                  int n_adapters) {
    if (!ctx) return PVX_ERR_INVALID_ARG;
    if (n_adapters < 0 || (n_adapters > 0 && (!adapters || !scales)))
        return PVX_ERR_INVALID_ARG;
    if (n_adapters > 0 && !adapters) return PVX_ERR_INVALID_ARG;
    for (int i = 0; i < n_adapters; i++) {
        if (!adapters[i]) return PVX_ERR_INVALID_ARG;           // boundary: adapter/handle
        if (!(scales[i] > 0.0f)) {
            ctx->last_error = "adapter/scale: scale for adapter " +
                              std::to_string(i) + " must be > 0";
            return PVX_ERR_INVALID_ARG;
        }
    }

    std::lock_guard<std::mutex> lock(ctx->infer_mtx);

    // n == 0 detaches (nullptr array, 0 count — upstream "no adapters" call).
    std::vector<llama_adapter_lora*> loras;
    std::vector<float> sc;
    for (int i = 0; i < n_adapters; i++) {
        loras.push_back(adapters[i]->lora);
        sc.push_back(scales[i]);
    }

    // All-or-nothing: upstream sets the full array or fails; on failure the
    // context keeps its previous set (typed error, no partial application).
    int32_t rc = llama_set_adapters_lora(ctx->ctx,
                                        loras.empty() ? nullptr : loras.data(),
                                        loras.size(),
                                        sc.empty() ? nullptr : sc.data());
    if (rc != 0) {
        ctx->last_error = "adapter/set: llama_set_adapters_lora failed (rc=" +
                          std::to_string(rc) + "; context keeps its previous adapter set)";
        return PVX_ERR_INTERNAL;
    }
    return PVX_OK;
}

// ── Control vectors (plan feature 10) ──
//
// Pattern-port of common_control_vector_load_one (common.cpp:1969):
// direction.<layer> F32 1-D tensors (layer >= 1, "layer 0 is not used")
// accumulate into one n_embd x max_layer buffer (layer 1 at [0]). Fixed
// strength 1.0 (single-file loads; multiple files can be layered by callers
// only via separate context sets — kept out of the ABI for now).

struct pvx_cvec_s {
    int n_embd = 0;
    std::vector<float> data;   // n_embd x max_layer (layer 1 at [0])
};

pvx_result_t pvx_control_vector_load(pvx_model_t* model, const char* path,
                                     pvx_cvec_t** out_cvec) {
    if (!model || !path || !out_cvec) return PVX_ERR_INVALID_ARG;
    *out_cvec = nullptr;

    // Direction tensors read raw through the stable gguf C API (same surface
    // the vendored loader uses — pattern-port, no link).
    ggml_context* gctx = nullptr;
    gguf_init_params ip = {};
    ip.no_alloc = false;
    ip.ctx = &gctx;
    gguf_context* gf = gguf_init_from_file(path, ip);
    if (!gf) {
        model->last_error = "cvector/file: cannot open '" + std::string(path) + "'";
        return PVX_ERR_LOAD_FAILED;
    }

    auto fail = [&](const std::string& boundary, const std::string& msg) {
        gguf_free(gf);
        if (gctx) ggml_free(gctx);
        model->last_error = boundary + ": " + msg;
        return PVX_ERR_LOAD_FAILED;
    };

    const int n_tensors = gguf_get_n_tensors(gf);
    if (n_tensors == 0) return fail("cvector/file", "no direction tensors in " + std::string(path));

    const int model_n_embd = llama_model_n_embd(model->model);
    auto* cvec = new pvx_cvec_s();
    cvec->n_embd = model_n_embd;

    // Walk direction.<L> tensors (upstream common_control_vector_load_one loop).
    for (int i = 0; i < n_tensors; i++) {
        const char* name = gguf_get_tensor_name(gf, i);
        if (!name) return fail("cvector/file", "tensor name read failed");

        int layer_idx = -1;
        const char* dot = strchr(name, '.');
        if (dot && strncmp(name, "direction", 9) == 0 && (size_t)(dot - name) == 9)
            layer_idx = atoi(dot + 1);
        if (layer_idx < 0)
            return fail("cvector/file", std::string("invalid/unparsable direction tensor layer index in '") + name + "'");
        if (layer_idx == 0)
            return fail("cvector/file", std::string("invalid (zero) direction tensor layer index in '") + name + "'");

        ggml_tensor* tensor = ggml_get_tensor(gctx, name);
        if (!tensor) return fail("cvector/file", std::string("tensor not found: ") + name);
        if (tensor->type != GGML_TYPE_F32)
            return fail("cvector/file", std::string("non-F32 direction tensor '") + name + "'");
        if (ggml_n_dims(tensor) != 1)
            return fail("cvector/file", std::string("non-1D direction tensor '") + name + "'");
        const int64_t ne = ggml_nelements(tensor);
        if ((int) ne != model_n_embd) {
            delete cvec;
            return fail("cvector/n-embd",
                        "direction width " + std::to_string(ne) +
                        " != model n_embd " + std::to_string(model_n_embd));
        }

        // Extend to cover layer L (layer 1 lives at [0]; layer 0 unused).
        if (cvec->data.size() < (size_t) model_n_embd * layer_idx)
            cvec->data.resize((size_t) model_n_embd * layer_idx, 0.0f);
        const float* src = (const float*) tensor->data;
        float* dst = cvec->data.data() + (size_t) model_n_embd * (layer_idx - 1);
        for (int j = 0; j < model_n_embd; j++) dst[j] += src[j];  // multiple files/entries accumulate
    }

    if (cvec->data.empty()) {
        delete cvec;
        return fail("cvector/file", "no valid direction tensors loaded from " + std::string(path));
    }

    gguf_free(gf);
    if (gctx) ggml_free(gctx);
    *out_cvec = cvec;
    return PVX_OK;
}

void pvx_control_vector_free(pvx_cvec_t* cvec) { delete cvec; }

pvx_result_t pvx_context_set_control_vector(pvx_context_t* ctx,
                                            const pvx_cvec_t* cvec,
                                            int layer_start,
                                            int layer_end) {
    if (!ctx) return PVX_ERR_INVALID_ARG;

    const int n_layer = llama_model_n_layer(ctx->model);

    if (!cvec) {
        // Detach: NULL data clears the applied vector (upstream contract).
        std::lock_guard<std::mutex> lock(ctx->infer_mtx);
        int32_t rc = llama_set_adapter_cvec(ctx->ctx, nullptr, 0, 0, 0, 0);
        if (rc != 0) {
            ctx->last_error = "cvector/clear: llama_set_adapter_cvec(null) failed (rc=" +
                              std::to_string(rc) + ")";
            return PVX_ERR_INTERNAL;
        }
        return PVX_OK;
    }

    // Defaults mirror the vendored common.cpp: layer_start<=0 -> 1,
    // layer_end<=0 -> n_layer. EXPLICIT out-of-range ends (> n_layer) are a
    // typed refusal (never a silent clamp — the caller must know).
    if (layer_start <= 0) layer_start = 1;
    if (layer_end <= 0) layer_end = n_layer;
    if (layer_end > n_layer) {
        ctx->last_error = "cvector/layer-range: layer_end " +
                          std::to_string(layer_end) + " beyond model n_layer " +
                          std::to_string(n_layer) + " (explicit range is never clamped)";
        return PVX_ERR_INVALID_ARG;
    }
    if (layer_start < 1 || layer_start > n_layer || layer_start > layer_end) {
        ctx->last_error = "cvector/layer-range: layer range [" +
                          std::to_string(layer_start) + ", " + std::to_string(layer_end) +
                          "] invalid for model with " + std::to_string(n_layer) + " layers";
        return PVX_ERR_INVALID_ARG;
    }

    std::lock_guard<std::mutex> lock(ctx->infer_mtx);
    // il_start/il_end are inclusive on llama_set_adapter_cvec. Data covers
    // [1, max_layer_from_file]; only the in-range slice is applied.
    int32_t rc = llama_set_adapter_cvec(ctx->ctx,
                                        cvec->data.data(),
                                        cvec->data.size(),
                                        cvec->n_embd,
                                        layer_start,
                                        layer_end);
    if (rc != 0) {
        ctx->last_error = "cvector/set: llama_set_adapter_cvec failed (rc=" +
                          std::to_string(rc) + ")";
        return PVX_ERR_INTERNAL;
    }
    return PVX_OK;
}
