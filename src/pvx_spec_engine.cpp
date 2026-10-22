// pvx_spec_engine.cpp — Standard-executor speculative-decoding session.
// See pvx_spec_engine.h for scope, semantics, and invariants.

#include "pvx_spec_engine.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include "speculative.h"
#include "llama-ext.h"
#include "gguf.h"

// Seams exported by pvxinference.cpp (same pipeline as the plain loop):
int32_t spec_sample(const float* logits, const llama_vocab* vocab,
                    const pvx_sampling_params_t* params,
                    const std::vector<llama_token>& recent_tokens);
void spec_snapshot(llama_context* lctx, std::vector<float>& dest, int row);

namespace {

// Upstream draft-simple defaults (vendored common/common.h).
constexpr int   kDraftMaxDefault = 16;
constexpr int   kDraftMinDefault = 0;
constexpr float kPMinDefault     = 0.0f;

// Upstream n-gram defaults (vendored common/common.h):
//   ngram_mod      n_match=24, n_max=64, n_min=48
//   ngram_map_k    size_n=12, size_m=48, min_hits=1
//   ngram_simple   size_n=12, size_m=48
// Divergence (documented): the executor's verify chain is bounded by
// draft_max/batch width (upstream verifies up to 64 rows), so the mod n_min
// floor defaults to 0 here — upstream's 48 would clear every chain the
// executor can physically verify.
constexpr int kNgSimpleNDefault  = 12;
constexpr int kNgSimpleMDefault  = 48;
constexpr int kNgMapKNDefault    = 12;
constexpr int kNgMapKMDefault    = 48;
constexpr int kNgMapKMinHitsDefault = 1;
constexpr int kNgModMatchDefault = 24;
constexpr int kNgModMaxDefault   = 64;
constexpr int kNgModMinDefault   = 0;
constexpr int kNgCacheMinDefault = 1;
constexpr int kNgCacheMaxDefault = 2;

// MOD hash-table capacity (upstream: 4M entries).
constexpr size_t kNgModTableSize = 4 * 1024 * 1024;


// dynamic-cache "not yet baselined" sentinel.
constexpr size_t kNgCacheNoBaseline = (size_t) -1;

// ── Companion-GGUF metadata prechecks (ABI v7) ─────────────────────────────
// Read metadata ONLY (gguf no_alloc): cheap, typed, fail-closed before any
// model load. Pattern-port of the constructors' own metadata reads (vendored
// speculative.cpp draft-eagle3 / draft-dflash constructors). GGUF key layout
// follows llama-arch.cpp: "%s.target_layers" with the arch prefix.
static bool gguf_meta_target_layers(const char* path, const char* arch_prefix,
                                    std::vector<int32_t>& out_ids, std::string& out_error) {
    const std::string boundary = std::string("spec-engine/") + arch_prefix + "-companion-metadata";
    out_error.clear();
    struct gguf_context* g = gguf_init_from_file(path, { /*no_alloc*/ true, /*ctx*/ nullptr });
    if (!g) {
        out_error = std::string("companion GGUF unreadable (boundary: spec-engine/") +
                    arch_prefix + "-companion-load): '" + path + "'";
        return false;
    }
    const std::string want_key = std::string(arch_prefix) + ".target_layers";
    bool ok = true;
    do {
        const int64_t arch_k = gguf_find_key(g, "general.architecture");
        const char* arch_c = nullptr;
        if (arch_k >= 0 && gguf_get_kv_type(g, arch_k) == GGUF_TYPE_STRING) {
            arch_c = gguf_get_val_str(g, arch_k);
        }
        if (!arch_c || std::string(arch_c) != arch_prefix) {
            out_error = std::string("companion GGUF is not a '") + arch_prefix +
                        "' draft model (general.architecture '" +
                        (arch_c ? arch_c : "<missing>") + "') (boundary: " + boundary + ")";
            ok = false;
            break;
        }
        const int64_t tl_k = gguf_find_key(g, want_key.c_str());
        if (tl_k < 0 || gguf_get_kv_type(g, tl_k) != GGUF_TYPE_ARRAY ||
            gguf_get_arr_type(g, tl_k) != GGUF_TYPE_INT32) {
            out_error = "companion GGUF missing metadata key '" + want_key +
                        "' (boundary: " + boundary + ")";
            ok = false;
            break;
        }
        const size_t n = gguf_get_arr_n(g, tl_k);
        const int32_t* ids = (const int32_t*) gguf_get_arr_data(g, tl_k);
        if (n == 0 || !ids) {
            out_error = "companion GGUF metadata key '" + want_key + "' is empty "
                        "(boundary: " + boundary + ")";
            ok = false;
            break;
        }
        for (size_t i = 0; i < n; i++) {
            if (ids[i] < 0) {
                out_error = "companion GGUF metadata key '" + want_key + "' carries a "
                            "negative layer id (boundary: " + boundary + ")";
                ok = false;
                break;
            }
            out_ids.push_back(ids[i]);
        }
    } while (false);
    gguf_free(g);
    return ok;
}

// EAGLE3 precheck: arch == eagle3 + target_layers present (3-entry contract is
// re-validated after the real load — llama_model_target_layer_ids_n — exactly
// like the vendored constructor).
static bool eagle_companion_meta_check(const char* path, std::string& out_error) {
    std::vector<int32_t> ids;
    if (!gguf_meta_target_layers(path, "eagle3", ids, out_error)) return false;
    pvx_log(PVX_LOG_INFO, "Spec", "eagle3 companion metadata: %zu target_layers",
            ids.size());
    return true;
}

// DFlash precheck: arch == dflash + dflash.target_layers present (the vendored
// constructor's GGML_ASSERT(target_layer_ids_n > 0) surfaced typed). The wiring
// boundary refuses afterwards; the metadata check exists so callers learn the
// companion contract BEFORE the typed wiring refusal.
static bool dflash_companion_meta_check(const char* path, std::string& out_error) {
    std::vector<int32_t> ids;
    return gguf_meta_target_layers(path, "dflash", ids, out_error);
}

struct TopLogit {
    llama_token token;
    float       logit;
};

static TopLogit drafter_top_logit(const float* logits, int n_vocab) {
    int best = 0;
    float best_logit = logits[0];
    for (int i = 1; i < n_vocab; i++) {
        if (logits[i] > best_logit) { best_logit = logits[i]; best = i; }
    }
    return { (llama_token) best, best_logit };
}

// Probability mass of the top token, given the full row (exact, stable):
// p_top = 1 / Σ exp(l_i − l_top). Only consulted when a p_min floor is set.
static float top_probability(const float* logits, int n_vocab, float top_logit) {
    double sum = 0.0;
    for (int i = 0; i < n_vocab; i++) {
        sum += std::exp((double) logits[i] - (double) top_logit);
    }
    return (float) (1.0 / sum);
}

} // namespace

// ── Construction (fail-closed) ─────────────────────────────────────────────

std::unique_ptr<PvxBLSpecEngine> PvxBLSpecEngine::create(pvx_model_t* model,
                                                         pvx_context_t* ctx,
                                                         const pvx_spec_params_t* params,
                                                         std::string& out_error) {
    if (!model || !ctx || !params) {
        out_error = "null model/context/params (boundary: spec-engine/config)";
        return nullptr;
    }

    const bool is_ngram = params->mode == PVX_SPEC_NGRAM_SIMPLE ||
                          params->mode == PVX_SPEC_NGRAM_MAP_K  ||
                          params->mode == PVX_SPEC_NGRAM_MOD    ||
                          params->mode == PVX_SPEC_NGRAM_CACHE;

    switch (params->mode) {
        case PVX_SPEC_OFF:
            out_error = "mode OFF passed to spec enable; use the spec_free entry point to "
                        "disable (boundary: spec-engine/config)";
            return nullptr;
        case PVX_SPEC_MTP_AUTO:
            // Named typed boundary: the capability is real, the wiring is
            // not built in this slice. Never faked.
            out_error = "PVX_SPEC_MTP_AUTO not wired in this build: MTP drafting requires "
                        "LLAMA_CONTEXT_TYPE_MTP context plumbing + nextn hidden-state "
                        "extraction this specialist does not construct yet; the mtp "
                        "capability reports model-level presence honestly "
                        "(boundary: spec-engine/mtp-wiring)";
            return nullptr;
        case PVX_SPEC_DFLASH: {
            // ABI v7 feature 3 (typed boundary slice): companion metadata is
            // validated honestly (the vendored draft-dflash constructor's own
            // checks — dflash.target_layers presence, block-size metadata), but
            // the block-diffusion draft loop (mask-token noise batch, draft-KV
            // feature injection, DFlash2 selector lattice / lilicorr paths) is
            // not built in this slice. Fail closed at the wiring boundary,
            // never faked — same posture as MTP_AUTO.
            if (!params->dflash_companion_path || params->dflash_companion_path[0] == '\0') {
                out_error = "PVX_SPEC_DFLASH configured without dflash_companion_path "
                            "(boundary: spec-engine/dflash-config)";
                return nullptr;
            }
            std::string meta_err;
            if (!dflash_companion_meta_check(params->dflash_companion_path, meta_err)) {
                out_error = meta_err;
                return nullptr;
            }
            out_error = "PVX_SPEC_DFLASH companion metadata validated, but the "
                        "block-diffusion draft loop (mask-token denoise batch, "
                        "draft-KV feature injection, DFlash2/lilicorr selector paths) "
                        "is not wired in this build — the capability is never faked "
                        "(boundary: spec-engine/dflash-wiring)";
            return nullptr;
        }
        case PVX_SPEC_DRAFT:
        case PVX_SPEC_EAGLE3:
        case PVX_SPEC_MTP_DRAFT:
        case PVX_SPEC_NGRAM_SIMPLE:
        case PVX_SPEC_NGRAM_MAP_K:
        case PVX_SPEC_NGRAM_MOD:
        case PVX_SPEC_NGRAM_CACHE:
            break;
        default:
            out_error = "unknown spec mode " + std::to_string((int) params->mode);
            return nullptr;
    }

    if (is_ngram) {
        // n-gram drafting needs only a standard vocab — no draft model, no
        // arch constraints. Config sanity is fail-closed.
        if (params->ngram.n_min < 0 || params->ngram.n_max < 0 ||
            (params->ngram.n_min > 0 && params->ngram.n_max > 0 &&
             params->ngram.n_min > params->ngram.n_max)) {
            out_error = "ngram params invalid: need 0 <= n_min <= n_max "
                        "(0 selects per-mode defaults) (boundary: spec-engine/ngram-config)";
            return nullptr;
        }
    }

    pvx_spec_capabilities_t caps{};
    if (pvx_model_spec_capabilities(model, &caps) != PVX_OK ||
        !(is_ngram ? (caps.types & PVX_SPEC_TYPE_NGRAM)
                   : caps.draft)) {
        out_error = is_ngram
            ? "n-gram spec rejected by capability report: this model has no standard "
              "vocabulary to draft from (boundary: spec-engine/target-memory)"
            : "PVX_SPEC_DRAFT rejected by capability report: draft-simple spec requires "
              "a dense (non-hybrid, non-recurrent) target memory with a standard "
              "vocab; this model's supported-types list omits draft_simple "
              "(boundary: spec-engine/target-memory)";
        return nullptr;
    }

    auto engine = std::unique_ptr<PvxBLSpecEngine>(new PvxBLSpecEngine());
    engine->m_mode      = params->mode;
    engine->m_model_tgt = model;
    engine->m_ctx_tgt   = ctx;
    engine->m_draft_max = params->draft_max > 0 ? params->draft_max : kDraftMaxDefault;
    engine->m_draft_min = params->draft_min > 0 ? params->draft_min : kDraftMinDefault;
    engine->m_p_min     = params->p_min >= 0.0f ? params->p_min : kPMinDefault;

    if (is_ngram) {
        const pvx_ngram_params_t& np = params->ngram;
        switch (params->mode) {
            case PVX_SPEC_NGRAM_SIMPLE:
                engine->m_ng_simple.size_ngram = (uint16_t) (np.n_min > 0 ? np.n_min : kNgSimpleNDefault);
                engine->m_ng_simple.size_mgram = (uint16_t) (np.n_max > 0 ? np.n_max : kNgSimpleMDefault);
                break;
            case PVX_SPEC_NGRAM_MAP_K:
                engine->m_ng_map = std::make_unique<pvx::ngram_map>(
                    (uint16_t) (np.n_min > 0 ? np.n_min : kNgMapKNDefault),
                    (uint16_t) (np.n_max > 0 ? np.n_max : kNgMapKMDefault),
                    /*only_keys=*/false,
                    (uint16_t) (np.min_hits > 0 ? np.min_hits : kNgMapKMinHitsDefault));
                break;
            case PVX_SPEC_NGRAM_MOD: {
                const int n_match = np.n_match > 0 ? np.n_match : kNgModMatchDefault;
                if (n_match < 2) {
                    out_error = "ngram mod n_match < 2 is not a meaningful lookup window "
                                "(boundary: spec-engine/ngram-config)";
                    return nullptr;
                }
                engine->m_ng_mod = std::make_unique<pvx::ngram_mod>((uint16_t) n_match,
                                                                   kNgModTableSize);
                engine->m_ng_mod_n_draft = 0;
                engine->m_ng_mod_i_last  = 0;
                engine->m_draft_min = np.n_min > 0 ? np.n_min : kNgModMinDefault;
                break;
            }
            case PVX_SPEC_NGRAM_CACHE: {
                engine->m_ng_cache_ctx     = std::make_unique<pvx::ngram_cache>();
                engine->m_ng_cache_dynamic = std::make_unique<pvx::ngram_cache>();
                engine->m_ng_cache_static  = std::make_unique<pvx::ngram_cache>();
                engine->m_ng_cache_n_min = np.n_min > 0 ? np.n_min : kNgCacheMinDefault;
                engine->m_ng_cache_n_max = np.n_max > 0 ? np.n_max : kNgCacheMaxDefault;
                if (np.cache_path && np.cache_path[0] != '\0') {
                    if (!pvx::ngram_cache_load(np.cache_path, *engine->m_ng_cache_static)) {
                        out_error = std::string("ngram cache file corrupt/unreadable "
                                                "(boundary: spec-engine/ngram-cache): '") +
                                    np.cache_path + "' — PVX_SPEC_NGRAM_CACHE stays disabled, "
                                    "no silent empty-cache fallback";
                        return nullptr;
                    }
                    engine->m_ng_cache_path = np.cache_path;
                }
                break;
            }
            default:
                break;
        }
        pvx_log(PVX_LOG_INFO, "Spec", "ngram session enabled: mode=%d n_min=%d n_max=%d",
                (int) params->mode, params->ngram.n_min, params->ngram.n_max);
        return engine;
    }

    // ── PVX_SPEC_EAGLE3: companion draft GGUF (ABI v7 feature 4) ──
    // Pattern-port of the vendored common_speculative_impl_draft_eagle3
    // constructor: GGUF metadata precheck (never linked — gguf metadata read
    // only), 3-extract-layers contract, target-layer-range check, vocab-family
    // compatibility, draft context with encoder/decoder extraction enabled,
    // target-side layer-input extraction flags (compute-only, KV-neutral).
    if (params->mode == PVX_SPEC_EAGLE3) {
        if (!params->eagle_companion_path || params->eagle_companion_path[0] == '\0') {
            out_error = "PVX_SPEC_EAGLE3 configured without eagle_companion_path "
                        "(boundary: spec-engine/eagle3-config)";
            return nullptr;
        }
        std::string meta_err;
        if (!eagle_companion_meta_check(params->eagle_companion_path, meta_err)) {
            out_error = meta_err;
            return nullptr;
        }
        llama_model_params mp = llama_model_default_params();
        if (params->draft_gpu_layers >= 0) mp.n_gpu_layers = params->draft_gpu_layers;
        engine->m_model_dft = llama_model_load_from_file(params->eagle_companion_path, mp);
        if (!engine->m_model_dft) {
            out_error = std::string("eagle3 companion draft failed to load (boundary: "
                                    "spec-engine/eagle3-companion-load): '") +
                        params->eagle_companion_path + "' — PVX_SPEC_EAGLE3 stays disabled";
            return nullptr;
        }
        const uint32_t n_extract = llama_model_target_layer_ids_n(engine->m_model_dft);
        const int32_t* layer_ids = llama_model_target_layer_ids(engine->m_model_dft);
        if (n_extract != 3) {
            out_error = "eagle3 companion is not EAGLE3 (expected 3 extract layers, got " +
                        std::to_string(n_extract) + ") — metadata key "
                        "'eagle3.target_layers' (boundary: spec-engine/eagle3-companion-metadata)";
            llama_model_free(engine->m_model_dft);
            engine->m_model_dft = nullptr;
            return nullptr;
        }
        engine->m_eagle_n_layers   = (int) n_extract;
        engine->m_eagle_n_embd_tgt = llama_model_n_embd(model->model);
        engine->m_eagle_n_embd_dec = llama_model_n_embd(engine->m_model_dft);
        engine->m_eagle_n_embd_enc = (int) n_extract * engine->m_eagle_n_embd_tgt;
        engine->m_eagle_n_layer_tgt = llama_model_n_layer(model->model);
        for (uint32_t k = 0; k < n_extract; k++) {
            if (layer_ids[k] > engine->m_eagle_n_layer_tgt) {
                out_error = "eagle3 companion target layer id " + std::to_string(layer_ids[k]) +
                            " exceeds target n_layer " + std::to_string(engine->m_eagle_n_layer_tgt) +
                            " (boundary: spec-engine/eagle3-companion-layers)";
                llama_model_free(engine->m_model_dft);
                engine->m_model_dft = nullptr;
                return nullptr;
            }
            engine->m_eagle_layer_ids[k] = layer_ids[k];
        }
        // Vocab compatibility REQUIRED (verification compares draft ids against
        // target samples — same family rule as draft-simple).
        if (!common_speculative_are_compatible(model->model, engine->m_model_dft)) {
            out_error = "eagle3 companion vocab type incompatible with target (boundary: "
                        "spec-engine/eagle3-companion-vocab)";
            llama_model_free(engine->m_model_dft);
            engine->m_model_dft = nullptr;
            return nullptr;
        }
        // Draft context: single stream; batch wide enough for one verify chain.
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx     = ctx->n_ctx;
        cp.n_batch   = std::max<uint32_t>(llama_n_batch(ctx->ctx), (uint32_t) (1 + engine->m_draft_max));
        cp.n_ubatch  = cp.n_batch;
        cp.n_seq_max = 1;
        cp.flash_attn_type = model->flash_attn ? LLAMA_FLASH_ATTN_TYPE_ENABLED
                                               : LLAMA_FLASH_ATTN_TYPE_DISABLED;
        engine->m_ctx_dft = llama_init_from_model(engine->m_model_dft, cp);
        if (!engine->m_ctx_dft) {
            out_error = "eagle3 draft context creation failed (boundary: "
                        "spec-engine/eagle3-companion-context)";
            llama_model_free(engine->m_model_dft);
            engine->m_model_dft = nullptr;
            return nullptr;
        }
        // Turn on extraction of the target layers' input embeddings (upstream
        // constructor pattern). Compute-side flags only — the target's KV and
        // state contracts are untouched (KV-neutrality).
        for (uint32_t k = 0; k < n_extract; k++) {
            if (layer_ids[k] < engine->m_eagle_n_layer_tgt) {
                llama_set_embeddings_layer_inp(ctx->ctx, (uint32_t) layer_ids[k], true);
            } else {
                // layer id == n_layer: the target's pre-norm embedding
                llama_set_embeddings_nextn(ctx->ctx, true, /*masked=*/ false);
            }
        }
        // Draft-side pre-norm hidden state (encoder output + decoder pre-norm).
        llama_set_embeddings_nextn(engine->m_ctx_dft, true, /*masked=*/ true);
        pvx_log(PVX_LOG_INFO, "Spec",
                "eagle3 session ready: n_extract=%u n_embd_tgt=%d n_embd_dec=%d n_embd_enc=%d",
                n_extract, engine->m_eagle_n_embd_tgt, engine->m_eagle_n_embd_dec,
                engine->m_eagle_n_embd_enc);
        return engine;
    }

    // ── PVX_SPEC_MTP_DRAFT: separate self-contained MTP head GGUF ──
    // (ABI v8 wave 4, upstream draft-mtp pattern-port). The head drafts on
    // the target trunk's nextn embeddings (DeepSeek-style (t_{P-1}, h_{P-1})
    // pairing), verified greedy-exact by the target — EAGLE3-lane round loop.
    if (params->mode == PVX_SPEC_MTP_DRAFT) {
        if (!params->mtp_draft_path || params->mtp_draft_path[0] == '\0') {
            out_error = "PVX_SPEC_MTP_DRAFT configured without mtp_draft_path "
                        "(boundary: spec-engine/mtp-config)";
            return nullptr;
        }
        llama_model_params mp = llama_model_default_params();
        if (params->draft_gpu_layers >= 0) mp.n_gpu_layers = params->draft_gpu_layers;
        // CRITICAL: the MTP head GGUF is nextn-only — the loader SKIPS every
        // blk.N.nextn.* tensor unless load_mtp is set (nextn_flags()/TENSOR_SKIP).
        // Without this the head context build GGML_ASSERTs "MTP block missing"
        // and aborts the process (graph_mtp eh_proj/enorm/hnorm/hc_head_norm null).
        mp.load_mtp = true;
        engine->m_model_dft = llama_model_load_from_file(params->mtp_draft_path, mp);
        if (!engine->m_model_dft) {
            out_error = std::string("MTP head draft failed to load (boundary: "
                                    "spec-engine/mtp-head-load): '") +
                        params->mtp_draft_path +
                        "' — PVX_SPEC_MTP_DRAFT stays disabled";
            return nullptr;
        }
        // The head must be an MTP head: nextn-layer blocks in the GGUF
        // (upstream ctor derives n_mtp_layers from n_layer_nextn and
        // GGML_ASSERTs the qwen4exp single-block contract — here typed).
        const int n_nextn = (int) llama_model_n_layer_nextn(engine->m_model_dft);
        if (n_nextn <= 0) {
            out_error = std::string("MTP head has no nextn layers (n_layer_nextn=0) — not "
                                    "an MTP head GGUF (boundary: spec-engine/mtp-head-nextn): '") +
                        params->mtp_draft_path + "'";
            llama_model_free(engine->m_model_dft);
            engine->m_model_dft = nullptr;
            return nullptr;
        }
        // Pairing row width contract (upstream ctor GGML_ASSERT): the head's
        // output hidden width must equal the target trunk's nextn width.
        engine->m_mtp_n_embd = (int) llama_model_n_embd_out(engine->m_model_dft);
        const int n_embd_tgt = (int) llama_model_n_embd_out(model->model);
        if (engine->m_mtp_n_embd != n_embd_tgt) {
            out_error = "MTP head hidden width (" + std::to_string(engine->m_mtp_n_embd) +
                        ") does not match the target trunk nextn width (" +
                        std::to_string(n_embd_tgt) + ") — verification pairing is "
                        "structurally impossible (boundary: spec-engine/mtp-head-width)";
            llama_model_free(engine->m_model_dft);
            engine->m_model_dft = nullptr;
            return nullptr;
        }
        // Vocab compatibility REQUIRED (draft ids are compared against
        // target samples — same family rule as draft-simple).
        if (!common_speculative_are_compatible(model->model, engine->m_model_dft)) {
            out_error = "MTP head vocab type incompatible with target (boundary: "
                        "spec-engine/mtp-head-vocab)";
            llama_model_free(engine->m_model_dft);
            engine->m_model_dft = nullptr;
            return nullptr;
        }
        // Head context: MTP context type (nextn-layer-only KV), single stream,
        // no recurrent seq budget; n_ctx mirrors the target. BL-42: ctx_other
        // is ALWAYS linked to the target trunk (upstream speculative.cpp ctor
        // sets cparams.ctx_other = ctx_tgt unconditionally) — self-contained
        // heads (qwen4exp class) ignore it; gemma4-assistant REQUIRES it
        // (shared-memory arch refuses standalone init). step35-style
        // multi-block chain heads (n_layer_nextn > 1, BL-45) ride the same
        // link-free context and select their per-step block via
        // llama_set_nextn_layer_offset.
        llama_context_params cp = llama_context_default_params();
        cp.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
        cp.ctx_other = ctx->ctx;
        cp.n_ctx     = ctx->n_ctx;
        cp.n_batch   = std::max<uint32_t>(llama_n_batch(ctx->ctx),
                                          (uint32_t) (engine->m_draft_max + 1));
        cp.n_ubatch  = cp.n_batch;
        cp.n_seq_max = 1;
        cp.n_rs_seq  = 0;
        cp.flash_attn_type = model->flash_attn ? LLAMA_FLASH_ATTN_TYPE_ENABLED
                                               : LLAMA_FLASH_ATTN_TYPE_DISABLED;
        engine->m_ctx_dft = llama_init_from_model(engine->m_model_dft, cp);
        if (!engine->m_ctx_dft) {
            out_error = "MTP head context creation failed (boundary: "
                        "spec-engine/mtp-head-context)";
            llama_model_free(engine->m_model_dft);
            engine->m_model_dft = nullptr;
            return nullptr;
        }
        // Target trunk nextn embeddings on (compute flag only, KV-neutral);
        // head-side nextn output on (masked — the chain pairs read it).
        llama_set_embeddings_nextn(ctx->ctx, true, /*masked=*/ false);
        llama_set_embeddings_nextn(engine->m_ctx_dft, true, /*masked=*/ true);
        // BL-42: upstream-faithful shared-memory detection — the head sees the
        // trunk KV through ctx_other (gemma4-assistant); plain heads ignore it.
        engine->m_mtp_mem_shared = (llama_get_ctx_other(engine->m_ctx_dft) == ctx->ctx);
        if (engine->m_mtp_mem_shared) {
            pvx_log(PVX_LOG_INFO, "Spec",
                    "mtp-draft shared-memory head (ctx_other linked): no catch-up, "
                    "same-position drafts");
        }
        // BL-45 (multi-block chain heads, step35 class): upstream ctor —
        // chain_heads = n_mtp_layers > 1 && !is_mem_shared; the chain budget
        // clamps to the trained block count. Chain mode drafts step i under
        // llama_set_nextn_layer_offset(i) (the head context KV holds every
        // nextn block; the MTP graph runs the block at the current offset).
        const PvxBLMtpChainPlan plan =
            pvx_mtp_chain_plan(n_nextn, engine->m_mtp_mem_shared, engine->m_draft_max);
        engine->m_mtp_n_layers    = plan.n_layers;
        engine->m_mtp_chain_heads = plan.chain;
        engine->m_draft_max       = plan.draft_max;
        if (plan.chain) {
            pvx_log(PVX_LOG_INFO, "Spec",
                    "mtp-draft chain head (multi-block): n_layers=%d draft_max clamped to %d",
                    plan.n_layers, plan.draft_max);
        }
        pvx_log(PVX_LOG_INFO, "Spec",
                "mtp-draft session ready: n_nextn=%d n_embd=%d draft_max=%d p_min=%.3f",
                n_nextn, engine->m_mtp_n_embd, engine->m_draft_max, engine->m_p_min);
        return engine;
    }

    // ── PVX_SPEC_DRAFT: load the draft model ──
    if (!params->draft_model_path || params->draft_model_path[0] == '\0') {
        out_error = "PVX_SPEC_DRAFT configured without draft_model_path; a draft-model "
                    "session without a draft model is fail-closed, not silently "
                    "disabled (boundary: spec-engine/draft-config)";
        return nullptr;
    }

    // Load the draft model; GPU offload mirrors the caller's explicit policy.
    llama_model_params mp = llama_model_default_params();
    if (params->draft_gpu_layers >= 0) mp.n_gpu_layers = params->draft_gpu_layers;

    engine->m_model_dft = llama_model_load_from_file(params->draft_model_path, mp);
    if (!engine->m_model_dft) {
        out_error = std::string("draft model failed to load (boundary: spec-engine/"
                                "draft-load): '") + params->draft_model_path +
                                "' — PVX_SPEC_DRAFT stays disabled, no silent fallback";
        return nullptr;
    }

    // Vocab compatibility REQUIRED (upstream draft-simple throws otherwise).
    if (!common_speculative_are_compatible(model->model, engine->m_model_dft)) {
        out_error = "draft model vocab type incompatible with target (boundary: "
                    "spec-engine/draft-vocab): verification cannot compare draft ids "
                    "against target samples across vocab families";
        llama_model_free(engine->m_model_dft);
        engine->m_model_dft = nullptr;
        return nullptr;
    }

    // Draft context: single stream sized so a draft chain always fits; same
    // attention policy as the target.
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx     = ctx->n_ctx;
    cp.n_batch   = std::max<uint32_t>(llama_n_batch(ctx->ctx), (uint32_t) (1 + engine->m_draft_max));
    cp.n_ubatch  = cp.n_batch;
    cp.n_seq_max = 1;
    cp.flash_attn_type = model->flash_attn ? LLAMA_FLASH_ATTN_TYPE_ENABLED
                                           : LLAMA_FLASH_ATTN_TYPE_DISABLED;

    engine->m_ctx_dft = llama_init_from_model(engine->m_model_dft, cp);
    if (!engine->m_ctx_dft) {
        out_error = "draft context creation failed (boundary: spec-engine/draft-context): "
                    "draft model loaded but its KV/context could not be constructed";
        llama_model_free(engine->m_model_dft);
        engine->m_model_dft = nullptr;
        return nullptr;
    }

    pvx_log(PVX_LOG_INFO, "Spec", "draft session enabled: draft_max=%d draft_min=%d p_min=%.3f",
            engine->m_draft_max, engine->m_draft_min, engine->m_p_min);
    return engine;
}

// ── Stream binding (ABI v7 feature 5) ──────────────────────────────────────

void PvxBLSpecEngine::bind(pvx_executor_t* exec) {
    m_stream.ctx           = exec->ctx_ref;
    m_stream.seq           = 0;
    m_stream.n_tokens      = &exec->n_tokens;
    m_stream.n_kv_pos      = &exec->n_kv_pos;
    m_stream.recent_tokens = &exec->recent_tokens;
    m_stream.last_logits   = &exec->last_logits;
    m_stream.mirror        = &exec->mirror;
    m_stream.lane          = "executor";
}

void PvxBLSpecEngine::bind_conversation(pvx_conversation_t* conv) {
    m_stream.ctx           = conv->ctx_ref;
    m_stream.seq           = conv->seq_id;
    m_stream.n_tokens      = &conv->n_tokens;
    m_stream.n_kv_pos      = &conv->n_kv_pos;
    m_stream.recent_tokens = &conv->recent_tokens;
    m_stream.last_logits   = &conv->last_logits;
    m_stream.mirror        = &conv->mirror;
    m_stream.lane          = "conversation";
}

void PvxBLSpecEngine::stream_push_token(llama_token id, llama_pos pos) {
    pvx_mirror_entry_t ent;
    ent.id = id;
    ent.pos = { pos, pos, pos, pos };
    m_stream.mirror->push_back(std::move(ent));
}

PvxBLSpecEngine::~PvxBLSpecEngine() {
    // ABI v8 wave 4: drop the target-side nextn extraction flag we set at
    // create time — compute-side cleanup only, the target's KV/state are
    // untouched.
    if (m_mode == PVX_SPEC_MTP_DRAFT && m_ctx_tgt && m_ctx_tgt->ctx) {
        llama_set_embeddings_nextn(m_ctx_tgt->ctx, false, /*masked=*/ false);
    }
    // EAGLE3: drop the target-side extraction flags we set at create time —
    // compute-side cleanup only, the target's KV/state are untouched.
    if (m_mode == PVX_SPEC_EAGLE3 && m_ctx_tgt && m_ctx_tgt->ctx && m_model_dft) {
        for (int k = 0; k < m_eagle_n_layers && k < 4; k++) {
            if (m_eagle_layer_ids[k] >= 0 &&
                m_eagle_layer_ids[k] < m_eagle_n_layer_tgt) {
                llama_set_embeddings_layer_inp(m_ctx_tgt->ctx,
                                               (uint32_t) m_eagle_layer_ids[k], false);
            }
        }
    }
    if (m_mode == PVX_SPEC_NGRAM_CACHE && !m_ng_cache_path.empty() &&
        m_ng_cache_ctx && m_ng_cache_dynamic) {
        // Persist the session's caches back to the configured file (upstream
        // saves the dynamic cache at exit): merge dynamic over static into a
        // combined cache. Save is best-effort telemetry — the typed boundary
        // is the load side (fail-closed at spec_init).
        pvx::ngram_cache combined;
        pvx::ngram_cache_merge(combined, *m_ng_cache_static);
        pvx::ngram_cache_merge(combined, *m_ng_cache_ctx);
        pvx::ngram_cache_merge(combined, *m_ng_cache_dynamic);
        pvx::ngram_cache_save(combined, m_ng_cache_path);
    }
    if (m_ctx_dft) {
        llama_free(m_ctx_dft);
        m_ctx_dft = nullptr;
    }
    if (m_model_dft) {
        llama_model_free(m_model_dft);
        m_model_dft = nullptr;
    }
}

bool PvxBLSpecEngine::draft_decode(PvxBLSpecEngine& eng, llama_token tok, llama_pos pos) {
    auto* dft = eng.m_ctx_dft;
    llama_batch b = llama_batch_init(1, 0, 1);
    b.token[0]     = tok;
    b.pos[0]       = pos;
    b.n_seq_id[0]  = 1;
    b.seq_id[0][0] = eng.m_stream.seq;
    b.logits[0]    = 1;
    b.n_tokens     = 1;
    const int rc = llama_decode(dft, b);
    llama_batch_free(b);
    if (rc != 0) {
        pvx_log(PVX_LOG_ERROR, "Spec", "draft ctx decode failed at pos=%d", (int) pos);
        return false;
    }
    return true;
}

// ── Mirror-consistency guard (ABI v6) ─────────────────────────────────────

bool PvxBLSpecEngine::mirror_consistent(std::string& why) const {
    const auto& mirror = *m_stream.mirror;
    if (mirror.size() != (size_t) *m_stream.n_tokens) {
        why = "spec-engine/mirror-consistency: mirror ring size (" +
              std::to_string(mirror.size()) + ") != committed token count (" +
              std::to_string(*m_stream.n_tokens) + ") — a commit path bypassed the ring; "
              "refusing to draft against a phantom stream";
        return false;
    }
    if (mirror.empty()) {
        return true;
    }
    if (mirror.front().pos[0] != *m_stream.n_kv_pos - *m_stream.n_tokens) {
        why = "spec-engine/mirror-consistency: mirror front position (" +
              std::to_string(mirror.front().pos[0]) + ") != window base (" +
              std::to_string(*m_stream.n_kv_pos - *m_stream.n_tokens) + ")";
        return false;
    }
    if (mirror.back().pos[0] != *m_stream.n_kv_pos - 1) {
        why = "spec-engine/mirror-consistency: mirror back position (" +
              std::to_string(mirror.back().pos[0]) + ") != last committed position (" +
              std::to_string(*m_stream.n_kv_pos - 1) + ")";
        return false;
    }
    for (size_t i = 1; i < mirror.size(); ++i) {
        if (mirror[i].pos[0] < mirror[i - 1].pos[0]) {
            why = "spec-engine/mirror-consistency: mirror positions not ascending at entry " +
                  std::to_string(i);
            return false;
        }
    }
    return true;
}

void PvxBLSpecEngine::mirror_token_stream(std::vector<llama_token>& out) const {
    out.clear();
    out.reserve(m_stream.mirror->size());
    for (const auto& e : *m_stream.mirror) {
        if (e.id != LLAMA_TOKEN_NULL) {
            out.push_back(e.id);
        }
    }
}

// ── Embd-aware mirror replay (ABI v6) ─────────────────────────────────────
//
// Replaces the plain-token replay: the executor's mirror ring carries token
// rows AND multimodal embedding rows at pinned positions. The pattern is the
// vendored draft-simple batch builder: token rows feed the draft by id, embd
// rows by pointer; a draft whose input width differs receives a ZERO row plus
// a one-time warning (upstream zeros_warned) — quality signal only, never an
// abort. Positions replay 1:1 (pinned per entry, section array set as-is).
bool PvxBLSpecEngine::mirror_prefix() {
    
    const int pos_base = *m_stream.n_kv_pos - *m_stream.n_tokens;

    const bool incremental = (m_mirror_base == pos_base && m_mirror_pos <= *m_stream.n_kv_pos);
    llama_memory_t dmem = llama_get_memory(m_ctx_dft);
    if (!dmem || !llama_memory_seq_rm(dmem, 0, incremental ? m_mirror_pos : 0, -1)) {
        pvx_log(PVX_LOG_ERROR, "Spec", "draft mirror: seq_rm refused (rebuild=%d)", !incremental);
        return false;
    }
    pvx_log(PVX_LOG_DEBUG, "Spec", "mirror start: rebuild=%d watermark=%d n_kv_pos=%d n_tokens=%d",
            !incremental, m_mirror_pos, *m_stream.n_kv_pos, *m_stream.n_tokens);

    const auto& ring = *m_stream.mirror;
    const int begin  = incremental ? m_mirror_pos : pos_base;
    if (begin < pos_base || begin > *m_stream.n_kv_pos) {
        m_last_error = "spec-engine/mirror-consistency: watermark outside the committed window";
        return false;
    }
    const int n_fill = *m_stream.n_kv_pos - begin;
    if (n_fill <= 0) {
        m_mirror_pos  = *m_stream.n_kv_pos;
        m_mirror_base = pos_base;
        return true;
    }

    // Draft input width for the zero-pad mismatch path (upstream: draft model
    // n_embd; mismatch => zeros + one-time warn).
    const int dft_width = llama_model_n_embd_inp(m_model_dft);
    std::vector<float> zeros;

    const size_t first = (size_t) (begin - pos_base);
    size_t filled = 0;
    const uint32_t d_batch = llama_n_batch(m_ctx_dft);
    while (filled < (size_t) n_fill) {
        const int slice = std::min<int>((int) d_batch, n_fill - (int) filled);
        llama_batch_ext* ext = llama_batch_ext_init(m_ctx_dft);
        bool ok = ext != nullptr;
        for (int i = 0; ok && i < slice; i++) {
            const pvx_mirror_entry_t& e = ring[first + filled + i];
            int32_t idx = -1;
            if (e.id != LLAMA_TOKEN_NULL) {
                idx = llama_batch_ext_add_token(ext, 0, e.id);
            } else {
                // embd row: zero-pad on width mismatch (upstream behavior —
                // warn once, keep positions contiguous, never abort).
                const float* row = e.embd.data();
                if ((int) e.embd.size() != dft_width) {
                    if (!m_width_warned) {
                        m_width_warned = true;
                        pvx_log(PVX_LOG_WARN, "Spec",
                                "target embeddings of size %zu do not fit the draft input "
                                "width %d; the draft receives zero rows for them and "
                                "drafting after multimodal input will be poor",
                                e.embd.size(), dft_width);
                    }
                    if ((int) zeros.size() != dft_width) zeros.assign((size_t) dft_width, 0.0f);
                    row = zeros.data();
                }
                const llama_embd le = { row, 1, (size_t) dft_width };
                idx = llama_batch_ext_add_embd(ext, 0, le);
            }
            if (idx < 0 || !llama_batch_ext_set_pos(ext, idx, e.pos.data())) {
                ok = false;
                break;
            }
        }
        if (ok) {
            const int rc = llama_process(m_ctx_dft, LLAMA_PROCESS_TYPE_DECODE, ext);
            if (rc != 0) ok = false;
        }
        llama_batch_ext_free(ext);
        if (!ok) {
            pvx_log(PVX_LOG_ERROR, "Spec", "draft mirror decode failed (%d rows)", slice);
            return false;
        }
        filled += (size_t) slice;
    }
    m_mirror_pos  = *m_stream.n_kv_pos;
    m_mirror_base = pos_base;
    pvx_log(PVX_LOG_DEBUG, "Spec", "mirror ok: filled [%d, %d) watermark=%d",
            begin, *m_stream.n_kv_pos, m_mirror_pos);
    return true;
}

// ── n-gram drafting (ABI v6) ──────────────────────────────────────────────

int PvxBLSpecEngine::ngram_draft(std::vector<llama_token>& draft, int max_draft,
                                 llama_token sampled) {
    draft.clear();
    if (max_draft <= 0) return 0;
    std::vector<llama_token> stream;
    mirror_token_stream(stream);

    switch (m_mode) {
        case PVX_SPEC_NGRAM_SIMPLE: {
            draft = pvx::ngram_simple_draft(m_ng_simple, stream, sampled);
            break;
        }
        case PVX_SPEC_NGRAM_MAP_K: {
            auto& map = *m_ng_map;
            // (Re)build the map index when the stream shrank (rewind) or on
            // first use — upstream calls begin() once per generation.
            if (stream.size() < map.idx_last_check ||
                (map.keys.empty() && map.key_map_last_idx == 0 && map.idx_last_check == 0)) {
                pvx::ngram_map_begin(map, stream);
            }
            pvx::ngram_map_draft(map, stream, sampled, draft);
            break;
        }
        case PVX_SPEC_NGRAM_MOD: {
            auto& mod = *m_ng_mod;
            const size_t n = mod.get_n();
            const size_t cur_len = stream.size();
            if (cur_len < n + 1) return 0;
            // feed new n-grams in chunks (upstream: 32-token hysteresis)
            if (m_ng_mod_i_last + 32 < cur_len) {
                for (size_t i = m_ng_mod_i_last; i + n < cur_len; ++i) {
                    mod.add(stream.data() + i);
                }
                m_ng_mod_i_last = cur_len - n;
                const double f = (double) mod.get_used() / (double) mod.size();
                if (f > 0.25) {
                    pvx_log(PVX_LOG_WARN, "Spec",
                            "ngram mod occupancy %.2f exceeds threshold (0.25) — resetting", f);
                    mod.reset();
                    m_ng_mod_i_last = cur_len - n;
                }
            }
            const int n_max_cont = std::min(m_draft_max, kNgModMaxDefault);
            std::vector<llama_token> result(n + (size_t) n_max_cont);
            for (size_t i = 0; i + 1 < n; ++i) {
                result[i] = stream[cur_len - n + 1 + i];
            }
            result[n - 1] = sampled;
            bool cleared = false;
            for (int i = 0; i < n_max_cont; ++i) {
                const llama_token token = mod.get(result.data() + i);
                if (token == pvx::ngram_mod::EMPTY) {
                    if (i < m_draft_min) {
                        cleared = true;
                    } else {
                        result.resize(n + (size_t) i);
                    }
                    break;
                }
                result[n + (size_t) i] = token;
            }
            if (!cleared) {
                if (result.size() > n) {
                    // only the drafted continuation tokens
                    draft.assign(result.begin() + (long) n, result.end());
                } else {
                    draft.clear();
                }
                m_ng_mod_n_draft = draft.size();
            } else {
                draft.clear();
                m_ng_mod_n_draft = 0;
            }
            break;
        }
        case PVX_SPEC_NGRAM_CACHE: {
            // draft[0] = sampled seed (upstream contract: draft arrives with
            // the sampled tail, cache_draft appends continuations).
            draft.assign(1, sampled);
            pvx::ngram_cache_draft(stream, draft, max_draft, m_ng_cache_n_min,
                                   m_ng_cache_n_max, *m_ng_cache_ctx,
                                   *m_ng_cache_dynamic, *m_ng_cache_static);
            break;
        }
        default:
            return 0;
    }

    if ((int) draft.size() > max_draft) {
        draft.resize((size_t) max_draft);
    }
    return (int) draft.size();
}

void PvxBLSpecEngine::ngram_after_round(const std::vector<llama_token>& stream,
                                        int n_accepted) {
    switch (m_mode) {
        case PVX_SPEC_NGRAM_MAP_K:
            if (m_ng_map) {
                pvx::ngram_map_accept(*m_ng_map, (uint16_t) std::max(0, n_accepted));
            }
            break;
        case PVX_SPEC_NGRAM_MOD:
            // Acceptance-fraction heuristics (upstream n_low rung) are a
            // quality-only adaptive reset; the drafter stays deterministic
            // without them, so the pattern-port keeps the table semantics.
            break;
        case PVX_SPEC_NGRAM_CACHE: {
            if (!m_ng_cache_ctx) break;
            std::vector<llama_token> snap = stream; // update needs a mutable ref
            if (stream.size() > m_ng_cache_fed_ctx) {
                const int nnew = (int) (stream.size() - m_ng_cache_fed_ctx);
                pvx::ngram_cache_update(*m_ng_cache_ctx, m_ng_cache_n_min,
                                        m_ng_cache_n_max, snap, nnew);
                m_ng_cache_fed_ctx = stream.size();
            }
            // dynamic cache: generated tokens only (beyond the first-round
            // prompt baseline; SIZE_MAX = not yet baselined)
            if (m_ng_cache_fed_dyn == kNgCacheNoBaseline) {
                m_ng_cache_fed_dyn = stream.size();
            } else if (stream.size() > m_ng_cache_fed_dyn) {
                const int nnew = (int) (stream.size() - m_ng_cache_fed_dyn);
                pvx::ngram_cache_update(*m_ng_cache_dynamic, m_ng_cache_n_min,
                                        m_ng_cache_n_max, snap, nnew);
                m_ng_cache_fed_dyn = stream.size();
            }
            break;
        }
        default:
            break;
    }
}

// ── Generation (upstream draft-simple verify loop, adapted to the pvx
// executor) ────────────────────────────────────────────────────────────────
//
// Entry contract (executor entry point's last_logits guard): the prefill is
// complete — the executor's stream is committed through recent_tokens.back()
// ("prev_last") at absolute position n_kv_pos-1, and exec->last_logits is the
// row AFTER prev_last (prediction for n_kv_pos). This matches the plain loop
// exactly at hand-off (prompt + infer), and the commit path restores the same
// state at round end.
//
// Per round:
//   0. Mirror consistency guard (fail-closed), then for draft-mode
//      mirror_prefix(): draft ctx holds the committed stream for positions
//      [m_mirror_base, m_mirror_pos); stale cells dropped, missing rows
//      decoded (full rebuild when the target state shrank or shifted).
//      n-gram modes skip the draft-context replay (no draft ctx) but run the
//      same guard — their lookup streams are built from the same ring.
//   1. Draft chain:
//      - draft mode: draft[0] from the TARGET's freshest row
//        (exec->last_logits), continuations greedy top-1 on the draft ctx,
//        gated by p_min and EOG.
//      - n-gram modes: lookup over the committed stream (mirror token
//        entries) with the target's top-1 as the tail — pvx_ngram.cpp.
//      Chain bounded by draft_max, the target's batch width, n_ctx room for
//      the commit row, and the remaining output budget.
//   2. verify batch = [draft0..draft_{d-1}] decoded on the TARGET in ONE
//      call, every row logits-enabled. prev_last is NOT re-decoded: it is
//      already committed at n_kv_pos-1.
//   3. sample-and-match with the executor pipeline: s0 = sample(last_logits)
//      vs draft0; s_{i+1} = sample(row_i) vs draft_{i+1}; the first
//      disagreeing sample IS the commit token. Penalty feed sees exactly the
//      accepted prefix in both paths.
//   4. KV repair: seq_rm past the accepted prefix — drops the unverified
//      draft cells; the accepted cells are already at the right positions.
//   5. commit: ONE decode of the commit token, counters/recent_tokens AND
//      the mirror ring advanced exactly like the plain loop's per-token
//      commits (penalty-window + mirror-position equivalence).

pvx_result_t PvxBLSpecEngine::generate(const pvx_sampling_params_t* params,
                                       int out_max,
                                       pvx_spec_token_callback_t on_token,
                                       void* user_data,
                                       int* out_generated) {
    
    auto* ctx  = m_ctx_tgt;
    if (!m_stream.ctx || !ctx || !params) return PVX_ERR_INVALID_ARG;
    if (m_stream.last_logits->empty()) return PVX_ERR_DECODE_FAILED;

    // ABI v7: EAGLE3 runs its own round structure (upstream draft-eagle3:
    // the corrected sample rides as the first row of the verify batch and
    // the deferred boundary advances per round) — one implementation per
    // draft-graph family, exactly like upstream has one impl per type.
    if (m_mode == PVX_SPEC_EAGLE3) {
        return generate_eagle(params, out_max, on_token, user_data, out_generated);
    }

    // ABI v8 wave 4: draft-MTP runs its own round structure (upstream
    // draft-mtp — the head's (t_p, h_{p-1}) pairing; one impl per family).
    if (m_mode == PVX_SPEC_MTP_DRAFT) {
        return generate_mtp(params, out_max, on_token, user_data, out_generated);
    }

    const bool is_ngram = m_mode != PVX_SPEC_DRAFT;
    const int n_vocab = llama_vocab_n_tokens(ctx->vocab);
    // EOG terminates the stream exactly like the plain pipeline: unless the
    // caller asked to sample past EOG (ignore_eos). The sampler seam already
    // honors ignore_eos (spec_sample == do_sample); the loop's termination
    // checks must honor it too or spec ON would stop where plain greedy
    // continues (token-exactness breaks for the whole emitted stream).
    const bool ignore_eos = params->ignore_eos;
    auto stop_at_eog = [&](llama_token t) {
        return !ignore_eos && llama_vocab_is_eog(ctx->vocab, t);
    };
    // Chain bound (upstream n_draft_max rule): remaining budget minus 1 so the
    // commit token always fits; never wider than the target's batch.
    int chain_cap = std::min(m_draft_max,
                             std::min((int) llama_n_batch(ctx->ctx) - 1,
                                      out_max - 1));
    // Hybrid/recurrent target: the KV repair rewinds the recurrent state via
    // trailing snapshots, bounded by n_rs_seq — upstream's server rule (drafts
    // must not exceed llama_n_rs_seq(ctx_tgt); server-context.cpp). Upstream
    // never seq_rm-repairs a recurrent target: it checkpoint/restores instead
    // (server-context.cpp:4562-4576) — pvx does not port that path, so with no
    // snapshot ring (pvx default) drafting is bounded to zero: every round
    // falls through to the plain single-token step. Output stays token-exact,
    // generation continues — poor drafting, correct stream (the documented
    // wave-1 posture for width/recurrent-degraded drafting).
    if (llama_model_is_recurrent(m_model_tgt->model) ||
        llama_model_is_hybrid(m_model_tgt->model)) {
        const int rs_cap = (int) llama_n_rs_seq(ctx->ctx);
        if (rs_cap < chain_cap) {
            pvx_log(PVX_LOG_WARN, "Spec",
                    "drafting disabled: hybrid/recurrent target memory cannot roll back "
                    "verify batches (llama_n_rs_seq=%d, upstream checkpoint-repair not wired); "
                    "plain exact steps only", rs_cap);
        }
        chain_cap = std::min(chain_cap, rs_cap);
    }
    if (chain_cap < 0) chain_cap = 0; // tiny budget/batch: drafting off, fallback steps still run

    int generated = 0;

    m_n_drafted = 0;
    m_n_accepted = 0;
    while (generated < out_max) {
        // 0. ── Mirror consistency guard (fail-closed, ABI v6) ──
        {
            std::string why;
            if (!mirror_consistent(why)) {
                m_last_error = why;
                ctx->last_error = why;
                pvx_log(PVX_LOG_ERROR, "Spec", "%s", why.c_str());
                return PVX_ERR_INTERNAL;
            }
        }
        // ── Draft-context replay (draft mode only) ──
        if (!is_ngram && !mirror_prefix()) return PVX_ERR_DECODE_FAILED;

        const int n_remaining = out_max - generated;
        const int max_draft   = std::min(chain_cap, n_remaining - 1);
        const int round_base  = *m_stream.n_kv_pos;

        // 1. ── Draft chain ──
        std::vector<llama_token> draft;
        if (max_draft > 0) {
            const float* last_logits = m_stream.last_logits->data();
            const TopLogit first_top = drafter_top_logit(last_logits, n_vocab);
            const bool first_usable =
                (m_p_min <= 0.0f ||
                 top_probability(last_logits, n_vocab, first_top.logit) >= m_p_min) &&
                !llama_vocab_is_eog(ctx->vocab, first_top.token);
            if (first_usable) {
                if (is_ngram) {
                    ngram_draft(draft, max_draft, first_top.token);
                    m_n_drafted += (int64_t) draft.size();
                    if (m_model_tgt && m_model_tgt->audit.enabled) m_model_tgt->audit.spec_drafted += (int64_t) draft.size();  // ABI v9
                } else {
                    draft.push_back(first_top.token);
                    m_n_drafted++;
                    if (m_model_tgt && m_model_tgt->audit.enabled) m_model_tgt->audit.spec_drafted++;  // ABI v9
                    // Draft token i sits at absolute position n_kv_pos + i.
                    llama_pos next_pos = *m_stream.n_kv_pos;
                    for (int i = 1; i <= max_draft; i++) {
                        if (!draft_decode(*this, draft.back(), next_pos)) {
                            return PVX_ERR_INTERNAL;
                        }
                        if (i == max_draft) break; // chain complete — no extra decode
                        next_pos++;
                        const float* d_logits = llama_get_logits_ith(m_ctx_dft, 0);
                        if (!d_logits) return PVX_ERR_INTERNAL;
                        const TopLogit d_top = drafter_top_logit(d_logits, n_vocab);
                        if (m_p_min > 0.0f &&
                            top_probability(d_logits, n_vocab, d_top.logit) < m_p_min) break;
                        if (llama_vocab_is_eog(ctx->vocab, d_top.token)) break;
                        draft.push_back(d_top.token);
                        m_n_drafted++;
                        if (m_model_tgt && m_model_tgt->audit.enabled) m_model_tgt->audit.spec_drafted++;  // ABI v9
                    }
                }
            }
        }

        if ((int) draft.size() < m_draft_min) draft.clear();

        if (draft.empty()) {
            // Drafter cannot help here (EOG/p_min floor/budget/no lookup hit):
            // fall through to the plain single-token step — honest progress
            // with the same sampler. Mirrors the plain loop's commit exactly;
            // the next round's mirror_prefix carries the token to the draft ctx.
            const llama_token t = spec_sample(m_stream.last_logits->data(), ctx->vocab,
                                              params, *m_stream.recent_tokens);
            if (t < 0) return PVX_ERR_DECODE_FAILED;
            llama_batch b = llama_batch_init(1, 0, ctx->n_seq_max);
            b.token[0]     = t;
            b.pos[0]       = *m_stream.n_kv_pos;
            b.n_seq_id[0]  = 1;
            b.seq_id[0][0] = m_stream.seq;
            b.logits[0]    = 1;
            b.n_tokens     = 1;
            const int rc = llama_decode(ctx->ctx, b);
            llama_batch_free(b);
            if (rc != 0) {
                ctx->last_error = "spec fallback decode failed (rc=" + std::to_string(rc) + ")";
                return PVX_ERR_DECODE_FAILED;
            }
            spec_snapshot(ctx->ctx, *m_stream.last_logits, 0);
            stream_push_token(t, *m_stream.n_kv_pos);
            (*m_stream.n_tokens)++;
            (*m_stream.n_kv_pos)++;
            m_stream.recent_tokens->push_back(t);
            generated++;
            if (is_ngram) {
                std::vector<llama_token> stream;
                mirror_token_stream(stream);
                ngram_after_round(stream, 0);
            }
            if (on_token && !on_token(t, user_data)) break;
            if (stop_at_eog(t)) break;
            continue;
        }

        // 2. ── Verify on the target: ONE decode of [draft0..draft_{d-1}] ──
        const int n_draft = (int) draft.size();
        llama_batch b = llama_batch_init(n_draft, 0, ctx->n_seq_max);
        for (int i = 0; i < n_draft; i++) {
            b.token[i]     = draft[i];
            b.pos[i]       = *m_stream.n_kv_pos + i;
            b.n_seq_id[i]  = 1;
            b.seq_id[i][0] = m_stream.seq;
            b.logits[i]    = 1;
        }
        b.n_tokens = n_draft;
        const int rc = llama_decode(ctx->ctx, b);
        llama_batch_free(b);
        if (rc != 0) {
            ctx->last_error = "spec verify decode failed (rc=" + std::to_string(rc) + ")";
            return PVX_ERR_DECODE_FAILED;
        }

        // Snapshot every row immediately (rows die at the next decode).
        std::vector<std::vector<float>> rows(n_draft);
        for (int i = 0; i < n_draft; i++) spec_snapshot(ctx->ctx, rows[i], i);

        // 3. ── Sample-and-match with the executor pipeline ──
        std::vector<llama_token> accepted;
        llama_token commit = 0;
        int  eog_row = -1;
        {
            std::vector<llama_token> feed = *m_stream.recent_tokens; // ends at prev_last
            llama_token s = spec_sample(m_stream.last_logits->data(), ctx->vocab,
                                        params, feed);
            if (s < 0) return PVX_ERR_DECODE_FAILED;
            int i = 0;
            for (; i < n_draft; i++) {
                if (s != draft[i]) break; // first disagreeing sample IS the commit
                accepted.push_back(draft[i]);
                m_n_accepted++;
                if (m_model_tgt && m_model_tgt->audit.enabled) m_model_tgt->audit.spec_accepted++;  // ABI v9
                feed.push_back(draft[i]);
                if (stop_at_eog(draft[i])) {
                    // Mid-chain EOG: accepted, nothing commits past an EOG.
                    eog_row = i;
                    break;
                }
                if (i + 1 < n_draft) {
                    s = spec_sample(rows[i].data(), ctx->vocab, params, feed);
                    if (s < 0) return PVX_ERR_DECODE_FAILED;
                }
            }
            if (i == n_draft && eog_row < 0) {
                // Fully accepted: the row AFTER the last draft gives the commit
                // token (upstream's tail sample — a fresh sample, never a reuse
                // of the verification samples).
                commit = spec_sample(rows[n_draft - 1].data(), ctx->vocab, params, feed);
                if (commit < 0) return PVX_ERR_DECODE_FAILED;
            } else if (eog_row < 0) {
                commit = s; // first disagreeing sample (row after accepted prefix)
            }
            pvx_log(PVX_LOG_DEBUG, "Spec", "round: draft=%zu accepted=%zu commit=%d feedtail=%d",
                    draft.size(), accepted.size(), (int) commit,
                    feed.empty() ? -1 : (int) feed.back());
        }

        // 4. ── KV repair: keep the accepted prefix (positions 0 … keep − 1) ──
        const int keep = *m_stream.n_kv_pos + (int) accepted.size();
        if (!llama_memory_seq_rm(ctx->mem, m_stream.seq, keep, -1)) {
            ctx->last_error = "spec KV repair refused by memory (partial seq_rm unsupported)";
            return PVX_ERR_REWIND_UNSUPPORTED;
        }

        if (eog_row >= 0) {
            // Mid-chain EOG: the generation ends with the accepted stream (EOG
            // last). The row after the EOG becomes the executor's next-step
            // last_logits — same contract as the plain loop decoding an EOG.
            *m_stream.last_logits = std::move(rows[eog_row]);
            (*m_stream.n_tokens) += (int) accepted.size();
            (*m_stream.n_kv_pos) += (int) accepted.size();
            for (llama_token t : accepted) m_stream.recent_tokens->push_back(t);
            for (int i = 0; i < (int) accepted.size(); i++) {
                stream_push_token(accepted[i], round_base + i);
            }
            if (is_ngram) {
                std::vector<llama_token> stream;
                mirror_token_stream(stream);
                ngram_after_round(stream, (int) accepted.size());
            }
            for (llama_token t : accepted) {
                generated++;
                if (on_token && !on_token(t, user_data)) {
                    if (out_generated) *out_generated = generated;
                    return PVX_OK;
                }
                if (stop_at_eog(t)) {
                    if (out_generated) *out_generated = generated;
                    return PVX_OK;
                }
            }
            continue; // unreachable (EOG always ends the stream) — kept for shape
        }

        // 5. ── Commit: ONE decode of the commit token (logits row becomes the
        //    next-step last_logits, same contract as pvx_executor_infer).
        {
            llama_batch rb = llama_batch_init(1, 0, ctx->n_seq_max);
            rb.token[0]     = commit;
            rb.pos[0]       = keep;
            rb.n_seq_id[0]  = 1;
            rb.seq_id[0][0] = m_stream.seq;
            rb.logits[0]    = 1;
            rb.n_tokens     = 1;
            const int rrc = llama_decode(ctx->ctx, rb);
            llama_batch_free(rb);
            if (rrc != 0) {
                ctx->last_error = "spec commit decode failed (rc=" + std::to_string(rrc) + ")";
                return PVX_ERR_DECODE_FAILED;
            }
            spec_snapshot(ctx->ctx, *m_stream.last_logits, 0);
        }

        (*m_stream.n_tokens) += (int) accepted.size() + 1;
        (*m_stream.n_kv_pos) += (int) accepted.size() + 1;
        for (llama_token t : accepted) m_stream.recent_tokens->push_back(t);
        m_stream.recent_tokens->push_back(commit);
        for (int i = 0; i < (int) accepted.size(); i++) {
            stream_push_token(accepted[i], round_base + i);
        }
        stream_push_token(commit, keep);
        if (is_ngram) {
            std::vector<llama_token> stream;
            mirror_token_stream(stream);
            ngram_after_round(stream, (int) accepted.size());
        }

        // Stream/EOG per committed token (upstream processes ids in order).
        pvx_log(PVX_LOG_DEBUG, "Spec", "round end: n_kv_pos=%d n_tokens=%d committed=%zu",
                *m_stream.n_kv_pos, *m_stream.n_tokens, accepted.size() + 1);
        for (llama_token t : accepted) {
            generated++;
            if (on_token && !on_token(t, user_data)) {
                if (out_generated) *out_generated = generated;
                return PVX_OK;
            }
            if (stop_at_eog(t)) {
                if (out_generated) *out_generated = generated;
                return PVX_OK;
            }
        }
        generated++;
        if (on_token && !on_token(commit, user_data)) break;
        if (stop_at_eog(commit)) break;
    }

    if (out_generated) *out_generated = generated;
    return PVX_OK;
}
// ═══════════════════════════════════════════════════════════════════════════
// ── EAGLE3 (ABI v7 feature 4) — pattern-port of the vendored
// common_speculative_impl_draft_eagle3 process()/draft()/accept()/get_state()
// ═══════════════════════════════════════════════════════════════════════════
//
// Input convention (upstream): the draft decoder row at memory pos P consumes
// the pair (t_{P+1}, g_P) — the NEXT stream token with the EAGLE3 encoder
// output of the target's extracted layer features at P. Draft decoder KV holds
// pairs for [0, boundary); the pair at the boundary position is DEFERRED
// (needs the next sample — the target's freshest corrected token) and is
// completed by the next round's seed (upstream pending_g_last/pending_pos_last,
// here derived from g_hist: pending == (g_hist.back(), pos n_tokens-1)).
//
// Round structure (adapted to the pvx verify pipeline — see generate_eagle):
//   1. c = the target sample from last_logits (the round's always-committed
//      token — upstream's dp.id_last / first verify-batch row).
//   2. seed: draft-decode (c, g_{n_tokens-1}) at pos n_tokens-1 → chain
//      candidates from greedy top-1 with prenorm g stepping (upstream draft()).
//   3. verify: ONE target decode of [c, chain...] at [n_kv_pos, ...]; the
//      eagle capture hook (process() pattern) extracts layer features for
//      every row, runs the encoder, appends g rows and writes the draft's
//      training pairs for rows [0, m-1] (last row deferred).
//   4. sample-and-match against the pipeline; the corrected sample becomes
//      the NEXT round's c (deferred into the next verify batch — identical
//      stream positions to the plain loop's commits).
//   5. KV repair to the committed prefix; g_hist truncated to the committed
//      count (the boundary pair is re-seeded next round, upstream accept()).
//
// KV-neutrality: the draft consumes its OWN KV (own context); the target only
// gains compute-side extraction flags. Target KV mutations are the same
// committed-position advances the plain loop performs — nothing else.

bool PvxBLSpecEngine::eagle_pair_decode(llama_token tok, const float* g_row, llama_pos pos) {
    llama_batch_ext* ext = llama_batch_ext_init(m_ctx_dft);
    if (!ext) return false;
    bool ok = false;
    do {
        const int32_t idx = llama_batch_ext_add_token(ext, 0, tok);
        if (idx < 0) break;
        const llama_embd le = { g_row, 1, (size_t) m_eagle_n_embd_dec };
        if (!llama_batch_ext_set_embd_token(ext, idx, le)) break;
        const llama_pos p[4] = { pos, pos, pos, pos };
        if (!llama_batch_ext_set_pos(ext, idx, p)) break;
        if (!llama_batch_ext_set_output_logits(ext, idx, true)) break;
        const int32_t rc = llama_process(m_ctx_dft, LLAMA_PROCESS_TYPE_DECODE, ext);
        if (rc != 0) {
            pvx_log(PVX_LOG_ERROR, "Spec", "eagle3 draft decode failed at pos=%d", (int) pos);
            break;
        }
        ok = true;
    } while (false);
    llama_batch_ext_free(ext);
    return ok;
}

bool PvxBLSpecEngine::eagle_process_target_rows(llama_pos pos_first, int n_rows,
                                                const llama_token* row_ids) {
    if (m_mode != PVX_SPEC_EAGLE3 || n_rows <= 0) return true;
    if (!m_ctx_dft || !row_ids) return false;
    // The capture hook must fire exactly once per decoded batch, in stream
    // order: g_hist stays aligned with the committed window [0, n_tokens).
    if ((size_t) pos_first != m_eagle_g_hist.size()) {
        m_last_error = "spec-engine/eagle3-capture-order: hook at pos " +
                       std::to_string((int) pos_first) + " but g_hist holds " +
                       std::to_string(m_eagle_g_hist.size()) + " rows";
        pvx_log(PVX_LOG_ERROR, "Spec", "%s", m_last_error.c_str());
        return false;
    }

    auto* tgt = m_ctx_tgt->ctx;

    // Interleave each extract layer's input features into [n_rows, n_embd_enc]
    // (upstream process() features_buf gather).
    m_eagle_features_buf.assign((size_t) n_rows * (size_t) m_eagle_n_embd_enc, 0.0f);
    for (int k = 0; k < m_eagle_n_layers; k++) {
        const int lid = m_eagle_layer_ids[k];
        const float* layer = (lid < m_eagle_n_layer_tgt)
            ? llama_get_embeddings_layer_inp(tgt, (uint32_t) lid)
            : llama_get_embeddings_nextn(tgt);
        if (!layer) {
            m_last_error = "spec-engine/eagle3-capture: target layer " +
                           std::to_string(lid) + " features not extracted";
            pvx_log(PVX_LOG_ERROR, "Spec", "%s", m_last_error.c_str());
            return false;
        }
        for (int i = 0; i < n_rows; i++) {
            memcpy(m_eagle_features_buf.data() + (size_t) i * m_eagle_n_embd_enc +
                       (size_t) k * m_eagle_n_embd_tgt,
                   layer + (size_t) i * m_eagle_n_embd_tgt,
                   (size_t) m_eagle_n_embd_tgt * sizeof(float));
        }
    }

    // EAGLE3 encoder pass on the draft context (upstream: per-token encoder
    // chunked by n_ubatch, output via the draft's pre-norm embeddings buffer).
    m_eagle_g_buf.assign((size_t) n_rows * (size_t) m_eagle_n_embd_dec, 0.0f);
    const int n_ubatch = (int) llama_n_ubatch(m_ctx_dft);
    for (int off = 0; off < n_rows; off += n_ubatch) {
        const int chunk = std::min(n_ubatch, n_rows - off);
        llama_batch_ext* enc = llama_batch_ext_init(m_ctx_dft);
        if (!enc) return false;
        bool ok = false;
        do {
            // placeholder positions from the draft memory state (upstream
            // comment: the per-token encoder does not use positions)
            llama_pos pos = llama_memory_seq_pos_max(llama_get_memory(m_ctx_dft), 0) + 1;
            bool rows_ok = true;
            for (int j = 0; j < chunk; j++) {
                const llama_embd le = {
                    m_eagle_features_buf.data() + (size_t) (off + j) * m_eagle_n_embd_enc,
                    1, (size_t) m_eagle_n_embd_enc };
                const int32_t idx = llama_batch_ext_add_embd(enc, 0, le);
                if (idx < 0) { rows_ok = false; break; }
                const llama_pos p[4] = { pos, pos, pos, pos };
                if (!llama_batch_ext_set_pos(enc, idx, p) ||
                    !llama_batch_ext_set_output_embd(enc, idx, true)) {
                    rows_ok = false; break;
                }
                pos++;
            }
            if (!rows_ok) break;
            if (llama_process(m_ctx_dft, LLAMA_PROCESS_TYPE_ENCODE, enc) != 0) break;
            const float* g_chunk = llama_get_embeddings_nextn(m_ctx_dft);
            if (!g_chunk) break;
            memcpy(m_eagle_g_buf.data() + (size_t) off * m_eagle_n_embd_dec, g_chunk,
                   (size_t) chunk * (size_t) m_eagle_n_embd_dec * sizeof(float));
            ok = true;
        } while (false);
        llama_batch_ext_free(enc);
        if (!ok) {
            m_last_error = "spec-engine/eagle3-encoder: encoder pass failed";
            pvx_log(PVX_LOG_ERROR, "Spec", "%s", m_last_error.c_str());
            return false;
        }
    }

    // One g row per committed position (upstream pending_g_last/verify_g are
    // snapshots of these rows; here the ring itself is the source of truth).
    m_eagle_g_hist.reserve(m_eagle_g_hist.size() + (size_t) n_rows);
    for (int i = 0; i < n_rows; i++) {
        m_eagle_g_hist.emplace_back(
            m_eagle_g_buf.data() + (size_t) i * m_eagle_n_embd_dec,
            m_eagle_g_buf.data() + (size_t) (i + 1) * m_eagle_n_embd_dec);
    }

    // Decoder training pairs: (t_{P+1}, g_P) at pos P for rows [0, n_rows-1);
    // the LAST row is deferred (needs the next sample). Cross-batch bridge:
    // the previous batch's deferred pair at pos_first-1 consumes THIS batch's
    // first token — skipped when the seed already completed it (upstream's
    // pending_pos > dft_pos_max check, verbatim pattern).
    llama_batch_ext* dec = llama_batch_ext_init(m_ctx_dft);
    if (!dec) return false;
    bool ok = false;
    int n_pairs = 0; // empty batch = deferred boundary only (n_rows == 1,
                     // bridge skipped): upstream guards batch.size() > 0 before
                     // llama_process — decoding 0 rows is an error (live-verified
                     // on the Qwen3-8B pair's final plain_step round).
    do {
        llama_memory_t dmem = llama_get_memory(m_ctx_dft);
        if (!dmem) break;
        // Drop the drafted-but-unverified chain cells [pos_first, ...) before
        // the training batch rewrites those positions from the verified
        // stream: upstream restores the draft context to the committed prefix
        // before process() (TAG_SPEC_AVOID_DRAFT_REEVAL — always re-evaluate).
        // The seed cell at pos_first-1 STAYS, so the bridge condition below
        // (pos_first-1 > dft_pos_max) correctly skips. Without the drop, the
        // training batch would start at pos_first while the draft KV still
        // ends at the chain tail — llama's Y = X+1 consecutiveness check
        // aborts the decode (live-verified on the Qwen3-8B pair: draft KV
        // pos_max 30 vs training batch start 24).
        if (!llama_memory_seq_rm(dmem, 0, (llama_pos) pos_first, -1)) break;
        if (pos_first > 0 &&
            (llama_pos) (pos_first - 1) > llama_memory_seq_pos_max(dmem, 0)) {
            const int32_t idx = llama_batch_ext_add_token(dec, 0, row_ids[0]);
            const llama_embd le = {
                m_eagle_g_hist[(size_t) (pos_first - 1)].data(), 1,
                (size_t) m_eagle_n_embd_dec };
            if (idx < 0 || !llama_batch_ext_set_embd_token(dec, idx, le)) break;
            const llama_pos p[4] = { pos_first - 1, pos_first - 1, pos_first - 1, pos_first - 1 };
            if (!llama_batch_ext_set_pos(dec, idx, p)) break;
            n_pairs++;
        }
        for (int k = 0; k + 1 < n_rows; k++) {
            const int32_t idx = llama_batch_ext_add_token(dec, 0, row_ids[k + 1]);
            const llama_embd le = {
                m_eagle_g_hist[(size_t) (pos_first + k)].data(), 1,
                (size_t) m_eagle_n_embd_dec };
            if (idx < 0 || !llama_batch_ext_set_embd_token(dec, idx, le)) break;
            const llama_pos pk = pos_first + k;
            const llama_pos p[4] = { pk, pk, pk, pk };
            if (!llama_batch_ext_set_pos(dec, idx, p)) break;
            if (!llama_batch_ext_set_output_logits(dec, idx, false)) break;
            n_pairs++;
        }
        if (n_pairs > 0 && llama_process(m_ctx_dft, LLAMA_PROCESS_TYPE_DECODE, dec) != 0) break;
        ok = true;
    } while (false);
    llama_batch_ext_free(dec);
    if (!ok) {
        m_last_error = "spec-engine/eagle3-training: decoder training write failed";
        pvx_log(PVX_LOG_ERROR, "Spec", "%s", m_last_error.c_str());
        return false;
    }
    pvx_log(PVX_LOG_DEBUG, "Spec", "eagle3 process: pos_first=%d n_rows=%d g_hist=%zu",
            (int) pos_first, n_rows, m_eagle_g_hist.size());
    return true;
}

void PvxBLSpecEngine::eagle_on_stream_shrunk(int new_n_tokens) {
    if (m_mode != PVX_SPEC_EAGLE3) return;
    if ((int) m_eagle_g_hist.size() > new_n_tokens) {
        m_eagle_g_hist.resize((size_t) std::max(0, new_n_tokens));
    }
    if (m_ctx_dft) {
        llama_memory_t dmem = llama_get_memory(m_ctx_dft);
        if (dmem) {
            const llama_pos boundary = (llama_pos) std::max(0, new_n_tokens - 1);
            llama_memory_seq_rm(dmem, 0, boundary, -1);
        }
    }
    // mirror watermark invalidation (draft-side only — the target's own
    // rewind already ran; the draft rebuilds from its g_hist/eagle path)
    m_mirror_pos  = new_n_tokens;
    m_mirror_base = 0;
}

void PvxBLSpecEngine::eagle_state_capture(std::vector<std::vector<float>>& out_rows) const {
    if (m_mode != PVX_SPEC_EAGLE3) return;
    out_rows = m_eagle_g_hist;
}

bool PvxBLSpecEngine::eagle_state_restore(std::vector<std::vector<float>>&& rows) {
    if (m_mode != PVX_SPEC_EAGLE3) return false;
    const int n = m_stream.n_tokens ? *m_stream.n_tokens : 0;
    if ((int) rows.size() != n || !m_stream.mirror ||
        (size_t) n != m_stream.mirror->size()) {
        m_last_error = "spec-engine/eagle3-state-mismatch: saved g rows (" +
                       std::to_string(rows.size()) + ") do not cover the restored "
                       "stream (" + std::to_string(n) + " tokens) — the EAGLE3 "
                       "session must be re-initialized after this restore";
        pvx_log(PVX_LOG_WARN, "Spec", "%s", m_last_error.c_str());
        return false;
    }
    for (const auto& e : *m_stream.mirror) {
        if (e.id == LLAMA_TOKEN_NULL) {
            m_last_error = "spec-engine/eagle3-state-restore: mirror carries "
                           "multimodal rows (vision + EAGLE3 is a typed refusal)";
            return false;
        }
    }
    // Full draft-KV clear, then training-pair replay [0, n-1) — the pair at
    // n-1 stays deferred (the next round's seed completes it, upstream
    // single-ubatch-prefill semantics).
    llama_memory_t dmem = llama_get_memory(m_ctx_dft);
    if (dmem) llama_memory_seq_rm(dmem, 0, 0, -1);
    m_eagle_g_hist = std::move(rows);
    const uint32_t d_batch = llama_n_batch(m_ctx_dft);
    for (int off = 0; off + 1 < n; off += (int) d_batch) {
        const int slice = std::min<int>((int) d_batch, n - 1 - off);
        llama_batch_ext* ext = llama_batch_ext_init(m_ctx_dft);
        if (!ext) return false;
        bool ok = true;
        for (int j = 0; ok && j < slice; j++) {
            const int k = off + j;
            const int32_t idx = llama_batch_ext_add_token(ext, 0, (*m_stream.mirror)[k + 1].id);
            if (idx < 0) { ok = false; break; }
            const llama_embd le = {
                m_eagle_g_hist[(size_t) k].data(), 1, (size_t) m_eagle_n_embd_dec };
            if (!llama_batch_ext_set_embd_token(ext, idx, le)) { ok = false; break; }
            const llama_pos p[4] = { (llama_pos) k, (llama_pos) k,
                                     (llama_pos) k, (llama_pos) k };
            if (!llama_batch_ext_set_pos(ext, idx, p)) { ok = false; break; }
        }
        if (ok && llama_process(m_ctx_dft, LLAMA_PROCESS_TYPE_DECODE, ext) != 0) ok = false;
        llama_batch_ext_free(ext);
        if (!ok) {
            m_last_error = "spec-engine/eagle3-state-restore: draft replay decode failed";
            pvx_log(PVX_LOG_ERROR, "Spec", "%s", m_last_error.c_str());
            return false;
        }
    }
    m_mirror_pos  = n;
    m_mirror_base = 0;
    return true;
}

// ── EAGLE3 round loop (see the block comment above for the convention) ─────

pvx_result_t PvxBLSpecEngine::generate_eagle(const pvx_sampling_params_t* params,
                                             int out_max,
                                             pvx_spec_token_callback_t on_token,
                                             void* user_data,
                                             int* out_generated) {
    auto* ctx = m_ctx_tgt;
    const int n_vocab = llama_vocab_n_tokens(ctx->vocab);
    const bool ignore_eos = params->ignore_eos;
    auto stop_at_eog = [&](llama_token t) {
        return !ignore_eos && llama_vocab_is_eog(ctx->vocab, t);
    };

    // Chain bound: batch width + budget, minus the always-present c row.
    int chain_cap = std::min(m_draft_max,
                             std::min((int) llama_n_batch(ctx->ctx) - 2,
                                      out_max - 2));
    if (llama_model_is_recurrent(m_model_tgt->model) ||
        llama_model_is_hybrid(m_model_tgt->model)) {
        const int rs_cap = (int) llama_n_rs_seq(ctx->ctx);
        if (rs_cap < chain_cap) {
            pvx_log(PVX_LOG_WARN, "Spec",
                    "eagle3 drafting disabled: hybrid/recurrent target memory cannot "
                    "roll back verify batches (llama_n_rs_seq=%d); plain exact steps only",
                    rs_cap);
        }
        chain_cap = std::min(chain_cap, rs_cap - 1);
    }
    if (chain_cap < 0) chain_cap = 0;

    int generated = 0;
    m_n_drafted = 0;
    m_n_accepted = 0;

    // Per-round plain single-token step (chain unusable / EOG / degraded
    // target): identical commit to the plain loop + the capture hook.
    // stopped=true when the stream ended (EOG committed or callback abort).
    bool stopped = false;
    auto plain_step = [&](llama_token t) -> pvx_result_t {
        const llama_pos pos = *m_stream.n_kv_pos;
        llama_batch b = llama_batch_init(1, 0, ctx->n_seq_max);
        b.token[0]     = t;
        b.pos[0]       = pos;
        b.n_seq_id[0]  = 1;
        b.seq_id[0][0] = m_stream.seq;
        b.logits[0]    = 1;
        b.n_tokens     = 1;
        const int rc = llama_decode(ctx->ctx, b);
        llama_batch_free(b);
        if (rc != 0) {
            ctx->last_error = "eagle3 fallback decode failed (rc=" + std::to_string(rc) + ")";
            return PVX_ERR_DECODE_FAILED;
        }
        if (!eagle_process_target_rows(pos, 1, &t)) return PVX_ERR_INTERNAL;
        spec_snapshot(ctx->ctx, *m_stream.last_logits, 0);
        stream_push_token(t, *m_stream.n_kv_pos);
        (*m_stream.n_tokens)++;
        (*m_stream.n_kv_pos)++;
        m_stream.recent_tokens->push_back(t);
        generated++;
        if (on_token && !on_token(t, user_data)) { stopped = true; return PVX_OK; }
        if (stop_at_eog(t)) { stopped = true; return PVX_OK; }
        return PVX_OK;
    };

    while (generated < out_max) {
        // Capture-order invariant: one g row per committed position.
        if (m_eagle_g_hist.size() != (size_t) *m_stream.n_tokens) {
            m_last_error = "spec-engine/eagle3-capture-order: g_hist (" +
                           std::to_string(m_eagle_g_hist.size()) + ") != committed (" +
                           std::to_string(*m_stream.n_tokens) + ")";
            ctx->last_error = m_last_error;
            pvx_log(PVX_LOG_ERROR, "Spec", "%s", m_last_error.c_str());
            return PVX_ERR_INTERNAL;
        }

        // 1. ── c: the target's sample from last_logits ──
        const llama_token c = spec_sample(m_stream.last_logits->data(), ctx->vocab,
                                          params, *m_stream.recent_tokens);
        if (c < 0) return PVX_ERR_DECODE_FAILED;
        // PER-ROUND budget: the round commits c + the accepted chain, so the
        // chain may never exceed out_max - generated - 1 (the call-wide cap
        // alone lets the last rounds overrun out_max — live-verified on the
        // Qwen3-8B pair: 26 emitted for out_max 24).
        const int round_cap = std::min(chain_cap, out_max - generated - 1);
        if (stop_at_eog(c) || round_cap <= 0) {
            const pvx_result_t rc = plain_step(c);
            if (out_generated) *out_generated = generated;
            if (rc != PVX_OK || stopped) return rc;
            continue;
        }

        // 2. ── seed + chain (upstream draft()) ──
        const llama_pos pb = (llama_pos) (*m_stream.n_tokens - 1); // deferred boundary
        llama_memory_t dmem = llama_get_memory(m_ctx_dft);
        if (!dmem || !llama_memory_seq_rm(dmem, 0, pb, -1)) {
            ctx->last_error = "eagle3 draft KV repair refused";
            return PVX_ERR_REWIND_UNSUPPORTED;
        }
        if (!eagle_pair_decode(c, m_eagle_g_hist.back().data(), pb)) {
            ctx->last_error = "eagle3 seed decode failed";
            return PVX_ERR_INTERNAL;
        }
        std::vector<llama_token> chain;
        std::vector<float> prenorm((size_t) m_eagle_n_embd_dec);
        llama_pos next_pos = pb + 1;
        for (int i = 0; i < round_cap; i++) {
            const float* d_logits = llama_get_logits_ith(m_ctx_dft, 0);
            const float* pn = llama_get_embeddings_nextn_ith(m_ctx_dft, 0);
            if (!d_logits || !pn) {
                ctx->last_error = "eagle3 draft logits/prenorm missing";
                return PVX_ERR_INTERNAL;
            }
            memcpy(prenorm.data(), pn, (size_t) m_eagle_n_embd_dec * sizeof(float));
            const TopLogit d_top = drafter_top_logit(d_logits, n_vocab);
            if (m_p_min > 0.0f &&
                top_probability(d_logits, n_vocab, d_top.logit) < m_p_min) break;
            if (llama_vocab_is_eog(ctx->vocab, d_top.token)) break;
            chain.push_back(d_top.token);
            m_n_drafted++;
            if (m_model_tgt && m_model_tgt->audit.enabled) m_model_tgt->audit.spec_drafted++;  // ABI v9
            if ((int) chain.size() >= round_cap) break;
            if (!eagle_pair_decode(chain.back(), prenorm.data(), next_pos)) {
                ctx->last_error = "eagle3 chain decode failed";
                return PVX_ERR_INTERNAL;
            }
            next_pos++;
        }
        if ((int) chain.size() < m_draft_min) chain.clear();

        if (chain.empty()) {
            // Drafter cannot help (p_min floor/EOG draft): the seed already
            // completed the deferred boundary pair on the draft side; commit
            // the round's single token honestly (plain pipeline).
            const pvx_result_t rc = plain_step(c);
            if (out_generated) *out_generated = generated;
            if (rc != PVX_OK || stopped) return rc;
            continue;
        }

        // 3. ── verify: ONE target decode of [c, chain...] ──
        const int m = (int) chain.size();
        const int base = *m_stream.n_kv_pos;
        std::vector<llama_token> verify_ids;
        verify_ids.push_back(c);
        verify_ids.insert(verify_ids.end(), chain.begin(), chain.end());
        llama_batch b = llama_batch_init(m + 1, 0, ctx->n_seq_max);
        for (int i = 0; i <= m; i++) {
            b.token[i]     = verify_ids[i];
            b.pos[i]       = base + i;
            b.n_seq_id[i]  = 1;
            b.seq_id[i][0] = m_stream.seq;
            b.logits[i]    = 1;
        }
        b.n_tokens = m + 1;
        const int rc = llama_decode(ctx->ctx, b);
        llama_batch_free(b);
        if (rc != 0) {
            ctx->last_error = "eagle3 verify decode failed (rc=" + std::to_string(rc) + ")";
            return PVX_ERR_DECODE_FAILED;
        }
        std::vector<std::vector<float>> rows(m + 1);
        for (int i = 0; i <= m; i++) spec_snapshot(ctx->ctx, rows[i], i);
        // capture hook IMMEDIATELY (features die at the next target decode)
        if (!eagle_process_target_rows(base, m + 1, verify_ids.data())) {
            ctx->last_error = m_last_error;
            return PVX_ERR_INTERNAL;
        }

        // 4. ── sample-and-match ──
        int n_acc = 0;
        bool hit_eog = false;
        llama_token corrected = -1;
        {
            std::vector<llama_token> feed = *m_stream.recent_tokens;
            feed.push_back(c); // c commits: penalties see it in both paths
            llama_token s = spec_sample(rows[0].data(), ctx->vocab, params, feed);
            if (s < 0) return PVX_ERR_DECODE_FAILED;
            int i = 0;
            for (; i < m; i++) {
                if (s != chain[i]) break;
                n_acc++;
                m_n_accepted++;
                if (m_model_tgt && m_model_tgt->audit.enabled) m_model_tgt->audit.spec_accepted++;  // ABI v9
                feed.push_back(chain[i]);
                if (stop_at_eog(chain[i])) { hit_eog = true; break; }
                if (i + 1 < m) {
                    s = spec_sample(rows[i + 1].data(), ctx->vocab, params, feed);
                    if (s < 0) return PVX_ERR_DECODE_FAILED;
                }
            }
            if (!hit_eog) {
                corrected = (n_acc == m)
                    ? spec_sample(rows[m].data(), ctx->vocab, params, feed)
                    : s; // first disagreeing sample — next round's c
                if (corrected < 0) return PVX_ERR_DECODE_FAILED;
            }
        }

        // 5. ── commit [c, accepted prefix]; KV repair; truncate g_hist ──
        const int committed = 1 + n_acc;
        const int keep = base + committed;
        if (!llama_memory_seq_rm(ctx->mem, m_stream.seq, keep, -1)) {
            ctx->last_error = "eagle3 KV repair refused by memory (partial seq_rm unsupported)";
            return PVX_ERR_REWIND_UNSUPPORTED;
        }
        *m_stream.last_logits = rows[n_acc]; // row of the last committed token
        (*m_stream.n_tokens) += committed;
        (*m_stream.n_kv_pos) += committed;
        m_stream.recent_tokens->push_back(c);
        for (int i = 0; i < n_acc; i++) m_stream.recent_tokens->push_back(chain[i]);
        stream_push_token(c, base);
        for (int i = 0; i < n_acc; i++) stream_push_token(chain[i], base + 1 + i);
        if ((int) m_eagle_g_hist.size() > *m_stream.n_tokens) {
            m_eagle_g_hist.resize((size_t) *m_stream.n_tokens);
        }

        // Emit the committed tokens in stream order (c first).
        generated++;
        if (on_token && !on_token(c, user_data)) {
            if (out_generated) *out_generated = generated;
            return PVX_OK;
        }
        if (hit_eog) {
            if (out_generated) *out_generated = generated;
            return PVX_OK;
        }
        for (int i = 0; i < n_acc; i++) {
            generated++;
            if (on_token && !on_token(chain[i], user_data)) {
                if (out_generated) *out_generated = generated;
                return PVX_OK;
            }
            if (stop_at_eog(chain[i])) {
                if (out_generated) *out_generated = generated;
                return PVX_OK;
            }
        }
        pvx_log(PVX_LOG_DEBUG, "Spec",
                "eagle3 round: chain=%d accepted=%d committed=%d corrected=%d pos=%d",
                m, n_acc, committed, (int) corrected, keep);
    }

    if (out_generated) *out_generated = generated;
    return PVX_OK;
}

// ── Separate-head draft-MTP internals (ABI v8 wave 4) ───────────────────────
//
// Pattern-port of the vendored common_speculative_impl_draft_mtp (upstream
// llama.cpp PR #28243, evolved form; single self-contained head only).
// Pairing convention (upstream draft()/process(), live-verified on qwen4exp):
// EVERY head-context row at stream position p consumes (t_p, h_{p-1}) — the
// token at p paired with the target trunk's nextn embedding at p-1 (zero row
// for p == 0 — upstream pending_h zeros init). The row at p predicts t_{p+1}
// (standard shift). Seed row = the round's commit token at its target
// position, paired h_hist.back() (= the last committed row's h, upstream
// pending_h); catch-up rows replay each committed position with the same
// pairing (upstream process()); chain rows pair the HEAD's own nextn output
// of the previous row (the head's h estimate — upstream non-shared draft()
// set_embd_state path). No seq_rm on the head context anywhere: stale chain
// cells are re-decoded-over with the correct committed inputs, exactly the
// upstream process() semantics (the eagle3 lane's seq_rm drop does NOT apply
// here — the head memory carries recurrent cells).

bool PvxBLSpecEngine::mtp_pair_decode(llama_token tok, const float* h_row,
                                      llama_pos pos, bool output_logits) {
    llama_batch_ext* ext = llama_batch_ext_init(m_ctx_dft);
    if (!ext) return false;
    // ABI v8 wave-4 fix (live-verified 2026-10-19): every seed/chain row must
    // DROP the head memory region above its position first — upstream draft()
    // seq_rm(pos0, -1) (speculative.cpp:1749) — else the previous round's
    // catch-up/draft cells occupy [pos, ...) and the decode hits the KV
    // contiguity check ("X = 30 / Y = 24", live log 20:55). Cells below pos
    // (the committed prefix) stay.
    if (!m_mtp_mem_shared) {
        llama_memory_t dmem = llama_get_memory(m_ctx_dft);
        if (dmem) llama_memory_seq_rm(dmem, 0, pos, -1);
    }
    // BL-42 (shared-memory heads): NO head seq_rm — upstream draft() keeps the
    // same-position overwrite semantics for gemma4-class assistants.
    bool ok = false;
    do {
        const int32_t idx = llama_batch_ext_add_token(ext, 0, tok);
        if (idx < 0) break;
        // ABI v8 wave-4 fix: the MTP head graph consumes the paired h row as
        // a STATE embedding (upstream draft()/process() set_embd_state path,
        // speculative.cpp:1637) — llm_graph_input_embd_h reads ubatch->embd_state.
        // A plain input embd (set_embd_token) fails width validation: the head's
        // token embd is 2560-wide while the h row is n_embd_out (10240) — the
        // live 2026-10-19 catch-up log "set_token_embd: embedding size mismatch,
        // got 10240, expected 2560".
        const llama_embd le = { h_row, 1, (size_t) m_mtp_n_embd };
        if (!llama_batch_ext_set_embd_state(ext, idx, le)) break;
        const llama_pos p[4] = { pos, pos, pos, pos };
        if (!llama_batch_ext_set_pos(ext, idx, p)) break;
        if (!llama_batch_ext_set_output_logits(ext, idx, output_logits)) break;
        if (llama_process(m_ctx_dft, LLAMA_PROCESS_TYPE_DECODE, ext) != 0) {
            pvx_log(PVX_LOG_ERROR, "Spec", "mtp head decode failed at pos=%d", (int) pos);
            break;
        }
        ok = true;
    } while (false);
    llama_batch_ext_free(ext);
    return ok;
}

// BL-39 pure pairing planner: see pvx_spec_engine.h (struct PvxBLMtpArm).
PvxBLMtpArm pvx_mtp_catch_up_arm(llama_pos p, const pvx_mirror_entry_t& e) {
    PvxBLMtpArm a;
    a.input_is_embd = (e.id == LLAMA_TOKEN_NULL);
    a.zero_h = (p == 0);
    a.h_index = (size_t) (p > 0 ? p - 1 : 0);
    return a;
}

// BL-45 pure planners (upstream speculative.cpp ctor + draft() chain_heads
// path): chain mode is decided ONCE at head init, per-step arms are pure
// functions of (step, drafted).
PvxBLMtpChainPlan pvx_mtp_chain_plan(int n_nextn, bool mem_shared, int draft_max) {
    PvxBLMtpChainPlan p{};
    p.n_layers  = std::max(1, n_nextn);
    p.chain     = p.n_layers > 1 && !mem_shared;
    p.draft_max = p.chain ? std::min(draft_max, p.n_layers) : draft_max;
    return p;
}

PvxBLMtpChainArm pvx_mtp_chain_arm(int step, int drafted) {
    PvxBLMtpChainArm a{};
    a.layer_offset = step;
    a.n_rows       = drafted + 1;
    a.i_last       = drafted;
    return a;
}

// BL-45 factoring: chunked (t_p, h_{p-1}) replay of n_rows mirror rows from
// pos_first into the head context — the shared body of process() catch-up
// and state-restore replay. No seq_rm inside: callers own the head-memory
// discipline (single: drop [pos_first, end) once; chain: per head).
bool PvxBLSpecEngine::mtp_replay_rows(const char* err_tag, llama_pos pos_first,
                                       int n_rows, const pvx_mirror_entry_t* rows) {
    const int d_batch = (int) llama_n_batch(m_ctx_dft);
    const int dft_width = (int) llama_model_n_embd_inp(m_model_dft);
    std::vector<float> zeros;
    // h_{-1} pairing row (p == 0): zero-padded scratch (upstream pending
    // zeros init), sized at create time.
    if (m_mtp_zeros.empty()) m_mtp_zeros.assign((size_t) m_mtp_n_embd, 0.0f);
    for (int off = 0; off < n_rows; off += d_batch) {
        const int chunk = std::min(d_batch, n_rows - off);
        llama_batch_ext* ext = llama_batch_ext_init(m_ctx_dft);
        if (!ext) return false;
        bool ok = false;
        do {
            for (int j = 0; j < chunk; j++) {
                const llama_pos p = pos_first + off + j;
                const pvx_mirror_entry_t& e = rows[off + j];
                const PvxBLMtpArm arm = pvx_mtp_catch_up_arm(p, e);
                const float* h_pair = arm.zero_h
                    ? m_mtp_zeros.data() : m_mtp_h_hist[arm.h_index].data();
                int32_t idx = -1;
                if (arm.input_is_embd) {
                    // Vision row: image embedding as head input (upstream
                    // add_embd arm), zero-pad on width mismatch.
                    const float* row = e.embd.data();
                    if ((int) e.embd.size() != dft_width) {
                        if (!m_width_warned) {
                            m_width_warned = true;
                            pvx_log(PVX_LOG_WARN, "Spec",
                                    "target embeddings of size %zu do not fit the mtp head "
                                    "input width %d; the head receives zero rows for them "
                                    "and drafting after multimodal input will be poor",
                                    e.embd.size(), dft_width);
                        }
                        if ((int) zeros.size() != dft_width)
                            zeros.assign((size_t) dft_width, 0.0f);
                        row = zeros.data();
                    }
                    const llama_embd le = { row, 1, (size_t) dft_width };
                    idx = llama_batch_ext_add_embd(ext, 0, le);
                } else {
                    idx = llama_batch_ext_add_token(ext, 0, e.id);
                }
                if (idx < 0) break;
                // ABI v8 wave-4 fix: state-embedding pairing (see mtp_pair_decode) —
                // upstream process() pairs each committed row via set_embd_state.
                const llama_embd le = { h_pair, 1, (size_t) m_mtp_n_embd };
                if (!llama_batch_ext_set_embd_state(ext, idx, le)) break;
                // Pinned 4-D position (token entries carry {p,p,p,p}; vision
                // entries carry their M-RoPE section-major row position —
                // exactly what the target decoded against).
                if (!llama_batch_ext_set_pos(ext, idx, e.pos.data())) break;
                if (!llama_batch_ext_set_output_logits(ext, idx, false)) break;
                ok = true;  // full chunk armed
            }
            if (!ok) break;
            if (llama_process(m_ctx_dft, LLAMA_PROCESS_TYPE_DECODE, ext) != 0) {
                ok = false;
                break;
            }
        } while (false);
        llama_batch_ext_free(ext);
        if (!ok) {
            m_last_error = std::string(err_tag) + ": head replay decode failed";
            pvx_log(PVX_LOG_ERROR, "Spec", "%s", m_last_error.c_str());
            return false;
        }
    }
    return true;
}

bool PvxBLSpecEngine::mtp_process_target_rows(llama_pos pos_first, int n_rows,
                                               const pvx_mirror_entry_t* rows) {
    if (m_mode != PVX_SPEC_MTP_DRAFT || n_rows <= 0) return true;
    if (!m_ctx_dft || !rows) return false;
    // Capture-order invariant: one h row per committed position — the hook
    // fires exactly once per decoded batch, in stream order.
    if ((size_t) pos_first != m_mtp_h_hist.size()) {
        m_last_error = "spec-engine/mtp-capture-order: hook at pos " +
                       std::to_string((int) pos_first) + " but h_hist holds " +
                       std::to_string(m_mtp_h_hist.size()) + " rows";
        pvx_log(PVX_LOG_ERROR, "Spec", "%s", m_last_error.c_str());
        return false;
    }

    auto* tgt = m_ctx_tgt->ctx;
    const float* h_tgt = llama_get_embeddings_nextn(tgt);
    if (!h_tgt) {
        m_last_error = "spec-engine/mtp-capture: target nextn embeddings not "
                       "extracted (llama_set_embeddings_nextn contract broken)";
        pvx_log(PVX_LOG_ERROR, "Spec", "%s", m_last_error.c_str());
        return false;
    }

    // One h row per committed position (upstream verify_h gather).
    m_mtp_h_hist.reserve(m_mtp_h_hist.size() + (size_t) n_rows);
    for (int i = 0; i < n_rows; i++) {
        m_mtp_h_hist.emplace_back(
            h_tgt + (size_t) i * (size_t) m_mtp_n_embd,
            h_tgt + (size_t) (i + 1) * (size_t) m_mtp_n_embd);
    }

    // BL-42 (shared-memory heads, gemma4-assistant): the head reads the trunk
    // KV through ctx_other — upstream process() skips the ENTIRE catch-up
    // decode for is_mem_shared heads; the h capture above is still required
    // (the seed pairs h_hist.back()).
    if (m_mtp_mem_shared) {
        pvx_log(PVX_LOG_DEBUG, "Spec", "mtp process (shared): pos_first=%d n_rows=%d h_hist=%zu (no catch-up)",
                (int) pos_first, n_rows, m_mtp_h_hist.size());
        return true;
    }

    // Catch-up decode: the whole batch into the head context with the
    // (t_p, h_{p-1}) pairing (zero-h row for p == 0 — upstream pending
    // zeros), chunked by the head's n_batch.
    //
    // BL-39 (upstream speculative.cpp:1630): vision rows ride the SAME
    // pairing — the image embedding is the head INPUT (add_embd arm), the
    // h_{p-1} state pairing is unchanged, so a vision chunk in the stream
    // shifts the row indices but never pairs an image-embd position as
    // history. Input width mismatch (target projector vs head input)
    // zero-pads with a one-time warn (upstream zeros_warned discipline —
    // quality signal only, never an abort).
    //
    // ABI v8 wave-4 fix (live-verified 2026-10-19, probe): the head KV fails
    // the contiguity check when a catch-up re-decodes positions it already
    // holds — the round's seed+chain pre-writes [pos_first, ...) into the
    // head (mtp_pair_decode), and every later catch-up then re-decodes the
    // same committed stream rows (rc=-1 "failed to initialize batch",
    // live log 20:50). Mirror upstream's draft-region drop
    // (speculative.cpp:1650/1749): seq_rm [pos_first, end) on the head
    // memory first — the committed prefix below pos_first stays — so the
    // catch-up replays over freed cells only. No-op for fresh positions
    // (prompt slices: nothing above pos_first yet).
    // BL-45 (multi-block chain heads): upstream process() replays the
    // catch-up batch once per head — per head: seq_rm [pos_first, end),
    // llama_set_nextn_layer_offset(head), decode, then restore offset 0
    // (speculative.cpp chain_heads path). Single-block and shared-memory
    // paths stay byte-identical (one replay, offset untouched).
    if (m_mtp_chain_heads) {
        for (int head = 0; head < m_mtp_n_layers; ++head) {
            llama_memory_t dmem = llama_get_memory(m_ctx_dft);
            if (dmem) llama_memory_seq_rm(dmem, 0, (llama_pos) pos_first, -1);
            llama_set_nextn_layer_offset(m_ctx_dft, head);
            if (!mtp_replay_rows("spec-engine/mtp-catchup", pos_first, n_rows, rows))
                return false;
        }
        llama_set_nextn_layer_offset(m_ctx_dft, 0); // restore default for non-draft decodes (upstream speculative.cpp:1665)
    } else {
        llama_memory_t dmem = llama_get_memory(m_ctx_dft);
        if (dmem) llama_memory_seq_rm(dmem, 0, (llama_pos) pos_first, -1);
        if (!mtp_replay_rows("spec-engine/mtp-catchup", pos_first, n_rows, rows))
            return false;
    }
    pvx_log(PVX_LOG_DEBUG, "Spec", "mtp process: pos_first=%d n_rows=%d h_hist=%zu",
            (int) pos_first, n_rows, m_mtp_h_hist.size());
    return true;
}

void PvxBLSpecEngine::mtp_on_stream_shrunk(int new_n_tokens) {
    if (m_mode != PVX_SPEC_MTP_DRAFT) return;
    // Truncate the h ring; stale head cells above the boundary are
    // re-decoded-over by later catch-ups (no seq_rm — the head memory carries
    // recurrent cells without a snapshot ring).
    if ((int) m_mtp_h_hist.size() > new_n_tokens) {
        m_mtp_h_hist.resize((size_t) std::max(0, new_n_tokens));
    }
}

void PvxBLSpecEngine::mtp_state_capture(std::vector<std::vector<float>>& out_rows) const {
    if (m_mode != PVX_SPEC_MTP_DRAFT) return;
    out_rows = m_mtp_h_hist;
}

bool PvxBLSpecEngine::mtp_state_restore(std::vector<std::vector<float>>&& rows) {
    if (m_mode != PVX_SPEC_MTP_DRAFT) return false;
    const int n = m_stream.n_tokens ? *m_stream.n_tokens : 0;
    if ((int) rows.size() != n || !m_stream.mirror ||
        (size_t) n != m_stream.mirror->size()) {
        m_last_error = "spec-engine/mtp-state-mismatch: saved h rows (" +
                       std::to_string(rows.size()) + ") do not cover the restored "
                       "stream (" + std::to_string(n) + " tokens) — the draft-MTP "
                       "session must be re-initialized after this restore";
        pvx_log(PVX_LOG_WARN, "Spec", "%s", m_last_error.c_str());
        return false;
    }
    for (const auto& r : rows) {
        if ((int) r.size() != m_mtp_n_embd) {
            m_last_error = "spec-engine/mtp-state-restore: saved h row width (" +
                           std::to_string(r.size()) + ") != head width (" +
                           std::to_string(m_mtp_n_embd) + ")";
            return false;
        }
    }
    // Full head-context clear, then a full catch-up replay [0, n): every
    // position re-decoded with the (t_p, h_{p-1}) pairing (no deferral —
    // every committed position carries its own pairing row). BL-39: mirror
    // entries may carry vision rows (id == LLAMA_TOKEN_NULL) — the image
    // embedding is the head INPUT (upstream add_embd arm), the h pairing
    // is unchanged, so a saved mid-vision session restores exactly like a
    // text one (the old typed multimodal refusal is gone).
    // BL-42 (shared-memory heads): no head KV to rebuild — the head reads the
    // trunk; only the h ring is restored.
    if (m_mtp_mem_shared) {
        m_mtp_h_hist = std::move(rows);
        return true;
    }
    m_mtp_h_hist = std::move(rows);
    // BL-45 (multi-block chain heads): the restore replay mirrors upstream
    // process() per head — seq_rm the whole head memory, set offset(head),
    // replay the committed stream, restore offset 0. Single-block and
    // shared-memory paths stay byte-identical (one replay, offset untouched).
    if (m_mtp_chain_heads) {
        for (int head = 0; head < m_mtp_n_layers; ++head) {
            llama_memory_t dmem = llama_get_memory(m_ctx_dft);
            if (dmem) llama_memory_seq_rm(dmem, 0, 0, -1);
            llama_set_nextn_layer_offset(m_ctx_dft, head);
            if (!mtp_replay_rows("spec-engine/mtp-state-restore", 0, n,
                                 m_stream.mirror->data()))
                return false;
        }
        llama_set_nextn_layer_offset(m_ctx_dft, 0); // restore default (upstream speculative.cpp:1665)
    } else {
        llama_memory_t dmem = llama_get_memory(m_ctx_dft);
        if (dmem) llama_memory_seq_rm(dmem, 0, 0, -1);
        if (!mtp_replay_rows("spec-engine/mtp-state-restore", 0, n,
                             m_stream.mirror->data()))
            return false;
    }
    return true;
}

// ── draft-MTP round loop ───────────────────────────────────────────────────
//
// Round structure (mirrors generate_eagle; see the pairing block comment
// above): (1) c = the target's sample from last_logits (the round's
// always-committed token at base = n_kv_pos); (2) seed: head row
// (c, h_hist[n_tokens-1]) at base with logits → d0 = greedy top-1; chain rows
// (d_{i-1}, head-own nextn of the previous row) at base+i with logits; (3)
// ONE target verify decode of [c, chain...] at [base, ...] — all logits; the
// capture hook fires immediately (catch-up + h_hist growth); (4)
// sample-and-match (greedy-exact), corrected = the first disagreeing sample
// (next round's c); (5) commit [c, accepted], target KV repair to the
// committed prefix, h_hist truncated to n_tokens.

pvx_result_t PvxBLSpecEngine::generate_mtp(const pvx_sampling_params_t* params,
                                            int out_max,
                                            pvx_spec_token_callback_t on_token,
                                            void* user_data,
                                            int* out_generated) {
    auto* ctx = m_ctx_tgt;
    const int n_vocab = llama_vocab_n_tokens(ctx->vocab);
    const bool ignore_eos = params->ignore_eos;
    auto stop_at_eog = [&](llama_token t) {
        return !ignore_eos && llama_vocab_is_eog(ctx->vocab, t);
    };

    // Chain bound: batch width + budget (minus the always-present c row).
    int chain_cap = std::min(m_draft_max,
                             std::min((int) llama_n_batch(ctx->ctx) - 2,
                                      out_max - 2));
    // Hybrid/recurrent target (qwen4exp): the verify repair rewinds the
    // recurrent state via trailing snapshots — the chain may never exceed
    // llama_n_rs_seq (upstream server rule; plain steps when the context was
    // created with n_rs_seq == 0).
    if (llama_model_is_recurrent(m_model_tgt->model) ||
        llama_model_is_hybrid(m_model_tgt->model)) {
        const int rs_cap = (int) llama_n_rs_seq(ctx->ctx);
        if (rs_cap < chain_cap) {
            pvx_log(PVX_LOG_WARN, "Spec",
                    "mtp drafting disabled: hybrid/recurrent target memory cannot "
                    "roll back verify batches (llama_n_rs_seq=%d, create the context "
                    "with n_rs_seq >= chain to enable); plain exact steps only", rs_cap);
        }
        chain_cap = std::min(chain_cap, rs_cap - 1);
    }
    if (chain_cap < 0) chain_cap = 0;

    pvx_log(PVX_LOG_DEBUG, "Spec",
            "mtp generate enter: n_kv_pos=%d n_tokens=%d h_hist=%zu chain_cap=%d rs=%d",
            (int) *m_stream.n_kv_pos, (int) *m_stream.n_tokens,
            m_mtp_h_hist.size(), chain_cap, (int) llama_n_rs_seq(ctx->ctx));

    int generated = 0;
    m_n_drafted = 0;
    m_n_accepted = 0;

    bool stopped = false;
    auto plain_step = [&](llama_token t) -> pvx_result_t {
        const llama_pos pos = *m_stream.n_kv_pos;
        llama_batch b = llama_batch_init(1, 0, ctx->n_seq_max);
        b.token[0]     = t;
        b.pos[0]       = pos;
        b.n_seq_id[0]  = 1;
        b.seq_id[0][0] = m_stream.seq;
        b.logits[0]    = 1;
        b.n_tokens     = 1;
        const int rc = llama_decode(ctx->ctx, b);
        llama_batch_free(b);
        if (rc != 0) {
            ctx->last_error = "mtp fallback decode failed (rc=" + std::to_string(rc) + ")";
            return PVX_ERR_DECODE_FAILED;
        }
        // BL-39: hook takes mirror entries (vision rows carry embd payloads).
        pvx_mirror_entry_t ent;
        ent.id = t;
        ent.pos = { pos, pos, pos, pos };
        if (!mtp_process_target_rows(pos, 1, &ent)) return PVX_ERR_INTERNAL;
        spec_snapshot(ctx->ctx, *m_stream.last_logits, 0);
        stream_push_token(t, *m_stream.n_kv_pos);
        (*m_stream.n_tokens)++;
        (*m_stream.n_kv_pos)++;
        m_stream.recent_tokens->push_back(t);
        generated++;
        if (on_token && !on_token(t, user_data)) { stopped = true; return PVX_OK; }
        if (stop_at_eog(t)) { stopped = true; return PVX_OK; }
        return PVX_OK;
    };

    while (generated < out_max) {
        // Capture-order invariant: one h row per committed position.
        if (m_mtp_h_hist.size() != (size_t) *m_stream.n_tokens) {
            m_last_error = "spec-engine/mtp-capture-order: h_hist (" +
                           std::to_string(m_mtp_h_hist.size()) + ") != committed (" +
                           std::to_string(*m_stream.n_tokens) + ")";
            ctx->last_error = m_last_error;
            pvx_log(PVX_LOG_ERROR, "Spec", "%s", m_last_error.c_str());
            return PVX_ERR_INTERNAL;
        }

        // 1. ── c: the target's sample from last_logits ──
        const llama_token c = spec_sample(m_stream.last_logits->data(), ctx->vocab,
                                          params, *m_stream.recent_tokens);
        if (c < 0) return PVX_ERR_DECODE_FAILED;
        // PER-ROUND budget (see generate_eagle).
        const int round_cap = std::min(chain_cap, out_max - generated - 1);
        if (stop_at_eog(c) || round_cap <= 0) {
            const pvx_result_t rc = plain_step(c);
            if (out_generated) *out_generated = generated;
            if (rc != PVX_OK || stopped) return rc;
            continue;
        }

        // 2. ── seed + chain (upstream draft(), single-head path) ──
        const llama_pos base = (llama_pos) *m_stream.n_kv_pos;
        std::vector<float> h_seed((size_t) m_mtp_n_embd);
        memcpy(h_seed.data(), m_mtp_h_hist.back().data(),
               (size_t) m_mtp_n_embd * sizeof(float));
        if (!mtp_pair_decode(c, h_seed.data(), base, /*output_logits=*/ true)) {
            ctx->last_error = "mtp seed decode failed";
            return PVX_ERR_INTERNAL;
        }
        std::vector<llama_token> chain;
        std::vector<float> h_step((size_t) m_mtp_n_embd);
        // BL-45 (multi-block chain heads, step35 class): upstream draft()
        // chain_heads path — draft step i decodes the WHOLE re-added prefix
        // (seed row + drafted so far) under llama_set_nextn_layer_offset(i),
        // per step seq_rm(base, -1) so each head rebuilds its OWN layer's KV
        // draft region; logits only on the last row; pairing row t rides
        // chain_h[t] (seed = pending h, then each head's own nextn output).
        // draft_max was clamped to n_layers at init (upstream n_max rule), so
        // round_cap can never address an untrained block.
        if (m_mtp_chain_heads) {
            std::vector<float> chain_h = h_seed;
            chain_h.reserve((size_t) (round_cap + 1) * (size_t) m_mtp_n_embd);
            for (int i = 0; i < round_cap; i++) {
                const PvxBLMtpChainArm arm = pvx_mtp_chain_arm(i, (int) chain.size());
                llama_memory_t dmem = llama_get_memory(m_ctx_dft);
                if (dmem) llama_memory_seq_rm(dmem, 0, base, -1);
                llama_set_nextn_layer_offset(m_ctx_dft, arm.layer_offset);
                llama_batch_ext* ext = llama_batch_ext_init(m_ctx_dft);
                bool ok = false;
                do {
                    for (int t = 0; t < arm.n_rows; t++) {
                        const llama_token tok = (t == 0) ? c : chain[(size_t) (t - 1)];
                        const int32_t idx = llama_batch_ext_add_token(ext, 0, tok);
                        if (idx < 0) break;
                        const llama_embd le = {
                            chain_h.data() + (size_t) t * (size_t) m_mtp_n_embd,
                            1, (size_t) m_mtp_n_embd };
                        if (!llama_batch_ext_set_embd_state(ext, idx, le)) break;
                        const llama_pos pos4[4] = { base + t, base + t,
                                                    base + t, base + t };
                        if (!llama_batch_ext_set_pos(ext, idx, pos4)) break;
                        if (!llama_batch_ext_set_output_logits(ext, idx,
                                t == arm.n_rows - 1)) break;
                        ok = (t == arm.n_rows - 1);
                    }
                    if (!ok) break;
                    if (llama_process(m_ctx_dft, LLAMA_PROCESS_TYPE_DECODE, ext) != 0) {
                        pvx_log(PVX_LOG_ERROR, "Spec",
                                "mtp chain head decode failed at step=%d pos=%d",
                                i, (int) base);
                        ok = false;
                        break;
                    }
                } while (false);
                llama_batch_ext_free(ext);
                if (!ok) {
                    ctx->last_error = "mtp chain head decode failed";
                    return PVX_ERR_INTERNAL;
                }
                const float* d_logits = llama_get_logits_ith(m_ctx_dft, arm.i_last);
                const float* h_row = llama_get_embeddings_nextn_ith(m_ctx_dft, arm.i_last);
                if (!d_logits || !h_row) {
                    ctx->last_error = "mtp head logits/h missing";
                    return PVX_ERR_INTERNAL;
                }
                memcpy(h_step.data(), h_row, (size_t) m_mtp_n_embd * sizeof(float));
                const TopLogit d_top = drafter_top_logit(d_logits, n_vocab);
                if (m_p_min > 0.0f &&
                    top_probability(d_logits, n_vocab, d_top.logit) < m_p_min) break;
                if (llama_vocab_is_eog(ctx->vocab, d_top.token)) break;
                chain.push_back(d_top.token);
                m_n_drafted++;
                if (m_model_tgt && m_model_tgt->audit.enabled) m_model_tgt->audit.spec_drafted++;  // ABI v9
                chain_h.insert(chain_h.end(), h_step.begin(), h_step.end());
            }
            llama_set_nextn_layer_offset(m_ctx_dft, 0); // restore default for non-draft decodes (upstream speculative.cpp:1844)
        } else {
        // BL-42 (shared-memory heads): every draft row rides the SAME position
        // (upstream is_mem_shared draft(), speculative.cpp — gemma4 assistants
        // share the trunk KV; advancing would collide with target cells).
        llama_pos next_pos = m_mtp_mem_shared ? base : (llama_pos) (base + 1);
        for (int i = 0; i < round_cap; i++) {
            const float* d_logits = llama_get_logits_ith(m_ctx_dft, 0);
            const float* h_row = llama_get_embeddings_nextn_ith(m_ctx_dft, 0);
            if (!d_logits || !h_row) {
                ctx->last_error = "mtp head logits/h missing";
                return PVX_ERR_INTERNAL;
            }
            memcpy(h_step.data(), h_row, (size_t) m_mtp_n_embd * sizeof(float));
            const TopLogit d_top = drafter_top_logit(d_logits, n_vocab);
            if (m_p_min > 0.0f &&
                top_probability(d_logits, n_vocab, d_top.logit) < m_p_min) break;
            if (llama_vocab_is_eog(ctx->vocab, d_top.token)) break;
            chain.push_back(d_top.token);
            m_n_drafted++;
            if (m_model_tgt && m_model_tgt->audit.enabled) m_model_tgt->audit.spec_drafted++;  // ABI v9
            if ((int) chain.size() >= round_cap) break;
            if (!mtp_pair_decode(chain.back(), h_step.data(), next_pos,
                                 /*output_logits=*/ true)) {
                ctx->last_error = "mtp chain decode failed";
                return PVX_ERR_INTERNAL;
            }
            if (!m_mtp_mem_shared) next_pos++;
        }
        } // BL-45: single-head/shared path above unchanged
        if ((int) chain.size() < m_draft_min) chain.clear();

        if (chain.empty()) {
            // Drafter cannot help (p_min floor / EOG draft / chain floor):
            // the seed row already wrote this round's catch-up cell on the
            // head; commit the round's single token honestly (plain pipeline).
            const pvx_result_t rc = plain_step(c);
            if (out_generated) *out_generated = generated;
            if (rc != PVX_OK || stopped) return rc;
            continue;
        }

        // 3. ── verify: ONE target decode of [c, chain...] ──
        const int m = (int) chain.size();
        std::vector<llama_token> verify_ids;
        verify_ids.push_back(c);
        verify_ids.insert(verify_ids.end(), chain.begin(), chain.end());
        llama_batch b = llama_batch_init(m + 1, 0, ctx->n_seq_max);
        for (int i = 0; i <= m; i++) {
            b.token[i]     = verify_ids[i];
            b.pos[i]       = base + i;
            b.n_seq_id[i]  = 1;
            b.seq_id[i][0] = m_stream.seq;
            b.logits[i]    = 1;
        }
        b.n_tokens = m + 1;
        const int rc = llama_decode(ctx->ctx, b);
        llama_batch_free(b);
        if (rc != 0) {
            ctx->last_error = "mtp verify decode failed (rc=" + std::to_string(rc) + ")";
            return PVX_ERR_DECODE_FAILED;
        }
        std::vector<std::vector<float>> rows(m + 1);
        for (int i = 0; i <= m; i++) spec_snapshot(ctx->ctx, rows[i], i);
        // capture hook IMMEDIATELY (h rows die at the next target decode);
        // BL-39: verify rows as mirror entries (token rows — sampled positions).
        std::vector<pvx_mirror_entry_t> vent((size_t) (m + 1));
        for (int i = 0; i <= m; i++) {
            vent[(size_t) i].id = verify_ids[(size_t) i];
            vent[(size_t) i].pos = { base + i, base + i, base + i, base + i };
        }
        if (!mtp_process_target_rows(base, m + 1, vent.data())) {
            ctx->last_error = m_last_error;
            return PVX_ERR_INTERNAL;
        }

        // 4. ── sample-and-match ──
        int n_acc = 0;
        bool hit_eog = false;
        llama_token corrected = -1;
        {
            std::vector<llama_token> feed = *m_stream.recent_tokens;
            feed.push_back(c); // c commits: penalties see it in both paths
            llama_token s = spec_sample(rows[0].data(), ctx->vocab, params, feed);
            if (s < 0) return PVX_ERR_DECODE_FAILED;
            int i = 0;
            for (; i < m; i++) {
                if (s != chain[i]) break;
                n_acc++;
                m_n_accepted++;
                if (m_model_tgt && m_model_tgt->audit.enabled) m_model_tgt->audit.spec_accepted++;  // ABI v9
                feed.push_back(chain[i]);
                if (stop_at_eog(chain[i])) { hit_eog = true; break; }
                if (i + 1 < m) {
                    s = spec_sample(rows[i + 1].data(), ctx->vocab, params, feed);
                    if (s < 0) return PVX_ERR_DECODE_FAILED;
                }
            }
            if (!hit_eog) {
                corrected = (n_acc == m)
                    ? spec_sample(rows[m].data(), ctx->vocab, params, feed)
                    : s; // first disagreeing sample — next round's c
                if (corrected < 0) return PVX_ERR_DECODE_FAILED;
            }
        }

        // 5. ── commit [c, accepted prefix]; target KV repair; truncate h_hist ──
        const int committed = 1 + n_acc;
        const int keep = base + committed;
        if (!llama_memory_seq_rm(ctx->mem, m_stream.seq, keep, -1)) {
            ctx->last_error = "mtp KV repair refused by memory (hybrid/recurrent "
                              "target needs n_rs_seq >= chain — create the context "
                              "with a recurrent snapshot ring)";
            return PVX_ERR_REWIND_UNSUPPORTED;
        }
        *m_stream.last_logits = rows[n_acc]; // row of the last committed token
        (*m_stream.n_tokens) += committed;
        (*m_stream.n_kv_pos) += committed;
        m_stream.recent_tokens->push_back(c);
        for (int i = 0; i < n_acc; i++) m_stream.recent_tokens->push_back(chain[i]);
        stream_push_token(c, base);
        for (int i = 0; i < n_acc; i++) stream_push_token(chain[i], base + 1 + i);
        if ((int) m_mtp_h_hist.size() > *m_stream.n_tokens) {
            m_mtp_h_hist.resize((size_t) *m_stream.n_tokens);
        }

        // Emit the committed tokens in stream order (c first).
        generated++;
        if (on_token && !on_token(c, user_data)) {
            if (out_generated) *out_generated = generated;
            return PVX_OK;
        }
        if (hit_eog) {
            if (out_generated) *out_generated = generated;
            return PVX_OK;
        }
        for (int i = 0; i < n_acc; i++) {
            generated++;
            if (on_token && !on_token(chain[i], user_data)) {
                if (out_generated) *out_generated = generated;
                return PVX_OK;
            }
            if (stop_at_eog(chain[i])) {
                if (out_generated) *out_generated = generated;
                return PVX_OK;
            }
        }
        pvx_log(PVX_LOG_DEBUG, "Spec",
                "mtp round: chain=%d accepted=%d committed=%d corrected=%d pos=%d",
                m, n_acc, committed, (int) corrected, keep);
    }

    if (out_generated) *out_generated = generated;
    return PVX_OK;
}
