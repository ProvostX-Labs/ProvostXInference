// pvx_spec_engine.cpp — Standard-executor speculative-decoding session.
// See pvx_spec_engine.h for scope, semantics, and invariants.

#include "pvx_spec_engine.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include "speculative.h"

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

struct TopLogit {
    llama_token token;
    float       logit;
};

TopLogit drafter_top_logit(const float* logits, int n_vocab) {
    int best = 0;
    float best_logit = logits[0];
    for (int i = 1; i < n_vocab; i++) {
        if (logits[i] > best_logit) { best_logit = logits[i]; best = i; }
    }
    return { (llama_token) best, best_logit };
}

// Probability mass of the top token, given the full row (exact, stable):
// p_top = 1 / Σ exp(l_i − l_top). Only consulted when a p_min floor is set.
float top_probability(const float* logits, int n_vocab, float top_logit) {
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
            out_error = "mode OFF passed to spec enable; use pvx_executor_spec_free to "
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
        case PVX_SPEC_DRAFT:
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
    engine->m_exec      = nullptr; // bound at spec_init time
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

PvxBLSpecEngine::~PvxBLSpecEngine() {
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
    b.seq_id[0][0] = 0;
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
    const auto& mirror = m_exec->mirror;
    if (mirror.size() != (size_t) m_exec->n_tokens) {
        why = "spec-engine/mirror-consistency: mirror ring size (" +
              std::to_string(mirror.size()) + ") != committed token count (" +
              std::to_string(m_exec->n_tokens) + ") — a commit path bypassed the ring; "
              "refusing to draft against a phantom stream";
        return false;
    }
    if (mirror.empty()) {
        return true;
    }
    if (mirror.front().pos[0] != m_exec->n_kv_pos - m_exec->n_tokens) {
        why = "spec-engine/mirror-consistency: mirror front position (" +
              std::to_string(mirror.front().pos[0]) + ") != window base (" +
              std::to_string(m_exec->n_kv_pos - m_exec->n_tokens) + ")";
        return false;
    }
    if (mirror.back().pos[0] != m_exec->n_kv_pos - 1) {
        why = "spec-engine/mirror-consistency: mirror back position (" +
              std::to_string(mirror.back().pos[0]) + ") != last committed position (" +
              std::to_string(m_exec->n_kv_pos - 1) + ")";
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
    out.reserve(m_exec->mirror.size());
    for (const auto& e : m_exec->mirror) {
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
    auto* exec = m_exec;
    const int pos_base = exec->n_kv_pos - exec->n_tokens;

    const bool incremental = (m_mirror_base == pos_base && m_mirror_pos <= exec->n_kv_pos);
    llama_memory_t dmem = llama_get_memory(m_ctx_dft);
    if (!dmem || !llama_memory_seq_rm(dmem, 0, incremental ? m_mirror_pos : 0, -1)) {
        pvx_log(PVX_LOG_ERROR, "Spec", "draft mirror: seq_rm refused (rebuild=%d)", !incremental);
        return false;
    }
    pvx_log(PVX_LOG_DEBUG, "Spec", "mirror start: rebuild=%d watermark=%d n_kv_pos=%d n_tokens=%d",
            !incremental, m_mirror_pos, exec->n_kv_pos, exec->n_tokens);

    const auto& ring = exec->mirror;
    const int begin  = incremental ? m_mirror_pos : pos_base;
    if (begin < pos_base || begin > exec->n_kv_pos) {
        m_last_error = "spec-engine/mirror-consistency: watermark outside the committed window";
        return false;
    }
    const int n_fill = exec->n_kv_pos - begin;
    if (n_fill <= 0) {
        m_mirror_pos  = exec->n_kv_pos;
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
    m_mirror_pos  = exec->n_kv_pos;
    m_mirror_base = pos_base;
    pvx_log(PVX_LOG_DEBUG, "Spec", "mirror ok: filled [%d, %d) watermark=%d",
            begin, exec->n_kv_pos, m_mirror_pos);
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
    auto* exec = m_exec;
    auto* ctx  = m_ctx_tgt;
    if (!exec || !ctx || !params) return PVX_ERR_INVALID_ARG;
    if (exec->last_logits.empty()) return PVX_ERR_DECODE_FAILED;

    const bool is_ngram = m_mode != PVX_SPEC_DRAFT;
    const int n_vocab = llama_vocab_n_tokens(ctx->vocab);
    // Chain bound (upstream n_draft_max rule): remaining budget minus 1 so the
    // commit token always fits; never wider than the target's batch.
    int chain_cap = std::min(m_draft_max,
                             std::min((int) llama_n_batch(ctx->ctx) - 1,
                                      out_max - 1));
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
        const int round_base  = exec->n_kv_pos;

        // 1. ── Draft chain ──
        std::vector<llama_token> draft;
        if (max_draft > 0) {
            const float* last_logits = exec->last_logits.data();
            const TopLogit first_top = drafter_top_logit(last_logits, n_vocab);
            const bool first_usable =
                (m_p_min <= 0.0f ||
                 top_probability(last_logits, n_vocab, first_top.logit) >= m_p_min) &&
                !llama_vocab_is_eog(ctx->vocab, first_top.token);
            if (first_usable) {
                if (is_ngram) {
                    ngram_draft(draft, max_draft, first_top.token);
                    m_n_drafted += (int64_t) draft.size();
                } else {
                    draft.push_back(first_top.token);
                    m_n_drafted++;
                    // Draft token i sits at absolute position n_kv_pos + i.
                    llama_pos next_pos = exec->n_kv_pos;
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
            const llama_token t = spec_sample(exec->last_logits.data(), ctx->vocab,
                                              params, exec->recent_tokens);
            if (t < 0) return PVX_ERR_DECODE_FAILED;
            llama_batch b = llama_batch_init(1, 0, ctx->n_seq_max);
            b.token[0]     = t;
            b.pos[0]       = exec->n_kv_pos;
            b.n_seq_id[0]  = 1;
            b.seq_id[0][0] = 0;
            b.logits[0]    = 1;
            b.n_tokens     = 1;
            const int rc = llama_decode(ctx->ctx, b);
            llama_batch_free(b);
            if (rc != 0) {
                ctx->last_error = "spec fallback decode failed (rc=" + std::to_string(rc) + ")";
                return PVX_ERR_DECODE_FAILED;
            }
            spec_snapshot(ctx->ctx, exec->last_logits, 0);
            pvx_mirror_push_token(exec, t, exec->n_kv_pos);
            exec->n_tokens++;
            exec->n_kv_pos++;
            exec->recent_tokens.push_back(t);
            generated++;
            if (is_ngram) {
                std::vector<llama_token> stream;
                mirror_token_stream(stream);
                ngram_after_round(stream, 0);
            }
            if (on_token && !on_token(t, user_data)) break;
            if (llama_vocab_is_eog(ctx->vocab, t)) break;
            continue;
        }

        // 2. ── Verify on the target: ONE decode of [draft0..draft_{d-1}] ──
        const int n_draft = (int) draft.size();
        llama_batch b = llama_batch_init(n_draft, 0, ctx->n_seq_max);
        for (int i = 0; i < n_draft; i++) {
            b.token[i]     = draft[i];
            b.pos[i]       = exec->n_kv_pos + i;
            b.n_seq_id[i]  = 1;
            b.seq_id[i][0] = 0;
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
            std::vector<llama_token> feed = exec->recent_tokens; // ends at prev_last
            llama_token s = spec_sample(exec->last_logits.data(), ctx->vocab,
                                        params, feed);
            if (s < 0) return PVX_ERR_DECODE_FAILED;
            int i = 0;
            for (; i < n_draft; i++) {
                if (s != draft[i]) break; // first disagreeing sample IS the commit
                accepted.push_back(draft[i]);
                m_n_accepted++;
                feed.push_back(draft[i]);
                if (llama_vocab_is_eog(ctx->vocab, draft[i])) {
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
        const int keep = exec->n_kv_pos + (int) accepted.size();
        if (!llama_memory_seq_rm(ctx->mem, 0, keep, -1)) {
            ctx->last_error = "spec KV repair refused by memory (partial seq_rm unsupported)";
            return PVX_ERR_REWIND_UNSUPPORTED;
        }

        if (eog_row >= 0) {
            // Mid-chain EOG: the generation ends with the accepted stream (EOG
            // last). The row after the EOG becomes the executor's next-step
            // last_logits — same contract as the plain loop decoding an EOG.
            exec->last_logits = std::move(rows[eog_row]);
            exec->n_tokens += (int) accepted.size();
            exec->n_kv_pos += (int) accepted.size();
            for (llama_token t : accepted) exec->recent_tokens.push_back(t);
            for (int i = 0; i < (int) accepted.size(); i++) {
                pvx_mirror_push_token(exec, accepted[i], round_base + i);
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
                if (llama_vocab_is_eog(ctx->vocab, t)) {
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
            rb.seq_id[0][0] = 0;
            rb.logits[0]    = 1;
            rb.n_tokens     = 1;
            const int rrc = llama_decode(ctx->ctx, rb);
            llama_batch_free(rb);
            if (rrc != 0) {
                ctx->last_error = "spec commit decode failed (rc=" + std::to_string(rrc) + ")";
                return PVX_ERR_DECODE_FAILED;
            }
            spec_snapshot(ctx->ctx, exec->last_logits, 0);
        }

        exec->n_tokens += (int) accepted.size() + 1;
        exec->n_kv_pos += (int) accepted.size() + 1;
        for (llama_token t : accepted) exec->recent_tokens.push_back(t);
        exec->recent_tokens.push_back(commit);
        for (int i = 0; i < (int) accepted.size(); i++) {
            pvx_mirror_push_token(exec, accepted[i], round_base + i);
        }
        pvx_mirror_push_token(exec, commit, keep);
        if (is_ngram) {
            std::vector<llama_token> stream;
            mirror_token_stream(stream);
            ngram_after_round(stream, (int) accepted.size());
        }

        // Stream/EOG per committed token (upstream processes ids in order).
        pvx_log(PVX_LOG_DEBUG, "Spec", "round end: n_kv_pos=%d n_tokens=%d committed=%zu",
                exec->n_kv_pos, exec->n_tokens, accepted.size() + 1);
        for (llama_token t : accepted) {
            generated++;
            if (on_token && !on_token(t, user_data)) {
                if (out_generated) *out_generated = generated;
                return PVX_OK;
            }
            if (llama_vocab_is_eog(ctx->vocab, t)) {
                if (out_generated) *out_generated = generated;
                return PVX_OK;
            }
        }
        generated++;
        if (on_token && !on_token(commit, user_data)) break;
        if (llama_vocab_is_eog(ctx->vocab, commit)) break;
    }

    if (out_generated) *out_generated = generated;
    return PVX_OK;
}
