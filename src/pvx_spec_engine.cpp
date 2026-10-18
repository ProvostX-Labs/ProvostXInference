// pvx_spec_engine.cpp — Standard-executor speculative-decoding session.
// See pvx_spec_engine.h for scope, semantics, and invariants.

#include "pvx_spec_engine.h"

#include <algorithm>
#include <cmath>
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
            break;
        default:
            out_error = "unknown spec mode " + std::to_string((int) params->mode);
            return nullptr;
    }

    pvx_spec_capabilities_t caps{};
    if (pvx_model_spec_capabilities(model, &caps) != PVX_OK || !caps.draft) {
        out_error = "PVX_SPEC_DRAFT rejected by capability report: draft-simple spec requires "
                    "a dense (non-hybrid, non-recurrent) target memory with a standard "
                    "vocab; this model's supported-types list omits draft_simple "
                    "(boundary: spec-engine/target-memory)";
        return nullptr;
    }

    if (!params->draft_model_path || params->draft_model_path[0] == '\0') {
        out_error = "PVX_SPEC_DRAFT configured without draft_model_path; a draft-model "
                    "session without a draft model is fail-closed, not silently "
                    "disabled (boundary: spec-engine/draft-config)";
        return nullptr;
    }

    auto engine = std::unique_ptr<PvxBLSpecEngine>(new PvxBLSpecEngine());
    engine->m_mode      = PVX_SPEC_DRAFT;
    engine->m_model_tgt = model;
    engine->m_ctx_tgt   = ctx;
    engine->m_exec      = nullptr; // bound at spec_init time
    engine->m_draft_max = params->draft_max > 0 ? params->draft_max : kDraftMaxDefault;
    engine->m_draft_min = params->draft_min > 0 ? params->draft_min : kDraftMinDefault;
    engine->m_p_min     = params->p_min >= 0.0f ? params->p_min : kPMinDefault;

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

bool PvxBLSpecEngine::mirror_prefix() {
    auto* exec = m_exec;
    auto* ctx  = m_ctx_tgt;
    // shift_left keeps KV positions absolute (n_tokens shrinks, n_kv_pos does
    // not): position of recent_tokens[i] is (n_kv_pos - n_tokens) + i.
    const int pos_base = exec->n_kv_pos - exec->n_tokens;
    // Validity: the mirror tracks the target's committed stream append-only.
    // A base change (shift_left) or a shrunken target (rewind/reset/restore)
    // invalidates the whole mirror → rebuild from the visible window.
    const bool incremental = (m_mirror_base == pos_base && m_mirror_pos <= exec->n_kv_pos);
    llama_memory_t dmem = llama_get_memory(m_ctx_dft);
    if (!dmem || !llama_memory_seq_rm(dmem, 0, incremental ? m_mirror_pos : 0, -1)) {
        pvx_log(PVX_LOG_ERROR, "Spec", "draft mirror: seq_rm refused (rebuild=%d)", !incremental);
        return false;
    }
    pvx_log(PVX_LOG_DEBUG, "Spec", "mirror start: rebuild=%d watermark=%d n_kv_pos=%d n_tokens=%d",
            !incremental, m_mirror_pos, exec->n_kv_pos, exec->n_tokens);
    const int begin  = incremental ? m_mirror_pos : pos_base;
    const int n_fill = exec->n_kv_pos - begin;
    if (n_fill <= 0) {
        m_mirror_pos  = exec->n_kv_pos;
        m_mirror_base = pos_base;
        return true;
    }

    const uint32_t d_batch = llama_n_batch(m_ctx_dft);
    int filled = 0;
    while (filled < n_fill) {
        const int slice = std::min<int>((int) d_batch, n_fill - filled);
        llama_batch b = llama_batch_init(slice, 0, 1);
        for (int i = 0; i < slice; i++) {
            b.token[i]     = exec->recent_tokens[begin - pos_base + filled + i];
            b.pos[i]       = begin + filled + i;
            b.n_seq_id[i]  = 1;
            b.seq_id[i][0] = 0;
            b.logits[i]    = 0; // mirror only — drafting seeds come from the target row
        }
        b.n_tokens = slice;
        const int rc = llama_decode(m_ctx_dft, b);
        llama_batch_free(b);
        if (rc != 0) {
            pvx_log(PVX_LOG_ERROR, "Spec", "draft mirror decode failed (%d tokens, rc=%d)",
                    slice, rc);
            return false;
        }
        filled += slice;
    }
    m_mirror_pos  = exec->n_kv_pos;
    m_mirror_base = pos_base;
    pvx_log(PVX_LOG_DEBUG, "Spec", "mirror ok: filled [%d, %d) watermark=%d",
            begin, exec->n_kv_pos, m_mirror_pos);
    return true;
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
//   0. mirror_prefix(): draft ctx holds the committed stream for positions
//      [m_mirror_base, m_mirror_pos); stale cells dropped, missing tokens
//      decoded (full rebuild when the target state shrank or shifted).
//   1. draft[0] comes from the TARGET's freshest row (exec->last_logits) —
//      the verifier's first compare uses the same row the plain loop samples,
//      so the first draft is target-greedy-agreeing at temp 0. Continuations
//      are greedy top-1 decodes on the draft ctx at positions n_kv_pos …
//      n_kv_pos+d-1 — the same positions the verify decode fills on the
//      target — gated by the draft-side top probability (p_min) and EOG.
//      Chain bounded by draft_max, the target's batch width, n_ctx room for
//      the commit row, and the remaining output budget (upstream n_draft_max
//      rule: remaining-1 so the commit token always fits).
//   2. verify batch = [draft0..draft_{d-1}] decoded on the TARGET in ONE
//      call, every row logits-enabled. prev_last is NOT re-decoded: it is
//      already committed at n_kv_pos-1 (upstream re-decodes id_last because
//      its loop keeps the last token out of KV; the pvx executor commits it,
//      so the fresh row is exec->last_logits).
//   3. sample-and-match with the executor pipeline: s0 = sample(last_logits)
//      vs draft0; s_{i+1} = sample(row_i) vs draft_{i+1}; the first
//      disagreeing sample IS the commit token (it is the target's sample of
//      the row after the accepted prefix) — upstream sample_and_accept_n
//      semantics. Penalty feed sees exactly the accepted prefix in both
//      paths.
//   4. KV repair: seq_rm past the accepted prefix — drops the unverified
//      draft cells; the accepted cells are already at the right positions
//      from the verify decode (upstream kv_remove(n_past+n_accept, -1)).
//   5. commit: ONE decode of the commit token (logits row becomes the
//      executor's next-step last_logits — same contract as pvx_executor_infer)
//      and counters/recent_tokens advanced exactly like the plain loop's
//      per-token commits (penalty-window equivalence).

pvx_result_t PvxBLSpecEngine::generate(const pvx_sampling_params_t* params,
                                       int out_max,
                                       pvx_spec_token_callback_t on_token,
                                       void* user_data,
                                       int* out_generated) {
    auto* exec = m_exec;
    auto* ctx  = m_ctx_tgt;
    if (!exec || !ctx || !params) return PVX_ERR_INVALID_ARG;
    if (exec->last_logits.empty()) return PVX_ERR_DECODE_FAILED;

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
        // 0. ── Mirror the committed stream onto the draft ctx ──
        if (!mirror_prefix()) return PVX_ERR_DECODE_FAILED;

        const int n_remaining = out_max - generated;
        const int max_draft   = std::min(chain_cap, n_remaining - 1);

        // 1. ── Draft chain (seed from the target row; greedy top-1 + p_min + EOG) ──
        std::vector<llama_token> draft;
        if (max_draft > 0) {
            const float* last_logits = exec->last_logits.data();
            const TopLogit first_top = drafter_top_logit(last_logits, n_vocab);
            const bool first_usable =
                (m_p_min <= 0.0f ||
                 top_probability(last_logits, n_vocab, first_top.logit) >= m_p_min) &&
                !llama_vocab_is_eog(ctx->vocab, first_top.token);
            if (first_usable) {
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

        if ((int) draft.size() < m_draft_min) draft.clear();

        if (draft.empty()) {
            // Drafter cannot help here (EOG/p_min floor/budget): fall through
            // to the plain single-token step — honest progress with the same
            // sampler. Mirrors the plain loop's commit exactly; the next
            // round's mirror_prefix carries the token to the draft ctx.
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
            exec->n_tokens++;
            exec->n_kv_pos++;
            exec->recent_tokens.push_back(t);
            generated++;
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
        // s0 samples last_logits (row after prev_last) vs draft0; s_{i+1}
        // samples row_i (row after draft_i) vs draft_{i+1}; the first
        // disagreeing sample is the commit token (the target's sample of the
        // row after the accepted prefix); full acceptance takes a fresh
        // tail sample from row_{d-1}. Penalty feed = accepted prefix.
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
                    // Mid-chain EOG: accepted, nothing commits past an EOG
                    // (upstream stops the accept loop; the stream ends here).
                    // The row after the EOG is already decoded — keep it as the
                    // executor's next-step last_logits.
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

        // 4. ── KV repair: keep the accepted prefix (positions 0 … keep − 1) —
        //    the accepted draft cells are already at the right positions from
        //    the verify decode; only the unverified cells are dropped.
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
        //    executor's next-step last_logits, same contract as pvx_executor_infer).
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
