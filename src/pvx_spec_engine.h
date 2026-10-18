// pvx_spec_engine.h — Standard-executor speculative-decoding session (INF BL-05).
//
// One PvxBLSpecEngine per executor, owned by pvx_executor_s::spec (unique_ptr),
// created via pvx_executor_spec_init and torn down in pvx_executor_free /
// pvx_executor_spec_free. The pvxinference.cpp layer only validates, binds, and
// tears down; ALL verify-loop mechanics live in this class.
//
// Scope (INF BL-05):
//   - PVX_SPEC_DRAFT: separate draft model, upstream draft-simple semantics —
//     verification by target-side sampling (sample-and-match over the executor's
//     own pipeline), verify batch = draft chain decoded in ONE target call,
//     KV repair to the accepted prefix, commit token = the first disagreeing
//     sample (or the row after full acceptance), identical recent-tokens and
//     position bookkeeping to the plain loop.
//   - Draft-context mirroring: the draft model is fed the target's committed
//     token stream (same ids, same absolute positions) at round start
//     (deferred mirroring: the first round prefills the full history, later
//     rounds decode just the tokens committed since — cost is one tiny-model
//     pass per round; no checkpoint machinery), then drafting extends it
//     greedily.
//   - PVX_SPEC_MTP_AUTO: typed NOT_SUPPORTED boundary. The model-level MTP
//     capability is reported honestly (nextn layers present in the GGUF /
//     arch), but the drafter requires LLAMA_CONTEXT_TYPE_MTP context plumbing
//     + nextn hidden-state extraction this specialist does not construct yet —
//     configured MTP fails typed, never faked.

#pragma once

#include "pvx_internal.h"

#include <memory>
#include <string>
#include <vector>

struct llama_context;
struct llama_model;

class PvxBLSpecEngine {
public:
    // Fail-closed construction: returns nullptr with out_error carrying the
    // typed boundary message (spec-engine/config | spec-engine/mtp-wiring |
    // spec-engine/target-memory | spec-engine/draft-config | spec-engine/
    // draft-load | spec-engine/draft-vocab | spec-engine/draft-context).
    // The caller maps the message's boundary tag to the typed result code.
    static std::unique_ptr<PvxBLSpecEngine> create(pvx_model_t* model,
                                                   pvx_context_t* ctx,
                                                   const pvx_spec_params_t* params,
                                                   std::string& out_error);

    // Bind the owning executor (called once by pvx_executor_spec_init after
    // successful construction). The draft context is empty here; the first
    // generate round mirrors the executor's committed stream lazily.
    void bind(pvx_executor_t* exec) { m_exec = exec; }

    // Last typed boundary message for pvx_last_error surfacing.
    const std::string& last_error() const { return m_last_error; }

    // Spec-session acceptance telemetry (drafted vs verified-accepted since
    // the last generate call; 0/0 when no session is attached).
    int64_t drafted() const { return m_n_drafted; }
    int64_t accepted() const { return m_n_accepted; }

    // Speculative token loop continuing from the executor state: the prefill
    // must be complete (executor entry point enforces a non-empty
    // last_logits); loops draft → verify → commit until EOG, out_max
    // generated tokens, stream-stop, or typed failure. Committed tokens land
    // in the executor state exactly as the plain loop would commit them.
    pvx_result_t generate(const pvx_sampling_params_t* params,
                          int out_max,
                          pvx_spec_token_callback_t on_token,
                          void* user_data,
                          int* out_generated);

    ~PvxBLSpecEngine();

private:
    PvxBLSpecEngine() = default;
    int64_t m_n_drafted = 0;
    int64_t m_n_accepted = 0;

    // Mirror the target's committed token stream onto the draft context.
    // Valid (append-only) mirror = positions [m_mirror_base, m_mirror_pos)
    // with m_mirror_base == current (n_kv_pos - n_tokens) and m_mirror_pos ==
    // the n_kv_pos at the last mirror. Per round: drop cells ≥ m_mirror_pos
    // (stale drafting), decode the missing committed tokens chunked by the
    // draft batch size (no logits — drafting seeds come from the target row).
    // A target-state reduction (rewind/reset/restore) or a base change
    // (shift_left keeps positions absolute) invalidates the mirror → the
    // whole visible window is rebuilt. Returns false on typed decode failure.
    bool mirror_prefix();

    // Draft-ctx mechanics (static members: no external state):
    //   draft_decode — one-token decode with logits enabled
    static bool draft_decode(PvxBLSpecEngine& eng, llama_token tok, llama_pos pos);

    // Draft-model resources (mode == PVX_SPEC_DRAFT). Teardown order matters:
    // context first, then model (backend device ownership).
    llama_model*   m_model_dft = nullptr;
    llama_context* m_ctx_dft   = nullptr;

    // Mirror watermark + base (see mirror_prefix).
    int m_mirror_pos  = 0;
    int m_mirror_base = 0;

    // Non-owning back-references (executor/context outlive the session).
    pvx_model_t*    m_model_tgt = nullptr;
    pvx_context_t*  m_ctx_tgt   = nullptr;
    pvx_executor_t* m_exec      = nullptr; // bound owner (spec_init)

    // Session mode (only PVX_SPEC_DRAFT constructible today).
    pvx_spec_mode_t m_mode = PVX_SPEC_OFF;

    // Upstream draft-simple knobs (defaults mirror common_params_speculative_draft).
    int   m_draft_max = 16;
    int   m_draft_min = 0;
    float m_p_min     = 0.0f;

    std::string m_last_error;
};
