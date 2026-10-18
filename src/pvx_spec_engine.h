// pvx_spec_engine.h — Standard-executor speculative-decoding session (INF BL-05).
//
// One PvxBLSpecEngine per executor, owned by pvx_executor_s::spec (unique_ptr),
// created via pvx_executor_spec_init and torn down in pvx_executor_free /
// pvx_executor_spec_free. The pvxinference.cpp layer only validates, binds, and
// tears down; ALL verify-loop mechanics live in this class.
//
// Scope (INF BL-05 + ABI v6 wave 1):
//   - PVX_SPEC_DRAFT: separate draft model, upstream draft-simple semantics —
//     verification by target-side sampling (sample-and-match over the executor's
//     own pipeline), verify batch = draft chain decoded in ONE target call,
//     KV repair to the accepted prefix, commit token = the first disagreeing
//     sample (or the row after full acceptance), identical recent-tokens and
//     position bookkeeping to the plain loop.
//   - Draft-context mirroring (EMBD-AWARE, ABI v6): the draft model is fed the
//     target's committed stream from the executor's mirror ring
//     (pvx_mirror_entry_t): plain token rows by id, multimodal rows by
//     projected embedding at their pinned positions — the pattern of the
//     vendored draft-simple batch builder (token row vs embd row, zero-pad +
//     one-time warn on draft input-width mismatch, never an abort; the target
//     still verifies the real row). The old plain-token replay desynced after
//     any image prompt (positions advanced with no ids to replay); the ring's
//     invariant mirror.size() == n_tokens is enforced fail-closed by
//     mirror_consistent() on every round.
//   - PVX_SPEC_NGRAM_SIMPLE / MAP_K / MOD / CACHE (ABI v6): drafting from the
//     committed stream itself — no draft model. The n-gram machinery is a
//     pattern-port of the vendored common/ngram-* implementation into
//     src/pvx_ngram.{h,cpp}; the verify loop is unchanged (drafts verify
//     against the target exactly like draft-model chains).
//   - PVX_SPEC_MTP_AUTO: typed NOT_SUPPORTED boundary. The model-level MTP
//     capability is reported honestly (nextn layers present in the GGUF /
//     arch), but the drafter requires LLAMA_CONTEXT_TYPE_MTP context plumbing
//     + nextn hidden-state extraction this specialist does not construct yet —
//     configured MTP fails typed, never faked.

#pragma once

#include "pvx_internal.h"
#include "pvx_ngram.h"

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
    // draft-load | spec-engine/draft-vocab | spec-engine/draft-context |
    // spec-engine/ngram-config | spec-engine/ngram-cache).
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
    // last_logits); loops draft → verify → commit until EOG (unless the
    // sampling params set ignore_eos — then EOG tokens are sampled through
    // exactly like the plain pipeline), out_max generated tokens, stream-stop,
    // or typed failure. Committed tokens land in the executor state exactly
    // as the plain loop would commit them.
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

    // Mirror the target's committed stream onto the draft context. Embd-aware:
    // token rows feed the draft by id, image rows by projected embedding at
    // their pinned positions (zero-pad + one-time warn when the draft input
    // width differs from the stored row width — upstream behavior, quality
    // warning only, never an abort).
    // Valid (append-only) mirror = draft KV positions [m_mirror_base,
    // m_mirror_pos) matching the executor's window base; a target-state
    // reduction (rewind/reset/restore) or a base change (shift_left keeps
    // positions absolute) invalidates the mirror → the whole visible window
    // is rebuilt. Returns false on typed decode failure.
    bool mirror_prefix();

    // Fail-closed mirror-consistency guard (ABI v6): the ring must cover
    // exactly the executor's committed window — mirror.size() == n_tokens,
    // positions contiguous ascending, front == n_kv_pos - n_tokens, back ==
    // n_kv_pos - 1. A violation means a commit path forgot the ring; spec
    // generation refuses (typed) rather than drafting against a phantom
    // stream. Returns false with `why` naming the broken invariant.
    bool mirror_consistent(std::string& why) const;

    // Draft-ctx mechanics (static members: no external state):
    //   draft_decode — one-token decode with logits enabled
    static bool draft_decode(PvxBLSpecEngine& eng, llama_token tok, llama_pos pos);

    // ── n-gram drafting (ABI v6, no draft model) ──
    // Produce a draft chain from the committed stream (mirror token entries
    // + the target's top-1 as the lookup tail). Returns the number of tokens
    // pushed; 0 = no usable draft (caller falls through to the plain step).
    int  ngram_draft(std::vector<llama_token>& draft, int max_draft, llama_token sampled);
    // Feed newly committed stream tokens into the mode's online state and
    // update acceptance statistics (map_k value bookkeeping).
    void ngram_after_round(const std::vector<llama_token>& stream, int n_accepted);
    // Rebuild the committed token stream (ids only) from the mirror ring.
    void mirror_token_stream(std::vector<llama_token>& out) const;

    // Draft-model resources (mode == PVX_SPEC_DRAFT). Teardown order matters:
    // context first, then model (backend device ownership).
    llama_model*   m_model_dft = nullptr;
    llama_context* m_ctx_dft   = nullptr;

    // Mirror watermark + base (see mirror_prefix).
    int m_mirror_pos  = 0;
    int m_mirror_base = 0;
    // One-time draft-width mismatch warning (upstream zeros_warned).
    bool m_width_warned = false;

    // Non-owning back-references (executor/context outlive the session).
    pvx_model_t*    m_model_tgt = nullptr;
    pvx_context_t*  m_ctx_tgt   = nullptr;
    pvx_executor_t* m_exec      = nullptr; // bound owner (spec_init)

    // Session mode (PVX_SPEC_DRAFT + the n-gram modes are constructible).
    pvx_spec_mode_t m_mode = PVX_SPEC_OFF;

    // Upstream draft-simple knobs (defaults mirror common_params_speculative).
    int   m_draft_max = 16;
    int   m_draft_min = 0;
    float m_p_min     = 0.0f;

    // n-gram state (mode == PVX_SPEC_NGRAM_*).
    pvx::ngram_simple_config m_ng_simple{};
    std::unique_ptr<pvx::ngram_map>  m_ng_map;    // NGRAM_MAP_K
    std::unique_ptr<pvx::ngram_mod>   m_ng_mod;    // NGRAM_MOD
    size_t m_ng_mod_i_last    = 0;  // last stream index fed into the mod table
    size_t m_ng_mod_n_draft  = 0;  // length of the last mod draft (accept stats)
    std::unique_ptr<pvx::ngram_cache> m_ng_cache_ctx;      // NGRAM_CACHE: context+dynamic caches
    std::unique_ptr<pvx::ngram_cache> m_ng_cache_dynamic;
    std::unique_ptr<pvx::ngram_cache> m_ng_cache_static;   // from cache_path (optional)
    int  m_ng_cache_n_min = 1;
    int  m_ng_cache_n_max = 2;
    size_t m_ng_cache_fed_ctx = 0;      // stream prefix already fed into m_ng_cache_ctx
    size_t m_ng_cache_fed_dyn = (size_t) -1; // SIZE_MAX = not yet baselined
    std::string m_ng_cache_path;        // save-back path (empty = none)

    std::string m_last_error;
};
