// pvx_spec_engine.h — Speculative-decoding session (INF BL-05 + ABI v6/v7 waves).
//
// One PvxBLSpecEngine per stream, owned by pvx_executor_s::spec (executor lane)
// or pvx_conversation_s::spec (batched/pool lane, ABI v7 feature 5), created
// via pvx_executor_spec_init / pvx_conversation_spec_init and torn down in the
// respective free paths. The pvxinference.cpp layer only validates, binds, and
// tears down; ALL verify-loop mechanics live in this class.
//
// Both lanes operate through pvx_spec_stream_s (a pointer view of ONE committed
// stream): the verify loop, mirror replay, KV repair and bookkeeping are a
// single implementation — token-exactness between lanes follows structurally.
//
// Scope:
//   - PVX_SPEC_DRAFT: separate draft model, upstream draft-simple semantics —
//     verification by target-side sampling (sample-and-match over the stream's
//     own pipeline), verify batch = draft chain decoded in ONE target call,
//     KV repair to the accepted prefix, commit token = the first disagreeing
//     sample (or the row after full acceptance), identical recent-tokens and
//     position bookkeeping to the plain loop.
//   - Draft-context mirroring (EMBD-AWARE, ABI v6): the draft model is fed the
//     target's committed stream from the mirror ring (pvx_mirror_entry_t):
//     plain token rows by id, multimodal rows by projected embedding at their
//     pinned positions — the pattern of the vendored draft-simple batch
//     builder (token row vs embd row, zero-pad + one-time warn on draft input
//     width mismatch, never an abort; the target still verifies the real
//     row). Invariant mirror.size() == n_tokens enforced fail-closed by
//     mirror_consistent() on every round.
//   - PVX_SPEC_NGRAM_SIMPLE / MAP_K / MOD / CACHE (ABI v6): drafting from the
//     committed stream itself — no draft model (pattern-port of the vendored
//     common/ngram-* into src/pvx_ngram.{h,cpp}).
//   - PVX_SPEC_EAGLE3 (ABI v7 feature 4): pattern-port of the vendored
//     common_speculative_impl_draft_eagle3. The draft decoder consumes
//     (token, g_embd) pairs; g rows are the EAGLE3 encoder output of the
//     target's extracted layer features (llama_set_embeddings_layer_inp on
//     the target context — a compute flag only, KV-neutral). Every committed
//     position carries one g row (g_hist, aligned with the mirror); a
//     per-position deferred boundary (the (token, g) pair needing the NEXT
//     sample) is completed at each round's seed. Fail-closed boundaries:
//     companion metadata (eagle3.target_layers, exactly 3), vocab family,
//     target layer ids, non-empty stream at enable (feature capture must see
//     every committed row: spec_init requires n_tokens == 0), multimodal rows
//     (vision prompts refuse typed), and the batched/pool lane (conversation
//     spec_init refuses typed). g_hist rides inside pvx_state_save/restore
//     (additive payload) so restore never resumes against a stale boundary.
//   - PVX_SPEC_MTP_AUTO / PVX_SPEC_DFLASH: typed NOT_SUPPORTED boundaries —
//     capabilities/metadata are validated honestly, the drafter wiring is not
//     built in this slice; configured modes fail typed, never faked.

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
    // spec-engine/dflash-wiring | spec-engine/eagle3-companion-load |
    // spec-engine/eagle3-companion-metadata | spec-engine/eagle3-companion-layers
    // spec-engine/target-memory | spec-engine/draft-config | spec-engine/
    // draft-load | spec-engine/draft-vocab | spec-engine/draft-context |
    // spec-engine/ngram-config | spec-engine/ngram-cache).
    // The caller maps the message's boundary tag to the typed result code.
    static std::unique_ptr<PvxBLSpecEngine> create(pvx_model_t* model,
                                                   pvx_context_t* ctx,
                                                   const pvx_spec_params_t* params,
                                                   std::string& out_error);

    // Bind the owning executor / conversation (called once by the respective
    // spec_init after successful construction). The draft context is empty
    // here; the first generate round mirrors the committed stream lazily.
    void bind(pvx_executor_t* exec);
    void bind_conversation(pvx_conversation_t* conv);

    // Last typed boundary message for pvx_last_error surfacing.
    const std::string& last_error() const { return m_last_error; }

    // Session mode accessor (pvxinference.cpp checks it for lane-specific
    // fail-closed refusals, e.g. EAGLE3 + vision prompts).
    pvx_spec_mode_t mode() const { return m_mode; }

    // Spec-session acceptance telemetry.
    int64_t drafted() const { return m_n_drafted; }
    int64_t accepted() const { return m_n_accepted; }

    // Speculative token loop continuing from the stream state (see
    // pvx_executor_generate_spec / pvx_conversation_generate_spec docs).
    pvx_result_t generate(const pvx_sampling_params_t* params,
                          int out_max,
                          pvx_spec_token_callback_t on_token,
                          void* user_data,
                          int* out_generated);

    // ── EAGLE3 capture hook (executor lane, ABI v7) ──
    // Called by commit sites right AFTER a successful target decode on the
    // bound stream: captures the decoded batch's per-layer input features,
    // runs the EAGLE3 encoder (draft context), appends one g row per
    // committed position to g_hist and writes the draft decoder's training
    // pairs (token[k+1], g[k]) at pos[k] for all but the last row (the
    // deferred boundary — completed by the next round's seed, exactly the
    // upstream process() pattern). No-op for non-EAGLE3 modes.
    // row_ids = the ids of the decoded batch rows, in order.
    bool eagle_process_target_rows(llama_pos pos_first, int n_rows,
                                   const llama_token* row_ids);

    // The stream shrank (rewind / partial restore) with an EAGLE3 session
    // active: truncate g_hist to the new committed count and drop draft-side
    // cells past the boundary. No-op for non-EAGLE3 modes.
    void eagle_on_stream_shrunk(int new_n_tokens);

    // EAGLE3 state handoff for pvx_state_save: fills the additive payload
    // (one g row per committed position). No-op/absent for other modes.
    void eagle_state_capture(std::vector<std::vector<float>>& out_rows) const;

    // EAGLE3 restore (additive ABI v7 payload): rebuild the draft decoder
    // from a saved g_hist — full training-pair replay [0, n_tokens), the
    // last row deferred. Returns false (typed message in last_error()) when
    // the payload does not match the restored stream — the caller must then
    // invalidate the session, never resume against a stale boundary.
    bool eagle_state_restore(std::vector<std::vector<float>>&& rows);

    // True when this session keeps a draft context whose KV is only repairable
    // through the mirror/g_hist (used by rewind/save sites to decide whether
    // shrink hooks must run).
    bool is_eagle() const { return m_mode == PVX_SPEC_EAGLE3; }

    ~PvxBLSpecEngine();

private:
    PvxBLSpecEngine() = default;
    int64_t m_n_drafted = 0;
    int64_t m_n_accepted = 0;

    // Mirror the target's committed stream onto the draft context (draft /
    // ngram modes). Embd-aware: token rows by id, image rows by projected
    // embedding at pinned positions (zero-pad + one-time warn on width
    // mismatch — upstream behavior, quality warning only, never an abort).
    bool mirror_prefix();

    // Fail-closed mirror-consistency guard (ABI v6): the bound stream's ring
    // must cover exactly its committed window. Returns false with `why`
    // naming the broken invariant.
    bool mirror_consistent(std::string& why) const;

    // Draft-ctx mechanics:
    //   draft_decode — one-token decode with logits enabled
    static bool draft_decode(PvxBLSpecEngine& eng, llama_token tok, llama_pos pos);
    //   eagle_pair_decode — one (token, g_embd) row at pos, output enabled
    bool eagle_pair_decode(llama_token tok, const float* g_row, llama_pos pos);
    //   stream_push_token — one token row into the bound stream's mirror ring
    void stream_push_token(llama_token id, llama_pos pos);

    // ── n-gram drafting (ABI v6, no draft model) ──
    int  ngram_draft(std::vector<llama_token>& draft, int max_draft, llama_token sampled);
    void ngram_after_round(const std::vector<llama_token>& stream, int n_accepted);
    void mirror_token_stream(std::vector<llama_token>& out) const;

    // ── EAGLE3 internals ──
    // One round of the EAGLE3 loop (upstream draft()/accept() pattern adapted
    // to the pvx verify pipeline). Returns via pvx_result_t like generate().
    pvx_result_t generate_eagle(const pvx_sampling_params_t* params,
                                 int out_max,
                                 pvx_spec_token_callback_t on_token,
                                 void* user_data,
                                 int* out_generated);

    // Draft-model resources (modes PVX_SPEC_DRAFT / PVX_SPEC_EAGLE3). Teardown
    // order matters: context first, then model (backend device ownership).
    llama_model*   m_model_dft = nullptr;
    llama_context* m_ctx_dft   = nullptr;

    // Mirror watermark + base (see mirror_prefix).
    int m_mirror_pos  = 0;
    int m_mirror_base = 0;
    // One-time draft-width mismatch warning (upstream zeros_warned).
    bool m_width_warned = false;

    // Non-owning back-references (target outlives the session).
    pvx_model_t*   m_model_tgt = nullptr;
    pvx_context_t* m_ctx_tgt   = nullptr;

    // Bound stream view (executor or conversation lane).
    pvx_spec_stream_s m_stream{};

    // Session mode.
    pvx_spec_mode_t m_mode = PVX_SPEC_OFF;

    // Upstream draft-simple knobs (defaults mirror common_params_speculative).
    int   m_draft_max = 16;
    int   m_draft_min = 0;
    float m_p_min     = 0.0f;

    // ── EAGLE3 state (mode == PVX_SPEC_EAGLE3) ──
    int  m_eagle_n_embd_dec = 0;   // draft hidden size (g row width)
    int  m_eagle_n_embd_enc = 0;   // target_layer_ids_n * n_embd_tgt (encoder input)
    int  m_eagle_n_embd_tgt = 0;
    int  m_eagle_n_layer_tgt = 0;
    int  m_eagle_layer_ids[4] = { -1, -1, -1, -1 }; // extracted target layers
    int  m_eagle_n_layers = 0;                      // == 3 (upstream check)
    // One encoder-output g row per committed position [0, n_tokens).
    // INVariant: g_hist.size() == *m_stream.n_tokens at round boundaries.
    std::vector<std::vector<float>> m_eagle_g_hist;
    // Scratch: [n_rows * n_embd_enc] feature buffer, [n_rows * n_embd_dec] g rows.
    std::vector<float> m_eagle_features_buf;
    std::vector<float> m_eagle_g_buf;

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
