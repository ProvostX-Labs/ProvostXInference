// pvx_internal.h — Internal structs shared between pvxinference.cpp and pvx_backend.cpp
// NOT part of the public API (pvxinference.h).

#ifndef PVX_INTERNAL_H
#define PVX_INTERNAL_H

#include "pvxinference.h"
#include "llama.h"
#include "ggml.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <string>
#include <vector>
#include <array>
#include <stack>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <set>

// ── Global destruction synchronization ──
//
// Metal (and potentially other backends) assert that all GPU resources
// (buffers, residency sets) are freed before the device is freed.
// pvx_free_context frees llama_context (which owns Metal buffers), and
// pvx_free_model frees llama_model (which owns Metal devices). If these
// run concurrently on different threads (e.g. .NET SafeHandle finalizers
// during process exit), the device destructor asserts because buffers
// haven't been freed yet.
//
// The destruction gate ensures:
// 1. pvx_free_context acquires the gate, frees the context, releases the gate
// 2. pvx_free_model acquires the gate, waits for all contexts to be freed,
//    then frees the model
// This is backend-agnostic — it protects ANY backend that requires
// ordered teardown (Metal, CUDA, Vulkan all have this constraint).
//
// The context count is per-model: pvx_free_model blocks until its
// model's context count reaches zero. This preserves parallelism —
// contexts from DIFFERENT models can be freed concurrently.

struct pvx_model_s {
    llama_model* model = nullptr;
    const llama_vocab* vocab = nullptr;
    mtmd_context* mtmd_ctx = nullptr;
    // For mmproj WRAPPERS: the text model whose mtmd_ctx this wrapper shares.
    // Null on real models. pvx_free_mmproj decrements the owner's refcount.
    pvx_model_t* mmproj_owner = nullptr;
    std::string last_error;
    // Config from pvx_model_params_t, read by pvx_create_context
    bool flash_attn = true;
    pvx_kv_type_t kv_cache_type = PVX_KV_F16;
    int threads = -1;  // -1 = auto
    // Reasoning-marker probe list (data-driven, owner-approved 2026-09-26):
    // tag strings probed against the vocabulary at template time; the first
    // single-special-token match drives the empty-block prefill.
    std::vector<std::string> reasoning_markers = { "<think>" };

    // Destruction synchronization: track how many contexts are alive for this model.
    // pvx_free_model waits on this CV until the count hits zero.
    // Requested gpu_layers from pvx_load_model (the CALLER's config) — read by
    // pvx_backend_fill_profile for truthful per-context gpu-layer visibility.
    int gpu_layers = 0;

    int active_context_count = 0;
    std::mutex destruction_mtx;
    std::condition_variable destruction_cv;
    // Shared mtmd context (vision): pvx_load_mmproj stores ONE mtmd_ctx here
    // AND on each mmproj wrapper. The refcount of live wrappers guards BOTH
    // directions of the lifetime: a second load while a wrapper is alive is
    // fail-closed (no UAF on the wrapper's shared pointer), and pvx_free_model
    // waits until every wrapper is freed before releasing mtmd_ctx.
    int mtmd_shared_refs = 0;   // live mmproj wrappers sharing model->mtmd_ctx
};

// Backend probe profile — what the probe discovered about THIS context's
// backend+config combination. Owned by the context (pvx_context_s::backend),
// never process-global: concurrent engines/probes on different contexts must
// not clobber each other (static-mutable-state rule).
enum pvx_backend_type_e {
    PVX_BACKEND_CPU = 0,
    PVX_BACKEND_METAL,
    PVX_BACKEND_CUDA,
    PVX_BACKEND_VULKAN,
    // Accelerator device whose vendor tag we do not classify — reported by
    // its raw ggml device name so a probe NEVER misreports a GPU as CPU.
    PVX_BACKEND_GPU_OTHER,
};

struct pvx_backend_profile {
    pvx_backend_type_e type = PVX_BACKEND_CPU;
    bool probe_passed = false;
    std::string name;
    std::string probe_error;
    // What was adjusted from the caller's requested config (for logging/diagnostics)
    bool gpu_layers_reduced = false;
    bool flash_attn_disabled = false;
    bool kv_cache_downgraded = false;   // q8_0 → f16
    int  original_gpu_layers = 0;
    int  effective_gpu_layers = 0;
    // Vulkan-specific: stale KV cells after clear (#26744)
    bool needs_kv_flush_after_clear = false;
};

struct pvx_context_s {
    llama_context* ctx = nullptr;
    llama_model* model = nullptr;       // raw llama_model pointer (non-owning)
    pvx_model_t* owning_model = nullptr; // back-reference for destruction accounting
    const llama_vocab* vocab = nullptr;
    llama_memory_t mem = nullptr;
    uint32_t n_ctx = 0;
    uint32_t n_seq_max = 0;
    pvx_pooling_type_t pool_type = PVX_POOL_NONE;
    std::string last_error;

    std::vector<pvx_conversation_t*> active_conversations;
    std::mutex infer_mtx;
    // Per-context backend probe state (detection half filled at context
    // creation by pvx_backend_fill_profile; decode-proven half by
    // pvx_probe_backend).
    pvx_backend_profile backend;
};

// Fill the detection half of a context's backend profile (backend type/name
// from the ggml device registry + gpu-layer fields from the owning model).
void pvx_backend_fill_profile(pvx_context_t* ctx);

// Per-executor/conversation grammar sampler holder. The chain MUST persist
// across sample calls within one generation (grammar is stateful — restarting
// from a fresh sampler per call makes the grammar expect position 0 forever).
// Rebuilt automatically when a different grammar string/root is passed.
//
// THREADING (LLM-24, 2026-10-01): the C# batched layer reaches one
// conversation's grammar state from MULTIPLE threads — the cycle/driver
// thread samples + rebuilds the chain (ApplyOp/StepAsync), while the owner's
// request thread resets it at generation boundaries (ResetGrammarState /
// checkpoint import), and a cancelled step lets the request thread proceed
// while a driver still holds the conversation in its ready set. ALL access to
// chain/grammar_str/grammar_root is serialized by this mutex. Without it, two
// racing resets both pass the non-null check and free the same
// llama_sampler* twice → `llama_grammar` dtor double-free under batch load.
struct pvx_grammar_state_s {
    std::mutex mtx;
    llama_sampler* chain = nullptr;
    std::string grammar_str;
    std::string grammar_root;
    ~pvx_grammar_state_s() { if (chain) llama_sampler_free(chain); }
};

// ── Embd-aware mirror entry ring (ABI v6 wave 1, spec-mirror substrate) ──
//
// One entry per COMMITTED KV position on the standard executor's stream
// (positions are the executor's own seq-0 window). A token row carries the
// committed token id; a multimodal row carries the projected embedding slice
// (mtmd output) with id == LLAMA_TOKEN_NULL — the pattern of the vendored
// draft-simple batch builder: token rows feed the draft by id, embd rows by
// pointer, and a draft whose input width differs from the target's receives
// a ZERO row plus a one-time warning (never an abort; the target still
// verifies the real row). Positions are pinned per entry (absolute KV
// positions, stable across shift_left which only frees the front).
//
// INVARIANT (spec-engine mirror-consistency guard): mirror.size() == n_tokens
// and the covered positions are exactly [n_kv_pos - n_tokens, n_kv_pos)
// (contiguous ascending, entry.pos pinned). Every site that advances or
// retreats the executor's committed window updates the ring in the same
// operation — there is no second code path.
struct pvx_mirror_entry_t {
    llama_token id = LLAMA_TOKEN_NULL; // LLAMA_TOKEN_NULL => embd row
    // Pinned position array, section-major [t, y, x, z] (M-RoPE order; for
    // non-M-RoPE streams all four equal the plain KV position).
    std::array<llama_pos, 4> pos = { 0, 0, 0, 0 };
    std::vector<float> embd;            // row payload iff id == LLAMA_TOKEN_NULL
};

struct pvx_conversation_s {
    llama_seq_id seq_id;
    llama_context* ctx;
    llama_memory_t mem;
    // Owning context (back-reference) — checkpoint export/import must hold
    // the context's infer mutex: seq-state serialization is not thread-safe
    // against concurrent llama_decode on the shared batched context.
    pvx_context_t* ctx_ref = nullptr;
    int n_tokens = 0;
    int n_kv_pos = 0;
    bool has_pending_prompt = false;
    std::vector<llama_token> pending_tokens;
    int last_batch_idx = -1;

    // ABI v7 (feature 5, batched-lane spec): per-conversation committed-stream
    // mirror ring — the conversation-lane twin of pvx_executor_s::mirror,
    // maintained by every conversation commit path (batched decode commit,
    // vision text/embd decodes, rewind, shift, reset, pool return, state
    // save/restore). Invariant mirror.size() == n_tokens enforced by the spec
    // engine's mirror-consistency guard before every round.
    std::vector<pvx_mirror_entry_t> mirror;

    // ABI v7 (feature 5): per-conversation spec session; created lazily via
    // pvx_conversation_spec_init, torn down in pool return / reset paths and
    // by the conversation destructor (unique_ptr: complete type where
    // pvx_conversation_s instances are deleted — pvxinference.cpp includes
    // pvx_spec_engine.h).
    std::unique_ptr<class PvxBLSpecEngine> spec;

    // ABI v8 (feature 12, stats export): prompt/decode counters kept by
    // the commit paths under the context infer mutex (snapshot semantics;
    // zero new locks). Additive fields only.
    int64_t n_prompt_tokens = 0;
    int64_t n_decode_steps = 0;
    // Snapshot of the last committed token's logits row, copied at decode
    // time. llama_get_logits_ith() rows are only addressable while their
    // batch is the shared context's CURRENT batch — any later decode on the
    // same ctx (another conversation's slice or cycle in batched inference)
    // replaces it, and sampling a stale row GGML_ABORTs the process
    // (get_logits_ith: "batch.logits[i] != true"). Sampling reads this
    // private copy, immune to interleaved decodes.
    std::vector<float> last_logits;

    // Rolling window of committed tokens for repeat/present penalties
    // (mirrors pvx_executor_s.recent_tokens)
    std::vector<llama_token> recent_tokens;

    // Persistent grammar sampler state (survives across sample calls within
    // one generation; rebuilt when a different grammar is passed)
    pvx_grammar_state_s grammar_state;
};

struct pvx_pool_s {
    pvx_context_t* ctx_ref;
    std::vector<std::unique_ptr<pvx_conversation_s>> conversations;
    std::stack<pvx_conversation_t*> available;
    std::mutex mtx;
};

struct pvx_executor_s {
    pvx_context_t* ctx_ref;
    // Speculative-decoding session (INF BL-05): owned; created lazily via
    // pvx_executor_spec_init, torn down in pvx_executor_free. Nullable.
    std::unique_ptr<class PvxBLSpecEngine> spec;

    int n_tokens = 0;
    int n_kv_pos = 0;
    bool has_pending = false;
    std::vector<llama_token> pending_tokens;
    int last_batch_idx = -1;
    // Private logits-row snapshot — same rationale as
    // pvx_conversation_s::last_logits (sampling must never read a batch row
    // that a later decode on the shared ctx may have invalidated).
    std::vector<float> last_logits;
    std::vector<llama_token> recent_tokens;
    // Embd-aware committed-stream mirror (see pvx_mirror_entry_t): the spec
    // engine replays THIS ring onto the draft context — plain token ids AND
    // multimodal embedding rows at their pinned positions. recent_tokens
    // alone cannot express image rows (no ids exist), which is why the old
    // id-only mirror desynced after any image prompt.
    std::vector<pvx_mirror_entry_t> mirror;
    pvx_grammar_state_s grammar_state;
    // ABI v8 (feature 12, stats export): prompt/decode counters kept by
    // the commit paths under the context infer mutex (snapshot semantics;
    // zero new locks). Additive fields only.
    int64_t n_prompt_tokens = 0;
    int64_t n_decode_steps = 0;
};

inline void pvx_mirror_push_token(pvx_executor_t* e, llama_token id, llama_pos pos) {
    pvx_mirror_entry_t ent;
    ent.id = id;
    ent.pos = { pos, pos, pos, pos };
    e->mirror.push_back(std::move(ent));
}

inline void pvx_mirror_push_embd(pvx_executor_t* e, const float* row, int n_embd,
                                 const std::array<llama_pos, 4>& pos) {
    pvx_mirror_entry_t ent;
    ent.id  = LLAMA_TOKEN_NULL;
    ent.pos = pos;
    ent.embd.assign(row, row + n_embd);
    e->mirror.push_back(std::move(ent));
}

struct pvx_state_s {
    int n_tokens;
    // Committed-stream mirror snapshot (ABI v6): taken at save, restored
    // verbatim (positions rebased to the restored window's [0, n) contract,
    // see pvx_state_restore) so an embd-aware spec session resumes against
    // the same rows the backup KV holds — including multimodal rows whose
    // embeddings are not recoverable from KV alone.
    std::vector<pvx_mirror_entry_t> mirror;
    // ABI v7 (feature 4, EAGLE3): per-position encoder-output g rows, aligned
    // with the committed stream [0, n_tokens). Present iff the saved executor
    // carried an active EAGLE3 session; absent (empty) in legacy checkpoints
    // and conversation-lane states. Restore with an active EAGLE3 session but
    // a mismatched/absent payload invalidates the session typed (the caller
    // must re-init spec) — never resumes against a stale deferred boundary.
    std::vector<std::vector<float>> eagle_g_rows;
};

// Conversation-lane mirror push helpers (ABI v7 feature 5) — the twins of
// pvx_mirror_push_token/push_embd for pvx_conversation_s::mirror.
inline void pvx_conv_mirror_push_token(pvx_conversation_t* c, llama_token id, llama_pos pos) {
    pvx_mirror_entry_t ent;
    ent.id = id;
    ent.pos = { pos, pos, pos, pos };
    c->mirror.push_back(std::move(ent));
}

inline void pvx_conv_mirror_push_embd(pvx_conversation_t* c, const float* row, int n_embd,
                                       const std::array<llama_pos, 4>& pos) {
    pvx_mirror_entry_t ent;
    ent.id  = LLAMA_TOKEN_NULL;
    ent.pos = pos;
    ent.embd.assign(row, row + n_embd);
    c->mirror.push_back(std::move(ent));
}

// ── Spec-session stream view (ABI v7 feature 5) ──────────────────────────
// The spec engine operates on ONE committed stream through this indirection:
// the executor lane binds its fields (seq 0), the batched/pool lane binds a
// conversation's fields (seq = conv->seq_id). All verify-loop mechanics —
// mirror replay, target batches, KV repair, bookkeeping — read/write through
// the pointers, so both lanes share the exact same loop code (token-exactness
// between lanes follows from having literally one implementation).
struct pvx_spec_stream_s {
    pvx_context_t* ctx = nullptr;              // target context wrapper
    llama_seq_id   seq = 0;                    // target KV sequence id
    int* n_tokens = nullptr;
    int* n_kv_pos = nullptr;
    std::vector<llama_token>* recent_tokens = nullptr;
    std::vector<float>* last_logits = nullptr;
    std::vector<pvx_mirror_entry_t>* mirror = nullptr;
    const char* lane = "executor";             // diagnostics only
};

#endif // PVX_INTERNAL_H
