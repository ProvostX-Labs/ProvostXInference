#ifndef PVXINFERENCE_H
#define PVXINFERENCE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

/* ── Opaque handles ── */
typedef struct pvx_model_s       pvx_model_t;
typedef struct pvx_context_s      pvx_context_t;
typedef struct pvx_pool_s         pvx_pool_t;
typedef struct pvx_conversation_s pvx_conversation_t;
typedef struct pvx_executor_s     pvx_executor_t;     /* standard (non-batched) */
typedef struct pvx_state_s        pvx_state_t;        /* saved KV state */

/* ── Error codes ── */
typedef enum {
    PVX_OK = 0,
    PVX_ERR_INVALID_ARG = 1,
    PVX_ERR_INVALID_HANDLE = 2,
    PVX_ERR_LOAD_FAILED = 3,
    PVX_ERR_NO_SLOT = 4,
    PVX_ERR_DECODE_FAILED = 5,
    PVX_ERR_OVERFLOW = 6,
    PVX_ERR_NOT_SUPPORTED = 7,
    PVX_ERR_REWIND_UNSUPPORTED = 8,
    PVX_ERR_INTERNAL = 99,
} pvx_result_t;

/* ── KV cache type ── */
typedef enum {
    PVX_KV_F16 = 0,
    PVX_KV_Q8_0 = 1,
    PVX_KV_Q4_0 = 2,
    PVX_KV_Q4_1 = 3,
} pvx_kv_type_t;

/* ── Pooling type (embeddings) ── */
typedef enum {
    PVX_POOL_NONE = 0,
    PVX_POOL_MEAN = 1,
    PVX_POOL_CLS = 2,
    PVX_POOL_LAST = 3,
    /* ABI v8: RANK pooling attaches the model's classification head to the
     * graph — the reranking lane (upstream LLAMA_POOLING_TYPE_RANK, the
     * llama-server --reranking mode). Contexts created with this pooling are
     * rerank-capable; pvx_rerank refuses typed on every other pooling. */
    PVX_POOL_RANK = 4,
} pvx_pooling_type_t;

/* ABI v8 wave-5 (BL-21b): GGUF metadata override (llama_model_kv_override
 * parity) — the general escape hatch for ANY model hyperparameter key.
 * Explicit fixed layout (no union): blittable from C#.
 * tag: 0 = bool, 1 = int64, 2 = float64, 3 = string. */
typedef struct {
    char    key[128];
    int32_t tag;
    int64_t val_i64;
    double  val_f64;
    uint8_t val_bool;
    char    val_str[128];
} pvx_kv_override_t;

/* ── Model params ── */
typedef struct {
    const char* model_path;
    int gpu_layers;          /* -1 = all, 0 = CPU only */
    int threads;             /* -1 = auto */
    bool flash_attn;
    pvx_kv_type_t kv_cache_type;
    /* Reasoning-marker probe list (data, not code): tag strings whose presence
     * as a SINGLE special token in the model's vocabulary marks the model as
     * thinking-capable. The FIRST matching marker drives the empty-block
     * prefill in pvx_apply_chat_template(enable_thinking=0): "<M>\n\n</M>\n\n".
     * NULL/n_markers<=0 = built-in default {"<think>"}. Extend the list via config
     * (no code change) — e.g. "<reason>" for DeepSeek-family vocabularies. */
    const char** reasoning_markers;  /* NULL = built-in default */
    int n_reasoning_markers;         /* <=0  = built-in default */
    /* ABI v8 wave-5 (BL-21b) appended load controls — zeroed = pre-wave-5
     * load behavior (llama defaults where pvx did not override). */
    const pvx_kv_override_t* kv_overrides; /* NULL = none */
    int32_t n_kv_overrides;                 /* <=0 = none */
    int32_t lazy_mode;                      /* 0 = llama default (AUTO); else llama_lazy_mode */
    int32_t use_extra_bufts_tri;             /* 0 = llama default (true); 1 force on; 2 force off */
    bool vocab_only;                        /* false = llama default */
    bool check_tensors;                     /* false = llama default */
    bool no_host;                           /* false = llama default */
    bool no_alloc;                          /* false = llama default */
    bool load_mtp;                          /* false = llama default (model's OWN MTP layers unloaded) */
    /* ABI v9 (per-engine audit/telemetry, owner order 2026-10-22): appended.
     * Zeroed struct = audit disabled — bit-identical loads as before. The
     * audit sink is PER-MODEL (own file handle + own mutex, no process
     * statics) so an LLM array of engines gets one audit log per engine. */
    const char* audit_path;                 /* NULL/empty = audit disabled */
    int32_t audit_include_prompts;          /* 0 = events only; 1 = also record prompt token counts + template events */
} pvx_model_params_t;

/* ── Context params ── */
typedef struct {
    uint32_t context_size;   /* n_ctx */
    uint32_t batch_size;     /* n_batch */
    uint32_t seq_max;        /* max sequences (batch mode) */
    pvx_pooling_type_t pooling_type;  /* for embedding models */
    /* ABI v8 wave-4 append (2026-10-19): recurrent-state snapshot ring per
     * sequence (upstream llama_context_params.n_rs_seq). 0 = llama default
     * (no snapshots — hybrid/recurrent targets then refuse partial seq_rm,
     * the pre-existing pvx_model_partial_rewind_supported contract). > 0 on
     * rs-rollback architectures (qwen35/qwen4exp/deepseek4/...) enables
     * partial seq_rm on hybrid memories — REQUIRED for speculative-decode
     * verify repair on those targets (the spec chains clamp to
     * llama_n_rs_seq, upstream server rule). Non-rs-rollback archs clamp
     * to 0 inside llama (harmless). */
    uint32_t n_rs_seq;
    /* ABI v8 wave-5 (BL-21c) appended context controls — zeroed = the
     * PRE-wave-5 pvx mapping / llama defaults (n_rs_seq comment above). */
    uint32_t ubatch_size;             /* 0 = mirror batch_size (pre-wave-5 mapping) */
    uint32_t n_outputs_max;           /* 0 = llama default (= n_batch) */
    uint32_t n_outputs_max_per_seq;   /* 0 = llama default (= n_outputs_max) */
    int32_t  threads_batch;           /* 0 = llama default (GGML auto, mirrors threads) */
    int32_t  rope_scaling_type;      /* 0 = UNSPECIFIED; 1 = NONE; 2 = LINEAR; 3 = YARN */
    float    rope_freq_base;          /* 0 = from model */
    float    rope_freq_scale;         /* 0 = from model */
    float    yarn_ext_factor;         /* <=0 = from model (-1) */
    float    yarn_attn_factor;        /* <=0 = from model (-1) */
    float    yarn_beta_fast;           /* <=0 = from model (-1) */
    float    yarn_beta_slow;           /* <=0 = from model (-1) */
    uint32_t yarn_orig_ctx;            /* 0 = from model */
    float    defrag_thold;             /* <=0 = disabled (llama default) */
    uint32_t moe_cache_size;           /* bytes; 0 = disabled (llama default) */
    int32_t  kv_type_k;               /* -1 = mirror model kv_cache_type; 0..3 explicit pvx_kv_type_t */
    int32_t  kv_type_v;               /* -1 = mirror model kv_cache_type */
    int32_t  offload_kqv_tri;          /* 0 = llama default (true); 1 on; 2 off */
    int32_t  op_offload_tri;           /* 0 = llama default (true); 1 on; 2 off */
    int32_t  swa_full_tri;             /* 0 = llama default (true); 1 on; 2 off */
    int32_t  kv_unified_tri;           /* 0 = llama default (false); 1 on; 2 off */
    int32_t  no_perf_tri;              /* 0 = llama default (true); 1 on; 2 off */
    bool     embeddings;                /* false = pre-wave-5 (pvx_get_embeddings drives it) */
    int32_t  attention_type;           /* 0 = UNSPECIFIED; 1 = causal; 2 = non-causal */
} pvx_context_params_t;

/* ABI v8 wave-5 (BL-21a, 2026-10-20): sampler-chain parity structs.
 * ZERO-VALUE SEMANTICS: every wave-5 sampling field at zero = the
 * pre-wave-5 chain (greedy / temp+topk+topp+minp+time-seeded dist).
 * logit_bias NULL / n<=0 = off. */
typedef struct {
    int32_t token;   /* llama token id */
    float   bias;    /* logit bias (llama_logit_bias parity) */
} pvx_logit_bias_t;

/* ABI v8 wave-5 sampling knobs (appended to pvx_sampling_params_t). */
typedef struct {
    int64_t seed;               /* <=0 = time entropy (pre-wave-5); >=1 deterministic */
    float   frequency_penalty;  /* 0 = off */
    float   typical_p;          /* 0 = off; (0,1) typical sampling (temp>0 path) */
    float   xtc_p;              /* 0 = off; xtc sampler when temp>0 */
    float   xtc_t;              /* xtc threshold; <=0 → 1.0f when xtc_p > 0 */
    float   top_n_sigma;        /* 0 = off; top-n-σ when > 0 */
    int32_t mirostat_mode;      /* 0 = off; 1 = v1; 2 = v2 (replaces dist) */
    float   mirostat_tau;       /* 0 → 5.0f default when mode > 0 */
    float   mirostat_eta;       /* 0 → 0.1f default when mode > 0 */
    float   dry_multiplier;     /* 0 = off; DRY repetition penalty when > 0 */
    float   dry_base;           /* 0 → 1.75f default when multiplier > 0 */
    int32_t dry_allowed_length; /* 0 → 2 default when multiplier > 0 */
    int32_t dry_penalty_last_n; /* 0 → -1 (full context) when multiplier > 0 */
    const pvx_logit_bias_t* logit_bias; /* NULL = off */
    int32_t n_logit_bias;              /* <=0 = off */
} pvx_sampling_params_wave5_t;

/* ── Sampling params (full pipeline) ── */
typedef struct {
    float temperature;
    float top_p;
    int   top_k;
    float min_p;
    float repeat_penalty;
    int   repeat_last_n;     /* -1 = full context */
    float penalty_present;  /* presence penalty */
    int   max_tokens;
    bool  ignore_eos;
    /* ABI v8 wave-5 (BL-21a) appended knobs — zeroed = pre-wave-5 chain. */
    pvx_sampling_params_wave5_t wave5;
} pvx_sampling_params_t;



/* ── Speculative decoding (sequential path, INF BL-05) ────────────── */

/* Streaming callback for pvx_executor_generate_spec: receives each accepted
 * token id; return true to continue, false to stop early. NULL = no stream. */
typedef bool (*pvx_spec_token_callback_t)(int32_t token, void* user_data);

/* Self-reported supported speculative types (the supported-types list).
 * Bitset of PVX_SPEC_TYPE_* — reported per MODEL LOAD from GGUF metadata
 * (arch + nextn/MTP tensor presence). Bits not listed below are reserved 0:
 * companion-draft types (eagle3/dflash/dspark) are per-target companion GGUFs
 * (not detectable from the target model alone) and are not wired in this
 * slice — enabling them is structurally unreachable through pvx_spec_params_t,
 * which is the named config-surface boundary. The n-gram types are
 * model-independent (drafting from the committed stream + caches, no draft
 * model) and are reported for every standard-vocab target (ABI v6). */
#define PVX_SPEC_TYPE_DRAFT_SIMPLE 0x1u /* second draft model, any eligible target */
#define PVX_SPEC_TYPE_MTP          0x2u /* model carries its own MTP (nextn) layers */
#define PVX_SPEC_TYPE_EAGLE3       0x4u /* ABI v7: companion draft GGUF (target_layers metadata); executor-lane wiring */
#define PVX_SPEC_TYPE_DFLASH       0x8u /* ABI v7: dflash companion metadata validated; drafting unwired — typed boundary */
#define PVX_SPEC_TYPE_DSPARK       0x10u /* not detected / not wired this slice */
#define PVX_SPEC_TYPE_NGRAM        0x20u /* n-gram simple: history-pattern lookup (ABI v6) */
#define PVX_SPEC_TYPE_NGRAM_MAP_K  0x40u /* n-gram map with value statistics (ABI v6) */
#define PVX_SPEC_TYPE_NGRAM_MOD    0x80u /* n-gram MOD: (n-1)-suffix match, one deviation (ABI v6) */
#define PVX_SPEC_TYPE_NGRAM_CACHE  0x100u /* n-gram cache: static+dynamic token-prob caches (ABI v6) */

typedef struct {
    uint32_t types;  /* supported-types list (PVX_SPEC_TYPE_* bitset); 0 = no native spec type */
    bool mtp;        /* convenience mirror of the MTP bit */
    bool draft;      /* convenience mirror of the DRAFT_SIMPLE bit */
    /* ABI v6 append: the spec session can continue drafting after multimodal
     * (image) inputs — the embd-aware mirror replays projected embedding rows
     * onto the draft side (zero-padded with a warning when the draft input
     * width differs). Model-independent property of the spec engine. */
    bool spec_vision;
} pvx_spec_capabilities_t;

/* Spec config for a sequential executor.
 * mode:
 *   PVX_SPEC_OFF       = disabled (default)
 *   PVX_SPEC_MTP_AUTO  = spec-decode this model's own MTP layers (needs mtp capability)
 *   PVX_SPEC_DRAFT     = separate draft model (needs draft capability + draft_model_path)
 * Fail-closed rule: a configured path that cannot be honored (missing/unloadable
 * draft model, missing capability, wrong vocab family) is a typed error naming
 * the boundary — never a silent fallback to non-spec generation. */
typedef enum {
    PVX_SPEC_OFF      = 0,
    PVX_SPEC_MTP_AUTO = 1,
    PVX_SPEC_DRAFT    = 2,
    /* ABI v6 appends — n-gram drafting needs NO draft model: drafts come from
     * the committed stream itself (lookup caches built online). Any
     * standard-vocab target qualifies. */
    PVX_SPEC_NGRAM_SIMPLE = 3, /* history-pattern lookup (upstream ngram-simple) */
    PVX_SPEC_NGRAM_MAP_K  = 4, /* n-gram -> m-gram map with accept statistics */
    PVX_SPEC_NGRAM_MOD   = 5, /* MOD: (n-1)-suffix match allowing one deviation */
    PVX_SPEC_NGRAM_CACHE = 6, /* static (file) + dynamic token-probability caches */
    /* ABI v7 appends — companion-GGUF draft-graph modes (upstream spec-decode
     * draft-eagle3 / draft-dflash). Eagle3 is wired on the SEQUENTIAL executor
     * lane; the batched/pool lane refuses typed (boundary:
     * spec-engine/eagle3-conversation-lane). DFlash validates companion
     * metadata and fails typed at the wiring boundary (spec-engine/dflash-wiring)
     * — never faked. */
    PVX_SPEC_EAGLE3 = 7, /* EAGLE3 draft decoder fed (token, g_embd) pairs from target layer features */
    PVX_SPEC_DFLASH = 8, /* DFlash block-diffusion draft (typed wiring boundary this slice) */
    /* ABI v8 wave-4 append (2026-10-19) — SEPARATE self-contained draft-MTP head
     * GGUF (upstream draft-mtp, llama.cpp PR #28243; qwen4exp-family targets).
     * mtp_draft_path is REQUIRED with this mode; draft_model_path stays
     * draft-simple-only. The head drafts on the target trunk's nextn
     * embeddings (DeepSeek-style (t_{P-1}, h_{P-1}) pairing), verified greedy-
     * exact by the target — upstream sample-and-match. Sequential executor
     * lane only; PVX_SPEC_MTP_AUTO (embedded nextn) stays a typed wiring
     * boundary (boundary: spec-engine/mtp-wiring). */
    PVX_SPEC_MTP_DRAFT = 9,
} pvx_spec_mode_t;

/* ABI v6 append — n-gram knobs (all modes PVX_SPEC_NGRAM_*). Zero-value
 * fields select per-mode upstream defaults; no env vars. cache_path is read
 * at spec_init (a corrupt/vocab-mismatched file is a typed PVX_ERR_LOAD_FAILED
 * naming the path — never a silently empty cache). */
typedef struct {
    int         n_min;      /* lookup min n-gram size; 0 = per-mode default */
    int         n_max;      /* lookup/draft max size (m-gram length); 0 = per-mode default */
    int         min_hits;   /* PVX_SPEC_NGRAM_MAP_K only: min key hits; 0 = default (1) */
    int         n_match;    /* PVX_SPEC_NGRAM_MOD only: hash n-gram window; 0 = default (24) */
    const char* cache_path; /* PVX_SPEC_NGRAM_CACHE only: optional static cache file; NULL = dynamic only */
} pvx_ngram_params_t;

typedef struct {
    pvx_spec_mode_t mode;
    const char* draft_model_path;  /* required iff mode == PVX_SPEC_DRAFT */
    int   draft_max;               /* max drafted tokens per verify step; <=0 = upstream default */
    int   draft_min;               /* reject drafts shorter than this;   <=0 = upstream default */
    float p_min;                   /* drafter top-token confidence floor; <0 = upstream default */
    int   draft_gpu_layers;        /* -1 = same offload as target, else explicit */
    pvx_ngram_params_t ngram;      /* ABI v6 append — active iff mode is PVX_SPEC_NGRAM_* */
    /* ABI v7 appends — zero-value = feature off (blittable struct growth).
     * eagle_companion_path required iff mode == PVX_SPEC_EAGLE3: companion
     * GGUF must carry eagle3.target_layers (exactly 3 entries) + compatible
     * vocab; validated at spec_init with typed errors naming the metadata
     * key. dflash_companion_path required iff mode == PVX_SPEC_DFLASH. */
    const char* eagle_companion_path;
    const char* dflash_companion_path;
    /* ABI v8 wave-4 append (2026-10-19) — zero-value = feature off (blittable
     * struct growth). mtp_draft_path is REQUIRED iff mode == PVX_SPEC_MTP_DRAFT:
     * a SELF-CONTAINED MTP head GGUF (own token embedding + output head + nextn
     * block). Validated typed at spec_init (nextn-layer presence, n_embd_out
     * match vs the target trunk, vocab family) — never a silent disable. */
    const char* mtp_draft_path;
} pvx_spec_params_t;

/* Capability query (see struct docs). Engines without spec support report
 * all-false and their spec entry points return PVX_ERR_NOT_SUPPORTED.
 * Out-param form (blittable across the managed boundary). */
pvx_result_t pvx_model_spec_capabilities(const pvx_model_t* model,
                                         pvx_spec_capabilities_t* out_caps);

/* Typed validation of a spec config against this model (mode/path/capability/
 * draft-vocab consistency) without enabling anything. On failure the reason
 * string is retrievable via pvx_model_last_error (model-scoped, ABI v5). */
pvx_result_t pvx_model_spec_validate(const pvx_model_t* model,
                                     const pvx_spec_params_t* params);

/* Enable/disable speculative decoding on a SEQUENTIAL executor (single stream).
 * PVX_SPEC_DRAFT with a missing/unloadable draft model returns PVX_ERR_LOAD_FAILED
 * and changes nothing. Disable with mode == PVX_SPEC_OFF. One session per executor. */
pvx_result_t pvx_executor_spec_init(pvx_executor_t* exec,
                                    const pvx_spec_params_t* params);

/* Teardown of the spec session (idempotent; pvx_executor_free also calls it). */
void pvx_executor_spec_free(pvx_executor_t* exec);

/* Spec-session acceptance telemetry for this executor (0/0 when none).
 * ABI v7 live-verification fix (BL-36): the definition previously shipped
 * without this declaration, exporting a C++-mangled symbol unreachable from
 * the C ABI — the header contract (extern "C" linkage) is restored here. */
pvx_result_t pvx_executor_spec_stats(pvx_executor_t* exec,
                                     int64_t* out_drafted,
                                     int64_t* out_accepted);

/* Speculative token loop over the executor's current prompt+history: draft →
 * target-verify → accept all accepted tokens into the executor's normal state
 * (n_tokens, recent_tokens, KV) until EOG (unless params->ignore_eos is set,
 * which samples past EOG exactly like the plain pipeline) or out_max tokens.
 * on_token (nullable) streams every accepted token; return false to stop early.
 * out_generated receives the number of generated tokens.
 * PVX_ERR_NOT_SUPPORTED when no spec session is active on this executor.
 * Sampling uses the same pvx_sampling_params pipeline as pvx_executor_sample,
 * so spec and non-spec generation agree token-for-token for identical params. */
pvx_result_t pvx_executor_generate_spec(pvx_executor_t* exec,
                                        const pvx_sampling_params_t* params,
                                        int out_max,
                                        pvx_spec_token_callback_t on_token,
                                        void* user_data,
                                        int* out_generated);

/* ══════════════════════════════════════════════════
 *  Speculative decoding — batched/pool lane (ABI v7, feature 5)
 * ══════════════════════════════════════════════════ */

/* Enable speculative decoding on ONE pooled conversation. Mirrors the executor
 * trio: the spec session is per-conversation (own draft state, keyed by the
 * conversation's seq_id); the committed-stream mirror ring is maintained per
 * conversation by every commit path (batched decode, rewind, reset, return).
 * Requires >= 2 seq slots on the context (typed PVX_ERR_NOT_SUPPORTED naming
 * "batched spec requires >=2 seq slots" otherwise). PVX_SPEC_EAGLE3 refuses
 * typed on this lane (boundary: spec-engine/eagle3-conversation-lane).
 * Enabling with a non-empty history whose mirror is inconsistent (e.g. after a
 * checkpoint import) is a typed PVX_ERR_NOT_SUPPORTED naming
 * spec-engine/mirror-consistency — fail closed, never draft against a phantom
 * stream. All spec ops take the context's infer mutex (same discipline as
 * sample ops); pool return tears the session down in the existing teardown
 * order. */
pvx_result_t pvx_conversation_spec_init(pvx_conversation_t* conv,
                                         const pvx_spec_params_t* params);

/* Teardown of the conversation's spec session (idempotent; pool return and
 * context teardown also call it). */
void pvx_conversation_spec_free(pvx_conversation_t* conv);

/* Speculative token loop over one pooled conversation. Identical contract to
 * pvx_executor_generate_spec (draft -> target-verify -> accept into the
 * conversation's normal state; token-exact with the plain lane for identical
 * params). Sampling uses the same pipeline as pvx_conversation_sample.
 * PVX_ERR_NOT_SUPPORTED when no spec session is active on this conversation. */
pvx_result_t pvx_conversation_generate_spec(pvx_conversation_t* conv,
                                             const pvx_sampling_params_t* params,
                                             int out_max,
                                             pvx_spec_token_callback_t on_token,
                                             void* user_data,
                                             int* out_generated);

/* Spec-session acceptance telemetry for one conversation (0/0 when none). */
pvx_result_t pvx_conversation_spec_stats(pvx_conversation_t* conv,
                                         int64_t* out_drafted,
                                         int64_t* out_accepted);

/* ══════════════════════════════════════════════════
 *  JSON-schema → GBNF grammar converter (ABI v7, feature 11)
 * ══════════════════════════════════════════════════ */

/* Convert a JSON Schema document (RFC 8259 JSON text) into a GBNF grammar
 * string consumable by pvx_grammar_create (pattern-port of the vendored
 * common/json-schema-to-grammar converter — never linked, self-contained in
 * src/pvx_json_schema_grammar.cpp).
 *   json_schema : NUL-terminated JSON object (the schema document)
 *   root_name   : optional rule name for the root (NULL/"" = "root")
 *   out_gbnf    : receives a malloc'd NUL-terminated grammar (free with
 *                 pvx_free_string); NULL on failure
 *   out_error   : optional; receives a malloc'd typed boundary message
 *                 (boundary: json-schema/parse | json-schema/keyword)
 * Fail-closed contract: invalid JSON → PVX_ERR_INVALID_ARG naming the parse
 * position; an unsupported schema KEYWORD (per-keyword port, e.g. regex
 * "pattern") → PVX_ERR_NOT_SUPPORTED naming the keyword — never a loose
 * best-effort grammar. KV-neutral: pure string -> string conversion. */
pvx_result_t pvx_grammar_from_json_schema(const char* json_schema,
                                           const char* root_name,
                                           char** out_gbnf,
                                           char** out_error);

/* ── Decode result ── */
typedef enum {
    PVX_DECODE_OK = 0,
    PVX_DECODE_NO_WORK = 1,
    PVX_DECODE_FAILED = 2,
} pvx_decode_result_t;

/* ════════════════════════════════════════════════════
 *  Model
 * ════════════════════════════════════════════════════ */

pvx_result_t pvx_load_model(const pvx_model_params_t* params, pvx_model_t** out_model);
void pvx_free_model(pvx_model_t* model);

/* ════════════════════════════════════════════════════
 *  Context (shared KV pool — used by both standard and batch paths)
 * ════════════════════════════════════════════════════ */

pvx_result_t pvx_create_context(pvx_model_t* model,
                                const pvx_context_params_t* params,
                                pvx_context_t** out_ctx);
void pvx_free_context(pvx_context_t* ctx);

uint32_t pvx_context_size(pvx_context_t* ctx);
uint32_t pvx_context_seq_max(pvx_context_t* ctx);

/* Partial rewind capability: whether this model's memory supports truncating a
 * sequence at an arbitrary position (llama_memory_seq_rm with a partial range).
 * Hybrid architectures (attention + recurrent layers, e.g. qwen35/qwen3next)
 * and purely recurrent architectures CANNOT: their recurrent state is not
 * preserved per token, so a partial seq_rm refuses (and would otherwise leave
 * the attention KV untruncated while counters move — a silent KV/position
 * desync; M-RoPE models then hard-fail the decode position invariant).
 * Callers that cache KV across requests must check this before retaining or
 * reusing slots via partial rewinds. */
bool pvx_model_partial_rewind_supported(const pvx_model_t* model);
int      pvx_context_n_ctx_used(pvx_context_t* ctx, pvx_conversation_t* conv);

/* ════════════════════════════════════════════════════
 *  Tokenization
 * ════════════════════════════════════════════════════ */

pvx_result_t pvx_tokenize(pvx_context_t* ctx, const char* text,
                          bool add_bos, bool parse_special,
                          int32_t** out_tokens, int* out_count);
void pvx_free_tokens(int32_t* tokens);

pvx_result_t pvx_detokenize(pvx_context_t* ctx, const int32_t* tokens, int count,
                            bool remove_special,
                            char** out_text);
void pvx_free_string(char* str);

/* Token to piece (single token → string) */
/// Config-driven native log file: when set (non-empty path), ALL native logs
/// (llama.cpp/ggml/Metal + pvx) are appended here by the dylib itself.
void pvx_set_log_file(const char* path);

/// Repair tokenizer EOG metadata in a GGUF (idempotent, weights stream-copied untouched).
/// in_path: source gguf; out_path: repaired copy; report: caller buffer for a human summary.
pvx_result_t pvx_repair_eog_metadata(const char* in_path, const char* out_path, char* report, int report_size);

pvx_result_t pvx_token_to_piece(pvx_context_t* ctx, int32_t token, char* buf, int buf_size);

/* ════════════════════════════════════════════════════
 *  Standard (non-batched) executor — one session, one context
 *  Replaces: StatefulExecutorBase / InteractiveExecutor
 * ════════════════════════════════════════════════════ */

pvx_result_t pvx_executor_create(pvx_context_t* ctx, pvx_executor_t** out_exec);
void pvx_executor_free(pvx_executor_t* exec);

/* Prompt the executor (tokenizes + feeds to KV cache) */
pvx_result_t pvx_executor_prompt(pvx_executor_t* exec, const char* text);

/* Prompt with raw tokens */
pvx_result_t pvx_executor_prompt_tokens(pvx_executor_t* exec, const int32_t* tokens, int count);

/* Decode one step (processes pending tokens) */
pvx_decode_result_t pvx_executor_infer(pvx_executor_t* exec);

/* Sample one token using full pipeline */
pvx_result_t pvx_executor_sample(pvx_executor_t* exec,
                                  const pvx_sampling_params_t* params,
                                  int32_t* out_token);

/* Check if token is end-of-generation */
bool pvx_token_is_eos(pvx_context_t* ctx, int32_t token);

/* Current token count in KV cache */
int pvx_executor_token_count(pvx_executor_t* exec);

/* Rewind (remove last n tokens from KV) */
pvx_result_t pvx_executor_rewind(pvx_executor_t* exec, int n_tokens);

/* Reset (clear entire KV cache) */
pvx_result_t pvx_executor_reset(pvx_executor_t* exec);

/* ════════════════════════════════════════════════════
 *  State save/restore (for rewind in standard path)
 *  Replaces: LLamaContext.State / StatefulExecutorBase.ExecutorBaseState
 * ════════════════════════════════════════════════════ */

pvx_result_t pvx_state_save(pvx_executor_t* exec, pvx_state_t** out_state);
pvx_result_t pvx_state_restore(pvx_executor_t* exec, pvx_state_t* state);
void pvx_state_free(pvx_state_t* state);

/* Get the token count saved in a state (for verification) */
int pvx_state_token_count(pvx_state_t* state);

/* ════════════════════════════════════════════════════
 *  Conversation pool (batched inference)
 *  Replaces: BatchedExecutor / Conversation
 * ════════════════════════════════════════════════════ */

pvx_result_t pvx_pool_create(pvx_context_t* ctx, pvx_pool_t** out_pool);
void pvx_pool_free(pvx_pool_t* pool);

pvx_result_t pvx_pool_lease(pvx_pool_t* pool, pvx_conversation_t** out_conv);
pvx_result_t pvx_pool_return(pvx_pool_t* pool, pvx_conversation_t* conv);
int pvx_pool_available(pvx_pool_t* pool);
uint32_t pvx_pool_size(pvx_pool_t* pool);

/* ════════════════════════════════════════════════════
 *  Conversation (leased from pool)
 * ════════════════════════════════════════════════════ */

pvx_result_t pvx_conversation_prompt(pvx_conversation_t* conv, const char* text);
pvx_result_t pvx_conversation_prompt_tokens(pvx_conversation_t* conv,
                                             const int32_t* tokens, int count);
pvx_result_t pvx_conversation_sample(pvx_conversation_t* conv,
                                     const pvx_sampling_params_t* params,
                                     int32_t* out_token);
int pvx_conversation_token_count(pvx_conversation_t* conv);
pvx_result_t pvx_conversation_rewind(pvx_conversation_t* conv, int n_tokens);
pvx_result_t pvx_conversation_reset(pvx_conversation_t* conv);
pvx_result_t pvx_conversation_grammar_reset(pvx_conversation_t* conv);
pvx_result_t pvx_executor_grammar_reset(pvx_executor_t* exec);

/* Save/restore conversation state (for batch rewind) */
pvx_result_t pvx_conversation_save(pvx_conversation_t* conv, pvx_state_t** out_state);
pvx_result_t pvx_conversation_restore(pvx_pool_t* pool, pvx_conversation_t* conv,
                                       pvx_state_t* state);
void pvx_conversation_state_free(pvx_state_t* state);

// ── Conversation checkpoint (suspend/resume to disk) ──
// Serializes the conversation's FULL inference state for eviction/paging:
// KV cells (llama_state_seq_get_data), counters, pending tokens and the
// repeat-penalty window. Blob layout (little-endian):
//   [int32 n_tokens][int32 n_kv_pos][int32 n_recent][int32 n_pending]
//   [recent x n_recent][pending x n_pending][llama seq-state data]
// Caller owns the buffer; free it with pvx_conversation_export_free.
// Thread-safe: takes the owning context's inference mutex.
pvx_result_t pvx_conversation_export(pvx_conversation_t* conv, uint8_t** out_buf, size_t* out_size);
void pvx_conversation_export_free(uint8_t* buf);
// Restores a previously exported blob INTO this conversation. The
// conversation must be EMPTY (fresh lease or after reset) — its seq is
// cleared first, then KV cells are re-created at its own seq_id and all
// counters/pending/penalty-window state is rebuilt from the header.
pvx_result_t pvx_conversation_import(pvx_conversation_t* conv, const uint8_t* data, size_t size);

/* ════════════════════════════════════════════════════
 *  Batched inference — ONE decode for ALL conversations
 * ════════════════════════════════════════════════════ */

pvx_decode_result_t pvx_infer(pvx_context_t* ctx);

/* ════════════════════════════════════════════════════
 *  Embeddings
 *  Replaces: LLamaEmbedder
 * ════════════════════════════════════════════════════ */

pvx_result_t pvx_get_embeddings(pvx_model_t* model, pvx_context_t* ctx,
                                const char* text,
                                float** out_embeddings, int* out_dim);
void pvx_free_embeddings(float* embeddings);

int pvx_embedding_dim(pvx_model_t* model);

/* ════════════════════════════════════════════════════
 *  Vision / MTMD
 *  Replaces: MtmdWeights
 * ════════════════════════════════════════════════════ */

pvx_result_t pvx_load_mmproj(pvx_model_t* model, const char* mmproj_path,
                             pvx_model_t** out_mmproj);
void pvx_free_mmproj(pvx_model_t* mmproj);

/* Load image bytes into a conversation's media queue */
pvx_result_t pvx_conversation_load_image(pvx_conversation_t* conv,
                                          pvx_model_t* mmproj,
                                          const uint8_t* data, int size);

/* Full prompt with images: tokenize prompt+images, encode, feed to conversation */
pvx_result_t pvx_conversation_prompt_with_images(pvx_conversation_t* conv,
                                                    pvx_model_t* model_with_mtmd,
                                                    const char* text,
                                                    const uint8_t** image_data,
                                                    const int* image_sizes,
                                                    int n_images);

/* Get the mtmd marker string (e.g. "<image>") */
pvx_result_t pvx_mtmd_marker(char** out_marker);
const char* pvx_mtmd_marker_static(void);

/* ════════════════════════════════════════════════════
 *  Granular KV cache manipulation (impossible via LLamaSharp)
 * ════════════════════════════════════════════════════ */

/* Shift left: remove oldest n tokens, keep recent ones.
 * Sliding window context — no reset needed when approaching n_ctx.
 * Re-positions remaining tokens to start at position 0.
 * Returns PVX_ERR_NOT_SUPPORTED if the backend can't reposition.
 *
 * Implementation: llama_memory_seq_rm(seq, 0, n) removes the front n tokens,
 * then llama_memory_seq_cp shifts remaining positions. */
pvx_result_t pvx_conversation_shift_left(pvx_conversation_t* conv, int n_tokens);
pvx_result_t pvx_executor_shift_left(pvx_executor_t* exec, int n_tokens);

/* KV copy: copy an entire conversation's KV to another seq_id.
 * Used for state backup without blob serialization.
 * Source and dest must be on the same context. */
pvx_result_t pvx_kv_copy(pvx_context_t* ctx, int src_seq_id, int dst_seq_id);

/* KV clear: clear KV for a specific seq_id only.
 * Does NOT affect other conversations. */
pvx_result_t pvx_kv_clear_seq(pvx_context_t* ctx, int seq_id);

/* KV size: how many tokens are stored for a seq_id.
 * Returns -1 on error. */
int pvx_kv_seq_tokens(pvx_context_t* ctx, int seq_id);

/* Causal attention control (per-context, per-call).
 * Set false before decoding image embeddings, true for text.
 * Vision path handles this internally — exposed for advanced use. */
pvx_result_t pvx_set_causal_attn(pvx_context_t* ctx, bool causal);

/* Non-causal check for mtmd: returns true if the current model
 * needs non-causal attention for image decode. */
bool pvx_mtmd_needs_non_causal(pvx_model_t* mmproj);

/* Check if generated text contains any anti-prompt string.
 * Returns index of matched anti-prompt (>=0) or -1 if none. */
int pvx_check_anti_prompts(const char* text, const char** anti_prompts, int count);

/* ════════════════════════════════════════════════════
 *  Grammar (GBNF)
 * ════════════════════════════════════════════════════ */

/* Opaque grammar handle. */
typedef struct pvx_grammar_s pvx_grammar_t;

/* Create a GBNF grammar sampler. Returns null if parsing fails.
 * The grammar is applied during sampling — only tokens matching the
 * grammar are selected. Pass null to pvx_*_sample_with_grammar to
 * sample without grammar constraint.
 * The grammar must be created from the same model's vocab. */
pvx_grammar_t* pvx_grammar_create(pvx_model_t* model, const char* grammar_str, const char* grammar_root);

/* Free a grammar. */
void pvx_grammar_free(pvx_grammar_t* grammar);

/* Sample with grammar constraint. Like pvx_executor_sample / pvx_conversation_sample
 * but applies the grammar sampler before token selection. */
pvx_result_t pvx_executor_sample_grammar(pvx_executor_t* exec, const pvx_sampling_params_t* params,
                                             pvx_grammar_t* grammar, int* out_token);
pvx_result_t pvx_conversation_sample_grammar(pvx_conversation_t* conv, const pvx_sampling_params_t* params,
                                                pvx_grammar_t* grammar, int* out_token);

/* ════════════════════════════════════════════════════
 *  Chat template
 * ════════════════════════════════════════════════════ */

/* Apply the model's built-in chat template to a list of messages.
 * Uses llama_chat_apply_template (supports Qwen, Llama, ChatML, etc.).
 * Pass tmpl=null to use the model's default template.
 * messages: array of {role, content} pairs.
 * add_assistant: append the assistant turn marker.
 * enable_thinking: -1 = auto (model/template default), 0 = off, 1 = on.
 *   When 0 and the vocabulary structurally contains a "<think>" special token
 *   (single-token match, e.g. Qwen3 family), an EMPTY think block is prefilled
 *   after the assistant header - the same mechanism Qwen3's official template
 *   uses for enable_thinking=false. Models without the token are unaffected
 *   (the flag is then advisory only; stream-side stripping lives in the server).
 * out_text: receives the formatted prompt (caller frees with pvx_free_string).
 * Returns PVX_OK or error. */
typedef struct { const char* role; const char* content; } pvx_chat_message_t;

pvx_result_t pvx_apply_chat_template(pvx_model_t* model, const char* tmpl,
                                        const pvx_chat_message_t* messages, int n_messages,
                                        bool add_assistant, int enable_thinking,
                                        char** out_text);

/* ABI v9 (format-agnostic tools, owner-approved 2026-10-22): tools-aware chat
 * template application. Renders the model's OWN embedded Jinja template with
 * the tool definitions in context, so each model family receives tool docs in
 * its own trained format — no per-model code anywhere.
 *
 * tools: array of {name, description, parameters_json} (OpenAI function shape;
 * parameters_json is the raw JSON Schema string, may be null/empty for a
 * schema-less tool). NULL/n_tools<=0 = EXACT legacy behavior (delegates to
 * pvx_apply_chat_template — parity pinned by tests).
 *
 * Template selection: the model's embedded Jinja template (tmpl override wins
 * when set). Fail-closed on templates WITHOUT tool rendering: the tool docs
 * are then not injected; the caller (server) still grammar-constrains output
 * shape, so calls stay parseable. Detection is structural — no model-name
 * checks anywhere.
 * enable_thinking semantics identical to pvx_apply_chat_template.
 * out_text: receives the formatted prompt (caller frees with pvx_free_string).
 * Returns PVX_OK or error. */
typedef struct {
    const char* name;            /* required */
    const char* description;     /* may be null/empty */
    const char* parameters_json; /* JSON Schema string; null/empty = schema-less */
} pvx_chat_tool_t;

pvx_result_t pvx_apply_chat_template_v2(pvx_model_t* model, const char* tmpl,
                                        const pvx_chat_message_t* messages, int n_messages,
                                        const pvx_chat_tool_t* tools, int n_tools,
                                        bool add_assistant, int enable_thinking,
                                        char** out_text);

/* ════════════════════════════════════════════════════
 *  Vision on standard executor
 * ════════════════════════════════════════════════════ */

/* Prompt the standard executor with text + images.
 * Like pvx_executor_prompt but processes image markers via the mmproj projector.
 * The executor uses seq_id 0 — images are encoded and decoded with non-causal attention.
 * Must be called with no pending tokens (flush before vision prompt).
 * After this call, call pvx_executor_infer as normal. */
pvx_result_t pvx_executor_prompt_with_images(pvx_executor_t* exec,
                                                 pvx_model_t* model,
                                                 const char* text,
                                                 const char** image_data_ptrs,
                                                 const int* image_sizes,
                                                 int n_images);

/* ════════════════════════════════════════════════════
 *  Logging / errors
 * ════════════════════════════════════════════════════ */

const char* pvx_last_error(pvx_context_t* ctx);

/* Model-scoped last error (ABI v5): the reason string for model-level typed
 * failures (e.g. pvx_model_spec_validate). Do NOT pass a model handle to
 * pvx_last_error — that function expects a context handle. */
const char* pvx_model_last_error(const pvx_model_t* model);
const char* pvx_result_str(pvx_result_t r);

/* ── Structured logging callback (2026-09-25) ──
 *
 * Zero-overhead when no callback is set: every log site is a single
 * `if (g_log_cb)` check — branch-predicted to skip. When the callback
 * is NULL (default), no string formatting, no argument marshalling.
 *
 * Levels: DEBUG < INFO < WARN < ERROR. Set the minimum level via
 * pvx_set_log_level. Messages below the minimum are not dispatched.
 * The callback receives: level (int), a tag string (static), and the
 * formatted message. The callback owns the received strings for the
 * duration of the call only — copy if needed.
 */
typedef enum {
    PVX_LOG_DEBUG = 0,
    PVX_LOG_INFO  = 1,
    PVX_LOG_WARN  = 2,
    PVX_LOG_ERROR = 3,
    PVX_LOG_NONE  = 99
} pvx_log_level_t;

typedef void (*pvx_log_callback_t)(int level, const char* tag, const char* message);

/* Set the global log callback. Pass NULL to disable (zero-overhead). */
void pvx_set_log_callback(pvx_log_callback_t cb);

/* Set the minimum log level. Messages below this level are not dispatched. */
void pvx_set_log_level(pvx_log_level_t level);

/* Internal log helper — not for direct use by callers; use PVX_LOG macro. */
void pvx_log(pvx_log_level_t level, const char* tag, const char* fmt, ...);

/* Convenience macro: evaluates g_log_cb first (zero-cost skip). */
#define PVX_LOG(lvl, tag, ...) do { \
    if (g_log_cb && (lvl) >= g_log_min_level) { \
        pvx_log((lvl), (tag), __VA_ARGS__); \
    } \
} while(0)

/* ════════════════════════════════════════════════════
 *  Backend safety layer
 * ════════════════════════════════════════════════════ */

/* Probe: validate the caller's config works on this backend.
 * Does a minimal decode. Does NOT change config -- just validates.
 * Returns PVX_OK if the config works, PVX_ERR_DECODE_FAILED if not. */
pvx_result_t pvx_probe_backend(pvx_context_t* ctx);

/* Safe context creation with progressive fallback.
 * Tries the caller's requested config first (from ProvostXLLM config).
 * If it fails, progressively reduces gpu_layers: N -> N/2 -> N/4 -> ... -> 0.
 * The caller's config (KV type, batch size, FA, seq_max) is RESPECTED --
 * only gpu_layers is reduced during fallback.
 * model may be reloaded with fewer gpu_layers (caller passes model**). */
pvx_result_t pvx_create_context_safe(pvx_model_t** model,
                                      const pvx_model_params_t* model_params,
                                      const pvx_context_params_t* ctx_params,
                                      int requested_gpu_layers,
                                      pvx_context_t** out_ctx);

/* VRAM guard: check if enough VRAM is available for an operation. */
bool pvx_vram_available(size_t needed_bytes);

/* Backend state queries -- what the probe found about THIS context's config.
 * Per-context (audit 2026-10-17): the probe profile lives on the context, not
 * in process-global state — concurrent engines never clobber each other's
 * probe results. Null context → false/""/0. */
bool        pvx_backend_probe_passed(pvx_context_t* ctx);
const char* pvx_backend_name(pvx_context_t* ctx);
const char* pvx_backend_probe_error(pvx_context_t* ctx);
bool        pvx_backend_needs_kv_flush(pvx_context_t* ctx);
bool        pvx_backend_gpu_layers_reduced(pvx_context_t* ctx);
int         pvx_backend_effective_gpu_layers(pvx_context_t* ctx);
int         pvx_backend_original_gpu_layers(pvx_context_t* ctx);

/* KV flush: flush stale K/V cells after clearing (Vulkan bug #26744).
 * Automatically called in pvx_pool_return when pvx_backend_needs_kv_flush() is true. */
pvx_result_t pvx_kv_flush(pvx_context_t* ctx, int seq_id);

/* ABI handshake (2026-09-27): returned by the native library; the C# binding
 * carries the same constant. A mismatch means a STALE native copy was loaded
 * (e.g. an orphaned app-root dylib predating a signature change) — the binding
 * fails FAST with a clear error instead of corrupting the stack/segfaulting.
 * Bump BOTH sides in lockstep on any native signature/ABI change.
 */
/* ══════════════════════════════════════════════════════════
 *  ABI v8 — Wave 3 (media + weight-surface modifiers + rerank + stats)
 *  ══════════════════════════════════════════════════════════ */

/* ── Audio input (plan feature 6; MTMD_INPUT_CHUNK_TYPE_AUDIO) ──
 *
 * Audio rides the same embd-row mechanics as vision: the mmproj projector
 * (audio adapter) converts PCM F32 rows into projected embeddings that are
 * decoded at pinned positions and pushed into the committed-stream mirror
 * ring (KV positions advance by n_audio_tokens; no plain ids exist).
 * Sample-rate contract: PCM must arrive at the projector's native rate
 * (pvx_mtmd_audio_sample_rate); product code does NOT resample — a rate
 * mismatch is the caller's to fix. Fail-closed boundaries (typed, named):
 *   media/audio-adapter   — model's mtmd context lacks an audio adapter
 *   media/audio-mixed     — tokenizer produced a non-text/audio chunk
 * KV-neutrality: identical to images — positions only, no KV-op changes. */

/* Capability probe: does the loaded (shared) mmproj carry an audio adapter? */
bool pvx_mtmd_supports_audio(const pvx_model_t* model_with_mtmd);

/* Native PCM sample rate the projector expects, or -1 when audio is
 * unsupported (mirrors mtmd_get_audio_sample_rate). */
int pvx_mtmd_audio_sample_rate(const pvx_model_t* model_with_mtmd);

/* Prompt the standard executor with text + ONE audio clip (raw PCM F32 at
 * the projector's native rate; text carries the media marker). Mirrors
 * pvx_executor_prompt_with_images; the projected rows enter the executor
 * mirror ring. Same flush/fail-closed contract as the image path. */
pvx_result_t pvx_executor_prompt_with_audio(pvx_executor_t* exec,
                                            pvx_model_t* model_with_mtmd,
                                            const char* text,
                                            const float* pcm,
                                            int n_samples);

/* Conversation-lane twin (same contract as the executor path). */
pvx_result_t pvx_conversation_prompt_with_audio(pvx_conversation_t* conv,
                                                pvx_model_t* model_with_mtmd,
                                                const char* text,
                                                const float* pcm,
                                                int n_samples);

/* ── Speech output / TTS (plan feature 7; mtmd gen-audio, QWEN3TTS) ──
 *
 * A TTS session is a SEPARATE decode context on the backbone (speech) model
 * plus a dedicated gen-audio mmproj — never entangled with any conversation
 * context (KV-neutral by construction: no conversation KV interaction).
 * Determinism: identical (text, params.seed) pairs produce identical PCM on
 * FRESH sessions (the session is SINGLE-SHOT — the gen-audio pipeline carries
 * cross-call state; a second generate on a live session is a typed refusal,
 * boundary tts/session-consumed). seed == UINT32_MAX selects a random draw.
 * Fail-closed boundaries (typed):
 *   tts/gen-audio-capability — mmproj has no gen-audio pipeline (info.type
 *                             == MTMD_GEN_AUDIO_TYPE_NONE; a QWEN3TTS-
 *                             capable mmproj is required)
 *   tts/prompt-stage|gen-stage — helper pipeline stage failure */

typedef struct pvx_tts_session_s pvx_tts_session_t;

/* TTS sampling knobs (zero-value = per-model upstream defaults from
 * mtmd_gen_inp_default; no env vars). max_frames caps the generation loop
 * (<=0 = 512). lang is an optional BCP-ish tag the pipeline may use. */
typedef struct {
    const char* lang;    /* NULL/"" = pipeline default */
    int   top_k;         /* 0 = default */
    float top_p;         /* 0 = default */
    float temp;          /* 0 = default */
    uint32_t seed;       /* UINT32_MAX = random */
    int   max_frames;    /* 0 = 512 */
} pvx_tts_params_t;

/* Create a TTS session: loads the gen-audio mmproj against the backbone
 * speech model and owns its own llama context (context_size tokens; 0 =
 * model default). Capability-truthful: a mmproj without a gen-audio
 * pipeline fails PVX_ERR_NOT_SUPPORTED (boundary: tts/gen-audio-capability). */
pvx_result_t pvx_tts_session_create(pvx_model_t* backbone_model,
                                    const char* tts_mmproj_path,
                                    uint32_t context_size,
                                    pvx_tts_session_t** out_session);

void pvx_tts_session_free(pvx_tts_session_t* session);

/* Generate PCM F32 mono for one text prompt. out_pcm is malloc'd (free with
 * pvx_free_audio_samples); out_sample_rate receives the pipeline rate. */
pvx_result_t pvx_tts_generate(pvx_tts_session_t* session,
                              const pvx_tts_params_t* params,
                              const char* text,
                              float** out_pcm,
                              int* out_n_samples,
                              int* out_sample_rate);

void pvx_free_audio_samples(float* pcm);

/* ── Reranking (plan feature 8; RANK-pooling score head) ──
 *
 * Upstream pattern: llama-server --reranking — the model's classification
 * head is attached via RANK pooling and scores each (query, document) pair.
 * The prompt layout follows the model's "rerank" chat template ({query}/
 * {document} substitution) or the token fallback (BOS + query + EOS + doc +
 * EOS). Single-shot, own seq usage, KV cleared per document — KV-neutral.
 * Fail-closed boundaries (typed, named):
 *   rerank/pooling — context pooling must be PVX_POOL_RANK
 *   rerank/score-head — embeddings output is not the 1-dim score head */
pvx_result_t pvx_rerank(pvx_context_t* ctx,
                        const char* query,
                        const char* const* documents,
                        int n_documents,
                        float** out_scores);
void pvx_free_rerank_scores(float* scores);

/* ── LoRA adapters runtime attach/detach (plan feature 9) ──
 *
 * Adapters are a WEIGHTS-surface modifier (llama_set_adapters_lora): they
 * change future logits exactly like a sampling change — existing KV content
 * and contexts stay valid, no rewind is required (upstream semantics; the
 * set is serialized against decodes by the context's infer mutex).
 * Set with n == 0 detaches all adapters (back to baseline logits).
 * Fail-closed boundaries (typed, named):
 *   adapter/file       — load failed (path named in the model error string)
 *   adapter/arch       — adapter base arch does not match the model's
 *   adapter/scale      — scale <= 0
 * All-or-nothing: llama_set_adapters_lora applies the full array or fails. */

typedef struct pvx_adapter_s pvx_adapter_t;

pvx_result_t pvx_adapter_load(pvx_model_t* model, const char* path,
                              pvx_adapter_t** out_adapter);
void pvx_adapter_free(pvx_adapter_t* adapter);

/* Attach (or detach, n == 0) adapters on a LIVE context. scales[i] applies to
 * adapters[i]; scale <= 0 is a typed arg error. */
pvx_result_t pvx_context_set_lora(pvx_context_t* ctx,
                                  const pvx_adapter_t* const* adapters,
                                  const float* scales,
                                  int n_adapters);

/* ── Control vectors (plan feature 10) ──
 *
 * Pattern-port of common_control_vector_load (never linked): direction.<L>
 * F32 1-D tensors (layer L >= 1) from a GGUF, accumulated into an
 * n_embd x max_layer buffer (layer 1 at [0]). Weights-surface modifier like
 * LoRA — KV-neutral, live attach/detach, serialized by the infer mutex.
 * Fail-closed boundaries (typed, named):
 *   cvector/file  — unreadable/invalid direction tensor file
 *   cvector/n-embd — direction width != model n_embd (both dims named)
 *   cvector/layer-range — layer_start/end outside [1, n_layer] */

typedef struct pvx_cvec_s pvx_cvec_t;

pvx_result_t pvx_control_vector_load(pvx_model_t* model, const char* path,
                                     pvx_cvec_t** out_cvec);
void pvx_control_vector_free(pvx_cvec_t* cvec);

/* Apply on a live context (layer range inclusive; 0/0 = full 1..n_layer).
 * cvec == NULL clears the applied vector (no residual: logits return to
 * baseline, gated by tests). */
pvx_result_t pvx_context_set_control_vector(pvx_context_t* ctx,
                                            const pvx_cvec_t* cvec,
                                            int layer_start,
                                            int layer_end);

/* ── Metrics / stats export (plan feature 12, additive snapshot) ──
 *
 * Copy-out snapshot (blittable, zero new locks): counters maintained by the
 * existing commit paths under the infer mutex. Spec counters mirror the
 * existing pvx_*_spec_stats telemetry (0/0 when no session). Counters never
 * hide failures: spec rejected drafts are derivable (drafted - accepted). */
typedef struct {
    int64_t n_tokens;       /* committed stream tokens (mirror rows) */
    int64_t n_kv_pos;       /* committed KV positions */
    int64_t n_prompt_tokens;/* tokens fed through prompt paths */
    int64_t n_decode_steps; /* llama decode calls issued on this lane */
    int64_t spec_drafted;   /* speculative drafts proposed (0 = none) */
    int64_t spec_accepted;  /* speculative drafts accepted */
} pvx_stats_t;

pvx_result_t pvx_executor_stats(pvx_executor_t* exec, pvx_stats_t* out_stats);
pvx_result_t pvx_conversation_stats(pvx_conversation_t* conv, pvx_stats_t* out_stats);

/* ── Per-engine audit + telemetry (ABI v9, append-only; owner order 2026-10-22) ──
 *
 * AUDIT: a per-MODEL event sink (own file handle + own mutex — never process
 * globals), configured via pvx_model_params_t.audit_path. Events are one JSON
 * line each (timestamped, monotonic seq) so an LLM running an ARRAY of engines
 * gets one tamper-evident trail per engine instance. Recorded events:
 *   model_loaded / model_freed / context_created / context_freed /
 *   spec_enabled / spec_disabled / kv_cleared (destructive KV ops) /
 *   load_failed / decode_failed / prompt_recorded (only when
 *   audit_include_prompts=1 — token counts, never prompt TEXT: prompts may
 *   carry user content and audit logs must not become a content sink).
 *
 * TELEMETRY: pvx_engine_telemetry_t is a MODEL-aggregate snapshot (roll-up of
 * all executor/conversation lanes + spec + error counts), pulled via
 * pvx_engine_telemetry — zero new locks (rides the audit mutex when audit is
 * on; reads atomic-ish int64 copies otherwise). Rates (tok/s) are DERIVED by
 * the caller between two snapshots — the native side stays stateless.
 */
typedef struct {
    int64_t n_prompt_tokens;   /* model-wide prompt tokens fed */
    int64_t n_decode_steps;    /* model-wide llama decode calls */
    int64_t n_committed_tokens;/* model-wide committed stream tokens */
    int64_t spec_drafted;      /* model-wide spec drafts proposed */
    int64_t spec_accepted;     /* model-wide spec drafts accepted */
    int64_t n_errors;          /* typed failures recorded in audit */
    int64_t n_contexts_alive;  /* live contexts right now */
} pvx_engine_telemetry_t;

pvx_result_t pvx_engine_telemetry(pvx_model_t* model, pvx_engine_telemetry_t* out);

#define PVX_ABI_VERSION 9
int pvx_abi_version(void);

#ifdef __cplusplus
}
#endif

#endif /* PVXINFERENCE_H */
