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
} pvx_pooling_type_t;

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
} pvx_model_params_t;

/* ── Context params ── */
typedef struct {
    uint32_t context_size;   /* n_ctx */
    uint32_t batch_size;     /* n_batch */
    uint32_t seq_max;        /* max sequences (batch mode) */
    pvx_pooling_type_t pooling_type;  /* for embedding models */
} pvx_context_params_t;

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
#define PVX_SPEC_TYPE_EAGLE3       0x4u /* not detected / not wired this slice */
#define PVX_SPEC_TYPE_DFLASH       0x8u /* not detected / not wired this slice */
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
#define PVX_ABI_VERSION 6
int pvx_abi_version(void);

#ifdef __cplusplus
}
#endif

#endif /* PVXINFERENCE_H */
