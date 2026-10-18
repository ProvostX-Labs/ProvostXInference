// pvxinference.cpp — Complete C API implementation for ProvostX Inference Engine
//
// Replaces LLamaSharp entirely. Links llama.cpp directly.
// Owns: model loading, context, standard executor, conversation pool,
// batched inference, full sampling pipeline, embeddings, vision/mtmd,
// state save/restore, tokenization.
//
// Thread safety: internal mutex serializes ALL native llama.cpp calls.
#include "pvx_internal.h"
#include "pvx_spec_engine.h"
#include "llama-arch.h"
#include "llama-model.h"
#include "gguf.h"
#include <fstream>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdarg>
#include <cstdio>
#include <random>
#include <vector>
#include <atomic>
#include <mutex>
#include <ctime>
#include <deque>
#include <thread>
#include <condition_variable>
#include <chrono>

// ── Logging globals (2026-09-25) — defined here so PVX_LOG macro works everywhere ──
static pvx_log_callback_t g_log_cb = nullptr;
static pvx_log_level_t g_log_min_level = PVX_LOG_NONE;
static char g_log_file[1024] = {0};
static void pvx_ggml_log_bridge(enum ggml_log_level level, const char* text, void* user_data);

// ── Async native log writer (owner 2026-10-06: "make it so it never becomes an issue") ──
// Producers only ENQUEUE (bounded, drop-oldest under flood — logging can never
// stall an engine thread at any verbosity). A dedicated writer thread owns the
// kept-open FILE handle and flushes every drain cycle, so a hard kill loses at
// most the in-flight line — same window as the old per-line open/close.
struct pvx_log_entry {
    pvx_log_level_t level;
    char tag[32];
    char msg[1024];
};
static std::deque<pvx_log_entry> g_log_q;
static std::mutex g_log_mu;
static std::condition_variable g_log_cv;
static std::thread* g_log_writer = nullptr;
static std::atomic<bool> g_log_writer_run{false};
static std::atomic<uint64_t> g_log_dropped{0};
constexpr size_t PVX_LOG_Q_CAP = 8192;

static void pvx_log_writer_loop() {
    char path[1024] = {0};
    FILE* f = nullptr;
    while (g_log_writer_run.load(std::memory_order_relaxed)) {
        std::unique_lock<std::mutex> lk(g_log_mu);
        g_log_cv.wait_for(lk, std::chrono::milliseconds(200), []{ return !g_log_q.empty() || !g_log_writer_run.load(); });
        if (g_log_q.empty() && g_log_writer_run.load()) continue;
        std::deque<pvx_log_entry> batch;
        batch.swap(g_log_q);
        bool path_changed = strncmp(path, g_log_file, sizeof(path)) != 0;
        lk.unlock();
        if (path_changed) {
            if (f) { fclose(f); f = nullptr; }
            snprintf(path, sizeof(path), "%s", g_log_file);
        }
        if (!f && path[0]) { f = fopen(path, "a"); }
        if (!f) continue;
        char ts[32];
        time_t now = time(nullptr);
        struct tm tmv;
        localtime_r(&now, &tmv);
        strftime(ts, sizeof(ts), "%H:%M:%S", &tmv);
        const char* lvl = "";
        for (auto& e : batch) {
            lvl = e.level == PVX_LOG_ERROR ? "Error" : e.level == PVX_LOG_WARN ? "Warn" : "Info";
            fprintf(f, "[%s] [%s] [%s] %s\n", ts, lvl, e.tag, e.msg);
        }
        uint64_t dropped = g_log_dropped.exchange(0);
        if (dropped > 0) fprintf(f, "[%s] [Warn] [Native] log queue flooded — %llu line(s) dropped\n", ts, (unsigned long long)dropped);
        fflush(f);
    }
    if (f) fclose(f);
}

// Owner 2026-10-06: config-driven native log file. When set (non-empty), pvx_log
// appends every native line here with timestamp+level+tag — the LLM server hands
// this path straight from its config (inference section), so the dylib owns its
// own output location. Empty/unset = no file sink (logs go to the callback only).
extern "C" void pvx_set_log_file(const char* path) {
    if (!path || !path[0]) { g_log_file[0] = '\0'; return; }
    ggml_log_set(pvx_ggml_log_bridge, nullptr);
    if (!g_log_writer) {
        g_log_writer_run.store(true);
        g_log_writer = new std::thread(pvx_log_writer_loop);
        g_log_writer->detach();
    }
    { std::lock_guard<std::mutex> lk(g_log_mu); snprintf(g_log_file, sizeof(g_log_file), "%s", path); }
    g_log_cv.notify_one();
}

// ── Grammar tier diagnostics (benchmark investigation 2026-09-26) ──
// Counters for rejection-sampling tier hits in do_sample_with_grammar.
static std::atomic<int> g_tier1_hits{0};
static std::atomic<int> g_tier2_hits{0};
static std::atomic<int> g_tier2b_hits{0};
static std::atomic<int> g_tier3_hits{0};
static std::atomic<uint64_t> g_tier3_ns{0};

// Forward declaration of the log implementation (defined at end of file).
extern "C" void pvx_log(pvx_log_level_t level, const char* tag, const char* fmt, ...);

// ── Helpers ──

static const char* result_strings[] = {
    "OK", "INVALID_ARG", "INVALID_HANDLE", "LOAD_FAILED",
    "NO_SLOT", "DECODE_FAILED", "OVERFLOW", "NOT_SUPPORTED", "REWIND_UNSUPPORTED", "INTERNAL"
};

// ABI handshake (2026-09-27): see include/pvxinference.h. Bump with the header
// AND the C# binding constant in lockstep on any native ABI change.
int pvx_abi_version(void) { return PVX_ABI_VERSION; }

const char* pvx_result_str(pvx_result_t r) {
    if (r >= 0 && r <= PVX_ERR_INTERNAL) return result_strings[r];
    return "UNKNOWN";
}

static void set_error(pvx_context_t* ctx, const std::string& msg) {
    PVX_LOG(PVX_LOG_ERROR, "Inference", "set_error: %s", msg.c_str());
    if (ctx) ctx->last_error = msg;
}

// ── Model ──

pvx_result_t pvx_load_model(const pvx_model_params_t* params, pvx_model_t** out_model) {
    if (!params || !params->model_path || !out_model) return PVX_ERR_INVALID_ARG;

    auto* m = new pvx_model_s();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = params->gpu_layers;

    m->model = llama_model_load_from_file(params->model_path, mp);
    if (!m->model) {
        m->last_error = std::string("Failed to load model: ") + params->model_path;
        PVX_LOG(PVX_LOG_ERROR, "Model", "Load failed: %s (gpu_layers=%d)", params->model_path, params->gpu_layers);
        delete m;
        return PVX_ERR_LOAD_FAILED;
    }
    m->vocab = llama_model_get_vocab(m->model);
    PVX_LOG(PVX_LOG_INFO, "Model", "Loaded: %s (gpu_layers=%d)", params->model_path, params->gpu_layers);
    // Capability self-report at model load (INF BL-05): the engine decides what
    // the loaded MODEL supports — log the supported-types list so operators and
    // the LLM slice see it at load time (the query entry point stays authoritative).
    {
        pvx_spec_capabilities_t caps {};
        if (pvx_model_spec_capabilities(m, &caps) == PVX_OK) {
            if (caps.types == 0) {
                PVX_LOG(PVX_LOG_INFO, "Spec", "supported-types: (none native; second draft model remains available via PVX_SPEC_DRAFT)");
            } else {
                PVX_LOG(PVX_LOG_INFO, "Spec", "supported-types: [0x%x]%s%s", caps.types,
                        (caps.types & PVX_SPEC_TYPE_DRAFT_SIMPLE) ? " draft_simple" : "",
                        (caps.types & PVX_SPEC_TYPE_MTP) ? " mtp" : "");
            }
        }
    }
    m->flash_attn = params->flash_attn;
    m->kv_cache_type = params->kv_cache_type;
    m->threads = params->threads;
    m->gpu_layers = params->gpu_layers;   // caller's requested config — backend
                                          // profile reads this for truthful
                                          // gpu-layer visibility (per-context)
    // Data-driven reasoning-marker list (owner-approved 2026-09-26): caller may
    // override the built-in default ({"think"}) - extend per release via config,
    // no code change. Kept as a COPY; caller owns its array lifetime.
    if (params->reasoning_markers && params->n_reasoning_markers > 0) {
        m->reasoning_markers.assign(params->reasoning_markers,
                                    params->reasoning_markers + params->n_reasoning_markers);
    }
    *out_model = m;
    return PVX_OK;
}

void pvx_free_model(pvx_model_t* model) {
    if (!model) return;

    // Wait for all contexts created from this model to be freed first.
    // Metal (and other GPU backends) assert that all buffers/residency sets
    // are freed before the device is freed. If pvx_free_model runs concurrently
    // with pvx_free_context (e.g. .NET SafeHandle finalizers during process
    // exit), the device destructor aborts because contexts still hold buffers.
    //
    // We block here — NOT in pvx_free_context — so context destruction stays
    // parallel. Only model destruction serializes against context teardown.
    {
        std::unique_lock<std::mutex> lk(model->destruction_mtx);
        model->destruction_cv.wait(lk, [model]() {
            return model->active_context_count == 0 && model->mtmd_shared_refs == 0;
        });
    }

    if (model->mtmd_ctx) {
        mtmd_free(model->mtmd_ctx);
        model->mtmd_ctx = nullptr;
    }
    if (model->model) llama_model_free(model->model);
    delete model;
}

// ── Context ──

pvx_result_t pvx_create_context(pvx_model_t* model,
                                const pvx_context_params_t* params,
                                pvx_context_t** out_ctx) {
    if (!model || !params || !out_ctx) return PVX_ERR_INVALID_ARG;

    auto* c = new pvx_context_s();
    c->model = model->model;
    c->owning_model = model;  // back-reference for destruction accounting
    c->vocab = model->vocab;
    c->pool_type = params->pooling_type;

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = params->context_size;
    cp.n_batch = params->batch_size;
    cp.n_ubatch = params->batch_size;
    cp.n_seq_max = params->seq_max > 0 ? params->seq_max : 1;
    // Flash attention: caller drives via model params (passed to pvx_load_model)
    // KV cache type: caller drives via model params
    if (model->kv_cache_type != PVX_KV_F16) {
        switch (model->kv_cache_type) {
            case PVX_KV_Q8_0: cp.type_k = GGML_TYPE_Q8_0; cp.type_v = GGML_TYPE_Q8_0; break;
            case PVX_KV_Q4_0: cp.type_k = GGML_TYPE_Q4_0; cp.type_v = GGML_TYPE_Q4_0; break;
            case PVX_KV_Q4_1: cp.type_k = GGML_TYPE_Q4_1; cp.type_v = GGML_TYPE_Q4_1; break;
            default: break;
        }
    }
    if (model->flash_attn) {
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    } else {
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    }

    // Pooling type for embedding models
    // Note: do NOT set cp.embeddings = true here — it creates an encoder context
    // that requires llama_encode() instead of llama_decode(). The original
    // llama_batch_get_one + llama_decode works for BERT/MiniLM models.
    // llama_set_embeddings(ctx, true) is called in pvx_get_embeddings instead.
    switch (params->pooling_type) {
        case PVX_POOL_MEAN: cp.pooling_type = LLAMA_POOLING_TYPE_MEAN; break;
        case PVX_POOL_CLS:  cp.pooling_type = LLAMA_POOLING_TYPE_CLS; break;
        case PVX_POOL_LAST: cp.pooling_type = LLAMA_POOLING_TYPE_LAST; break;
        default:           cp.pooling_type = LLAMA_POOLING_TYPE_NONE; break;
    }

    c->ctx = llama_new_context_with_model(model->model, cp);
    if (!c->ctx) { c->last_error = "Failed to create context"; delete c; return PVX_ERR_LOAD_FAILED; }

    c->mem = llama_get_memory(c->ctx);
    c->n_ctx = params->context_size;
    c->n_seq_max = cp.n_seq_max;

    // Backend profile detection half: EVERY context creation path (facade
    // included) gets truthful backend type/name + gpu-layer fields from birth.
    // The decode probe (pvx_probe_backend) adds probe_passed on demand.
    pvx_backend_fill_profile(c);

    // Register this context with the owning model for destruction ordering.
    // pvx_free_model will wait until active_context_count reaches zero
    // before freeing the model (and its GPU device).
    {
        std::lock_guard<std::mutex> lk(model->destruction_mtx);
        model->active_context_count++;
    }

    *out_ctx = c;
    return PVX_OK;
}

void pvx_free_context(pvx_context_t* ctx) {
    if (!ctx) return;

    // Free the llama context first. llama_free releases all GPU buffers
    // (Metal residency sets, CUDA allocations, etc.) owned by this context.
    if (ctx->ctx) llama_free(ctx->ctx);

    // Decrement the owning model's context count and notify any waiting
    // pvx_free_model call that one fewer context holds GPU resources.
    if (ctx->owning_model) {
        std::lock_guard<std::mutex> lk(ctx->owning_model->destruction_mtx);
        ctx->owning_model->active_context_count--;
        ctx->owning_model->destruction_cv.notify_all();
    }

    delete ctx;
}

uint32_t pvx_context_size(pvx_context_t* ctx) { return ctx ? ctx->n_ctx : 0; }
uint32_t pvx_context_seq_max(pvx_context_t* ctx) { return ctx ? ctx->n_seq_max : 0; }

int pvx_context_n_ctx_used(pvx_context_t* ctx, pvx_conversation_t* conv) {
    if (!conv) return 0;
    return conv->n_tokens;
}

// ── Tokenization ──

static int do_tokenize(const llama_vocab* vocab, const char* text,
                       std::vector<llama_token>& out, bool add_bos, bool parse_special) {
    std::string s(text);
    int n_max = (int)s.size() + 2;
    out.resize(n_max);
    int n = llama_tokenize(vocab, s.c_str(), (int)s.size(), out.data(), n_max, add_bos, parse_special);
    if (n < 0) { n_max = -n; out.resize(n_max); n = llama_tokenize(vocab, s.c_str(), (int)s.size(), out.data(), n_max, add_bos, parse_special); }
    if (n < 0) return -1;
    out.resize(n);
    return n;
}

pvx_result_t pvx_tokenize(pvx_context_t* ctx, const char* text,
                          bool add_bos, bool parse_special,
                          int32_t** out_tokens, int* out_count) {
    if (!ctx || !text || !out_tokens || !out_count) return PVX_ERR_INVALID_ARG;
    std::vector<llama_token> tokens;
    if (do_tokenize(ctx->vocab, text, tokens, add_bos, parse_special) < 0) return PVX_ERR_INTERNAL;
    *out_count = (int)tokens.size();
    if (tokens.empty()) {
        *out_tokens = nullptr;
        return PVX_OK;
    }
    *out_tokens = (int32_t*)malloc(tokens.size() * sizeof(int32_t));
    if (!*out_tokens) return PVX_ERR_INTERNAL;
    memcpy(*out_tokens, tokens.data(), tokens.size() * sizeof(int32_t));
    return PVX_OK;
}

void pvx_free_tokens(int32_t* tokens) { free(tokens); }

pvx_result_t pvx_detokenize(pvx_context_t* ctx, const int32_t* tokens, int count,
                           bool remove_special, char** out_text) {
    if (!ctx || !tokens || !out_text || count <= 0) return PVX_ERR_INVALID_ARG;
    std::string result;
    for (int i = 0; i < count; i++) {
        char buf[256];
        int n = llama_token_to_piece(ctx->vocab, tokens[i], buf, sizeof(buf), 0, !remove_special);
        if (n > 0) result.append(buf, n);
    }
    *out_text = (char*)malloc(result.size() + 1);
    if (!*out_text) return PVX_ERR_INTERNAL;
    strcpy(*out_text, result.c_str());
    return PVX_OK;
}

void pvx_free_string(char* str) { free(str); }

pvx_result_t pvx_token_to_piece(pvx_context_t* ctx, int32_t token, char* buf, int buf_size) {
    if (!ctx || !buf || buf_size <= 0) return PVX_ERR_INVALID_ARG;
    int n = llama_token_to_piece(ctx->vocab, token, buf, buf_size, 0, true);
    return n > 0 ? PVX_OK : PVX_ERR_INTERNAL;
}

bool pvx_token_is_eos(pvx_context_t* ctx, int32_t token) {
    return ctx ? llama_vocab_is_eog(ctx->vocab, token) : false;
}

// ── Full sampling pipeline ──
// Replaces: DefaultSamplingPipeline
// Implements: temperature, top_k, top_p, min_p, repeat_penalty, presence penalty


// Forward declarations (defined in the Grammar section below)
struct pvx_grammar_state_s;
static void reset_grammar_state(pvx_grammar_state_s* gs);

// ── Per-thread RNG (mt19937) ─────────────────────────────────────────────────
// Replaces srand()/rand(): srand was global — under the batched server multiple
// conversations sampling on different threads share one seed state, and rand()
// is not thread-safe (data race) and has poor distribution for top-p sampling.
// Each thread gets its own mt19937 seeded once from random_device.
static thread_local std::mt19937 t_rng{std::random_device{}()};
static thread_local bool t_rng_seeded = false;
static void ensure_rng() {
    if (!t_rng_seeded) { t_rng.seed(std::random_device{}()); t_rng_seeded = true; }
}
static float rng_unit01() {
    ensure_rng();
    return std::uniform_real_distribution<float>(0.0f, 1.0f)(t_rng);
}

// ── Thread-local candidate-array scratch ────────────────────────────────────
// do_sample_with_grammar() previously allocated a fresh std::vector of 150k
// llama_token_data entries per attempt (up to 16 re-draws with ignore_eos) —
// that push_back loop + realloc was the dominant per-token sampling cost.
// Reusing a thread-local buffer keeps the raw block across attempts and calls;
// capacity grows once to n_vocab and is then reused. Token ids are rewritten
// every attempt, so no stale state can leak into sampling.
static llama_token_data* t_candidates(size_t n_vocab, llama_token_data_array& arr) {
    thread_local std::vector<llama_token_data> buf;
    if (buf.size() < n_vocab) buf.resize(n_vocab);
    for (int i = 0; i < (int)n_vocab; i++) {
        buf[i] = { (llama_token)i, 0.0f, 0.0f };
    }
    arr = { buf.data(), n_vocab, -1, false };
    return buf.data();
}

// ── Sampling-params fingerprint (for future pooled-chain gating) ────────────
// Any change in these params invalidates a persisted/pooled selection chain.
// Dist is intentionally NOT fingerprinted (seed changes per call by design);
// penalties are intentionally NOT fingerprinted (they are history-feeding and
// force rebuilds anyway); grammar is conversation-owned (Tier-1, persistent).
static uint64_t params_fingerprint(const pvx_sampling_params_t* p) {
    if (!p) return 0;
    uint64_t h = 1469598103934665603ULL;
    auto mix = [&h](uint64_t v) { h ^= v; h *= 1099511628211ULL; };
    union { float f; uint32_t u; } tf;
    tf.f = p->temperature;            mix(tf.u);
    tf.f = p->top_p;                  mix(tf.u);
    mix((uint32_t)p->top_k);
    tf.f = p->min_p;                  mix(tf.u);
    mix(p->ignore_eos ? 1u : 0u);
    // Penalty params — REQUIRED for chain pooling: the penalties sampler is
    // part of the pooled chain, so any penalty change must force a rebuild.
    tf.f = p->repeat_penalty;        mix(tf.u);
    tf.f = p->penalty_present;       mix(tf.u);
    mix((uint32_t)p->repeat_last_n);
    return h;
}

// ── Pooled per-thread selection chain (2026-09-26, ARCHITECTURE pm6) ──
// do_sample_with_grammar used to build + free a fresh chain per sample call
// (~1-2ms; ×N per batched step). The chain depends ONLY on the sampling
// params, so we pool one chain per thread, keyed by params_fingerprint.
// Invariants (owner-approved):
//   1. Fingerprint covers EVERY chain-affecting param (incl. penalties) —
//      any mismatch → free + rebuild, never wrong params.
//   2. NO llama_sampler_accept on the pooled chain — stateless w.r.t. accept,
//      preserving today's token-selection semantics exactly.
//      (GATED FUTURE OPTION, owner noted: feed recent_tokens via
//      reset+replay to activate repeat/present penalties — deliberate
//      behavior change, needs own LLM validation. See ARCHITECTURE pm6.)
//   3. thread_local with destructor → freed at thread exit; rebuilt on
//      mismatch. Nothing shared across threads, nothing to leak.
struct pooled_chain {
    llama_sampler* smpl = nullptr;
    llama_sampler* pen = nullptr;   // penalties sampler inside the chain (null if penalties off)
    uint64_t fingerprint = 0;
    ~pooled_chain() { if (smpl) llama_sampler_free(smpl); pen = nullptr; }
};
static thread_local pooled_chain t_pooled;

// Build or fetch the per-thread selection chain for these params.
static llama_sampler* pooled_selection_chain(const pvx_sampling_params_t* params,
                                             const std::vector<llama_token>& recent_tokens,
                                             int n_vocab) {
    const uint64_t fp = params_fingerprint(params);
    if (t_pooled.smpl && t_pooled.fingerprint == fp) return t_pooled.smpl;

    if (t_pooled.smpl) llama_sampler_free(t_pooled.smpl);
    t_pooled.pen = nullptr;
    llama_sampler* smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());

    if (params->repeat_penalty != 1.0f || params->penalty_present > 0.0f) {
        // Auto window (-1): FIXED constant (llama-server parity) — NOT
        // recent_tokens.size(). The chain is pooled/thread-persistent; sizing
        // the ring from build-time recent_tokens.size() would make the penalty
        // window depend on whichever request first touched this thread.
        int pen_last_n = params->repeat_last_n >= 0 ? params->repeat_last_n : 64;
        if (pen_last_n > 0) {
            llama_sampler* pen = llama_sampler_init_penalties(
                n_vocab, pen_last_n, params->repeat_penalty, 0.0f, params->penalty_present);
            llama_sampler_chain_add(smpl, pen);
            t_pooled.pen = pen;
        }
    }
    if (params->temperature <= 0) {
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    } else {
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(params->temperature));
        if (params->top_k > 0)
            llama_sampler_chain_add(smpl, llama_sampler_init_top_k(params->top_k));
        if (params->top_p < 1.0f)
            llama_sampler_chain_add(smpl, llama_sampler_init_top_p(params->top_p, 1));
        if (params->min_p > 0.0f)
            llama_sampler_chain_add(smpl, llama_sampler_init_min_p(params->min_p, 1));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist((uint32_t)t_rng()));
    }

    t_pooled.smpl = smpl;
    t_pooled.fingerprint = fp;
    return smpl;
}

// Snapshot a decoded logits row into `dest`. MUST be called immediately after
// the llama_decode that produced `row` — the row is only addressable while
// that batch is the context's CURRENT batch state. Any later decode on the
// shared context (another slice of a chunked batch, another conversation's
// cycle) replaces it, and llama_get_logits_ith() on the stale row
// GGML_ABORTs the process (get_logits_ith: "batch.logits[i] != true").
static void snapshot_logits_row(llama_context* lctx, std::vector<float>& dest, int row) {
    const float* logits = llama_get_logits_ith(lctx, row);
    if (!logits) { dest.clear(); return; }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(lctx)));
    dest.assign(logits, logits + n_vocab);
}

static int32_t do_sample_with_grammar(const float* logits_raw, const llama_vocab* vocab,
                                      const pvx_sampling_params_t* params,
                                      const std::vector<llama_token>& recent_tokens,
                                      pvx_grammar_t* grammar,
                                      pvx_grammar_state_s* grammar_state);

static int32_t do_sample(const float* logits, const llama_vocab* vocab,
                         const pvx_sampling_params_t* params,
                         const std::vector<llama_token>& recent_tokens) {
    // Single sampling pipeline (2026-09-26): no-grammar calls delegate to the
    // same chain used by do_sample_with_grammar — thread-local candidate
    // scratch, per-thread mt19937, llama-server chain semantics
    // (temp -> top_k -> top_p -> min_p -> dist, llama-style penalties).
    // Replaces the old hand-rolled sampler (~8.6ms/token measured).
    if (!logits) return -1;
    static const pvx_sampling_params_t kDefaultParams{
        .temperature = 0.3f, .top_p = 0.95f, .top_k = 40, .min_p = 0.0f,
        .repeat_penalty = 1.1f, .repeat_last_n = -1, .penalty_present = 0.0f,
        .max_tokens = 0, .ignore_eos = false,
    };
    return do_sample_with_grammar(logits, vocab,
                                  params ? params : &kDefaultParams,
                                  recent_tokens, /*grammar=*/nullptr,
                                  /*grammar_state=*/nullptr);
}

// ── Spec-engine sampling seams (INF BL-05) ──
// The spec engine verifies with the SAME token-selection pipeline as the plain
// loop: pooled per-thread selection chain + grammar-free do_sample, and the
// same logits-row snapshot rule (row dies at the next decode on the context).
int32_t spec_sample(const float* logits, const llama_vocab* vocab,
                           const pvx_sampling_params_t* params,
                           const std::vector<llama_token>& recent_tokens) {
    return do_sample(logits, vocab, params, recent_tokens);
}

void spec_snapshot(llama_context* lctx, std::vector<float>& dest, int row) {
    snapshot_logits_row(lctx, dest, row);
}

// Thread-safe last_error setter for engine-side typed boundaries.
void pvx_spec_set_context_error(pvx_context_t* ctx, const std::string& msg) {
    if (ctx) ctx->last_error = msg;
}

// ── Standard executor (non-batched) ──
// Replaces: InteractiveExecutor / StatefulExecutorBase

pvx_result_t pvx_executor_create(pvx_context_t* ctx, pvx_executor_t** out_exec) {
    if (!ctx || !out_exec) return PVX_ERR_INVALID_ARG;
    auto* exec = new pvx_executor_s();
    exec->ctx_ref = ctx;
    *out_exec = exec;
    return PVX_OK;
}

void pvx_executor_free(pvx_executor_t* exec) {
    if (!exec) return;
    pvx_executor_spec_free(exec);    // Leave seq 0 KV clean: executors always use seq 0, and the next consumer on
    // this context (fresh executor or pooled conversation on seq 0) assumes its
    // sequence starts empty. Stale cells cause position conflicts that ABORT
    // the next llama_decode ("failed to initialize batch", rc=-1). Mirrors the
    // cleanup pvx_pool_return does for pooled conversations.
    {
        std::lock_guard<std::mutex> lock(exec->ctx_ref->infer_mtx);
        if (exec->n_tokens > 0) {
            llama_memory_seq_rm(exec->ctx_ref->mem, 0, 0, -1);
            // Vulkan stale-cells fix (#26744): flush after clearing — same as pvx_pool_return
            if (pvx_backend_needs_kv_flush(exec->ctx_ref)) pvx_kv_flush(exec->ctx_ref, 0);
        }
    }
    delete exec;
}

pvx_result_t pvx_executor_prompt(pvx_executor_t* exec, const char* text) {
    if (!exec || !text) return PVX_ERR_INVALID_ARG;
    reset_grammar_state(&exec->grammar_state);  // new prompt = new generation
    std::vector<llama_token> tokens;
    if (do_tokenize(exec->ctx_ref->vocab, text, tokens, false, true) < 0) return PVX_ERR_INTERNAL;
    exec->pending_tokens.insert(exec->pending_tokens.end(), tokens.begin(), tokens.end());
    exec->has_pending = true;
    return PVX_OK;
}

pvx_result_t pvx_executor_prompt_tokens(pvx_executor_t* exec, const int32_t* tokens, int count) {
    if (!exec || !tokens || count <= 0) return PVX_ERR_INVALID_ARG;
    // NOTE: no grammar reset here — PromptTokens feeds sampled tokens back into
    // the executor DURING generation; resetting would restart the grammar at
    // position 0 every token. Text prompts (pvx_executor_prompt) reset instead.
    exec->pending_tokens.insert(exec->pending_tokens.end(), tokens, tokens + count);
    exec->has_pending = true;
    return PVX_OK;
}

pvx_decode_result_t pvx_executor_infer(pvx_executor_t* exec) {
    if (!exec) return PVX_DECODE_FAILED;
    std::lock_guard<std::mutex> lock(exec->ctx_ref->infer_mtx);

    if (exec->pending_tokens.empty()) return PVX_DECODE_NO_WORK;

    // Chunk large prompts into ≤ n_batch slices — llama_decode ABORTS the process
    // (GGML_ASSERT n_tokens_all <= n_batch) if handed more tokens in one call.
    // LLamaSharp chunked transparently; this layer must do it itself.
    const uint32_t n_batch = llama_n_batch(exec->ctx_ref->ctx);
    if (n_batch == 0) return PVX_DECODE_FAILED;
    int n_tokens = (int)exec->pending_tokens.size();
    const int kv_base = exec->n_kv_pos;  // positions advance per slice — capture once

    int decoded = 0;
    while (decoded < n_tokens) {
        int slice = std::min((int)n_batch, n_tokens - decoded);
        bool last_slice = (decoded + slice == n_tokens);
        llama_batch batch = llama_batch_init(slice, 0, exec->ctx_ref->n_seq_max);
        llama_seq_id seq0 = 0;

        for (int i = 0; i < slice; i++) {
            batch.token[i] = exec->pending_tokens[decoded + i];
            batch.pos[i] = kv_base + decoded + i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = seq0;
            batch.logits[i] = last_slice && (i == slice - 1) ? 1 : 0;
        }
        batch.n_tokens = slice;

        int rc = llama_decode(exec->ctx_ref->ctx, batch);
        llama_batch_free(batch);

        if (rc != 0) {
            // Decode failed — keep REMAINING pending tokens so caller can retry.
            // Already-decoded slices were committed inside the loop (n_tokens/
            // n_kv_pos/recent_tokens updated per slice — do NOT re-add here).
            exec->recent_tokens.insert(exec->recent_tokens.end(),
                                       exec->pending_tokens.begin(),
                                       exec->pending_tokens.begin() + decoded);
            for (int i = 0; i < decoded; i++)
                pvx_mirror_push_token(exec, exec->pending_tokens[i], kv_base + i);
            exec->pending_tokens.erase(exec->pending_tokens.begin(),
                                       exec->pending_tokens.begin() + decoded);
            exec->ctx_ref->last_error = "llama_decode failed (rc=" + std::to_string(rc) + ")";
            PVX_LOG(PVX_LOG_ERROR, "Decode", "llama_decode failed rc=%d, slice=%d of %d tokens", rc, slice, exec->n_tokens);
            return PVX_DECODE_FAILED;
        }

        // ABI v7 (feature 4): EAGLE3 feature capture — features for these rows
        // die at the NEXT decode on this context; the hook must see them now.
        if (exec->spec && exec->spec->is_eagle()) {
            if (!exec->spec->eagle_process_target_rows(
                    kv_base + decoded, slice, exec->pending_tokens.data() + decoded)) {
                exec->ctx_ref->last_error = exec->spec->last_error();
                PVX_LOG(PVX_LOG_ERROR, "Decode", "eagle3 capture failed on infer slice");
                return PVX_DECODE_FAILED;
            }
        }
        exec->recent_tokens.insert(exec->recent_tokens.end(),
                                   exec->pending_tokens.begin() + decoded,
                                   exec->pending_tokens.begin() + decoded + slice);
        for (int i = 0; i < slice; i++)
            pvx_mirror_push_token(exec, exec->pending_tokens[decoded + i],
                                  kv_base + decoded + i);
        exec->n_tokens += slice;
        exec->n_kv_pos += slice;
        if (last_slice) {
            exec->last_batch_idx = slice - 1;  // index into THIS slice's logits
            // Snapshot NOW — the row is only valid until the next decode on
            // this shared context (any later slice/cycle invalidates it).
            snapshot_logits_row(exec->ctx_ref->ctx, exec->last_logits, slice - 1);
        }
        decoded += slice;
    }

    exec->pending_tokens.clear();
    exec->has_pending = false;
    return PVX_DECODE_OK;
}

pvx_result_t pvx_executor_sample(pvx_executor_t* exec,
                                  const pvx_sampling_params_t* params,
                                  int32_t* out_token) {
    if (!exec || !out_token) return PVX_ERR_INVALID_ARG;
    std::lock_guard<std::mutex> lock(exec->ctx_ref->infer_mtx);

    if (exec->last_logits.empty()) return PVX_ERR_DECODE_FAILED;
    int32_t token = do_sample(exec->last_logits.data(), exec->ctx_ref->vocab, params,
                              exec->recent_tokens);
    if (token < 0) return PVX_ERR_DECODE_FAILED;

    // Track for repeat penalty
    exec->recent_tokens.push_back(token);
    *out_token = token;
    return PVX_OK;
}

int pvx_executor_token_count(pvx_executor_t* exec) { return exec ? exec->n_tokens : 0; }

pvx_result_t pvx_executor_rewind(pvx_executor_t* exec, int n_tokens) {
    if (!exec || n_tokens < 0) return PVX_ERR_INVALID_ARG;
    if (n_tokens > exec->n_tokens) n_tokens = exec->n_tokens;
    if (n_tokens == 0) return PVX_OK;
    std::lock_guard<std::mutex> lock(exec->ctx_ref->infer_mtx);
    // Note: the spec session (if any) needs no hook here — mirror_prefix()
    // detects the shrunken target state at the next round and rebuilds the
    // draft mirror (append-only validity rule in pvx_spec_engine.h).
    int start = exec->n_tokens - n_tokens;
    if (!llama_memory_seq_rm(exec->ctx_ref->mem, 0, start, -1)) {
        exec->ctx_ref->last_error = "partial rewind refused by memory (hybrid/recurrent model "
                                    "cannot partially erase state): seq_rm(" +
                                    std::to_string(start) + ", -1) failed";
        return PVX_ERR_REWIND_UNSUPPORTED;
    }
    reset_grammar_state(&exec->grammar_state);  // generation restarts after rewind
    exec->n_tokens -= n_tokens;
    exec->n_kv_pos -= n_tokens;
    // Rewound tokens include the sampled row's token — snapshot is stale.
    exec->last_batch_idx = -1;
    exec->last_logits.clear();
    if ((int)exec->recent_tokens.size() > n_tokens)
        exec->recent_tokens.resize(exec->recent_tokens.size() - n_tokens);
    else
        exec->recent_tokens.clear();
    // Mirror ring retreats with the window: the rewound positions are gone
    // from KV, so their entries (token or embd) are dropped with them.
    if ((int)exec->mirror.size() > n_tokens)
        exec->mirror.resize(exec->mirror.size() - n_tokens);
    else
        exec->mirror.clear();
    return PVX_OK;
}

pvx_result_t pvx_executor_reset(pvx_executor_t* exec) {
    if (!exec) return PVX_ERR_INVALID_ARG;
    std::lock_guard<std::mutex> lock(exec->ctx_ref->infer_mtx);
    reset_grammar_state(&exec->grammar_state);
    llama_memory_seq_rm(exec->ctx_ref->mem, 0, 0, -1);
    exec->n_tokens = 0;
    exec->n_kv_pos = 0;
    exec->has_pending = false;
    exec->pending_tokens.clear();
    exec->recent_tokens.clear();
    exec->mirror.clear();
    exec->last_batch_idx = -1;
    exec->last_logits.clear();
    return PVX_OK;
}

// ── Speculative decoding (sequential path, INF BL-05) ───────────────────────
// Capability self-report + fail-closed config + per-executor spec session.
// The engine (pvx_spec_engine.h) owns all verify-loop mechanics; this layer
// only validates, binds, and tears down.

pvx_result_t pvx_model_spec_capabilities(const pvx_model_t* model,
                                         pvx_spec_capabilities_t* out_caps) {
    if (!model || !model->model || !out_caps) return PVX_ERR_INVALID_ARG;
    // Supported-types list from GGUF-derived metadata (llama.cpp already parsed
    // arch + tensor inventory at load):
    //   - nextn/MTP layers present  -> PVX_SPEC_TYPE_MTP (wiring boundary:
    //     enabling MTP fails typed NOT_SUPPORTED naming spec-engine/mtp-wiring)
    //   - any arch with a standard vocab -> PVX_SPEC_TYPE_DRAFT_SIMPLE: since
    //     llama.cpp bump 5e03bdd8->f3363593 (2026-10-16), speculative.cpp
    //     handles draft-model speculation on recurrent/hybrid targets too
    //     (deferred-boundary g_embd stash, recurrent-rollback anchors —
    //     #29923/#29928/#30397). The old dense-only gate was removed; fail-
    //     closed behavior stays (empty list only when there is no vocab).
    const bool mtp = llama_model_n_layer_nextn(model->model) > 0;
    const bool vocab_std = llama_vocab_type(model->vocab) != LLAMA_VOCAB_TYPE_NONE;
    uint32_t types = 0;
    if (mtp) types |= PVX_SPEC_TYPE_MTP;
    if (vocab_std) {
        types |= PVX_SPEC_TYPE_DRAFT_SIMPLE;
        // n-gram drafting needs no draft model and no arch support: any
        // standard-vocab target qualifies (ABI v6).
        types |= PVX_SPEC_TYPE_NGRAM;
        types |= PVX_SPEC_TYPE_NGRAM_MAP_K;
        types |= PVX_SPEC_TYPE_NGRAM_MOD;
        types |= PVX_SPEC_TYPE_NGRAM_CACHE;
    }
    out_caps->types = types;
    out_caps->mtp   = mtp;
    out_caps->draft = vocab_std;
    // ABI v6: the spec engine's mirror is embd-aware (token rows by id,
    // image rows by projected embedding at pinned positions, draft-width
    // mismatch zero-pads with a warning) — spec sessions continue drafting
    // after multimodal inputs on every standard-vocab target.
    out_caps->spec_vision = true;
    return PVX_OK;
}

pvx_result_t pvx_model_spec_validate(const pvx_model_t* model,
                                     const pvx_spec_params_t* params) {
    if (!model || !params) return PVX_ERR_INVALID_ARG;
    // Model-scoped typed-error plumbing (2026-10-18 matrix fix): the validate
    // path is model-level, so the reason string goes to model->last_error —
    // readable via pvx_model_last_error (ABI v5). const_cast: last_error is a
    // logical out-param on a model owned by the caller; the read-only model
    // contract still holds for everything this function inspects.
    pvx_model_t* m = const_cast<pvx_model_t*>(model);
    switch (params->mode) {
        case PVX_SPEC_OFF:
            return PVX_OK;
        case PVX_SPEC_MTP_AUTO: {
            pvx_spec_capabilities_t caps{};
            pvx_result_t rc = pvx_model_spec_capabilities(model, &caps);
            if (rc != PVX_OK) return rc;
            if (!caps.mtp) {
                m->last_error = "model has no MTP capability (no nextn tensors in "
                                "the GGUF; boundary: spec-engine/capability)";
                return PVX_ERR_NOT_SUPPORTED;
            }
            // capability real; wiring boundary (pvx_spec_engine)
            m->last_error = "MTP capability is present but auto-MTP is not wired "
                            "in this build (requires LLAMA_CONTEXT_TYPE_MTP plumbing "
                            "+ nextn hidden-state extraction; boundary: spec-engine/mtp-wiring)";
            return PVX_ERR_NOT_SUPPORTED;
        }
        case PVX_SPEC_DRAFT: {
            pvx_spec_capabilities_t caps{};
            pvx_result_t rc = pvx_model_spec_capabilities(model, &caps);
            if (rc != PVX_OK) return rc;
            if (!caps.draft) {
                m->last_error = "model has no standard vocabulary for draft-model "
                                "speculation (boundary: spec-engine/capability)";
                return PVX_ERR_NOT_SUPPORTED;
            }
            if (!params->draft_model_path || params->draft_model_path[0] == '\0')
                return PVX_ERR_INVALID_ARG;
            // Full load + vocab compatibility is exercised by spec_init; here we
            // only type-check the config shape (fail-closed at enable time).
            return PVX_OK;
        }
        case PVX_SPEC_NGRAM_SIMPLE:
        case PVX_SPEC_NGRAM_MAP_K:
        case PVX_SPEC_NGRAM_MOD: {
            pvx_spec_capabilities_t caps{};
            pvx_result_t rc = pvx_model_spec_capabilities(model, &caps);
            if (rc != PVX_OK) return rc;
            if (!(caps.types & PVX_SPEC_TYPE_NGRAM)) {
                m->last_error = "model has no standard vocabulary for n-gram "
                                "speculation (boundary: spec-engine/capability)";
                return PVX_ERR_NOT_SUPPORTED;
            }
            if (params->ngram.n_min < 0 || params->ngram.n_max < 0 ||
                (params->ngram.n_min > 0 && params->ngram.n_max > 0 &&
                 params->ngram.n_min > params->ngram.n_max)) {
                m->last_error = "ngram params invalid: need 0 <= n_min <= n_max "
                                "(0 selects per-mode defaults) (boundary: spec-engine/ngram-config)";
                return PVX_ERR_INVALID_ARG;
            }
            return PVX_OK;
        }
        case PVX_SPEC_NGRAM_CACHE: {
            pvx_spec_capabilities_t caps{};
            pvx_result_t rc = pvx_model_spec_capabilities(model, &caps);
            if (rc != PVX_OK) return rc;
            if (!(caps.types & PVX_SPEC_TYPE_NGRAM)) {
                m->last_error = "model has no standard vocabulary for n-gram "
                                "speculation (boundary: spec-engine/capability)";
                return PVX_ERR_NOT_SUPPORTED;
            }
            if (params->ngram.n_min < 0 || params->ngram.n_max < 0 ||
                (params->ngram.n_min > 0 && params->ngram.n_max > 0 &&
                 params->ngram.n_min > params->ngram.n_max)) {
                m->last_error = "ngram params invalid: need 0 <= n_min <= n_max "
                                "(0 selects per-mode defaults) (boundary: spec-engine/ngram-config)";
                return PVX_ERR_INVALID_ARG;
            }
            // Cache file readability is exercised by spec_init; here we type-check
            // the config shape (fail-closed typed error at enable time on a corrupt
            // or unreadable file — never a silent empty cache).
            return PVX_OK;
        }
        case PVX_SPEC_EAGLE3: {
            pvx_spec_capabilities_t caps{};
            pvx_result_t rc = pvx_model_spec_capabilities(model, &caps);
            if (rc != PVX_OK) return rc;
            if (!caps.draft) {
                m->last_error = "model has no standard vocabulary for EAGLE3 "
                                "draft decoding (boundary: spec-engine/capability)";
                return PVX_ERR_NOT_SUPPORTED;
            }
            if (!params->eagle_companion_path || params->eagle_companion_path[0] == '\0')
                return PVX_ERR_INVALID_ARG;
            // Companion metadata + load + target-layer contract + vocab family
            // are exercised typed at spec_init (per-target pair validation —
            // capability is a property of the LOADED PAIR, not the target alone).
            return PVX_OK;
        }
        case PVX_SPEC_DFLASH: {
            pvx_spec_capabilities_t caps{};
            pvx_result_t rc = pvx_model_spec_capabilities(model, &caps);
            if (rc != PVX_OK) return rc;
            if (!caps.draft) {
                m->last_error = "model has no standard vocabulary for DFlash "
                                "draft decoding (boundary: spec-engine/capability)";
                return PVX_ERR_NOT_SUPPORTED;
            }
            if (!params->dflash_companion_path || params->dflash_companion_path[0] == '\0')
                return PVX_ERR_INVALID_ARG;
            // Companion metadata is validated typed at spec_init; the wiring
            // boundary refuses afterwards (never faked).
            m->last_error = "DFlash companion metadata contract is validated at "
                            "spec_init, but the block-diffusion draft loop is not "
                            "wired in this build (boundary: spec-engine/dflash-wiring)";
            return PVX_ERR_NOT_SUPPORTED;
        }
        default:
            return PVX_ERR_INVALID_ARG;
    }
}

pvx_result_t pvx_executor_spec_init(pvx_executor_t* exec,
                                    const pvx_spec_params_t* params) {
    if (!exec || !exec->ctx_ref || !params) return PVX_ERR_INVALID_ARG;
    std::lock_guard<std::mutex> lock(exec->ctx_ref->infer_mtx);

    if (params->mode == PVX_SPEC_OFF) {
        exec->spec.reset();
        return PVX_OK;
    }

    auto engine = PvxBLSpecEngine::create(exec->ctx_ref->owning_model,
                                          exec->ctx_ref, params,
                                          exec->ctx_ref->last_error);
    if (engine && engine->mode() == PVX_SPEC_EAGLE3 && exec->n_tokens > 0) {
        // ABI v7 (feature 4): EAGLE3 g rows are captured per committed position
        // by the target-decode hook — a stream that predates the session has
        // positions the hook never saw, so drafting would run against a
        // phantom prefix. Fail closed at enable: init on a fresh executor
        // (reset) or before the first prompt.
        exec->ctx_ref->last_error =
            "eagle3 spec requires an empty stream at enable: feature capture must "
            "observe every committed position (n_tokens=" + std::to_string(exec->n_tokens) +
            "); reset the executor first (boundary: spec-engine/eagle3-prefill-boundary)";
        engine.reset();
    }
    if (!engine) {
        // last_error carries the typed boundary message; result stays typed:
        //   config-shape violations        -> PVX_ERR_INVALID_ARG
        //   draft model fails to load      -> PVX_ERR_LOAD_FAILED
        //   capability/wiring boundaries   -> PVX_ERR_NOT_SUPPORTED
        const std::string& err = exec->ctx_ref->last_error;
        if (err.find("draft-load") != std::string::npos ||
            err.find("companion-load") != std::string::npos ||
            err.find("ngram-cache") != std::string::npos ||
            err.find("companion-context") != std::string::npos) return PVX_ERR_LOAD_FAILED;
        if (err.find("draft-config") != std::string::npos ||
            err.find("companion-config") != std::string::npos ||
            err.find("eagle3-config") != std::string::npos ||
            err.find("dflash-config") != std::string::npos ||
            err.find("ngram-config") != std::string::npos ||
            err.find("spec-engine/config") != std::string::npos) return PVX_ERR_INVALID_ARG;
        return PVX_ERR_NOT_SUPPORTED;
    }
    engine->bind(exec);
    exec->spec = std::move(engine);
    return PVX_OK;
}

void pvx_executor_spec_free(pvx_executor_t* exec) {
    if (!exec) return;
    if (exec->ctx_ref) {
        std::lock_guard<std::mutex> lock(exec->ctx_ref->infer_mtx);
        exec->spec.reset();
        return;
    }
    exec->spec.reset();
}

pvx_result_t pvx_executor_spec_stats(pvx_executor_t* exec,
                                     int64_t* out_drafted,
                                     int64_t* out_accepted) {
    if (!exec || !out_drafted || !out_accepted) return PVX_ERR_INVALID_ARG;
    std::lock_guard<std::mutex> lock(exec->ctx_ref->infer_mtx);
    if (!exec->spec) {
        *out_drafted = 0;
        *out_accepted = 0;
        return PVX_OK;
    }
    *out_drafted  = exec->spec->drafted();
    *out_accepted = exec->spec->accepted();
    return PVX_OK;
}

pvx_result_t pvx_executor_generate_spec(pvx_executor_t* exec,
                                        const pvx_sampling_params_t* params,
                                        int out_max,
                                        pvx_spec_token_callback_t on_token,
                                        void* user_data,
                                        int* out_generated) {
    if (!exec || !exec->ctx_ref || !out_generated) return PVX_ERR_INVALID_ARG;
    std::lock_guard<std::mutex> lock(exec->ctx_ref->infer_mtx);
    if (!exec->spec) {
        exec->ctx_ref->last_error =
            "no spec session on this executor; enable via pvx_executor_spec_init "
            "(boundary: spec-engine/session)";
        return PVX_ERR_NOT_SUPPORTED;
    }
    return exec->spec->generate(params, out_max, on_token, user_data, out_generated);
}

// ── Batched/pool-lane spec (ABI v7 feature 5) ─────────────────────────────

pvx_result_t pvx_conversation_spec_init(pvx_conversation_t* conv,
                                        const pvx_spec_params_t* params) {
    if (!conv || !conv->ctx_ref || !params) return PVX_ERR_INVALID_ARG;
    std::lock_guard<std::mutex> lock(conv->ctx_ref->infer_mtx);

    if (params->mode == PVX_SPEC_OFF) {
        conv->spec.reset();
        return PVX_OK;
    }
    // Plan feature 5(f): batched spec needs spare seq room — typed refusal
    // naming the boundary (never a silent single-seq degradation).
    if (conv->ctx_ref->n_seq_max < 2) {
        conv->ctx_ref->last_error =
            "batched spec requires >=2 seq slots (context n_seq_max=" +
            std::to_string(conv->ctx_ref->n_seq_max) + "; boundary: spec-engine/batched-seq-budget)";
        return PVX_ERR_NOT_SUPPORTED;
    }
    // EAGLE3: per-position feature capture is executor-lane wiring (the
    // conversation commit sites have no capture hook) — typed refusal on
    // this lane, never a degraded draft.
    if (params->mode == PVX_SPEC_EAGLE3) {
        conv->ctx_ref->last_error =
            "eagle3 spec is wired on the sequential executor lane only; the "
            "batched/pool conversation lane refuses typed (boundary: "
            "spec-engine/eagle3-conversation-lane)";
        return PVX_ERR_NOT_SUPPORTED;
    }
    // Fail closed when the stream predates the ring's coverage (e.g. after a
    // checkpoint import): drafting would replay a phantom prefix.
    if (conv->n_tokens > 0 && conv->mirror.size() != (size_t) conv->n_tokens) {
        conv->ctx_ref->last_error =
            "conversation mirror ring does not cover the committed stream (" +
            std::to_string(conv->mirror.size()) + " rows vs " +
            std::to_string(conv->n_tokens) + " tokens); the spec engine refuses to "
            "draft against a phantom prefix (boundary: spec-engine/mirror-consistency)";
        return PVX_ERR_NOT_SUPPORTED;
    }
    auto engine = PvxBLSpecEngine::create(conv->ctx_ref->owning_model,
                                          conv->ctx_ref, params,
                                          conv->ctx_ref->last_error);
    if (!engine) {
        const std::string& err = conv->ctx_ref->last_error;
        if (err.find("draft-load") != std::string::npos ||
            err.find("companion-load") != std::string::npos ||
            err.find("ngram-cache") != std::string::npos ||
            err.find("companion-context") != std::string::npos) return PVX_ERR_LOAD_FAILED;
        if (err.find("draft-config") != std::string::npos ||
            err.find("companion-config") != std::string::npos ||
            err.find("eagle3-config") != std::string::npos ||
            err.find("dflash-config") != std::string::npos ||
            err.find("ngram-config") != std::string::npos ||
            err.find("spec-engine/config") != std::string::npos) return PVX_ERR_INVALID_ARG;
        return PVX_ERR_NOT_SUPPORTED;
    }
    engine->bind_conversation(conv);
    conv->spec = std::move(engine);
    return PVX_OK;
}

void pvx_conversation_spec_free(pvx_conversation_t* conv) {
    if (!conv || !conv->ctx_ref) { return; }
    std::lock_guard<std::mutex> lock(conv->ctx_ref->infer_mtx);
    conv->spec.reset();
}

pvx_result_t pvx_conversation_generate_spec(pvx_conversation_t* conv,
                                            const pvx_sampling_params_t* params,
                                            int out_max,
                                            pvx_spec_token_callback_t on_token,
                                            void* user_data,
                                            int* out_generated) {
    if (!conv || !conv->ctx_ref || !out_generated) return PVX_ERR_INVALID_ARG;
    // Same lock discipline as the sample/generate ops on the shared batched
    // context: a spec round performs several target decodes (draft chain
    // verify + commit); the cycle-gate (exclusive use) is the CALLER's
    // contract — same as the plain sample/infer cycle. Locking here prevents
    // cross-conversation decode interleaving INSIDE a round; plain
    // conversations continue between rounds.
    std::lock_guard<std::mutex> lock(conv->ctx_ref->infer_mtx);
    if (!conv->spec) {
        conv->ctx_ref->last_error =
            "no spec session on this conversation; enable via pvx_conversation_spec_init "
            "(boundary: spec-engine/session)";
        return PVX_ERR_NOT_SUPPORTED;
    }
    if (conv->n_tokens == 0 || conv->last_logits.empty()) return PVX_ERR_DECODE_FAILED;
    return conv->spec->generate(params, out_max, on_token, user_data, out_generated);
}

pvx_result_t pvx_conversation_spec_stats(pvx_conversation_t* conv,
                                         int64_t* out_drafted,
                                         int64_t* out_accepted) {
    if (!conv || !conv->ctx_ref || !out_drafted || !out_accepted) return PVX_ERR_INVALID_ARG;
    std::lock_guard<std::mutex> lock(conv->ctx_ref->infer_mtx);
    if (!conv->spec) {
        *out_drafted = 0;
        *out_accepted = 0;
        return PVX_OK;
    }
    *out_drafted  = conv->spec->drafted();
    *out_accepted = conv->spec->accepted();
    return PVX_OK;
}

// ── State save/restore ──
// Uses llama_memory_seq_cp to copy KV to a spare seq_id, then restore by copying back

pvx_result_t pvx_state_save(pvx_executor_t* exec, pvx_state_t** out_state) {
    if (!exec || !out_state) return PVX_ERR_INVALID_ARG;
    auto* state = new pvx_state_s();
    state->n_tokens = exec->n_tokens;
    // Snapshot the committed-stream mirror (embd rows included) so a restore
    // resumes the spec substrate against the same rows the backup KV holds.
    state->mirror = exec->mirror;
    // ABI v7 (feature 4): EAGLE3 per-position g rows ride with the backup —
    // restore replays the draft decoder from them (pending boundary included),
    // never resuming against a stale deferred state.
    if (exec->spec && exec->spec->is_eagle()) {
        exec->spec->eagle_state_capture(state->eagle_g_rows);
    }
    // Copy KV from seq 0 to seq 1 (backup)
    if (exec->n_tokens > 0) {
        if (exec->ctx_ref->n_seq_max < 2) {
            delete state;
            return PVX_ERR_NOT_SUPPORTED;  // need >=2 seq_ids for backup
        }
        std::lock_guard<std::mutex> lock(exec->ctx_ref->infer_mtx);
        // Clear the backup seq FIRST — llama_memory_seq_cp MERGES into the
        // destination; a stale seq 1 from a previous save/restore cycle would
        // corrupt the backup (restores would resurrect old cells).
        llama_memory_seq_rm(exec->ctx_ref->mem, 1, 0, -1);
        llama_memory_seq_cp(exec->ctx_ref->mem, 0, 1, 0, -1);
    }
    *out_state = state;
    return PVX_OK;
}

pvx_result_t pvx_state_restore(pvx_executor_t* exec, pvx_state_t* state) {
    if (!exec || !state) return PVX_ERR_INVALID_ARG;
    std::lock_guard<std::mutex> lock(exec->ctx_ref->infer_mtx);
    // Clear current KV for seq 0, then copy from backup seq 1
    llama_memory_seq_rm(exec->ctx_ref->mem, 0, 0, -1);
    if (state->n_tokens > 0 && exec->ctx_ref->n_seq_max > 1) {
        llama_memory_seq_cp(exec->ctx_ref->mem, 1, 0, 0, -1);
    } else if (state->n_tokens > 0 && exec->ctx_ref->n_seq_max <= 1) {
        // Can't restore from backup seq — no spare seq_id
        exec->n_tokens = 0;
        exec->last_batch_idx = -1;
        exec->last_logits.clear();
        return PVX_OK;
    }
    exec->n_tokens = state->n_tokens;
    exec->n_kv_pos = state->n_tokens;
    // Restore the mirror verbatim, rebased to the restored window's position
    // contract [0, n_tokens): restore semantics pin the continuation at
    // position n_tokens regardless of any pre-save shift_left, so the ring's
    // pinned positions are rebased by the saved front offset to match. Embd
    // rows are preserved exactly (they cannot be re-derived from KV).
    exec->mirror = state->mirror;
    if (!exec->mirror.empty()) {
        const llama_pos front = exec->mirror.front().pos[0];
        if (front != 0)
            for (auto& e : exec->mirror)
                for (auto& p : e.pos) p -= front;
    }
    // ABI v7 (feature 4): EAGLE3 restore — the draft decoder rebuilds from the
    // saved g rows (full training-pair replay, boundary deferred). A mismatched
    // or absent payload on a live session invalidates it typed (the caller
    // must re-init spec; the next generate refuses NOT_SUPPORTED naming the
    // boundary) — restore itself still succeeds (the stream is fine).
    if (exec->spec && exec->spec->is_eagle()) {
        std::vector<std::vector<float>> rows = std::move(state->eagle_g_rows);
        if (!exec->spec->eagle_state_restore(std::move(rows))) {
            PVX_LOG(PVX_LOG_WARN, "Spec", "eagle3 session invalidated by restore: %s",
                    exec->spec->last_error().c_str());
            exec->spec.reset();
        }
    }
    // Restored KV may not include the sampled row's token — snapshot is stale.
    exec->last_batch_idx = -1;
    exec->last_logits.clear();
    return PVX_OK;
}

void pvx_state_free(pvx_state_t* state) { delete state; }

int pvx_state_token_count(pvx_state_t* state) { return state ? state->n_tokens : 0; }

// ── Conversation pool ──

pvx_result_t pvx_pool_create(pvx_context_t* ctx, pvx_pool_t** out_pool) {
    if (!ctx || !out_pool) return PVX_ERR_INVALID_ARG;
    auto* pool = new pvx_pool_s();
    pool->ctx_ref = ctx;
    for (uint32_t i = 0; i < ctx->n_seq_max; i++) {
        auto conv = std::make_unique<pvx_conversation_s>();
        conv->seq_id = (llama_seq_id)i;
        conv->ctx = ctx->ctx;
        conv->mem = ctx->mem;
        conv->ctx_ref = ctx;
        pool->conversations.push_back(std::move(conv));
        pool->available.push(pool->conversations.back().get());
    }
    *out_pool = pool;
    return PVX_OK;
}

void pvx_pool_free(pvx_pool_t* pool) { delete pool; }

pvx_result_t pvx_pool_lease(pvx_pool_t* pool, pvx_conversation_t** out_conv) {
    if (!pool || !out_conv) return PVX_ERR_INVALID_ARG;
    std::lock_guard<std::mutex> lock(pool->mtx);
    if (pool->available.empty()) return PVX_ERR_NO_SLOT;
    *out_conv = pool->available.top();
    pool->available.pop();
    std::lock_guard<std::mutex> ilock(pool->ctx_ref->infer_mtx);
    pool->ctx_ref->active_conversations.push_back(*out_conv);
    return PVX_OK;
}

pvx_result_t pvx_pool_return(pvx_pool_t* pool, pvx_conversation_t* conv) {
    if (!pool || !conv) return PVX_ERR_INVALID_ARG;
    {
        std::lock_guard<std::mutex> ilock(pool->ctx_ref->infer_mtx);
        auto& active = pool->ctx_ref->active_conversations;
        auto it = std::find(active.begin(), active.end(), conv);
        if (it != active.end()) active.erase(it);
    }
    // LLM-24 (2026-10-01): lock order pool->mtx → infer_mtx now matches
    // pvx_pool_lease (the previous order was inverted: infer_mtx first, then
    // pool->mtx — a lease/return pair could interleave deadlocks). The seq_rm
    // and KV flush additionally MUST hold infer_mtx: a concurrently running
    // pvx_infer (InferAll) on the shared context raced this teardown and
    // corrupted Metal buffers — the sporadic ggml_metal aborts on teardown.
    std::lock_guard<std::mutex> lock(pool->mtx);
    {
        std::lock_guard<std::mutex> ilock(pool->ctx_ref->infer_mtx);
        // ABI v7 (feature 5): the conversation's spec session dies with the
        // lease — draft state, caches and g rows are per-conversation. Freed
        // HERE (inside the existing lock order pool->mtx → infer_mtx) so the
        // teardown cannot race a spec round or another lane's session.
        conv->spec.reset();
        if (conv->n_tokens > 0) {
            llama_memory_seq_rm(conv->mem, conv->seq_id, 0, -1);
            conv->n_tokens = 0; conv->n_kv_pos = 0;
            // Vulkan stale cells fix (#26744): flush stale K/V after clearing
            if (pvx_backend_needs_kv_flush(pool->ctx_ref)) {
                pvx_kv_flush(pool->ctx_ref, conv->seq_id);
            }
        }
        conv->mirror.clear();  // ABI v7: ring retreats with the window
    }
    conv->has_pending_prompt = false;
    conv->pending_tokens.clear();
    conv->last_batch_idx = -1;
    conv->last_logits.clear();
    pool->available.push(conv);
    return PVX_OK;
}

int pvx_pool_available(pvx_pool_t* pool) {
    if (!pool) return 0;
    std::lock_guard<std::mutex> lock(pool->mtx);
    return (int)pool->available.size();
}

uint32_t pvx_pool_size(pvx_pool_t* pool) { return pool ? (uint32_t)pool->conversations.size() : 0; }

// ── Conversation ──

pvx_result_t pvx_conversation_prompt(pvx_conversation_t* conv, const char* text) {
    if (!conv || !text) return PVX_ERR_INVALID_ARG;
    const llama_vocab* vocab = llama_model_get_vocab(llama_get_model(conv->ctx));
    std::vector<llama_token> tokens;
    if (do_tokenize(vocab, text, tokens, false, true) < 0) return PVX_ERR_INTERNAL;
    conv->pending_tokens.insert(conv->pending_tokens.end(), tokens.begin(), tokens.end());
    conv->has_pending_prompt = true;
    return PVX_OK;
}

pvx_result_t pvx_conversation_prompt_tokens(pvx_conversation_t* conv,
                                             const int32_t* tokens, int count) {
    if (!conv || !tokens || count <= 0) return PVX_ERR_INVALID_ARG;
    conv->pending_tokens.insert(conv->pending_tokens.end(), tokens, tokens + count);
    conv->has_pending_prompt = true;
    return PVX_OK;
}

pvx_result_t pvx_conversation_sample(pvx_conversation_t* conv,
                                     const pvx_sampling_params_t* params,
                                     int32_t* out_token) {
    if (!conv || !out_token) return PVX_ERR_INVALID_ARG;
    // Sample from the PRIVATE logits snapshot (taken at decode time). Reading
    // llama_get_logits_ith(last_batch_idx) directly here would check against
    // the context's CURRENT batch — under batched continuous inference any
    // interleaved decode (other conversations' slices/cycles) invalidates
    // recorded rows and GGML_ABORTs the process.
    if (conv->n_tokens == 0 || conv->last_logits.empty()) return PVX_ERR_INVALID_ARG;
    // Repeat/present penalties now apply in batch mode via conv->recent_tokens
    // (mirrors the executor path).
    int32_t token = do_sample(conv->last_logits.data(),
                              llama_model_get_vocab(llama_get_model(conv->ctx)),
                              params, conv->recent_tokens);
    if (token < 0) return PVX_ERR_DECODE_FAILED;
    conv->recent_tokens.push_back(token);
    *out_token = token;
    return PVX_OK;
}

int pvx_conversation_token_count(pvx_conversation_t* conv) { return conv ? conv->n_tokens : 0; }

bool pvx_model_partial_rewind_supported(const pvx_model_t* model) {
    if (!model || !model->model) return false;
    // Hybrid (attention+recurrent) and purely recurrent memories refuse partial
    // seq_rm ranges: recurrent state cannot be partially erased (the bounded
    // per-token rollback ring is disabled by default, single-use, and not
    // suitable for arbitrary prefix-cache rewinds). Dense attention memories —
    // including M-RoPE ones — truncate fine.
    const auto arch = model->model->arch;
    return !llm_arch_is_hybrid(arch) && !llm_arch_is_recurrent(arch);
}

// Shared guard for partial rewinds: llama_memory_seq_rm returning false means
// the memory refused the partial range (hybrid/recurrent state cannot be
// rewound). Counters MUST stay untouched in that case — the KV still holds the
// old cells, so moving token-count bookkeeping would silently desync positions
// from the cache (M-RoPE: hard decode failure; scalar RoPE: X+1 continuity
// violation). Surface the refusal instead of corrupting state.
static pvx_result_t rewind_seq_rm_or_refuse(pvx_context_t* ctx_ref, llama_memory_t mem,
                                            llama_seq_id seq_id, llama_pos start) {
    if (!llama_memory_seq_rm(mem, seq_id, start, -1)) {
        ctx_ref->last_error = "partial rewind refused by memory (hybrid/recurrent model "
                              "cannot partially erase state): seq_rm(" +
                              std::to_string(start) + ", -1) failed";
        return PVX_ERR_REWIND_UNSUPPORTED;
    }
    return PVX_OK;
}

pvx_result_t pvx_conversation_rewind(pvx_conversation_t* conv, int n_tokens) {
    if (!conv || n_tokens < 0) return PVX_ERR_INVALID_ARG;
    if (n_tokens > conv->n_tokens) n_tokens = conv->n_tokens;
    if (n_tokens == 0) return PVX_OK;
    // Truncate the memory FIRST — only then move counters/bookkeeping. A
    // refused partial seq_rm must leave the conversation exactly as it was.
    // infer_mtx: KV mutation must not race a concurrent pvx_infer on the
    // shared batched context (same discipline as the executor path).
    std::lock_guard<std::mutex> lock(conv->ctx_ref->infer_mtx);
    int start = conv->n_tokens - n_tokens;
    pvx_result_t rm = rewind_seq_rm_or_refuse(conv->ctx_ref, conv->mem, conv->seq_id, start);
    if (rm != PVX_OK) return rm;
    reset_grammar_state(&conv->grammar_state);  // generation restarts after rewind
    if ((int)conv->recent_tokens.size() > n_tokens)
        conv->recent_tokens.resize(conv->recent_tokens.size() - n_tokens);
    else
        conv->recent_tokens.clear();
    // ABI v7 feature 5: the conversation mirror retreats with the window
    // (executor-lane rewind discipline — dropped positions leave the ring).
    if ((int)conv->mirror.size() > n_tokens)
        conv->mirror.resize(conv->mirror.size() - n_tokens);
    else
        conv->mirror.clear();
    conv->n_tokens -= n_tokens; conv->n_kv_pos -= n_tokens;
    // Rewound tokens include the sampled row's token — snapshot is stale.
    conv->last_batch_idx = -1;
    conv->last_logits.clear();
    return PVX_OK;
}

// Internal: reset WITHOUT taking infer_mtx — callers that already hold the
// context's infer mutex (shift_left full-clear) use this directly.
static void conv_reset_unlocked(pvx_conversation_t* conv) {
    reset_grammar_state(&conv->grammar_state);
    conv->recent_tokens.clear();
    conv->mirror.clear();
    if (conv->n_tokens > 0) llama_memory_seq_rm(conv->mem, conv->seq_id, 0, -1);
    conv->n_tokens = 0; conv->n_kv_pos = 0;
    conv->has_pending_prompt = false;
    conv->pending_tokens.clear();
    conv->last_batch_idx = -1;
    conv->last_logits.clear();
}

pvx_result_t pvx_conversation_reset(pvx_conversation_t* conv) {
    if (!conv) return PVX_ERR_INVALID_ARG;
    std::lock_guard<std::mutex> lock(conv->ctx_ref->infer_mtx);
    conv_reset_unlocked(conv);
    return PVX_OK;
}

// Conversation state save/restore (for batch rewind)
pvx_result_t pvx_conversation_save(pvx_conversation_t* conv, pvx_state_t** out_state) {
    if (!conv || !out_state) return PVX_ERR_INVALID_ARG;
    auto* state = new pvx_state_s();
    state->n_tokens = conv->n_tokens;
    // ABI v7 (feature 5): snapshot the per-conversation mirror with the
    // rewind point — the batched lane's rewind (restore) retreats the ring.
    state->mirror = conv->mirror;
    *out_state = state;
    return PVX_OK;
}

pvx_result_t pvx_conversation_restore(pvx_pool_t* pool, pvx_conversation_t* conv,
                                       pvx_state_t* state) {
    (void)pool;  // not needed — restore operates on conv directly
    if (!pool || !conv || !state) return PVX_ERR_INVALID_ARG;
    std::lock_guard<std::mutex> lock(conv->ctx_ref->infer_mtx);
    reset_grammar_state(&conv->grammar_state);  // generation rewind → fresh grammar
    int to_rewind = conv->n_tokens - state->n_tokens;
    if (to_rewind > 0) {
        pvx_result_t rm = rewind_seq_rm_or_refuse(conv->ctx_ref, conv->mem, conv->seq_id,
                                                  state->n_tokens);
        if (rm != PVX_OK) return rm;
        conv->n_tokens = state->n_tokens; conv->n_kv_pos = state->n_tokens;
        // ABI v7 (feature 5): the mirror retreats with the rewind point
        // (executor-lane discipline). The conversation's spec session (if
        // any) rebuilds its draft state from the shrunk ring at the next
        // round — its lookup streams and draft mirror derive from the ring.
        conv->mirror = state->mirror;
        // Rewound tokens include the sampled row's token — snapshot is stale.
        conv->last_batch_idx = -1;
        conv->last_logits.clear();
    }
    (void)pool;
    return PVX_OK;
}

void pvx_conversation_state_free(pvx_state_t* state) { delete state; }

// ── Conversation checkpoint (suspend/resume to disk) ──
// Blob layout (little-endian):
//   [int32 n_tokens][int32 n_kv_pos][int32 n_recent][int32 n_pending]
//   [recent x n_recent][pending x n_pending][llama seq-state data]

static constexpr size_t PVX_EXPORT_HEADER = 4 * sizeof(int32_t);

pvx_result_t pvx_conversation_export(pvx_conversation_t* conv, uint8_t** out_buf, size_t* out_size) {
    if (!conv || !out_buf || !out_size) return PVX_ERR_INVALID_ARG;

    size_t state_size = 0;
    const size_t n_recent = conv->recent_tokens.size();
    const size_t n_pending = conv->pending_tokens.size();
    size_t total = 0;
    uint8_t* buf = nullptr;
    {
        // State serialization AND the token-array copies are not thread-safe
        // against concurrent llama_decode / pending-token commits on the shared
        // context — capture sizes, copy recent/pending, and serialize the seq
        // state under ONE infer_mtx hold (suspend can be called from a registry
        // thread while other conversations are mid-batch).
        std::lock_guard<std::mutex> lock(conv->ctx_ref->infer_mtx);
        state_size = llama_state_seq_get_size(conv->ctx, conv->seq_id);
        total = PVX_EXPORT_HEADER
            + n_recent * sizeof(llama_token)
            + n_pending * sizeof(llama_token)
            + state_size;
        buf = (uint8_t*)malloc(total);
        if (!buf) return PVX_ERR_INTERNAL;

        auto* hdr = (int32_t*)buf;
        hdr[0] = conv->n_tokens;
        hdr[1] = conv->n_kv_pos;
        hdr[2] = (int32_t)n_recent;
        hdr[3] = (int32_t)n_pending;
        memcpy(buf + PVX_EXPORT_HEADER, conv->recent_tokens.data(), n_recent * sizeof(llama_token));
        memcpy(buf + PVX_EXPORT_HEADER + n_recent * sizeof(llama_token),
               conv->pending_tokens.data(), n_pending * sizeof(llama_token));
        if (state_size > 0) {
            size_t written = llama_state_seq_get_data(conv->ctx,
                buf + PVX_EXPORT_HEADER + (n_recent + n_pending) * sizeof(llama_token),
                state_size, conv->seq_id);
            if (written == 0) { free(buf); return PVX_ERR_INTERNAL; }
            (void)written;
        }
    }

    *out_buf = buf;
    *out_size = total;
    return PVX_OK;
}

void pvx_conversation_export_free(uint8_t* buf) { free(buf); }

pvx_result_t pvx_conversation_import(pvx_conversation_t* conv, const uint8_t* data, size_t size) {
    if (!conv || !data || size < PVX_EXPORT_HEADER) return PVX_ERR_INVALID_ARG;

    auto* hdr = (const int32_t*)data;
    const int32_t n_tokens = hdr[0];
    const int32_t n_kv_pos = hdr[1];
    const int32_t n_recent = hdr[2];
    const int32_t n_pending = hdr[3];
    if (n_tokens < 0 || n_kv_pos < 0 || n_recent < 0 || n_pending < 0) return PVX_ERR_INVALID_ARG;

    const size_t tokens_off = PVX_EXPORT_HEADER;
    const size_t recent_off = tokens_off + (size_t)n_recent * sizeof(llama_token);
    const size_t pending_off = recent_off + (size_t)n_pending * sizeof(llama_token);
    const size_t state_off = pending_off;
    if (size < state_off) return PVX_ERR_INVALID_ARG;
    const size_t state_size = size - state_off;

    const llama_token* recent = (const llama_token*)(data + tokens_off);
    const llama_token* pending = (const llama_token*)(data + recent_off);

    {
        std::lock_guard<std::mutex> lock(conv->ctx_ref->infer_mtx);

        // Destination seq must be empty — clear any residue (defensive: a
        // previous pool-return could have skipped cleaning if the counters
        // were zero while cells remained).
        llama_memory_seq_rm(conv->mem, conv->seq_id, 0, -1);

        if (state_size > 0) {
            size_t read = llama_state_seq_set_data(conv->ctx, data + state_off, state_size, conv->seq_id);
            if (read == 0) return PVX_ERR_DECODE_FAILED;
        }
    }

    conv->n_tokens = n_tokens;
    conv->n_kv_pos = n_kv_pos;
    conv->recent_tokens.assign(recent, recent + n_recent);
    conv->pending_tokens.assign(pending, pending + n_pending);
    conv->has_pending_prompt = n_pending > 0;
    // Restored KV cannot trust any prior logits snapshot — force re-sample path.
    conv->last_batch_idx = -1;
    conv->last_logits.clear();
    reset_grammar_state(&conv->grammar_state);
    return PVX_OK;
}

// ── Batched inference ──

pvx_decode_result_t pvx_infer(pvx_context_t* ctx) {
    if (!ctx) return PVX_DECODE_FAILED;
    std::lock_guard<std::mutex> lock(ctx->infer_mtx);

    std::vector<llama_token> all_tokens;
    std::vector<llama_seq_id> all_seq_ids;
    std::vector<int> all_pos;
    std::set<int> logits_indices;

    int current_idx = 0;
    // Snapshot pending tokens — only commit on successful decode
    struct conv_pending {
        pvx_conversation_t* conv;
        std::vector<llama_token> tokens;
        int first_idx;  // first index in combined batch
        int committed = 0;  // tokens already decoded (committed) from this conv
    };
    std::vector<conv_pending> pending;
    for (auto* conv : ctx->active_conversations) {
        if (conv->pending_tokens.empty()) continue;
        conv_pending cp;
        cp.conv = conv;
        cp.tokens = conv->pending_tokens;
        cp.first_idx = current_idx;
        for (int i = 0; i < (int)cp.tokens.size(); i++) {
            all_tokens.push_back(cp.tokens[i]);
            all_seq_ids.push_back(conv->seq_id);
            all_pos.push_back(conv->n_kv_pos + i);
            current_idx++;
        }
        logits_indices.insert(current_idx - 1);
        pending.push_back(std::move(cp));
    }

    if (all_tokens.empty()) return PVX_DECODE_NO_WORK;

    // Chunk into ≤ n_batch slices — llama_decode ABORTS the process
    // (GGML_ASSERT n_tokens_all <= n_batch) if handed more tokens in one call.
    // LLamaSharp chunked transparently; this layer must do it itself.
    const uint32_t n_batch = llama_n_batch(ctx->ctx);
    if (n_batch == 0) { ctx->last_error = "n_batch is 0"; return PVX_DECODE_FAILED; }

    int total = (int)all_tokens.size();
    int gi = 0;  // global index of current slice start
    while (gi < total) {
        int end = std::min(gi + (int)n_batch, total);
        int n_slice = end - gi;
        llama_batch batch = llama_batch_init(n_slice, 0, ctx->n_seq_max);

        for (int i = gi; i < end; i++) {
            batch.token[i - gi] = all_tokens[i];
            batch.pos[i - gi] = all_pos[i];
            batch.n_seq_id[i - gi] = 1;
            batch.seq_id[i - gi][0] = all_seq_ids[i];
            // logits only for a conv's final pending token when THIS slice covers it
            batch.logits[i - gi] = logits_indices.count(i) ? 1 : 0;
        }
        batch.n_tokens = n_slice;

        int rc = llama_decode(ctx->ctx, batch);
        llama_batch_free(batch);

        if (rc != 0) {
            ctx->last_error = "llama_decode failed (rc=" + std::to_string(rc) + ")";
            PVX_LOG(PVX_LOG_ERROR, "Batch", "llama_decode failed rc=%d", rc);
            // Successfully decoded earlier slices are already committed. Trim the
            // committed prefix from each conv's pending_tokens so a retry does not
            // re-decode them (positions would then diverge from KV).
            for (auto& cp : pending) {
                if (cp.committed > 0)
                    cp.conv->pending_tokens.erase(cp.conv->pending_tokens.begin(),
                                                  cp.conv->pending_tokens.begin() + cp.committed);
            }
            return PVX_DECODE_FAILED;
        }

        // Commit per-conversation coverage of this slice
        for (auto& cp : pending) {
            int conv_start = cp.first_idx;
            int conv_end = cp.first_idx + (int)cp.tokens.size();
            int s = std::max(gi, conv_start);
            int e = std::min(end, conv_end);
            if (s >= e) continue;
            int covered = e - s;
            cp.conv->n_tokens += covered;
            cp.conv->n_kv_pos += covered;
            cp.conv->recent_tokens.insert(cp.conv->recent_tokens.end(),
                                          cp.tokens.begin() + (s - conv_start),
                                          cp.tokens.begin() + (s - conv_start) + covered);
            // ABI v7 feature 5: per-conversation mirror ring — one entry per
            // committed row (batched lane positions are plain 1-D).
            for (int j = 0; j < covered; j++) {
                pvx_conv_mirror_push_token(cp.conv,
                                           cp.tokens[(size_t)(s - conv_start) + j],
                                           cp.conv->n_kv_pos - covered + j);
            }
            cp.committed += covered;
            if (e == conv_end) {
                cp.conv->last_batch_idx = conv_end - 1 - gi;  // logits index within THIS slice
                // Snapshot the logits row NOW — it is only addressable while
                // THIS slice's batch is the ctx's current batch. Conversations
                // ending in EARLIER slices of a chunked batch (or sampled after
                // a later infer cycle) would otherwise read a stale row and
                // GGML_ABORT (get_logits_ith: "batch.logits[i] != true").
                snapshot_logits_row(ctx->ctx, cp.conv->last_logits, cp.conv->last_batch_idx);
            }
        }
        gi = end;
    }

    for (auto& cp : pending) {
        cp.conv->pending_tokens.clear();
        cp.conv->has_pending_prompt = false;
    }
    return PVX_DECODE_OK;
}

// ── Embeddings ──

pvx_result_t pvx_get_embeddings(pvx_model_t* model, pvx_context_t* ctx,
                                const char* text,
                                float** out_embeddings, int* out_dim) {
    if (!model || !ctx || !text || !out_embeddings || !out_dim) return PVX_ERR_INVALID_ARG;
    std::lock_guard<std::mutex> lock(ctx->infer_mtx);

    std::vector<llama_token> tokens;
    if (do_tokenize(ctx->vocab, text, tokens, false, true) < 0) return PVX_ERR_INTERNAL;

    // Use llama_batch_get_one — llama.cpp handles encoder-only models (BERT) correctly.
    // Manual llama_batch_init + llama_decode fails on BERT ("calling encode() instead").
    int n = (int)tokens.size();
    if (n <= 0) {
        ctx->last_error = "embeddings: input produced no tokens";
        return PVX_ERR_INVALID_ARG;
    }
    llama_batch batch = llama_batch_get_one(tokens.data(), n);

    // Route encode vs decode by the MODEL'S OWN architecture — never by trial.
    // llama_encode on a decoder-only model builds an ENCODER graph on a causal
    // arch (the graph path the model does not define) and crashes the process
    // (2026-10-18 matrix: rc=139, EXC_BAD_ACCESS on Llama-3B and Qwen3.5-4B).
    // llama_decode on an encoder-only model has the mirror-image failure, so
    // each arch goes to its own entry point; llama_decode internally forwards
    // to encode() when the context has no decode memory, keeping enc-dec models
    // working. Fail CLOSED with a typed error for models that have neither.
    const bool has_enc = llama_model_has_encoder(model->model);
    const bool has_dec = llama_model_has_decoder(model->model);
    if (!has_enc && !has_dec) {
        ctx->last_error = "embeddings: model exposes neither encoder nor decoder (arch "+
                         std::string("cannot produce embeddings through this path; boundary: embeddings/arch)");
        return PVX_ERR_NOT_SUPPORTED;
    }

    // Decoder-only models whose OUTPUT head dimension differs from the hidden
    // state (e.g. merged multi-token/MTP heads: Qwen3.8-Flash-Next reports
    // n_embd_out = 4 * n_embd) cannot be pooled through this path: llama.cpp's
    // internal MEAN/CLS/LAST extraction reads n_embd_out floats from an
    // n_embd-wide pooled tensor and its upstream bounds assert ABORTS the
    // process (observed rc=134 on the 2026-10-18 verify run). Fail CLOSED with
    // a typed error instead - never an abort, never truncated-head garbage
    // dressed up as an embedding.
    if (has_dec && !has_enc &&
        llama_model_n_embd_out(model->model) != llama_model_n_embd(model->model)) {
        ctx->last_error = "embeddings: model's output head (n_embd_out=" +
                         std::to_string(llama_model_n_embd_out(model->model)) +
                         ") differs from its hidden state (n_embd=" +
                         std::to_string(llama_model_n_embd(model->model)) +
                         ") - merged multi-token output head cannot be pooled "
                         "honestly (boundary: embeddings/pooling)";
        return PVX_ERR_NOT_SUPPORTED;
    }

    // Must enable embeddings output before calling encode/decode.
    llama_set_embeddings(ctx->ctx, true);
    int enc_rc = has_enc ? llama_encode(ctx->ctx, batch) : llama_decode(ctx->ctx, batch);
    if (enc_rc != 0) {
        llama_set_embeddings(ctx->ctx, false);
        ctx->last_error = "Embedding decode/encode failed (rc=" + std::to_string(enc_rc) + ")";
        return PVX_ERR_DECODE_FAILED;
    }

    int n_embd = llama_model_n_embd(model->model);
    float* embd = (float*)malloc(n_embd * sizeof(float));
    if (!embd) return PVX_ERR_INTERNAL;

    memset(embd, 0, n_embd * sizeof(float));

    // Fail-closed extraction: if NO source produces data, hand back a typed
    // error — never zeros dressed up as an embedding, never a segv.
    bool got_data = false;

    switch (ctx->pool_type) {
        case PVX_POOL_CLS: {
            float* seq_emb = llama_get_embeddings_seq(ctx->ctx, 0);
            if (seq_emb) { memcpy(embd, seq_emb, n_embd * sizeof(float)); got_data = true; }
            else {
                float* tok = llama_get_embeddings_ith(ctx->ctx, 0);
                if (tok) { memcpy(embd, tok, n_embd * sizeof(float)); got_data = true; }
            }
            break;
        }
        case PVX_POOL_LAST: {
            float* seq_emb = llama_get_embeddings_seq(ctx->ctx, 0);
            if (seq_emb) { memcpy(embd, seq_emb, n_embd * sizeof(float)); got_data = true; }
            else {
                float* tok = llama_get_embeddings_ith(ctx->ctx, n - 1);
                if (tok) { memcpy(embd, tok, n_embd * sizeof(float)); got_data = true; }
            }
            break;
        }
        case PVX_POOL_MEAN:
        default: {
            // Try sequence-level embedding first (llama.cpp does pooling internally)
            float* seq_emb = llama_get_embeddings_seq(ctx->ctx, 0);
            if (seq_emb) {
                memcpy(embd, seq_emb, n_embd * sizeof(float));
                got_data = true;
            } else {
                // Fallback: manual mean pooling from token embeddings
                int got_tokens = 0;
                for (int i = 0; i < n; i++) {
                    float* tok = llama_get_embeddings_ith(ctx->ctx, i);
                    if (tok) {
                        for (int j = 0; j < n_embd; j++) embd[j] += tok[j];
                        got_tokens++;
                    }
                }
                if (got_tokens > 0) {
                    for (int j = 0; j < n_embd; j++) embd[j] /= got_tokens;
                    got_data = true;
                }
            }
            break;
        }
    }

    if (!got_data) {
        free(embd);
        llama_set_embeddings(ctx->ctx, false);
        ctx->last_error = "embeddings: model produced no embedding output for this "
                          "configuration (arch + pooling combination unsupported; "
                          "boundary: embeddings/pooling)";
        return PVX_ERR_NOT_SUPPORTED;
    }

    // Normalize
    float norm = 0;
    for (int j = 0; j < n_embd; j++) norm += embd[j] * embd[j];
    norm = sqrtf(norm);
    if (norm > 0) for (int j = 0; j < n_embd; j++) embd[j] /= norm;

    llama_set_embeddings(ctx->ctx, false);

    *out_embeddings = embd;
    *out_dim = n_embd;
    return PVX_OK;
}

void pvx_free_embeddings(float* embeddings) { free(embeddings); }
int pvx_embedding_dim(pvx_model_t* model) {
    return model ? llama_model_n_embd(model->model) : 0;
}

// ── Vision / MTMD ──
// Replaces: MtmdWeights
//
// Flow:
// 1. pvx_load_mmproj: load mmproj file → mtmd_context (stored on pvx_model_t)
// 2. pvx_convision_load_image: decode image bytes → mtmd_bitmap
// 3. pvx_conversation_prompt_with_images: tokenize prompt+images into chunks,
//    encode image chunks, feed text tokens + image embeddings into the conversation

pvx_result_t pvx_load_mmproj(pvx_model_t* model, const char* mmproj_path,
                             pvx_model_t** out_mmproj) {
    if (!model || !mmproj_path || !out_mmproj)
        return PVX_ERR_INVALID_ARG;

    // We don't load mmproj as a regular model — mtmd_init_from_file handles it.
    // Create a lightweight wrapper to hold the mtmd context.
    auto* mmproj_model = new pvx_model_s();

    // Initialize mtmd context with the text model reference
    mtmd_context_params mparams = mtmd_context_params_default();
    mparams.use_gpu = true;
    mparams.print_timings = false;
    mparams.n_threads = 1;

    mmproj_model->mtmd_ctx = mtmd_init_from_file(mmproj_path, model->model, mparams);
    if (!mmproj_model->mtmd_ctx) {
        delete mmproj_model;
        return PVX_ERR_LOAD_FAILED;
    }
    // pvxllm standalone (2026-10-06): executor-path vision needs mtmd_ctx on the MAIN
    // model (pvx_executor_prompt_with_images reads model->mtmd_ctx). Shared pointer;
    // freed in pvx_free_model. pvx_free_mmproj must not free it.
    // Lifetime guard (audit 2026-10-17): a SECOND mmproj load while a wrapper is
    // alive would mtmd_free the context that wrapper still references (UAF).
    // Fail closed instead; after all wrappers are freed a replacement load is
    // allowed (old context freed first — zero live references then).
    if (model->mtmd_shared_refs > 0) {
        delete mmproj_model;
        model->last_error = "a vision encoder is already loaded on this model; "
                            "free it before loading another (mtmd context is shared)";
        return PVX_ERR_NOT_SUPPORTED;
    }
    if (model->mtmd_ctx) mtmd_free(model->mtmd_ctx);
    model->mtmd_ctx = mmproj_model->mtmd_ctx;
    mmproj_model->mmproj_owner = model;
    {
        std::lock_guard<std::mutex> lk(model->destruction_mtx);
        model->mtmd_shared_refs++;
    }

    *out_mmproj = mmproj_model;
    return PVX_OK;
}

void pvx_free_mmproj(pvx_model_t* mmproj) {
    if (mmproj) {
        // mtmd_ctx shared with the main model (see pvx_load_mmproj) — do NOT free here.
        if (mmproj->model) llama_model_free(mmproj->model);
        if (mmproj->mmproj_owner) {
            pvx_model_t* owner = mmproj->mmproj_owner;
            std::lock_guard<std::mutex> lk(owner->destruction_mtx);
            owner->mtmd_shared_refs--;
            owner->destruction_cv.notify_all();
            mmproj->mmproj_owner = nullptr;
        }
        mmproj->mtmd_ctx = nullptr;  // shared — owner frees it
        delete mmproj;
    }
}

// Image loading is handled directly in pvx_conversation_prompt_with_images.
// This legacy entry point is kept for API compatibility but is a no-op.
pvx_result_t pvx_conversation_load_image(pvx_conversation_t* conv,
                                          pvx_model_t* mmproj,
                                          const uint8_t* data, int size) {
    (void)conv; (void)mmproj; (void)data; (void)size;
    return PVX_OK;
}

// Full prompt with images: tokenize prompt+images into chunks, encode, feed to conversation
pvx_result_t pvx_conversation_prompt_with_images(pvx_conversation_t* conv,
                                                    pvx_model_t* model_with_mtmd,
                                                    const char* text,
                                                    const uint8_t** image_data,
                                                    const int* image_sizes,
                                                    int n_images) {
    if (!conv || !model_with_mtmd || !model_with_mtmd->mtmd_ctx || !text)
        return PVX_ERR_INVALID_ARG;

    mtmd_context* mctx = model_with_mtmd->mtmd_ctx;

    // Step 1: Load all images as bitmaps
    std::vector<mtmd_bitmap*> bitmaps;
    for (int i = 0; i < n_images; i++) {
        auto wrapper = mtmd_helper_bitmap_init_from_buf(
            mctx, image_data[i], image_sizes[i], false, mtmd_helper_init_opt_default());
        if (!wrapper.bitmap) {
            for (auto* b : bitmaps) mtmd_bitmap_free(b);
            return PVX_ERR_INTERNAL;
        }
        bitmaps.push_back(wrapper.bitmap);
    }

    // Step 2: Tokenize prompt + images into chunks
    mtmd_input_text input_text = {};
    input_text.text = text;
    input_text.text_len = strlen(text);
    input_text.add_special = false;
    input_text.parse_special = false;

    mtmd_input_chunks* chunks = mtmd_input_chunks_init();
    // Debug: log marker + prompt for diagnosis

    int32_t rc = mtmd_tokenize(mctx, chunks, &input_text,
                               bitmaps.data(), bitmaps.size());

    // Free bitmaps (tokenize consumed them)
    for (auto* b : bitmaps) mtmd_bitmap_free(b);

    if (rc != 0) {
        mtmd_input_chunks_free(chunks);
        return PVX_ERR_INTERNAL;
    }

    // Step 3: Process each chunk sequentially.
    // Text chunks → queue tokens for the next pvx_infer() call (batched with other conversations).
    // Image chunks → encode + decode immediately (embd batches can't be queued as pending tokens).
    //   Before each image decode, flush any pending text tokens so they go in their own token-batch.
    //
    // IMPORTANT: All llama_decode calls in this function must be serialized via infer_mtx.
    // The C# layer guarantees exclusive cycle-gate access for vision, but we lock here
    // too for safety against direct calls.
    std::mutex* infer_mtx = nullptr;
    // Find the context's mutex via the conversation's context
    // conv->ctx is llama_context*, we need pvx_context_t* for the mutex
    // We lock per-decode below using a fallback: since we don't have pvx_context_t* here,
    // the C# layer MUST ensure exclusive access. This is documented in CAPABILITY.md.
    size_t n_chunks = mtmd_input_chunks_size(chunks);

    for (size_t i = 0; i < n_chunks; i++) {
        const mtmd_input_chunk* chunk = mtmd_input_chunks_get(chunks, i);
        auto chunk_type = mtmd_input_chunk_get_type(chunk);

        if (chunk_type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            // Text chunk — extract tokens, queue them
            size_t n_tokens = 0;
            const llama_token* tokens = mtmd_input_chunk_get_tokens_text(chunk, &n_tokens);
            if (tokens && n_tokens > 0) {
                conv->pending_tokens.insert(conv->pending_tokens.end(),
                                           tokens, tokens + n_tokens);
            }
        } else if (chunk_type == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
            // Before encoding the image, flush any pending text tokens.
            // Text tokens use batch.token; image embeddings use batch.embd —
            // they cannot coexist in the same llama_batch.
            if (!conv->pending_tokens.empty()) {
                // Chunked flush: llama_decode aborts if n_tokens_all > n_batch
                const uint32_t vflush_n_batch = llama_n_batch(conv->ctx);
                if (vflush_n_batch == 0) {
                    mtmd_input_chunks_free(chunks);
                    set_error(nullptr, "n_batch is 0");
                    return PVX_ERR_DECODE_FAILED;
                }
                const int vflush_base = conv->n_kv_pos;  // capture once — advance per slice
                int flushed = 0;
                while (flushed < (int)conv->pending_tokens.size()) {
                    int n_slice = std::min((int)vflush_n_batch,
                                           (int)conv->pending_tokens.size() - flushed);
                    llama_batch tbatch = llama_batch_init(n_slice, 0, 1);
                    for (int j = 0; j < n_slice; j++) {
                        tbatch.token[j] = conv->pending_tokens[flushed + j];
                        tbatch.pos[j] = vflush_base + flushed + j;
                        tbatch.n_seq_id[j] = 1;
                        tbatch.seq_id[j][0] = conv->seq_id;
                        tbatch.logits[j] = 0;
                    }
                    tbatch.n_tokens = n_slice;
                    int rc2 = llama_decode(conv->ctx, tbatch);
                    llama_batch_free(tbatch);
                    if (rc2 != 0) {
                        mtmd_input_chunks_free(chunks);
                        set_error(nullptr, std::string("text flush before image decode failed: ") + std::to_string(rc2));
                        return PVX_ERR_DECODE_FAILED;
                    }
                    conv->n_tokens += n_slice; conv->n_kv_pos += n_slice;
                    conv->recent_tokens.insert(conv->recent_tokens.end(),
                                               conv->pending_tokens.begin() + flushed,
                                               conv->pending_tokens.begin() + flushed + n_slice);
                    for (int j = 0; j < n_slice; j++) {
                        pvx_conv_mirror_push_token(conv,
                                                   conv->pending_tokens[(size_t) flushed + j],
                                                   conv->n_kv_pos - n_slice + j);
                    }
                    flushed += n_slice;
                }
                conv->pending_tokens.clear();
            }

            // Check if this model needs non-causal attention for image decoding
            bool need_non_causal = mtmd_decode_use_non_causal(mctx, chunk);
            if (need_non_causal) {
                llama_set_causal_attn(conv->ctx, false);
            }

            // Encode the image chunk
            int32_t enc_rc = mtmd_encode_chunk(mctx, chunk);
            if (enc_rc != 0) {
                if (need_non_causal) llama_set_causal_attn(conv->ctx, true);
                mtmd_input_chunks_free(chunks);
                return PVX_ERR_DECODE_FAILED;
            }

            // Get the output embeddings
            float* embd = mtmd_get_output_embd(mctx);
            size_t n_img_tokens = mtmd_input_chunk_get_n_tokens(chunk);
            int n_embd = llama_model_n_embd_inp(llama_get_model(conv->ctx));

            if (embd && n_img_tokens > 0 && n_embd > 0) {
                // Create an embd batch — feed image embeddings directly to llama_decode.
                // embd size = n_img_tokens * n_embd * sizeof(float)
                int n_tokens = (int)n_img_tokens;
                // llama_batch_init with embd=n_embd allocates batch.embd
                llama_batch ebatch = llama_batch_init(n_tokens, n_embd, 1);

                // Copy embeddings into the batch
                memcpy(ebatch.embd, embd, (size_t)n_tokens * n_embd * sizeof(float));

                // Set positions, seq_ids, logits — do NOT touch token (NULL in embd mode)
                for (int j = 0; j < n_tokens; j++) {
                    ebatch.pos[j] = conv->n_kv_pos + j;
                    ebatch.n_seq_id[j] = 1;
                    ebatch.seq_id[j][0] = conv->seq_id;
                    ebatch.logits[j] = (j == n_tokens - 1) ? 1 : 0;
                }
                ebatch.n_tokens = n_tokens;

                int rc3 = llama_decode(conv->ctx, ebatch);
                llama_batch_free(ebatch);

                // Restore causal attention for text decoding
                if (need_non_causal) {
                    llama_set_causal_attn(conv->ctx, true);
                }

                if (rc3 != 0) {
                    mtmd_input_chunks_free(chunks);
                    return PVX_ERR_DECODE_FAILED;
                }

                conv->n_tokens += n_tokens; conv->n_kv_pos += n_tokens;
                // ABI v7 feature 5: embd rows enter the conversation mirror at
                // their pinned (plain, batched-lane) positions — the spec
                // engine replays them by pointer exactly like the executor
                // lane's vision mirror rows.
                for (int j = 0; j < n_tokens; j++) {
                    const std::array<llama_pos, 4> rp = {
                        (llama_pos)(conv->n_kv_pos - n_tokens + j),
                        (llama_pos)(conv->n_kv_pos - n_tokens + j),
                        (llama_pos)(conv->n_kv_pos - n_tokens + j),
                        (llama_pos)(conv->n_kv_pos - n_tokens + j) };
                    pvx_conv_mirror_push_embd(conv,
                                              embd + (size_t) j * n_embd, n_embd, rp);
                }
                // Image chunk token positions are embeddings — no token ids exist,
                // so they cannot contribute to repeat-penalty history.
                conv->last_batch_idx = n_tokens - 1;
                // Snapshot now — the row is only valid until the next decode
                // on this shared context.
                snapshot_logits_row(conv->ctx, conv->last_logits, n_tokens - 1);
            }
        }
    }

    conv->has_pending_prompt = true;
    mtmd_input_chunks_free(chunks);
    return PVX_OK;
}

pvx_result_t pvx_mtmd_marker(char** out_marker) {
    if (!out_marker) return PVX_ERR_INVALID_ARG;
    const char* marker = mtmd_default_marker();
    if (!marker) marker = "<image>";
    *out_marker = (char*)malloc(strlen(marker) + 1);
    strcpy(*out_marker, marker);
    return PVX_OK;
}

const char* pvx_mtmd_marker_static(void) {
    return mtmd_default_marker();
}

// ── Granular KV cache manipulation ──
// These replace LLamaSharp workarounds with direct native operations.

// Shift left: remove oldest n tokens, reposition remaining to start at 0.
// This implements sliding window context — when approaching n_ctx, drop the
// oldest tokens instead of resetting the entire conversation.
pvx_result_t pvx_conversation_shift_left(pvx_conversation_t* conv, int n_tokens) {
    if (!conv) return PVX_ERR_INVALID_ARG;
    if (n_tokens <= 0) return PVX_ERR_INVALID_ARG;
    // infer_mtx: KV mutation must not race a concurrent pvx_infer on the
    // shared batched context (same discipline as the executor path).
    std::lock_guard<std::mutex> lock(conv->ctx_ref->infer_mtx);
    reset_grammar_state(&conv->grammar_state);  // context shift → grammar desyncs
    if (n_tokens >= conv->n_tokens) {
        // Shifting everything = full clear (unlocked helper — lock already held)
        conv_reset_unlocked(conv);
        return PVX_OK;
    }

    // Remove the front n_tokens from KV
    llama_memory_seq_rm(conv->mem, conv->seq_id, 0, n_tokens);

    // Shift remaining tokens: copy from position n_tokens to position 0.
    // llama.cpp handles position reindexing internally when you remove from the front
    // — the remaining tokens stay at their positions but the KV entries after the
    // removed range are now at a lower effective offset. However, the position
    // counter (n_past) for FUTURE decodes must reflect the new base.
    //
    // Actually, llama_memory_seq_rm with p0=0, p1=n removes those positions from KV.
    // The remaining KV entries (positions n..end) are still at their original positions.
    // For the NEXT decode, we must set batch.pos = (n_tokens_old - n_shifted + i),
    // which means our conv->n_tokens must reflect the NEW logical base.
    //
    // The simplest correct approach: after removing front tokens, the conversation's
    // logical position base shifts. We track n_tokens as the count AFTER the shift,
    // and set positions relative to the new base on the next decode.
    //
    // But llama.cpp's KV cache uses absolute positions for RoPE. Removing positions
    // 0..n from KV doesn't change the positions of remaining entries — they're still
    // at n..end. So the next decode must continue from position (old_n_tokens),
    // NOT from (n_tokens - n_shifted).
    //
    // This means shift_left is a KV-only operation: it frees memory but doesn't
    // change the position counter. The conversation continues from where it was,
    // just with the oldest KV entries freed. RoPE positions remain correct.
    conv->n_tokens -= n_tokens;  // logical count shrinks
    // n_kv_pos stays the same — KV positions are absolute
    // NOTE: positions for future decodes still use conv->n_tokens as the base.
    // The KV entries that remain are at positions [n_tokens_shifted .. old_n_tokens-1]
    // in absolute terms, but the model sees them as a contiguous block starting
    // from an offset. The position for the NEXT token is conv->n_tokens (post-shift).
    // This is correct because llama.cpp resolves positions via the KV cache's
    // stored positions, not our counter.

    // Trim recent tokens for repeat penalty
    if ((int)conv->recent_tokens.size() > n_tokens)
        conv->recent_tokens.erase(conv->recent_tokens.begin(),
                                  conv->recent_tokens.begin() + n_tokens);
    else
        conv->recent_tokens.clear();

    // ABI v7 feature 5: mirror ring — freed front positions drop out of the
    // window; surviving entries keep pinned absolute positions (executor-lane
    // shift_left discipline; the spec replay uses entry.pos).
    if ((int)conv->mirror.size() > n_tokens)
        conv->mirror.erase(conv->mirror.begin(),
                           conv->mirror.begin() + n_tokens);
    else
        conv->mirror.clear();

    if (conv->last_batch_idx >= conv->n_tokens) { conv->last_batch_idx = -1; conv->last_logits.clear(); }
    return PVX_OK;
}

pvx_result_t pvx_executor_shift_left(pvx_executor_t* exec, int n_tokens) {
    if (!exec || n_tokens <= 0) return PVX_ERR_INVALID_ARG;
    if (n_tokens >= exec->n_tokens) {
        return pvx_executor_reset(exec);
    }

    std::lock_guard<std::mutex> lock(exec->ctx_ref->infer_mtx);
    llama_memory_seq_rm(exec->ctx_ref->mem, 0, 0, n_tokens);
    exec->n_tokens -= n_tokens;  // logical count shrinks
    // n_kv_pos stays the same — KV positions are absolute, not relative
    if (exec->last_batch_idx >= exec->n_tokens) { exec->last_batch_idx = -1; exec->last_logits.clear(); }

    // Trim recent tokens for repeat penalty
    if ((int)exec->recent_tokens.size() > n_tokens)
        exec->recent_tokens.erase(exec->recent_tokens.begin(),
                                  exec->recent_tokens.begin() + n_tokens);
    else
        exec->recent_tokens.clear();

    // Mirror ring: the freed front positions [0, n_tokens) drop out of the
    // window; surviving entries keep their pinned absolute positions (the
    // draft replay uses entry.pos, not index arithmetic).
    if ((int)exec->mirror.size() > n_tokens)
        exec->mirror.erase(exec->mirror.begin(), exec->mirror.begin() + n_tokens);
    else
        exec->mirror.clear();

    return PVX_OK;
}

// KV copy: copy an entire seq_id's KV to another seq_id.
// Used for state backup without blob serialization.
pvx_result_t pvx_kv_copy(pvx_context_t* ctx, int src_seq_id, int dst_seq_id) {
    if (!ctx) return PVX_ERR_INVALID_ARG;
    if (src_seq_id < 0 || (uint32_t)src_seq_id >= ctx->n_seq_max) return PVX_ERR_INVALID_ARG;
    if (dst_seq_id < 0 || (uint32_t)dst_seq_id >= ctx->n_seq_max) return PVX_ERR_INVALID_ARG;
    std::lock_guard<std::mutex> lock(ctx->infer_mtx);
    // Clear dest first, then copy
    llama_memory_seq_rm(ctx->mem, dst_seq_id, 0, -1);
    llama_memory_seq_cp(ctx->mem, src_seq_id, dst_seq_id, 0, -1);
    return PVX_OK;
}

// KV clear: clear KV for a specific seq_id only.
pvx_result_t pvx_kv_clear_seq(pvx_context_t* ctx, int seq_id) {
    if (!ctx) return PVX_ERR_INVALID_ARG;
    if (seq_id < 0 || (uint32_t)seq_id >= ctx->n_seq_max) return PVX_ERR_INVALID_ARG;
    std::lock_guard<std::mutex> lock(ctx->infer_mtx);
    llama_memory_seq_rm(ctx->mem, seq_id, 0, -1);
    return PVX_OK;
}

// KV size: query token count for a seq_id.
// llama.cpp doesn't expose per-seq KV token count directly.
// Use pvx_conversation_token_count(conv) or pvx_executor_token_count(exec) instead.
int pvx_kv_seq_tokens(pvx_context_t* ctx, int seq_id) {
    (void)seq_id;
    return ctx ? -1 : -1;  // not directly queryable
}

// Causal attention control (per-context, per-call).
pvx_result_t pvx_set_causal_attn(pvx_context_t* ctx, bool causal) {
    if (!ctx) return PVX_ERR_INVALID_ARG;
    std::lock_guard<std::mutex> lock(ctx->infer_mtx);
    llama_set_causal_attn(ctx->ctx, causal);
    return PVX_OK;
}

// Non-causal check for mtmd.
bool pvx_mtmd_needs_non_causal(pvx_model_t* mmproj) {
    if (!mmproj || !mmproj->mtmd_ctx) return false;
    // chunk=nullptr means default case (image chunk)
    return mtmd_decode_use_non_causal(mmproj->mtmd_ctx, nullptr);
}

// ── Anti-prompt detection ──

int pvx_check_anti_prompts(const char* text, const char** anti_prompts, int count) {
    if (!text || !anti_prompts || count <= 0) return -1;
    std::string s(text);
    for (int i = 0; i < count; i++) {
        if (anti_prompts[i] && s.find(anti_prompts[i]) != std::string::npos)
            return i;
    }
    return -1;
}

// ── Grammar (GBNF) ──

struct pvx_grammar_s {
    const llama_vocab* vocab = nullptr;
    std::string grammar_str;
    std::string grammar_root;
};

pvx_grammar_t* pvx_grammar_create(pvx_model_t* model, const char* grammar_str, const char* grammar_root) {
    if (!model || !model->vocab || !grammar_str || !grammar_root) return nullptr;
    // Validate by creating a test sampler
    auto* test = llama_sampler_init_grammar(model->vocab, grammar_str, grammar_root);
    if (!test) return nullptr;
    llama_sampler_free(test);
    // Store strings; the persistent sampler chain is built lazily per generation
// by ensure_grammar_chain_locked() and rebuilt when the grammar changes
    auto* g = new pvx_grammar_s();
    g->vocab = model->vocab;
    g->grammar_str = grammar_str;
    g->grammar_root = grammar_root;
    return g;
}

void pvx_grammar_free(pvx_grammar_t* grammar) {
    if (!grammar) return;
    delete grammar;
}

// Helper: reset grammar sampler state (new generation starts from position 0)
static void reset_grammar_state(pvx_grammar_state_s* gs) {
    if (!gs) return;
    std::lock_guard<std::mutex> lock(gs->mtx);
    if (gs->chain) { llama_sampler_free(gs->chain); gs->chain = nullptr; }
    gs->grammar_str.clear();
    gs->grammar_root.clear();
}

// Ensure the persistent grammar sampler chain matches the requested grammar.
// Rebuilds when the grammar string/root changed (new request → fresh state).
// LOCKING (LLM-24): caller MUST hold gs->mtx — do_sample_with_grammar locks it
// for the whole grammar section so the rebuild can never race a concurrent
// reset from another thread (request-thread grammar reset / checkpoint import).
static bool ensure_grammar_chain_locked(pvx_grammar_state_s* gs, pvx_grammar_t* grammar) {
    if (!gs || !grammar) return false;
    if (gs->chain && (gs->grammar_str != grammar->grammar_str ||
                      gs->grammar_root != grammar->grammar_root)) {
        // inline reset (no re-lock — we already hold gs->mtx)
        llama_sampler_free(gs->chain); gs->chain = nullptr;
        gs->grammar_str.clear();
        gs->grammar_root.clear();
    }
    if (!gs->chain) {
        llama_sampler* chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
        llama_sampler* gsampler = llama_sampler_init_grammar(
            grammar->vocab, grammar->grammar_str.c_str(), grammar->grammar_root.c_str());
        if (!gsampler) {
            llama_sampler_free(chain);
            return false;
        }
        llama_sampler_chain_add(chain, gsampler);
        gs->chain = chain;
        gs->grammar_str = grammar->grammar_str;
        gs->grammar_root = grammar->grammar_root;
    }
    return true;
}

static int32_t do_sample_with_grammar(const float* logits_raw, const llama_vocab* vocab,
                                       const pvx_sampling_params_t* params,
                                       const std::vector<llama_token>& recent_tokens,
                                       pvx_grammar_t* grammar,
                                       pvx_grammar_state_s* grammar_state) {
    if (!params || !logits_raw) return -1;
    if (grammar && !grammar_state) return -1;  // no state holder → cannot sample

    int n_vocab = llama_vocab_n_tokens(vocab);

    // Pooled per-thread selection chain — grammar is NOT in it. The grammar
    // lives in the PERSISTENT conversation/executor chain (state advances via
    // accept()) and is applied via rejection sampling (tiers below). The chain
    // is fetched/validated by fingerprint; NEVER freed per call (thread_local
    // ownership, freed at thread exit or fingerprint change).
    llama_sampler* smpl = pooled_selection_chain(params, recent_tokens, n_vocab);

    // ── Penalties activation (2026-09-26, ARCHITECTURE pm7, owner-approved) ──
    // Feed THIS caller's recent_tokens into the pooled penalties ring via
    // targeted reset+replay. The ring is thread-local (pooled), so it MUST be
    // rebuilt per call to stay conversation-correct under batched
    // interleaving (reset+replay of the caller's window — no cross-contamination).
    // CRITICAL: reset ONLY the penalties sampler, NEVER the chain — chain reset
    // would reseed the dist RNG to its creation seed and corrupt the draw
    // sequence. Penalties off (pen == nullptr) → zero work, zero change.
    if (t_pooled.pen) {
        llama_sampler_reset(t_pooled.pen);
        // Auto window: same FIXED constant as the chain build (64) — must match
        // the ring size or the window would drift between build and replay.
        int pen_last_n = params->repeat_last_n >= 0 ? params->repeat_last_n : 64;
        if (pen_last_n > (int)recent_tokens.size()) pen_last_n = (int)recent_tokens.size();
        for (int i = (int)recent_tokens.size() - pen_last_n; i < (int)recent_tokens.size(); i++)
            llama_sampler_accept(t_pooled.pen, recent_tokens[i]);
    }

    const float* logits = logits_raw;

    // Advance grammar state with the committed token. Skip EOG/control tokens
    // (chat-template control tokens would desync the parse state). Applying the
    // grammar (llama_sampler_apply) is stateless — only accept() advances the
    // automaton — so the single-token and tier-2 checks below cannot corrupt it.
    auto grammar_accept = [&](llama_token t) {
        if (grammar && t >= 0 &&
            !llama_vocab_is_eog(vocab, t) && !llama_vocab_is_control(vocab, t)) {
            llama_sampler_accept(grammar_state->chain, t);
        }
    };

    // ── Grammar: rejection sampling (2026-09-26) ─────────────────────────────
    // llama-server / LLamaSharp semantics. The grammar-first full-vocab mask
    // costs ~130ms/token on a 152k vocab (token_to_piece + decode_utf8 with a
    // heap alloc PER candidate). Instead:
    //   Tier 1: run the selection chain WITHOUT grammar, validate the chosen
    //           token with a 1-element grammar check — O(1), passes ~always for
    //           loose grammars.
    //   Tier 2: grammar-mask the already top_k-truncated candidate array (sorted
    //           desc, size <= top_k) and take the highest-logit survivor — O(k).
    //   Tier 3: full grammar-first mask over the whole vocab (the old path) —
    //           only when the top-k set contains no grammar-valid token.
    // "Output always conforms" holds: every returned token passed a grammar
    // check (tier-1 validation, tier-2 mask, or tier-3 full mask).
    if (grammar) {
        // LLM-24 (2026-10-01): serialize the ENTIRE grammar section against
        // concurrent resets from other threads (owner's request thread resets
        // at generation boundaries; a cancelled step can overlap a driver
        // still holding this conversation). Holding gs->mtx across
        // ensure/apply/accept makes every chain pointer stable for the whole
        // call; a racing reset simply waits and runs before the next token.
        std::lock_guard<std::mutex> glock(grammar_state->mtx);
        if (!ensure_grammar_chain_locked(grammar_state, grammar)) return -1;
        llama_token token = -1;
        // Selection WITHOUT grammar + ignore_eos redraw loop.
        llama_token_data_array kept = {};
        for (int attempt = 0; attempt < 16; attempt++) {
            llama_token_data_array arr;
            llama_token_data* cur = t_candidates((size_t)n_vocab, arr);
            for (int i = 0; i < n_vocab; i++) cur[i].logit = logits[i];
            llama_sampler_apply(smpl, &arr);
            token = arr.data[arr.selected].id;
            // Keep the applied array: sorted desc, size <= top_k (top_k impl
            // truncates). Needed for tier 2.
            kept = arr;
            if (!params->ignore_eos || !llama_vocab_is_eog(vocab, token)) break;
        }

        // Tier 1: single-token grammar validation — the common fast path.
        llama_token_data single = { token, 1.0f, 0.0f };
        llama_token_data_array single_arr = { &single, 1, -1, false };
        llama_sampler_apply(grammar_state->chain, &single_arr);
        if (single.logit != -INFINITY) {
            g_tier1_hits++;
            grammar_accept(token);
            return token;
        }

        // Tier 2: mask the truncated top-k candidates, take best survivor.
        if (params->top_k > 0 && kept.size > 0 && kept.data) {
            llama_sampler_apply(grammar_state->chain, &kept);
            for (size_t i = 0; i < kept.size; i++) {
                llama_token t = kept.data[i].id;
                if (kept.data[i].logit == -INFINITY) continue;
                // ignore_eos: never return EOG through the fallback tiers.
                if (params->ignore_eos && llama_vocab_is_eog(vocab, t)) continue;
                g_tier2_hits++;
                grammar_accept(t);
                return t;
            }
        }

        // Tier 2.5: rescue net — grammar-mask a top-256 slice (raw-logit order,
        // NO top_p truncation), take the highest-logit survivor. Tier 2's `kept`
        // set is top_p-trimmed after top_k; with sharp distributions it holds
        // only a handful of tokens, so when the model fixates on an invalid
        // token (e.g. '+' outside the grammar charset) the whole set is invalid
        // and we would pay tier 3 (~12ms full-vocab mask). nth_element over the
        // thread-local candidates + a 256-candidate mask ≈ 0.5ms.
        {
            const size_t wide_k = std::min<size_t>(256, (size_t)n_vocab);
            llama_token_data_array wide;
            llama_token_data* wcur = t_candidates((size_t)n_vocab, wide);
            for (int i = 0; i < n_vocab; i++) wcur[i].logit = logits[i];
            std::nth_element(wcur, wcur + wide_k, wcur + n_vocab,
                [](const llama_token_data& a, const llama_token_data& b) { return a.logit > b.logit; });
            llama_token_data_array wide_arr = { wcur, wide_k, -1, false };
            llama_sampler_apply(grammar_state->chain, &wide_arr);
            llama_token best = -1; float best_logit = -INFINITY;
            for (size_t i = 0; i < wide_k; i++) {
                if (wcur[i].logit == -INFINITY) continue;
                if (params->ignore_eos && llama_vocab_is_eog(vocab, wcur[i].id)) continue;
                if (wcur[i].logit > best_logit) { best_logit = wcur[i].logit; best = wcur[i].id; }
            }
            if (best >= 0) {
                g_tier2b_hits++;
                grammar_accept(best);
                return best;
            }
        }

        // Tier 3: full grammar-first mask (strict grammars, e.g. JSON schemas).
        {
            auto t0 = std::chrono::steady_clock::now();
            g_tier3_hits++;
            int t1 = g_tier1_hits, t2 = g_tier2_hits;
            PVX_LOG(PVX_LOG_DEBUG, "Grammar", "tier3 full mask: t1=%d t2=%d t2b=%d t3=%d",
                    t1, t2, g_tier2b_hits.load(), g_tier3_hits.load());
            for (int attempt = 0; attempt < 16; attempt++) {
            llama_token_data_array arr;
            llama_token_data* cur = t_candidates((size_t)n_vocab, arr);
            for (int i = 0; i < n_vocab; i++) cur[i].logit = logits[i];
                llama_sampler_apply(grammar_state->chain, &arr);
                llama_sampler_apply(smpl, &arr);
                token = arr.data[arr.selected].id;
                if (!params->ignore_eos || !llama_vocab_is_eog(vocab, token)) break;
            }
            auto t1c = std::chrono::steady_clock::now();
            g_tier3_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t1c - t0).count();
        }
        grammar_accept(token);
        return token;
    }

    // ── No grammar: plain selection chain ──
    llama_token token = -1;
    for (int attempt = 0; attempt < 16; attempt++) {
        llama_token_data_array arr;
        llama_token_data* cur = t_candidates((size_t)n_vocab, arr);
        for (int i = 0; i < n_vocab; i++) cur[i].logit = logits[i];
        llama_sampler_apply(smpl, &arr);
        // OOB GUARD (2026-09-26): dist sets selected=-1 when the array arrives
        // EMPTY — reading data[-1] was undefined behavior. Map it to a clean
        // failure so the diagnostic below fires deterministically.
        token = (arr.selected >= 0 && (size_t)arr.selected < arr.size)
            ? arr.data[arr.selected].id : (llama_token)-1;
        if (!params->ignore_eos || !llama_vocab_is_eog(vocab, token)) break;
    }

    // DIAG (2026-09-26): concurrent streams were failing here with token<0 —
    // capture the exact chain state at the moment of failure (structured log;
    // the raw stderr print leaked into product output).
    if (token < 0) {
        PVX_LOG(PVX_LOG_WARN, "Sample", "plain chain returned token<0: n_vocab=%d temp=%.3f top_k=%d top_p=%.3f "
                "min_p=%.3f rep_pen=%.3f rep_last_n=%d pen=%d recent=%zu",
                n_vocab, params->temperature, params->top_k, params->top_p, params->min_p,
                params->repeat_penalty, params->repeat_last_n, t_pooled.pen ? 1 : 0,
                recent_tokens.size());
    }

    return token;
}

pvx_result_t pvx_executor_sample_grammar(pvx_executor_t* exec, const pvx_sampling_params_t* params,
                                             pvx_grammar_t* grammar, int* out_token) {
    if (!exec || !params || !out_token) return PVX_ERR_INVALID_ARG;
    if (exec->n_tokens == 0 || exec->last_batch_idx < 0) return PVX_ERR_DECODE_FAILED;

    std::lock_guard<std::mutex> lock(exec->ctx_ref->infer_mtx);
    if (exec->last_logits.empty()) return PVX_ERR_DECODE_FAILED;
    int32_t token = do_sample_with_grammar(exec->last_logits.data(), exec->ctx_ref->vocab,
                                            params, exec->recent_tokens, grammar,
                                            &exec->grammar_state);
    if (token < 0) return PVX_ERR_DECODE_FAILED;
    *out_token = token;
    return PVX_OK;
}

// Explicit generation-boundary reset: the caller (C# BatchSession) invokes this
// once per generation BEFORE the sample loop. The conversation re-prompts each
// generated piece as text, so prompt hooks cannot distinguish new turns from
// per-token pieces.
pvx_result_t pvx_conversation_grammar_reset(pvx_conversation_t* conv) {
    if (!conv) return PVX_ERR_INVALID_ARG;
    reset_grammar_state(&conv->grammar_state);
    return PVX_OK;
}

pvx_result_t pvx_executor_grammar_reset(pvx_executor_t* exec) {
    if (!exec) return PVX_ERR_INVALID_ARG;
    reset_grammar_state(&exec->grammar_state);
    return PVX_OK;
}

pvx_result_t pvx_conversation_sample_grammar(pvx_conversation_t* conv, const pvx_sampling_params_t* params,
                                                pvx_grammar_t* grammar, int* out_token) {
    if (!conv || !params || !out_token) return PVX_ERR_INVALID_ARG;
    if (conv->n_tokens == 0 || conv->last_logits.empty()) return PVX_ERR_DECODE_FAILED;

    int32_t token = do_sample_with_grammar(conv->last_logits.data(),
                                            llama_model_get_vocab(llama_get_model(conv->ctx)),
                                            params, conv->recent_tokens, grammar,
                                            &conv->grammar_state);
    if (token < 0) return PVX_ERR_DECODE_FAILED;
    conv->recent_tokens.push_back(token);
    *out_token = token;
    return PVX_OK;
}

// ── Chat template ──

// "</M>" from "<M>": insert a '/' after the opening angle bracket.
static std::string marker_with_slash(const std::string& marker) {
    if (marker.size() < 2 || marker[0] != '<') return marker;
    return marker.substr(0, 1) + "/" + marker.substr(1);
}

pvx_result_t pvx_apply_chat_template(pvx_model_t* model, const char* tmpl,
                                        const pvx_chat_message_t* messages, int n_messages,
                                        bool add_assistant, int enable_thinking,
                                        char** out_text) {
    if (!model || !messages || n_messages <= 0 || !out_text) return PVX_ERR_INVALID_ARG;

    // Convert to llama_chat_message array
    std::vector<llama_chat_message> chat(n_messages);
    for (int i = 0; i < n_messages; i++) {
        chat[i].role = messages[i].role;
        chat[i].content = messages[i].content;
    }

    // First call to get the required size
    int32_t needed = llama_chat_apply_template(tmpl, chat.data(), n_messages, add_assistant, nullptr, 0);
    if (needed <= 0) {
        model->last_error = "chat template apply failed";
        return PVX_ERR_INTERNAL;
    }

    // Allocate and apply
    // Add some slack — llama_chat_apply_template can sometimes need more than reported
    int32_t buf_size = needed + 256;
    char* buf = (char*)malloc(buf_size);
    if (!buf) return PVX_ERR_INTERNAL;

    int32_t written = llama_chat_apply_template(tmpl, chat.data(), n_messages, add_assistant, buf, buf_size);
    if (written <= 0) {
        free(buf);
        model->last_error = "chat template apply failed on second call";
        return PVX_ERR_INTERNAL;
    }

    // v15 (owner-approved 2026-09-26): thinking OFF as structured template state,
    // not prompt text. When enable_thinking==0 and the vocabulary structurally
    // carries a "think" special token (single-token match - Qwen3 family), prefill
    // an EMPTY think block directly after the assistant header - identical to
    // Qwen3's official enable_thinking=false mechanism. Detection is structural
    // (one special token), never model-name-based; vocabularies without the token
    // are untouched and the server-side stream stripping remains the guarantee.
    if (enable_thinking == 0 && add_assistant) {
        // Probe the model's marker list (data-driven): the FIRST marker that
        // exists as a single special token in the vocabulary wins and its
        // empty block is prefilled: "<M>\n\n</M>\n\n".
        for (const auto& marker : model->reasoning_markers) {
            llama_token probe[4];
            int32_t n_tok = llama_tokenize(model->vocab, marker.c_str(),
                                            (int32_t)marker.size(), probe, 4, false, true);
            if (n_tok != 1) continue;
            std::string block = marker + "\n\n" + marker_with_slash(marker) + "\n\n";
            const int32_t block_len = (int32_t)block.size();
            char* grown = (char*)realloc(buf, buf_size + block_len);
            if (!grown) { free(buf); return PVX_ERR_INTERNAL; }
            buf = grown;
            memcpy(buf + written, block.c_str(), block_len);
            written += block_len;
            buf[written] = '\0';
            buf_size += block_len;
            break;  // first match wins
        }
    }

        *out_text = buf;
    return PVX_OK;
}

// ── Vision on standard executor ──

pvx_result_t pvx_executor_prompt_with_images(pvx_executor_t* exec,
                                                 pvx_model_t* model,
                                                 const char* text,
                                                 const char** image_data,
                                                 const int* image_sizes,
                                                 int n_images) {
    if (!exec || !model || !text) return PVX_ERR_INVALID_ARG;
    // ABI v7 (feature 4): EAGLE3 drafting consumes per-position target layer
    // features; multimodal rows are skipped by the upstream eagle3 process()
    // hook, so drafting after an image prompt cannot be kept exact — refuse
    // the prompt typed while the session is live (free the session first, or
    // re-init after the vision turn). Fail closed, never a degraded draft.
    if (exec->spec && exec->spec->is_eagle()) {
        exec->ctx_ref->last_error =
            "eagle3 spec session active: multimodal prompts are a typed refusal "
            "while EAGLE3 drafting runs (upstream process() skips embd rows; "
            "free the session before image prompts) (boundary: spec-engine/eagle3-vision)";
        return PVX_ERR_NOT_SUPPORTED;
    }
    if (!model->mtmd_ctx) {
        model->last_error = "No mmproj loaded — cannot process images";
        return PVX_ERR_INVALID_ARG;
    }
    if (n_images > 0 && (!image_data || !image_sizes)) return PVX_ERR_INVALID_ARG;

    mtmd_context* mctx = model->mtmd_ctx;

    // Step 1: Load all images as bitmaps
    std::vector<mtmd_bitmap*> bitmaps;
    for (int i = 0; i < n_images; i++) {
        auto wrapper = mtmd_helper_bitmap_init_from_buf(
            mctx, (const unsigned char*)image_data[i], image_sizes[i], false, mtmd_helper_init_opt_default());
        if (!wrapper.bitmap) {
            for (auto* b : bitmaps) mtmd_bitmap_free(b);
            return PVX_ERR_INTERNAL;
        }
        bitmaps.push_back(wrapper.bitmap);
    }

    // Step 2: Tokenize prompt + images into chunks
    mtmd_input_text input_text = {};
    input_text.text = text;
    input_text.text_len = strlen(text);
    input_text.add_special = false;
    input_text.parse_special = false;

    mtmd_input_chunks* chunks = mtmd_input_chunks_init();
    int32_t rc = mtmd_tokenize(mctx, chunks, &input_text,
                               bitmaps.data(), bitmaps.size());

    for (auto* b : bitmaps) mtmd_bitmap_free(b);

    if (rc != 0) {
        mtmd_input_chunks_free(chunks);
        return PVX_ERR_INTERNAL;
    }

    // Step 3: Process chunks — text → pending tokens, images → embd decode
    std::lock_guard<std::mutex> lock(exec->ctx_ref->infer_mtx);

    size_t n_chunks = mtmd_input_chunks_size(chunks);
    for (size_t i = 0; i < n_chunks; i++) {
        const mtmd_input_chunk* chunk = mtmd_input_chunks_get(chunks, i);
        auto chunk_type = mtmd_input_chunk_get_type(chunk);

        if (chunk_type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            size_t n_tokens = 0;
            const llama_token* tokens = mtmd_input_chunk_get_tokens_text(chunk, &n_tokens);
            if (tokens && n_tokens > 0) {
                exec->pending_tokens.insert(exec->pending_tokens.end(), tokens, tokens + n_tokens);
                exec->has_pending = true;
            }
        } else if (chunk_type == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
            // Flush pending text tokens first (separate decode batch)
            if (exec->has_pending) {
                int n = (int)exec->pending_tokens.size();
                llama_batch batch = llama_batch_init(n, 0, exec->ctx_ref->n_seq_max);
                for (int j = 0; j < n; j++) {
                    batch.token[j] = exec->pending_tokens[j];
                    batch.pos[j] = exec->n_kv_pos + j;
                    batch.n_seq_id[j] = 1;
                    batch.seq_id[j][0] = 0;
                    batch.logits[j] = 0;
                }
                batch.n_tokens = n;
                int flush_rc = llama_decode(exec->ctx_ref->ctx, batch);
                llama_batch_free(batch);
                if (flush_rc != 0) {
                    // Fail loud: silently dropping the text prefix would push
                    // the image embeddings at the WRONG positions and desync
                    // n_tokens/n_kv_pos from the KV. Pending tokens are kept
                    // so the caller can retry.
                    exec->ctx_ref->last_error =
                        "text flush before image decode failed (rc=" + std::to_string(flush_rc) + ")";
                    mtmd_input_chunks_free(chunks);
                    return PVX_ERR_DECODE_FAILED;
                }
                exec->n_tokens += n;
                exec->n_kv_pos += n;
                // Committed window grew: mirror + penalty history advance with
                // it (same bookkeeping as the conversation lane's text flush).
                exec->recent_tokens.insert(exec->recent_tokens.end(),
                                           exec->pending_tokens.begin(), exec->pending_tokens.end());
                for (int j = 0; j < n; j++)
                    pvx_mirror_push_token(exec, exec->pending_tokens[j], exec->n_kv_pos - n + j);
                exec->pending_tokens.clear();
                exec->has_pending = false;
            }

            // Encode the image chunk and decode its projected embeddings
            // DIRECTLY (conversation-lane pattern) instead of the batched
            // helper: the embd rows must be captured into the executor's
            // mirror ring (spec substrate) — the helper hides them. Non-causal
            // attention only when the projector requires it.
            const bool need_non_causal = mtmd_decode_use_non_causal(mctx, chunk);
            if (need_non_causal) llama_set_causal_attn(exec->ctx_ref->ctx, false);
            int32_t enc_rc = mtmd_encode_chunk(mctx, chunk);
            if (enc_rc != 0) {
                if (need_non_causal) llama_set_causal_attn(exec->ctx_ref->ctx, true);
                exec->ctx_ref->last_error = "image chunk encode failed (rc=" +
                                            std::to_string(enc_rc) + ")";
                mtmd_input_chunks_free(chunks);
                return PVX_ERR_DECODE_FAILED;
            }

            float* embd = mtmd_get_output_embd(mctx);
            const int n_tokens = (int) mtmd_input_chunk_get_n_tokens(chunk);
            const int n_embd   = llama_model_n_embd_inp(llama_get_model(exec->ctx_ref->ctx));
            const bool use_mrope = mtmd_decode_use_mrope(mctx);
            const mtmd_image_tokens* image_tokens = mtmd_input_chunk_get_tokens_image(chunk);

            if (embd && n_tokens > 0 && n_embd > 0 && (!use_mrope || image_tokens)) {
                const llama_pos embd_base = exec->n_kv_pos;
                // Pinned per-row positions, helper semantics exactly
                // (mtmd_helper_decode_image_chunk): M-RoPE image rows carry
                // [t, y, x, z] section-major; normal rows are plain 1-D.
                std::vector<std::array<llama_pos, 4>> rowpos(n_tokens);
                if (use_mrope) {
                    for (int i = 0; i < n_tokens; i++) {
                        const mtmd_decoder_pos dp =
                            mtmd_image_tokens_get_decoder_pos(image_tokens, embd_base, (size_t) i);
                        rowpos[i] = { (llama_pos) dp.t, (llama_pos) dp.y,
                                      (llama_pos) dp.x, (llama_pos) dp.z };
                    }
                } else {
                    for (int i = 0; i < n_tokens; i++)
                        rowpos[i] = { embd_base + i, embd_base + i,
                                      embd_base + i, embd_base + i };
                }

                // Chunked decode through the stable extended-batch API — the
                // same surface the vendored helper uses (never linked, only
                // pattern-ported). Only the final row of the chunk is output.
                const uint32_t n_batch = llama_n_batch(exec->ctx_ref->ctx);
                int offset = 0;
                bool failed = false;
                while (offset < n_tokens) {
                    const int n_slice = std::min((int) n_batch, n_tokens - offset);
                    llama_batch_ext* ext = llama_batch_ext_init(exec->ctx_ref->ctx);
                    for (int j = 0; j < n_slice; j++) {
                        const int i = offset + j;
                        const llama_embd e = { embd + (size_t) i * n_embd, 1, (size_t) n_embd };
                        const int32_t idx = llama_batch_ext_add_embd(ext, 0, e);
                        if (idx < 0 || !llama_batch_ext_set_pos(ext, idx, rowpos[i].data())) {
                            failed = true; break;
                        }
                        if (i == n_tokens - 1)
                            llama_batch_ext_set_output_logits(ext, idx, true);
                    }
                    if (!failed) {
                        const int32_t dec_rc = llama_process(exec->ctx_ref->ctx,
                                                             LLAMA_PROCESS_TYPE_DECODE, ext);
                        if (dec_rc != 0) failed = true;
                    }
                    llama_batch_ext_free(ext);
                    if (failed) break;
                    offset += n_slice;
                }
                if (need_non_causal) llama_set_causal_attn(exec->ctx_ref->ctx, true);
                if (failed) {
                    exec->ctx_ref->last_error = "image embd decode failed";
                    mtmd_input_chunks_free(chunks);
                    return PVX_ERR_DECODE_FAILED;
                }
                exec->n_tokens += n_tokens;
                exec->n_kv_pos += n_tokens;
                exec->last_batch_idx = n_tokens - 1;
                // Mirror: one embd-aware entry per committed image position
                // (id == LLAMA_TOKEN_NULL) — the rows that keep mirror-size ==
                // n_kv_pos after a multimodal prompt.
                for (int i = 0; i < n_tokens; i++)
                    pvx_mirror_push_embd(exec, embd + (size_t) i * n_embd, n_embd, rowpos[i]);
                // Snapshot the last row now — it is only valid until the next
                // decode on this shared context (same discipline as the
                // conversation lane).
                snapshot_logits_row(exec->ctx_ref->ctx, exec->last_logits, n_tokens - 1);
            } else {
                if (need_non_causal) llama_set_causal_attn(exec->ctx_ref->ctx, true);
                // Fail closed: a projector that produced no usable rows must
                // not silently advance nothing while the caller believes the
                // image landed.
                exec->ctx_ref->last_error = "image chunk produced no usable embeddings";
                mtmd_input_chunks_free(chunks);
                return PVX_ERR_DECODE_FAILED;
            }
        }
    }

    mtmd_input_chunks_free(chunks);
    return PVX_OK;
}

// ── Last error ──

const char* pvx_last_error(pvx_context_t* ctx) {
    return ctx ? ctx->last_error.c_str() : "";
}

// Model-scoped last error (ABI v5): model-level failures (spec validate,
// capability queries) land here — NOT on pvx_last_error, which takes a
// CONTEXT handle. Passing a model handle to pvx_last_error reads the wrong
// struct layout (the 2026-10-18 "NOT_SUPPORTED: " empty-message bug).
const char* pvx_model_last_error(const pvx_model_t* model) {
    return model ? model->last_error.c_str() : "";
}

// ── Structured logging implementation (2026-09-25) ──
// Master native-log routing (owner 2026-10-06): ggml_log_set is the ONE hook all
// engine output flows through (llama.cpp core + ggml/Metal backends). Bridging it
// into pvx_log means every native log line reaches the managed callback — the LLM
// server forwards it into its logger, whose file/level are driven by its config
// (logging.file / logging.level). One config, one location, everything included.
static void pvx_ggml_log_bridge(ggml_log_level level, const char* text, void* /*user*/) {
    pvx_log_level_t lv = level == GGML_LOG_LEVEL_ERROR ? PVX_LOG_ERROR
        : level == GGML_LOG_LEVEL_WARN ? PVX_LOG_WARN : PVX_LOG_INFO;
    std::string t(text ? text : "");
    if (!t.empty() && t.back() == '\n') t.pop_back();
    pvx_log(lv, "Native", "%s", t.c_str());
}

extern "C" void pvx_set_log_callback(pvx_log_callback_t cb) {
    g_log_cb = cb;
    if (cb) ggml_log_set(pvx_ggml_log_bridge, nullptr);
}

extern "C" void pvx_set_log_level(pvx_log_level_t level) {
    g_log_min_level = level;
}

extern "C" void pvx_log(pvx_log_level_t level, const char* tag, const char* fmt, ...) {
    char buf[1024];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    if (g_log_file[0]) {
        std::lock_guard<std::mutex> lk(g_log_mu);
        if (g_log_q.size() >= PVX_LOG_Q_CAP) {
            g_log_q.pop_front(); // drop-oldest: a flooding logger never stalls the engine
            g_log_dropped.fetch_add(1, std::memory_order_relaxed);
        }
        pvx_log_entry e{};
        e.level = level;
        snprintf(e.tag, sizeof(e.tag), "%s", tag ? tag : "");
        snprintf(e.msg, sizeof(e.msg), "%s", buf);
        g_log_q.push_back(std::move(e));
        g_log_cv.notify_one();
    }
    if (!g_log_cb || level < g_log_min_level) return;
    g_log_cb((int)level, tag, buf);
}

// ── GGUF EOG metadata repair (owner 2026-10-06) ──────────────────────────────
// Some HuggingFace GGUFs ship with tokenizer.ggml.special_eos_id NOT listed in
// tokenizer.ggml.special_eog_ids — llama.cpp's is_eog() then misses the model's
// own end-of-turn token and generation runs past EOS. This repairs the METADATA
// ONLY (weights are stream-copied byte-for-byte, never decoded, never resident
// in RAM) using the same write sequence as the in-tree gguf-split merge tool.
// Returns PVX_OK + a human report; the repair is idempotent (already-clean files
// report "clean" and are not rewritten).

extern "C" pvx_result_t pvx_repair_eog_metadata(const char* in_path, const char* out_path,
                                                char* report, int report_size) {
    if (!in_path || !out_path || !report || report_size <= 0) return PVX_ERR_INVALID_ARG;
    snprintf(report, report_size, "");

    struct ggml_context* ctx_meta_host = nullptr;
    gguf_init_params ip{ /*no_alloc*/ true, /*ctx*/ &ctx_meta_host };
    gguf_context* g = gguf_init_from_file(in_path, ip);
    if (!g) { snprintf(report, report_size, "failed to open GGUF: %s", in_path); return PVX_ERR_INTERNAL; }

    const char* K_EOS = "tokenizer.ggml.special_eos_id";
    const char* K_EOS2 = "tokenizer.ggml.eos_token_id";
    const char* K_EOG = "tokenizer.ggml.special_eog_ids";
    int64_t eos_k = gguf_find_key(g, K_EOS);
    int64_t eog_k = gguf_find_key(g, K_EOG);

    // Fallback source (verified live): some uploads declare EOS only via
    // tokenizer.ggml.eos_token_id (the qwen gguf has no special_eos_id key).
    bool eos_from_fallback = false;
    if (eos_k < 0) {
        eos_k = gguf_find_key(g, K_EOS2);
        eos_from_fallback = eos_k >= 0;
    }

    // Nothing to do when the model declares no EOS or no EOG list at all —
    // those are valid shapes (is_eog then trusts token type metadata instead).
    if (eos_k < 0 || eog_k < 0) {
        snprintf(report, report_size, "clean: %s absent — nothing to repair",
                 eos_k < 0 ? "special_eos_id" : "special_eog_ids");
        gguf_free(g); ggml_free(ctx_meta_host);
        return PVX_OK;
    }
    if (gguf_get_kv_type(g, eos_k) != GGUF_TYPE_UINT32 || gguf_get_arr_type(g, eog_k) != GGUF_TYPE_UINT32) {
        snprintf(report, report_size, "unexpected value shapes (eos=%d eog_arr=%d) — refusing to touch",
                 (int)gguf_get_kv_type(g, eos_k), (int)gguf_get_arr_type(g, eog_k));
        gguf_free(g); ggml_free(ctx_meta_host);
        return PVX_ERR_INVALID_ARG;
    }
    uint32_t eos = gguf_get_val_u32(g, eos_k);
    size_t n = gguf_get_arr_n(g, eog_k);
    const uint32_t* arr = (const uint32_t*)gguf_get_arr_data(g, eog_k);
    if (n == 0 || !arr) {
        gguf_free(g); ggml_free(ctx_meta_host);
        snprintf(report, report_size, "special_eog_ids empty/invalid — refusing");
        return PVX_ERR_INVALID_ARG;
    }
    if (std::find(arr, arr + n, eos) != arr + n) {
        snprintf(report, report_size, "clean: eos %u already in special_eog_ids (%zu ids)%s", eos, n,
                 eos_from_fallback ? " (source: eos_token_id)" : "");
        gguf_free(g); ggml_free(ctx_meta_host);
        return PVX_OK; // idempotent
    }

    // ── repair: rebuild metadata + stream-copy weights (gguf-split merge sequence) ──
    std::vector<uint32_t> fixed(arr, arr + n);
    fixed.push_back(eos);

    gguf_context* out = gguf_init_empty();
    gguf_set_kv(out, g); // copies ALL metadata
    gguf_set_arr_data(out, K_EOG, GGUF_TYPE_UINT32, fixed.data(), fixed.size()); // overwrites the one key

    size_t n_tensors = gguf_get_n_tensors(g);
    for (size_t i = 0; i < n_tensors; i++) {
        const char* t_name = gguf_get_tensor_name(g, (int64_t)i);
        ggml_tensor* t = ggml_get_tensor(ctx_meta_host, t_name);
        if (!t) { snprintf(report, report_size, "tensor '%s' missing meta", t_name); gguf_free(out); gguf_free(g); ggml_free(ctx_meta_host); return PVX_ERR_INTERNAL; }
        gguf_add_tensor(out, t);
    }

    std::ifstream fin(in_path, std::ios::binary);
    std::ofstream fout(out_path, std::ios::binary | std::ios::trunc);
    if (!fin.is_open() || !fout.is_open()) {
        snprintf(report, report_size, "cannot open files for copy (out: %s)", out_path);
        gguf_free(out); gguf_free(g); ggml_free(ctx_meta_host);
        return PVX_ERR_INTERNAL;
    }

    // placeholder meta block, then tensor bytes + alignment padding, then real meta
    const size_t meta_size = gguf_get_meta_size(out);
    fout.write(std::string(meta_size, '\0').c_str(), meta_size);

    const size_t data_off = gguf_get_data_offset(g);
    std::vector<char> buf;
    for (size_t i = 0; i < n_tensors; i++) {
        const char* t_name = gguf_get_tensor_name(g, (int64_t)i);
        ggml_tensor* t = ggml_get_tensor(ctx_meta_host, t_name);
        const size_t n_bytes = ggml_nbytes(t);
        if (buf.size() < n_bytes) buf.resize(n_bytes);
        fin.seekg((std::streamoff)(data_off + gguf_get_tensor_offset(g, (int64_t)i)));
        fin.read(buf.data(), (std::streamsize)n_bytes);
        if (!fin) { snprintf(report, report_size, "short read on tensor '%s'", t_name); gguf_free(out); gguf_free(g); ggml_free(ctx_meta_host); return PVX_ERR_INTERNAL; }
        fout.write(buf.data(), (std::streamsize)n_bytes);
        const size_t pad = GGML_PAD(n_bytes, GGUF_DEFAULT_ALIGNMENT) - n_bytes;
        fout.write(std::string(pad, '\0').c_str(), (std::streamsize)pad);
    }
    std::vector<uint8_t> meta(meta_size);
    gguf_get_meta_data(out, meta.data());
    fout.seekp(0);
    fout.write((const char*)meta.data(), (std::streamsize)meta.size());
    fout.close();

    gguf_free(out); gguf_free(g); ggml_free(ctx_meta_host);
    snprintf(report, report_size,
             "repaired: eos %u appended to special_eog_ids (%zu -> %zu ids), %zu tensors stream-copied%s",
             eos, n, fixed.size(), n_tensors,
             eos_from_fallback ? " (source: eos_token_id)" : "");
    return PVX_OK;
}
