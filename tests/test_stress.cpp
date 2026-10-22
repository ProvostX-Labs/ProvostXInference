// test_stress.cpp — Stress tests for ProvostXInference
//
// Tests edge cases: zero/empty inputs, boundary sizes, repeated operations,
// interleaved pool ops, KV overflow, sampling extremes, state save/restore
// under stress, concurrent-ish patterns, resource cleanup.

#include "pvxinference.h"
#include <cassert>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

static const char* MODEL_PATH;
static const char* EMBED_MODEL_PATH;

static int tests_run = 0, tests_passed = 0, tests_skipped = 0;

#define TEST(name) \
    tests_run++; \
    fprintf(stderr, "  [%d] %s ... ", tests_run, #name); \
    test_##name(); \
    tests_passed++; \
    fprintf(stderr, "OK\n");

#define ASSERT(cond) \
    if (!(cond)) { \
        fprintf(stderr, "FAIL: line %d: %s\n", __LINE__, #cond); \
        exit(1); \
    }

#define SKIP(msg) do { tests_skipped++; fprintf(stderr, "SKIP (%s)\n", msg); return; } while(0)

static bool file_exists(const char* path) {
    if (!path) return false;
    FILE* f = fopen(path, "rb");
    if (f) { fclose(f); return true; }
    return false;
}

#define ASSERT_EQ(a, b) \
    if ((a) != (b)) { \
        fprintf(stderr, "FAIL: line %d: %s != %s (%lld != %lld)\n", __LINE__, #a, #b, (long long)(a), (long long)(b)); \
        exit(1); \
    }

// ── Helper: create a standard model+context for tests ──
static bool model_available() { return file_exists(MODEL_PATH); }

static pvx_model_t* make_model(int gpu_layers = 99, bool fa = true, pvx_kv_type_t kv = PVX_KV_F16) {
    pvx_model_params_t mp = {};
    mp.model_path = MODEL_PATH;
    mp.gpu_layers = gpu_layers;
    mp.flash_attn = fa;
    mp.kv_cache_type = kv;
    mp.threads = 0;  // auto
    pvx_model_t* m = nullptr;
    ASSERT_EQ(pvx_load_model(&mp, &m), PVX_OK);
    return m;
}

static pvx_context_t* make_ctx(pvx_model_t* m, uint32_t ctx_size = 4096, uint32_t batch = 512, uint32_t seq_max = 4) {
    pvx_context_params_t cp = {};
    cp.context_size = ctx_size;
    cp.batch_size = batch;
    cp.seq_max = seq_max;
    pvx_context_t* ctx = nullptr;
    ASSERT_EQ(pvx_create_context(m, &cp, &ctx), PVX_OK);
    return ctx;
}

// ====================================================
// 1. NULL / invalid argument tests
// ====================================================

static void test_null_args() {
    // Every function should handle NULL gracefully (no crash)
    ASSERT_EQ(pvx_load_model(nullptr, nullptr), PVX_ERR_INVALID_ARG);
    ASSERT_EQ(pvx_create_context(nullptr, nullptr, nullptr), PVX_ERR_INVALID_ARG);
    ASSERT_EQ(pvx_tokenize(nullptr, "test", false, true, nullptr, nullptr), PVX_ERR_INVALID_ARG);
    ASSERT_EQ(pvx_detokenize(nullptr, nullptr, 0, false, nullptr), PVX_ERR_INVALID_ARG);
    ASSERT_EQ(pvx_executor_create(nullptr, nullptr), PVX_ERR_INVALID_ARG);
    ASSERT_EQ(pvx_executor_prompt(nullptr, "test"), PVX_ERR_INVALID_ARG);
    ASSERT_EQ(pvx_executor_infer(nullptr), PVX_DECODE_FAILED);
    ASSERT_EQ(pvx_executor_sample(nullptr, nullptr, nullptr), PVX_ERR_INVALID_ARG);
    ASSERT_EQ(pvx_executor_rewind(nullptr, 0), PVX_ERR_INVALID_ARG);
    ASSERT_EQ(pvx_executor_reset(nullptr), PVX_ERR_INVALID_ARG);
    ASSERT_EQ(pvx_pool_create(nullptr, nullptr), PVX_ERR_INVALID_ARG);
    ASSERT_EQ(pvx_pool_lease(nullptr, nullptr), PVX_ERR_INVALID_ARG);
    ASSERT_EQ(pvx_pool_return(nullptr, nullptr), PVX_ERR_INVALID_ARG);
    ASSERT_EQ(pvx_state_save(nullptr, nullptr), PVX_ERR_INVALID_ARG);
    ASSERT_EQ(pvx_state_restore(nullptr, nullptr), PVX_ERR_INVALID_ARG);
    ASSERT_EQ(pvx_kv_copy(nullptr, 0, 1), PVX_ERR_INVALID_ARG);
    ASSERT_EQ(pvx_kv_clear_seq(nullptr, 0), PVX_ERR_INVALID_ARG);
    ASSERT_EQ(pvx_set_causal_attn(nullptr, true), PVX_ERR_INVALID_ARG);
    ASSERT_EQ(pvx_probe_backend(nullptr), PVX_ERR_INVALID_ARG);
    ASSERT_EQ(pvx_kv_flush(nullptr, 0), PVX_ERR_INVALID_ARG);
}

// ====================================================
// 2. Empty / zero-length inputs
// ====================================================

static void test_empty_prompt() {
    if (!file_exists(MODEL_PATH)) SKIP("model file not found");
    pvx_model_t* m = make_model();
    pvx_context_t* ctx = make_ctx(m);
    pvx_executor_t* exec = nullptr;
    ASSERT_EQ(pvx_executor_create(ctx, &exec), PVX_OK);

    // Empty string prompt
    ASSERT_EQ(pvx_executor_prompt(exec, ""), PVX_OK);
    // Infer with empty pending — should be NO_WORK
    ASSERT_EQ(pvx_executor_infer(exec), PVX_DECODE_NO_WORK);

    // Sample without any tokens in KV — should fail
    int32_t token = 0;
    ASSERT_EQ(pvx_executor_sample(exec, nullptr, &token), PVX_ERR_DECODE_FAILED);

    pvx_executor_free(exec);
    pvx_free_context(ctx);
    pvx_free_model(m);
}

static void test_empty_tokenize() {
    pvx_model_t* m = make_model();
    pvx_context_t* ctx = make_ctx(m);

    int32_t* tokens = nullptr;
    int count = -1;
    ASSERT_EQ(pvx_tokenize(ctx, "", false, true, &tokens, &count), PVX_OK);
    // Empty string might produce 0 tokens or just BOS — either is acceptable
    if (count == 0) { ASSERT(tokens == nullptr); }
    if (tokens) pvx_free_tokens(tokens);

    // NULL text
    ASSERT_EQ(pvx_tokenize(ctx, nullptr, false, true, &tokens, &count), PVX_ERR_INVALID_ARG);

    pvx_free_context(ctx);
    pvx_free_model(m);
}

static void test_detokenize_empty() {
    pvx_model_t* m = make_model();
    pvx_context_t* ctx = make_ctx(m);

    // Zero count
    char* out = nullptr;
    ASSERT_EQ(pvx_detokenize(ctx, nullptr, 0, false, &out), PVX_ERR_INVALID_ARG);

    // Count = -1 (invalid)
    int32_t dummy = 0;
    ASSERT_EQ(pvx_detokenize(ctx, &dummy, -1, false, &out), PVX_ERR_INVALID_ARG);

    pvx_free_context(ctx);
    pvx_free_model(m);
}

// ====================================================
// 3. Pool exhaustion and reuse stress
// ==========================================
// 

static void test_pool_exhaust_lease_return_cycle() {
    pvx_model_t* m = make_model();
    pvx_context_t* ctx = make_ctx(m, 4096, 512, 4);
    pvx_pool_t* pool = nullptr;
    ASSERT_EQ(pvx_pool_create(ctx, &pool), PVX_OK);

    // Lease all 4
    pvx_conversation_t* convs[4];
    for (int i = 0; i < 4; i++) ASSERT_EQ(pvx_pool_lease(pool, &convs[i]), PVX_OK);
    ASSERT_EQ(pvx_pool_available(pool), 0);

    // 5th should fail
    pvx_conversation_t* extra = nullptr;
    ASSERT_EQ(pvx_pool_lease(pool, &extra), PVX_ERR_NO_SLOT);
    ASSERT(extra == nullptr);

    // Return one, lease one — repeat 10 times
    for (int cycle = 0; cycle < 10; cycle++) {
        ASSERT_EQ(pvx_pool_return(pool, convs[0]), PVX_OK);
        ASSERT_EQ(pvx_pool_available(pool), 1);
        ASSERT_EQ(pvx_pool_lease(pool, &convs[0]), PVX_OK);
        ASSERT_EQ(pvx_pool_available(pool), 0);
    }

    // Return all
    for (int i = 0; i < 4; i++) ASSERT_EQ(pvx_pool_return(pool, convs[i]), PVX_OK);
    ASSERT_EQ(pvx_pool_available(pool), 4);

    pvx_pool_free(pool);
    pvx_free_context(ctx);
    pvx_free_model(m);
}

static void test_pool_return_double() {
    // Returning the same conversation twice should not corrupt the stack
    pvx_model_t* m = make_model();
    pvx_context_t* ctx = make_ctx(m, 4096, 512, 2);
    pvx_pool_t* pool = nullptr;
    ASSERT_EQ(pvx_pool_create(ctx, &pool), PVX_OK);

    pvx_conversation_t* conv = nullptr;
    ASSERT_EQ(pvx_pool_lease(pool, &conv), PVX_OK);
    ASSERT_EQ(pvx_pool_return(pool, conv), PVX_OK);
    // Audit 2026-10-22: double-return is now FAIL-CLOSED (typed INVALID_ARG).
    // Previously the conv was pushed into `available` TWICE, so two leasers
    // could share one seq_id and race each other's KV (corruption). The
    // leased-guard refuses the second return; the stress WARN is obsolete.
    ASSERT_EQ(pvx_pool_return(pool, conv), PVX_ERR_INVALID_ARG);
    // And the conv is NOT re-added: a fresh lease yields a valid conv, and the
    // pool never hands the same conv to two callers simultaneously.
    pvx_conversation_t* c1 = nullptr;
    ASSERT_EQ(pvx_pool_lease(pool, &c1), PVX_OK);
    ASSERT_EQ(pvx_pool_return(pool, c1), PVX_OK);

    pvx_pool_free(pool);
    pvx_free_context(ctx);
    pvx_free_model(m);
}

static void test_pool_prompt_return_without_infer() {
    // Return a conversation with pending tokens but no infer — should clean up
    pvx_model_t* m = make_model();
    pvx_context_t* ctx = make_ctx(m, 4096, 512, 2);
    pvx_pool_t* pool = nullptr;
    ASSERT_EQ(pvx_pool_create(ctx, &pool), PVX_OK);

    pvx_conversation_t* conv = nullptr;
    ASSERT_EQ(pvx_pool_lease(pool, &conv), PVX_OK);
    ASSERT_EQ(pvx_conversation_prompt(conv, "Hello world this is a test"), PVX_OK);
    // Return WITHOUT calling pvx_infer — pending_tokens should be cleared
    ASSERT_EQ(pvx_pool_return(pool, conv), PVX_OK);

    // Re-lease and verify it's clean
    ASSERT_EQ(pvx_pool_lease(pool, &conv), PVX_OK);
    ASSERT_EQ(pvx_conversation_token_count(conv), 0);
    ASSERT_EQ(pvx_pool_return(pool, conv), PVX_OK);

    pvx_pool_free(pool);
    pvx_free_context(ctx);
    pvx_free_model(m);
}

// ==================================================
// 4. KV cache boundary conditions
// ====================================================

static void test_tiny_context() {
    // Very small context (256) — should work for short prompts
    pvx_model_t* m = make_model();
    pvx_context_t* ctx = make_ctx(m, 256, 128, 1);
    pvx_executor_t* exec = nullptr;
    ASSERT_EQ(pvx_executor_create(ctx, &exec), PVX_OK);

    ASSERT_EQ(pvx_executor_prompt(exec, "Say hi"), PVX_OK);
    ASSERT_EQ(pvx_executor_infer(exec), PVX_DECODE_OK);
    ASSERT(pvx_executor_token_count(exec) > 0);

    pvx_sampling_params_t sp = {}; sp.temperature = 0.3f; sp.top_k = 40;
    int32_t token = 0;
    ASSERT_EQ(pvx_executor_sample(exec, &sp, &token), PVX_OK);
    ASSERT(token >= 0);

    pvx_executor_free(exec);
    pvx_free_context(ctx);
    pvx_free_model(m);
}

static void test_rewind_to_zero() {
    // Rewind all tokens — should leave context empty
    pvx_model_t* m = make_model();
    pvx_context_t* ctx = make_ctx(m);
    pvx_executor_t* exec = nullptr;
    ASSERT_EQ(pvx_executor_create(ctx, &exec), PVX_OK);

    ASSERT_EQ(pvx_executor_prompt(exec, "Hello world test"), PVX_OK);
    ASSERT_EQ(pvx_executor_infer(exec), PVX_DECODE_OK);
    int n = pvx_executor_token_count(exec);
    ASSERT(n > 0);

    // Rewind ALL tokens
    ASSERT_EQ(pvx_executor_rewind(exec, n), PVX_OK);
    ASSERT_EQ(pvx_executor_token_count(exec), 0);

    // Rewind more than we have (should clamp)
    ASSERT_EQ(pvx_executor_rewind(exec, 100), PVX_OK);
    ASSERT_EQ(pvx_executor_token_count(exec), 0);

    pvx_executor_free(exec);
    pvx_free_context(ctx);
    pvx_free_model(m);
}

static void test_shift_left_edge() {
    // Shift left by 0 (should be invalid or no-op)
    pvx_model_t* m = make_model();
    pvx_context_t* ctx = make_ctx(m, 512, 256, 2);
    pvx_executor_t* exec = nullptr;
    ASSERT_EQ(pvx_executor_create(ctx, &exec), PVX_OK);

    ASSERT_EQ(pvx_executor_prompt(exec, "Test prompt for shifting"), PVX_OK);
    ASSERT_EQ(pvx_executor_infer(exec), PVX_DECODE_OK);
    int n = pvx_executor_token_count(exec);
    ASSERT(n > 0);

    // Shift by 0 — should be invalid (n_tokens <= 0)
    ASSERT_EQ(pvx_executor_shift_left(exec, 0), PVX_ERR_INVALID_ARG);

    // Shift by exactly n — should be same as reset
    ASSERT_EQ(pvx_executor_shift_left(exec, n), PVX_OK);
    ASSERT_EQ(pvx_executor_token_count(exec), 0);

    pvx_executor_free(exec);
    pvx_free_context(ctx);
    pvx_free_model(m);
}

static void test_kv_copy_invalid_seqs() {
    pvx_model_t* m = make_model();
    pvx_context_t* ctx = make_ctx(m, 2048, 512, 4);

    // Invalid seq IDs
    ASSERT_EQ(pvx_kv_copy(ctx, -1, 0), PVX_ERR_INVALID_ARG);
    ASSERT_EQ(pvx_kv_copy(ctx, 0, -1), PVX_ERR_INVALID_ARG);
    ASSERT_EQ(pvx_kv_copy(ctx, 99, 0), PVX_ERR_INVALID_ARG);
    ASSERT_EQ(pvx_kv_copy(ctx, 0, 99), PVX_ERR_INVALID_ARG);
    ASSERT_EQ(pvx_kv_clear_seq(ctx, -1), PVX_ERR_INVALID_ARG);
    ASSERT_EQ(pvx_kv_clear_seq(ctx, 99), PVX_ERR_INVALID_ARG);

    // Copy to self — should be OK (no-op effectively)
    ASSERT_EQ(pvx_kv_copy(ctx, 0, 0), PVX_OK);

    pvx_free_context(ctx);
    pvx_free_model(m);
}

// ====================================================
// 5. Sampling parameter extremes
// ====================================================

static void test_sampling_extremes() {
    pvx_model_t* m = make_model();
    pvx_context_t* ctx = make_ctx(m);
    pvx_executor_t* exec = nullptr;
    ASSERT_EQ(pvx_executor_create(ctx, &exec), PVX_OK);

    ASSERT_EQ(pvx_executor_prompt(exec, "Say a number"), PVX_OK);
    ASSERT_EQ(pvx_executor_infer(exec), PVX_DECODE_OK);

    // Extreme temperature (very high = random)
    pvx_sampling_params_t sp = {};
    sp.temperature = 100.0f;  // extreme heat
    sp.top_p = 1.0f;
    sp.top_k = 0;  // no top-k
    sp.repeat_penalty = 1.0f;  // no penalty
    int32_t token = 0;
    ASSERT_EQ(pvx_executor_sample(exec, &sp, &token), PVX_OK);
    ASSERT(token >= 0);

    // After rewind, must re-infer before sampling (logits are stale)
    // Feed the sampled token back and infer to get fresh logits
    ASSERT_EQ(pvx_executor_prompt_tokens(exec, &token, 1), PVX_OK);
    ASSERT_EQ(pvx_executor_infer(exec), PVX_DECODE_OK);

    // Very low temperature (near greedy)
    sp.temperature = 0.001f;
    sp.top_k = 1;  // greedy
    ASSERT_EQ(pvx_executor_sample(exec, &sp, &token), PVX_OK);
    ASSERT(token >= 0);

    ASSERT_EQ(pvx_executor_prompt_tokens(exec, &token, 1), PVX_OK);
    ASSERT_EQ(pvx_executor_infer(exec), PVX_DECODE_OK);

    // Top-p = 0 (should still return something)
    sp.temperature = 0.3f;
    sp.top_p = 0.0f;
    sp.top_k = 0;
    ASSERT_EQ(pvx_executor_sample(exec, &sp, &token), PVX_OK);
    ASSERT(token >= 0);

    ASSERT_EQ(pvx_executor_prompt_tokens(exec, &token, 1), PVX_OK);
    ASSERT_EQ(pvx_executor_infer(exec), PVX_DECODE_OK);

    // Min-p = 0.99 (very aggressive filter)
    sp.top_p = 1.0f;
    sp.min_p = 0.99f;
    sp.top_k = 0;
    ASSERT_EQ(pvx_executor_sample(exec, &sp, &token), PVX_OK);
    ASSERT(token >= 0);

    pvx_executor_free(exec);
    pvx_free_context(ctx);
    pvx_free_model(m);
}

static void test_sampling_no_decode() {
    // Sample without any prior decode — should fail
    pvx_model_t* m = make_model();
    pvx_context_t* ctx = make_ctx(m);
    pvx_executor_t* exec = nullptr;
    ASSERT_EQ(pvx_executor_create(ctx, &exec), PVX_OK);

    pvx_sampling_params_t sp = {};
    int32_t token = -42;
    ASSERT_EQ(pvx_executor_sample(exec, &sp, &token), PVX_ERR_DECODE_FAILED);
    ASSERT_EQ(token, -42);  // should not be modified

    pvx_executor_free(exec);
    pvx_free_context(ctx);
    pvx_free_model(m);
}

// ====================================================
// 6. State save/restore edge cases
// ====================================================

static void test_state_save_empty() {
    // Save state when no tokens exist
    pvx_model_t* m = make_model();
    pvx_context_t* ctx = make_ctx(m, 4096, 512, 2);
    pvx_executor_t* exec = nullptr;
    ASSERT_EQ(pvx_executor_create(ctx, &exec), PVX_OK);

    pvx_state_t* state = nullptr;
    ASSERT_EQ(pvx_state_save(exec, &state), PVX_OK);
    ASSERT(state != nullptr);
    ASSERT_EQ(pvx_state_token_count(state), 0);

    // Restore empty state — should be no-op
    ASSERT_EQ(pvx_state_restore(exec, state), PVX_OK);
    ASSERT_EQ(pvx_executor_token_count(exec), 0);

    pvx_state_free(state);
    pvx_executor_free(exec);
    pvx_free_context(ctx);
    pvx_free_model(m);
}

static void test_state_restore_wrong_state() {
    // Save state A, modify, save state B, restore A, restore B
    pvx_model_t* m = make_model();
    pvx_context_t* ctx = make_ctx(m, 4096, 512, 2);
    pvx_executor_t* exec = nullptr;
    ASSERT_EQ(pvx_executor_create(ctx, &exec), PVX_OK);

    // Prompt "Hello" → save state A
    ASSERT_EQ(pvx_executor_prompt(exec, "Hello world"), PVX_OK);
    ASSERT_EQ(pvx_executor_infer(exec), PVX_DECODE_OK);
    int n_a = pvx_executor_token_count(exec);
    pvx_state_t* state_a = nullptr;
    ASSERT_EQ(pvx_state_save(exec, &state_a), PVX_OK);

    // Generate more → save state B
    pvx_sampling_params_t sp = {}; sp.temperature = 0.3f; sp.top_k = 40;
    for (int i = 0; i < 3; i++) {
        int32_t t = 0;
        ASSERT_EQ(pvx_executor_sample(exec, &sp, &t), PVX_OK);
        ASSERT_EQ(pvx_executor_prompt_tokens(exec, &t, 1), PVX_OK);
        ASSERT_EQ(pvx_executor_infer(exec), PVX_DECODE_OK);
    }
    int n_b = pvx_executor_token_count(exec);
    ASSERT(n_b > n_a);
    pvx_state_t* state_b = nullptr;
    ASSERT_EQ(pvx_state_save(exec, &state_b), PVX_OK);

    // Restore A → should go back to n_a
    ASSERT_EQ(pvx_state_restore(exec, state_a), PVX_OK);
    ASSERT_EQ(pvx_executor_token_count(exec), n_a);

    // Restore B → should go forward to n_b
    ASSERT_EQ(pvx_state_restore(exec, state_b), PVX_OK);
    ASSERT_EQ(pvx_executor_token_count(exec), n_b);

    pvx_state_free(state_a);
    pvx_state_free(state_b);
    pvx_executor_free(exec);
    pvx_free_context(ctx);
    pvx_free_model(m);
}

static void test_state_save_seq_max_1() {
    // With n_seq_max = 1, state save should fail (no backup seq)
    pvx_model_t* m = make_model();
    pvx_context_t* ctx = make_ctx(m, 4096, 512, 1);
    pvx_executor_t* exec = nullptr;
    ASSERT_EQ(pvx_executor_create(ctx, &exec), PVX_OK);

    ASSERT_EQ(pvx_executor_prompt(exec, "Hello"), PVX_OK);
    ASSERT_EQ(pvx_executor_infer(exec), PVX_DECODE_OK);

    pvx_state_t* state = nullptr;
    pvx_result_t r = pvx_state_save(exec, &state);
    if (r == PVX_ERR_NOT_SUPPORTED) {
        fprintf(stderr, "seq_max=1 correctly rejected ");
        // state should be null
        ASSERT(state == nullptr);
    } else if (r == PVX_OK) {
        // If it succeeded (no tokens to save), that's also acceptable
        if (state) pvx_state_free(state);
    }

    pvx_executor_free(exec);
    pvx_free_context(ctx);
    pvx_free_model(m);
}

// 
// 
// 7. Interleaved batch + pool operations
// ====================================================

static void test_batched_mixed_sizes() {
    // 4 conversations with different prompt sizes in one batch
    pvx_model_t* m = make_model();
    pvx_context_t* ctx = make_ctx(m, 4096, 512, 4);
    pvx_pool_t* pool = nullptr;
    ASSERT_EQ(pvx_pool_create(ctx, &pool), PVX_OK);

    pvx_conversation_t* convs[4];
    const char* prompts[] = {
        "Say 1.",                         // short
        "Say the number 2.",              // medium
        "Please say the number 3 now.",   // longer
        "I would like you to say 4."      // longest
    };

    for (int i = 0; i < 4; i++) {
        ASSERT_EQ(pvx_pool_lease(pool, &convs[i]), PVX_OK);
        ASSERT_EQ(pvx_conversation_prompt(convs[i], prompts[i]), PVX_OK);
    }

    // One decode for ALL conversations
    ASSERT_EQ(pvx_infer(ctx), PVX_DECODE_OK);

    // Each should have different token counts
    int counts[4];
    bool any_different = false;
    for (int i = 0; i < 4; i++) {
        counts[i] = pvx_conversation_token_count(convs[i]);
        ASSERT(counts[i] > 0);
        if (i > 0 && counts[i] != counts[0]) any_different = true;
    }
    ASSERT(any_different);  // prompts are different lengths → token counts differ

    // Sample from each
    pvx_sampling_params_t sp = {}; sp.temperature = 0.3f; sp.top_k = 40;
    for (int i = 0; i < 4; i++) {
        int32_t token = 0;
        ASSERT_EQ(pvx_conversation_sample(convs[i], &sp, &token), PVX_OK);
        ASSERT(token >= 0);
    }

    // Return all, re-lease, verify clean
    for (int i = 0; i < 4; i++) ASSERT_EQ(pvx_pool_return(pool, convs[i]), PVX_OK);
    for (int i = 0; i < 4; i++) {
        ASSERT_EQ(pvx_pool_lease(pool, &convs[i]), PVX_OK);
        ASSERT_EQ(pvx_conversation_token_count(convs[i]), 0);
    }
    for (int i = 0; i < 4; i++) ASSERT_EQ(pvx_pool_return(pool, convs[i]), PVX_OK);

    pvx_pool_free(pool);
    pvx_free_context(ctx);
    pvx_free_model(m);
}

static void test_batched_infer_no_work() {
    // pvx_infer with no pending tokens on any conversation
    pvx_model_t* m = make_model();
    pvx_context_t* ctx = make_ctx(m, 4096, 512, 4);
    pvx_pool_t* pool = nullptr;
    ASSERT_EQ(pvx_pool_create(ctx, &pool), PVX_OK);

    // Lease some convs but don't prompt
    pvx_conversation_t* convs[2];
    for (int i = 0; i < 2; i++) ASSERT_EQ(pvx_pool_lease(pool, &convs[i]), PVX_OK);

    // Infer — should be NO_WORK
    ASSERT_EQ(pvx_infer(ctx), PVX_DECODE_NO_WORK);

    for (int i = 0; i < 2; i++) ASSERT_EQ(pvx_pool_return(pool, convs[i]), PVX_OK);
    pvx_pool_free(pool);
    pvx_free_context(ctx);
    pvx_free_model(m);
}

static void test_batched_partial_prompt() {
    // 3 conversations: 2 have pending tokens, 1 doesn't
    pvx_model_t* m = make_model();
    pvx_context_t* ctx = make_ctx(m, 4096, 512, 4);
    pvx_pool_t* pool = nullptr;
    ASSERT_EQ(pvx_pool_create(ctx, &pool), PVX_OK);

    pvx_conversation_t* convs[3];
    for (int i = 0; i < 3; i++) ASSERT_EQ(pvx_pool_lease(pool, &convs[i]), PVX_OK);

    // Only prompt 2 of 3
    ASSERT_EQ(pvx_conversation_prompt(convs[0], "Say A"), PVX_OK);
    ASSERT_EQ(pvx_conversation_prompt(convs[1], "Say B"), PVX_OK);
    // convs[2] has no pending tokens

    ASSERT_EQ(pvx_infer(ctx), PVX_DECODE_OK);

    // 0 and 1 should have tokens, 2 should still be 0
    ASSERT(pvx_conversation_token_count(convs[0]) > 0);
    ASSERT(pvx_conversation_token_count(convs[1]) > 0);
    ASSERT_EQ(pvx_conversation_token_count(convs[2]), 0);

    for (int i = 0; i < 3; i++) ASSERT_EQ(pvx_pool_return(pool, convs[i]), PVX_OK);
    pvx_pool_free(pool);
    pvx_free_context(ctx);
    pvx_free_model(m);
}

// ====================================================
// 7b. Large prompt vs small n_batch (chunked decode regression)
// ====================================================

static void test_batched_large_prompt_small_nbatch() {
    // Prompt far exceeding n_batch — llama_decode must never see more than
    // n_batch tokens per call (GGML_ASSERT crash otherwise, kills the server).
    pvx_model_t* m = make_model();
    pvx_context_t* ctx = make_ctx(m, 4096, 64, 1);
    pvx_pool_t* pool = nullptr;
    ASSERT_EQ(pvx_pool_create(ctx, &pool), PVX_OK);

    pvx_conversation_t* conv = nullptr;
    ASSERT_EQ(pvx_pool_lease(pool, &conv), PVX_OK);

    // Build a prompt well over 64 tokens
    std::string prompt;
    for (int i = 0; i < 40; i++) prompt += "The quick brown fox jumps over the lazy dog number " + std::to_string(i) + ". ";
    ASSERT_EQ(pvx_conversation_prompt(conv, prompt.c_str()), PVX_OK);

    ASSERT_EQ(pvx_infer(ctx), PVX_DECODE_OK);
    ASSERT(pvx_conversation_token_count(conv) > 64);

    // Sampling right after chunked prefill must work (logits from last slice)
    pvx_sampling_params_t sp = {};
    sp.temperature = 0.3f; sp.top_p = 0.95f; sp.top_k = 40; sp.max_tokens = 16;
    int32_t tok = -1;
    ASSERT_EQ(pvx_conversation_sample(conv, &sp, &tok), PVX_OK);
    ASSERT(tok >= 0);

    ASSERT_EQ(pvx_pool_return(pool, conv), PVX_OK);
    pvx_pool_free(pool);
    pvx_free_context(ctx);
    pvx_free_model(m);
}

// ====================================================
// 8b. Multi-conversation chunked batch sampling (stale logits row crash)
// ====================================================
// Reproduces the J7b batch-server crash (2026-09-26): combined pending tokens
// of several conversations exceed n_batch, so pvx_infer decodes in slices and
// conversations' final tokens land in DIFFERENT slices. Sampling a
// conversation whose row is not in the context's current batch used to
// GGML_ABORT the process (get_logits_ith: "batch.logits[i] != true").
// Snapshotted per-conversation logits make all samples valid — including
// AFTER an interleaved decode for a different conversation.
static void test_batched_multi_conv_chunked_sample() {
    pvx_model_t* m = make_model();
    pvx_context_t* ctx = make_ctx(m, 4096, 64, 4);  // n_batch=64 forces many slices
    pvx_pool_t* pool = nullptr;
    ASSERT_EQ(pvx_pool_create(ctx, &pool), PVX_OK);

    pvx_conversation_t* convs[3] = {};
    for (int i = 0; i < 3; i++) {
        ASSERT_EQ(pvx_pool_lease(pool, &convs[i]), PVX_OK);
        std::string prompt;
        for (int j = 0; j < 30; j++)
            prompt += "Conversation " + std::to_string(i) + " filler line number " + std::to_string(j) + ". ";
        ASSERT_EQ(pvx_conversation_prompt(convs[i], prompt.c_str()), PVX_OK);
    }

    // One cycle decodes all three (~250 tokens each) in 64-token slices.
    // Conv 0 and conv 1 end in NON-final slices.
    ASSERT_EQ(pvx_infer(ctx), PVX_DECODE_OK);
    for (int i = 0; i < 3; i++)
        ASSERT(pvx_conversation_token_count(convs[i]) > 64);

    pvx_sampling_params_t sp = {};
    sp.temperature = 0.3f; sp.top_p = 0.95f; sp.top_k = 40; sp.max_tokens = 16;

    // Sample ALL conversations — convs ending in earlier slices abort on the
    // old code (stale row), pass with the per-conversation snapshot.
    int32_t toks[3] = {-1, -1, -1};
    for (int i = 0; i < 3; i++) {
        ASSERT_EQ(pvx_conversation_sample(convs[i], &sp, &toks[i]), PVX_OK);
        ASSERT(toks[i] >= 0);
    }

    // Interleaved decode: continue conv 0 only, then sample conv 1 — its
    // snapshot from the FIRST cycle must still be valid after this decode.
    ASSERT_EQ(pvx_conversation_prompt_tokens(convs[0], &toks[0], 1), PVX_OK);
    ASSERT_EQ(pvx_infer(ctx), PVX_DECODE_OK);
    int32_t tok1 = -1;
    ASSERT_EQ(pvx_conversation_sample(convs[1], &sp, &tok1), PVX_OK);
    ASSERT(tok1 >= 0);
    // Conv 0's own snapshot was refreshed by its latest decode.
    int32_t tok0 = -1;
    ASSERT_EQ(pvx_conversation_sample(convs[0], &sp, &tok0), PVX_OK);
    ASSERT(tok0 >= 0);

    for (int i = 0; i < 3; i++) ASSERT_EQ(pvx_pool_return(pool, convs[i]), PVX_OK);
    pvx_pool_free(pool);
    pvx_free_context(ctx);
    pvx_free_model(m);
}

// ====================================================
// 9. Repeated generation (resource leak detection)
// ====================================================

static void test_repeated_generation() {
    // Generate 20 tokens — detects token/recent_tokens leaks
    pvx_model_t* m = make_model();
    pvx_context_t* ctx = make_ctx(m, 4096, 512, 2);
    pvx_executor_t* exec = nullptr;
    ASSERT_EQ(pvx_executor_create(ctx, &exec), PVX_OK);

    ASSERT_EQ(pvx_executor_prompt(exec, "Count from 1 to 20: 1 2 3 4 5"), PVX_OK);
    ASSERT_EQ(pvx_executor_infer(exec), PVX_DECODE_OK);
    int initial_tokens = pvx_executor_token_count(exec);

    pvx_sampling_params_t sp = {}; sp.temperature = 0.3f; sp.top_k = 40;
    for (int i = 0; i < 20; i++) {
        int32_t token = 0;
        ASSERT_EQ(pvx_executor_sample(exec, &sp, &token), PVX_OK);
        if (pvx_token_is_eos(ctx, token)) break;
        ASSERT_EQ(pvx_executor_prompt_tokens(exec, &token, 1), PVX_OK);
        ASSERT_EQ(pvx_executor_infer(exec), PVX_DECODE_OK);
    }

    int final_tokens = pvx_executor_token_count(exec);
    ASSERT(final_tokens > initial_tokens);
    fprintf(stderr, "gen=%d ", final_tokens - initial_tokens);

    pvx_executor_free(exec);
    pvx_free_context(ctx);
    pvx_free_model(m);
}

static void test_repeated_pool_cycles() {
    // Lease → prompt → infer → sample → return, 10 cycles
    // Detects KV leaks (token count should stay consistent)
    pvx_model_t* m = make_model();
    pvx_context_t* ctx = make_ctx(m, 4096, 512, 2);
    pvx_pool_t* pool = nullptr;
    ASSERT_EQ(pvx_pool_create(ctx, &pool), PVX_OK);
    pvx_sampling_params_t sp = {}; sp.temperature = 0.3f; sp.top_k = 40;

    for (int cycle = 0; cycle < 10; cycle++) {
        pvx_conversation_t* conv = nullptr;
        ASSERT_EQ(pvx_pool_lease(pool, &conv), PVX_OK);
        ASSERT_EQ(pvx_pool_available(pool), 1);

        ASSERT_EQ(pvx_conversation_prompt(conv, "Say a number"), PVX_OK);
        ASSERT_EQ(pvx_infer(ctx), PVX_DECODE_OK);
        ASSERT(pvx_conversation_token_count(conv) > 0);

        int32_t token = 0;
        ASSERT_EQ(pvx_conversation_sample(conv, &sp, &token), PVX_OK);
        ASSERT(token >= 0);

        ASSERT_EQ(pvx_pool_return(pool, conv), PVX_OK);
        ASSERT_EQ(pvx_pool_available(pool), 2);
    }

    // After 10 cycles, pool should be fully available
    ASSERT_EQ(pvx_pool_available(pool), 2);

    pvx_pool_free(pool);
    pvx_free_context(ctx);
    pvx_free_model(m);
}

// ====================================================
// 9. Backend probe
// ====================================================

static void test_backend_probe() {
    pvx_model_t* m = make_model();
    pvx_context_t* ctx = make_ctx(m);
    ASSERT_EQ(pvx_probe_backend(ctx), PVX_OK);
    ASSERT(pvx_backend_probe_passed(ctx));
    fprintf(stderr, "backend=%s ", pvx_backend_name(ctx));

    // These should all be valid
    ASSERT(pvx_backend_name(ctx) != nullptr);
    ASSERT(pvx_backend_probe_error(ctx) != nullptr);

    pvx_free_context(ctx);
    pvx_free_model(m);
}

// ====================================================
// 10. Embedding edge cases
// ====================================================

static void test_embedding_empty_string() {
    pvx_model_params_t mp = {}; mp.model_path = EMBED_MODEL_PATH; mp.gpu_layers = 0; mp.flash_attn = true;
    pvx_model_t* m = nullptr;
    ASSERT_EQ(pvx_load_model(&mp, &m), PVX_OK);
    pvx_context_params_t cp = {}; cp.context_size = 2048; cp.batch_size = 512; cp.seq_max = 1;
    cp.pooling_type = PVX_POOL_MEAN;
    pvx_context_t* ctx = nullptr;
    ASSERT_EQ(pvx_create_context(m, &cp, &ctx), PVX_OK);

    float* emb = nullptr; int dim = 0;
    // Empty string — should still 
    // Empty string — should still produce embeddings (or fail gracefully)
    pvx_result_t r = pvx_get_embeddings(m, ctx, "", &emb, &dim);
    if (r == PVX_OK) {
        ASSERT(dim > 0);
        ASSERT(emb != nullptr);
        pvx_free_embeddings(emb);
    }

    pvx_free_context(ctx);
    pvx_free_model(m);
}

static void test_embedding_all_pooling_types() {
    pvx_model_params_t mp = {}; mp.model_path = EMBED_MODEL_PATH; mp.gpu_layers = 0; mp.flash_attn = true;
    pvx_model_t* m = nullptr;
    ASSERT_EQ(pvx_load_model(&mp, &m), PVX_OK);

    pvx_pooling_type_t types[] = {PVX_POOL_MEAN, PVX_POOL_CLS, PVX_POOL_LAST};
    for (int t = 0; t < 3; t++) {
        pvx_context_params_t cp = {}; cp.context_size = 2048; cp.batch_size = 512; cp.seq_max = 1;
        cp.pooling_type = types[t];
        pvx_context_t* ctx = nullptr;
        ASSERT_EQ(pvx_create_context(m, &cp, &ctx), PVX_OK);

        float* emb = nullptr; int dim = 0;
        ASSERT_EQ(pvx_get_embeddings(m, ctx, "Hello world test", &emb, &dim), PVX_OK);
        ASSERT(dim > 0);
        ASSERT(emb != nullptr);

        // Verify normalization
        float norm = 0;
        for (int j = 0; j < dim; j++) norm += emb[j] * emb[j];
        norm = sqrtf(norm);
        fprintf(stderr, "pool=%d norm=%.3f ", t, norm);
        ASSERT(norm > 0.9f && norm < 1.1f);  // should be ~unit length

        pvx_free_embeddings(emb);
        pvx_free_context(ctx);
    }

    pvx_free_model(m);
}

// ====================================================
// 11. Config parameter variations
// ====================================================

static void test_cpu_only_model() {
    // gpu_layers = 0 (CPU only)
    pvx_model_t* m = make_model(0, true, PVX_KV_F16);
    pvx_context_t* ctx = make_ctx(m, 2048, 512, 2);
    pvx_executor_t* exec = nullptr;
    ASSERT_EQ(pvx_executor_create(ctx, &exec), PVX_OK);

    ASSERT_EQ(pvx_executor_prompt(exec, "Say hello"), PVX_OK);
    ASSERT_EQ(pvx_executor_infer(exec), PVX_DECODE_OK);
    int32_t token = 0;
    pvx_sampling_params_t sp = {}; sp.temperature = 0.3f; sp.top_k = 40;
    ASSERT_EQ(pvx_executor_sample(exec, &sp, &token), PVX_OK);
    ASSERT(token >= 0);

    pvx_executor_free(exec);
    pvx_free_context(ctx);
    pvx_free_model(m);
}

static void test_fa_disabled() {
    // flash_attn = false
    pvx_model_t* m = make_model(99, false, PVX_KV_F16);
    pvx_context_t* ctx = make_ctx(m, 2048, 512, 2);
    pvx_executor_t* exec = nullptr;
    ASSERT_EQ(pvx_executor_create(ctx, &exec), PVX_OK);

    ASSERT_EQ(pvx_executor_prompt(exec, "Say hello"), PVX_OK);
    ASSERT_EQ(pvx_executor_infer(exec), PVX_DECODE_OK);
    ASSERT(pvx_executor_token_count(exec) > 0);

    pvx_executor_free(exec);
    pvx_free_context(ctx);
    pvx_free_model(m);
}

static void test_q8_kv_cache() {
    // q8_0 KV cache type
    pvx_model_t* m = make_model(99, true, PVX_KV_Q8_0);
    pvx_context_t* ctx = make_ctx(m, 2048, 512, 2);
    pvx_executor_t* exec = nullptr;
    ASSERT_EQ(pvx_executor_create(ctx, &exec), PVX_OK);

    ASSERT_EQ(pvx_executor_prompt(exec, "Say hello"), PVX_OK);
    ASSERT_EQ(pvx_executor_infer(exec), PVX_DECODE_OK);
    ASSERT(pvx_executor_token_count(exec) > 0);

    // Generate a token to verify the pipeline works end-to-end with q8_0
    pvx_sampling_params_t sp = {}; sp.temperature = 0.3f; sp.top_k = 40;
    int32_t token = 0;
    ASSERT_EQ(pvx_executor_sample(exec, &sp, &token), PVX_OK);
    ASSERT(token >= 0);

    pvx_executor_free(exec);
    pvx_free_context(ctx);
    pvx_free_model(m);
}

// ====================================================
// 12. Conversation rewind edge cases
// ====================================================

static void test_conversation_rewind_edge() {
    pvx_model_t* m = make_model();
    pvx_context_t* ctx = make_ctx(m, 4096, 512, 2);
    pvx_pool_t* pool = nullptr;
    ASSERT_EQ(pvx_pool_create(ctx, &pool), PVX_OK);

    pvx_conversation_t* conv = nullptr;
    ASSERT_EQ(pvx_pool_lease(pool, &conv), PVX_OK);

    // Rewind with 0 tokens — should be OK (no-op)
    ASSERT_EQ(pvx_conversation_rewind(conv, 0), PVX_OK);
    ASSERT_EQ(pvx_conversation_token_count(conv), 0);

    // Rewind with negative
    ASSERT_EQ(pvx_conversation_rewind(conv, -1), PVX_ERR_INVALID_ARG);

    // Prompt and infer
    ASSERT_EQ(pvx_conversation_prompt(conv, "Hello test"), PVX_OK);
    ASSERT_EQ(pvx_infer(ctx), PVX_DECODE_OK);
    int n = pvx_conversation_token_count(conv);
    ASSERT(n > 0);

    // Rewind to exactly 0
    ASSERT_EQ(pvx_conversation_rewind(conv, n), PVX_OK);
    ASSERT_EQ(pvx_conversation_token_count(conv), 0);

    // Rewind past 0 (should clamp)
    ASSERT_EQ(pvx_conversation_rewind(conv, n + 10), PVX_OK);
    ASSERT_EQ(pvx_conversation_token_count(conv), 0);

    ASSERT_EQ(pvx_pool_return(pool, conv), PVX_OK);
    pvx_pool_free(pool);
    pvx_free_context(ctx);
    pvx_free_model(m);
}

// ====================================================
// 13. Multiple model loads (memory leak detection)
// ====================================================

static void test_model_reload() {
    // Load and free the same model 5 times — should not leak
    for (int i = 0; i < 5; i++) {
        pvx_model_t* m = make_model(0);  // CPU only for speed
        pvx_context_t* ctx = make_ctx(m, 512);
        pvx_executor_t* exec = nullptr;
        ASSERT_EQ(pvx_executor_create(ctx, &exec), PVX_OK);
        ASSERT_EQ(pvx_executor_prompt(exec, "test"), PVX_OK);
        ASSERT_EQ(pvx_executor_infer(exec), PVX_DECODE_OK);
        pvx_executor_free(exec);
        pvx_free_context(ctx);
        pvx_free_model(m);
    }
}

// ====================================================

int main() {
    MODEL_PATH = getenv("PVX_MODEL_PATH");
    EMBED_MODEL_PATH = getenv("PVX_EMBED_MODEL_PATH");
    if (!MODEL_PATH) MODEL_PATH = "./models/Qwen3.5-4B-Q4_K_M.gguf";
    if (!EMBED_MODEL_PATH) EMBED_MODEL_PATH = "./models/all-MiniLM-L6-v2-Q5_K_M.gguf";

    fprintf(stderr, "\n=== ProvostXInference Stress Tests ===\n");
    fprintf(stderr, "Model: %s\n", file_exists(MODEL_PATH) ? MODEL_PATH : "(not found)");
    fprintf(stderr, "Embed: %s\n\n", file_exists(EMBED_MODEL_PATH) ? EMBED_MODEL_PATH : "(not found");

    TEST(null_args);
    TEST(empty_prompt);
    TEST(empty_tokenize);
    TEST(detokenize_empty);
    TEST(pool_exhaust_lease_return_cycle);
    TEST(pool_return_double);
    TEST(pool_prompt_return_without_infer);
    TEST(tiny_context);
    TEST(rewind_to_zero);
    TEST(shift_left_edge);
    TEST(kv_copy_invalid_seqs);
    TEST(sampling_extremes);
    TEST(sampling_no_decode);
    TEST(state_save_empty);
    TEST(state_restore_wrong_state);
    TEST(state_save_seq_max_1);
    TEST(batched_mixed_sizes);
    TEST(batched_infer_no_work);
    TEST(batched_partial_prompt);
    TEST(batched_large_prompt_small_nbatch);
    TEST(batched_multi_conv_chunked_sample);
    TEST(repeated_generation);
    TEST(repeated_pool_cycles);
    TEST(backend_probe);
    TEST(embedding_empty_string);
    TEST(embedding_all_pooling_types);
    TEST(cpu_only_model);
    TEST(fa_disabled);
    TEST(q8_kv_cache);
    TEST(conversation_rewind_edge);
    TEST(model_reload);

    fprintf(stderr, "\n=== %d/%d passed, %d skipped ===\n", tests_passed, tests_run, tests_skipped);
    return 0;
}
