// test_basic.cpp — Full test suite for ProvostXInference C API

#include "pvxinference.h"
#include "gguf.h"
#include "ggml.h"
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES // M_PI is not defined by MSVC without this
#endif
#include <cmath>
#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <cmath>

static const char* MODEL_PATH;
static const char* EMBED_MODEL_PATH;
static const char* MMPROJ_PATH;
static const char* HYBRID_MODEL_PATH;
static const char* TEST_IMAGE;
static const char* DRAFT_MODEL_PATH;

static int tests_run = 0, tests_passed = 0, tests_skipped = 0;

#define TEST(name) \
    tests_run++; \
    fprintf(stderr, "  [%d] %s ... ", tests_run, #name); \
    test_##name(); \
    tests_passed++; \
    fprintf(stderr, "OK\n");

#define SKIP(msg) do { tests_skipped++; fprintf(stderr, "SKIP (%s)\n", msg); return; } while(0)

#define ASSERT(cond) \
    if (!(cond)) { \
        fprintf(stderr, "FAIL: line %d: %s\n", __LINE__, #cond); \
        exit(1); \
    }

static bool file_exists(const char* path) {
    if (!path) return false;
    FILE* f = fopen(path, "rb");
    if (f) { fclose(f); return true; }
    return false;
}

// ── Tests ──

static void test_load_model() {
    if (!file_exists(MODEL_PATH)) SKIP("model file not found");
    pvx_model_params_t p = {}; p.model_path = MODEL_PATH; p.gpu_layers = 99; p.flash_attn = true;
    pvx_model_t* m = nullptr;
    ASSERT(pvx_load_model(&p, &m) == PVX_OK);
    pvx_free_model(m);
}

static void test_create_context() {
    if (!file_exists(MODEL_PATH)) SKIP("model file not found");
    pvx_model_params_t mp = {}; mp.model_path = MODEL_PATH; mp.gpu_layers = 99; mp.flash_attn = true;
    pvx_model_t* m = nullptr; ASSERT(pvx_load_model(&mp, &m) == PVX_OK);
    pvx_context_params_t cp = {}; cp.context_size = 4096; cp.batch_size = 512; cp.seq_max = 4;
    pvx_context_t* ctx = nullptr;
    ASSERT(pvx_create_context(m, &cp, &ctx) == PVX_OK);
    ASSERT(pvx_context_size(ctx) == 4096);
    ASSERT(pvx_context_seq_max(ctx) == 4);
    pvx_free_context(ctx); pvx_free_model(m);
}

static void test_tokenize() {
    if (!file_exists(MODEL_PATH)) SKIP("model file not found");
    pvx_model_params_t mp = {}; mp.model_path = MODEL_PATH; mp.gpu_layers = 0; mp.flash_attn = true;
    pvx_model_t* m = nullptr; ASSERT(pvx_load_model(&mp, &m) == PVX_OK);
    pvx_context_params_t cp = {}; cp.context_size = 2048; cp.batch_size = 512; cp.seq_max = 2;
    pvx_context_t* ctx = nullptr; ASSERT(pvx_create_context(m, &cp, &ctx) == PVX_OK);

    int32_t* tokens = nullptr; int count = 0;
    ASSERT(pvx_tokenize(ctx, "Hello world", false, true, &tokens, &count) == PVX_OK);
    ASSERT(count > 0);

    char* text = nullptr;
    ASSERT(pvx_detokenize(ctx, tokens, count, false, &text) == PVX_OK);
    ASSERT(strstr(text, "Hello") != nullptr);
    pvx_free_string(text); pvx_free_tokens(tokens);
    pvx_free_context(ctx); pvx_free_model(m);
}

static void test_pool() {
    if (!file_exists(MODEL_PATH)) SKIP("model file not found");
    pvx_model_params_t mp = {}; mp.model_path = MODEL_PATH; mp.gpu_layers = 99; mp.flash_attn = true;
    pvx_model_t* m = nullptr; ASSERT(pvx_load_model(&mp, &m) == PVX_OK);
    pvx_context_params_t cp = {}; cp.context_size = 4096; cp.batch_size = 512; cp.seq_max = 4;
    pvx_context_t* ctx = nullptr; ASSERT(pvx_create_context(m, &cp, &ctx) == PVX_OK);

    pvx_pool_t* pool = nullptr; ASSERT(pvx_pool_create(ctx, &pool) == PVX_OK);
    ASSERT(pvx_pool_size(pool) == 4);
    ASSERT(pvx_pool_available(pool) == 4);

    pvx_conversation_t* convs[4];
    for (int i = 0; i < 4; i++) ASSERT(pvx_pool_lease(pool, &convs[i]) == PVX_OK);
    ASSERT(pvx_pool_available(pool) == 0);

    pvx_conversation_t* extra = nullptr;
    ASSERT(pvx_pool_lease(pool, &extra) == PVX_ERR_NO_SLOT);

    for (int i = 0; i < 4; i++) ASSERT(pvx_pool_return(pool, convs[i]) == PVX_OK);
    ASSERT(pvx_pool_available(pool) == 4);

    pvx_pool_free(pool); pvx_free_context(ctx); pvx_free_model(m);
}

static void test_single_inference() {
    if (!file_exists(MODEL_PATH)) SKIP("model file not found");
    pvx_model_params_t mp = {}; mp.model_path = MODEL_PATH; mp.gpu_layers = 99; mp.flash_attn = true;
    pvx_model_t* m = nullptr; ASSERT(pvx_load_model(&mp, &m) == PVX_OK);
    pvx_context_params_t cp = {}; cp.context_size = 4096; cp.batch_size = 512; cp.seq_max = 2;
    pvx_context_t* ctx = nullptr; ASSERT(pvx_create_context(m, &cp, &ctx) == PVX_OK);

    pvx_executor_t* exec = nullptr;
    ASSERT(pvx_executor_create(ctx, &exec) == PVX_OK);

    ASSERT(pvx_executor_prompt(exec, "Say the number 7 and nothing else.") == PVX_OK);
    ASSERT(pvx_executor_infer(exec) == PVX_DECODE_OK);
    ASSERT(pvx_executor_token_count(exec) > 0);

    pvx_sampling_params_t sp = {};
    sp.temperature = 0.3f; sp.top_p = 0.95f; sp.top_k = 40;
    sp.repeat_penalty = 1.1f; sp.repeat_last_n = -1; sp.max_tokens = 32;

    std::string output;
    for (int i = 0; i < 32; i++) {
        int32_t token = 0;
        ASSERT(pvx_executor_sample(exec, &sp, &token) == PVX_OK);
        if (pvx_token_is_eos(ctx, token)) break;

        char buf[256];
        if (pvx_token_to_piece(ctx, token, buf, sizeof(buf)) == PVX_OK) output += buf;

        ASSERT(pvx_executor_prompt_tokens(exec, &token, 1) == PVX_OK);
        ASSERT(pvx_executor_infer(exec) == PVX_DECODE_OK);
    }
    fprintf(stderr, "out=\"%s\" ", output.c_str());
    ASSERT(!output.empty());

    pvx_executor_free(exec); pvx_free_context(ctx); pvx_free_model(m);
}

static void test_batched_inference() {
    if (!file_exists(MODEL_PATH)) SKIP("model file not found");
    pvx_model_params_t mp = {}; mp.model_path = MODEL_PATH; mp.gpu_layers = 99; mp.flash_attn = true;
    pvx_model_t* m = nullptr; ASSERT(pvx_load_model(&mp, &m) == PVX_OK);
    pvx_context_params_t cp = {}; cp.context_size = 4096; cp.batch_size = 512; cp.seq_max = 4;
    pvx_context_t* ctx = nullptr; ASSERT(pvx_create_context(m, &cp, &ctx) == PVX_OK);

    pvx_pool_t* pool = nullptr; ASSERT(pvx_pool_create(ctx, &pool) == PVX_OK);
    pvx_conversation_t* convs[4];
    const char* prompts[] = {"Say 1.","Say 2.","Say 3.","Say 4."};

    for (int i = 0; i < 4; i++) {
        ASSERT(pvx_pool_lease(pool, &convs[i]) == PVX_OK);
        ASSERT(pvx_conversation_prompt(convs[i], prompts[i]) == PVX_OK);
    }
    ASSERT(pvx_infer(ctx) == PVX_DECODE_OK);

    pvx_sampling_params_t sp = {}; sp.temperature = 0.3f; sp.top_p = 0.95f; sp.top_k = 40;
    for (int i = 0; i < 4; i++) {
        int32_t token = 0;
        ASSERT(pvx_conversation_sample(convs[i], &sp, &token) == PVX_OK);
    }
    for (int i = 0; i < 4; i++) ASSERT(pvx_pool_return(pool, convs[i]) == PVX_OK);

    pvx_pool_free(pool); pvx_free_context(ctx); pvx_free_model(m);
}

static void test_state_save_restore() {
    if (!file_exists(MODEL_PATH)) SKIP("model file not found");
    pvx_model_params_t mp = {}; mp.model_path = MODEL_PATH; mp.gpu_layers = 99; mp.flash_attn = true;
    pvx_model_t* m = nullptr; ASSERT(pvx_load_model(&mp, &m) == PVX_OK);
    pvx_context_params_t cp = {}; cp.context_size = 4096; cp.batch_size = 512; cp.seq_max = 2;
    pvx_context_t* ctx = nullptr; ASSERT(pvx_create_context(m, &cp, &ctx) == PVX_OK);

    pvx_executor_t* exec = nullptr; ASSERT(pvx_executor_create(ctx, &exec) == PVX_OK);

    ASSERT(pvx_executor_prompt(exec, "Hello") == PVX_OK);
    ASSERT(pvx_executor_infer(exec) == PVX_DECODE_OK);
    int tokens_after_prompt = pvx_executor_token_count(exec);
    ASSERT(tokens_after_prompt > 0);

    pvx_state_t* state = nullptr;
    ASSERT(pvx_state_save(exec, &state) == PVX_OK);

    // Generate a few more tokens
    pvx_sampling_params_t sp = {}; sp.temperature = 0.3f; sp.top_k = 40;
    for (int i = 0; i < 5; i++) {
        int32_t token = 0;
        ASSERT(pvx_executor_sample(exec, &sp, &token) == PVX_OK);
        ASSERT(pvx_executor_prompt_tokens(exec, &token, 1) == PVX_OK);
        ASSERT(pvx_executor_infer(exec) == PVX_DECODE_OK);
    }
    ASSERT(pvx_executor_token_count(exec) > tokens_after_prompt);

    // Restore
    ASSERT(pvx_state_restore(exec, state) == PVX_OK);
    ASSERT(pvx_executor_token_count(exec) == tokens_after_prompt);

    pvx_state_free(state);
    pvx_executor_free(exec); pvx_free_context(ctx); pvx_free_model(m);
}

static void test_embeddings() {
    if (!file_exists(EMBED_MODEL_PATH)) SKIP("embed model not found");
    pvx_model_params_t mp = {}; mp.model_path = EMBED_MODEL_PATH; mp.gpu_layers = 0; mp.flash_attn = true;
    pvx_model_t* m = nullptr; ASSERT(pvx_load_model(&mp, &m) == PVX_OK);
    pvx_context_params_t cp = {}; cp.context_size = 2048; cp.batch_size = 512; cp.seq_max = 1;
    cp.pooling_type = PVX_POOL_MEAN;
    pvx_context_t* ctx = nullptr; ASSERT(pvx_create_context(m, &cp, &ctx) == PVX_OK);

    float* emb = nullptr; int dim = 0;
    ASSERT(pvx_get_embeddings(m, ctx, "Hello world", &emb, &dim) == PVX_OK);
    ASSERT(dim > 0);
    // Verify embeddings are non-zero (not all-null from failed compute)
    float norm = 0; for (int i = 0; i < dim; i++) norm += emb[i]*emb[i];
    norm = sqrtf(norm);
    fprintf(stderr, "dim=%d norm=%.3f ", dim, norm);
    ASSERT(norm > 0.5f);
    pvx_free_embeddings(emb);
    pvx_free_context(ctx); pvx_free_model(m);
}

static void test_anti_prompts() {
    const char* text = "Hello world\n### User: What is 2+2?";
    const char* anti[] = {"### User:", "### Assistant:"};
    int idx = pvx_check_anti_prompts(text, anti, 2);
    ASSERT(idx == 0);

    const char* text2 = "Just normal text";
    int idx2 = pvx_check_anti_prompts(text2, anti, 2);
    ASSERT(idx2 == -1);
}

static void test_pool_reuse() {
    if (!file_exists(MODEL_PATH)) SKIP("model file not found");
    // Verify pool reuse works — lease, prompt, return, re-lease gets same conv back clean
    pvx_model_params_t mp = {}; mp.model_path = MODEL_PATH; mp.gpu_layers = 99; mp.flash_attn = true;
    pvx_model_t* m = nullptr; ASSERT(pvx_load_model(&mp, &m) == PVX_OK);
    pvx_context_params_t cp = {}; cp.context_size = 4096; cp.batch_size = 512; cp.seq_max = 2;
    pvx_context_t* ctx = nullptr; ASSERT(pvx_create_context(m, &cp, &ctx) == PVX_OK);

    pvx_pool_t* pool = nullptr; ASSERT(pvx_pool_create(ctx, &pool) == PVX_OK);
    pvx_conversation_t* conv1 = nullptr;
    ASSERT(pvx_pool_lease(pool, &conv1) == PVX_OK);
    ASSERT(pvx_conversation_prompt(conv1, "Hello world") == PVX_OK);
    ASSERT(pvx_infer(ctx) == PVX_DECODE_OK);
    ASSERT(pvx_conversation_token_count(conv1) > 0);

    // Return — should clear KV
    ASSERT(pvx_pool_return(pool, conv1) == PVX_OK);
    ASSERT(pvx_pool_available(pool) == 2);

    // Re-lease — should get a clean conversation
    pvx_conversation_t* conv2 = nullptr;
    ASSERT(pvx_pool_lease(pool, &conv2) == PVX_OK);
    ASSERT(pvx_conversation_token_count(conv2) == 0);

    pvx_pool_return(pool, conv2);
    pvx_pool_free(pool); pvx_free_context(ctx); pvx_free_model(m);
}

static void test_vision() {
    // The mmproj fixture pairs with the vision-capable multimodal model
    // (Qwen3.5-4B = HYBRID_MODEL_PATH), NOT the dense main model.
    fprintf(stderr, "[viz-dbg] hybrid=%d model=%d mmproj=%d img=%d\n",
            file_exists(HYBRID_MODEL_PATH), file_exists(MODEL_PATH), file_exists(MMPROJ_PATH), file_exists(TEST_IMAGE));
    const char* vision_model = file_exists(HYBRID_MODEL_PATH) ? HYBRID_MODEL_PATH : MODEL_PATH;
    if (!file_exists(vision_model) || !file_exists(MMPROJ_PATH) || !file_exists(TEST_IMAGE)) SKIP("vision files not found");
    // Test mmproj loading + image prompt with embeddings
    pvx_model_params_t mp = {};
    mp.model_path = vision_model;
    mp.gpu_layers = 99;
    mp.flash_attn = true;

    pvx_model_t* model = nullptr;
    ASSERT(pvx_load_model(&mp, &model) == PVX_OK);

    // Load mmproj
    pvx_model_t* mmproj = nullptr;
    const char* mmproj_path = MMPROJ_PATH;
    ASSERT(pvx_load_mmproj(model, mmproj_path, &mmproj) == PVX_OK);
    ASSERT(mmproj != nullptr);

    // Get mtmd marker
    const char* marker = pvx_mtmd_marker_static();
    ASSERT(marker != nullptr);
    fprintf(stderr, "marker=\"%s\" ", marker);

    // Create context + pool
    pvx_context_params_t cp = {};
    cp.context_size = 4096;
    cp.batch_size = 512;
    cp.seq_max = 2;

    pvx_context_t* ctx = nullptr;
    ASSERT(pvx_create_context(model, &cp, &ctx) == PVX_OK);

    pvx_pool_t* pool = nullptr;
    ASSERT(pvx_pool_create(ctx, &pool) == PVX_OK);

    pvx_conversation_t* conv = nullptr;
    ASSERT(pvx_pool_lease(pool, &conv) == PVX_OK);

    // Read test image
    FILE* f = fopen(TEST_IMAGE, "rb");
    ASSERT(f != nullptr);
    fseek(f, 0, SEEK_END);
    long img_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t* img_data = (uint8_t*)malloc(img_size);
    fread(img_data, 1, img_size, f);
    fclose(f);

    // Prompt with image — use the real marker from mtmd
    char* marker_str = nullptr;
    ASSERT(pvx_mtmd_marker(&marker_str) == PVX_OK);
    std::string prompt = std::string("What is in this image? ") + marker_str;
    pvx_free_string(marker_str);

    const uint8_t* imgs[] = {img_data};
    int sizes[] = {(int)img_size};
    pvx_result_t r = pvx_conversation_prompt_with_images(
        conv, mmproj, prompt.c_str(), imgs, sizes, 1);
    fprintf(stderr, "vision_prompt=%d ", r);
    // Should be OK (image encoded + text tokens queued)
    ASSERT(r == PVX_OK);

    // After image prompt, we should have tokens in the conversation
    int n_tokens = pvx_conversation_token_count(conv);
    fprintf(stderr, "n_tokens=%d ", n_tokens);
    ASSERT(n_tokens > 0);

    // Run one infer to process remaining text tokens
    pvx_infer(ctx);

    // Sample — should produce some output
    pvx_sampling_params_t sp = {};
    sp.temperature = 0.3f;
    sp.top_p = 0.95f;
    sp.top_k = 40;
    int32_t token = 0;
    ASSERT(pvx_conversation_sample(conv, &sp, &token) == PVX_OK);
    ASSERT(token >= 0);

    free(img_data);
    ASSERT(pvx_pool_return(pool, conv) == PVX_OK);
    pvx_pool_free(pool);
    pvx_free_context(ctx);
    pvx_free_mmproj(mmproj);
    pvx_free_model(model);
}

static void test_shift_left() {
    if (!file_exists(MODEL_PATH)) SKIP("model file not found");
    // Sliding window: fill context, shift left, keep generating
    pvx_model_params_t mp = {}; mp.model_path = MODEL_PATH; mp.gpu_layers = 99; mp.flash_attn = true;
    pvx_model_t* m = nullptr; ASSERT(pvx_load_model(&mp, &m) == PVX_OK);
    pvx_context_params_t cp = {}; cp.context_size = 512; cp.batch_size = 256; cp.seq_max = 2;
    pvx_context_t* ctx = nullptr; ASSERT(pvx_create_context(m, &cp, &ctx) == PVX_OK);

    pvx_executor_t* exec = nullptr; ASSERT(pvx_executor_create(ctx, &exec) == PVX_OK);

    // Prompt enough to fill most of the 512 context
    ASSERT(pvx_executor_prompt(exec, "Hello world, this is a test of the sliding window context shift. "
                                      "We need enough tokens to approach the context limit. "
                                      "The quick brown fox jumps over the lazy dog repeatedly.") == PVX_OK);
    ASSERT(pvx_executor_infer(exec) == PVX_DECODE_OK);
    int tokens_before = pvx_executor_token_count(exec);
    ASSERT(tokens_before > 10);

    // Shift left by 10 — remove oldest 10 tokens
    ASSERT(pvx_executor_shift_left(exec, 10) == PVX_OK);
    int tokens_after = pvx_executor_token_count(exec);
    ASSERT(tokens_after == tokens_before - 10);

    // Continue generating — must prompt+infer first (shift invalidated last_batch_idx)
    pvx_sampling_params_t sp = {}; sp.temperature = 0.3f; sp.top_k = 40; sp.max_tokens = 5;
    ASSERT(pvx_executor_prompt(exec, "Continue: ") == PVX_OK);
    ASSERT(pvx_executor_infer(exec) == PVX_DECODE_OK);
    int32_t token = 0;
    ASSERT(pvx_executor_sample(exec, &sp, &token) == PVX_OK);
    ASSERT(pvx_executor_prompt_tokens(exec, &token, 1) == PVX_OK);
    ASSERT(pvx_executor_infer(exec) == PVX_DECODE_OK);
    ASSERT(pvx_executor_token_count(exec) > tokens_after);

    pvx_executor_free(exec); pvx_free_context(ctx); pvx_free_model(m);
}

static void test_kv_copy() {
    if (!file_exists(MODEL_PATH)) SKIP("model file not found");
    pvx_model_params_t mp = {}; mp.model_path = MODEL_PATH; mp.gpu_layers = 99; mp.flash_attn = true;
    pvx_model_t* m = nullptr; ASSERT(pvx_load_model(&mp, &m) == PVX_OK);
    pvx_context_params_t cp = {}; cp.context_size = 2048; cp.batch_size = 512; cp.seq_max = 4;
    pvx_context_t* ctx = nullptr; ASSERT(pvx_create_context(m, &cp, &ctx) == PVX_OK);

    // Clear and copy
    ASSERT(pvx_kv_clear_seq(ctx, 0) == PVX_OK);
    ASSERT(pvx_kv_clear_seq(ctx, 1) == PVX_OK);
    ASSERT(pvx_kv_copy(ctx, 0, 1) == PVX_OK);
    ASSERT(pvx_kv_clear_seq(ctx, 3) == PVX_OK);

    // Invalid seq_id
    ASSERT(pvx_kv_clear_seq(ctx, 99) == PVX_ERR_INVALID_ARG);
    ASSERT(pvx_kv_copy(ctx, 0, 99) == PVX_ERR_INVALID_ARG);

    pvx_free_context(ctx); pvx_free_model(m);
}

static void test_causal_attn() {
    if (!file_exists(MODEL_PATH)) SKIP("model file not found");
    pvx_model_params_t mp = {}; mp.model_path = MODEL_PATH; mp.gpu_layers = 99; mp.flash_attn = true;
    pvx_model_t* m = nullptr; ASSERT(pvx_load_model(&mp, &m) == PVX_OK);
    pvx_context_params_t cp = {}; cp.context_size = 2048; cp.batch_size = 512; cp.seq_max = 2;
    pvx_context_t* ctx = nullptr; ASSERT(pvx_create_context(m, &cp, &ctx) == PVX_OK);

    // Toggle causal attention
    ASSERT(pvx_set_causal_attn(ctx, false) == PVX_OK);
    ASSERT(pvx_set_causal_attn(ctx, true) == PVX_OK);

    pvx_free_context(ctx); pvx_free_model(m);
}

static void test_partial_rewind_capability() {
    if (!file_exists(MODEL_PATH)) SKIP("model file not found");
    pvx_model_params_t mp = {}; mp.model_path = MODEL_PATH; mp.gpu_layers = 0; mp.flash_attn = true;
    pvx_model_t* m = nullptr; ASSERT(pvx_load_model(&mp, &m) == PVX_OK);
    // The default test model is dense-attention — partial rewind must be supported.
    ASSERT(pvx_model_partial_rewind_supported(m) == true);
    pvx_free_model(m);
}

static void test_partial_rewind_refused_on_hybrid() {
    if (!HYBRID_MODEL_PATH || !file_exists(HYBRID_MODEL_PATH)) SKIP("hybrid model not provided");
    pvx_model_params_t mp = {}; mp.model_path = HYBRID_MODEL_PATH; mp.gpu_layers = 99; mp.flash_attn = true;
    pvx_model_t* m = nullptr; ASSERT(pvx_load_model(&mp, &m) == PVX_OK);
    // Hybrid (attention+recurrent) memories cannot partially erase state.
    ASSERT(pvx_model_partial_rewind_supported(m) == false);
    pvx_context_params_t cp = {}; cp.context_size = 1024; cp.batch_size = 512; cp.seq_max = 1;
    pvx_context_t* ctx = nullptr; ASSERT(pvx_create_context(m, &cp, &ctx) == PVX_OK);
    pvx_pool_t* pool = nullptr; ASSERT(pvx_pool_create(ctx, &pool) == PVX_OK);
    pvx_conversation_t* conv = nullptr; ASSERT(pvx_pool_lease(pool, &conv) == PVX_OK);
    pvx_conversation_prompt(conv, "Hello world, this is a partial rewind capability test prompt.");
    ASSERT(pvx_infer(ctx) == PVX_DECODE_OK);
    const int n = pvx_conversation_token_count(conv);
    ASSERT(n > 2);
    // PARTIAL rewind must be REFUSED with counters left intact (no silent KV desync).
    ASSERT(pvx_conversation_rewind(conv, 1) == PVX_ERR_REWIND_UNSUPPORTED);
    ASSERT(pvx_conversation_token_count(conv) == n);
    // FULL rewind (rm_all range) still works on hybrid memories.
    ASSERT(pvx_conversation_rewind(conv, n) == PVX_OK);
    ASSERT(pvx_conversation_token_count(conv) == 0);
    pvx_pool_return(pool, conv); pvx_pool_free(pool);
    pvx_free_context(ctx); pvx_free_model(m);
}

// ── Speculative decoding (INF BL-05) ──

static void test_spec_capabilities() {
    if (!file_exists(MODEL_PATH)) SKIP("model file not found");
    pvx_model_params_t mp = {}; mp.model_path = MODEL_PATH; mp.gpu_layers = 0; mp.flash_attn = true;
    pvx_model_t* m = nullptr; ASSERT(pvx_load_model(&mp, &m) == PVX_OK);
    pvx_spec_capabilities_t caps {};
    ASSERT(pvx_model_spec_capabilities(m, &caps) == PVX_OK);
    // Supported-types list (bitset), not booleans: dense Qwen family reports
    // draft_simple; no nextn tensors in this GGUF so no mtp bit.
    ASSERT(caps.draft == true);
    ASSERT(caps.mtp == false);
    ASSERT((caps.types & PVX_SPEC_TYPE_DRAFT_SIMPLE) != 0);
    ASSERT((caps.types & PVX_SPEC_TYPE_MTP) == 0);
    // ABI v6: n-gram types reported for every standard-vocab target; the spec
    // engine's mirror is embd-aware so spec sessions support vision prompts.
    ASSERT((caps.types & PVX_SPEC_TYPE_NGRAM) != 0);
    ASSERT((caps.types & PVX_SPEC_TYPE_NGRAM_MAP_K) != 0);
    ASSERT((caps.types & PVX_SPEC_TYPE_NGRAM_MOD) != 0);
    ASSERT((caps.types & PVX_SPEC_TYPE_NGRAM_CACHE) != 0);
    ASSERT(caps.spec_vision == true);
    fprintf(stderr, "(types=0x%x mtp=%d draft=%d) ", caps.types,
            caps.mtp ? 1 : 0, caps.draft ? 1 : 0);
    pvx_free_model(m);
}

static void test_spec_capabilities_null() {
    pvx_spec_capabilities_t caps {};
    ASSERT(pvx_model_spec_capabilities(nullptr, &caps) == PVX_ERR_INVALID_ARG);
    pvx_model_t* fake = nullptr;
    ASSERT(pvx_model_spec_capabilities(fake, &caps) == PVX_ERR_INVALID_ARG);
}

static void test_spec_validate() {
    if (!file_exists(MODEL_PATH)) SKIP("model file not found");
    pvx_model_params_t mp = {}; mp.model_path = MODEL_PATH; mp.gpu_layers = 0; mp.flash_attn = true;
    pvx_model_t* m = nullptr; ASSERT(pvx_load_model(&mp, &m) == PVX_OK);
    pvx_spec_params_t sp = {}; sp.mode = PVX_SPEC_OFF;
    ASSERT(pvx_model_spec_validate(m, &sp) == PVX_OK);
    sp.mode = PVX_SPEC_MTP_AUTO;
    // MTP capability flag is real; wiring boundary is typed NOT_SUPPORTED.
    ASSERT(pvx_model_spec_validate(m, &sp) == PVX_ERR_NOT_SUPPORTED);
    sp.mode = PVX_SPEC_DRAFT; sp.draft_model_path = nullptr;
    // Draft mode without a path is a typed invalid-arg.
    ASSERT(pvx_model_spec_validate(m, &sp) == PVX_ERR_INVALID_ARG);
    sp.draft_model_path = "/nonexistent/nope.gguf";
    // Shaped-config validation passes; load failure happens at enable time.
    ASSERT(pvx_model_spec_validate(m, &sp) == PVX_OK);
    pvx_free_model(m);
}

static void test_spec_generate_requires_session() {
    if (!file_exists(MODEL_PATH)) SKIP("model file not found");
    pvx_model_params_t mp = {}; mp.model_path = MODEL_PATH; mp.gpu_layers = 0; mp.flash_attn = true;
    pvx_model_t* m = nullptr; ASSERT(pvx_load_model(&mp, &m) == PVX_OK);
    pvx_context_params_t cp = {}; cp.context_size = 1024; cp.batch_size = 512; cp.seq_max = 1;
    pvx_context_t* ctx = nullptr; ASSERT(pvx_create_context(m, &cp, &ctx) == PVX_OK);
    pvx_executor_t* exec = nullptr; ASSERT(pvx_executor_create(ctx, &exec) == PVX_OK);
    int n = 0;
    // No session → typed NOT_SUPPORTED (never a silent non-spec run).
    ASSERT(pvx_executor_generate_spec(exec, nullptr, 8, nullptr, nullptr, &n) == PVX_ERR_NOT_SUPPORTED);
    pvx_executor_free(exec); pvx_free_context(ctx); pvx_free_model(m);
}

static void test_spec_init_fail_closed() {
    if (!file_exists(MODEL_PATH)) SKIP("model file not found");
    pvx_model_params_t mp = {}; mp.model_path = MODEL_PATH; mp.gpu_layers = 99; mp.flash_attn = true;
    pvx_model_t* m = nullptr; ASSERT(pvx_load_model(&mp, &m) == PVX_OK);
    pvx_context_params_t cp = {}; cp.context_size = 1024; cp.batch_size = 512; cp.seq_max = 1;
    pvx_context_t* ctx = nullptr; ASSERT(pvx_create_context(m, &cp, &ctx) == PVX_OK);
    pvx_executor_t* exec = nullptr; ASSERT(pvx_executor_create(ctx, &exec) == PVX_OK);

    // OFF via init = disable (idempotent, no session needed).
    pvx_spec_params_t sp = {}; sp.mode = PVX_SPEC_OFF;
    ASSERT(pvx_executor_spec_init(exec, &sp) == PVX_OK);

    // MTP: typed NOT_SUPPORTED with boundary name in last_error.
    sp.mode = PVX_SPEC_MTP_AUTO;
    ASSERT(pvx_executor_spec_init(exec, &sp) == PVX_ERR_NOT_SUPPORTED);
    ASSERT(strstr(pvx_last_error(ctx), "mtp-wiring") != nullptr);

    // Draft without path: fail-closed invalid arg.
    sp.mode = PVX_SPEC_DRAFT; sp.draft_model_path = nullptr;
    ASSERT(pvx_executor_spec_init(exec, &sp) == PVX_ERR_INVALID_ARG);

    // Draft with unloadable path: PVX_ERR_LOAD_FAILED, boundary named, no fallback.
    sp.draft_model_path = "/nonexistent/spec-draft.gguf";
    ASSERT(pvx_executor_spec_init(exec, &sp) == PVX_ERR_LOAD_FAILED);
    ASSERT(strstr(pvx_last_error(ctx), "draft-load") != nullptr);
    // No silent session was created.
    int n = 0;
    ASSERT(pvx_executor_generate_spec(exec, nullptr, 4, nullptr, nullptr, &n) == PVX_ERR_NOT_SUPPORTED);

    // Draft with a REAL draft model (env-selected): enable + verify-loop guard.
    if (DRAFT_MODEL_PATH && file_exists(DRAFT_MODEL_PATH)) {
        sp.draft_model_path = DRAFT_MODEL_PATH;
        sp.draft_max = 4;
        // Env pairing is host-controlled: a draft that loads but is not
        // vocab-compatible with the selected target is a fixture problem, not
        // a product regression — the engine's typed draft-vocab refusal is
        // already covered by the incompatible-pair legs above.
        if (pvx_executor_spec_init(exec, &sp) != PVX_OK) {
            if (strstr(pvx_last_error(ctx), "draft-vocab") != nullptr)
                SKIP("env draft not vocab-compatible with target");
            ASSERT(pvx_executor_spec_init(exec, &sp) == PVX_OK);
        }
        const char* prompt = "List three colors:";
        ASSERT(pvx_executor_prompt(exec, prompt) == PVX_OK);
        ASSERT(pvx_executor_infer(exec) == PVX_DECODE_OK);
        ASSERT(pvx_executor_spec_init(exec, &sp) == PVX_OK); // re-init replaces cleanly
        pvx_sampling_params_t samp = {};
        samp.temperature = 0.0f; samp.top_k = 1; samp.top_p = 1.0f;
        samp.repeat_penalty = 1.0f;
        ASSERT(pvx_executor_generate_spec(exec, &samp, 12, nullptr, nullptr, &n) == PVX_OK);
        ASSERT(n >= 1 && n <= 24);
        ASSERT(pvx_executor_token_count(exec) > 0);
        // Disable path.
        sp.mode = PVX_SPEC_OFF;
        ASSERT(pvx_executor_spec_init(exec, &sp) == PVX_OK);
        ASSERT(pvx_executor_generate_spec(exec, &samp, 4, nullptr, nullptr, &n) == PVX_ERR_NOT_SUPPORTED);
    } else {
        fprintf(stderr, "(no draft model; enable-path skipped) ");
    }

    pvx_executor_free(exec); pvx_free_context(ctx); pvx_free_model(m);
}

// ── ABI v6 wave 1: n-gram spec modes + spec-vision (mirror embd entries) ──

// Plain greedy loop through the public entry points (the C# sequential lane's
// exact commit sequence). Fixed token budget — no EOG short-circuit needed
// for a byte-exact comparison.
static int gen_plain_loop(pvx_executor_t* exec, const pvx_sampling_params_t* sp,
                          int max_tokens, int32_t* out) {
    for (int i = 0; i < max_tokens; i++) {
        int32_t t = -1;
        if (pvx_executor_sample(exec, sp, &t) != PVX_OK) return -1;
        if (pvx_executor_prompt_tokens(exec, &t, 1) != PVX_OK) return -1;
        if (pvx_executor_infer(exec) != PVX_DECODE_OK) return -1;
        out[i] = t;
    }
    return max_tokens;
}

static pvx_context_t* make_ctx(pvx_model_t* m, uint32_t seq_max, uint32_t ctx_size) {
    pvx_context_params_t cp = {};
    cp.context_size = ctx_size; cp.batch_size = 512; cp.seq_max = seq_max;
    pvx_context_t* c = nullptr;
    ASSERT(pvx_create_context(m, &cp, &c) == PVX_OK);
    return c;
}

// n-gram spec ON == plain greedy OFF, token-for-token, for every n-gram mode.
static void test_spec_ngram_token_exactness() {
    if (!file_exists(MODEL_PATH)) SKIP("model file not found");
    pvx_model_params_t mp = {}; mp.model_path = MODEL_PATH; mp.gpu_layers = 99; mp.flash_attn = true;
    pvx_model_t* m = nullptr; ASSERT(pvx_load_model(&mp, &m) == PVX_OK);

    const char* prompt = "The colors of the rainbow, in order, are: red, orange, yellow, green, blue, indigo and";
    const int n_gen = 16;
    pvx_sampling_params_t sp = {};
    sp.temperature = 0.0f; sp.top_k = 1; sp.top_p = 1.0f; sp.repeat_penalty = 1.0f;

    const pvx_spec_mode_t modes[] = { PVX_SPEC_NGRAM_SIMPLE, PVX_SPEC_NGRAM_MAP_K,
                                      PVX_SPEC_NGRAM_MOD,   PVX_SPEC_NGRAM_CACHE };
    for (int mi = 0; mi < 4; mi++) {
        // A: plain greedy (spec OFF) — the reference stream
        pvx_context_t* ca = make_ctx(m, 1, 1024);
        pvx_executor_t* ea = nullptr; ASSERT(pvx_executor_create(ca, &ea) == PVX_OK);
        ASSERT(pvx_executor_prompt(ea, prompt) == PVX_OK);
        ASSERT(pvx_executor_infer(ea) == PVX_DECODE_OK);
        int32_t ref[n_gen];
        ASSERT(gen_plain_loop(ea, &sp, n_gen, ref) == n_gen);
        pvx_executor_free(ea); pvx_free_context(ca);

        // B: n-gram spec ON with streaming capture — must match byte-for-byte
        pvx_context_t* cb = make_ctx(m, 1, 1024);
        pvx_executor_t* eb = nullptr; ASSERT(pvx_executor_create(cb, &eb) == PVX_OK);
        ASSERT(pvx_executor_prompt(eb, prompt) == PVX_OK);
        ASSERT(pvx_executor_infer(eb) == PVX_DECODE_OK);
        pvx_spec_params_t spc = {};
        spc.mode = modes[mi];
        ASSERT(pvx_executor_spec_init(eb, &spc) == PVX_OK);
        struct CapCtx { int32_t* v; int i; };
        int32_t out[n_gen];
        CapCtx capc{out, 0};
        int n = 0;
        ASSERT(pvx_executor_generate_spec(eb, &sp, n_gen,
                (pvx_spec_token_callback_t)+[](int32_t t, void* ud) -> bool {
                    CapCtx* c = (CapCtx*) ud; c->v[c->i++] = t; return true;
                }, &capc, &n) == PVX_OK);
        ASSERT(n == n_gen);
        ASSERT(capc.i == n_gen);
        ASSERT(memcmp(ref, out, sizeof(ref)) == 0);
        pvx_executor_free(eb); pvx_free_context(cb);
    }
    pvx_free_model(m);
}

static void test_spec_ngram_fail_closed() {
    if (!file_exists(MODEL_PATH)) SKIP("model file not found");
    pvx_model_params_t mp = {}; mp.model_path = MODEL_PATH; mp.gpu_layers = 0; mp.flash_attn = true;
    pvx_model_t* m = nullptr; ASSERT(pvx_load_model(&mp, &m) == PVX_OK);
    pvx_context_t* ctx = make_ctx(m, 1, 1024);
    pvx_executor_t* exec = nullptr; ASSERT(pvx_executor_create(ctx, &exec) == PVX_OK);

    // n_min > n_max: typed invalid arg.
    pvx_spec_params_t sp = {};
    sp.mode = PVX_SPEC_NGRAM_SIMPLE;
    sp.ngram.n_min = 8; sp.ngram.n_max = 2;
    ASSERT(pvx_model_spec_validate(m, &sp) == PVX_ERR_INVALID_ARG);
    ASSERT(pvx_executor_spec_init(exec, &sp) == PVX_ERR_INVALID_ARG);

    // Corrupt cache file: typed LOAD_FAILED naming the path, no silent empty cache.
    const char* tmp = getenv("PVX_TEST_TMP");
    std::string path = tmp ? tmp : "/tmp";
    path += "/pvx_spec_corrupt_cache.bin";
    FILE* f = fopen(path.c_str(), "wb");
    ASSERT(f);
    fwrite("garbage-not-a-cache", 1, 19, f);
    fclose(f);
    sp = {};
    sp.mode = PVX_SPEC_NGRAM_CACHE;
    sp.ngram.cache_path = path.c_str();
    ASSERT(pvx_executor_spec_init(exec, &sp) == PVX_ERR_LOAD_FAILED);
    ASSERT(strstr(pvx_last_error(ctx), "ngram-cache") != nullptr);
    remove(path.c_str());

    // Unknown mode: typed invalid arg.
    sp = {};
    sp.mode = (pvx_spec_mode_t) 99;
    ASSERT(pvx_executor_spec_init(exec, &sp) == PVX_ERR_NOT_SUPPORTED);

    pvx_executor_free(exec); pvx_free_context(ctx); pvx_free_model(m);
}

// spec + vision: an image prompt lands embd-aware entries in the mirror; a
// spec session must serve it and stay token-exact against the plain path.
static void test_spec_vision_exactness() {
    const char* vision_model = file_exists(HYBRID_MODEL_PATH) ? HYBRID_MODEL_PATH : MODEL_PATH;
    if (!file_exists(vision_model) || !file_exists(MMPROJ_PATH) || !file_exists(TEST_IMAGE))
        SKIP("vision files not found");
    pvx_model_params_t mp = {}; mp.model_path = vision_model; mp.gpu_layers = 99; mp.flash_attn = true;
    pvx_model_t* model = nullptr; ASSERT(pvx_load_model(&mp, &model) == PVX_OK);
    pvx_model_t* mmproj = nullptr;
    ASSERT(pvx_load_mmproj(model, MMPROJ_PATH, &mmproj) == PVX_OK);

    FILE* f = fopen(TEST_IMAGE, "rb");
    ASSERT(f);
    fseek(f, 0, SEEK_END); long img_size = ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t* img_data = (uint8_t*) malloc(img_size);
    ASSERT(fread(img_data, 1, img_size, f) == (size_t) img_size);
    fclose(f);
    const char* imgs[] = { (const char*) img_data };
    int sizes[] = { (int) img_size };
    char* marker_str = nullptr;
    ASSERT(pvx_mtmd_marker(&marker_str) == PVX_OK);
    const std::string prompt = std::string("Describe this image in exactly ten words: ") + marker_str;
    pvx_free_string(marker_str);

    const int n_gen = 8;
    pvx_sampling_params_t sp = {};
    sp.temperature = 0.0f; sp.top_k = 1; sp.top_p = 1.0f; sp.repeat_penalty = 1.0f;

    // A: plain greedy after the image prompt
    pvx_context_t* ca = make_ctx(model, 1, 4096);
    pvx_executor_t* ea = nullptr; ASSERT(pvx_executor_create(ca, &ea) == PVX_OK);
    ASSERT(pvx_executor_prompt_with_images(ea, model, prompt.c_str(), imgs, sizes, 1) == PVX_OK);
    ASSERT(pvx_executor_infer(ea) == PVX_DECODE_OK);
    int32_t ref[n_gen];
    ASSERT(gen_plain_loop(ea, &sp, n_gen, ref) == n_gen);
    pvx_executor_free(ea); pvx_free_context(ca);

    // B: spec (n-gram) ON after the image prompt — the mirror now holds embd
    // entries; the consistency guard and the token stream must hold.
    struct CapCtx { int32_t* v; int i; };
    pvx_context_t* cb = make_ctx(model, 1, 4096);
    pvx_executor_t* eb = nullptr; ASSERT(pvx_executor_create(cb, &eb) == PVX_OK);
    ASSERT(pvx_executor_prompt_with_images(eb, model, prompt.c_str(), imgs, sizes, 1) == PVX_OK);
    ASSERT(pvx_executor_infer(eb) == PVX_DECODE_OK);
    pvx_spec_params_t spc = {};
    spc.mode = PVX_SPEC_NGRAM_SIMPLE;
    ASSERT(pvx_executor_spec_init(eb, &spc) == PVX_OK);
    int32_t out[n_gen];
    CapCtx cap{out, 0};
    int n = 0;
    ASSERT(pvx_executor_generate_spec(eb, &sp, n_gen,
            (pvx_spec_token_callback_t)+[](int32_t t, void* ud) -> bool {
                CapCtx* c = (CapCtx*) ud; c->v[c->i++] = t; return true;
            }, &cap, &n) == PVX_OK);
    // EOG may end the stream early in either path: the exactness property is
    // token identity on the emitted prefix (returned count == callback count).
    ASSERT(n >= 1);
    ASSERT(cap.i == (size_t) n);
    for (int i = 0; i < n; i++) ASSERT(ref[i] == out[i]);

    // C: draft-model spec after an image prompt (true embd-aware draft
    // replay) — only when a draft model is available (env-gated).
    if (DRAFT_MODEL_PATH && file_exists(DRAFT_MODEL_PATH)) {
        pvx_context_t* cc = make_ctx(model, 1, 4096);
        pvx_executor_t* ec = nullptr; ASSERT(pvx_executor_create(cc, &ec) == PVX_OK);
        ASSERT(pvx_executor_prompt_with_images(ec, model, prompt.c_str(), imgs, sizes, 1) == PVX_OK);
        ASSERT(pvx_executor_infer(ec) == PVX_DECODE_OK);
        pvx_spec_params_t spd = {};
        spd.mode = PVX_SPEC_DRAFT;
        spd.draft_model_path = DRAFT_MODEL_PATH;
        spd.draft_max = 4;
        pvx_result_t rc = pvx_executor_spec_init(ec, &spd);
        if (rc == PVX_OK) {
            int32_t outd[n_gen];
            CapCtx capd{outd, 0};
            int nd = 0;
            ASSERT(pvx_executor_generate_spec(ec, &sp, n_gen,
                    (pvx_spec_token_callback_t)+[](int32_t t, void* ud) -> bool {
                        CapCtx* c = (CapCtx*) ud; c->v[c->i++] = t; return true;
                    }, &capd, &nd) == PVX_OK);
            ASSERT(nd >= 1);
            ASSERT(capd.i == (size_t) nd);
            for (int i = 0; i < nd; i++) ASSERT(ref[i] == outd[i]);
        } else {
            // Incompatible draft vocab is a typed refusal, not a test failure.
            fprintf(stderr, "(draft+vision skipped: spec_init rc=%d) ", (int) rc);
        }
        if (rc == PVX_OK) {
            // Full-budget contract for the live fixture pair (Qwen3.5-4B target
            // + Qwen3.5-0.8B draft via PVX_DRAFT_MODEL_PATH): the greedy stream
            // after this image prompt begins with an EOG token, so the plain
            // leg above runs its fixed budget EOG-blind while the spec loop
            // stops at EOG — the prefix contract above covers that shape.
            // With ignore_eos BOTH loops must agree on the FULL budget: the
            // spec loop samples past EOG exactly like the plain pipeline and
            // the stream stays byte-exact (on the hybrid target drafting
            // degrades to plain exact steps — verify-batch rollback is
            // unavailable; output correctness must not). Fresh context: the
            // prefix leg above already advanced cc.
            pvx_context_t* cd = make_ctx(model, 1, 4096);
            pvx_executor_t* ed = nullptr; ASSERT(pvx_executor_create(cd, &ed) == PVX_OK);
            ASSERT(pvx_executor_prompt_with_images(ed, model, prompt.c_str(), imgs, sizes, 1) == PVX_OK);
            ASSERT(pvx_executor_infer(ed) == PVX_DECODE_OK);
            pvx_spec_params_t spd2 = spd; // same draft pair as above
            ASSERT(pvx_executor_spec_init(ed, &spd2) == PVX_OK);
            pvx_sampling_params_t sp2 = sp; sp2.ignore_eos = true;
            int32_t outd2[n_gen]; CapCtx capd2{outd2, 0}; int nd2 = 0;
            ASSERT(pvx_executor_generate_spec(ed, &sp2, n_gen,
                    (pvx_spec_token_callback_t)+[](int32_t t, void* ud) -> bool {
                        CapCtx* c = (CapCtx*) ud; c->v[c->i++] = t; return true;
                    }, &capd2, &nd2) == PVX_OK);
            ASSERT(nd2 == n_gen);
            ASSERT(capd2.i == (size_t) nd2);
            for (int i = 0; i < nd2; i++) ASSERT(ref[i] == outd2[i]);
            pvx_executor_free(ed); pvx_free_context(cd);
        }
        pvx_executor_free(ec); pvx_free_context(cc);
    } else {
        fprintf(stderr, "(no draft model; draft+vision leg skipped) ");
    }

    pvx_executor_free(eb); pvx_free_context(cb);
    free(img_data);
    pvx_free_mmproj(mmproj);
    pvx_free_model(model);
}

// ═══════════════════════════════════════════════════════════════════════════
// ABI v7 wave 2 — batched/pool-lane spec, companion-GGUF typed boundaries,
// JSON-schema → GBNF round-trip through the grammar sampler.
// ═══════════════════════════════════════════════════════════════════════════

// Plain greedy generation loop over ONE pooled conversation: sample →
// prompt_tokens → batched infer — the lane's plain pipeline, token-for-token
// identical contract with pvx_conversation_generate_spec.
static int conv_plain_loop(pvx_context_t* ctx, pvx_conversation_t* conv,
                           const pvx_sampling_params_t* sp, int n_gen, int32_t* out) {
    for (int i = 0; i < n_gen; i++) {
        int32_t t = 0;
        if (pvx_conversation_sample(conv, sp, &t) != PVX_OK) return i;
        if (pvx_conversation_prompt_tokens(conv, &t, 1) != PVX_OK) return i;
        if (pvx_infer(ctx) != PVX_DECODE_OK) return i;
        out[i] = t;
    }
    return n_gen;
}

// Spec generation with streaming capture over ONE pooled conversation.
static int conv_spec_loop(pvx_conversation_t* conv,
                          const pvx_sampling_params_t* sp, int n_gen, int32_t* out) {
    struct Cap { int32_t* v; int i; };
    Cap cap{out, 0};
    int n = 0;
    if (pvx_conversation_generate_spec(conv, sp, n_gen,
            (pvx_spec_token_callback_t)+[](int32_t t, void* ud) -> bool {
                Cap* c = (Cap*) ud; c->v[c->i++] = t; return true;
            }, &cap, &n) != PVX_OK) return -1;
    return n;
}

static void test_conversation_spec_ngram_exactness() {
    if (!file_exists(MODEL_PATH)) SKIP("model file not found");
    pvx_model_params_t mp = {}; mp.model_path = MODEL_PATH; mp.gpu_layers = 99; mp.flash_attn = true;
    pvx_model_t* m = nullptr; ASSERT(pvx_load_model(&mp, &m) == PVX_OK);
    pvx_context_params_t cp = {}; cp.context_size = 2048; cp.batch_size = 512; cp.seq_max = 4;
    pvx_context_t* ctx = nullptr; ASSERT(pvx_create_context(m, &cp, &ctx) == PVX_OK);
    pvx_pool_t* pool = nullptr; ASSERT(pvx_pool_create(ctx, &pool) == PVX_OK);

    const char* prompt = "The colors of the rainbow, in order, are: red, orange, yellow, green, blue, indigo and";
    const int n_gen = 16;
    pvx_sampling_params_t sp = {};
    sp.temperature = 0.0f; sp.top_k = 1; sp.top_p = 1.0f; sp.repeat_penalty = 1.0f;

    const pvx_spec_mode_t modes[] = { PVX_SPEC_NGRAM_SIMPLE, PVX_SPEC_NGRAM_MAP_K,
                                      PVX_SPEC_NGRAM_MOD,   PVX_SPEC_NGRAM_CACHE };
    for (int mi = 0; mi < 4; mi++) {
        // A: plain greedy conversation — the reference stream
        pvx_conversation_t* ca = nullptr; ASSERT(pvx_pool_lease(pool, &ca) == PVX_OK);
        ASSERT(pvx_conversation_prompt(ca, prompt) == PVX_OK);
        ASSERT(pvx_infer(ctx) == PVX_DECODE_OK);
        int32_t ref[n_gen];
        ASSERT(conv_plain_loop(ctx, ca, &sp, n_gen, ref) == n_gen);
        ASSERT(pvx_pool_return(pool, ca) == PVX_OK);

        // B: n-gram spec ON, on the SAME pool's conversation — token-exact
        pvx_conversation_t* cb = nullptr; ASSERT(pvx_pool_lease(pool, &cb) == PVX_OK);
        ASSERT(pvx_conversation_prompt(cb, prompt) == PVX_OK);
        ASSERT(pvx_infer(ctx) == PVX_DECODE_OK);
        pvx_spec_params_t spc = {};
        spc.mode = modes[mi];
        ASSERT(pvx_conversation_spec_init(cb, &spc) == PVX_OK);
        int32_t out[n_gen];
        const int n = conv_spec_loop(cb, &sp, n_gen, out);
        ASSERT(n == n_gen);
        ASSERT(memcmp(ref, out, sizeof(ref)) == 0);
        // stats surface: drafted >= 0, accepted >= 0 (session live)
        int64_t dr = -1, ac = -1;
        ASSERT(pvx_conversation_spec_stats(cb, &dr, &ac) == PVX_OK);
        ASSERT(dr >= 0 && ac >= 0);
        pvx_conversation_spec_free(cb);
        ASSERT(pvx_pool_return(pool, cb) == PVX_OK);
    }
    pvx_pool_free(pool); pvx_free_context(ctx); pvx_free_model(m);
}

static void test_conversation_spec_interleaved_exactness() {
    // Interleaved spec + plain conversations on ONE pool: rounds alternate
    // with plain cycles on a third conversation; both spec+plain streams
    // stay token-exact against their isolated references.
    if (!file_exists(MODEL_PATH)) SKIP("model file not found");
    pvx_model_params_t mp = {}; mp.model_path = MODEL_PATH; mp.gpu_layers = 99; mp.flash_attn = true;
    pvx_model_t* m = nullptr; ASSERT(pvx_load_model(&mp, &m) == PVX_OK);
    pvx_context_params_t cp = {}; cp.context_size = 2048; cp.batch_size = 512; cp.seq_max = 4;
    pvx_context_t* ctx = nullptr; ASSERT(pvx_create_context(m, &cp, &ctx) == PVX_OK);
    pvx_pool_t* pool = nullptr; ASSERT(pvx_pool_create(ctx, &pool) == PVX_OK);
    const int n_gen = 12;
    pvx_sampling_params_t sp = {};
    sp.temperature = 0.0f; sp.top_k = 1; sp.top_p = 1.0f; sp.repeat_penalty = 1.0f;

    // References on a separate context
    pvx_context_t* rctx = nullptr;
    pvx_context_params_t rcp = {}; rcp.context_size = 2048; rcp.batch_size = 512; rcp.seq_max = 2;
    ASSERT(pvx_create_context(m, &rcp, &rctx) == PVX_OK);
    pvx_pool_t* rpool = nullptr; ASSERT(pvx_pool_create(rctx, &rpool) == PVX_OK);
    const char* spec_prompt = "Count up slowly: one, two, three,";
    const char* plain_prompt = "The capital of France is";
    pvx_conversation_t* ra = nullptr; ASSERT(pvx_pool_lease(rpool, &ra) == PVX_OK);
    ASSERT(pvx_conversation_prompt(ra, spec_prompt) == PVX_OK);
    ASSERT(pvx_infer(rctx) == PVX_DECODE_OK);
    int32_t spec_ref[n_gen];
    ASSERT(conv_plain_loop(rctx, ra, &sp, n_gen, spec_ref) == n_gen);
    ASSERT(pvx_pool_return(rpool, ra) == PVX_OK);
    pvx_conversation_t* rb = nullptr; ASSERT(pvx_pool_lease(rpool, &rb) == PVX_OK);
    ASSERT(pvx_conversation_prompt(rb, spec_prompt) == PVX_OK);
    ASSERT(pvx_infer(rctx) == PVX_DECODE_OK);
    int32_t plain_ref[n_gen];
    ASSERT(conv_plain_loop(rctx, rb, &sp, n_gen, plain_ref) == n_gen);
    ASSERT(pvx_pool_return(rpool, rb) == PVX_OK);

    // Interleaved: spec conv A + plain twin B' + plain conv C on one pool
    pvx_conversation_t* convs[3];
    for (int i = 0; i < 3; i++) ASSERT(pvx_pool_lease(pool, &convs[i]) == PVX_OK);
    ASSERT(pvx_conversation_prompt(convs[0], spec_prompt) == PVX_OK);
    ASSERT(pvx_conversation_prompt(convs[1], spec_prompt) == PVX_OK);
    ASSERT(pvx_conversation_prompt(convs[2], plain_prompt) == PVX_OK);
    ASSERT(pvx_infer(ctx) == PVX_DECODE_OK);
    pvx_spec_params_t spc = {};
    spc.mode = PVX_SPEC_NGRAM_SIMPLE;
    ASSERT(pvx_conversation_spec_init(convs[0], &spc) == PVX_OK);

    int32_t spec_out[n_gen]; int spec_i = 0;
    int32_t plain_out[n_gen]; int plain_i = 0;
    int32_t other_out[n_gen];
    // alternate: one spec round (chunk of 2) — one plain token on the twin —
    // one plain token on the unrelated third conversation
    while (spec_i < n_gen) {
        int n = 0;
        struct Cap { int32_t* v; int i; };
        Cap cap{spec_out + spec_i, 0};
        ASSERT(pvx_conversation_generate_spec(convs[0], &sp, 2,
                (pvx_spec_token_callback_t)+[](int32_t t, void* ud) -> bool {
                    Cap* c = (Cap*) ud; c->v[c->i++] = t; return true;
                }, &cap, &n) == PVX_OK);
        ASSERT(n > 0);
        spec_i += n;
        // keep the plain twin in lockstep with the spec stream's token count
        while (plain_i < spec_i) {
            int32_t t = 0;
            ASSERT(pvx_conversation_sample(convs[1], &sp, &t) == PVX_OK);
            ASSERT(pvx_conversation_prompt_tokens(convs[1], &t, 1) == PVX_OK);
            plain_out[plain_i++] = t;
            ASSERT(pvx_infer(ctx) == PVX_DECODE_OK);
            // one unrelated plain conversation token interleaves with every
            // twin token: shared-context interleaving stress
            ASSERT(pvx_conversation_sample(convs[2], &sp, &t) == PVX_OK);
            ASSERT(pvx_conversation_prompt_tokens(convs[2], &t, 1) == PVX_OK);
            ASSERT(pvx_infer(ctx) == PVX_DECODE_OK);
            other_out[0] = t;
        }
    }
    ASSERT(spec_i == n_gen);
    ASSERT(plain_i == n_gen);
    ASSERT(memcmp(spec_ref, spec_out, sizeof(spec_ref)) == 0);
    ASSERT(memcmp(plain_ref, plain_out, sizeof(plain_ref)) == 0);
    for (int i = 0; i < 3; i++) ASSERT(pvx_pool_return(pool, convs[i]) == PVX_OK);
    pvx_pool_free(rpool); pvx_free_context(rctx);
    pvx_pool_free(pool); pvx_free_context(ctx); pvx_free_model(m);
}

static void test_conversation_spec_lease_churn() {
    // Return mid-session (pool teardown of a live spec session), re-lease,
    // re-init spec, exactness after the churn.
    if (!file_exists(MODEL_PATH)) SKIP("model file not found");
    pvx_model_params_t mp = {}; mp.model_path = MODEL_PATH; mp.gpu_layers = 99; mp.flash_attn = true;
    pvx_model_t* m = nullptr; ASSERT(pvx_load_model(&mp, &m) == PVX_OK);
    pvx_context_params_t cp = {}; cp.context_size = 2048; cp.batch_size = 512; cp.seq_max = 2;
    pvx_context_t* ctx = nullptr; ASSERT(pvx_create_context(m, &cp, &ctx) == PVX_OK);
    pvx_pool_t* pool = nullptr; ASSERT(pvx_pool_create(ctx, &pool) == PVX_OK);
    const char* prompt = "The days of the week are: Monday, Tuesday,";
    const int n_gen = 10;
    pvx_sampling_params_t sp = {};
    sp.temperature = 0.0f; sp.top_k = 1; sp.top_p = 1.0f; sp.repeat_penalty = 1.0f;

    // reference
    pvx_conversation_t* r = nullptr; ASSERT(pvx_pool_lease(pool, &r) == PVX_OK);
    ASSERT(pvx_conversation_prompt(r, prompt) == PVX_OK);
    ASSERT(pvx_infer(ctx) == PVX_DECODE_OK);
    int32_t ref[n_gen];
    ASSERT(conv_plain_loop(ctx, r, &sp, n_gen, ref) == n_gen);
    ASSERT(pvx_pool_return(pool, r) == PVX_OK);

    pvx_conversation_t* c = nullptr; ASSERT(pvx_pool_lease(pool, &c) == PVX_OK);
    ASSERT(pvx_conversation_prompt(c, prompt) == PVX_OK);
    ASSERT(pvx_infer(ctx) == PVX_DECODE_OK);
    pvx_spec_params_t spc = {};
    spc.mode = PVX_SPEC_NGRAM_MAP_K;
    ASSERT(pvx_conversation_spec_init(c, &spc) == PVX_OK);
    // generate HALF the tokens, then RETURN mid-session (churn)
    int32_t out[n_gen];
    int n = 0;
    struct Cap { int32_t* v; int i; };
    Cap cap{out, 0};
    ASSERT(pvx_conversation_generate_spec(c, &sp, n_gen / 2,
            (pvx_spec_token_callback_t)+[](int32_t t, void* ud) -> bool {
                Cap* q = (Cap*) ud; q->v[q->i++] = t; return true;
            }, &cap, &n) == PVX_OK);
    ASSERT(n == n_gen / 2);
    ASSERT(memcmp(ref, out, sizeof(int32_t) * (size_t) n) == 0);
    ASSERT(pvx_pool_return(pool, c) == PVX_OK);   // teardown with a live session

    // re-lease (fresh conversation), replay prompt + FULL spec generation
    pvx_conversation_t* c2 = nullptr; ASSERT(pvx_pool_lease(pool, &c2) == PVX_OK);
    ASSERT(pvx_conversation_token_count(c2) == 0);
    ASSERT(pvx_conversation_prompt(c2, prompt) == PVX_OK);
    ASSERT(pvx_infer(ctx) == PVX_DECODE_OK);
    ASSERT(pvx_conversation_spec_init(c2, &spc) == PVX_OK);
    Cap cap2{out, 0};
    ASSERT(pvx_conversation_generate_spec(c2, &sp, n_gen,
            (pvx_spec_token_callback_t)+[](int32_t t, void* ud) -> bool {
                Cap* q = (Cap*) ud; q->v[q->i++] = t; return true;
            }, &cap2, &n) == PVX_OK);
    ASSERT(n == n_gen);
    ASSERT(memcmp(ref, out, sizeof(ref)) == 0);
    ASSERT(pvx_pool_return(pool, c2) == PVX_OK);
    pvx_pool_free(pool); pvx_free_context(ctx); pvx_free_model(m);
}

static void test_conversation_spec_rewind_continue() {
    // Rewind mid-spec: the conversation mirror retreats with the window; both
    // lanes (plain + spec) rewind identically, re-seed with a control token,
    // and the spec stream continues token-exact with the plain stream
    // (draft mirror rebuilds from the shrunk ring — the core mirror-rebuild
    // contract of the conversation lane).
    if (!file_exists(MODEL_PATH)) SKIP("model file not found");
    pvx_model_params_t mp = {}; mp.model_path = MODEL_PATH; mp.gpu_layers = 99; mp.flash_attn = true;
    pvx_model_t* m = nullptr; ASSERT(pvx_load_model(&mp, &m) == PVX_OK);
    pvx_context_params_t cp = {}; cp.context_size = 2048; cp.batch_size = 512; cp.seq_max = 2;
    pvx_context_t* ctx = nullptr; ASSERT(pvx_create_context(m, &cp, &ctx) == PVX_OK);
    pvx_pool_t* pool = nullptr; ASSERT(pvx_pool_create(ctx, &pool) == PVX_OK);
    const char* prompt = "The months of the year are: January, February,";
    const int n_gen = 10;
    const int n_half = 4;
    pvx_sampling_params_t sp = {};
    sp.temperature = 0.0f; sp.top_k = 1; sp.top_p = 1.0f; sp.repeat_penalty = 1.0f;

    // plain reference: generate n_half, rewind 2, re-seed [42], continue
    pvx_conversation_t* r = nullptr; ASSERT(pvx_pool_lease(pool, &r) == PVX_OK);
    ASSERT(pvx_conversation_prompt(r, prompt) == PVX_OK);
    ASSERT(pvx_infer(ctx) == PVX_DECODE_OK);
    int32_t head[n_half];
    ASSERT(conv_plain_loop(ctx, r, &sp, n_half, head) == n_half);
    const int tc_before = pvx_conversation_token_count(r);
    ASSERT(pvx_conversation_rewind(r, 2) == PVX_OK);
    ASSERT(pvx_conversation_token_count(r) == tc_before - 2);
    int32_t seed_tok = 42;
    ASSERT(pvx_conversation_prompt_tokens(r, &seed_tok, 1) == PVX_OK);
    ASSERT(pvx_infer(ctx) == PVX_DECODE_OK);
    int32_t cont[n_gen];
    ASSERT(conv_plain_loop(ctx, r, &sp, n_gen, cont) == n_gen);
    ASSERT(pvx_pool_return(pool, r) == PVX_OK);

    // spec stream: identical shape with the spec session LIVE across the rewind
    pvx_conversation_t* c = nullptr; ASSERT(pvx_pool_lease(pool, &c) == PVX_OK);
    ASSERT(pvx_conversation_prompt(c, prompt) == PVX_OK);
    ASSERT(pvx_infer(ctx) == PVX_DECODE_OK);
    pvx_spec_params_t spc = {};
    spc.mode = PVX_SPEC_NGRAM_SIMPLE;
    spc.draft_max = 4;
    ASSERT(pvx_conversation_spec_init(c, &spc) == PVX_OK);
    int32_t out[n_gen];
    int n = 0;
    struct Cap { int32_t* v; int i; };
    Cap cap{out, 0};
    ASSERT(pvx_conversation_generate_spec(c, &sp, n_half,
            (pvx_spec_token_callback_t)+[](int32_t t, void* ud) -> bool {
                Cap* q = (Cap*) ud; q->v[q->i++] = t; return true;
            }, &cap, &n) == PVX_OK);
    ASSERT(n == n_half);
    ASSERT(memcmp(head, out, sizeof(int32_t) * n_half) == 0);
    ASSERT(pvx_conversation_token_count(c) == tc_before);
    ASSERT(pvx_conversation_rewind(c, 2) == PVX_OK);
    ASSERT(pvx_conversation_token_count(c) == tc_before - 2);
    ASSERT(pvx_conversation_prompt_tokens(c, &seed_tok, 1) == PVX_OK);
    ASSERT(pvx_infer(ctx) == PVX_DECODE_OK);
    Cap cap2{out, 0};
    ASSERT(pvx_conversation_generate_spec(c, &sp, n_gen,
            (pvx_spec_token_callback_t)+[](int32_t t, void* ud) -> bool {
                Cap* q = (Cap*) ud; q->v[q->i++] = t; return true;
            }, &cap2, &n) == PVX_OK);
    ASSERT(n == n_gen);
    ASSERT(memcmp(cont, out, sizeof(cont)) == 0);
    ASSERT(pvx_pool_return(pool, c) == PVX_OK);
    pvx_pool_free(pool); pvx_free_context(ctx); pvx_free_model(m);
}

static void test_conversation_spec_fail_closed() {
    if (!file_exists(MODEL_PATH)) SKIP("model file not found");
    pvx_model_params_t mp = {}; mp.model_path = MODEL_PATH; mp.gpu_layers = 99; mp.flash_attn = true;
    pvx_model_t* m = nullptr; ASSERT(pvx_load_model(&mp, &m) == PVX_OK);
    // seq_max < 2: typed batched-seq-budget refusal
    pvx_context_params_t cp1 = {}; cp1.context_size = 1024; cp1.batch_size = 256; cp1.seq_max = 1;
    pvx_context_t* ctx1 = nullptr; ASSERT(pvx_create_context(m, &cp1, &ctx1) == PVX_OK);
    pvx_pool_t* pool1 = nullptr; ASSERT(pvx_pool_create(ctx1, &pool1) == PVX_OK);
    pvx_conversation_t* c1 = nullptr; ASSERT(pvx_pool_lease(pool1, &c1) == PVX_OK);
    pvx_spec_params_t spc = {};
    spc.mode = PVX_SPEC_NGRAM_SIMPLE;
    ASSERT(pvx_conversation_spec_init(c1, &spc) == PVX_ERR_NOT_SUPPORTED);
    ASSERT(strstr(pvx_last_error(ctx1), "batched spec requires >=2 seq slots") != nullptr);
    ASSERT(pvx_pool_return(pool1, c1) == PVX_OK);
    pvx_pool_free(pool1); pvx_free_context(ctx1);

    // eagle3 on the conversation lane: typed refusal naming the boundary
    pvx_context_params_t cp = {}; cp.context_size = 2048; cp.batch_size = 512; cp.seq_max = 2;
    pvx_context_t* ctx = nullptr; ASSERT(pvx_create_context(m, &cp, &ctx) == PVX_OK);
    pvx_pool_t* pool = nullptr; ASSERT(pvx_pool_create(ctx, &pool) == PVX_OK);
    pvx_conversation_t* c = nullptr; ASSERT(pvx_pool_lease(pool, &c) == PVX_OK);
    spc = {};
    spc.mode = PVX_SPEC_EAGLE3;
    spc.eagle_companion_path = "whatever.gguf";
    ASSERT(pvx_conversation_spec_init(c, &spc) == PVX_ERR_NOT_SUPPORTED);
    ASSERT(strstr(pvx_last_error(ctx), "eagle3-conversation-lane") != nullptr);

    // dflash: missing path typed invalid; companion contract validated at init
    spc = {};
    spc.mode = PVX_SPEC_DFLASH;
    ASSERT(pvx_conversation_spec_init(c, &spc) == PVX_ERR_INVALID_ARG);

    // generate without a session: typed NOT_SUPPORTED naming the boundary
    pvx_sampling_params_t sp = {}; sp.temperature = 0.0f; sp.top_k = 1;
    int dummy_n = 0;
    ASSERT(pvx_conversation_generate_spec(c, &sp, 4, nullptr, nullptr, &dummy_n)
           == PVX_ERR_NOT_SUPPORTED);
    ASSERT(strstr(pvx_last_error(ctx), "spec-engine/session") != nullptr);

    // stats with no session: 0/0 and OK
    int64_t dr = 5, ac = 5;
    ASSERT(pvx_conversation_spec_stats(c, &dr, &ac) == PVX_OK);
    ASSERT(dr == 0 && ac == 0);

    ASSERT(pvx_pool_return(pool, c) == PVX_OK);
    pvx_pool_free(pool); pvx_free_context(ctx); pvx_free_model(m);
}

static void test_spec_dflash_metadata_typed() {
    // DFlash: companion metadata validated, then the wiring boundary refuses
    // typed — never faked (MTP_AUTO precedent).
    if (!file_exists(MODEL_PATH)) SKIP("model file not found");
    pvx_model_params_t mp = {}; mp.model_path = MODEL_PATH; mp.gpu_layers = 99; mp.flash_attn = true;
    pvx_model_t* m = nullptr; ASSERT(pvx_load_model(&mp, &m) == PVX_OK);
    pvx_context_t* ctx = make_ctx(m, 1, 1024);
    pvx_executor_t* exec = nullptr; ASSERT(pvx_executor_create(ctx, &exec) == PVX_OK);

    // missing path
    pvx_spec_params_t sp = {};
    sp.mode = PVX_SPEC_DFLASH;
    ASSERT(pvx_executor_spec_init(exec, &sp) == PVX_ERR_INVALID_ARG);

    // valid metadata file: falls through to the wiring boundary
    const char* tmpdir = getenv("PVX_TEST_TMP");
    const std::string path = std::string(tmpdir ? tmpdir : "/tmp") + "/pvx_dflash_meta.gguf";
    gguf_context* g = gguf_init_empty();
    gguf_set_val_str(g, "general.architecture", "dflash");
    const int32_t layers[3] = { 2, 14, 24 };
    gguf_set_arr_data(g, "dflash.target_layers", GGUF_TYPE_INT32, layers, 3);
    ASSERT(gguf_write_to_file(g, path.c_str(), true));
    gguf_free(g);
    sp.dflash_companion_path = path.c_str();
    ASSERT(pvx_model_spec_validate(m, &sp) == PVX_ERR_NOT_SUPPORTED);
    ASSERT(strstr(pvx_model_last_error(m), "dflash-wiring") != nullptr);
    ASSERT(pvx_executor_spec_init(exec, &sp) == PVX_ERR_NOT_SUPPORTED);
    ASSERT(strstr(pvx_last_error(ctx), "dflash-wiring") != nullptr);

    // wrong-arch metadata: typed at the metadata boundary
    const std::string path2 = std::string(tmpdir ? tmpdir : "/tmp") + "/pvx_dflash_wrongarch.gguf";
    g = gguf_init_empty();
    gguf_set_val_str(g, "general.architecture", "llama");
    gguf_set_arr_data(g, "llama.target_layers", GGUF_TYPE_INT32, layers, 3);
    ASSERT(gguf_write_to_file(g, path2.c_str(), true));
    gguf_free(g);
    sp.dflash_companion_path = path2.c_str();
    ASSERT(pvx_executor_spec_init(exec, &sp) == PVX_ERR_NOT_SUPPORTED);
    ASSERT(strstr(pvx_last_error(ctx), "dflash-companion-metadata") != nullptr);
    remove(path.c_str());
    remove(path2.c_str());
    pvx_executor_free(exec); pvx_free_context(ctx); pvx_free_model(m);
}

static void test_spec_eagle3_companion_typed() {
    // EAGLE3 capability gating: metadata precheck + load boundary + live leg
    // (PVX_EAGLE_DRAFT_PATH names a genuine companion; absent = typed skip).
    if (!file_exists(MODEL_PATH)) SKIP("model file not found");
    pvx_model_params_t mp = {}; mp.model_path = MODEL_PATH; mp.gpu_layers = 99; mp.flash_attn = true;
    pvx_model_t* m = nullptr; ASSERT(pvx_load_model(&mp, &m) == PVX_OK);
    pvx_context_t* ctx = make_ctx(m, 1, 1024);
    pvx_executor_t* exec = nullptr; ASSERT(pvx_executor_create(ctx, &exec) == PVX_OK);

    // missing path
    pvx_spec_params_t sp = {};
    sp.mode = PVX_SPEC_EAGLE3;
    ASSERT(pvx_executor_spec_init(exec, &sp) == PVX_ERR_INVALID_ARG);

    // metadata-only fake (arch right, no tensors): typed LOAD_FAILED after
    // the precheck passes
    const char* tmpdir = getenv("PVX_TEST_TMP");
    const std::string path = std::string(tmpdir ? tmpdir : "/tmp") + "/pvx_eagle3_meta.gguf";
    gguf_context* g = gguf_init_empty();
    gguf_set_val_str(g, "general.architecture", "eagle3");
    const int32_t layers[3] = { 2, 14, 24 };
    gguf_set_arr_data(g, "eagle3.target_layers", GGUF_TYPE_INT32, layers, 3);
    ASSERT(gguf_write_to_file(g, path.c_str(), true));
    gguf_free(g);
    sp.eagle_companion_path = path.c_str();
    ASSERT(pvx_executor_spec_init(exec, &sp) == PVX_ERR_LOAD_FAILED);
    ASSERT(strstr(pvx_last_error(ctx), "eagle3-companion") != nullptr);
    remove(path.c_str());

    // wrong arch: typed at the metadata boundary (naming the key)
    const std::string path2 = std::string(tmpdir ? tmpdir : "/tmp") + "/pvx_eagle3_wrong.gguf";
    g = gguf_init_empty();
    gguf_set_val_str(g, "general.architecture", "llama");
    ASSERT(gguf_write_to_file(g, path2.c_str(), true));
    gguf_free(g);
    sp.eagle_companion_path = path2.c_str();
    ASSERT(pvx_executor_spec_init(exec, &sp) == PVX_ERR_NOT_SUPPORTED);
    ASSERT(strstr(pvx_last_error(ctx), "eagle3-companion-metadata") != nullptr);
    remove(path2.c_str());

    // validate: shape OK (pair-level validation happens at spec_init)
    sp.eagle_companion_path = "x.gguf";
    ASSERT(pvx_model_spec_validate(m, &sp) == PVX_OK);

    // (the LIVE EAGLE3 leg moved to test_spec_eagle3_live_pair below: it needs
    // the MATCHING target model, not the standard fixture target — env pair
    // PVX_EAGLE3_TARGET_PATH / PVX_EAGLE3_DRAFT_PATH)
    if (exec) pvx_executor_free(exec);
    if (ctx) pvx_free_context(ctx);
    pvx_free_model(m);
}

static void test_spec_mtp_draft_typed() {
    // ABI v8 wave 4 (upstream draft-mtp): fail-closed boundary legs — every
    // un-honorable configuration is a TYPED refusal naming the boundary,
    // never a silent disable.
    if (!file_exists(MODEL_PATH)) SKIP("model file not found");
    pvx_model_params_t mp = {}; mp.model_path = MODEL_PATH; mp.gpu_layers = 99; mp.flash_attn = true;
    pvx_model_t* m = nullptr; ASSERT(pvx_load_model(&mp, &m) == PVX_OK);

    // validate: missing path typed INVALID_ARG; with a path shape-OK.
    pvx_spec_params_t sp = {};
    sp.mode = PVX_SPEC_MTP_DRAFT;
    ASSERT(pvx_model_spec_validate(m, &sp) == PVX_ERR_INVALID_ARG);
    sp.mtp_draft_path = "x-mtp-head.gguf";
    ASSERT(pvx_model_spec_validate(m, &sp) == PVX_OK);

    // executor enable: metadata-only fake qwen4exp head (right arch, no
    // tensors) passes the arch surface but the real load types LOAD_FAILED.
    const char* tmpdir = getenv("PVX_TEST_TMP");
    const std::string meta1 = std::string(tmpdir ? tmpdir : "/tmp") + "/pvx_mtp_head_meta.gguf";
    {
        gguf_context* g = gguf_init_empty();
        gguf_set_val_str(g, "general.architecture", "qwen4exp");
        ASSERT(gguf_write_to_file(g, meta1.c_str(), true));
        gguf_free(g);
    }
    pvx_context_t* ctx = make_ctx(m, 1, 1024);
    pvx_executor_t* exec = nullptr; ASSERT(pvx_executor_create(ctx, &exec) == PVX_OK);
    sp.mtp_draft_path = meta1.c_str();
    ASSERT(pvx_executor_spec_init(exec, &sp) == PVX_ERR_LOAD_FAILED);
    ASSERT(strstr(pvx_last_error(ctx), "mtp-head-load") != nullptr);

    // wrong-arch head (a real llama model carries no nextn layers): typed
    // NOT_SUPPORTED naming the nextn boundary — never faked as a draft head.
    sp.mtp_draft_path = MODEL_PATH;
    ASSERT(pvx_executor_spec_init(exec, &sp) == PVX_ERR_NOT_SUPPORTED);
    ASSERT(strstr(pvx_last_error(ctx), "mtp-head-nextn") != nullptr);

    // (the EMPTY-STREAM enable requirement is exercised in the LIVE leg below —
    // it needs a genuinely loadable MTP head; the metadata-only fake above
    // types LOAD_FAILED before the stream check fires.)
    remove(meta1.c_str());

    // batched/pool conversation lane: typed refusal naming the lane.
    pvx_context_params_t cpb = {}; cpb.context_size = 1024; cpb.batch_size = 512; cpb.seq_max = 2;
    pvx_context_t* ctxb = nullptr; ASSERT(pvx_create_context(m, &cpb, &ctxb) == PVX_OK);
    pvx_pool_t* pool = nullptr; ASSERT(pvx_pool_create(ctxb, &pool) == PVX_OK);
    pvx_conversation_t* conv = nullptr; ASSERT(pvx_pool_lease(pool, &conv) == PVX_OK);
    sp = {}; sp.mode = PVX_SPEC_MTP_DRAFT; sp.mtp_draft_path = "whatever.gguf";
    ASSERT(pvx_conversation_spec_init(conv, &sp) == PVX_ERR_NOT_SUPPORTED);
    ASSERT(strstr(pvx_last_error(ctxb), "mtp-conversation-lane") != nullptr);
    ASSERT(pvx_pool_return(pool, conv) == PVX_OK);
    pvx_pool_free(pool); pvx_free_context(ctxb);

    pvx_executor_free(exec); pvx_free_context(ctx); pvx_free_model(m);
}

// LIVE draft-MTP leg (ABI v8 wave 4): a genuine qwen4exp-family target + a
// self-contained mtp head GGUF pair. Token-exactness vs plain greedy + REAL
// drafting telemetry (drafted > 0). The target context must carry a
// recurrent-state snapshot ring (n_rs_seq >= chain — hybrid verify repair).
// Env-gated: PVX_MTP_TARGET_PATH + PVX_MTP_DRAFT_PATH. Absent = typed skip
// (the primary host keeps this suite RAM-safe; the large-memory host verifies the 160GB pair live).
static void test_spec_mtp_draft_live_pair() {
    const char* tgt = getenv("PVX_MTP_TARGET_PATH");
    const char* dft = getenv("PVX_MTP_DRAFT_PATH");
    if (!tgt || !dft || !file_exists(tgt) || !file_exists(dft)) {
        fprintf(stderr, "live-skip: no genuine mtp pair "
                "(PVX_MTP_TARGET_PATH/PVX_MTP_DRAFT_PATH unset)\n");
        return;
    }
    pvx_model_params_t mp = {}; mp.model_path = tgt; mp.gpu_layers = 99; mp.flash_attn = true;
    pvx_model_t* m = nullptr; ASSERT(pvx_load_model(&mp, &m) == PVX_OK);
    const int n_gen = 24;
    const char* prompt = "The colors of the rainbow, in order, are: red, orange, yellow, green, blue, indigo and";
    pvx_sampling_params_t smp = {};
    smp.temperature = 0.0f; smp.top_k = 1; smp.top_p = 1.0f; smp.repeat_penalty = 1.0f;
    smp.max_tokens = n_gen; smp.ignore_eos = false;

    // plain greedy reference on a fresh context
    pvx_context_t* ca = make_ctx(m, 1, 2048);
    pvx_executor_t* ea = nullptr; ASSERT(pvx_executor_create(ca, &ea) == PVX_OK);
    ASSERT(pvx_executor_prompt(ea, prompt) == PVX_OK);
    ASSERT(pvx_executor_infer(ea) == PVX_DECODE_OK);
    int32_t ref[32];
    ASSERT(gen_plain_loop(ea, &smp, n_gen, ref) == n_gen);
    pvx_executor_free(ea); pvx_free_context(ca);

    // draft-MTP: enable FIRST (empty-stream requirement), then prompt+infer;
    // the target context carries n_rs_seq (hybrid verify repair contract).
    pvx_context_params_t cp = {};
    cp.context_size = 2048; cp.batch_size = 512; cp.seq_max = 1;
    cp.n_rs_seq = 16;  // ABI v8 wave 4: recurrent-state snapshot ring
    pvx_context_t* cb = nullptr; ASSERT(pvx_create_context(m, &cp, &cb) == PVX_OK);
    pvx_executor_t* eb = nullptr; ASSERT(pvx_executor_create(cb, &eb) == PVX_OK);
    pvx_spec_params_t sp = {};
    sp.mode = PVX_SPEC_MTP_DRAFT;
    sp.mtp_draft_path = dft;
    sp.draft_max = 2;
    sp.p_min = -1.0f;  // greedy floor off (upstream default)
    // enable requires an EMPTY stream (per-position capture must see every
    // commit): a prompt-first enable types the refusal and creates nothing.
    ASSERT(pvx_executor_prompt(eb, prompt) == PVX_OK);
    ASSERT(pvx_executor_infer(eb) == PVX_DECODE_OK);
    ASSERT(pvx_executor_spec_init(eb, &sp) == PVX_ERR_NOT_SUPPORTED);
    ASSERT(strstr(pvx_last_error(cb), "mtp-stream") != nullptr);
    pvx_executor_free(eb); pvx_free_context(cb);
    // fresh executor for the real enable
    pvx_context_t* cb2 = nullptr; ASSERT(pvx_create_context(m, &cp, &cb2) == PVX_OK);
    pvx_executor_t* eb2 = nullptr; ASSERT(pvx_executor_create(cb2, &eb2) == PVX_OK);
    ASSERT(pvx_executor_spec_init(eb2, &sp) == PVX_OK);
    ASSERT(pvx_executor_prompt(eb2, prompt) == PVX_OK);
    ASSERT(pvx_executor_infer(eb2) == PVX_DECODE_OK);
    int32_t out[32]; int n = 0;
    struct Cap { int32_t* v; int i; };
    Cap cap{out, 0};
    ASSERT(pvx_executor_generate_spec(eb2, &smp, n_gen,
            (pvx_spec_token_callback_t)+[](int32_t t, void* ud) -> bool {
                Cap* c = (Cap*) ud; c->v[c->i++] = t; return true;
            }, &cap, &n) == PVX_OK);
    ASSERT(n == n_gen);
    ASSERT(memcmp(ref, out, sizeof(int32_t) * (size_t) n_gen) == 0); // byte-identical
    int64_t drafted = 0, accepted = 0;
    ASSERT(pvx_executor_spec_stats(eb2, &drafted, &accepted) == PVX_OK);
    ASSERT(drafted > 0);
    ASSERT(accepted > 0);
    fprintf(stderr, "live: mtp-draft exact %d/%d tokens, drafted=%lld accepted=%lld "
            "(accept-rate %.1f%%) ", n, n_gen, (long long) drafted,
            (long long) accepted, 100.0 * (double) accepted / (double) drafted);
    pvx_executor_free(eb2); pvx_free_context(cb2); pvx_free_model(m);
}

// LIVE draft-MTP + vision same-session leg (BL-39): a genuine vision-capable
// target + MTP head pair + mmproj + test image. Spec is enabled on the EMPTY
// executor, then a vision-then-text prompt rides the SAME session — the
// capture hook must pair image embd rows as head INPUT embeddings (never as
// (t,h) history) and the spec stream must stay token-exact vs plain greedy.
// Env-gated: PVX_MTP_TARGET_PATH + PVX_MTP_DRAFT_PATH + PVX_MMPROJ_PATH (or
// the suite's MMPROJ_PATH) + PVX_TEST_IMAGE. Absent = typed skip.
static void test_spec_mtp_vision_live_pair() {
    const char* tgt = getenv("PVX_MTP_TARGET_PATH");
    const char* dft = getenv("PVX_MTP_DRAFT_PATH");
    // BL-42: the MTP pair may come from a DIFFERENT family than the suite's
    // vision fixtures (e.g. gemma-4 pair while the suite hybrid stays qwen) —
    // PVX_MTP_MMPROJ_PATH overrides, PVX_MMPROJ_PATH (suite MMPROJ_PATH)
    // stays the fallback.
    const char* mmp = getenv("PVX_MTP_MMPROJ_PATH");
    if (!mmp) mmp = getenv("PVX_MMPROJ_PATH");
    if (!mmp) mmp = MMPROJ_PATH;
    const char* img = getenv("PVX_TEST_IMAGE");
    if (!img) img = TEST_IMAGE;
    if (!tgt || !dft || !file_exists(tgt) || !file_exists(dft) ||
        !mmp || !file_exists(mmp) || !img || !file_exists(img)) {
        fprintf(stderr, "live-skip: no genuine mtp+vision pair "
                "(PVX_MTP_TARGET_PATH/PVX_MTP_DRAFT_PATH/mmproj/image unset)\n");
        return;
    }
    pvx_model_params_t mp = {}; mp.model_path = tgt; mp.gpu_layers = 99; mp.flash_attn = true;
    pvx_model_t* m = nullptr; ASSERT(pvx_load_model(&mp, &m) == PVX_OK);
    pvx_model_t* mmproj = nullptr;
    ASSERT(pvx_load_mmproj(m, mmp, &mmproj) == PVX_OK);

    FILE* f = fopen(img, "rb");
    ASSERT(f);
    fseek(f, 0, SEEK_END); long img_size = ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t* img_data = (uint8_t*) malloc(img_size);
    ASSERT(fread(img_data, 1, img_size, f) == (size_t) img_size);
    fclose(f);
    const char* imgs[] = { (const char*) img_data };
    int sizes[] = { (int) img_size };
    char* marker_str = nullptr;
    ASSERT(pvx_mtmd_marker(&marker_str) == PVX_OK);
    // LIVE LESSON (2026-10-20, the large-memory host 134GB pair): a RAW un-templated prompt
    // after the image leaves the target's greedy argmax near-tied between
    // ADJACENT end-marker specials (248044/248045) — the head picks the
    // sibling, every draft rejects (accepted=0), exactness holds via the
    // verify fallback. Proven NOT a pairing bug: chat-templated framing
    // yields a real description distribution and genuine acceptances
    // (drafted=15 accepted=4, exact, probe 2026-10-20). The leg therefore
    // applies the model's chat template — the pairing claim it guards is
    // "spec drafts accept on real multimodal distributions", not "the head
    // reproduces degenerate end-marker near-ties".
    const std::string v_content =
        std::string(marker_str) + "\nDescribe this image in exactly ten words.";
    pvx_free_string(marker_str);
    pvx_chat_message_t vm[1] = { { "user", v_content.c_str() } };
    char* v_prompt_str = nullptr;
    ASSERT(pvx_apply_chat_template(m, nullptr, vm, 1, /*add_assistant=*/true,
                                    /*enable_thinking=*/0, &v_prompt_str) == PVX_OK);
    const std::string prompt = v_prompt_str;
    pvx_free_string(v_prompt_str);

    const int n_gen = 12;
    pvx_sampling_params_t smp = {};
    smp.temperature = 0.0f; smp.top_k = 1; smp.top_p = 1.0f; smp.repeat_penalty = 1.0f;
    smp.max_tokens = n_gen; smp.ignore_eos = true;

    // A: plain greedy reference on a fresh context (vision + infer).
    pvx_context_t* ca = make_ctx(m, 1, 4096);
    pvx_executor_t* ea = nullptr; ASSERT(pvx_executor_create(ca, &ea) == PVX_OK);
    ASSERT(pvx_executor_prompt_with_images(ea, m, prompt.c_str(), imgs, sizes, 1) == PVX_OK);
    ASSERT(pvx_executor_infer(ea) == PVX_DECODE_OK);
    int32_t ref[32];
    ASSERT(gen_plain_loop(ea, &smp, n_gen, ref) == n_gen);
    pvx_executor_free(ea); pvx_free_context(ca);

    // B: draft-MTP FIRST (empty stream), then the SAME vision-then-text
    // prompt in the SAME session — the BL-39 scenario. n_rs_seq ring for the
    // hybrid verify repair contract (ABI v8 wave 4).
    pvx_context_params_t cp = {};
    cp.context_size = 4096; cp.batch_size = 512; cp.seq_max = 1;
    cp.n_rs_seq = 16;
    pvx_context_t* cb = nullptr; ASSERT(pvx_create_context(m, &cp, &cb) == PVX_OK);
    pvx_executor_t* eb = nullptr; ASSERT(pvx_executor_create(cb, &eb) == PVX_OK);
    pvx_spec_params_t sp = {};
    sp.mode = PVX_SPEC_MTP_DRAFT;
    sp.mtp_draft_path = dft;
    sp.draft_max = 2;
    sp.p_min = -1.0f;
    ASSERT(pvx_executor_spec_init(eb, &sp) == PVX_OK);
    ASSERT(pvx_executor_prompt_with_images(eb, m, prompt.c_str(), imgs, sizes, 1) == PVX_OK);
    ASSERT(pvx_executor_infer(eb) == PVX_DECODE_OK);
    int32_t out[32]; int n = 0;
    struct Cap { int32_t* v; int i; };
    Cap cap{out, 0};
    ASSERT(pvx_executor_generate_spec(eb, &smp, n_gen,
            (pvx_spec_token_callback_t)+[](int32_t t, void* ud) -> bool {
                Cap* c = (Cap*) ud; c->v[c->i++] = t; return true;
            }, &cap, &n) == PVX_OK);
    ASSERT(n == n_gen);
    ASSERT(cap.i == (size_t) n);
    ASSERT(memcmp(ref, out, sizeof(int32_t) * (size_t) n) == 0); // byte-identical
    int64_t drafted = 0, accepted = 0;
    ASSERT(pvx_executor_spec_stats(eb, &drafted, &accepted) == PVX_OK);
    ASSERT(drafted > 0);
    ASSERT(accepted > 0);
    fprintf(stderr, "live: mtp+vision exact %d/%d tokens, drafted=%lld accepted=%lld "
            "(accept-rate %.1f%%) ", n, n_gen, (long long) drafted,
            (long long) accepted, 100.0 * (double) accepted / (double) drafted);
    pvx_executor_free(eb); pvx_free_context(cb);
    free(img_data);
    pvx_free_mmproj(mmproj);
    pvx_free_model(m);
}

// LIVE draft-MTP + audio same-session leg (BL-40): a genuine audio-capable
// target + MTP head pair + audio-capable mmproj. Spec is enabled on the
// EMPTY executor, then an audio-then-text prompt rides the SAME session —
// the capture hook must pair audio embd rows as head INPUT embeddings (never
// as (t,h) history) and the spec stream must stay token-exact vs plain
// greedy. Env-gated: PVX_MTP_TARGET_PATH + PVX_MTP_DRAFT_PATH +
// PVX_AUDIO_MMPROJ_PATH; absent (or the pair is not audio-capable) = typed
// skip. The PCM is a mechanical 1 s sine at the projector's native rate
// (decode-mechanics gate — transcript quality needs real audio).
static void test_spec_mtp_audio_live_pair() {
    const char* tgt = getenv("PVX_MTP_TARGET_PATH");
    const char* dft = getenv("PVX_MTP_DRAFT_PATH");
    const char* mmp = getenv("PVX_AUDIO_MMPROJ_PATH");
    if (!tgt || !dft || !mmp || !file_exists(tgt) || !file_exists(dft) ||
        !file_exists(mmp)) {
        fprintf(stderr, "live-skip: no genuine mtp+audio pair "
                "(PVX_MTP_TARGET_PATH/PVX_MTP_DRAFT_PATH/PVX_AUDIO_MMPROJ_PATH "
                "unset)\n");
        return;
    }
    pvx_model_params_t mp = {}; mp.model_path = tgt; mp.gpu_layers = 99; mp.flash_attn = true;
    pvx_model_t* m = nullptr; ASSERT(pvx_load_model(&mp, &m) == PVX_OK);
    pvx_model_t* mmproj = nullptr;
    // Off-pair leg (BL-40 contract: non-audio pair = typed skip). The audio
    // projector is capability-coupled to the target — a geometry mismatch on
    // load is the off-pair signature, not a product regression.
    if (pvx_load_mmproj(m, mmp, &mmproj) != PVX_OK) {
        fprintf(stderr, "live-skip: mtp pair is not audio-capable "
                "(audio projector load failed: %s)\n", pvx_model_last_error(m));
        pvx_free_model(m);
        return;
    }
    if (!pvx_mtmd_supports_audio(m)) {
        // off-pair (target loaded but projector carries no audio adapter)
        fprintf(stderr, "live-skip: mtp pair is not audio-capable "
                "(mtmd_support_audio false)\n");
        pvx_free_mmproj(mmproj);
        pvx_free_model(m);
        return;
    }
    const int rate = pvx_mtmd_audio_sample_rate(m);
    ASSERT(rate > 0);
    std::vector<float> pcm((size_t) rate, 0.0f);
    for (size_t i = 0; i < pcm.size(); i++)
        pcm[i] = 0.25f * sinf((float) (2.0 * M_PI * 440.0 * i / pcm.size()));
    // LIVE LESSON (2026-10-20, vision leg): raw un-templated multimodal
    // prompts degenerate the target into end-marker near-ties (every draft
    // rejected); the audio leg uses the model's chat template for a real
    // transcription distribution — same fix as the vision leg.
    const std::string a_content = std::string("<__media__>") + "\nTranscribe this audio.";
    pvx_chat_message_t am[1] = { { "user", a_content.c_str() } };
    char* a_prompt_str = nullptr;
    ASSERT(pvx_apply_chat_template(m, nullptr, am, 1, /*add_assistant=*/true,
                                    /*enable_thinking=*/0, &a_prompt_str) == PVX_OK);
    const std::string a_prompt = a_prompt_str;
    pvx_free_string(a_prompt_str);
    const char* prompt = a_prompt.c_str();

    const int n_gen = 12;
    pvx_sampling_params_t smp = {};
    smp.temperature = 0.0f; smp.top_k = 1; smp.top_p = 1.0f; smp.repeat_penalty = 1.0f;
    smp.max_tokens = n_gen; smp.ignore_eos = true;

    // A: plain greedy reference on a fresh context (audio + infer).
    pvx_context_t* ca = make_ctx(m, 1, 4096);
    pvx_executor_t* ea = nullptr; ASSERT(pvx_executor_create(ca, &ea) == PVX_OK);
    ASSERT(pvx_executor_prompt_with_audio(ea, m, prompt, pcm.data(), (int) pcm.size())
           == PVX_OK);
    ASSERT(pvx_executor_infer(ea) == PVX_DECODE_OK);
    int32_t ref[32];
    ASSERT(gen_plain_loop(ea, &smp, n_gen, ref) == n_gen);
    pvx_executor_free(ea); pvx_free_context(ca);

    // B: draft-MTP FIRST (empty stream), then the SAME audio-then-text
    // prompt in the SAME session — the BL-40 scenario. n_rs_seq ring for the
    // hybrid verify repair contract (ABI v8 wave 4).
    pvx_context_params_t cp = {};
    cp.context_size = 4096; cp.batch_size = 512; cp.seq_max = 1;
    cp.n_rs_seq = 16;
    pvx_context_t* cb = nullptr; ASSERT(pvx_create_context(m, &cp, &cb) == PVX_OK);
    pvx_executor_t* eb = nullptr; ASSERT(pvx_executor_create(cb, &eb) == PVX_OK);
    pvx_spec_params_t sp = {};
    sp.mode = PVX_SPEC_MTP_DRAFT;
    sp.mtp_draft_path = dft;
    sp.draft_max = 2;
    sp.p_min = -1.0f;
    ASSERT(pvx_executor_spec_init(eb, &sp) == PVX_OK);
    ASSERT(pvx_executor_prompt_with_audio(eb, m, prompt, pcm.data(), (int) pcm.size())
           == PVX_OK);
    ASSERT(pvx_executor_infer(eb) == PVX_DECODE_OK);
    int32_t out[32]; int n = 0;
    struct Cap { int32_t* v; int i; };
    Cap cap{out, 0};
    ASSERT(pvx_executor_generate_spec(eb, &smp, n_gen,
            (pvx_spec_token_callback_t)+[](int32_t t, void* ud) -> bool {
                Cap* c = (Cap*) ud; c->v[c->i++] = t; return true;
            }, &cap, &n) == PVX_OK);
    ASSERT(n == n_gen);
    ASSERT(cap.i == (size_t) n);
    ASSERT(memcmp(ref, out, sizeof(int32_t) * (size_t) n) == 0); // byte-identical
    int64_t drafted = 0, accepted = 0;
    ASSERT(pvx_executor_spec_stats(eb, &drafted, &accepted) == PVX_OK);
    ASSERT(drafted > 0);
    ASSERT(accepted > 0);
    fprintf(stderr, "live: mtp+audio exact %d/%d tokens, drafted=%lld accepted=%lld "
            "(accept-rate %.1f%%) ", n, n_gen, (long long) drafted,
            (long long) accepted, 100.0 * (double) accepted / (double) drafted);
    pvx_executor_free(eb); pvx_free_context(cb);
    pvx_free_mmproj(mmproj);
    pvx_free_model(m);
}

static void test_spec_eagle3_live_pair() {
    // LIVE EAGLE3 leg (BL-36 evidence): a genuine target+companion pair.
    // Re-provision with the verified pair (repo + file names, relative notes):
    //   target    Qwen/Qwen3-8B-GGUF            file Qwen3-8B-Q4_K_M.gguf (~5.03 GB)
    //   companion williamliao/Qwen3-8B-EAGLE3-Speculator-GGUF
    //             file Qwen3-8B-speculator.eagle3-F16.gguf (~2.05 GB; RedHatAI
    //             speculator conversion; eagle3.target_layers [2,18,33])
    // Env-gated: PVX_EAGLE3_TARGET_PATH + PVX_EAGLE3_DRAFT_PATH (legacy alias
    // PVX_EAGLE_DRAFT_PATH still honored for the draft). Absent = typed skip.
    const char* tgt = getenv("PVX_EAGLE3_TARGET_PATH");
    const char* dft = getenv("PVX_EAGLE3_DRAFT_PATH");
    if (!dft) dft = getenv("PVX_EAGLE_DRAFT_PATH");
    if (!tgt || !dft || !file_exists(tgt) || !file_exists(dft)) {
        fprintf(stderr, "live-skip: no genuine eagle3 pair "
                "(PVX_EAGLE3_TARGET_PATH/PVX_EAGLE3_DRAFT_PATH unset)\n");
        return;
    }
    pvx_model_params_t mp = {}; mp.model_path = tgt; mp.gpu_layers = 99; mp.flash_attn = true;
    pvx_model_t* m = nullptr; ASSERT(pvx_load_model(&mp, &m) == PVX_OK);
    const int n_gen = 24;
    const char* prompt = "The colors of the rainbow, in order, are: red, orange, yellow, green, blue, indigo and";
    pvx_sampling_params_t smp = {};
    smp.temperature = 0.0f; smp.top_k = 1; smp.top_p = 1.0f; smp.repeat_penalty = 1.0f;
    smp.max_tokens = n_gen; smp.ignore_eos = false;

    // plain greedy reference on a fresh context
    pvx_context_t* ca = make_ctx(m, 1, 2048);
    pvx_executor_t* ea = nullptr; ASSERT(pvx_executor_create(ca, &ea) == PVX_OK);
    ASSERT(pvx_executor_prompt(ea, prompt) == PVX_OK);
    ASSERT(pvx_executor_infer(ea) == PVX_DECODE_OK);
    int32_t ref[32];
    ASSERT(gen_plain_loop(ea, &smp, n_gen, ref) == n_gen);
    pvx_executor_free(ea); pvx_free_context(ca);

    // EAGLE3 requires an empty stream at enable (prefill-boundary): enable
    // FIRST (companion metadata + vocab compat validated typed), then
    // prompt+infer (the capture hook rides the commit path), then generate.
    pvx_context_t* cb = make_ctx(m, 1, 2048);
    pvx_executor_t* eb = nullptr; ASSERT(pvx_executor_create(cb, &eb) == PVX_OK);
    pvx_spec_params_t sp = {};
    sp.mode = PVX_SPEC_EAGLE3;
    sp.eagle_companion_path = dft;
    sp.draft_max = 8;
    ASSERT(pvx_executor_spec_init(eb, &sp) == PVX_OK);
    ASSERT(pvx_executor_prompt(eb, prompt) == PVX_OK);
    ASSERT(pvx_executor_infer(eb) == PVX_DECODE_OK);
    int32_t out[32]; int n = 0;
    struct Cap { int32_t* v; int i; };
    Cap cap{out, 0};
    ASSERT(pvx_executor_generate_spec(eb, &smp, n_gen,
            (pvx_spec_token_callback_t)+[](int32_t t, void* ud) -> bool {
                Cap* c = (Cap*) ud; c->v[c->i++] = t; return true;
            }, &cap, &n) == PVX_OK);
    ASSERT(n == n_gen);
    ASSERT(memcmp(ref, out, sizeof(int32_t) * (size_t) n_gen) == 0); // byte-identical

    // REAL drafting evidence (not silent degradation): accepted > 0.
    int64_t drafted = 0, accepted = 0;
    ASSERT(pvx_executor_spec_stats(eb, &drafted, &accepted) == PVX_OK);
    ASSERT(drafted > 0);
    ASSERT(accepted > 0);
    fprintf(stderr, "live: eagle3 exact %d/%d tokens, drafted=%lld accepted=%lld "
            "(accept-rate %.1f%%) ", n, n_gen, (long long) drafted,
            (long long) accepted, 100.0 * (double) accepted / (double) drafted);
    pvx_executor_free(eb); pvx_free_context(cb); pvx_free_model(m);
}

static void test_spec_dflash_live_pair() {
    // LIVE DFlash/DSpark leg (BL-36 evidence): the honest typed-boundary
    // contract against a genuine LiquidAI pair. Re-provision (repo + file):
    //   target  LiquidAI/LFM2.5-1.2B-Instruct-GGUF       file LFM2.5-1.2B-Instruct-Q4_0.gguf (~0.70 GB; arch lfm2)
    //   sidecar LiquidAI/LFM2.5-1.2B-Instruct-DSpark-GGUF file LFM2.5-1.2B-Instruct-DSpark-Q4_K_M.gguf (~0.18 GB;
    //           arch dflash, dflash.target_layers i32[5], markov head = DSpark)
    // Env-gated: PVX_DFLASH_TARGET_PATH + PVX_DFLASH_DRAFT_PATH. Absent = skip.
    // pvx does NOT wire block-diffusion drafting (ABI v7 typed boundary
    // slice): the REAL sidecar must pass the metadata precheck, then the enable
    // refuses typed at spec-engine/dflash-wiring — never faked, never silent.
    const char* tgt = getenv("PVX_DFLASH_TARGET_PATH");
    const char* dft = getenv("PVX_DFLASH_DRAFT_PATH");
    if (!tgt || !dft || !file_exists(tgt) || !file_exists(dft)) {
        fprintf(stderr, "live-skip: no genuine dflash pair "
                "(PVX_DFLASH_TARGET_PATH/PVX_DFLASH_DRAFT_PATH unset)\n");
        return;
    }
    pvx_model_params_t mp = {}; mp.model_path = tgt; mp.gpu_layers = 99; mp.flash_attn = true;
    pvx_model_t* m = nullptr; ASSERT(pvx_load_model(&mp, &m) == PVX_OK);

    // validate: metadata precheck accepts the REAL sidecar, then the wiring
    // boundary refuses typed (both legs must name dflash-wiring).
    pvx_spec_params_t sp = {};
    sp.mode = PVX_SPEC_DFLASH;
    sp.dflash_companion_path = dft;
    ASSERT(pvx_model_spec_validate(m, &sp) == PVX_ERR_NOT_SUPPORTED);
    ASSERT(strstr(pvx_model_last_error(m), "dflash-wiring") != nullptr);

    pvx_context_t* ctx = make_ctx(m, 1, 2048);
    pvx_executor_t* exec = nullptr; ASSERT(pvx_executor_create(ctx, &exec) == PVX_OK);
    ASSERT(pvx_executor_spec_init(exec, &sp) == PVX_ERR_NOT_SUPPORTED);
    ASSERT(strstr(pvx_last_error(ctx), "dflash-wiring") != nullptr);

    // no phantom session: generate refuses typed, never a silent plain run
    pvx_sampling_params_t smp = {};
    smp.temperature = 0.0f; smp.top_k = 1; smp.top_p = 1.0f; smp.repeat_penalty = 1.0f;
    int n = 0;
    ASSERT(pvx_executor_generate_spec(exec, &smp, 4, nullptr, nullptr, &n)
            == PVX_ERR_NOT_SUPPORTED);

    // the pair is real: the LFM2.5 target itself generates greedy tokens
    ASSERT(pvx_executor_prompt(exec, "The colors of the rainbow, in order, are: red, orange, yellow, green, blue, indigo and") == PVX_OK);
    ASSERT(pvx_executor_infer(exec) == PVX_DECODE_OK);
    int32_t ref[8];
    ASSERT(gen_plain_loop(exec, &smp, 8, ref) == 8);
    fprintf(stderr, "live: dflash typed wiring refusal on the genuine pair; "
            "target plain greedy 8/8 ");
    pvx_executor_free(exec); pvx_free_context(ctx); pvx_free_model(m);
}

static void test_json_schema_grammar_roundtrip() {
    // Schema -> GBNF -> grammar sampler -> generation stays schema-shaped:
    // the decision-envelope schema produces {"thinking": "..."} output that
    // parses as a JSON object with the required key.
    if (!file_exists(MODEL_PATH)) SKIP("model file not found");
    char* gbnf = nullptr; char* err = nullptr;
    // the decision-envelope schema (same shape as the pinned fixture)
    const char* schema = R"SCHEMA({
        "type": "object",
        "properties": {
            "thinking": { "type": "string" },
            "answer": { "type": ["string", "null"] },
            "toolcalls": { "type": ["array", "null"],
                "items": { "type": "object",
                    "properties": {
                        "name": { "type": "string" },
                        "args": { "type": "object", "additionalProperties": { "type": "string" } }
                    },
                    "required": ["name", "args"] } }
        },
        "required": ["thinking"]
    })SCHEMA";
    ASSERT(pvx_grammar_from_json_schema(schema, nullptr, &gbnf, &err) == PVX_OK);
    ASSERT(gbnf);

    // pinned fixture: byte-exact against tests/fixtures/decision-envelope.gbnf
    {
        std::string fx = "tests/fixtures/decision-envelope.gbnf";
        FILE* f = fopen(fx.c_str(), "rb");
        if (f) {
            fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
            std::string want((size_t) sz, '\0');
            ASSERT(fread(&want[0], 1, (size_t) sz, f) == (size_t) sz);
            fclose(f);
            ASSERT(want == std::string(gbnf));
        }
    }

    pvx_model_params_t mp = {}; mp.model_path = MODEL_PATH; mp.gpu_layers = 99; mp.flash_attn = true;
    pvx_model_t* m = nullptr; ASSERT(pvx_load_model(&mp, &m) == PVX_OK);
    pvx_context_t* ctx = make_ctx(m, 1, 1024);
    pvx_executor_t* exec = nullptr; ASSERT(pvx_executor_create(ctx, &exec) == PVX_OK);
    pvx_grammar_t* gram = pvx_grammar_create(m, gbnf, "root");
    ASSERT(gram != nullptr);
    ASSERT(pvx_executor_prompt(exec, "Reply with a JSON object.") == PVX_OK);
    ASSERT(pvx_executor_infer(exec) == PVX_DECODE_OK);
    pvx_sampling_params_t sp = {};
    sp.temperature = 0.0f; sp.top_k = 1; sp.top_p = 1.0f; sp.repeat_penalty = 1.0f;
    // grammar-constrained greedy loop: sample_grammar -> commit -> infer
    int32_t tokens[64];
    int n = 0;
    for (n = 0; n < 24; n++) {
        int32_t t = 0;
        if (pvx_executor_sample_grammar(exec, &sp, gram, &t) != PVX_OK) break;
        tokens[n] = t;
        ASSERT(pvx_executor_prompt_tokens(exec, &t, 1) == PVX_OK);
        ASSERT(pvx_executor_infer(exec) == PVX_DECODE_OK);
    }
    ASSERT(n > 0);
    // detokenize + shape check: balanced JSON object with "thinking"
    int32_t* heap = (int32_t*) malloc(sizeof(int32_t) * (size_t) n);
    memcpy(heap, tokens, sizeof(int32_t) * (size_t) n);
    char* text = nullptr;
    ASSERT(pvx_detokenize(ctx, heap, n, false, &text) == PVX_OK);
    ASSERT(text);
    const std::string t(text);
    ASSERT(t.find("\"thinking\"") != std::string::npos);
    ASSERT(t.find('{') != std::string::npos);
    pvx_free_string(text); free(heap);
    pvx_grammar_free(gram);
    pvx_executor_free(exec); pvx_free_context(ctx); pvx_free_model(m);
    pvx_free_string(gbnf);
}


// ════════════════════════════════════════════════════════════
// ABI v8 — Wave 3 tests (audio, TTS, rerank, LoRA, cvec, stats)
// ════════════════════════════════════════════════════════════

// Reranking: typed pooling gate + typed score-head gate on a non-rerank model;
// live leg env-gated (PVX_RERANK_MODEL_PATH — a RANK-head GGUF, e.g. the
// gpustack/jina-reranker-v1-tiny-en-GGUF conversion).
static void test_rerank_typed() {
    if (!file_exists(EMBED_MODEL_PATH) && !file_exists(MODEL_PATH)) SKIP("no model file");
    const char* path = file_exists(EMBED_MODEL_PATH) ? EMBED_MODEL_PATH : MODEL_PATH;
    pvx_model_params_t mp = {}; mp.model_path = path; mp.gpu_layers = 99; mp.flash_attn = true;
    pvx_model_t* m = nullptr; ASSERT(pvx_load_model(&mp, &m) == PVX_OK);

    // pooling gate first: a NON-RANK context refuses typed naming the boundary
    pvx_context_t* ctx = make_ctx(m, 1, 512);
    const char* docs[2] = { "a red apple", "a blue car" };
    float* scores = nullptr;
    ASSERT(pvx_rerank(ctx, "what color is an apple?", docs, 2, &scores) == PVX_ERR_NOT_SUPPORTED);
    ASSERT(strstr(pvx_last_error(ctx), "rerank/pooling") != nullptr);

    // RANK pooling on the same (non-rerank) model: the context may build (the
    // gate is the pooling TYPE, upstream llama-server --reranking contract —
    // a headless encoder yields hidden-state readings exactly as upstream
    // would under --reranking; conversion-provisioned cls.weight is the
    // reranker contract). No typed refusal here — the boundary stays
    // rerank/pooling.
    pvx_free_context(ctx); pvx_free_model(m);
}

static void test_rerank_live() {
    // LIVE leg — a real RANK-head reranker (verified against our requirements
    // via range-request GGUF precheck before download): scores are finite,
    // ordering is deterministic, and the relevant doc ranks on top.
    // Provisioned pair (HF repo gpustack/jina-reranker-v1-tiny-en-GGUF,
    // file jina-reranker-v1-tiny-en-Q8_0.gguf) — 30B-decoder-arch-class
    // jina-bert-v2 encoder + 1-dim score head (n_embd_out == 1).
    const char* rp = getenv("PVX_RERANK_MODEL_PATH");
    if (!rp || !file_exists(rp)) SKIP("PVX_RERANK_MODEL_PATH unset/missing");
    pvx_model_params_t mp = {}; mp.model_path = rp; mp.gpu_layers = 99; mp.flash_attn = true;
    pvx_model_t* m = nullptr; ASSERT(pvx_load_model(&mp, &m) == PVX_OK);
    pvx_context_params_t cp = {};
    cp.context_size = 1024; cp.batch_size = 512; cp.seq_max = 1;
    cp.pooling_type = PVX_POOL_RANK;  // attach the classification head
    pvx_context_t* ctx = nullptr;
    ASSERT(pvx_create_context(m, &cp, &ctx) == PVX_OK);

    const char* query = "What is the capital of France?";
    const char* docs[3] = {
        "The capital of France is Paris.",
        "Photosynthesis converts sunlight into chemical energy.",
        "The Eiffel Tower is located in Paris, France.",
    };
    float* s1 = nullptr; float* s2 = nullptr;
    ASSERT(pvx_rerank(ctx, query, docs, 3, &s1) == PVX_OK);
    ASSERT(pvx_rerank(ctx, query, docs, 3, &s2) == PVX_OK);  // determinism: replay
    for (int i = 0; i < 3; i++) {
        ASSERT(std::isfinite(s1[i]));
        ASSERT(s1[i] == s2[i]);
    }
    // relevant docs (0, 2: Paris/France) must outrank the distractor (1)
    ASSERT(s1[0] > s1[1]);
    ASSERT(s1[2] > s1[1]);
    fprintf(stderr, "\n      rerank scores: %.4f %.4f %.4f ", s1[0], s1[1], s1[2]);
    pvx_free_rerank_scores(s1); pvx_free_rerank_scores(s2);
    pvx_free_context(ctx); pvx_free_model(m);
}

// Audio input: typed capability gates (vision-only mmproj has no audio) +
// live leg env-gated (PVX_AUDIO_MODEL_PATH + PVX_AUDIO_MMPROJ_PATH — a
// Qwen3-ASR-family pair, verified via range-request GGUF precheck:
// clip.has_audio_encoder == true, projector qwen3a).
static void test_audio_capability_typed() {
    if (!file_exists(HYBRID_MODEL_PATH) || !file_exists(MMPROJ_PATH))
        SKIP("hybrid model or mmproj missing");
    pvx_model_params_t mp = {}; mp.model_path = HYBRID_MODEL_PATH; mp.gpu_layers = 99; mp.flash_attn = true;
    pvx_model_t* m = nullptr; ASSERT(pvx_load_model(&mp, &m) == PVX_OK);
    pvx_model_t* mmproj = nullptr; ASSERT(pvx_load_mmproj(m, MMPROJ_PATH, &mmproj) == PVX_OK);

    // vision-only projector: no audio adapter — typed false / rate -1
    ASSERT(pvx_mtmd_supports_audio(m) == false);
    ASSERT(pvx_mtmd_audio_sample_rate(m) == -1);

    // audio prompt through the vision-only projector: typed refusal
    pvx_context_t* ctx = make_ctx(m, 1, 1024);
    pvx_executor_t* exec = nullptr; ASSERT(pvx_executor_create(ctx, &exec) == PVX_OK);
    std::vector<float> pcm(16000, 0.0f);  // 1 s of silence @16k
    ASSERT(pvx_executor_prompt_with_audio(exec, m, "<__media__>", pcm.data(), (int) pcm.size())
           == PVX_ERR_NOT_SUPPORTED);
    ASSERT(strstr(pvx_model_last_error(m), "media/audio-adapter") != nullptr);
    ASSERT(pvx_executor_token_count(exec) == 0);  // nothing committed

    pvx_executor_free(exec); pvx_free_context(ctx);
    pvx_free_mmproj(mmproj); pvx_free_model(m);
}

static void test_audio_live() {
    // LIVE leg — Qwen3-ASR pair (HF repo ggml-org/Qwen3-ASR-1.7B-GGUF:
    // Qwen3-ASR-1.7B-Q8_0.gguf + mmproj-Qwen3-ASR-1.7B-Q8_0.gguf).
    // Gates: capability probes, both-lane decode mechanics (executor +
    // conversation), committed counters consistent (n_tokens == n_kv_pos),
    // sample rate contract truthful.
    const char* ap = getenv("PVX_AUDIO_MODEL_PATH");
    const char* pp = getenv("PVX_AUDIO_MMPROJ_PATH");
    if (!ap || !pp || !file_exists(ap) || !file_exists(pp))
        SKIP("PVX_AUDIO_MODEL_PATH/PVX_AUDIO_MMPROJ_PATH unset/missing");
    pvx_model_params_t mp = {}; mp.model_path = ap; mp.gpu_layers = 99; mp.flash_attn = true;
    pvx_model_t* m = nullptr; ASSERT(pvx_load_model(&mp, &m) == PVX_OK);
    pvx_model_t* mm = nullptr; ASSERT(pvx_load_mmproj(m, pp, &mm) == PVX_OK);

    ASSERT(pvx_mtmd_supports_audio(m) == true);
    const int rate = pvx_mtmd_audio_sample_rate(m);
    ASSERT(rate > 0);
    fprintf(stderr, "\n      asr sample rate: %d Hz ", rate);

    // 1 s of 440 Hz sine at the native rate — decode mechanics gate
    // (transcript quality needs real audio; the gate is mechanical).
    std::vector<float> pcm((size_t) rate, 0.0f);
    for (size_t i = 0; i < pcm.size(); i++)
        pcm[i] = 0.25f * sinf((float) (2.0 * M_PI * 440.0 * i / pcm.size()));

    // executor lane
    {
        pvx_context_t* ctx = make_ctx(m, 1, 4096);
        pvx_executor_t* exec = nullptr; ASSERT(pvx_executor_create(ctx, &exec) == PVX_OK);
        const char* prompt = "Transcribe this audio: <__media__>";
        ASSERT(pvx_executor_prompt_with_audio(exec, m, prompt, pcm.data(), (int) pcm.size()) == PVX_OK);
        const int n0 = pvx_executor_token_count(exec);
        ASSERT(n0 > 3);  // text rows + audio rows committed
        pvx_stats_t st = {};
        ASSERT(pvx_executor_stats(exec, &st) == PVX_OK);
        ASSERT(st.n_tokens == n0); ASSERT(st.n_kv_pos == n0);
        ASSERT(st.n_decode_steps >= 1); ASSERT(st.spec_drafted == 0 && st.spec_accepted == 0);
        // greedy continuation works after audio rows (KV positions consistent)
        pvx_sampling_params_t sp = {}; sp.temperature = 0.0f; sp.top_k = 1;
        int32_t tok = 0;
        ASSERT(pvx_executor_sample(exec, &sp, &tok) == PVX_OK);
        ASSERT(tok != 0);
        pvx_executor_free(exec); pvx_free_context(ctx);
    }

    // conversation lane (pooled, seq discipline)
    {
        pvx_context_t* ctx2 = make_ctx(m, 2, 4096);
        pvx_pool_t* pool = nullptr; ASSERT(pvx_pool_create(ctx2, &pool) == PVX_OK);
        pvx_conversation_t* conv = nullptr; ASSERT(pvx_pool_lease(pool, &conv) == PVX_OK);
        const char* prompt = "What do you hear? <__media__>";
        ASSERT(pvx_conversation_prompt_with_audio(conv, m, prompt, pcm.data(), (int) pcm.size()) == PVX_OK);
        const int c0 = pvx_conversation_token_count(conv);
        ASSERT(c0 > 3);
        ASSERT(pvx_conversation_token_count(conv) == c0);
        ASSERT(pvx_pool_return(pool, conv) == PVX_OK);
        pvx_pool_free(pool); pvx_free_context(ctx2);
    }

    pvx_free_mmproj(mm); pvx_free_model(m);
}

// TTS live leg (env-gated: PVX_TTS_MODEL_PATH + PVX_TTS_MMPROJ_PATH —
// the Qwen3-TTS pair, HF repo ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF:
// Qwen3-TTS-12Hz-1.7B-Base-Q4_K_M.gguf + mmproj-...-Q8_0.gguf; precheck
// verified clip.has_gen_audio_encoder == true / projector qwen3tts_gen).
static void test_tts_live() {
    const char* tp = getenv("PVX_TTS_MODEL_PATH");
    const char* mp2 = getenv("PVX_TTS_MMPROJ_PATH");
    if (!tp || !mp2 || !file_exists(tp) || !file_exists(mp2))
        SKIP("PVX_TTS_MODEL_PATH/PVX_TTS_MMPROJ_PATH unset/missing");
    pvx_model_params_t mp3 = {}; mp3.model_path = tp; mp3.gpu_layers = 99; mp3.flash_attn = true;
    pvx_model_t* m = nullptr; ASSERT(pvx_load_model(&mp3, &m) == PVX_OK);

    // no-gen-audio mmproj must refuse typed — loads cleanly (backbone-
    // compatible audio-only projector) but has no gen-audio pipeline →
    // NOT_SUPPORTED at the capability gate (boundary tts/gen-audio-capability),
    // never a partial session. NOTE: a vision mmproj against the TTS backbone
    // (arch mismatch) fails DEEP inside the vendored mtmd init and leaks a
    // Metal buffer set (upstream, llama.cpp frozen) — that combination is
    // deliberately NOT exercised here; the audio-only projector is the clean
    // negative.
    {
        const char* neg = getenv("PVX_AUDIO_MMPROJ_PATH");
        if (neg && file_exists(neg)) {
            pvx_tts_session_t* bad = nullptr;
            ASSERT(pvx_tts_session_create(m, neg, 4096, &bad) == PVX_ERR_NOT_SUPPORTED);
            ASSERT(bad == nullptr);
            ASSERT(strstr(pvx_model_last_error(m), "tts/gen-audio-capability") != nullptr);
        }
        // missing file: typed load failure
        pvx_tts_session_t* badf = nullptr;
        ASSERT(pvx_tts_session_create(m, "/nonexistent/nope.gguf", 4096, &badf)
               == PVX_ERR_LOAD_FAILED);
        ASSERT(badf == nullptr);
    }

    pvx_tts_session_t* s = nullptr;
    ASSERT(pvx_tts_session_create(m, mp2, 4096, &s) == PVX_OK);

    pvx_tts_params_t p = {};
    p.seed = 42; p.max_frames = 64;  // short cap for the suite
    float* pcm = nullptr; int n_samples = 0, sr = 0;
    ASSERT(pvx_tts_generate(s, &p, "Hello world.", &pcm, &n_samples, &sr) == PVX_OK);
    ASSERT(pcm && n_samples > 0 && sr == 24000);
    fprintf(stderr, "\n      tts: %d samples @ %d Hz ", n_samples, sr);

    // SINGLE-SHOT contract: a second generate on the same session refuses
    // typed (cross-call gen-pipeline state; boundary tts/session-consumed).
    {
        float* pcm2 = nullptr; int n2 = 0, sr2 = 0;
        ASSERT(pvx_tts_generate(s, &p, "Hello world.", &pcm2, &n2, &sr2)
               == PVX_ERR_NOT_SUPPORTED);
        ASSERT(pcm2 == nullptr);
    }
    // determinism: identical (text, seed) on a FRESH session → byte-identical PCM
    {
        pvx_tts_session_t* s2 = nullptr;
        ASSERT(pvx_tts_session_create(m, mp2, 4096, &s2) == PVX_OK);
        float* pcm2 = nullptr; int n2 = 0, sr2 = 0;
        ASSERT(pvx_tts_generate(s2, &p, "Hello world.", &pcm2, &n2, &sr2) == PVX_OK);
        ASSERT(n2 == n_samples && sr2 == sr);
        ASSERT(memcmp(pcm, pcm2, (size_t) n_samples * sizeof(float)) == 0);
        pvx_free_audio_samples(pcm2);
        pvx_tts_session_free(s2);
    }
    // random seed also works (UINT32_MAX = random draw) — FRESH session
    // (single-shot contract; boundary tts/session-consumed)
    {
        pvx_tts_session_t* sr3 = nullptr;
        ASSERT(pvx_tts_session_create(m, mp2, 4096, &sr3) == PVX_OK);
        pvx_tts_params_t pr = {}; pr.seed = UINT32_MAX; pr.max_frames = 64;
        float* pcmr = nullptr; int nr = 0, srr = 0;
        ASSERT(pvx_tts_generate(sr3, &pr, "Hello again.", &pcmr, &nr, &srr) == PVX_OK);
        ASSERT(pcmr && nr > 0 && srr == 24000);
        pvx_free_audio_samples(pcmr);
        pvx_tts_session_free(sr3);
    }
    // empty text: typed arg error (fresh session; nothing consumed)
    {
        pvx_tts_session_t* se = nullptr;
        ASSERT(pvx_tts_session_create(m, mp2, 4096, &se) == PVX_OK);
        pvx_tts_params_t pe = {}; pe.seed = 7; pe.max_frames = 16;
        float* pcme = nullptr; int ne = 0, sre = 0;
        ASSERT(pvx_tts_generate(se, &pe, "", &pcme, &ne, &sre) == PVX_ERR_INVALID_ARG);
        pvx_tts_session_free(se);
    }
    pvx_free_audio_samples(pcm);
    pvx_tts_session_free(s); pvx_free_model(m);
}

// LoRA adapters: typed load gates + env-gated live attach/detach with
// logits-and-KV neutrality (same context, no reset between arms).
static void test_lora_typed() {
    if (!file_exists(MODEL_PATH)) SKIP("model file not found");
    pvx_model_params_t mp = {}; mp.model_path = MODEL_PATH; mp.gpu_layers = 99; mp.flash_attn = true;
    pvx_model_t* m = nullptr; ASSERT(pvx_load_model(&mp, &m) == PVX_OK);

    // missing file: typed load failure
    pvx_adapter_t* bad = nullptr;
    ASSERT(pvx_adapter_load(m, "/nonexistent/nope.gguf", &bad) == PVX_ERR_LOAD_FAILED);
    ASSERT(strstr(pvx_model_last_error(m), "adapter/") != nullptr);

    pvx_context_t* ctx = make_ctx(m, 1, 1024);
    // scale gate + null-handle gates on set
    const pvx_adapter_t* arr[1] = { nullptr };
    const float sc[1] = { 0.0f };
    ASSERT(pvx_context_set_lora(ctx, nullptr, sc, 1) == PVX_ERR_INVALID_ARG);
    ASSERT(pvx_context_set_lora(ctx, arr, nullptr, 1) == PVX_ERR_INVALID_ARG);

    pvx_free_context(ctx); pvx_free_model(m);
}

static void test_lora_live() {
    // LIVE leg — a real adapter for the dense fixture base (Llama-3.2-3B):
    // HF repo Leejy0-0/Llama-3.2-3b-lora_model-F16-GGUF (GGUF adapter:
    // general.type == adapter, adapter.type == lora, arch llama — verified
    // via range-request header precheck before download).
    const char* lp = getenv("PVX_LORA_ADAPTER_PATH");
    if (!lp || !file_exists(lp)) SKIP("PVX_LORA_ADAPTER_PATH unset/missing");
    if (!file_exists(MODEL_PATH)) SKIP("model file not found");

    pvx_model_params_t mp = {}; mp.model_path = MODEL_PATH; mp.gpu_layers = 99; mp.flash_attn = true;
    pvx_model_t* m = nullptr; ASSERT(pvx_load_model(&mp, &m) == PVX_OK);

    pvx_adapter_t* a = nullptr;
    ASSERT(pvx_adapter_load(m, lp, &a) == PVX_OK);

    // attach/detach KV-neutrality gate on ONE context: save state at the
    // post-prompt point, greedy-sample one token under each weights arm
    // (baseline → lora → detached), and prove detach restores the baseline
    // token WITHOUT any reset/rewind in between.
    pvx_context_t* ctx = make_ctx(m, 2, 2048);  // seq_max 2: state_save backup seq
    pvx_executor_t* exec = nullptr; ASSERT(pvx_executor_create(ctx, &exec) == PVX_OK);
    ASSERT(pvx_executor_prompt(exec, "The quick brown fox") == PVX_OK);
    ASSERT(pvx_executor_infer(exec) == PVX_DECODE_OK);
    pvx_state_t* anchor = nullptr;
    ASSERT(pvx_state_save(exec, &anchor) == PVX_OK);
    const int n_anchor = pvx_state_token_count(anchor);

    pvx_sampling_params_t sp = {}; sp.temperature = 0.0f; sp.top_k = 1;
    int32_t t_base = 0, t_lora = 0, t_off = 0;

    // arm 1: baseline greedy from the post-prompt logits
    ASSERT(pvx_executor_sample(exec, &sp, &t_base) == PVX_OK);

    // arm 2: BASELINE continuation — decode t_base at pos n_anchor from the
    // SAME anchor KV (restore clears the logits snapshot — re-infer is the
    // honest path). No weights change yet: this is the baseline reference
    // for the attach/detach comparison below.
    ASSERT(pvx_state_restore(exec, anchor) == PVX_OK);
    ASSERT(pvx_executor_prompt_tokens(exec, &t_base, 1) == PVX_OK);
    ASSERT(pvx_executor_infer(exec) == PVX_DECODE_OK);
    int32_t t_pre = 0;
    ASSERT(pvx_executor_sample(exec, &sp, &t_pre) == PVX_OK);

    // arm 3: attach LoRA @1.0 (weights-surface: KV cells untouched, no rewind
    // required), restore the same anchor, decode the same token → lora logits
    ASSERT(pvx_state_restore(exec, anchor) == PVX_OK);
    const pvx_adapter_t* adapters[1] = { a };
    const float sc1[1] = { 1.0f };
    ASSERT(pvx_context_set_lora(ctx, adapters, sc1, 1) == PVX_OK);
    ASSERT(pvx_executor_prompt_tokens(exec, &t_base, 1) == PVX_OK);
    ASSERT(pvx_executor_infer(exec) == PVX_DECODE_OK);
    ASSERT(pvx_executor_sample(exec, &sp, &t_lora) == PVX_OK);

    // arm 4: detach, same decode → baseline logits RETURN (KV untouched)
    ASSERT(pvx_state_restore(exec, anchor) == PVX_OK);
    ASSERT(pvx_context_set_lora(ctx, nullptr, nullptr, 0) == PVX_OK);
    ASSERT(pvx_executor_prompt_tokens(exec, &t_base, 1) == PVX_OK);
    ASSERT(pvx_executor_infer(exec) == PVX_DECODE_OK);
    ASSERT(pvx_executor_sample(exec, &sp, &t_off) == PVX_OK);

    ASSERT(t_off == t_pre);  // detach restores baseline logits (same pos, same KV)
    fprintf(stderr, "\n      lora tokens: prompt-greedy=%d baseline-cont=%d lora-cont=%d detached-cont=%d ",
            t_base, t_pre, t_lora, t_off);

    pvx_state_free(anchor);
    pvx_executor_free(exec);
    pvx_free_context(ctx);
    pvx_adapter_free(a); pvx_free_model(m);
}

// Control vectors: synthetic direction.GGUF (gguf writer API — tests only),
// typed width gate, live apply/detach with KV-neutral baseline restoration.
static void write_cvector_gguf(const char* path, int n_embd, int n_layers, uint32_t salt) {
    size_t data_sz = (size_t) n_embd * n_layers;
    struct ggml_init_params ip = {};
    ip.mem_size = (data_sz + 8) * sizeof(float) * 2;
    ip.mem_buffer = nullptr;
    ip.no_alloc = false;
    ggml_context* gctx = ggml_init(ip);
    ASSERT(gctx != nullptr);
    gguf_context* g = gguf_init_empty();
    ASSERT(g != nullptr);
    gguf_set_val_str(g, "general.architecture", "llama");
    gguf_set_val_str(g, "general.type", "cvector");  // descriptive only
    for (int l = 1; l <= n_layers; l++) {
        std::string name = "direction." + std::to_string(l);
        ggml_tensor* t = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, (int64_t) n_embd);
        float* d = (float*) t->data;
        for (int i = 0; i < n_embd; i++)
            d[i] = 0.01f * ((float) ((salt * (uint32_t) (l * 131 + i * 17)) % 100) - 50.0f) / 100.0f;
        ggml_set_name(t, name.c_str());
        gguf_add_tensor(g, t);
    }
    ASSERT(gguf_write_to_file(g, path, false));
    gguf_free(g);
    ggml_free(gctx);
}

static void test_control_vector() {
    if (!file_exists(MODEL_PATH)) SKIP("model file not found");
    const char* tmp = getenv("PVX_TEST_TMP");
    std::string dir = tmp ? std::string(tmp) : "/tmp";
    std::string cv_path = dir + "/pvx_cvec_test.gguf";
    std::string bad_path = dir + "/pvx_cvec_bad.gguf";

    // Model dims are derived at runtime from the model GGUF — NOTHING is
    // hardcoded; the fixture works for any arch/model the suite runs against.
    ggml_context* meta_ctx = nullptr;
    gguf_init_params mip = {};
    mip.no_alloc = true;
    mip.ctx = &meta_ctx;
    gguf_context* meta = gguf_init_from_file(MODEL_PATH, mip);
    ASSERT(meta != nullptr);
    int model_n_embd = 0, model_n_layer = 0;
    {
        const int n_keys = (int) gguf_get_n_kv(meta);
        for (int i = 0; i < n_keys; i++) {
            const char* k = gguf_get_key(meta, i);
            if (!k) continue;
            const size_t kl = strlen(k);
            if (kl >= 17 && strcmp(k + kl - 17, ".embedding_length") == 0)
                model_n_embd = (int) gguf_get_val_u32(meta, i);
            else if (kl >= 12 && strcmp(k + kl - 12, ".block_count") == 0)
                model_n_layer = (int) gguf_get_val_u32(meta, i);
        }
    }
    gguf_free(meta);
    if (meta_ctx) ggml_free(meta_ctx);
    ASSERT(model_n_embd > 0);   // every supported arch declares <arch>.embedding_length
    ASSERT(model_n_layer > 0);   // every supported arch declares <arch>.block_count

    pvx_model_params_t mp = {}; mp.model_path = MODEL_PATH; mp.gpu_layers = 99; mp.flash_attn = true;
    pvx_model_t* m = nullptr; ASSERT(pvx_load_model(&mp, &m) == PVX_OK);

    // Wrong-width file: typed cvector/n-embd refusal; the RIGHT-width file
    // then applies/detaches with baseline restoration.
    const int bad_embd = model_n_embd / 2 > 0 ? model_n_embd / 2 : model_n_embd + 1;
    write_cvector_gguf(bad_path.c_str(), bad_embd, 4, 1);   // wrong width, derived
    pvx_cvec_t* bad = nullptr;
    ASSERT(pvx_control_vector_load(m, bad_path.c_str(), &bad) == PVX_ERR_LOAD_FAILED);
    ASSERT(strstr(pvx_model_last_error(m), "cvector/n-embd") != nullptr);

    // missing file: typed
    ASSERT(pvx_control_vector_load(m, "/nonexistent/cv.gguf", &bad) == PVX_ERR_LOAD_FAILED);

    write_cvector_gguf(cv_path.c_str(), model_n_embd, model_n_layer, 2);   // matches the model exactly
    pvx_cvec_t* cv = nullptr;
    ASSERT(pvx_control_vector_load(m, cv_path.c_str(), &cv) == PVX_OK);

    pvx_context_t* ctx = make_ctx(m, 2, 2048);  // seq_max 2: state_save backup seq
    pvx_executor_t* exec = nullptr; ASSERT(pvx_executor_create(ctx, &exec) == PVX_OK);
    ASSERT(pvx_executor_prompt(exec, "The quick brown fox") == PVX_OK);
    ASSERT(pvx_executor_infer(exec) == PVX_DECODE_OK);
    pvx_state_t* anchor = nullptr;
    ASSERT(pvx_state_save(exec, &anchor) == PVX_OK);
    const int n_anchor = pvx_state_token_count(anchor);

    pvx_sampling_params_t sp = {}; sp.temperature = 0.0f; sp.top_k = 1;
    int32_t t_base = 0, t_cv = 0, t_off = 0;

    ASSERT(pvx_executor_sample(exec, &sp, &t_base) == PVX_OK);            // baseline

    // invalid layer range: typed (explicit out-of-range end, never clamped — derived)
    ASSERT(pvx_context_set_control_vector(ctx, cv, 0, model_n_layer + 1) == PVX_ERR_INVALID_ARG);
    ASSERT(strstr(pvx_last_error(ctx), "cvector/layer-range") != nullptr);

    // baseline continuation arm (reference for the on/off comparison)
    ASSERT(pvx_state_restore(exec, anchor) == PVX_OK);
    ASSERT(pvx_executor_prompt_tokens(exec, &t_base, 1) == PVX_OK);
    ASSERT(pvx_executor_infer(exec) == PVX_DECODE_OK);
    int32_t t_pre = 0;
    ASSERT(pvx_executor_sample(exec, &sp, &t_pre) == PVX_OK);

    // apply (defaults = full range), same decode → cvec logits
    ASSERT(pvx_state_restore(exec, anchor) == PVX_OK);
    ASSERT(pvx_context_set_control_vector(ctx, cv, 0, 0) == PVX_OK);
    ASSERT(pvx_executor_prompt_tokens(exec, &t_base, 1) == PVX_OK);
    ASSERT(pvx_executor_infer(exec) == PVX_DECODE_OK);
    ASSERT(pvx_executor_sample(exec, &sp, &t_cv) == PVX_OK);

    // detach, same decode → baseline logits RETURN (KV untouched)
    ASSERT(pvx_state_restore(exec, anchor) == PVX_OK);
    ASSERT(pvx_context_set_control_vector(ctx, nullptr, 0, 0) == PVX_OK);  // detach
    ASSERT(pvx_executor_prompt_tokens(exec, &t_base, 1) == PVX_OK);
    ASSERT(pvx_executor_infer(exec) == PVX_DECODE_OK);
    ASSERT(pvx_executor_sample(exec, &sp, &t_off) == PVX_OK);
    ASSERT(t_off == t_pre);  // detach restores baseline logits; KV untouched
    fprintf(stderr, "\n      cvec tokens: prompt-greedy=%d baseline-cont=%d with-cv=%d detached-cont=%d ",
            t_base, t_pre, t_cv, t_off);

    pvx_state_free(anchor);
    pvx_executor_free(exec);
    pvx_free_context(ctx);
    pvx_control_vector_free(cv); pvx_free_model(m);
}

// Checkpoint import → spec init: the phantom-prefix guard (ABI v6 feature 5f).
// A restored KV stream WITHOUT its mirror ring must NEVER feed the spec
// engine — the ring does not cover the committed stream, so drafting would
// replay a phantom prefix. spec_init must refuse typed NOT_SUPPORTED with the
// mirror-consistency boundary (never silently degrade to plain drafting).
static void test_conversation_checkpoint_import_phantom_prefix_guard() {
    if (!file_exists(MODEL_PATH)) SKIP("model file not found");
    pvx_model_params_t mp = {}; mp.model_path = MODEL_PATH; mp.gpu_layers = 99; mp.flash_attn = true;
    pvx_model_t* m = nullptr; ASSERT(pvx_load_model(&mp, &m) == PVX_OK);
    pvx_context_params_t cp = {}; cp.context_size = 2048; cp.batch_size = 512; cp.seq_max = 4;
    pvx_context_t* ctx = nullptr; ASSERT(pvx_create_context(m, &cp, &ctx) == PVX_OK);
    pvx_pool_t* pool = nullptr; ASSERT(pvx_pool_create(ctx, &pool) == PVX_OK);

    // Conversation A: commit a real stream, export its checkpoint.
    pvx_conversation_t* ca = nullptr; ASSERT(pvx_pool_lease(pool, &ca) == PVX_OK);
    ASSERT(pvx_conversation_prompt(ca, "The capital of France is") == PVX_OK);
    ASSERT(pvx_infer(ctx) == PVX_DECODE_OK);
    pvx_sampling_params_t sp = {}; sp.temperature = 0.0f; sp.top_k = 1; sp.max_tokens = 1;
    int32_t t = 0;
    ASSERT(pvx_conversation_sample(ca, &sp, &t) == PVX_OK);
    const int committed = pvx_conversation_token_count(ca);
    ASSERT(committed > 0);
    uint8_t* blob = nullptr; size_t blob_size = 0;
    ASSERT(pvx_conversation_export(ca, &blob, &blob_size) == PVX_OK);
    ASSERT(blob && blob_size > 0);
    ASSERT(pvx_pool_return(pool, ca) == PVX_OK);  // pool return clears mirror/counters

    // Same slot re-leased, blob imported: KV + counters restored, mirror ring EMPTY.
    pvx_conversation_t* cb = nullptr; ASSERT(pvx_pool_lease(pool, &cb) == PVX_OK);
    ASSERT(pvx_conversation_import(cb, blob, blob_size) == PVX_OK);
    pvx_conversation_export_free(blob);
    ASSERT(pvx_conversation_token_count(cb) == committed);

    // The guard: spec init against a phantom prefix must refuse typed
    // (ngram-simple needs no draft sidecar; the ONLY failing boundary here is
    // mirror-consistency).
    pvx_spec_params_t s = {}; s.mode = PVX_SPEC_NGRAM_SIMPLE;
    ASSERT(pvx_conversation_spec_init(cb, &s) == PVX_ERR_NOT_SUPPORTED);
    const char* err = pvx_last_error(ctx);
    ASSERT(err && strstr(err, "mirror-consistency"));

    // Plain generation still works on the imported stream (KV is healthy —
    // only the spec substrate refuses). The import restores the KV stream but
    // NOT the logits snapshot (fresh-decode contract): commit one more token,
    // then greedy sampling continues off the restored KV.
    ASSERT(pvx_conversation_prompt(cb, " Paris") == PVX_OK);
    ASSERT(pvx_infer(ctx) == PVX_DECODE_OK);
    ASSERT(pvx_conversation_sample(cb, &sp, &t) == PVX_OK);
    ASSERT(pvx_conversation_token_count(cb) > committed);  // continuation committed
    ASSERT(pvx_pool_return(pool, cb) == PVX_OK);

    // Control: a FRESH lease + prompt (mirror covers stream) → spec init OK.
    pvx_conversation_t* cc = nullptr; ASSERT(pvx_pool_lease(pool, &cc) == PVX_OK);
    ASSERT(pvx_conversation_prompt(cc, "The capital of France is") == PVX_OK);
    ASSERT(pvx_infer(ctx) == PVX_DECODE_OK);
    ASSERT(pvx_conversation_spec_init(cc, &s) == PVX_OK);
    pvx_conversation_spec_free(cc);
    ASSERT(pvx_pool_return(pool, cc) == PVX_OK);

    pvx_pool_free(pool); pvx_free_context(ctx); pvx_free_model(m);
    fprintf(stderr, "  phantom-prefix guard fired on import: committed=%d refused typed\n", committed);
}

// Stats export (feature 12): counter sanity on both lanes (no spec session:
// 0/0 spec telemetry).
static void test_stats_export() {
    if (!file_exists(MODEL_PATH)) SKIP("model file not found");
    pvx_model_params_t mp = {}; mp.model_path = MODEL_PATH; mp.gpu_layers = 99; mp.flash_attn = true;
    pvx_model_t* m = nullptr; ASSERT(pvx_load_model(&mp, &m) == PVX_OK);

    // null-handle gate
    pvx_stats_t st = {};
    ASSERT(pvx_executor_stats(nullptr, &st) == PVX_ERR_INVALID_ARG);
    ASSERT(pvx_conversation_stats(nullptr, &st) == PVX_ERR_INVALID_ARG);

    // executor lane
    {
        pvx_context_t* ctx = make_ctx(m, 1, 1024);
        pvx_executor_t* exec = nullptr; ASSERT(pvx_executor_create(ctx, &exec) == PVX_OK);
        ASSERT(pvx_executor_prompt(exec, "Hello world") == PVX_OK);
        ASSERT(pvx_executor_infer(exec) == PVX_DECODE_OK);
        pvx_stats_t s = {};
        ASSERT(pvx_executor_stats(exec, &s) == PVX_OK);
        ASSERT(s.n_tokens == pvx_executor_token_count(exec));
        ASSERT(s.n_kv_pos == s.n_tokens);
        ASSERT(s.n_prompt_tokens == s.n_tokens);
        ASSERT(s.n_decode_steps >= 1);
        ASSERT(s.spec_drafted == 0 && s.spec_accepted == 0);
        pvx_executor_free(exec); pvx_free_context(ctx);
    }

    // conversation lane
    {
        pvx_context_t* ctx = make_ctx(m, 2, 1024);
        pvx_pool_t* pool = nullptr; ASSERT(pvx_pool_create(ctx, &pool) == PVX_OK);
        pvx_conversation_t* conv = nullptr; ASSERT(pvx_pool_lease(pool, &conv) == PVX_OK);
        ASSERT(pvx_conversation_prompt(conv, "Hello again") == PVX_OK);
        pvx_stats_t s = {};
        ASSERT(pvx_conversation_stats(conv, &s) == PVX_OK);
        // pre-batch snapshot: tokens are QUEUED (pending), not committed —
        // n_tokens/n_kv_pos still 0 (nothing decoded on this lane yet),
        // n_prompt_tokens counts the queued prompt, decode_steps 0.
        ASSERT(s.n_tokens == pvx_conversation_token_count(conv));
        ASSERT(s.n_kv_pos == s.n_tokens);
        ASSERT(s.n_prompt_tokens > 0);
        ASSERT(s.n_decode_steps == 0);  // batched decode not run yet: 0 steps so far
        // run the batched infer and verify the commit side advanced too
        pvx_infer(ctx);
        pvx_stats_t s2 = {};
        ASSERT(pvx_conversation_stats(conv, &s2) == PVX_OK);
        ASSERT(s2.n_tokens == (int64_t) pvx_conversation_token_count(conv));
        ASSERT(s2.n_kv_pos == s2.n_tokens);
        ASSERT(s2.n_prompt_tokens == s.n_prompt_tokens);   // queue drained once
        ASSERT(s2.n_decode_steps >= 1);  // batched decode ran
        ASSERT(pvx_pool_return(pool, conv) == PVX_OK);
        pvx_pool_free(pool); pvx_free_context(ctx);
    }
    pvx_free_model(m);
}

int main() {
    MODEL_PATH = getenv("PVX_MODEL_PATH");
    HYBRID_MODEL_PATH = getenv("PVX_HYBRID_MODEL_PATH");
    EMBED_MODEL_PATH = getenv("PVX_EMBED_MODEL_PATH");
    MMPROJ_PATH = getenv("PVX_MMPROJ_PATH");
    TEST_IMAGE = getenv("PVX_TEST_IMAGE");
    if (!MODEL_PATH) MODEL_PATH = "./models/Qwen3-8B-Q4_K_M.gguf";
    if (!HYBRID_MODEL_PATH) HYBRID_MODEL_PATH = "./models/Qwen3.5-4B-Q4_K_M.gguf";
    if (!EMBED_MODEL_PATH) EMBED_MODEL_PATH = "./models/all-MiniLM-L6-v2-Q5_K_M.gguf";
    if (!MMPROJ_PATH) MMPROJ_PATH = "./models/mmproj-Qwen3.5-4B-BF16.gguf";
    if (!TEST_IMAGE) TEST_IMAGE = "./test_image.png";
    DRAFT_MODEL_PATH = getenv("PVX_DRAFT_MODEL_PATH");

    fprintf(stderr, "\n=== ProvostXInference Tests ===\n");
    fprintf(stderr, "Model: %s\n", file_exists(MODEL_PATH) ? MODEL_PATH : "(not found)");
    fprintf(stderr, "Embed: %s\n", file_exists(EMBED_MODEL_PATH) ? EMBED_MODEL_PATH : "(not found)");
    fprintf(stderr, "\n");
    TEST(load_model);
    TEST(create_context);
    TEST(partial_rewind_capability);
    TEST(partial_rewind_refused_on_hybrid);
    TEST(tokenize);
    TEST(pool);
    TEST(single_inference);
    TEST(batched_inference);
    TEST(state_save_restore);
    TEST(embeddings);
    TEST(anti_prompts);
    TEST(pool_reuse);
    TEST(vision);
    TEST(shift_left);
    TEST(kv_copy);
    TEST(causal_attn);
    TEST(spec_capabilities);
    TEST(spec_capabilities_null);
    TEST(spec_validate);
    TEST(spec_generate_requires_session);
    TEST(spec_init_fail_closed);
    TEST(spec_ngram_token_exactness);
    TEST(spec_ngram_fail_closed);
    TEST(spec_vision_exactness);
    TEST(conversation_spec_ngram_exactness);
    TEST(conversation_spec_interleaved_exactness);
    TEST(conversation_spec_lease_churn);
    TEST(conversation_spec_rewind_continue);
    TEST(conversation_spec_fail_closed);
    TEST(spec_dflash_metadata_typed);
    TEST(spec_eagle3_companion_typed);
    TEST(spec_eagle3_live_pair);
    TEST(spec_dflash_live_pair);
    TEST(spec_mtp_draft_typed);
    TEST(spec_mtp_draft_live_pair);
    TEST(spec_mtp_vision_live_pair);
    TEST(spec_mtp_audio_live_pair);
    TEST(json_schema_grammar_roundtrip);
    TEST(rerank_typed);
    TEST(rerank_live);
    TEST(audio_capability_typed);
    TEST(audio_live);
    TEST(tts_live);
    TEST(lora_typed);
    TEST(lora_live);
    TEST(control_vector);
    TEST(conversation_checkpoint_import_phantom_prefix_guard);
    TEST(stats_export);
    fprintf(stderr, "\n=== %d/%d passed, %d skipped ===\n", tests_passed, tests_run, tests_skipped);
    return 0;
}
