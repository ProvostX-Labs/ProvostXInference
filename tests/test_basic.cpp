// test_basic.cpp — Full test suite for ProvostXInference C API

#include "pvxinference.h"
#include "gguf.h"
#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <string>
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
        ASSERT(pvx_executor_spec_init(exec, &sp) == PVX_OK);
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
    TEST(json_schema_grammar_roundtrip);
    fprintf(stderr, "\n=== %d/%d passed, %d skipped ===\n", tests_passed, tests_run, tests_skipped);
    return 0;
}
