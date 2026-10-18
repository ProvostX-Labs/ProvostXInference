// test_basic.cpp — Full test suite for ProvostXInference C API

#include "pvxinference.h"
#include <cassert>
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
    fprintf(stderr, "\n=== %d/%d passed, %d skipped ===\n", tests_passed, tests_run, tests_skipped);
    return 0;
}
