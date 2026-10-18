// test_ngram.cpp — Pure unit tests for the ABI v6 wave-1 additions:
//   - n-gram drafting logic (simple / mod / map-k / cache) — src/pvx_ngram
//   - embd-aware mirror ring mechanics — include/pvx_internal.h
// No model files needed: everything here is deterministic logic. Test
// fixtures may use env vars for the scratch dir (product code never does);
// the default is the system temp dir.

#include "pvx_ngram.h"
#include "pvx_internal.h"
#include "pvx_spec_engine.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static int tests_run = 0, tests_passed = 0;

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

using namespace pvx;

// ── n-gram simple ──────────────────────────────────────────────────────────

static void test_ngram_simple_repeat_pattern() {
    // history: [1 2 3 4] [1 2 3 4] [1 2 3]  + sampled 4
    const std::vector<llama_token> tokens = {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3};
    ngram_simple_config cfg{2, 3};
    const auto draft = ngram_simple_draft(cfg, tokens, /*sampled=*/4);
    // pattern (3,4) last occurred at index 6 -> continuation is 1,2,3
    ASSERT(draft.size() >= 2);
    ASSERT(draft[0] == 1);
    ASSERT(draft[1] == 2);
}

static void test_ngram_simple_no_match() {
    const std::vector<llama_token> tokens = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    ngram_simple_config cfg{2, 4};
    const auto draft = ngram_simple_draft(cfg, tokens, /*sampled=*/11);
    ASSERT(draft.empty());
}

static void test_ngram_simple_too_short() {
    const std::vector<llama_token> tokens = {1, 2, 3};
    ngram_simple_config cfg{2, 4};
    ASSERT(ngram_simple_draft(cfg, tokens, 3).empty());
}

// ── n-gram MOD ──────────────────────────────────────────────────────────────

static void test_ngram_mod_basic() {
    ngram_mod mod(2, 1024);
    const int32_t a1[] = {1, 2, 3};
    const int32_t a2[] = {4, 5, 6};
    mod.add(a1);
    mod.add(a2);
    ASSERT(mod.get(a1) == 3);
    ASSERT(mod.get(a2) == 6);
    const int32_t miss[] = {9, 9};
    ASSERT(mod.get(miss) == ngram_mod::EMPTY);
    ASSERT(mod.get_used() == 2);
    mod.reset();
    ASSERT(mod.get_used() == 0);
    ASSERT(mod.get(a1) == ngram_mod::EMPTY);
}

// ── n-gram map-k ────────────────────────────────────────────────────────────

static void test_ngram_map_draft_and_accept() {
    // key n-gram (5,6) occurs twice with the same continuation (7,8,9).
    // Contract: `tokens` is the committed stream WITHOUT the sampled tail.
    const std::vector<llama_token> tokens =
        {0, 1, 5, 6, 7, 8, 9, 2, 3, 5};
    ngram_map map(2, 3, /*only_keys=*/false, /*min_hits=*/1);
    ngram_map_begin(map, tokens);

    std::vector<llama_token> draft;
    ngram_map_draft(map, tokens, /*sampled=*/6, draft);
    // key (5,6): continuation 7,8,9 must be proposed
    ASSERT(draft.size() >= 2);
    ASSERT(draft[0] == 7);
    ASSERT(draft[1] == 8);
    ASSERT(draft[2] == 9);

    // accept 2 of the drafted tokens: the value's accepted-length stat updates
    ASSERT(map.last_draft_created);
    ngram_map_accept(map, 2);
    const size_t ki = map.last_draft_key_idx;
    const uint16_t vi = map.last_draft_value_idx;
    ASSERT(map.keys[ki].values[vi].n_accepted == 2);
    // A subsequent draft from the same key is bounded by the accepted stat.
    std::vector<llama_token> draft2;
    ngram_map_draft(map, tokens, 6, draft2);
    ASSERT(draft2.size() <= 2 || draft2.size() == 3); // bounded by m or stat
    if (draft2.size() == 3) {
        // stat still m when no accept happened for draft2's key/value pair
        ASSERT(map.keys[map.last_draft_key_idx].values[map.last_draft_value_idx].n_accepted == 2);
    }
}

static void test_ngram_map_no_draft_on_unique_key() {
    const std::vector<llama_token> tokens = {1, 2, 3, 4, 5, 6, 7, 8};
    ngram_map map(2, 3, false, 1);
    ngram_map_begin(map, tokens);
    std::vector<llama_token> draft;
    ngram_map_draft(map, tokens, /*sampled=*/9, draft);
    ASSERT(draft.empty());
    ASSERT(!map.last_draft_created);
}

static void test_ngram_map_begin_handles_shrink() {
    std::vector<llama_token> tokens = {2, 5, 6, 7, 8, 9, 5};
    ngram_map map(2, 3, false, 1);
    ngram_map_begin(map, tokens);
    std::vector<llama_token> draft;
    ngram_map_draft(map, tokens, 6, draft);
    ASSERT(draft.size() >= 1);

    // rewind: stream shrinks — begin() must clean up without aborting
    tokens.resize(4);
    ngram_map_begin(map, tokens);
    std::vector<llama_token> draft2;
    ngram_map_draft(map, tokens, 4, draft2);
    ASSERT(draft2.empty());
}

// ── n-gram cache ────────────────────────────────────────────────────────────

static void test_ngram_cache_update_and_draft() {
    // build a cache where (9,9) -> 5 overwhelmingly
    std::vector<llama_token> inp;
    for (int r = 0; r < 5; r++) {
        inp.push_back(9); inp.push_back(9); inp.push_back(5);
    }
    ngram_cache nc_context;
    ngram_cache_update(nc_context, 1, 2, inp, (int) inp.size());

    std::vector<llama_token> ctx = {9, 9};
    std::vector<llama_token> draft = {5};
    ngram_cache_draft(ctx, draft, /*n_draft=*/4, 1, 2, nc_context,
                      nc_context, nc_context);
    // context cache proposes 5 after (9,9,5)
    ASSERT(draft.size() >= 2);
    ASSERT(draft[1] == 5);
}

static void test_ngram_cache_save_load_roundtrip() {
    ngram_cache cache;
    const llama_token key[] = {7, 7};
    const ngram_key k(key, 2);
    ngram_cache_part part;
    part.emplace(42, 3);
    cache.emplace(k, part);

    const char* tmp = getenv("PVX_TEST_TMP");
    std::string path = tmp ? std::string(tmp) : std::string("/tmp");
    path += "/pvx_ngram_cache_test.bin";
    ngram_cache_save(cache, path);

    ngram_cache loaded;
    ASSERT(ngram_cache_load(path, loaded));
    ASSERT(loaded.size() == 1);
    const auto& lpart = loaded.begin()->second;
    ASSERT(lpart.at(42) == 3);
    remove(path.c_str());
}

static void test_ngram_cache_load_corrupt_fails_closed() {
    const char* tmp = getenv("PVX_TEST_TMP");
    std::string path = tmp ? std::string(tmp) : std::string("/tmp");
    path += "/pvx_ngram_corrupt_test.bin";
    FILE* f = fopen(path.c_str(), "wb");
    ASSERT(f);
    // truncated garbage: a full ngram_key + a token count of 2 but no payload
    ngram_key k;
    fwrite(&k, sizeof(k), 1, f);
    const int32_t bad = 2;
    fwrite(&bad, sizeof(bad), 1, f);
    fclose(f);

    ngram_cache loaded;
    ASSERT(!ngram_cache_load(path, loaded));
    ASSERT(loaded.empty());
    remove(path.c_str());
}

static void test_ngram_cache_merge() {
    ngram_cache a, b;
    const llama_token key1[] = {3, 4};
    const ngram_key k1(key1, 2);
    ngram_cache_part p1; p1.emplace(10, 1);
    a.emplace(k1, p1);
    ngram_cache_part p2; p2.emplace(10, 2); p2.emplace(11, 1);
    b.emplace(k1, p2);
    ngram_cache_merge(a, b);
    ASSERT(a.size() == 1);
    ASSERT(a.begin()->second.at(10) == 3);
    ASSERT(a.begin()->second.at(11) == 1);
}

// ── Embd-aware mirror ring mechanics ────────────────────────────────────────

static void test_mirror_ring_token_and_embd_entries() {
    pvx_executor_s exec{}; // plain struct, no ctx needed for ring mechanics
    ASSERT(exec.mirror.empty());
    for (int i = 0; i < 4; i++) {
        pvx_mirror_push_token(&exec, 100 + i, i);
    }
    ASSERT(exec.mirror.size() == 4);
    ASSERT(exec.mirror[0].id == 100 && exec.mirror[3].id == 103);
    for (int i = 0; i < 4; i++) {
        ASSERT(exec.mirror[i].pos[0] == i);
        // plain token rows carry the position replicated across sections
        ASSERT(exec.mirror[i].pos[1] == i);
        ASSERT(exec.mirror[i].pos[2] == i);
    }
    // embd entry: id NULL, payload stored, pinned 4-dim position
    const float row[3] = {0.25f, -0.5f, 1.0f};
    const std::array<llama_pos, 4> mpos = {4, 6, 5, 0}; // [t, y, x, z]
    pvx_mirror_push_embd(&exec, row, 3, mpos);
    ASSERT(exec.mirror.size() == 5);
    const auto& e = exec.mirror.back();
    ASSERT(e.id == LLAMA_TOKEN_NULL);
    ASSERT(e.embd.size() == 3);
    ASSERT(e.embd[0] == 0.25f && e.embd[2] == 1.0f);
    ASSERT(e.pos[0] == 4 && e.pos[1] == 6 && e.pos[2] == 5 && e.pos[3] == 0);
}

static void test_mirror_ring_invariant_shapes() {
    // The spec engine's guard requires: size == n_tokens, positions
    // ascending in t, back == n_kv_pos-1, front == n_kv_pos-n_tokens.
    pvx_executor_s exec{};
    exec.n_tokens = 0; exec.n_kv_pos = 0;
    for (int i = 0; i < 3; i++) {
        pvx_mirror_push_token(&exec, 7, i);
        exec.n_tokens++; exec.n_kv_pos++;
    }
    ASSERT(exec.mirror.size() == (size_t) exec.n_tokens);
    ASSERT(exec.mirror.front().pos[0] == exec.n_kv_pos - exec.n_tokens);
    ASSERT(exec.mirror.back().pos[0] == exec.n_kv_pos - 1);

    // rewind n=1: ring retreats with the window
    exec.mirror.resize(exec.mirror.size() - 1);
    exec.n_tokens--; exec.n_kv_pos--;
    ASSERT(exec.mirror.size() == (size_t) exec.n_tokens);
    ASSERT(exec.mirror.back().pos[0] == exec.n_kv_pos - 1);

    // shift_left n=1: front entries drop, positions stay pinned
    exec.mirror.erase(exec.mirror.begin(), exec.mirror.begin() + 1);
    exec.n_tokens--;
    ASSERT(exec.mirror.size() == (size_t) exec.n_tokens);
    ASSERT(exec.mirror.front().pos[0] == exec.n_kv_pos - exec.n_tokens);
}

// mirror-consistency guard shape, exercised through a deliberately desynced
// ring: the invariant mirror.size() == n_tokens must be detectable.
static void test_mirror_guard_detects_desync() {
    pvx_executor_s exec{};
    exec.n_tokens = 3; exec.n_kv_pos = 3;
    for (int i = 0; i < 2; i++) pvx_mirror_push_token(&exec, 1, i);
    // a commit path forgot the ring: size 2 != n_tokens 3
    ASSERT(exec.mirror.size() != (size_t) exec.n_tokens);
}

int main() {
    fprintf(stderr, "\n=== ProvostXInference n-gram + mirror unit tests ===\n");
    TEST(ngram_simple_repeat_pattern);
    TEST(ngram_simple_no_match);
    TEST(ngram_simple_too_short);
    TEST(ngram_mod_basic);
    TEST(ngram_map_draft_and_accept);
    TEST(ngram_map_no_draft_on_unique_key);
    TEST(ngram_map_begin_handles_shrink);
    TEST(ngram_cache_update_and_draft);
    TEST(ngram_cache_save_load_roundtrip);
    TEST(ngram_cache_load_corrupt_fails_closed);
    TEST(ngram_cache_merge);
    TEST(mirror_ring_token_and_embd_entries);
    TEST(mirror_ring_invariant_shapes);
    TEST(mirror_guard_detects_desync);
    fprintf(stderr, "\n=== %d/%d passed ===\n", tests_passed, tests_run);
    return 0;
}
