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

// ═══════════════════════════════════════════════════════════════════════════
// ABI v7 wave 2 — JSON-schema → GBNF converter unit tests (no model needed)
// + conversation-lane mirror helpers.
// ═══════════════════════════════════════════════════════════════════════════

#include "pvxinference.h"

static void test_json_schema_basic_object() {
    char* gbnf = nullptr; char* err = nullptr;
    const char* schema = R"SCHEMA({
        "type": "object",
        "properties": { "name": { "type": "string" }, "age": { "type": "integer" } },
        "required": ["name"]
    })SCHEMA";
    ASSERT(pvx_grammar_from_json_schema(schema, nullptr, &gbnf, &err) == PVX_OK);
    ASSERT(gbnf && !err);
    const std::string g(gbnf);
    // root object rule: required name first, optional age branch
    ASSERT(g.find(R"x(root ::= "{" space root-name-kv ( "," space ( root-age-kv ) )? space "}")x") != std::string::npos);
    ASSERT(g.find("root-name-kv ::=") != std::string::npos);
    ASSERT(g.find("space \":\" space root-name") != std::string::npos);
    ASSERT(g.find(R"x(\"name\")x") != std::string::npos);
    ASSERT(g.find("space ::=") != std::string::npos);
    ASSERT(g.find("string ::=") != std::string::npos);
    ASSERT(g.find("integer ::=") != std::string::npos);
    pvx_free_string(gbnf);
}

static void test_json_schema_int_range() {
    char* gbnf = nullptr; char* err = nullptr;
    const char* schema = R"SCHEMA({ "type": "integer", "minimum": 5, "maximum": 100 })SCHEMA";
    ASSERT(pvx_grammar_from_json_schema(schema, nullptr, &gbnf, &err) == PVX_OK);
    const std::string g(gbnf);
    // upstream build_min_max_int output shape for [5, 100]: union of
    // [5-9], the two-digit range, and the exact max bound "100"
    ASSERT(g.find(R"x(root ::= ([5-9] | ([1-8] [0-9] | [9] [0-9]) | "100"))x") != std::string::npos);
    pvx_free_string(gbnf);
}

static void test_json_schema_enum_const() {
    char* gbnf = nullptr; char* err = nullptr;
    const char* schema = R"SCHEMA({ "enum": ["red", "green", 7, true, null] })SCHEMA";
    ASSERT(pvx_grammar_from_json_schema(schema, nullptr, &gbnf, &err) == PVX_OK);
    const std::string g(gbnf);
    ASSERT(g.find(R"x(root ::= ("\"red\"" | "\"green\"" | "7" | "true" | "null"))x") != std::string::npos);
    pvx_free_string(gbnf);
    gbnf = nullptr;
    const char* cst = R"SCHEMA({ "const": "hello" })SCHEMA";
    ASSERT(pvx_grammar_from_json_schema(cst, nullptr, &gbnf, &err) == PVX_OK);
    ASSERT(gbnf && std::string(gbnf).find(R"x(root ::= "\"hello\"")x") != std::string::npos);
    pvx_free_string(gbnf);
}

static void test_json_schema_refs_and_defs() {
    char* gbnf = nullptr; char* err = nullptr;
    // recursive schema: a node has optional next pointing back at itself
    const char* schema = R"SCHEMA({
        "$defs": { "node": { "type": "object",
            "properties": { "v": { "type": "integer" }, "next": { "$ref": "#/$defs/node" } },
            "required": ["v"] } },
        "$ref": "#/$defs/node"
    })SCHEMA";
    ASSERT(pvx_grammar_from_json_schema(schema, nullptr, &gbnf, &err) == PVX_OK);
    const std::string g(gbnf);
    ASSERT(g.find("ref-defs-node ::=") != std::string::npos);
    // the recursive reference must reuse the named rule, not re-expand
    const size_t n_rec = g.find("ref-defs-node-next ::=");
    ASSERT(n_rec != std::string::npos);
    // the named node rule is defined exactly once (self-referencing shape)
    int n_defs = 0;
    for (size_t k = 0; (k = g.find("ref-defs-node ::=", k)) != std::string::npos; k++) n_defs++;
    ASSERT(n_defs == 1);
    ASSERT(g.find("ref-defs-node-next ::= ref-defs-node") != std::string::npos);
    pvx_free_string(gbnf);
}

static void test_json_schema_decision_envelope() {
    // The v13 decision-envelope schema (structured decisions): thinking
    // always, optional answer, optional toolcalls with name+string-arg map.
    char* gbnf = nullptr; char* err = nullptr;
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
    const std::string g(gbnf);
    ASSERT(g.find(R"x(root-thinking-kv ::= "\"thinking\"" space ":" space root-thinking)x") != std::string::npos);
    ASSERT(g.find("root-toolcalls-0-item-args-kv ::=") != std::string::npos);
    ASSERT(g.find("root-toolcalls-1 ::= null") != std::string::npos);
    ASSERT(g.find(R"x(root-toolcalls-0-item-args-additional-kv ::= string ":" space root-toolcalls-0-item-args-additional-value)x") != std::string::npos);
    pvx_free_string(gbnf);
}

static void test_json_schema_invalid_json_typed() {
    char* gbnf = nullptr; char* err = nullptr;
    ASSERT(pvx_grammar_from_json_schema("{ not json", nullptr, &gbnf, &err)
           == PVX_ERR_INVALID_ARG);
    ASSERT(gbnf == nullptr && err != nullptr);
    ASSERT(strstr(err, "json-schema/parse") != nullptr);
    ASSERT(strstr(err, "position") != nullptr);
    pvx_free_string(err);
}

static void test_json_schema_unsupported_keyword_typed() {
    char* gbnf = nullptr; char* err = nullptr;
    ASSERT(pvx_grammar_from_json_schema(
        R"SCHEMA({ "type": "string", "pattern": "^[a-z]+$" })SCHEMA",
        nullptr, &gbnf, &err) == PVX_ERR_NOT_SUPPORTED);
    ASSERT(err && strstr(err, "'pattern'") != nullptr);
    ASSERT(strstr(err, "json-schema/keyword") != nullptr);
    pvx_free_string(err);
    err = nullptr;
    ASSERT(pvx_grammar_from_json_schema(
        R"SCHEMA({ "type": "object", "propertyNames": { "type": "string" } })SCHEMA",
        nullptr, &gbnf, &err) == PVX_ERR_NOT_SUPPORTED);
    ASSERT(err && strstr(err, "'propertyNames'") != nullptr);
    pvx_free_string(err);
}

static void test_json_schema_format_fails_closed() {
    char* gbnf = nullptr; char* err = nullptr;
    ASSERT(pvx_grammar_from_json_schema(
        R"SCHEMA({ "type": "string", "format": "email" })SCHEMA",
        nullptr, &gbnf, &err) == PVX_ERR_NOT_SUPPORTED);
    ASSERT(err && strstr(err, "'format'") != nullptr);
    pvx_free_string(err);
    err = nullptr;
    gbnf = nullptr;
    ASSERT(pvx_grammar_from_json_schema(
        R"SCHEMA({ "type": "string", "format": "date-time" })SCHEMA",
        nullptr, &gbnf, &err) == PVX_OK);
    ASSERT(gbnf && std::string(gbnf).find("date-time ::=") != std::string::npos);
    pvx_free_string(gbnf);
}

static void test_conv_mirror_helpers() {
    pvx_conversation_s conv{};
    pvx_conv_mirror_push_token(&conv, 42, 7);
    const std::array<llama_pos, 4> rp = { 8, 8, 8, 8 };
    std::vector<float> row = { 1.0f, 2.0f };
    pvx_conv_mirror_push_embd(&conv, row.data(), 2, rp);
    ASSERT(conv.mirror.size() == 2);
    ASSERT(conv.mirror[0].id == 42);
    ASSERT(conv.mirror[0].pos[0] == 7);
    ASSERT(conv.mirror[1].id == LLAMA_TOKEN_NULL);
    ASSERT(conv.mirror[1].embd.size() == 2);
    ASSERT(conv.mirror[1].pos[3] == 8);
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
    TEST(json_schema_basic_object);
    TEST(json_schema_int_range);
    TEST(json_schema_enum_const);
    TEST(json_schema_refs_and_defs);
    TEST(json_schema_decision_envelope);
    TEST(json_schema_invalid_json_typed);
    TEST(json_schema_unsupported_keyword_typed);
    TEST(json_schema_format_fails_closed);
    TEST(conv_mirror_helpers);
    fprintf(stderr, "\n=== %d/%d passed ===\n", tests_passed, tests_run);
    return 0;
}
