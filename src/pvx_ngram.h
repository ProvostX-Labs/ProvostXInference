// pvx_ngram.h — n-gram speculative drafting (ABI v6 wave 1).
//
// Pattern-port of the vendored n-gram drafters (llama.cpp/common/ngram-map.h,
// ngram-mod.h, ngram-cache.h). Per the rules of engagement the frozen tree is
// reference material only: nothing under llama.cpp/ is included, linked, or
// called from here — every routine below lives in src/ against the stable
// llama.h token types only. Semantics follow the vendored implementation
// line-for-line where observable (draft contents, acceptance statistics,
// cache file format) so behavior is comparable against upstream.
//
// Four drafters:
//   - simple : history-pattern lookup (common_ngram_simple_draft)
//   - map_k  : n-gram -> m-gram map with per-value accept statistics
//              (common_ngram_map begin/draft/accept)
//   - mod    : hashed (n-1)-suffix table, one deviation per prediction
//              (common_ngram_mod)
//   - cache  : static (file) + dynamic token-probability caches
//              (common_ngram_cache update/draft/save/load)
//
// None of them owns a draft model or context: drafts are produced from the
// committed token stream (the executor's recent_tokens window) and verified
// by the target exactly like draft-model chains (pvx_spec_engine.cpp).

#pragma once

#include "llama.h"

#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace pvx {

// ── n-gram simple ──────────────────────────────────────────────────────────

struct ngram_simple_config {
    uint16_t size_ngram = 0; // n-gram lookup size
    uint16_t size_mgram = 0; // m-gram draft size
};

// Upstream common_ngram_simple_draft: search the history for the trailing
// n-gram (ending at `sampled`) and copy the m-gram that followed the last
// match. Empty result = no draft.
std::vector<llama_token> ngram_simple_draft(const ngram_simple_config& config,
                                            const std::vector<llama_token>& tokens,
                                            llama_token sampled);

// ── n-gram MOD ────────────────────────────────────────────────────────────

// Upstream common_ngram_mod: open-addressed hash of n-gram -> next token.
// add() takes a pointer to n+1 tokens (the last is the value). Port is
// semantics-identical (same hash constant, same EMPTY sentinel).
class ngram_mod {
public:
    using entry_t = int32_t;
    static constexpr entry_t EMPTY = -1;

    ngram_mod(uint16_t n, size_t size);

    size_t   idx(const entry_t* tokens) const;
    void     add(const entry_t* tokens);
    entry_t  get(const entry_t* tokens) const;
    void     reset();

    size_t get_n()    const;
    size_t get_used() const;
    size_t size()       const;
    size_t size_bytes() const;

private:
    size_t n;
    size_t used;
    std::vector<entry_t> entries;
};

// ── n-gram map-k ──────────────────────────────────────────────────────────

constexpr uint16_t NGRAM_MAX_VALUES     = 4;    // m-gram values per key
constexpr size_t   NGRAM_HASH_MAP_SIZE   = 262144; // 2^18 hash slots
constexpr uint16_t NGRAM_MAX_VALUE_COUNT = 16380;  // saturating occurrence counter

struct ngram_map_value {
    size_t   value_idx = 0;  // index of the value m-gram in the history (0 = unused)
    uint16_t value_num = 0;  // occurrences of this m-gram after the key
    int16_t  n_accepted = -1; // accepted count at the last draft (-1 = unused)
};

struct ngram_map_key {
    size_t   key_idx  = 0;
    size_t   stat_idx = 0;
    uint16_t key_num  = 0;
    ngram_map_value values[NGRAM_MAX_VALUES];
};

struct ngram_map {
    uint16_t size_key   = 0;
    uint16_t size_value = 0;
    bool     key_only   = false;
    uint16_t min_hits   = 0;

    std::vector<ngram_map_key> keys;

    size_t  size_last_begin      = 0;
    bool    last_draft_created   = false;
    size_t  last_draft_key_idx   = 0;
    uint16_t last_draft_value_idx = 0;
    size_t  idx_last_check       = 0;

    std::vector<uint32_t> key_map;        // hash -> ngram index (0 = unused)
    uint32_t              key_map_last_idx = 0;

    ngram_map(uint16_t sz_key, uint16_t sz_value, bool only_keys, uint16_t min_hits);
};

void ngram_map_begin(ngram_map& map, const std::vector<llama_token>& tokens);
void ngram_map_draft(ngram_map& map, const std::vector<llama_token>& inp,
                     llama_token sampled, std::vector<llama_token>& draft);
void ngram_map_accept(ngram_map& map, uint16_t n_accepted);

// ── n-gram cache ──────────────────────────────────────────────────────────

constexpr int NGRAM_MIN    = 1;
constexpr int NGRAM_MAX    = 4;
constexpr int NGRAM_STATIC = 2;

struct ngram_key {
    llama_token tokens[NGRAM_MAX];

    ngram_key();
    ngram_key(const llama_token* input, int ngram_size);

    bool operator==(const ngram_key& other) const;
};

struct ngram_token_hash {
    size_t operator()(llama_token token) const;
};

struct ngram_key_hash {
    size_t operator()(const ngram_key& ngram) const;
};

using ngram_cache_part = std::unordered_map<llama_token, int32_t>;
using ngram_cache = std::unordered_map<ngram_key, ngram_cache_part, ngram_key_hash>;

void ngram_cache_update(ngram_cache& cache, int ngram_min, int ngram_max,
                        std::vector<llama_token>& inp, int nnew);
void ngram_cache_draft(std::vector<llama_token>& inp, std::vector<llama_token>& draft,
                       int n_draft, int ngram_min, int ngram_max,
                       ngram_cache& nc_context, ngram_cache& nc_dynamic, ngram_cache& nc_static);
void ngram_cache_save(const ngram_cache& cache, const std::string& filename);
// Fail-closed load: corrupt/truncated files return false (typed error naming
// the path is the caller's job) instead of aborting the process.
bool ngram_cache_load(const std::string& filename, ngram_cache& out_cache);
void ngram_cache_merge(ngram_cache& target, const ngram_cache& add);

} // namespace pvx
