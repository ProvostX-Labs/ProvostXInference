// pvx_ngram.cpp — n-gram speculative drafting (ABI v6 wave 1).
// Pattern-port of the vendored llama.cpp/common n-gram drafters; see
// pvx_ngram.h for the scope and provenance of each routine. The frozen tree
// is never included — this TU depends on stable llama.h token types only.

#include "pvx_ngram.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>

namespace pvx {

// prime number used for the LCG hash (32 bit), near (sqrt(5)-1)/2 * 2^32.
static constexpr uint32_t kLcgFactor = 2654435761UL;

static uint32_t ngram_map_hash(const std::vector<llama_token>& tokens,
                               size_t start, size_t len) {
    uint32_t hash = 0;
    for (size_t i = 0; i < len; ++i) {
        hash = hash * kLcgFactor + tokens[start + i];
    }
    return hash;
}

// ── n-gram simple ──────────────────────────────────────────────────────────

std::vector<llama_token> ngram_simple_draft(const ngram_simple_config& config,
                                            const std::vector<llama_token>& tokens,
                                            llama_token sampled) {
    std::vector<llama_token> draft_tokens;

    const size_t cur_len      = tokens.size();
    const size_t n_draft_min  = config.size_ngram;
    const size_t n_draft_max  = config.size_mgram;

    if (cur_len <= n_draft_min + n_draft_max + 1) {
        return draft_tokens;
    }

    std::vector<llama_token> pattern;
    pattern.reserve(n_draft_min);
    for (size_t j = cur_len - n_draft_min + 1; j < cur_len; ++j) {
        pattern.push_back(tokens[j]);
    }
    pattern.push_back(sampled);

    size_t match_pos = 0; // position 0 == no match (we ignore it)
    for (size_t j = cur_len - n_draft_min - 1; j > 0; --j) {
        bool match = true;
        for (size_t k = 0; k < pattern.size(); ++k) {
            if (tokens[j + k] != pattern[k]) {
                match = false;
                break;
            }
        }
        if (match) {
            match_pos = j;
            break;
        }
    }
    if (match_pos == 0) {
        return draft_tokens;
    }

    const size_t copy_max = std::min(n_draft_max, cur_len - (match_pos + n_draft_min));
    if (copy_max < n_draft_min) {
        return draft_tokens;
    }

    draft_tokens.reserve(copy_max);
    for (size_t j = 0; j < copy_max; ++j) {
        draft_tokens.push_back(tokens[match_pos + n_draft_min + j]);
    }
    return draft_tokens;
}

// ── n-gram MOD ──────────────────────────────────────────────────────────────

ngram_mod::ngram_mod(uint16_t n_, size_t size) : n(n_), used(0) {
    entries.resize(size);
    reset();
}

size_t ngram_mod::idx(const entry_t* tokens) const {
    size_t res = 0;
    for (size_t i = 0; i < n; ++i) {
        res = res * 6364136223846793005ULL + tokens[i];
    }
    return res % entries.size();
}

void ngram_mod::add(const entry_t* tokens) {
    const size_t i = idx(tokens);
    if (entries[i] == EMPTY) {
        used++;
    }
    entries[i] = tokens[n];
}

ngram_mod::entry_t ngram_mod::get(const entry_t* tokens) const {
    return entries[idx(tokens)];
}

void ngram_mod::reset() {
    std::fill(entries.begin(), entries.end(), EMPTY);
    used = 0;
}

size_t ngram_mod::get_n()    const { return n; }
size_t ngram_mod::get_used() const { return used; }
size_t ngram_mod::size()       const { return entries.size(); }
size_t ngram_mod::size_bytes() const { return entries.size() * sizeof(entries[0]); }

// ── n-gram map-k ───────────────────────────────────────────────────────────

ngram_map::ngram_map(uint16_t sz_key, uint16_t sz_value, bool only_keys, uint16_t min_hits_)
    : size_key(sz_key), size_value(sz_value), key_only(only_keys), min_hits(min_hits_) {
    key_map.resize(NGRAM_HASH_MAP_SIZE);
}

void ngram_map_begin(ngram_map& map, const std::vector<llama_token>& tokens) {
    const size_t size_begin = tokens.size();

    size_t idx_begin_cleanup = map.size_last_begin;
    if (idx_begin_cleanup > size_begin) {
        if (size_begin > (size_t) map.size_key + map.size_value) {
            idx_begin_cleanup = size_begin - map.size_key - map.size_value;
        } else {
            idx_begin_cleanup = 0;
        }
    }

    size_t count_map_entries_upd = 0;
    if (!map.key_map.empty() && size_begin < map.idx_last_check) {
        for (size_t i = 0; i < map.key_map.size(); ++i) {
            uint32_t key_idx = map.key_map[i];
            if (key_idx != 0 && key_idx >= idx_begin_cleanup) {
                map.key_map[i] = 0;
                count_map_entries_upd++;
            }
        }
        map.key_map_last_idx = (idx_begin_cleanup > 0) ? (uint32_t) (idx_begin_cleanup - 1) : 0;
    }
    (void) count_map_entries_upd;

    if (size_begin < map.idx_last_check && !map.keys.empty()) {
        for (int32_t i = (int32_t) map.keys.size() - 1; i >= 0; --i) {
            ngram_map_key& key = map.keys[i];
            if (key.key_idx >= idx_begin_cleanup) {
                map.keys.erase(map.keys.begin() + i);
                continue;
            }
            if (map.key_only) {
                continue;
            }
            for (int16_t j = NGRAM_MAX_VALUES - 1; j >= 0; --j) {
                ngram_map_value& value = key.values[j];
                if (value.value_idx != 0 && value.value_idx >= idx_begin_cleanup) {
                    for (uint16_t k = j; k < NGRAM_MAX_VALUES - 1; ++k) {
                        key.values[k] = key.values[k + 1];
                    }
                    key.values[NGRAM_MAX_VALUES - 1].value_idx = 0;
                    key.values[NGRAM_MAX_VALUES - 1].value_num  = 0;
                }
            }
            if (key.values[0].value_idx == 0) {
                map.keys.erase(map.keys.begin() + i);
            }
        }
    }

    map.idx_last_check = size_begin;
    map.size_last_begin = size_begin;
}

void ngram_map_draft(ngram_map& map, const std::vector<llama_token>& inp,
                     llama_token sampled, std::vector<llama_token>& draft) {
    map.last_draft_created    = false;
    map.last_draft_key_idx    = 0;
    map.last_draft_value_idx  = 0;

    const size_t cur_len = inp.size();
    const uint16_t n = map.size_key;
    const uint16_t m = map.size_value;
    if (cur_len < (size_t) (2 * n + m)) {
        return;
    }
    if (map.idx_last_check > cur_len) {
        // Upstream aborts; a shrunken window can only come from a rewind, so
        // the map is stale — draft nothing and let begin() rebuild it.
        return;
    }
    map.idx_last_check = cur_len;

    std::vector<llama_token> key_tokens;
    key_tokens.reserve(n);
    for (size_t j = cur_len - n + 1; j < cur_len; ++j) {
        key_tokens.push_back(inp[j]);
    }
    key_tokens.push_back(sampled);

    size_t match_pos = 0;
    if (map.size_last_begin > cur_len) {
        return; // stale map state — upstream aborts; rebuild via begin()
    }
    if (!map.key_map.empty()) {
        const uint32_t idx_hash = ngram_map_hash(key_tokens, 0, n) % map.key_map.size();
        const uint32_t idx_key  = map.key_map[idx_hash];
        if (idx_key != 0 && idx_key < cur_len - n - m - 1) {
            bool match = true;
            for (size_t k = 0; k < n; ++k) {
                if (inp[idx_key + k] != key_tokens[k]) {
                    match = false;
                    break;
                }
            }
            if (match) {
                match_pos = idx_key;
            }
        }
    }
    if (match_pos == 0 && map.size_last_begin > (size_t) (n + m + 1)) {
        for (size_t j = map.size_last_begin - n - m - 1; j > map.key_map_last_idx && j > 0; --j) {
            bool match = true;
            for (size_t k = 0; k < n; ++k) {
                if (inp[j + k] != key_tokens[k]) {
                    match = false;
                    break;
                }
            }
            if (match) {
                match_pos = j;
                break;
            }
        }
    }
    if (match_pos == 0) {
        for (size_t j = cur_len - n - m - 1; j > map.size_last_begin && j > map.key_map_last_idx; --j) {
            bool match = true;
            for (size_t k = 0; k < n; ++k) {
                if (inp[j + k] != key_tokens[k]) {
                    match = false;
                    break;
                }
            }
            if (match) {
                match_pos = j;
                break;
            }
        }
    }

    if (!map.key_map.empty()) {
        if (map.size_last_begin > (size_t) (n + m + 1)) {
            for (size_t j = map.size_last_begin - n - m - 1; j > map.key_map_last_idx && j > 0; --j) {
                const uint32_t idx_hash = ngram_map_hash(inp, j, n) % map.key_map.size();
                if (map.key_map[idx_hash] == 0) {
                    map.key_map[idx_hash] = (uint32_t) j;
                }
            }
        }
        for (size_t j = cur_len - n - m - 1; j > map.size_last_begin && j > map.key_map_last_idx; --j) {
            const uint32_t idx_hash = ngram_map_hash(inp, j, n) % map.key_map.size();
            if (map.key_map[idx_hash] == 0) {
                map.key_map[idx_hash] = (uint32_t) j;
            }
        }
        map.key_map_last_idx = std::max((uint32_t) (cur_len - n - m - 1), map.key_map_last_idx);
    }

    if (match_pos == 0) {
        return;
    }

    size_t key_offset = map.keys.size();
    for (size_t i = 0; i < map.keys.size(); ++i) {
        bool match = true;
        for (size_t j = 0; j < n; ++j) {
            if (inp[map.keys[i].key_idx + j] != key_tokens[j]) {
                match = false;
                break;
            }
        }
        if (match) {
            key_offset = i;
            break;
        }
    }
    if (key_offset == map.keys.size()) {
        ngram_map_key new_key;
        new_key.key_idx  = match_pos;
        new_key.stat_idx = 0;
        new_key.key_num  = 0;
        for (int i = 0; i < NGRAM_MAX_VALUES; ++i) {
            new_key.values[i].value_num = 0;
            new_key.values[i].n_accepted = m;
        }
        map.keys.push_back(new_key);
    }

    ngram_map_key& curr_key = map.keys[key_offset];
    curr_key.key_num = (uint16_t) std::min((int) map.keys[key_offset].key_num + 1,
                                           (int) NGRAM_MAX_VALUE_COUNT);

    if (map.key_only) {
        const int n_draft_tokens = std::min((int) m, (int) curr_key.values[0].n_accepted);
        for (int i = 0; i < n_draft_tokens; ++i) {
            draft.push_back(inp[match_pos + n + i]);
        }
        map.last_draft_created   = true;
        map.last_draft_key_idx   = key_offset;
        map.last_draft_value_idx = 0;
        return;
    }

    if (curr_key.key_num < map.min_hits) {
        return;
    }

    for (size_t i = curr_key.stat_idx; i <= match_pos; ++i) {
        bool match_key = true;
        for (size_t k = 0; k < n; ++k) {
            if (inp[i + k] != key_tokens[k]) {
                match_key = false;
                break;
            }
        }
        if (!match_key) {
            continue;
        }

        const size_t idx_begin_value_key = i + n;
        int idx_value = -1;
        for (int v = 0; v < NGRAM_MAX_VALUES; ++v) {
            const size_t idx_begin_value_v = curr_key.values[v].value_idx;
            if (idx_begin_value_v == 0) {
                curr_key.values[v].value_idx = idx_begin_value_key;
                curr_key.values[v].value_num = 0;
                curr_key.values[v].n_accepted = m;
                idx_value = v;
                break;
            }
            bool match = true;
            for (size_t j = 0; j < m; ++j) {
                if (inp[idx_begin_value_key + j] != inp[idx_begin_value_v + j]) {
                    match = false;
                    break;
                }
            }
            if (match) {
                idx_value = v;
                break;
            }
        }
        if (idx_value >= 0) {
            curr_key.values[idx_value].value_num =
                (uint16_t) std::min((int) curr_key.values[idx_value].value_num + 1,
                                    (int) NGRAM_MAX_VALUE_COUNT);
        }
    }
    curr_key.stat_idx = match_pos;

    uint16_t max_occur = 0;
    int slot_max = 0;
    for (int v = 0; v < NGRAM_MAX_VALUES; ++v) {
        if (curr_key.values[v].value_num > max_occur) {
            max_occur = curr_key.values[v].value_num;
            slot_max = v;
        }
    }
    uint32_t sum_occur = 0;
    for (int v = 0; v < NGRAM_MAX_VALUES; ++v) {
        if (v == slot_max) continue;
        sum_occur += curr_key.values[v].value_num;
    }

    if (sum_occur > 0 && max_occur < 2 * sum_occur) {
        return;
    }

    const int n_draft_tokens = std::min((int) m, (int) curr_key.values[slot_max].n_accepted);
    for (int i = 0; i < n_draft_tokens; ++i) {
        draft.push_back(inp[match_pos + n + i]);
    }

    map.last_draft_created   = true;
    map.last_draft_key_idx   = key_offset;
    map.last_draft_value_idx = (uint16_t) slot_max;
}

void ngram_map_accept(ngram_map& map, uint16_t n_accepted) {
    if (!map.last_draft_created) {
        return;
    }
    if (map.last_draft_key_idx >= map.keys.size()) {
        return; // stale (map shrank between draft and accept) — upstream would abort
    }
    ngram_map_key& curr_key = map.keys[map.last_draft_key_idx];
    curr_key.values[map.last_draft_value_idx].n_accepted = n_accepted;
}

// ── n-gram cache ───────────────────────────────────────────────────────────

ngram_key::ngram_key() {
    for (int i = 0; i < NGRAM_MAX; ++i) {
        tokens[i] = LLAMA_TOKEN_NULL;
    }
}

ngram_key::ngram_key(const llama_token* input, int ngram_size) {
    for (int i = 0; i < NGRAM_MAX; ++i) {
        tokens[i] = i < ngram_size ? input[i] : LLAMA_TOKEN_NULL;
    }
}

bool ngram_key::operator==(const ngram_key& other) const {
    for (int i = 0; i < NGRAM_MAX; ++i) {
        if (tokens[i] != other.tokens[i]) {
            return false;
        }
    }
    return true;
}

size_t ngram_token_hash::operator()(llama_token token) const {
    return (size_t) token * 11400714819323198485ULL;
}

size_t ngram_key_hash::operator()(const ngram_key& ngram) const {
    size_t hash = ngram_token_hash{}(ngram.tokens[0]);
    for (int i = 1; i < NGRAM_MAX; ++i) {
        hash ^= ngram_token_hash{}(ngram.tokens[i]);
    }
    return hash;
}

void ngram_cache_update(ngram_cache& cache, int ngram_min, int ngram_max,
                        std::vector<llama_token>& inp, int nnew) {
    const int64_t inp_size = (int64_t) inp.size();
    for (int64_t ngram_size = ngram_min; ngram_size <= ngram_max; ++ngram_size) {
        const int64_t i_start = std::max(inp_size - nnew, ngram_size);
        for (int64_t i = i_start; i < inp_size; ++i) {
            const int64_t ngram_start = i - ngram_size;
            const ngram_key ngram(&inp[ngram_start], (int) ngram_size);
            const llama_token token = inp[i];

            auto part_it = cache.find(ngram);
            if (part_it == cache.end()) {
                ngram_cache_part part;
                part.emplace(token, 1);
                cache.emplace(ngram, std::move(part));
            } else {
                auto token_count_it = part_it->second.find(token);
                if (token_count_it == part_it->second.end()) {
                    part_it->second.emplace(token, 1);
                } else {
                    token_count_it->second++;
                }
            }
        }
    }
}

// token i of the combined sequence [inp, draft[1..]] (draft[0] == sampled tail)
static llama_token get_token(const std::vector<llama_token>& inp,
                             const std::vector<llama_token>& draft, size_t i) {
    return i < inp.size() ? inp[i] : draft[1 + i - inp.size()];
}

static constexpr int draft_min_sample_size_lax[NGRAM_MAX]    = { 2, 2, 1, 1 };
static constexpr int draft_min_percent_lax[NGRAM_MAX]        = { 66, 50, 50, 50 };
static constexpr int draft_min_sample_size_strict[NGRAM_MAX] = { 4, 3, 2, 2 };
static constexpr int draft_min_percent_strict[NGRAM_MAX]     = { 75, 66, 66, 66 };

static llama_token try_draft(const ngram_cache& nc_static, const ngram_key& ngram_static) {
    auto part_static_it = nc_static.find(ngram_static);
    if (part_static_it == nc_static.end()) {
        return LLAMA_TOKEN_NULL;
    }
    const ngram_cache_part& part_static = part_static_it->second;

    int max_count_static = 0;
    int sum_count_static = 0;
    llama_token max_token = LLAMA_TOKEN_NULL;

    for (const auto& token_count_static : part_static) {
        const llama_token token = token_count_static.first;
        const int32_t count_static = token_count_static.second;
        if (count_static > max_count_static) {
            max_token = token;
            max_count_static = count_static;
        }
        sum_count_static += count_static;
    }

    if (sum_count_static < draft_min_sample_size_lax[NGRAM_STATIC - 1]) {
        return LLAMA_TOKEN_NULL;
    }
    if (100 * max_count_static < draft_min_percent_lax[NGRAM_STATIC - 1] * sum_count_static) {
        return LLAMA_TOKEN_NULL;
    }
    return max_token;
}

static llama_token try_draft(const ngram_cache& nc_primary,
                             const std::vector<ngram_key>& ngrams_primary,
                             const ngram_cache_part& part_static,
                             const int* min_sample_size, const int* min_percent) {
    llama_token drafted_token = LLAMA_TOKEN_NULL;

    for (int i = (int) ngrams_primary.size() - 1; i >= 0 && drafted_token == LLAMA_TOKEN_NULL; --i) {
        const ngram_key& ngram_primary = ngrams_primary[i];
        auto part_primary_it = nc_primary.find(ngram_primary);
        if (part_primary_it == nc_primary.end()) {
            continue;
        }
        const ngram_cache_part& part_primary = part_primary_it->second;

        int max_count_primary = 0;
        int max_count_static  = 0;
        int sum_count_primary = 0;
        llama_token max_token = LLAMA_TOKEN_NULL;

        for (const auto& token_count_primary : part_primary) {
            const llama_token token = token_count_primary.first;
            auto token_count_static_it = part_static.find(token);

            const int32_t count_primary = token_count_primary.second;
            const int32_t count_static =
                token_count_static_it != part_static.end() ? 100 * token_count_static_it->second : 1;

            if (count_primary * count_static > max_count_primary * max_count_static) {
                max_token = token;
                max_count_primary = count_primary;
                max_count_static  = count_static;
            }
            sum_count_primary += count_primary;
        }

        if (sum_count_primary < min_sample_size[i]) {
            continue;
        }
        if (100 * max_count_primary < min_percent[i] * sum_count_primary) {
            continue;
        }
        drafted_token = max_token;
    }

    return drafted_token;
}

void ngram_cache_draft(std::vector<llama_token>& inp, std::vector<llama_token>& draft,
                       int n_draft, int ngram_min, int ngram_max,
                       ngram_cache& nc_context, ngram_cache& nc_dynamic, ngram_cache& nc_static) {
    // Upstream asserts draft.size() == 1 (the sampled tail is draft[0]).
    if (draft.size() != 1) {
        return;
    }
    const int inp_size = (int) inp.size();
    if (inp_size < NGRAM_STATIC) {
        return;
    }

    static const ngram_cache_part part_static_empty;

    while ((int) draft.size() - 1 < n_draft) {
        llama_token drafted_token = LLAMA_TOKEN_NULL;

        const int ngram_start_static = inp_size - NGRAM_STATIC + (int) draft.size() - 1;
        ngram_key ngram_static;
        for (int j = ngram_start_static; j < ngram_start_static + NGRAM_STATIC; ++j) {
            ngram_static.tokens[j - ngram_start_static] = get_token(inp, draft, (size_t) j);
        }
        auto part_static_it = nc_static.find(ngram_static);
        const ngram_cache_part& part_static =
            part_static_it != nc_static.end() ? part_static_it->second : part_static_empty;

        std::vector<ngram_key> ngrams_cd;
        for (int ngram_size_cd = ngram_min; ngram_size_cd <= ngram_max; ++ngram_size_cd) {
            const int ngram_start_cd = inp_size - ngram_size_cd + (int) draft.size() - 1;
            ngram_key ngram_cd;
            for (int j = ngram_start_cd; j < ngram_start_cd + ngram_size_cd; ++j) {
                ngram_cd.tokens[j - ngram_start_cd] = get_token(inp, draft, (size_t) j);
            }
            ngrams_cd.push_back(ngram_cd);
        }
        if (drafted_token == LLAMA_TOKEN_NULL) {
            drafted_token = try_draft(nc_context, ngrams_cd, part_static,
                                      draft_min_sample_size_lax, draft_min_percent_lax);
        }
        if (drafted_token == LLAMA_TOKEN_NULL) {
            drafted_token = try_draft(nc_dynamic, ngrams_cd, part_static,
                                      draft_min_sample_size_strict, draft_min_percent_strict);
        }
        if (drafted_token == LLAMA_TOKEN_NULL) {
            drafted_token = try_draft(nc_static, ngram_static);
        }
        if (drafted_token == LLAMA_TOKEN_NULL) {
            break;
        }
        draft.push_back(drafted_token);
    }
}

void ngram_cache_save(const ngram_cache& cache, const std::string& filename) {
    std::ofstream file_out(filename, std::ios::binary);
    if (!file_out) {
        return; // save is best-effort telemetry — load is the typed boundary
    }
    for (const auto& item : cache) {
        const ngram_key ngram = item.first;
        const ngram_cache_part& token_counts = item.second;
        if (token_counts.empty()) {
            continue;
        }
        const int32_t ntokens = (int32_t) token_counts.size();
        file_out.write(reinterpret_cast<const char*>(&ngram), sizeof(ngram_key));
        file_out.write(reinterpret_cast<const char*>(&ntokens), sizeof(int32_t));
        for (const auto& item2 : token_counts) {
            const llama_token token = item2.first;
            const int32_t count = item2.second;
            file_out.write(reinterpret_cast<const char*>(&token), sizeof(llama_token));
            file_out.write(reinterpret_cast<const char*>(&count), sizeof(int32_t));
        }
    }
}

bool ngram_cache_load(const std::string& filename, ngram_cache& out_cache) {
    std::ifstream hashmap_file(filename, std::ios::binary);
    if (!hashmap_file) {
        return false;
    }
    ngram_cache cache;

    ngram_key ngram;
    int32_t ntokens = 0;
    llama_token token = 0;
    int32_t count = 0;

    char* ngramc   = reinterpret_cast<char*>(&ngram);
    char* ntokensc = reinterpret_cast<char*>(&ntokens);
    char* tokenc   = reinterpret_cast<char*>(&token);
    char* countc   = reinterpret_cast<char*>(&count);
    while (hashmap_file.read(ngramc, sizeof(ngram_key))) {
        if (!hashmap_file.read(ntokensc, sizeof(int32_t)) || ntokens <= 0) {
            return false;
        }
        ngram_cache_part token_counts;
        for (int i = 0; i < ntokens; ++i) {
            if (!hashmap_file.read(tokenc, sizeof(llama_token))) return false;
            if (!hashmap_file.read(countc, sizeof(int32_t)) || count <= 0) return false;
            token_counts.emplace(token, count);
        }
        cache.emplace(ngram, std::move(token_counts));
    }
    // A short/garbage file dies mid-record: the last read failed but pulled
    // a partial record — that is corruption, not a valid empty cache.
    if (hashmap_file.gcount() != 0) {
        return false;
    }

    out_cache = std::move(cache);
    return true;
}

void ngram_cache_merge(ngram_cache& target, const ngram_cache& add) {
    for (const auto& ngram_part : add) {
        const ngram_key ngram = ngram_part.first;
        const ngram_cache_part& part = ngram_part.second;

        auto part_merged_it = target.find(ngram);
        if (part_merged_it == target.end()) {
            target.emplace(ngram, part);
            continue;
        }
        for (const auto& token_count : part) {
            const llama_token token = token_count.first;
            const int32_t count = token_count.second;
            auto token_count_merged_it = part_merged_it->second.find(token);
            if (token_count_merged_it == part_merged_it->second.end()) {
                part_merged_it->second.emplace(token, count);
                continue;
            }
            token_count_merged_it->second += count;
        }
    }
}

} // namespace pvx
