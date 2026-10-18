// pvx_json_schema_grammar.cpp — JSON Schema → GBNF grammar converter (ABI v7, feature 11).
//
// Pattern-port of the vendored common/json-schema-to-grammar.{h,cpp} (never
// linked — self-contained, no nlohmann/common dependency): a minimal JSON
// parser (sufficient for schema documents), the same primitive/space rules,
// the same object/array/union/enum/int-range builders, and the same
// fail-closed contract — invalid JSON is a typed parse error, an unsupported
// schema keyword is a typed PVX_ERR_NOT_SUPPORTED naming the keyword, never a
// best-effort loose grammar (plan feature 11(f)).
//
// Supported subset (typed fail-closed outside it): type (incl. union arrays),
// properties/required/additionalProperties, items/prefixItems/minItems/
// maxItems, enum/const, minimum/maximum/exclusiveMinimum/exclusiveMaximum
// (integers), minLength/maxLength, format (date/time/date-time/uuid),
// $defs/definitions + $ref (JSON-pointer, recursive schemas included),
// anyOf/oneOf, allOf (object/enum merge — upstream's add_component subset).
//
// KV-neutrality: pure string -> string conversion; the generated grammar is
// consumed through the existing pvx_grammar_* chain (acceptance happens at
// commit only).

#include "pvx_internal.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <deque>
#include <vector>

namespace pvx {
namespace schema {

// ── Minimal JSON value (parse + dump for enum/const literals) ──────────────

class Json {
public:
    enum Kind { NUL, BOOL, NUM_INT, NUM_DBL, STR, ARR, OBJ };
    Kind kind = NUL;
    bool b = false;
    int64_t i = 0;
    double d = 0.0;
    std::string s;
    std::vector<Json> arr;
    std::vector<std::pair<std::string, Json>> obj; // insertion-ordered

    bool is_obj() const { return kind == OBJ; }
    bool is_arr() const { return kind == ARR; }
    bool is_str() const { return kind == STR; }
    bool is_num() const { return kind == NUM_INT || kind == NUM_DBL; }
    bool is_int() const { return kind == NUM_INT; }
    bool is_null() const { return kind == NUL; }
    bool is_bool() const { return kind == BOOL; }

    const Json* find(const std::string& key) const {
        if (kind != OBJ) return nullptr;
        for (const auto& kv : obj) if (kv.first == key) return &kv.second;
        return nullptr;
    }

    static std::string dump(const Json& v) {
        switch (v.kind) {
            case NUL: return "null";
            case BOOL: return v.b ? "true" : "false";
            case NUM_INT: return std::to_string(v.i);
            case NUM_DBL: {
                std::ostringstream os;
                os.precision(17);
                os << v.d;
                return os.str();
            }
            case STR: {
                std::string out = "\"";
                for (char c : v.s) {
                    switch (c) {
                        case '"': out += "\\\""; break;
                        case '\\': out += "\\\\"; break;
                        case '\n': out += "\\n"; break;
                        case '\r': out += "\\r"; break;
                        case '\t': out += "\\t"; break;
                        default:
                            if ((unsigned char) c < 0x20) {
                                char buf[8];
                                snprintf(buf, sizeof(buf), "\\u%04x", c);
                                out += buf;
                            } else {
                                out += c;
                            }
                    }
                }
                return out + "\"";
            }
            case ARR: {
                std::string out = "[";
                for (size_t k = 0; k < v.arr.size(); k++) {
                    if (k) out += ",";
                    out += dump(v.arr[k]);
                }
                return out + "]";
            }
            case OBJ: {
                std::string out = "{";
                for (size_t k = 0; k < v.obj.size(); k++) {
                    if (k) out += ",";
                    out += dump(Json::str(v.obj[k].first)) + ":" + dump(v.obj[k].second);
                }
                return out + "}";
            }
        }
        return "";
    }

    static Json str(const std::string& s) { Json j; j.kind = STR; j.s = s; return j; }
};

// Typed parse failure: position of the offending input.
struct ParseError {
    size_t pos;
    std::string msg;
};

class Parser {
public:
    explicit Parser(const std::string& text) : t(text) {}

    Json parse() {
        skip_ws();
        Json v = parse_value(0);
        skip_ws();
        if (p != t.size()) err("trailing characters after JSON value");
        return v;
    }

private:
    const std::string t;
    size_t p = 0;

    void err(const std::string& m) { throw ParseError{ p, m }; }
    void skip_ws() { while (p < t.size() && (t[p]==' '||t[p]=='\t'||t[p]=='\n'||t[p]=='\r')) p++; }
    char peek() {
        if (p >= t.size()) err("unexpected end of input");
        return t[p];
    }
    void expect(char c) { if (peek() != c) err(std::string("expected '") + c + "'"); p++; }

    Json parse_value(int depth) {
        if (depth > 200) err("nesting too deep");
        switch (peek()) {
            case '{': return parse_obj(depth);
            case '[': return parse_arr(depth);
            case '"': { Json v; v.kind = Json::STR; v.s = parse_string(); return v; }
            case 't': { parse_word("true");  Json v; v.kind = Json::BOOL; v.b = true;  return v; }
            case 'f': { parse_word("false"); Json v; v.kind = Json::BOOL; v.b = false; return v; }
            case 'n': { parse_word("null");  return Json{}; }
            default: return parse_number();
        }
    }

    void parse_word(const char* lit) {
        const size_t n = strlen(lit);
        if (t.compare(p, n, lit) != 0) err(std::string("invalid literal, expected '") + lit + "'");
        p += n;
    }

    Json parse_obj(int depth) {
        expect('{');
        Json v; v.kind = Json::OBJ;
        skip_ws();
        if (peek() == '}') { p++; return v; }
        while (true) {
            skip_ws();
            std::string key = parse_string();
            skip_ws();
            expect(':');
            skip_ws();
            v.obj.emplace_back(std::move(key), parse_value(depth + 1));
            skip_ws();
            char c = peek();
            if (c == ',') { p++; continue; }
            if (c == '}') { p++; break; }
            err("expected ',' or '}' in object");
        }
        return v;
    }

    Json parse_arr(int depth) {
        expect('[');
        Json v; v.kind = Json::ARR;
        skip_ws();
        if (peek() == ']') { p++; return v; }
        while (true) {
            skip_ws();
            v.arr.push_back(parse_value(depth + 1));
            skip_ws();
            char c = peek();
            if (c == ',') { p++; continue; }
            if (c == ']') { p++; break; }
            err("expected ',' or ']' in array");
        }
        return v;
    }

    std::string parse_string() {
        expect('"');
        std::string out;
        while (true) {
            if (p >= t.size()) err("unterminated string");
            char c = t[p++];
            if (c == '"') break;
            if (c == '\\') {
                if (p >= t.size()) err("trailing backslash");
                char e = t[p++];
                switch (e) {
                    case '"': out += '"'; break;
                    case '\\': out += '\\'; break;
                    case '/': out += '/'; break;
                    case 'b': out += '\b'; break;
                    case 'f': out += '\f'; break;
                    case 'n': out += '\n'; break;
                    case 'r': out += '\r'; break;
                    case 't': out += '\t'; break;
                    case 'u': {
                        if (p + 4 > t.size()) err("bad \\u escape");
                        unsigned cp = 0;
                        for (int k = 0; k < 4; k++) {
                            char h = t[p + k];
                            cp <<= 4;
                            if (h >= '0' && h <= '9') cp |= (unsigned)(h - '0');
                            else if (h >= 'a' && h <= 'f') cp |= (unsigned)(h - 'a' + 10);
                            else if (h >= 'A' && h <= 'F') cp |= (unsigned)(h - 'A' + 10);
                            else err("bad \\u escape");
                        }
                        p += 4;
                        // surrogate pair
                        if (cp >= 0xD800 && cp <= 0xDBFF && p + 6 <= t.size() &&
                            t[p] == '\\' && t[p+1] == 'u') {
                            unsigned lo = 0;
                            bool ok = true;
                            for (int k = 0; k < 4; k++) {
                                char h = t[p + 2 + k];
                                lo <<= 4;
                                if (h >= '0' && h <= '9') lo |= (unsigned)(h - '0');
                                else if (h >= 'a' && h <= 'f') lo |= (unsigned)(h - 'a' + 10);
                                else if (h >= 'A' && h <= 'F') lo |= (unsigned)(h - 'A' + 10);
                                else { ok = false; break; }
                            }
                            if (ok && lo >= 0xDC00 && lo <= 0xDFFF) {
                                cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                                p += 6;
                            }
                        }
                        // UTF-8 encode
                        if (cp < 0x80) out += (char) cp;
                        else if (cp < 0x800) {
                            out += (char)(0xC0 | (cp >> 6));
                            out += (char)(0x80 | (cp & 0x3F));
                        } else if (cp < 0x10000) {
                            out += (char)(0xE0 | (cp >> 12));
                            out += (char)(0x80 | ((cp >> 6) & 0x3F));
                            out += (char)(0x80 | (cp & 0x3F));
                        } else {
                            out += (char)(0xF0 | (cp >> 18));
                            out += (char)(0x80 | ((cp >> 12) & 0x3F));
                            out += (char)(0x80 | ((cp >> 6) & 0x3F));
                            out += (char)(0x80 | (cp & 0x3F));
                        }
                        break;
                    }
                    default: err("unsupported escape");
                }
            } else {
                out += c;
            }
        }
        return out;
    }

    Json parse_number() {
        size_t start = p;
        if (peek() == '-') p++;
        if (p >= t.size() || t[p] < '0' || t[p] > '9') err("invalid number");
        bool is_int = true;
        while (p < t.size() && t[p] >= '0' && t[p] <= '9') p++;
        if (p < t.size() && t[p] == '.') {
            is_int = false;
            p++;
            if (p >= t.size() || t[p] < '0' || t[p] > '9') err("invalid number");
            while (p < t.size() && t[p] >= '0' && t[p] <= '9') p++;
        }
        if (p < t.size() && (t[p] == 'e' || t[p] == 'E')) {
            is_int = false;
            p++;
            if (p < t.size() && (t[p] == '+' || t[p] == '-')) p++;
            if (p >= t.size() || t[p] < '0' || t[p] > '9') err("invalid number");
            while (p < t.size() && t[p] >= '0' && t[p] <= '9') p++;
        }
        Json v;
        const std::string tok = t.substr(start, p - start);
        if (is_int) {
            v.kind = Json::NUM_INT;
            try {
                v.i = std::stoll(tok);
            } catch (...) {
                // integer overflow: keep as double
                v.kind = Json::NUM_DBL;
                v.d = std::stod(tok);
            }
        } else {
            v.kind = Json::NUM_DBL;
            v.d = std::stod(tok);
        }
        return v;
    }
};

// ── Typed keyword failure (surfaced as PVX_ERR_NOT_SUPPORTED) ───────────────

struct KeywordError {
    std::string keyword;
    std::string msg;
};

[[noreturn]] void unsupported(const std::string& kw, const std::string& why) {
    throw KeywordError{ kw, why };
}

// ── Grammar helpers (upstream pattern, verbatim where possible) ────────────

static std::string string_repeat(const std::string& s, size_t n) {
    std::string out;
    out.reserve(s.size() * n);
    for (size_t k = 0; k < n; k++) out += s;
    return out;
}

static std::string build_repetition(const std::string& item_rule, int min_items,
                                    int max_items, const std::string& separator_rule = "") {
    const bool has_max = max_items != std::numeric_limits<int>::max();
    if (max_items == 0) return "";
    if (min_items == 0 && max_items == 1) return item_rule + "?";
    if (separator_rule.empty()) {
        if (min_items == 1 && !has_max) return item_rule + "+";
        if (min_items == 0 && !has_max) return item_rule + "*";
        return item_rule + "{" + std::to_string(min_items) + "," +
               (has_max ? std::to_string(max_items) : "") + "}";
    }
    auto result = item_rule + " " +
        build_repetition("(" + separator_rule + " " + item_rule + ")",
                          min_items == 0 ? 0 : min_items - 1,
                          has_max ? max_items - 1 : max_items);
    if (min_items == 0) result = "(" + result + ")?";
    return result;
}

static void build_min_max_int(int64_t min_value, int64_t max_value,
                              std::ostringstream& out, int decimals_left = 16,
                              bool top_level = true) {
    const bool has_min = min_value != std::numeric_limits<int64_t>::min();
    const bool has_max = max_value != std::numeric_limits<int64_t>::max();

    auto digit_range = [&](char from, char to) {
        out << "[";
        if (from == to) out << from; else out << from << "-" << to;
        out << "]";
    };
    auto more_digits = [&](int min_digits, int max_digits) {
        out << "[0-9]";
        if (min_digits == max_digits && min_digits == 1) return;
        out << "{";
        out << min_digits;
        if (max_digits != min_digits) {
            out << ",";
            if (max_digits != std::numeric_limits<int>::max()) out << max_digits;
        }
        out << "}";
    };
    std::function<void(const std::string&, const std::string&)> uniform_range =
        [&](const std::string& from, const std::string& to) {
            size_t i = 0;
            while (i < from.length() && i < to.length() && from[i] == to[i]) i++;
            if (i > 0) out << "\"" << from.substr(0, i) << "\"";
            if (i < from.length() && i < to.length()) {
                if (i > 0) out << " ";
                const size_t sub_len = from.length() - i - 1;
                if (sub_len > 0) {
                    const std::string from_sub = from.substr(i + 1);
                    const std::string to_sub = to.substr(i + 1);
                    const std::string sub_zeros = string_repeat("0", sub_len);
                    const std::string sub_nines = string_repeat("9", sub_len);
                    bool to_reached = false;
                    out << "(";
                    if (from_sub == sub_zeros) {
                        digit_range(from[i], to[i] - 1);
                        out << " ";
                        more_digits(sub_len, sub_len);
                    } else {
                        out << "[" << from[i] << "] ";
                        out << "(";
                        uniform_range(from_sub, sub_nines);
                        out << ")";
                        if (from[i] < to[i] - 1) {
                            out << " | ";
                            if (to_sub == sub_nines) {
                                digit_range(from[i] + 1, to[i]);
                                to_reached = true;
                            } else {
                                digit_range(from[i] + 1, to[i] - 1);
                            }
                            out << " ";
                            more_digits(sub_len, sub_len);
                        }
                    }
                    if (!to_reached) {
                        out << " | ";
                        digit_range(to[i], to[i]);
                        out << " ";
                        uniform_range(sub_zeros, to_sub);
                    }
                    out << ")";
                } else {
                    out << "[" << from[i] << "-" << to[i] << "]";
                }
            }
        };

    if (has_min && has_max) {
        if (min_value < 0 && max_value < 0) {
            out << "\"-\" (";
            build_min_max_int(-max_value, -min_value, out, decimals_left, true);
            out << ")";
            return;
        }
        if (min_value < 0) {
            out << "\"-\" (";
            build_min_max_int(0, -min_value, out, decimals_left, true);
            out << ") | ";
            min_value = 0;
        }
        auto min_s = std::to_string(min_value);
        auto max_s = std::to_string(max_value);
        auto min_digits = min_s.length();
        auto max_digits = max_s.length();
        for (auto digits = min_digits; digits < max_digits; digits++) {
            uniform_range(min_s, string_repeat("9", digits));
            min_s = "1" + string_repeat("0", digits);
            out << " | ";
        }
        uniform_range(min_s, max_s);
        return;
    }

    const int less_decimals = std::max(decimals_left - 1, 1);

    if (has_min) {
        if (min_value < 0) {
            out << "\"-\" (";
            build_min_max_int(std::numeric_limits<int64_t>::min(), -min_value, out, decimals_left, false);
            out << ") | [0] | [1-9] ";
            more_digits(0, decimals_left - 1);
        } else if (min_value == 0) {
            if (top_level) {
                out << "[0] | [1-9] ";
                more_digits(0, less_decimals);
            } else {
                more_digits(1, decimals_left);
            }
        } else if (min_value <= 9) {
            char c = '0' + (char) min_value;
            char range_start = top_level ? '1' : '0';
            if (c > range_start) {
                digit_range(range_start, c - 1);
                out << " ";
                more_digits(1, less_decimals);
                out << " | ";
            }
            digit_range(c, '9');
            out << " ";
            more_digits(0, less_decimals);
        } else {
            auto min_s = std::to_string(min_value);
            auto len = min_s.length();
            auto c = min_s[0];
            if (c > '1') {
                digit_range(top_level ? '1' : '0', c - 1);
                out << " ";
                more_digits((int) len, less_decimals);
                out << " | ";
            }
            digit_range(c, c);
            out << " (";
            build_min_max_int(std::stoll(min_s.substr(1)),
                              std::numeric_limits<int64_t>::max(), out, less_decimals, false);
            out << ")";
            if (c < '9') {
                out << " | ";
                digit_range(c + 1, '9');
                out << " ";
                more_digits((int) len - 1, less_decimals);
            }
        }
        return;
    }

    if (has_max) {
        if (max_value >= 0) {
            if (top_level) {
                out << "\"-\" [1-9] ";
                more_digits(0, decimals_left);
                out << " | ";
            }
            build_min_max_int(0, max_value, out, decimals_left, true);
        } else {
            out << "\"-\" (";
            build_min_max_int(-max_value, std::numeric_limits<int64_t>::max(), out, decimals_left, false);
            out << ")";
        }
        return;
    }

    throw std::runtime_error("At least one of min_value or max_value must be set");
}

static const std::string SPACE_RULE = "| \" \" | \"\\n\"{1,2} [ \\t]{0,20}";

struct BuiltinRule {
    std::string content;
    std::vector<std::string> deps;
};

static const std::unordered_map<std::string, BuiltinRule> PRIMITIVE_RULES = {
    {"boolean", {"(\"true\" | \"false\")", {}}},
    {"decimal-part", {"[0-9]{1,16}", {}}},
    {"integral-part", {"[0] | [1-9] [0-9]{0,15}", {}}},
    {"number", {"(\"-\"? integral-part) (\".\" decimal-part)? ([eE] [-+]? integral-part)?", {"integral-part", "decimal-part"}}},
    {"integer", {"(\"-\"? integral-part)", {"integral-part"}}},
    {"value", {"object | array | string | number | boolean | null", {"object", "array", "string", "number", "boolean", "null"}}},
    {"object", {"\"{\" space ( string \":\" space value (\",\" space string \":\" space value)* )? space \"}\"", {"string", "value"}}},
    {"array", {"\"[\" space ( value (\",\" space value)* )? space \"]\"", {"value"}}},
    {"uuid", {"\"\\\"\" [0-9a-fA-F]{8} \"-\" [0-9a-fA-F]{4} \"-\" [0-9a-fA-F]{4} \"-\" [0-9a-fA-F]{4} \"-\" [0-9a-fA-F]{12} \"\\\"\"", {}}},
    {"char", {"[^\"\\\\\\x7F\\x00-\\x1F] | [\\\\] ([\"\\\\bfnrt] | \"u\" [0-9a-fA-F]{4})", {}}},
    {"string", {"\"\\\"\" char* \"\\\"\"", {"char"}}},
    {"null", {"\"null\"", {}}},
};

static const std::unordered_map<std::string, BuiltinRule> STRING_FORMAT_RULES = {
    {"date", {"[0-9]{4} \"-\" ( \"0\" [1-9] | \"1\" [0-2] ) \"-\" ( \"0\" [1-9] | [1-2] [0-9] | \"3\" [0-1] )", {}}},
    {"time", {"([01] [0-9] | \"2\" [0-3]) \":\" [0-5] [0-9] \":\" [0-5] [0-9] ( \".\" [0-9]{3} )? ( \"Z\" | ( \"+\" | \"-\" ) ( [01] [0-9] | \"2\" [0-3] ) \":\" [0-5] [0-9] )", {}}},
    {"date-time", {"date \"T\" time", {"date", "time"}}},
    {"date-string", {"\"\\\"\" date \"\\\"\"", {"date"}}},
    {"time-string", {"\"\\\"\" time \"\\\"\"", {"time"}}},
    {"date-time-string", {"\"\\\"\" date-time \"\\\"\"", {"date-time"}}},
};

static bool is_reserved_name(const std::string& name) {
    if (name == "root" || name == "space") return true;
    return PRIMITIVE_RULES.count(name) > 0 || STRING_FORMAT_RULES.count(name) > 0;
}

static const std::regex GRAMMAR_LITERAL_ESCAPE_RE("[\r\n\"\\\\]");
static const std::unordered_map<char, std::string> GRAMMAR_LITERAL_ESCAPES = {
    {'\r', "\\r"}, {'\n', "\\n"}, {'"', "\\\""}, {'\\', "\\\\"}
};

static std::string format_literal(const std::string& literal) {
    std::string escaped;
    for (char c : literal) {
        auto it = GRAMMAR_LITERAL_ESCAPES.find(c);
        escaped += (it != GRAMMAR_LITERAL_ESCAPES.end()) ? it->second : std::string(1, c);
    }
    return "\"" + escaped + "\"";
}

// ── Converter ───────────────────────────────────────────────────────────────

class Converter {
public:
    explicit Converter(const Json& root) : m_root(root) {
        m_rules["space"] = SPACE_RULE;
    }

    std::string convert(const std::string& root_name) {
        const std::string n = root_name.empty() ? "root" : root_name;
        visit(m_root, n, n);
        if (!m_errors.empty()) {
            throw std::runtime_error("JSON schema conversion failed:\n" + m_errors[0]);
        }
        return format_grammar();
    }

private:
    const Json& m_root;
    std::map<std::string, std::string> m_rules; // ordered for stable output
    std::unordered_set<std::string> m_refs_in_progress;
    std::vector<std::string> m_errors;

    std::string add_rule(const std::string& name, const std::string& rule) {
        std::string final_name = (is_reserved_name(name) && name != "root") || name.empty()
            ? name + "-" : name;
        m_rules[final_name] = rule;
        return final_name;
    }

    // Primitives register under their OWN names (never renamed — a rule named
    // `string` IS the primitive; property collisions get `-` names at the
    // visit() layer like upstream).
    std::string add_primitive(const std::string& name, const BuiltinRule& rule) {
        const std::string final_name = (name == "root") ? "root" : name;
        const auto n = m_rules.count(final_name) ? final_name
                       : (m_rules[final_name] = rule.content, final_name);
        for (const auto& dep : rule.deps) {
            const BuiltinRule* dep_rule = nullptr;
            auto it = PRIMITIVE_RULES.find(dep);
            if (it != PRIMITIVE_RULES.end()) dep_rule = &it->second;
            else {
                auto it2 = STRING_FORMAT_RULES.find(dep);
                if (it2 != STRING_FORMAT_RULES.end()) dep_rule = &it2->second;
            }
            if (!dep_rule) {
                m_errors.push_back("Rule " + dep + " not known");
                continue;
            }
            if (m_rules.find(dep) == m_rules.end()) {
                add_primitive(dep, *dep_rule);
            }
        }
        return n;
    }

    static std::string visit_primitive_rule(const std::string& rule_name,
                                            const std::string& type) {
        return rule_name; // caller maps through add_primitive
    }

    // not-strings trie (upstream _not_strings, ASCII-simplified: property
    // names with bytes > 0x7F are typed refusals — the vendored trie is
    // codepoint-aware; the pvx port keeps ASCII key sets honest).
    std::string not_strings(const std::vector<std::string>& strings) {
        for (const auto& s : strings) {
            for (unsigned char c : s) {
                if (c > 0x7F) {
                    unsupported("additionalProperties",
                                "non-ASCII property names are not supported in the "
                                "not-property key trie this slice");
                }
                if (c == '"' || c == '\\') {
                    unsupported("additionalProperties",
                                "property names with quote/backslash characters are not "
                                "supported in the not-property key trie this slice");
                }
            }
        }
        const std::string char_rule = add_primitive("char", PRIMITIVE_RULES.at("char"));
        // Build a byte trie.
        struct Node { std::map<char, int> children; bool terminal = false; };
        std::vector<Node> nodes(1);
        for (const auto& s : strings) {
            int cur = 0;
            for (char ch : s) {
                auto& kids = nodes[cur].children;
                auto it = kids.find(ch);
                if (it == kids.end()) {
                    nodes.push_back(Node{});
                    kids[ch] = (int) nodes.size() - 1;
                    cur = (int) nodes.size() - 1;
                } else {
                    cur = it->second;
                }
            }
            nodes[cur].terminal = true;
        }
        std::ostringstream out;
        out << "[\"] ( ";
        std::function<void(int)> visit_node = [&](int idx) {
            const auto& node = nodes[idx];
            std::string rejects;
            bool first = true;
            for (const auto& kv : node.children) {
                rejects += kv.first;
                if (!first) out << " | ";
                first = false;
                out << "[" << kv.first << "]";
                if (!nodes[kv.second].children.empty()) {
                    out << " (";
                    visit_node(kv.second);
                    out << ")";
                } else {
                    out << " " << char_rule << "+";
                }
            }
            if (!node.children.empty()) {
                out << " | [^\"" << rejects << "] " << char_rule << "*";
            }
        };
        visit_node(0);
        out << " )";
        if (!nodes[0].children.empty()) out << "?";
        else out << "?"; // root terminal (no strings) — match any string
        out << " [\"]";
        return out.str();
    }

    // JSON-pointer $ref resolution against the root schema document
    const Json* resolve_ref(const std::string& ref, std::string& out_name) {
        static const std::regex nonalphanumeric(R"([^a-zA-Z0-9-]+)");
        const size_t hash = ref.find('#');
        std::string fragment = hash != std::string::npos ? ref.substr(hash + 1) : ref;
        out_name = "ref" + std::regex_replace(fragment, nonalphanumeric, "-");
        if (out_name == "ref" || out_name.empty()) out_name = "ref-";
        const Json* target = &m_root;
        if (fragment.empty()) return target;
        if (!fragment.empty() && fragment[0] == '/') fragment = fragment.substr(1);
        size_t pos = 0;
        while (pos < fragment.size() && target) {
            size_t slash = fragment.find('/', pos);
            std::string tok = fragment.substr(pos, slash == std::string::npos
                                                ? std::string::npos : slash - pos);
            // unescape ~1 -> '/', ~0 -> '~'
            std::string unesc;
            for (size_t k = 0; k < tok.size(); k++) {
                if (tok[k] == '~' && k + 1 < tok.size() && tok[k+1] == '1') { unesc += '/'; k++; }
                else if (tok[k] == '~' && k + 1 < tok.size() && tok[k+1] == '0') { unesc += '~'; k++; }
                else unesc += tok[k];
            }
            if (target->is_arr()) {
                try {
                    const size_t idx = (size_t) std::stoul(unesc);
                    target = idx < target->arr.size() ? &target->arr[idx] : nullptr;
                } catch (...) {
                    target = nullptr;
                }
            } else if (target->is_obj()) {
                target = target->find(unesc);
            } else {
                target = nullptr;
            }
            if (slash == std::string::npos) break;
            pos = slash + 1;
        }
        return target;
    }

    std::string build_object_rule(
            const std::vector<std::pair<std::string, const Json*>>& properties,
            const std::unordered_set<std::string>& required,
            const std::string& name,
            const Json* additional_properties) {
        std::vector<std::string> required_props;
        std::vector<std::string> optional_props;
        std::unordered_map<std::string, std::string> prop_kv_rule_names;
        std::vector<std::string> prop_names;
        for (const auto& kv : properties) {
            const auto& prop_name = kv.first;
            const auto* prop_schema = kv.second;
            const std::string prop_rule_name =
                visit(*prop_schema, name + (name.empty() ? "" : "-") + prop_name,
                      name + (name.empty() ? "" : "-") + prop_name);
            prop_kv_rule_names[prop_name] = add_rule(
                name + (name.empty() ? "" : "-") + prop_name + "-kv",
                format_literal(Json::dump(Json::str(prop_name))) + " space \":\" space " + prop_rule_name);
            if (required.count(prop_name)) required_props.push_back(prop_name);
            else optional_props.push_back(prop_name);
            prop_names.push_back(prop_name);
        }
        if (additional_properties) {
            const std::string sub_name = name + (name.empty() ? "" : "-") + "additional";
            const bool is_any = additional_properties->is_bool() && additional_properties->b;
            const std::string value_rule = !is_any
                ? visit(*additional_properties, sub_name + "-value", sub_name + "-value")
                : add_primitive("value", PRIMITIVE_RULES.at("value"));
            const std::string key_rule = prop_names.empty()
                ? add_primitive("string", PRIMITIVE_RULES.at("string"))
                : add_rule(sub_name + "-k", not_strings(prop_names));
            const std::string kv_rule = add_rule(sub_name + "-kv", key_rule + " \":\" space " + value_rule);
            prop_kv_rule_names["*"] = kv_rule;
            optional_props.push_back("*");
        }

        if (required_props.empty() && optional_props.empty()) {
            return "\"{\" space \"}\"";
        }

        std::string rule = "\"{\" space ";
        for (size_t i = 0; i < required_props.size(); i++) {
            if (i > 0) rule += " \",\" space ";
            rule += prop_kv_rule_names[required_props[i]];
        }

        if (!optional_props.empty()) {
            rule += " (";
            if (!required_props.empty()) rule += " \",\" space ( ";

            std::function<std::string(const std::vector<std::string>&, bool)> get_recursive_refs =
                [&](const std::vector<std::string>& ks, bool first_is_optional) {
                    std::string res;
                    if (ks.empty()) return res;
                    const std::string& k = ks[0];
                    const std::string kv_rule_name = prop_kv_rule_names[k];
                    const std::string comma_ref = "( \",\" space " + kv_rule_name + " )";
                    if (first_is_optional) {
                        res = comma_ref + (k == "*" ? "*" : "?");
                    } else {
                        res = kv_rule_name + (k == "*" ? " " + comma_ref + "*" : "");
                    }
                    if (ks.size() > 1) {
                        res += " " + add_rule(
                            name + (name.empty() ? "" : "-") + k + "-rest",
                            get_recursive_refs(std::vector<std::string>(ks.begin() + 1, ks.end()), true));
                    }
                    return res;
                };

            for (size_t i = 0; i < optional_props.size(); i++) {
                if (i > 0) rule += " | ";
                rule += get_recursive_refs(
                    std::vector<std::string>(optional_props.begin() + i, optional_props.end()), false);
            }
            if (!required_props.empty()) rule += " )";
            rule += " )?";
        }

        rule += " space \"}\"";
        return rule;
    }

    static std::string constant_rule(const Json& value) {
        return format_literal(Json::dump(value));
    }

    // The keyword gate: every key in the schema must be understood, or the
    // conversion fails closed naming the keyword (plan 11(f)).
    static void check_keywords(const Json& schema) {
        // Ignored metadata (no output-constraining semantics):
        static const std::unordered_set<std::string> IGNORED = {
            "$schema", "$id", "title", "description", "default", "examples",
            "deprecated", "readOnly", "writeOnly", "definitions", "$comment",
            "$defs", "$anchor"
        };
        if (!schema.is_obj()) {
            unsupported("<schema>",
                        "a JSON Schema must be an object at the position it appears");
        }
        for (const auto& kv : schema.obj) {
            const std::string& k = kv.first;
            if (IGNORED.count(k)) continue;
            static const std::unordered_set<std::string> SUPPORTED = {
                "type", "properties", "required", "additionalProperties",
                "items", "prefixItems", "minItems", "maxItems", "enum", "const",
                "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
                "minLength", "maxLength", "format", "$ref", "anyOf", "oneOf",
                "allOf"
            };
            if (!SUPPORTED.count(k)) {
                unsupported(k, "the JSON-schema keyword '" + k +
                               "' is not supported by this converter slice — the "
                               "grammar fails closed, never best-effort");
            }
        }
    }

    bool extract_type(const Json& schema, const char* want) {
        const Json* t = schema.find("type");
        if (!t) return false;
        if (t->is_str()) return t->s == want;
        if (t->is_arr()) {
            for (const auto& v : t->arr) if (v.is_str() && v.s == want) return true;
        }
        return false;
    }

    std::string visit_union(const std::vector<const Json*>& children, const std::string& rule_name) {
        std::vector<std::string> parts;
        for (size_t i = 0; i < children.size(); i++) {
            parts.push_back(visit(*children[i], rule_name + "-" + std::to_string(i),
                                  rule_name + "-" + std::to_string(i)));
        }
        return add_rule(rule_name, "(" + [&] {
            std::string joined;
            for (size_t i = 0; i < parts.size(); i++) {
                if (i) joined += " | ";
                joined += parts[i];
            }
            return joined;
        }() + ")");
    }

    // Deep copy (synthesized type-union members keep every constraint of
    // the shared schema; only the narrowed `type` differs).
    static Json clone(const Json& v) {
        Json out = v;
        out.obj.clear();
        out.arr.clear();
        for (const auto& kv : v.obj) out.obj.emplace_back(kv.first, clone(kv.second));
        for (const auto& e : v.arr) out.arr.push_back(clone(e));
        return out;
    }

    std::string visit(const Json& schema, const std::string& name,
                      const std::string& rule_name_in) {
        const std::string rule_name = (is_reserved_name(rule_name_in) && rule_name_in != "root")
            ? rule_name_in + "-" : rule_name_in;
        const std::string sub_name = name + (name.empty() ? "" : "-");

        check_keywords(schema);

        // $ref: resolve against the root document; rule-name reuse makes
        // recursive schemas work (a rule may reference itself).
        if (const Json* ref = schema.find("$ref")) {
            if (!ref->is_str()) {
                unsupported("$ref", "$ref must be a string");
            }
            std::string ref_name;
            const Json* target = resolve_ref(ref->s, ref_name);
            if (!target) {
                m_errors.push_back("Unresolved $ref " + ref->s);
                return add_rule(rule_name, "\"\"");
            }
            if (m_rules.find(ref_name) == m_rules.end()) {
                // Cycle guard (upstream _refs_being_resolved): a schema that
                // references itself while its rule is being emitted resolves
                // to the rule NAME — self-referencing grammars are the
                // supported recursive shape, infinite expansion is not.
                if (m_refs_in_progress.count(ref->s)) {
                    m_rules[ref_name] = ref_name; // self-reference placeholder
                } else {
                    m_refs_in_progress.insert(ref->s);
                    visit(*target, ref_name, ref_name);
                    m_refs_in_progress.erase(ref->s);
                }
            }
            return add_rule(rule_name, ref_name);
        }

        if (const Json* any_of = schema.find("anyOf")) {
            if (!any_of->is_arr() || any_of->arr.empty())
                unsupported("anyOf", "anyOf must be a non-empty array");
            std::vector<const Json*> kids;
            for (const auto& v : any_of->arr) kids.push_back(&v);
            return visit_union(kids, rule_name);
        }
        if (const Json* one_of = schema.find("oneOf")) {
            if (!one_of->is_arr() || one_of->arr.empty())
                unsupported("oneOf", "oneOf must be a non-empty array");
            std::vector<const Json*> kids;
            for (const auto& v : one_of->arr) kids.push_back(&v);
            return visit_union(kids, rule_name);
        }
        if (const Json* all_of = schema.find("allOf")) {
            if (!all_of->is_arr() || all_of->arr.empty())
                unsupported("allOf", "allOf must be a non-empty array");
            // Upstream add_component subset: objects merge (properties/required),
            // enums intersect, refs recurse; anything else fails closed.
            std::unordered_set<std::string> required;
            std::vector<std::pair<std::string, const Json*>> properties;
            std::map<std::string, int> enum_values;
            std::vector<std::string> enum_constants;
            std::function<void(const Json&, bool)> add_component = [&](const Json& comp, bool is_req) {
                if (const Json* ref = comp.find("$ref")) {
                    std::string ref_name;
                    const Json* target = resolve_ref(ref->is_str() ? ref->s : "", ref_name);
                    if (target) add_component(*target, is_req);
                    else m_errors.push_back("Unresolved $ref in allOf");
                    return;
                }
                if (const Json* props = comp.find("properties")) {
                    if (!props->is_obj())
                        unsupported("allOf", "allOf member 'properties' must be an object");
                    const Json* req = comp.find("required");
                    for (const auto& pkv : props->obj) {
                        properties.emplace_back(pkv.first, &pkv.second);
                        if (is_req && req && req->is_arr()) {
                            for (const auto& r : req->arr)
                                if (r.is_str() && r.s == pkv.first) required.insert(pkv.first);
                        }
                    }
                    return;
                }
                if (const Json* en = comp.find("enum")) {
                    if (!en->is_arr())
                        unsupported("allOf", "allOf member 'enum' must be an array");
                    for (const auto& v : en->arr) {
                        const std::string c = constant_rule(v);
                        if (std::find(enum_constants.begin(), enum_constants.end(), c) == enum_constants.end())
                            enum_constants.push_back(c);
                        enum_values[c] += 1;
                    }
                    return;
                }
                unsupported("allOf",
                            "allOf members must be objects with 'properties' or 'enum' "
                            "schemas (the mergeable subset); arbitrary allOf shapes fail closed");
            };
            for (const auto& child : all_of->arr) {
                add_component(child, true);
            }
            if (!enum_values.empty()) {
                std::vector<std::string> intersection;
                for (const auto& p : enum_values)
                    if (p.second == (int) all_of->arr.size()) intersection.push_back(p.first);
                if (!intersection.empty()) {
                    std::string joined;
                    for (size_t i = 0; i < intersection.size(); i++) {
                        if (i) joined += " | ";
                        joined += intersection[i];
                    }
                    return add_rule(rule_name, "(" + joined + ")");
                }
            }
            if (properties.empty())
                unsupported("allOf", "allOf produced no mergeable properties/enums");
            return add_rule(rule_name, build_object_rule(properties, required, name, nullptr));
        }

        if (const Json* en = schema.find("enum")) {
            if (!en->is_arr() || en->arr.empty())
                unsupported("enum", "enum must be a non-empty array");
            std::string joined;
            for (size_t i = 0; i < en->arr.size(); i++) {
                if (i) joined += " | ";
                joined += constant_rule(en->arr[i]);
            }
            return add_rule(rule_name, "(" + joined + ")");
        }
        if (const Json* cst = schema.find("const")) {
            return add_rule(rule_name, constant_rule(*cst));
        }

        const Json* type = schema.find("type");
        std::string type_s;
        if (type) {
            if (type->is_str()) type_s = type->s;
            else if (type->is_arr()) {
                // type union — upstream union semantics: each member keeps
                // every constraint of the shared schema with the narrowed type
                // (e.g. ["array","null"] + items: the array member carries items).
                std::vector<const Json*> kids;
                for (const auto& v : type->arr) {
                    if (!v.is_str())
                        unsupported("type", "type array entries must be strings");
                    Json prim = clone(schema);
                    bool replaced = false;
                    for (auto& kv : prim.obj) {
                        if (kv.first == "type") { kv.second = v; replaced = true; break; }
                    }
                    if (!replaced) prim.obj.emplace_back("type", v);
                    m_temp.push_back(std::move(prim));
                    kids.push_back(&m_temp.back());
                }
                if (kids.empty()) unsupported("type", "type array must be non-empty");
                return visit_union(kids, rule_name);
            }
        }

        // Structural keywords imply object/array like upstream (json-schema.cpp
        // infers kinds from the keywords present).
        const Json* props_kw = schema.find("properties");
        const Json* items_kw = schema.find("items");
        const Json* pre_kw  = schema.find("prefixItems");
        if (type_s == "object" || (type_s.empty() &&
            (props_kw || schema.find("required") || schema.find("additionalProperties")))) {
            std::vector<std::pair<std::string, const Json*>> properties;
            std::unordered_set<std::string> required;
            if (const Json* props = schema.find("properties")) {
                if (!props->is_obj())
                    unsupported("properties", "properties must be an object");
                if (const Json* req = schema.find("required")) {
                    if (!req->is_arr())
                        unsupported("required", "required must be an array");
                    for (const auto& r : req->arr) {
                        if (!r.is_str())
                            unsupported("required", "required entries must be strings");
                        required.insert(r.s);
                    }
                }
                for (const auto& pkv : props->obj) properties.emplace_back(pkv.first, &pkv.second);
            } else if (const Json* req = schema.find("required")) {
                unsupported("properties", "required without properties is not representable");
            }
            const Json* additional = nullptr;
            if (const Json* ap = schema.find("additionalProperties")) {
                if (ap->is_bool()) {
                    if (!ap->b) additional = nullptr; // false = no extra props
                    else {
                        m_temp.push_back(Json{});
                        m_temp.back().kind = Json::OBJ; // any
                        additional = &m_temp.back();
                    }
                } else if (ap->is_obj()) {
                    additional = ap;
                } else {
                    unsupported("additionalProperties",
                                "additionalProperties must be a boolean or schema object");
                }
            }
            if (properties.empty() && (additional == nullptr || additional->obj.empty())) {
                // free-form object (any shape, or any values for unknown keys)
                return add_rule(rule_name, add_primitive("object", PRIMITIVE_RULES.at("object")));
            }
            return add_rule(rule_name, build_object_rule(properties, required, name, additional));
        }

        if (type_s == "array" || (type_s.empty() &&
            (items_kw || pre_kw || schema.find("minItems") || schema.find("maxItems")))) {
            int min_items = 0;
            int max_items = std::numeric_limits<int>::max();
            if (const Json* mi = schema.find("minItems")) {
                if (!mi->is_int() || mi->i < 0)
                    unsupported("minItems", "minItems must be a non-negative integer");
                min_items = (int) mi->i;
            }
            if (const Json* ma = schema.find("maxItems")) {
                if (!ma->is_int() || ma->i < 0)
                    unsupported("maxItems", "maxItems must be a non-negative integer");
                max_items = (int) ma->i;
            }
            if (const Json* pre = schema.find("prefixItems")) {
                if (!pre->is_arr())
                    unsupported("prefixItems", "prefixItems must be an array");
                // tuple: prefixItems + (optional) items for the tail
                std::string rule = "\"[\" space ";
                for (size_t i = 0; i < pre->arr.size(); i++) {
                    if (i > 0) rule += " \",\" space ";
                    rule += visit(pre->arr[i], sub_name + "tuple-" + std::to_string(i),
                                  sub_name + "tuple-" + std::to_string(i));
                }
                const Json* items = schema.find("items");
                if (items && (min_items > (int) pre->arr.size() || max_items < (int) pre->arr.size())) {
                    // combined bounds with prefixItems tail — keep the honest
                    // bound on the tail repetition
                    const int tail_min = std::max(0, min_items - (int) pre->arr.size());
                    const int tail_max = max_items == std::numeric_limits<int>::max()
                        ? max_items : std::max(0, max_items - (int) pre->arr.size());
                    const std::string item_rule = visit(*items, sub_name + "item", sub_name + "item");
                    if (pre->arr.size() > 0 || tail_min > 0)
                        rule += " \",\" space ";
                    rule += build_repetition(item_rule, tail_min, tail_max, "\",\" space");
                }
                rule += " space \"]\"";
                return add_rule(rule_name, rule);
            }
            const Json* items = schema.find("items");
            if (!items || (items->is_obj() && items->obj.empty())) {
                if (min_items == 0 && max_items == std::numeric_limits<int>::max()) {
                    return add_rule(rule_name, add_primitive("array", PRIMITIVE_RULES.at("array")));
                }
                m_temp.push_back(Json{}); // any
                m_temp.back().kind = Json::OBJ;
                items = &m_temp.back();
            } else if (items->is_bool() && items->b) {
                m_temp.push_back(Json{});
                m_temp.back().kind = Json::OBJ;
                items = &m_temp.back();
            } else if (items->is_bool() && !items->b) {
                if (min_items > 0)
                    unsupported("items", "items=false with minItems>0 is unsatisfiable");
                return add_rule(rule_name, "\"[\" space \"]\"");
            }
            const std::string item_rule = visit(*items, sub_name + "item", sub_name + "item");
            return add_rule(rule_name, "\"[\" space " +
                           build_repetition(item_rule, min_items, max_items, "\",\" space") +
                           " space \"]\"");
        }

        if (type_s == "string" || (type_s.empty() &&
            (schema.find("minLength") || schema.find("maxLength") || schema.find("format")))) {
            if (const Json* pat = schema.find("pattern")) {
                (void) pat;
                unsupported("pattern",
                            "the regex->GBNF compiler is not ported in this slice — "
                            "schemas using 'pattern' fail closed");
            }
            if (const Json* fmt = schema.find("format")) {
                if (!fmt->is_str())
                    unsupported("format", "format must be a string");
                const std::string& f = fmt->s;
                if (f == "uuid") {
                    return add_rule(rule_name, add_primitive("uuid", PRIMITIVE_RULES.at("uuid")));
                }
                if (f == "date" || f == "time" || f == "date-time") {
                    const std::string prim = f + "-string";
                    const auto it = STRING_FORMAT_RULES.find(prim);
                    if (it == STRING_FORMAT_RULES.end())
                        unsupported("format", "unknown string format '" + f + "'");
                    return add_rule(rule_name, add_primitive(prim, it->second));
                }
                unsupported("format", "string format '" + f + "' is not supported");
            }
            int min_len = 0;
            int max_len = -1;
            if (const Json* ml = schema.find("minLength")) {
                if (!ml->is_int() || ml->i < 0)
                    unsupported("minLength", "minLength must be a non-negative integer");
                min_len = (int) ml->i;
            }
            if (const Json* ml = schema.find("maxLength")) {
                if (!ml->is_int() || ml->i < 0)
                    unsupported("maxLength", "maxLength must be a non-negative integer");
                max_len = (int) ml->i;
            }
            if (min_len > 0 || max_len >= 0) {
                const std::string char_rule = add_primitive("char", PRIMITIVE_RULES.at("char"));
                const int max_c = max_len < 0 ? std::numeric_limits<int>::max() : max_len;
                return add_rule(rule_name, "\"\\\"\" " +
                                build_repetition(char_rule, min_len, max_c) + " \"\\\"\"");
            }
            return add_rule(rule_name, add_primitive(
                rule_name == "root" ? "root" : "string", PRIMITIVE_RULES.at("string")));
        }

        if (type_s == "integer" || schema.find("minimum") || schema.find("maximum") ||
            schema.find("exclusiveMinimum") || schema.find("exclusiveMaximum")) {
            int64_t min_v = std::numeric_limits<int64_t>::min();
            int64_t max_v = std::numeric_limits<int64_t>::max();
            auto int_or_fail = [&](const Json* v, const char* kw) -> int64_t {
                if (!v || !v->is_int())
                    unsupported(kw, std::string(kw) + " bounds must be integers (this slice)");
                return v->i;
            };
            if (const Json* v = schema.find("minimum")) {
                const Json* excl = schema.find("exclusiveMinimum");
                if (excl && excl->is_bool()) {
                    min_v = excl->b ? int_or_fail(v, "exclusiveMinimum") + 1 : int_or_fail(v, "minimum");
                } else {
                    min_v = int_or_fail(v, "minimum");
                }
            } else if (const Json* v2 = schema.find("exclusiveMinimum")) {
                min_v = int_or_fail(v2, "exclusiveMinimum") + 1;
            }
            if (const Json* v = schema.find("maximum")) {
                const Json* excl = schema.find("exclusiveMaximum");
                if (excl && excl->is_bool()) {
                    max_v = excl->b ? int_or_fail(v, "exclusiveMaximum") - 1 : int_or_fail(v, "maximum");
                } else {
                    max_v = int_or_fail(v, "maximum");
                }
            } else if (const Json* v2 = schema.find("exclusiveMaximum")) {
                max_v = int_or_fail(v2, "exclusiveMaximum") - 1;
            }
            if (min_v == std::numeric_limits<int64_t>::min() &&
                max_v == std::numeric_limits<int64_t>::max()) {
                return add_rule(rule_name, add_primitive(
                    rule_name == "root" ? "root" : "integer", PRIMITIVE_RULES.at("integer")));
            }
            std::ostringstream out;
            out << "(";
            build_min_max_int(min_v, max_v, out);
            out << ")";
            return add_rule(rule_name, out.str());
        }

        if (type_s == "number") {
            return add_rule(rule_name, add_primitive(
                rule_name == "root" ? "root" : "number", PRIMITIVE_RULES.at("number")));
        }
        if (type_s == "boolean") {
            return add_rule(rule_name, add_primitive(
                rule_name == "root" ? "root" : "boolean", PRIMITIVE_RULES.at("boolean")));
        }
        if (type_s == "null") {
            return add_rule(rule_name, add_primitive(
                rule_name == "root" ? "root" : "null", PRIMITIVE_RULES.at("null")));
        }
        if (type_s.empty()) {
            // No type, no structural keywords: any JSON value.
            return add_rule(rule_name, add_primitive(
                rule_name == "root" ? "root" : "value", PRIMITIVE_RULES.at("value")));
        }
        unsupported("type", "unknown schema type '" + type_s + "'");
    }

    std::string format_grammar() {
        std::ostringstream ss;
        for (const auto& kv : m_rules) ss << kv.first << " ::= " << kv.second << '\n';
        return ss.str();
    }

    std::deque<Json> m_temp; // scratch storage for synthesized schemas (STABLE pointers)
};

} // namespace schema
} // namespace pvx

// ── C ABI (typed failure surfaces) ────────────────────────────────────────

extern "C" pvx_result_t pvx_grammar_from_json_schema(const char* json_schema,
                                                     const char* root_name,
                                                     char** out_gbnf,
                                                     char** out_error) {
    using pvx::schema::Converter;
    using pvx::schema::Json;
    using pvx::schema::KeywordError;
    using pvx::schema::ParseError;

    if (!out_gbnf) return PVX_ERR_INVALID_ARG;
    *out_gbnf = nullptr;
    if (out_error) *out_error = nullptr;
    if (!json_schema) return PVX_ERR_INVALID_ARG;

    const std::string rn = (root_name && root_name[0]) ? root_name : "root";

    try {
        pvx::schema::Parser parser(json_schema);
        const Json doc = parser.parse();
        Converter conv(doc);
        const std::string gbnf = conv.convert(rn);
        *out_gbnf = (char*) malloc(gbnf.size() + 1);
        if (!*out_gbnf) return PVX_ERR_INTERNAL;
        memcpy(*out_gbnf, gbnf.c_str(), gbnf.size() + 1);
        return PVX_OK;
    } catch (const ParseError& e) {
        if (out_error) {
            const std::string msg = "invalid JSON schema at position " +
                                    std::to_string(e.pos) + ": " + e.msg +
                                    " (boundary: json-schema/parse)";
            *out_error = (char*) malloc(msg.size() + 1);
            if (*out_error) memcpy(*out_error, msg.c_str(), msg.size() + 1);
        }
        return PVX_ERR_INVALID_ARG;
    } catch (const KeywordError& e) {
        if (out_error) {
            const std::string msg = "unsupported JSON-schema keyword '" + e.keyword +
                                    "': " + e.msg + " (boundary: json-schema/keyword)";
            *out_error = (char*) malloc(msg.size() + 1);
            if (*out_error) memcpy(*out_error, msg.c_str(), msg.size() + 1);
        }
        return PVX_ERR_NOT_SUPPORTED;
    } catch (const std::exception& e) {
        if (out_error) {
            const std::string msg = std::string("schema conversion failed: ") + e.what() +
                                    " (boundary: json-schema/keyword)";
            *out_error = (char*) malloc(msg.size() + 1);
            if (*out_error) memcpy(*out_error, msg.c_str(), msg.size() + 1);
        }
        return PVX_ERR_NOT_SUPPORTED;
    }
}
