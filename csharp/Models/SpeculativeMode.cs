namespace ProvostXInference.Models;

/// <summary>Speculative-decoding mode, values mirror pvx_spec_mode_t exactly.
/// Fail-closed rule: a configured path that cannot be honored (missing/unloadable
/// draft model, missing capability, wrong vocab family) is a typed error naming
/// the boundary — never a silent fallback to non-spec generation.</summary>
public enum SpeculativeMode
{
    Off = 0,
    /// <summary>Model-native MTP drafting — requires an mtp capability; enabling
    /// currently fails typed NOT_SUPPORTED while the MTP wiring boundary stands.</summary>
    Mtp = 1,
    /// <summary>Second draft model (upstream draft-simple) — requires a
    /// draft_simple capability plus a user-provided DraftModelPath.</summary>
    Draft = 2,
    /// <summary>ABI v6 — n-gram drafting from the committed stream: no draft
    /// model. History-pattern lookup (upstream ngram-simple).</summary>
    NgramSimple = 3,
    /// <summary>ABI v6 — n-gram -> m-gram map with per-value accept statistics
    /// (upstream ngram-map-k).</summary>
    NgramMapK = 4,
    /// <summary>ABI v6 — MOD: hashed (n-1)-suffix match, one deviation
    /// (upstream ngram-mod).</summary>
    NgramMod = 5,
    /// <summary>ABI v6 — static (file) + dynamic token-probability caches
    /// (upstream ngram-cache).</summary>
    NgramCache = 6,
}
