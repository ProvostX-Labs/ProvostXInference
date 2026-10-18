namespace ProvostXInference.Models;

/// <summary>n-gram speculative-dcoding knobs (ABI v6). Active iff the
/// speculative mode is an n-gram mode; zero values select the per-mode native
/// defaults. Sentinel discipline mirrors the rest of SpeculativeConfig:
/// the config is blitted to pvx_ngram_params_t at the native boundary.</summary>
public sealed record NgramConfig
{
    /// <summary>Lookup min n-gram size; 0 = per-mode default. Must satisfy
    /// 0 &lt;= NMin &lt;= NMax (fail-closed typed error otherwise).</summary>
    public int NMin { get; init; }
    /// <summary>Lookup/draft max size (m-gram length); 0 = per-mode default.</summary>
    public int NMax { get; init; }
    /// <summary>Map-k mode only: minimum key hits before drafting; 0 = default.</summary>
    public int MinHits { get; init; }
    /// <summary>Mod mode only: hashed (n-1)-suffix window; 0 = default.</summary>
    public int NMatch { get; init; }
    /// <summary>Cache mode only: optional static cache file. A corrupt or
    /// unreadable file is a typed LOAD_FAILED error naming the path — never a
    /// silently empty cache.</summary>
    public string? CachePath { get; init; }
}
