namespace ProvostXInference.Models;

/// <summary>Tag for a GGUF metadata override value (BL-21b wave-5).</summary>
public enum KvOverrideTag
{
    Bool = 0,
    Int64 = 1,
    Float64 = 2,
    Str = 3,
}

/// <summary>One GGUF metadata override — the GENERAL model-hyperparameter
/// escape hatch (upstream llama_model_kv_override parity, BL-21b):
/// any key (e.g. "<arch>.rope.freq_base") overridden at model load.
/// The native union is split into explicit properties (only the Tag's
/// value crosses the ABI).</summary>
public sealed record KvOverride
{
    public required string Key { get; init; }
    public KvOverrideTag Tag { get; init; }
    public bool Bool { get; init; }
    public long Int64 { get; init; }
    public double Float64 { get; init; }
    public string? Str { get; init; }
}