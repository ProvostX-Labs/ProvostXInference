namespace ProvostXInference.Models;

/// <summary>User-facing speculative-decoding configuration (sequential mode).
/// Sentinels mirror the native defaults (upstream draft-simple): DraftMax <= 0
/// and DraftMin <= 0 select the upstream defaults; PMin < 0 disables the
/// drafter confidence floor; DraftGpuLayers -1 mirrors the target's offload.</summary>
public sealed record SpeculativeConfig
{
    public SpeculativeMode Mode { get; init; } = SpeculativeMode.Off;
    /// <summary>Required iff Mode == Draft (fail-closed: missing path is a typed
    /// error, never a silent disable).</summary>
    public string? DraftModelPath { get; init; }
    public int DraftMax { get; init; } = 0;
    public int DraftMin { get; init; } = 0;
    public float PMin { get; init; } = 0.0f;
    public int DraftGpuLayers { get; init; } = -1;
}
