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
    /// <summary>ABI v6: n-gram knobs — active iff Mode is an n-gram mode
    /// (Simple/MapK/Mod/Cache). Null = all native defaults.</summary>
    public NgramConfig? Ngram { get; init; }
    /// <summary>ABI v7: EAGLE3 companion draft GGUF — required iff Mode ==
    /// Eagle3 (missing path is a typed native refusal naming the boundary,
    /// never a silent disable). The companion must carry eagle3.target_layers
    /// metadata (exactly 3 entries) plus a compatible vocab family.</summary>
    public string? EagleCompanionPath { get; init; }
    /// <summary>ABI v7: DFlash companion draft GGUF — required iff Mode ==
    /// Dflash. Metadata is validated at spec init; the unwired drafting
    /// boundary types fail-closed.</summary>
    public string? DflashCompanionPath { get; init; }
}
