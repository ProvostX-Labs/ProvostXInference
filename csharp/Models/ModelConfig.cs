namespace ProvostXInference.Models;

/// <summary>Managed enum mirroring PvxKvType.</summary>
public enum KvType { F16 = 0, Q8_0 = 1, Q4_0 = 2, Q4_1 = 3 }

/// <summary>User-facing model configuration (not blittable — converted to PvxModelParams internally).</summary>
public sealed record ModelConfig
{
    public required string Path { get; init; }

    /// <summary>
    /// Engine selection (per-model config knob, BL-02): WHICH specialist dylib
    /// serves this model — one of the 5-product matrix ids
    /// ("metal" = macOS only; "cuda"/"vulkan" = Windows/Linux). Fail-closed:
    /// unknown engine ids and wrong-OS selections throw at load time; different
    /// models in one process may select different specialists (per-handle
    /// dispatch routes each model's calls onto its own image). Null = platform
    /// default (macOS: metal; elsewhere: cuda when installed, else vulkan).
    /// </summary>
    public string? Engine { get; init; }

    public int GpuLayers { get; init; } = -1;      // -1 = all
    public int Threads { get; init; } = 0;          // 0 = auto
    public bool FlashAttention { get; init; } = true;
    public KvType KvCacheType { get; init; } = KvType.F16;

    /// <summary>
    /// Reasoning-marker probe list (data-driven, 2026-09-26): tag strings probed
    /// against the vocabulary at template time; the first single-special-token
    /// match drives the empty-block prefill when thinking is off. Default
    /// ({"think"}) covers Qwen3-family + GLM-4.5-family vocabularies; extend
    /// via config (e.g. "reason" for DeepSeek) — no code change needed.
    /// </summary>
    public string[]? ReasoningMarkers { get; init; }
}