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

    // ── ABI v8 wave-5 (BL-21b): model-load controls — null/zero = llama defaults ──
    /// <summary>GGUF metadata overrides — the GENERAL escape hatch: any
    /// model hyperparameter key (e.g. "<arch>.rope.freq_base") overridden
    /// at load. null = none.</summary>
    public IReadOnlyList<KvOverride>? KvOverrides { get; init; }
    /// <summary>Lazy tensor load (upstream llama_lazy_mode). 0/null = llama
    /// default (AUTO); else 1 none / 2 auto / 3 mmap / 4 mlock.</summary>
    public int LazyMode { get; init; } = 0;
    /// <summary>Extra buffer types (weight repacking). 0 = llama default
    /// (true); 1 force on; 2 force off.</summary>
    public int UseExtraBuftsTri { get; init; } = 0;
    /// <summary>Vocab-only load (no weights). null = false (llama default).</summary>
    public bool VocabOnly { get; init; } = false;
    /// <summary>Validate tensor data at load. default false.</summary>
    public bool CheckTensors { get; init; } = false;
    /// <summary>Bypass the host buffer. default false.</summary>
    public bool NoHost { get; init; } = false;
    /// <summary>Metadata-only load simulation. default false.</summary>
    public bool NoAlloc { get; init; } = false;
    /// <summary>Load the model's OWN MTP layers (upstream load_mtp — llama
    /// default false: MTP tensors stay unloaded unless asked).</summary>
    public bool LoadMtp { get; init; } = false;
}
