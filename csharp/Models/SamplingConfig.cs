namespace ProvostXInference.Models;

/// <summary>User-facing sampling configuration.</summary>
public sealed record SamplingConfig
{
    public float Temperature { get; init; } = 0.3f;
    public float TopP { get; init; } = 0.95f;
    public int TopK { get; init; } = 40;
    public float MinP { get; init; } = 0.0f;
    public float RepeatPenalty { get; init; } = 1.1f;
    public int RepeatLastN { get; init; } = -1;
    public float PenaltyPresent { get; init; } = 0.0f;
    public int MaxTokens { get; init; } = 256;
    public bool IgnoreEos { get; init; } = false;

    // ── ABI v8 wave-5 (BL-21a) sampler knobs — zeroed = pre-wave-5 chain ──
    /// <summary>Deterministic sampling seed. null/&lt;0 = time entropy.</summary>
    public long? Seed { get; init; }
    /// <summary>Frequency penalty (logit sampling bias). null/0 = off.</summary>
    public float? FrequencyPenalty { get; init; }
    /// <summary>Typical sampling p in (0,1). null/0 = off.</summary>
    public float? TypicalP { get; init; }
    /// <summary>XTC sampler probability. null/0 = off.</summary>
    public float? XtcP { get; init; }
    /// <summary>XTC logit threshold. null/&lt;=0 → 1.0f when XtcP &gt; 0.</summary>
    public float? XtcT { get; init; }
    /// <summary>Top-n-σ sampling. null/0 = off.</summary>
    public float? TopNSigma { get; init; }
    /// <summary>Mirostat: 0/null = off; 1 = v1; 2 = v2 (replaces the dist sampler).</summary>
    public int? MirostatMode { get; init; }
    /// <summary>Mirostat target surprise (tau). null/0 → 5.0f when mode &gt; 0.</summary>
    public float? MirostatTau { get; init; }
    /// <summary>Mirostat learning rate (eta). null/0 → 0.1f when mode &gt; 0.</summary>
    public float? MirostatEta { get; init; }
    /// <summary>DRY repetition multiplier. null/0 = off.</summary>
    public float? DryMultiplier { get; init; }
    /// <summary>DRY base. null/0 → 1.75f when multiplier &gt; 0.</summary>
    public float? DryBase { get; init; }
    /// <summary>DRY allowed raw length. null/0 → 2 when multiplier &gt; 0.</summary>
    public int? DryAllowedLength { get; init; }
    /// <summary>DRY window (last n). null/0 → -1 (full context) when multiplier &gt; 0.</summary>
    public int? DryPenaltyLastN { get; init; }
    /// <summary>Token → logit-bias pairs (upstream llama_logit_bias). null/empty = off.</summary>
    public IReadOnlyList<(int Token, float Bias)>? LogitBias { get; init; }
}
