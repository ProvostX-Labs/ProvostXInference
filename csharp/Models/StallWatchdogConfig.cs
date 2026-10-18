namespace ProvostXInference.Models;

/// <summary>Configuration for the engine-level stall-safe prefill watchdog
/// (stall_watchdog config section; INF BL-33). Defaults follow the owner
/// decision: multiplier 5 over the rolling median chunk wall, 3 retries, the
/// full four-step recovery ladder.</summary>
public sealed record StallWatchdogConfig
{
    /// <summary>Config section name used by the owning server config
    /// (kebab snake-case as specified by the owner).</summary>
    public const string SectionName = "stall_watchdog";

    /// <summary>Watchdog master switch. Disabled = single-shot prefill with
    /// zero watchdog overhead (behavior identical to the plain layer below).</summary>
    public bool Enabled { get; init; } = true;

    /// <summary>A chunk stalls when its wall exceeds GapMultiplier × the
    /// rolling median of recent chunk walls. Default 5.</summary>
    public double GapMultiplier { get; init; } = 5.0;

    /// <summary>Maximum recovery attempts before the ladder fails loud. Default 3.</summary>
    public int MaxRetries { get; init; } = 3;

    /// <summary>Chunk size (tokens) for committed chunked prefill. Default 2048
    /// (mirrors the upstream ub-chunking the stall evidence was gathered on).</summary>
    public int ChunkSizeTokens { get; init; } = 2048;

    /// <summary>Floor for the SmallerChunkRetry halving step. Default 128.</summary>
    public int MinChunkSizeTokens { get; init; } = 128;

    /// <summary>Rolling baseline window (recent chunk walls feeding the median).
    /// Default 8.</summary>
    public int BaselineWindow { get; init; } = 8;

    /// <summary>Recovery ladder, executed in order across retry attempts.
    /// Defaults to the full owner-specified ladder.</summary>
    public IReadOnlyList<PrefillRecoveryStep> LadderSteps { get; init; } = DefaultLadder();

    /// <summary>Owner default ladder: fresh-slot → smaller-chunk → draft/ngram
    /// disabled → fail loud.</summary>
    public static IReadOnlyList<PrefillRecoveryStep> DefaultLadder() => new[]
    {
        PrefillRecoveryStep.FreshSlotRetry,
        PrefillRecoveryStep.SmallerChunkRetry,
        PrefillRecoveryStep.DraftDisabledRetry,
        PrefillRecoveryStep.FailLoud,
    };

    /// <summary>Fail-closed validation: every constraint violation is a typed
    /// argument error naming the field — never a silent clamp.</summary>
    public void Validate()
    {
        if (GapMultiplier <= 0) throw new ArgumentException($"{SectionName}.gap_multiplier must be > 0", nameof(GapMultiplier));
        if (MaxRetries < 0) throw new ArgumentException($"{SectionName}.max_retries must be >= 0", nameof(MaxRetries));
        if (ChunkSizeTokens <= 0) throw new ArgumentException($"{SectionName}.chunk_size_tokens must be > 0", nameof(ChunkSizeTokens));
        if (MinChunkSizeTokens <= 0) throw new ArgumentException($"{SectionName}.min_chunk_size_tokens must be > 0", nameof(MinChunkSizeTokens));
        if (MinChunkSizeTokens > ChunkSizeTokens) throw new ArgumentException($"{SectionName}.min_chunk_size_tokens must be <= chunk_size_tokens", nameof(MinChunkSizeTokens));
        if (BaselineWindow < 2) throw new ArgumentException($"{SectionName}.baseline_window must be >= 2", nameof(BaselineWindow));
        if (LadderSteps.Count == 0) throw new ArgumentException($"{SectionName}.ladder_steps must not be empty", nameof(LadderSteps));
        if (LadderSteps[^1] != PrefillRecoveryStep.FailLoud) throw new ArgumentException($"{SectionName}.ladder_steps must terminate in FailLoud", nameof(LadderSteps));
    }
}