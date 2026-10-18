using ProvostXInference.Models;

namespace ProvostXInference.Exceptions;

/// <summary>Fail-loud terminal outcome of the stall-safe prefill recovery
/// ladder (INF BL-33): a prefill chunk kept stalling past the configured
/// retries. Carries the evidence (chunk index, wall, baseline, steps
/// attempted) so the failure is diagnosable — never a silent hang or a
/// quiet degradation.</summary>
public sealed class PrefillStallUnrecoveredException : Exception
{
    /// <summary>0-based index of the chunk that could not be committed.</summary>
    public int ChunkIndex { get; }

    /// <summary>Token count of the failing chunk.</summary>
    public int ChunkTokens { get; }

    /// <summary>Wall seconds of the last stall observation.</summary>
    public double LastStallWallSeconds { get; }

    /// <summary>Rolling baseline median at the last stall observation.</summary>
    public double BaselineMedianSeconds { get; }

    /// <summary>Ladder steps attempted, in order, before failing loud.</summary>
    public IReadOnlyList<PrefillRecoveryStep> AttemptedSteps { get; }

    public PrefillStallUnrecoveredException(
        int chunkIndex, int chunkTokens, double lastStallWallSeconds,
        double baselineMedianSeconds, IReadOnlyList<PrefillRecoveryStep> attemptedSteps,
        string? reason = null)
        : base(Describe(chunkIndex, chunkTokens, lastStallWallSeconds, baselineMedianSeconds, attemptedSteps, reason))
    {
        ChunkIndex = chunkIndex;
        ChunkTokens = chunkTokens;
        LastStallWallSeconds = lastStallWallSeconds;
        BaselineMedianSeconds = baselineMedianSeconds;
        AttemptedSteps = attemptedSteps;
    }

    private static string Describe(int chunkIndex, int chunkTokens, double wall, double median,
        IReadOnlyList<PrefillRecoveryStep> steps, string? reason) =>
        $"prefill stall unrecovered: chunk {chunkIndex} ({chunkTokens} tokens) wall {wall:F2}s vs baseline median {median:F2}s; "
        + $"ladder exhausted after [{string.Join(", ", steps)}]"
        + (string.IsNullOrEmpty(reason) ? "" : $"; reason: {reason}");
}