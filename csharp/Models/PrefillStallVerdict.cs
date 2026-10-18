namespace ProvostXInference.Models;

/// <summary>Typed result of one stall-detector observation (INF BL-33).
/// Immutable; carries the evidence so callers can log or fail with real
/// numbers instead of a bare boolean.</summary>
public sealed record PrefillStallVerdict
{
    public bool IsStall { get; init; }

    /// <summary>0-based index of the observed chunk within the prefill.</summary>
    public int ChunkIndex { get; init; }

    /// <summary>Token count of the observed chunk.</summary>
    public int ChunkTokens { get; init; }

    /// <summary>Wall seconds measured around the substrate chunk commit
    /// (backend-independent wall clock).</summary>
    public double ChunkWallSeconds { get; init; }

    /// <summary>Rolling median (seconds) of the recent baseline chunk walls at
    /// observation time; 0 while the baseline is still warming up.</summary>
    public double BaselineMedianSeconds { get; init; }

    /// <summary>Ratio chunk-wall / baseline-median; 0 when no baseline yet.</summary>
    public double Ratio { get; init; }

    public static PrefillStallVerdict Clean(int chunkIndex, int chunkTokens, double wallSeconds) =>
        new() { IsStall = false, ChunkIndex = chunkIndex, ChunkTokens = chunkTokens, ChunkWallSeconds = wallSeconds };
}