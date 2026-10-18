using ProvostXInference.Models;

namespace ProvostXInference.Abstractions;

/// <summary>Typed stall detector over per-chunk prefill wall timings
/// (INF BL-33). Stateful across the chunks of one prefill: it maintains the
/// rolling baseline itself.</summary>
public interface IPrefillStallDetector
{
    /// <summary>Observe one completed chunk commit and return the typed
    /// verdict (stall vs clean) with the evidence attached.</summary>
    PrefillStallVerdict Observe(int chunkIndex, int chunkTokens, double wallSeconds);
}