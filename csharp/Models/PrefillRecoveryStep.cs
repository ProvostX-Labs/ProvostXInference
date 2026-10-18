namespace ProvostXInference.Models;

/// <summary>Recovery-ladder step executed after a detected prefill stall
/// (stall-safe prefill layer, INF BL-33). Ordered from cheapest to
/// fail-loud; the ladder always terminates in <see cref="PrefillRecoveryStep.FailLoud"/>
/// so a persistently stalling prefill never spins silently.</summary>
public enum PrefillRecoveryStep
{
    /// <summary>Return the current conversation to the pool and continue on a
    /// freshly leased slot (replays the committed prefix first).</summary>
    FreshSlotRetry,

    /// <summary>Retry the stalled chunk with the chunk size halved (floor:
    /// <see cref="StallWatchdogConfig.MinChunkSizeTokens"/>).</summary>
    SmallerChunkRetry,

    /// <summary>Retry the stalled chunk with draft/ngram speculative helpers
    /// disabled via the constructor-injected suspension hook.</summary>
    DraftDisabledRetry,

    /// <summary>Fail loud: throw a typed <c>PrefillStallUnrecoveredException</c>
    /// naming the chunk, the timings, and the ladder steps attempted.</summary>
    FailLoud,
}