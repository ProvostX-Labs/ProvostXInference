using ProvostXInference.Abstractions;
using ProvostXInference.Models;

namespace ProvostXInference.Engines;

/// <summary>
/// Facade factory for the stall-watchdog helper family (INF BL-33, owner
/// black-box directive 2026-10-19): the rolling-median detector, the
/// recovery ladder and the production wall clock are ENGINE-LEVEL utilities
/// (backend-independent, pure managed) — but callers must consume them
/// through the facade surface, never the Implementation namespace. This
/// static factory is the sole construction seam; the returned types are the
/// Abstractions interfaces. Fail-closed: a null/invalid config throws (the
/// underlying helpers validate in their constructors).
/// </summary>
public static class StallWatchdogOps
{
    /// <summary>Rolling-median stall detector for one prefill sequence.</summary>
    public static IPrefillStallDetector CreateDetector(StallWatchdogConfig config)
        => new Implementation.RollingPrefillStallDetector(config ?? throw new ArgumentNullException(nameof(config)));

    /// <summary>Recovery ladder (step order + retry budget) for one watchdog.</summary>
    public static IStallRecoveryLadder CreateRecoveryLadder(StallWatchdogConfig config)
        => new Implementation.StallRecoveryLadder(config ?? throw new ArgumentNullException(nameof(config)));

    /// <summary>Production wall clock backed by the injected <see cref="TimeProvider"/>.</summary>
    public static IChunkWallClock CreateWallClock(TimeProvider timeProvider)
        => new Implementation.SystemChunkWallClock(timeProvider);
}
