using ProvostXInference.Models;

namespace ProvostXInference.Abstractions;

/// <summary>Facade view of the stall-recovery ladder (INF BL-33): owns the
/// configured step order and the retry budget; returns the step for a
/// given 0-based retry attempt and fails closed when the budget is
/// exhausted (the caller is expected to throw the typed unrecovered error
/// then — the ladder itself never invents steps). Surfaced behind the
/// Engines facade (<c>StallWatchdogOps.CreateRecoveryLadder</c>) so callers
/// never touch the Implementation namespace.</summary>
public interface IStallRecoveryLadder
{
    /// <summary>Number of recovery attempts before the ladder is exhausted.</summary>
    int MaxRetries { get; }

    /// <summary>Step to execute for <paramref name="attempt"/> (0-based).
    /// Returns null when the retry budget is exhausted — the caller must fail
    /// loud (typed error) rather than loop.</summary>
    PrefillRecoveryStep? StepFor(int attempt);
}
