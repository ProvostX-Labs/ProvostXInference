using ProvostXInference.Models;

namespace ProvostXInference.Implementation;

/// <summary>Recovery ladder for the stall-safe prefill layer (INF BL-33).
/// Owns the configured step order and the retry budget; returns the step for
/// a given 0-based retry attempt and fails closed when the budget is
/// exhausted (the caller is expected to throw the typed unrecovered error
/// then — the ladder itself never invents steps).</summary>
public sealed class StallRecoveryLadder
{
    private readonly IReadOnlyList<PrefillRecoveryStep> _steps;
    private readonly int _maxRetries;

    public StallRecoveryLadder(StallWatchdogConfig config)
    {
        config.Validate();
        _steps = config.LadderSteps;
        _maxRetries = config.MaxRetries;
    }

    /// <summary>Number of recovery attempts before the ladder is exhausted.</summary>
    public int MaxRetries => _maxRetries;

    /// <summary>Step to execute for <paramref name="attempt"/> (0-based).
    /// Returns null when the retry budget is exhausted — the caller must fail
    /// loud (typed error) rather than loop.</summary>
    public PrefillRecoveryStep? StepFor(int attempt)
    {
        if (attempt < 0) throw new ArgumentOutOfRangeException(nameof(attempt));
        if (attempt >= _maxRetries) return null;
        // Extra attempts beyond the configured step list fail loud directly.
        return attempt >= _steps.Count ? PrefillRecoveryStep.FailLoud : _steps[attempt];
    }
}