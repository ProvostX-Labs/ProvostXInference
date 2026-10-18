namespace ProvostXInference.Decision.Engine;

/// <summary>
/// Raw model output for one judged record (executor → response mapper boundary).
/// The mapper owns the risk-escalation rule; the executor reports raw scores only.
/// </summary>
/// <param name="RawAction">Model's raw verdict action (pre-escalation).</param>
/// <param name="Confidence">0..1 model confidence.</param>
/// <param name="Risk">0..4 risk score.</param>
/// <param name="NeedsSandbox">0..1 sandbox recommendation score.</param>
/// <param name="ModelVersion">Judge model version reported by the executor.</param>
public sealed record JudgeScore(
    JudgeAction RawAction,
    double Confidence,
    double Risk,
    double NeedsSandbox,
    string ModelVersion);