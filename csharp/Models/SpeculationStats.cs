namespace ProvostXInference.Models;

/// <summary>Spec-session acceptance telemetry for ONE speculative session
/// (executor or pooled conversation; ABI v7 adds the conversation lane):
/// cumulative drafted/accepted token counters since session start. Zero/zero
/// when no session is active — counters never hide rejections.</summary>
public sealed record SpeculationStats(long Drafted, long Accepted)
{
    public static readonly SpeculationStats None = new(0, 0);
}