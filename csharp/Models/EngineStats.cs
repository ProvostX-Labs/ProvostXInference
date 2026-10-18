namespace ProvostXInference.Models;

/// <summary>
/// ABI v8 (wave 3, feature 12): copy-out stats snapshot of one lane
/// (executor or conversation). Spec counters mirror the spec telemetry
/// (0/0 when no session is active); rejected drafts derive as drafted-accepted.
/// </summary>
public sealed record EngineStats(
    long Tokens,
    long KvPositions,
    long PromptTokens,
    long DecodeSteps,
    long SpecDrafted,
    long SpecAccepted);
