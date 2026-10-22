namespace ProvostXInference.Models;

/// <summary>
/// ABI v9 (per-engine telemetry, owner order 2026-10-22): MODEL-aggregate
/// snapshot — the roll-up of every executor/conversation lane on ONE engine
/// instance (never process-wide). Rates (tok/s etc.) are derived by the
/// caller between two snapshots; the engine stays stateless. Counters never
/// hide failures: rejected spec drafts derive as Drafted - Accepted.
/// </summary>
public sealed record EngineTelemetry(
    long PromptTokens,
    long DecodeSteps,
    long CommittedTokens,
    long SpecDrafted,
    long SpecAccepted,
    long Errors,
    long ContextsAlive);
