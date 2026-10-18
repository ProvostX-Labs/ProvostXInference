namespace ProvostXInference.Decision.Engine;

/// <summary>
/// Discriminator for judge decision records on the decision-engine wire
/// /v1/systemone (two record kinds on the same endpoint — the enriched arms;
/// the raw Jev shape carries a free-form state instead).
/// </summary>
public enum JudgeRecordKind
{
    /// <summary>Shell command decision: input {command, cwd, purpose}.</summary>
    Shell = 0,

    /// <summary>Process decision: PROCESS os= | exe= | chain= | cmd= | net= | writes=.</summary>
    Process = 1,
}