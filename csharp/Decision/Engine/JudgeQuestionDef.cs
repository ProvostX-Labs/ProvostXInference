namespace ProvostXInference.Decision.Engine;

/// <summary>
/// One typed judge question as trained (id, type, instructions, ordered options).
/// Pure data — the judge spec carries these in the exact training order
/// (action, needs_sandbox, risk_level).
/// </summary>
public sealed record JudgeQuestionDef(
    string Id,
    JudgeQuestionType Type,
    string Instructions,
    IReadOnlyList<JudgeOption> Options);