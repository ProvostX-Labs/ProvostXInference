namespace ProvostXInference.Decision.Engine;

/// <summary>
/// The FULL serialized inference specification for one judge model: record kind,
/// sequence budget (max_len/head_max_len exactly as trained), the confidence
/// temperatures fitted at export time, and the three typed questions in training
/// order. Loaded from a spec JSON (Assets/models/judge/judge-spec-*.json) —
/// the questions MUST match the training questions verbatim or the model scores
/// text it was never trained on.
/// </summary>
public sealed record JudgeSequenceSpec(
    JudgeRecordKind Kind,
    int MaxLen,
    int HeadMaxLen,
    IReadOnlyList<double> Temperature,
    IReadOnlyList<JudgeQuestionDef> Questions);
