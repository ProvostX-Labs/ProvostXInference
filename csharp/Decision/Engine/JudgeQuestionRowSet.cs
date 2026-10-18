namespace ProvostXInference.Decision.Engine;

/// <summary>
/// Encoded question rows for ONE Jev state + the pre-truncation token count of
/// the serialized state (usage.input_tokens source). One row per question, in
/// question order. Pure data.
/// </summary>
public sealed record JudgeQuestionRowSet(
    IReadOnlyList<JudgeRowEncoding> Rows,
    int StateTokens);
