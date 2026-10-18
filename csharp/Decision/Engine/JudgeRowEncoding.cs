namespace ProvostXInference.Decision.Engine;

/// <summary>
/// Encoded ONE question row of a judge record: token ids, marker (MASK) offsets,
/// and the question's type. One record → one row per question, in question order.
/// Pure data.
/// </summary>
public sealed record JudgeRowEncoding(
    JudgeQuestionType QuestionType,
    int[] InputIds,
    int[] MarkerPositions);
