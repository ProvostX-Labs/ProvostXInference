namespace ProvostXInference.Decision.Engine;

/// <summary>
/// The raw answers for ONE Jev request (one state), in question order, plus the
/// USAGE inputs: the tokenized state length (actual tokenizer count of the
/// serialized state — the usage.input_tokens definition) and the question count
/// (the usage.output_tokens definition: one answer per question). Pure data.
/// </summary>
public sealed record JudgeQuestionAnswerSet(
    IReadOnlyList<JudgeQuestionAnswer> Answers,
    int StateTokens);
