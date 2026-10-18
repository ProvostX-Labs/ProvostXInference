namespace ProvostXInference.Decision.Engine;

/// <summary>
/// One decision-engine question batch on the Jev wire: the SERIALIZED state text
/// (rule owned by the server layer's state serializer — the engine plane only sees
/// text) and the typed questions to answer. Questions = null → the judge's
/// CANONICAL trained question set (the loaded session's spec — our additive
/// default for omitted questions; real Jev requires questions). One request → one
/// answer set. Pure data.
/// </summary>
public sealed record JudgeStateQuestions(
    string StateText,
    IReadOnlyList<JudgeQuestionDef>? Questions);
