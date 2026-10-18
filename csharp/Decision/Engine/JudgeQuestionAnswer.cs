namespace ProvostXInference.Decision.Engine;

/// <summary>
/// ONE raw model answer for one Jev question (session decode → response mapper
/// boundary). RAW semantics: the model's argmax choice / expected score / noul
/// probability — NO risk-escalation rule applied (the escalation owner is
/// JudgeVerdictMapper on the enriched arms; the raw wire stays raw).
/// Field populations per type: choice → Choice + Confidence + Probabilities;
/// score → Score + Confidence + Probabilities (level-index keyed) + Legend;
/// noul → Noul. Pure data.
/// </summary>
public sealed record JudgeQuestionAnswer(
    string QuestionId,
    JudgeQuestionType Type,
    string? Choice = null,
    double? Score = null,
    double? Confidence = null,
    double? Noul = null,
    IReadOnlyDictionary<string, double>? Probabilities = null,
    IReadOnlyDictionary<string, string>? Legend = null);
