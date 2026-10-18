namespace ProvostXInference.Decision.Engine;

/// <summary>
/// Judge question types, numbered EXACTLY as the judge models were trained
/// (laya QTYPES): choice=0, score=1, noul=2. The value feeds the model's
/// qtype input — never renumber.
/// </summary>
public enum JudgeQuestionType
{
    Choice = 0,
    Score = 1,
    Noul = 2,
}