using ProvostXInference.Decision.Engine;

namespace ProvostXInference.Engines;

/// <summary>
/// Decision (ONNX judge) lane ops view: batch judging on the ORT session with
/// truthful active-provider reporting, plus the tokenizer/spec the caller needs
/// to build records. Wraps <see cref="DecisionSession"/>; adds nothing native.
/// </summary>
public interface IDecisionEngineOps
{
    /// <summary>The provider that ACTUALLY serves this session (never a silent GPU name).</summary>
    string ActiveProvider { get; }

    /// <summary>The judge tokenizer (shared asset).</summary>
    JudgeTokenizer Tokenizer { get; }

    /// <summary>The judge sequence spec.</summary>
    JudgeSequenceSpec Spec { get; }

    /// <summary>Batch-judge decision records (training-eval decode semantics).</summary>
    IReadOnlyList<JudgeScore> Judge(IReadOnlyList<DecisionRecord> records);

    /// <summary>Batch-judge state question sets.</summary>
    IReadOnlyList<JudgeQuestionAnswerSet> JudgeQuestionSet(IReadOnlyList<JudgeStateQuestions> requests);
}