using ProvostXInference.Decision.Engine;

namespace ProvostXInference.Engines;

/// <summary>
/// Decision lane ops over an existing <see cref="DecisionSession"/> (loaded by
/// the factory through <see cref="Decision.Abstractions.IDecisionSessionFactory"/>).
/// </summary>
public sealed class DecisionEngineOps : IDecisionEngineOps, IDisposable
{
    private readonly DecisionSession _session;

    public DecisionEngineOps(DecisionSession session)
    {
        ArgumentNullException.ThrowIfNull(session);
        _session = session;
    }

    public string ActiveProvider => _session.ActiveProvider;
    public JudgeTokenizer Tokenizer => _session.Tokenizer;
    public JudgeSequenceSpec Spec => _session.Spec;
    public IReadOnlyList<JudgeScore> Judge(IReadOnlyList<DecisionRecord> records) => _session.Judge(records);
    public IReadOnlyList<JudgeQuestionAnswerSet> JudgeQuestionSet(IReadOnlyList<JudgeStateQuestions> requests) =>
        _session.JudgeQuestionSet(requests);

    public void Dispose() => _session.Dispose();
}