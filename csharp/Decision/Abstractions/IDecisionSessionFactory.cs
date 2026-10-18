using ProvostXInference.Decision.Engine;

namespace ProvostXInference.Decision.Abstractions;

/// <summary>
/// Creates <see cref="DecisionSession"/>s for a decision (judge) model file:
/// builds the ORT session options (full graph optimization, matching the
/// training runtime) and walks the parsed <see cref="ExecutionProviderChain"/>
/// in preference order. EP FALLBACK RULE: a provider that fails to initialize
/// degrades to the next entry — CPU is always the terminal fallback and is the
/// only provider REQUIRED to work. The session's ACTUALLY-active provider is
/// recorded so health reporting is truthful. One factory per host; thread-safe.
/// </summary>
public interface IDecisionSessionFactory
{
    /// <summary>
    /// Initialize an <see cref="DecisionSession"/> for <paramref name="modelPath"/>.
    /// Throws when NO provider in the chain could initialize (mapped to model_invalid).
    /// </summary>
    DecisionSession Create(
        string modelPath,
        JudgeTokenizer tokenizer,
        JudgeSequenceSpec spec,
        ExecutionProviderChain chain,
        JudgeNumericsGate? numericsGate = null);
}
