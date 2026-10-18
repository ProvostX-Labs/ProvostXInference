using Microsoft.ML.OnnxRuntime;

namespace ProvostXInference.Decision.Engine;

/// <summary>
/// Test-support surface for the decision engine slice (Inference-side, consumed
/// by both repos' test suites): opens an ORT session with ONE pinned provider.
/// Keeping the only direct ORT-construction site inside ProvostXInference is the
/// LLM→Inference extraction contract — LLM-side code references no ORT types.
/// </summary>
public static class DecisionTestSupport
{
    /// <summary>Open a <see cref="DecisionSession"/> over <paramref name="modelPath"/>
    /// with the provider pinned by name (test stubs report a known provider).</summary>
    public static DecisionSession OpenSessionWithProvider(
        string modelPath, JudgeTokenizer tokenizer, JudgeSequenceSpec spec, string provider)
        => new(tokenizer, spec, new InferenceSession(modelPath), provider);
}
