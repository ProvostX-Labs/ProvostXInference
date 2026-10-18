using ProvostXInference.Decision.Engine;

namespace ProvostXInference.Engines;

/// <summary>
/// Fluent builder for <see cref="DecisionEngineRequest"/> (ONNX judge lane).
/// <see cref="Build"/> yields a VALIDATED request (fail-closed).
/// </summary>
public sealed class DecisionEngineRequestBuilder
{
    private readonly string _modelPath;

    public DecisionEngineRequestBuilder(string modelPath) => _modelPath = modelPath;

    public string? ExecutionProviders { get; private set; }
    public JudgeNumericsGate? NumericsGate { get; private set; }
    public string? TokenizerJsonPath { get; private set; }
    public string? SpecJsonPath { get; private set; }

    /// <summary>Tokenizer JSON path (required by the judge lane).</summary>
    public DecisionEngineRequestBuilder WithTokenizer(string tokenizerJsonPath)
        { TokenizerJsonPath = tokenizerJsonPath; return this; }

    /// <summary>Judge sequence-spec JSON path (required by the judge lane).</summary>
    public DecisionEngineRequestBuilder WithSpec(string specJsonPath)
        { SpecJsonPath = specJsonPath; return this; }

    /// <summary>Raw EP chain (LLM-10 format, e.g. "coreml-&gt;cpu"); null = OS default.</summary>
    public DecisionEngineRequestBuilder WithExecutionProviders(string? chain)
        { ExecutionProviders = chain; return this; }

    /// <summary>Numerics gate for non-CPU EPs (drift = walk down the chain).</summary>
    public DecisionEngineRequestBuilder WithNumericsGate(JudgeNumericsGate? gate)
        { NumericsGate = gate; return this; }

    /// <summary>Validated request (fail-closed — Build() itself validates).</summary>
    public DecisionEngineRequest Build()
    {
        var request = new DecisionEngineRequest
        {
            ModelPath = _modelPath,
            TokenizerJsonPath = TokenizerJsonPath ?? string.Empty,
            SpecJsonPath = SpecJsonPath ?? string.Empty,
            ExecutionProviders = ExecutionProviders,
            NumericsGate = NumericsGate,
        };
        request.Validate();
        return request;
    }

    /// <summary>Build + create the engine (still UNLOADED — call Initialize()).</summary>
    public IInferenceEngine Create(IInferenceEngineFactory factory) => factory.Create(Build());
}