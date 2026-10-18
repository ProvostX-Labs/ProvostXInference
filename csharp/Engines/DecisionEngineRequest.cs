using ProvostXInference.Exceptions;

namespace ProvostXInference.Engines;

/// <summary>
/// Contract for an ONNX decision (judge) engine on the managed ORT lane:
/// model path, tokenizer JSON, judge sequence-spec JSON, execution-provider
/// chain (raw string, LLM-10 format; null = OS default), optional numerics gate.
/// </summary>
public sealed record DecisionEngineRequest : EngineRequest
{
    /// <summary>Tokenizer JSON path (required by the judge lane).</summary>
    public required string TokenizerJsonPath { get; init; }

    /// <summary>Judge sequence-spec JSON path (required by the judge lane).</summary>
    public required string SpecJsonPath { get; init; }

    /// <summary>
    /// Raw execution-provider chain (LLM-10 format, e.g. "coreml->cpu").
    /// Null/empty = OS default chain. Parsed fail-closed by the lane.
    /// </summary>
    public string? ExecutionProviders { get; init; }

    /// <summary>
    /// Numerics gate for non-CPU EPs (LLM-10): drift vs the anchor record rejects
    /// the EP and walks down the chain. Null = no gate.
    /// </summary>
    public Decision.Engine.JudgeNumericsGate? NumericsGate { get; init; }

    protected override void ValidateCore()
    {
        if (string.IsNullOrWhiteSpace(TokenizerJsonPath))
            throw new InvalidEngineRequestException("TokenizerJsonPath is required (null/empty/whitespace).");
        if (string.IsNullOrWhiteSpace(SpecJsonPath))
            throw new InvalidEngineRequestException("SpecJsonPath is required (null/empty/whitespace).");
    }
}