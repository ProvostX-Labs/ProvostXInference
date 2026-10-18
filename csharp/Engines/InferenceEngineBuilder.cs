namespace ProvostXInference.Engines;

/// <summary>
/// Fluent entry point of the engine-object contract:
/// <code>
/// var engine = InferenceEngineBuilder.Gguf("/path/model.gguf")
///     .WithMmproj("/path/mmproj.gguf").WithGpuLayers(99)
///     .Build();                    // → validated GgufEngineRequest
/// engine.Create(factory).Initialize();   // → live IInferenceEngine
/// </code>
/// No static mutable state — the builder is a local, disposable scratch record.
/// </summary>
public static class InferenceEngineBuilder
{
    /// <summary>Start a GGUF generative engine request.</summary>
    public static GgufEngineRequestBuilder Gguf(string modelPath) => new(modelPath);

    /// <summary>Start an ONNX decision (judge) engine request.</summary>
    public static DecisionEngineRequestBuilder Decision(string modelPath) => new(modelPath);

    /// <summary>Start a vision-encoder engine request (model + required mmproj).</summary>
    public static EncoderEngineRequestBuilder Encoder(string modelPath) => new(modelPath);
}