namespace ProvostXInference.Engines;

/// <summary>
/// Which backend lane an <see cref="IInferenceEngine"/> serves. One engine object
/// per lane; the facade routes ops views by this kind (fail-closed on mismatch).
/// </summary>
public enum EngineKind
{
    /// <summary>GGUF generative model on the native llama.cpp backend (ABI v4).</summary>
    Gguf = 0,

    /// <summary>ONNX judge/decision model on the managed ORT lane.</summary>
    Decision = 1,

    /// <summary>GGUF model + required vision projector (mmproj): image-embedding lane.</summary>
    Encoder = 2,
}