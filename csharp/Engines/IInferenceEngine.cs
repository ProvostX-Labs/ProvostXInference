namespace ProvostXInference.Engines;

/// <summary>
/// The SOLE caller-facing object of Inference's world: one loaded backend (GGUF
/// llama.cpp, ONNX decision, or vision encoder) behind a typed facade with
/// lifecycle, inference ops, context management, KV-cache ops, and every knob
/// as a property. Callers (e.g. ProvostXLLM) hold an array of these and route
/// requests by <see cref="Kind"/>/<see cref="ModelPath"/>.
/// </summary>
public interface IInferenceEngine : IDisposable
{
    /// <summary>Which backend lane this engine serves.</summary>
    EngineKind Kind { get; }

    /// <summary>The validated contract this engine was built from.</summary>
    EngineRequest Request { get; }

    /// <summary>Primary model artifact path (identity for routing).</summary>
    string ModelPath { get; }

    /// <summary>
    /// Native specialist id actually serving GGUF/Encoder engines (BL-02 matrix);
    /// null = platform default. Always null on the managed Decision lane.
    /// </summary>
    string? ActiveSpecialist { get; }

    /// <summary>Whether <see cref="Initialize"/> has loaded the backend.</summary>
    bool IsLoaded { get; }

    /// <summary>Load the backend (fail-closed; throws on any load error). Idempotent.</summary>
    void Initialize();

    /// <summary>GGUF lane view. Throws <see cref="Exceptions.EngineKindMismatchException"/> off-lane.</summary>
    IGgufEngineOps Gguf { get; }

    /// <summary>Decision lane view. Throws <see cref="Exceptions.EngineKindMismatchException"/> off-lane.</summary>
    IDecisionEngineOps Decision { get; }

    /// <summary>Encoder lane view. Throws <see cref="Exceptions.EngineKindMismatchException"/> off-lane.</summary>
    IEncoderEngineOps Encoder { get; }
}