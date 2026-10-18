using ProvostXInference.Exceptions;

namespace ProvostXInference.Engines;

/// <summary>
/// The typed engine CONTRACT base: everything a caller must tell Inference to get
/// an <see cref="IInferenceEngine"/>. Derived records add lane-specific parameters.
/// <see cref="Validate"/> is fail-closed — any invalid field throws
/// <see cref="InvalidEngineRequestException"/> before a backend is ever touched.
/// </summary>
public abstract record EngineRequest
{
    /// <summary>
    /// Native specialist engine id ("metal" / "cuda" / "vulkan", BL-02 matrix).
    /// Null = platform default. Ignored by the managed Decision lane.
    /// </summary>
    public string? EngineId { get; init; }

    /// <summary>Filesystem path of the primary model artifact (GGUF or ONNX). Required.</summary>
    public required string ModelPath { get; init; }

    /// <summary>Fail-closed validation. Sealed core + lane hooks.</summary>
    public void Validate()
    {
        if (string.IsNullOrWhiteSpace(ModelPath))
            throw new InvalidEngineRequestException("ModelPath is required (null/empty/whitespace).");
        ValidateCore();
    }

    /// <summary>Lane-specific validation hook (base is valid by construction).</summary>
    protected virtual void ValidateCore() { }
}