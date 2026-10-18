using ProvostXInference.Abstractions;
using ProvostXInference.Models;

namespace ProvostXInference.Engines;

/// <summary>
/// Testability seam over the native GGUF loader: the engine factory depends on
/// THIS interface, not on <see cref="Implementation.NativeInferenceModel"/>'s static
/// factory, so the facade is unit-testable without a native image.
/// </summary>
public interface IGgufModelLoader
{
    /// <summary>Load a native model (fail-closed engine selection inside).</summary>
    IInferenceModel Load(ModelConfig config);

    /// <summary>Load the mmproj vision projector for an already-loaded model.</summary>
    IVisionEncoder LoadVisionEncoder(IInferenceModel model, string mmprojPath);
}