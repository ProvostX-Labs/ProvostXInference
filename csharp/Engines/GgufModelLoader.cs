using ProvostXInference.Abstractions;
using ProvostXInference.Models;
using ProvostXInference.Implementation;

namespace ProvostXInference.Engines;

/// <summary>
/// Production <see cref="IGgufModelLoader"/>: thin wrapper over the native
/// wrapper-layer factories (<see cref="NativeInferenceModel.Load"/> +
/// <see cref="IInferenceModel.LoadVisionEncoder"/>). No state, no fallback.
/// </summary>
public sealed class GgufModelLoader : IGgufModelLoader
{
    public IInferenceModel Load(ModelConfig config) => NativeInferenceModel.Load(config);

    public IVisionEncoder LoadVisionEncoder(IInferenceModel model, string mmprojPath)
    {
        ArgumentNullException.ThrowIfNull(model);
        return model.LoadVisionEncoder(mmprojPath);
    }
}