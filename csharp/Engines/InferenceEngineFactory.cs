using ProvostXInference.Abstractions;
using ProvostXInference.Decision.Abstractions;
using ProvostXInference.Decision.Engine;
using ProvostXInference.Implementation;

namespace ProvostXInference.Engines;

/// <summary>
/// Default factory (composition root of the facade): native GGUF loading via
/// <see cref="GgufModelLoader"/> (thin seam over the wrapper-layer static
/// factory), decision loading via the existing
/// <see cref="IDecisionSessionFactory"/> lane. No static mutable state.
/// </summary>
public sealed class InferenceEngineFactory : IInferenceEngineFactory
{
    private readonly IGgufModelLoader _ggufLoader;
    private readonly IDecisionSessionFactory _decisionFactory;
    private readonly IExecutionProviderDefaultSource _epDefaultSource;
    private readonly IInferenceBackend _backend;

    public InferenceEngineFactory(IGgufModelLoader? ggufLoader = null,
        IDecisionSessionFactory? decisionFactory = null,
        IExecutionProviderDefaultSource? epDefaultSource = null,
        IInferenceBackend? backend = null)
    {
        _ggufLoader = ggufLoader ?? new GgufModelLoader();
        _decisionFactory = decisionFactory ?? new DecisionSessionFactory();
        _epDefaultSource = epDefaultSource ?? new OsExecutionProviderDefaultSource();
        _backend = backend ?? new NativeInferenceBackend();
    }

    public IInferenceEngine Create(EngineRequest request)
    {
        ArgumentNullException.ThrowIfNull(request);
        request.Validate();
        return new InferenceEngine(request, _ggufLoader, _decisionFactory, _epDefaultSource, _backend);
    }
}
