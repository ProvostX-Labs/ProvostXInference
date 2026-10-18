using ProvostXInference.Abstractions;
using ProvostXInference.Decision.Abstractions;
using ProvostXInference.Decision.Engine;
using ProvostXInference.Exceptions;
using ProvostXInference.Models;

namespace ProvostXInference.Engines;

/// <summary>
/// Facade implementation: created UNLOADED by <see cref="InferenceEngineFactory"/>,
/// loaded on <see cref="Initialize"/> per kind, torn down on Dispose (ops first,
/// then backend). All deps constructor-injected; no static state.
/// </summary>
public sealed class InferenceEngine : IInferenceEngine
{
    private readonly IGgufModelLoader _ggufLoader;
    private readonly IDecisionSessionFactory _decisionFactory;
    private readonly IExecutionProviderDefaultSource _epDefaultSource;
    private readonly IInferenceBackend _backend;
    private readonly SamplingConfig _defaultSampling;
    private Abstractions.IInferenceModel? _model;   // Gguf + Encoder backends
    private GgufEngineOps? _ggufOps;
    private DecisionEngineOps? _decisionOps;
    private EncoderEngineOps? _encoderOps;
    private bool _disposed;

    internal InferenceEngine(EngineRequest request, IGgufModelLoader ggufLoader,
        IDecisionSessionFactory decisionFactory, IExecutionProviderDefaultSource epDefaultSource,
        IInferenceBackend backend, SamplingConfig? defaultSampling = null)
    {
        Request = request;
        _ggufLoader = ggufLoader;
        _decisionFactory = decisionFactory;
        _epDefaultSource = epDefaultSource;
        _backend = backend;
        _defaultSampling = defaultSampling ?? new SamplingConfig();
    }

    public EngineKind Kind => Request switch
    {
        GgufEngineRequest => EngineKind.Gguf,
        DecisionEngineRequest => EngineKind.Decision,
        EncoderEngineRequest => EngineKind.Encoder,
        _ => throw new InvalidEngineRequestException($"Unknown request type {Request.GetType().Name}."),
    };

    public EngineRequest Request { get; }
    public string ModelPath => Request.ModelPath;
    public string? ActiveSpecialist => Request.EngineId;
    public bool IsLoaded => _ggufOps is not null || _decisionOps is not null || _encoderOps is not null;

    public void Initialize()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (IsLoaded) return;
        Request.Validate(); // fail-closed re-validation at the load boundary
        switch (Request)
        {
            case GgufEngineRequest gguf:
                _model = _ggufLoader.Load(gguf.ToModelConfig());
                _ggufOps = new GgufEngineOps(_model, gguf.ToContextConfig(), _defaultSampling, _backend);
                break;
            case EncoderEngineRequest enc:
                _model = _ggufLoader.Load(enc.ToModelConfig());
                _encoderOps = new EncoderEngineOps(_model, _ggufLoader.LoadVisionEncoder(_model, enc.MmprojPath));
                break;
            case DecisionEngineRequest dec:
                var tokenizer = new JudgeTokenizer(dec.TokenizerJsonPath);
                var spec = DecisionAssetCatalog.LoadSpec(dec.SpecJsonPath);
                var chain = string.IsNullOrWhiteSpace(dec.ExecutionProviders)
                    ? _epDefaultSource.DefaultChain()
                    : ExecutionProviderChain.Parse(dec.ExecutionProviders, "engine-request");
                _decisionOps = new DecisionEngineOps(_decisionFactory.Create(dec.ModelPath, tokenizer, spec, chain, dec.NumericsGate));
                break;
            default:
                throw new InvalidEngineRequestException($"Unknown request type {Request.GetType().Name}.");
        }
    }

    public IGgufEngineOps Gguf => _ggufOps
        ?? throw new EngineKindMismatchException($"Engine kind {Kind} serves no Gguf view.");
    public IDecisionEngineOps Decision => _decisionOps
        ?? throw new EngineKindMismatchException($"Engine kind {Kind} serves no Decision view.");
    public IEncoderEngineOps Encoder => _encoderOps
        ?? throw new EngineKindMismatchException($"Engine kind {Kind} serves no Encoder view.");

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        // Ops views own their pools/contexts/sessions; model disposal follows.
        _ggufOps?.Dispose();
        _encoderOps?.Dispose();
        _decisionOps?.Dispose();
        _model?.Dispose();
        _ggufOps = null;
        _encoderOps = null;
        _decisionOps = null;
        _model = null;
    }
}