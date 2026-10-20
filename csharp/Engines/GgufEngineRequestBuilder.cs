using ProvostXInference.Models;

namespace ProvostXInference.Engines;

/// <summary>
/// Fluent builder for <see cref="GgufEngineRequest"/>. <see cref="Build"/> yields a
/// VALIDATED request (fail-closed); <see cref="Create"/> builds + factory-creates
/// an engine in one step.
/// </summary>
public sealed class GgufEngineRequestBuilder
{
    private readonly string _modelPath;

    public GgufEngineRequestBuilder(string modelPath) => _modelPath = modelPath;

    public string? EngineId { get; private set; }
    public string? MmprojPath { get; private set; }
    public string? DraftModelPath { get; private set; }
    public int GpuLayers { get; private set; } = -1;
    public int Threads { get; private set; }
    public bool FlashAttention { get; private set; } = true;
    public KvType KvCacheType { get; private set; } = KvType.F16;
    public string[]? ReasoningMarkers { get; private set; }
    public uint ContextSize { get; private set; } = 4096;
    public uint BatchSize { get; private set; } = 512;
    public uint SeqMax { get; private set; } = 4;
    public PoolingType PoolingType { get; private set; } = PoolingType.None;

    /// <summary>BL-21 wave-5 advanced knobs (set via <see cref="WithAdvanced"/>).</summary>
    public ModelConfig? AdvancedModel { get; private set; }
    public ContextConfig? AdvancedContext { get; private set; }

    /// <summary>Native specialist id ("metal"/"cuda"/"vulkan"); null = platform default.</summary>
    public GgufEngineRequestBuilder WithEngineId(string? engineId) { EngineId = engineId; return this; }

    /// <summary>Optional vision projector sidecar.</summary>
    public GgufEngineRequestBuilder WithMmproj(string? path) { MmprojPath = path; return this; }

    /// <summary>Optional speculative-decoding draft model sidecar.</summary>
    public GgufEngineRequestBuilder WithDraftModel(string? path) { DraftModelPath = path; return this; }

    /// <summary>GPU layers (-1 = all).</summary>
    public GgufEngineRequestBuilder WithGpuLayers(int gpuLayers) { GpuLayers = gpuLayers; return this; }

    /// <summary>CPU threads (0 = auto).</summary>
    public GgufEngineRequestBuilder WithThreads(int threads) { Threads = threads; return this; }

    public GgufEngineRequestBuilder WithFlashAttention(bool on) { FlashAttention = on; return this; }

    /// <summary>llama.cpp KV cache quantization knob.</summary>
    public GgufEngineRequestBuilder WithKvCacheType(KvType kvType) { KvCacheType = kvType; return this; }

    /// <summary>Reasoning-marker probe list (data-driven; default {"think"}-family).</summary>
    public GgufEngineRequestBuilder WithReasoningMarkers(params string[] markers)
        { ReasoningMarkers = markers; return this; }

    /// <summary>Context window size.</summary>
    public GgufEngineRequestBuilder WithContextSize(uint ctx) { ContextSize = ctx; return this; }

    public GgufEngineRequestBuilder WithBatchSize(uint batch) { BatchSize = batch; return this; }

    /// <summary>Max parallel KV sequences.</summary>
    public GgufEngineRequestBuilder WithSeqMax(uint seqMax) { SeqMax = seqMax; return this; }

    public GgufEngineRequestBuilder WithPooling(PoolingType pooling) { PoolingType = pooling; return this; }

    /// <summary>BL-21 wave-5 pass-through: advanced model-load and/or context knobs.
    /// The classic With* values take precedence; the advanced records supply the
    /// extended knob surface (kv overrides, rope/yarn, tri-states, …).</summary>
    public GgufEngineRequestBuilder WithAdvanced(ModelConfig? model = null, ContextConfig? context = null)
    {
        AdvancedModel = model;
        AdvancedContext = context;
        return this;
    }

    /// <summary>Validated request (fail-closed — Build() itself validates).</summary>
    public GgufEngineRequest Build()
    {
        var request = new GgufEngineRequest
        {
            ModelPath = _modelPath,
            EngineId = EngineId,
            MmprojPath = MmprojPath,
            DraftModelPath = DraftModelPath,
            GpuLayers = GpuLayers,
            Threads = Threads,
            FlashAttention = FlashAttention,
            KvCacheType = KvCacheType,
            ReasoningMarkers = ReasoningMarkers,
            ContextSize = ContextSize,
            BatchSize = BatchSize,
            SeqMax = SeqMax,
            PoolingType = PoolingType,
            AdvancedModel = AdvancedModel,
            AdvancedContext = AdvancedContext,
        };
        request.Validate();
        return request;
    }

    /// <summary>Build + create the engine (still UNLOADED — call Initialize()).</summary>
    public IInferenceEngine Create(IInferenceEngineFactory factory) => factory.Create(Build());
}