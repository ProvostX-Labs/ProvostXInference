using ProvostXInference.Models;

namespace ProvostXInference.Engines;

/// <summary>
/// Fluent builder for <see cref="EncoderEngineRequest"/> (vision encoder lane:
/// GGUF model + REQUIRED mmproj). <see cref="Build"/> yields a VALIDATED request.
/// </summary>
public sealed class EncoderEngineRequestBuilder
{
    private readonly string _modelPath;

    public EncoderEngineRequestBuilder(string modelPath) => _modelPath = modelPath;

    public string? MmprojPath { get; private set; }
    public string? EngineId { get; private set; }
    public int GpuLayers { get; private set; } = -1;
    public int Threads { get; private set; }
    public bool FlashAttention { get; private set; } = true;
    public KvType KvCacheType { get; private set; } = KvType.F16;

    /// <summary>REQUIRED mmproj projector path.</summary>
    public EncoderEngineRequestBuilder WithMmproj(string mmprojPath) { MmprojPath = mmprojPath; return this; }

    /// <summary>Native specialist id; null = platform default.</summary>
    public EncoderEngineRequestBuilder WithEngineId(string? engineId) { EngineId = engineId; return this; }

    public EncoderEngineRequestBuilder WithGpuLayers(int gpuLayers) { GpuLayers = gpuLayers; return this; }

    public EncoderEngineRequestBuilder WithThreads(int threads) { Threads = threads; return this; }

    public EncoderEngineRequestBuilder WithFlashAttention(bool on) { FlashAttention = on; return this; }

    public EncoderEngineRequestBuilder WithKvCacheType(KvType kvType) { KvCacheType = kvType; return this; }

    /// <summary>Validated request (fail-closed — Build() itself validates).</summary>
    public EncoderEngineRequest Build()
    {
        var request = new EncoderEngineRequest
        {
            ModelPath = _modelPath,
            MmprojPath = MmprojPath ?? string.Empty,
            EngineId = EngineId,
            GpuLayers = GpuLayers,
            Threads = Threads,
            FlashAttention = FlashAttention,
            KvCacheType = KvCacheType,
        };
        request.Validate();
        return request;
    }

    /// <summary>Build + create the engine (still UNLOADED — call Initialize()).</summary>
    public IInferenceEngine Create(IInferenceEngineFactory factory) => factory.Create(Build());
}