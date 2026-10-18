using ProvostXInference.Exceptions;
using ProvostXInference.Models;

namespace ProvostXInference.Engines;

/// <summary>
/// Contract for a vision ENCODER engine: a GGUF model (for its vocabulary/
/// tokenizer context) plus a REQUIRED mmproj projector. The engine's primary
/// lane is image embedding; generative GGUF ops are NOT exposed on this kind.
/// </summary>
public sealed record EncoderEngineRequest : EngineRequest
{
    /// <summary>REQUIRED mmproj projector path (an encoder without one is meaningless).</summary>
    public required string MmprojPath { get; init; }

    public int GpuLayers { get; init; } = -1;
    public int Threads { get; init; } = 0;
    public bool FlashAttention { get; init; } = true;
    public KvType KvCacheType { get; init; } = KvType.F16;

    protected override void ValidateCore()
    {
        if (string.IsNullOrWhiteSpace(MmprojPath))
            throw new InvalidEngineRequestException("Encoder requires a non-empty MmprojPath.");
        if (GpuLayers < -1) throw new InvalidEngineRequestException("GpuLayers must be >= -1 (-1 = all).");
        if (Threads < 0) throw new InvalidEngineRequestException("Threads must be >= 0 (0 = auto).");
    }

    /// <summary>Map to the native loader config (validation already passed).</summary>
    public ModelConfig ToModelConfig() => new()
    {
        Path = ModelPath,
        Engine = EngineId,
        GpuLayers = GpuLayers,
        Threads = Threads,
        FlashAttention = FlashAttention,
        KvCacheType = KvCacheType,
    };
}