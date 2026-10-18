using ProvostXInference.Exceptions;
using ProvostXInference.Models;

namespace ProvostXInference.Engines;

/// <summary>
/// Contract for a GGUF generative engine on the native llama.cpp backend:
/// model path, optional vision projector (mmproj) sidecar, optional speculative
/// draft-model sidecar, and the full native load knob set (BL-02 engine matrix,
/// GPU layers, threads, flash attention, KV type, reasoning markers) plus
/// context sizing. Knobs are load-time; live knobs live on the engine object.
/// </summary>
public sealed record GgufEngineRequest : EngineRequest
{
    /// <summary>Optional mmproj vision projector sidecar. Null = text-only engine.</summary>
    public string? MmprojPath { get; init; }

    /// <summary>Optional speculative-decoding draft model path (validated on enable).</summary>
    public string? DraftModelPath { get; init; }

    public int GpuLayers { get; init; } = -1;
    public int Threads { get; init; } = 0;
    public bool FlashAttention { get; init; } = true;
    public KvType KvCacheType { get; init; } = KvType.F16;
    public string[]? ReasoningMarkers { get; init; }

    public uint ContextSize { get; init; } = 4096;
    public uint BatchSize { get; init; } = 512;
    public uint SeqMax { get; init; } = 4;
    public PoolingType PoolingType { get; init; } = PoolingType.None;

    protected override void ValidateCore()
    {
        if (MmprojPath is not null && MmprojPath.Trim().Length == 0)
            throw new InvalidEngineRequestException("MmprojPath must be null or a non-empty path.");
        if (DraftModelPath is not null && DraftModelPath.Trim().Length == 0)
            throw new InvalidEngineRequestException("DraftModelPath must be null or a non-empty path.");
        if (GpuLayers < -1) throw new InvalidEngineRequestException("GpuLayers must be >= -1 (-1 = all).");
        if (Threads < 0) throw new InvalidEngineRequestException("Threads must be >= 0 (0 = auto).");
        if (ContextSize == 0) throw new InvalidEngineRequestException("ContextSize must be > 0.");
        if (BatchSize == 0) throw new InvalidEngineRequestException("BatchSize must be > 0.");
        if (SeqMax == 0) throw new InvalidEngineRequestException("SeqMax must be > 0.");
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
        ReasoningMarkers = ReasoningMarkers,
    };

    /// <summary>Map to the context config (validation already passed).</summary>
    public ContextConfig ToContextConfig() => new()
    {
        ContextSize = ContextSize,
        BatchSize = BatchSize,
        SeqMax = SeqMax,
        PoolingType = PoolingType,
    };
}