using ProvostXInference.Interop;

namespace ProvostXInference.Models;

/// <summary>User-facing context configuration.</summary>
public sealed record ContextConfig
{
    public uint ContextSize { get; init; } = 4096;
    public uint BatchSize { get; init; } = 512;
    public uint SeqMax { get; init; } = 4;
    public PoolingType PoolingType { get; init; } = PoolingType.None;
}

/// <summary>Managed enum mirroring PvxPoolingType.</summary>
public enum PoolingType { None = 0, Mean = 1, Cls = 2, Last = 3, /// <summary>ABI v8: rerank lane (classification head).</summary>
    Rank = 4 }
