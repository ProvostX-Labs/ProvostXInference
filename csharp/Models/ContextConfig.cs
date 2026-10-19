using ProvostXInference.Interop;

namespace ProvostXInference.Models;

/// <summary>User-facing context configuration.</summary>
public sealed record ContextConfig
{
    public uint ContextSize { get; init; } = 4096;
    public uint BatchSize { get; init; } = 512;
    public uint SeqMax { get; init; } = 4;
    public PoolingType PoolingType { get; init; } = PoolingType.None;
    /// <summary>ABI v8 wave-4: recurrent-state snapshot ring per sequence
    /// (upstream llama_context_params.n_rs_seq). 0 = no snapshots (llama
    /// default — pre-existing behavior). &gt;= 1 on rs-rollback archs
    /// (qwen4exp/qwen35/...) enables partial sequence rewind on hybrid
    /// memories — REQUIRED for draft-MTP speculative verify repair;
    /// the spec chain clamps to llama_n_rs_seq (upstream server rule).
    /// Non-rs-rollback archs clamp to 0 inside llama (harmless).</summary>
    public uint NRsSeq { get; init; } = 0;
}

/// <summary>Managed enum mirroring PvxPoolingType.</summary>
public enum PoolingType { None = 0, Mean = 1, Cls = 2, Last = 3, /// <summary>ABI v8: rerank lane (classification head).</summary>
    Rank = 4 }
