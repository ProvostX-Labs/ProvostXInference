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

    // ── ABI v8 wave-5 (BL-21c) context controls — zeroed = pre-wave-5 ──
    /// <summary>n_ubatch. 0 = mirror BatchSize (pre-wave-5 mapping).</summary>
    public uint UbatchSize { get; init; } = 0;
    /// <summary>Max outputs per ubatch. 0 = llama default (= n_batch).</summary>
    public uint NOutputsMax { get; init; } = 0;
    /// <summary>Max outputs per sequence. 0 = llama default (= n_outputs_max).</summary>
    public uint NOutputsMaxPerSeq { get; init; } = 0;
    /// <summary>Batch threads. 0 = llama default (GGML auto).</summary>
    public int ThreadsBatch { get; init; } = 0;
    /// <summary>RoPE scaling: 0 UNSPECIFIED, 1 NONE, 2 LINEAR, 3 YARN.</summary>
    public int RopeScalingType { get; init; } = 0;
    /// <summary>RoPE base frequency. 0 = from model.</summary>
    public float RopeFreqBase { get; init; } = 0f;
    /// <summary>RoPE frequency scale. 0 = from model.</summary>
    public float RopeFreqScale { get; init; } = 0f;
    /// <summary>YaRN extrapolation mix. null/&lt;=0 = from model.</summary>
    public float YarnExtFactor { get; init; } = 0f;
    /// <summary>YaRN magnitude scale. null/&lt;=0 = from model.</summary>
    public float YarnAttnFactor { get; init; } = 0f;
    /// <summary>YaRN low-freq correction. null/&lt;=0 = from model.</summary>
    public float YarnBetaFast { get; init; } = 0f;
    /// <summary>YaRN high-freq correction. null/&lt;=0 = from model.</summary>
    public float YarnBetaSlow { get; init; } = 0f;
    /// <summary>YaRN original context. 0 = from model.</summary>
    public uint YarnOrigCtx { get; init; } = 0;
    /// <summary>KV defrag threshold (deprecated upstream). null/&lt;=0 = disabled.</summary>
    public float DefragThold { get; init; } = 0f;
    /// <summary>MoE expert host cache (bytes). 0 = disabled.</summary>
    public uint MoeCacheSize { get; init; } = 0;
    /// <summary>K cache type override: null = mirror the model KV type;
    /// 0..3 = explicit pvx_kv_type_t (F16/Q8_0/Q4_0/Q4_1).</summary>
    public int? KvTypeK { get; init; }
    /// <summary>V cache type override (see KvTypeK).</summary>
    public int? KvTypeV { get; init; }
    /// <summary>Offload KQV ops to GPU: 0 = llama default (true); 1 on; 2 off.</summary>
    public int OffloadKqvTri { get; init; } = 0;
    /// <summary>Offload host ops to device: 0 = llama default (true); 1 on; 2 off.</summary>
    public int OpOffloadTri { get; init; } = 0;
    /// <summary>Full-size SWA cache: 0 = llama default (true); 1 on; 2 off.</summary>
    public int SwaFullTri { get; init; } = 0;
    /// <summary>Unified KV buffer across sequences: 0 = llama default (false); 1 on; 2 off.</summary>
    public int KvUnifiedTri { get; init; } = 0;
    /// <summary>Disable perf timing: 0 = llama default (true); 1 on; 2 off.</summary>
    public int NoPerfTri { get; init; } = 0;
    /// <summary>Extract embeddings at decode (encoder context): false = pre-wave-5.</summary>
    public bool Embeddings { get; init; } = false;
    /// <summary>Attention type: 0 UNSPECIFIED, 1 causal, 2 non-causal.</summary>
    public int AttentionType { get; init; } = 0;
}

/// <summary>Managed enum mirroring PvxPoolingType.</summary>
public enum PoolingType
{
    None = 0, Mean = 1, Cls = 2, Last = 3,
    /// <summary>ABI v8: rerank lane (classification head).</summary>
    Rank = 4,
}
