using System.Runtime.InteropServices;

namespace ProvostXInference.Interop;

/// <summary>Blittable mirror of pvx_context_params_t.</summary>
[StructLayout(LayoutKind.Sequential)]
internal readonly struct PvxContextParams
{
    public readonly uint ContextSize;
    public readonly uint BatchSize;
    public readonly uint SeqMax;
    public readonly PvxPoolingType PoolingType;
    /// <summary>ABI v8 wave-4: recurrent-state snapshot ring per sequence
    /// (0 = llama default). MUST mirror the native 5th field — the marshal
    /// size is pinned by PvxContextParamsMarshalTests.</summary>
    public readonly uint NRsSeq;

    // ── ABI v8 wave-5 (BL-21c) appended fields — zeroed = pre-wave-5 ──
    public readonly uint UbatchSize;             // 0 = mirror batch_size
    public readonly uint NOutputsMax;            // 0 = llama default (= n_batch)
    public readonly uint NOutputsMaxPerSeq;      // 0 = llama default
    public readonly int ThreadsBatch;            // 0 = llama default (GGML auto)
    public readonly int RopeScalingType;         // 0 UNSPEC, 1 NONE, 2 LINEAR, 3 YARN
    public readonly float RopeFreqBase;          // 0 = from model
    public readonly float RopeFreqScale;         // 0 = from model
    public readonly float YarnExtFactor;          // <=0 = from model (-1)
    public readonly float YarnAttnFactor;         // <=0 = from model (-1)
    public readonly float YarnBetaFast;           // <=0 = from model (-1)
    public readonly float YarnBetaSlow;           // <=0 = from model (-1)
    public readonly uint YarnOrigCtx;             // 0 = from model
    public readonly float DefragThold;            // <=0 = disabled (default)
    public readonly uint MoeCacheSize;            // bytes; 0 = disabled (default)
    public readonly int KvTypeK;                 // -1 = mirror model KV type; 0..3 explicit
    public readonly int KvTypeV;                 // -1 = mirror model KV type
    public readonly int OffloadKqvTri;            // 0 = llama default (true); 1 on; 2 off
    public readonly int OpOffloadTri;             // 0 = llama default (true); 1 on; 2 off
    public readonly int SwaFullTri;              // 0 = llama default (true); 1 on; 2 off
    public readonly int KvUnifiedTri;            // 0 = llama default (false); 1 on; 2 off
    public readonly int NoPerfTri;               // 0 = llama default (true); 1 on; 2 off
    public readonly byte Embeddings;              // false = pre-wave-5 (pvx_get_embeddings drives)
    public readonly int AttentionType;           // 0 UNSPEC; 1 causal; 2 non-causal

    public PvxContextParams(uint contextSize, uint batchSize, uint seqMax,
        PvxPoolingType poolingType, uint nRsSeq = 0,
        uint ubatchSize = 0, uint nOutputsMax = 0, uint nOutputsMaxPerSeq = 0,
        int threadsBatch = 0, int ropeScalingType = 0,
        float ropeFreqBase = 0f, float ropeFreqScale = 0f,
        float yarnExtFactor = 0f, float yarnAttnFactor = 0f,
        float yarnBetaFast = 0f, float yarnBetaSlow = 0f, uint yarnOrigCtx = 0,
        float defragThold = 0f, uint moeCacheSize = 0,
        int kvTypeK = -1, int kvTypeV = -1,
        int offloadKqvTri = 0, int opOffloadTri = 0, int swaFullTri = 0,
        int kvUnifiedTri = 0, int noPerfTri = 0, bool embeddings = false,
        int attentionType = 0)
    {
        ContextSize = contextSize;
        BatchSize = batchSize;
        SeqMax = seqMax;
        PoolingType = poolingType;
        NRsSeq = nRsSeq;
        UbatchSize = ubatchSize;
        NOutputsMax = nOutputsMax;
        NOutputsMaxPerSeq = nOutputsMaxPerSeq;
        ThreadsBatch = threadsBatch;
        RopeScalingType = ropeScalingType;
        RopeFreqBase = ropeFreqBase;
        RopeFreqScale = ropeFreqScale;
        YarnExtFactor = yarnExtFactor;
        YarnAttnFactor = yarnAttnFactor;
        YarnBetaFast = yarnBetaFast;
        YarnBetaSlow = yarnBetaSlow;
        YarnOrigCtx = yarnOrigCtx;
        DefragThold = defragThold;
        MoeCacheSize = moeCacheSize;
        KvTypeK = kvTypeK;
        KvTypeV = kvTypeV;
        OffloadKqvTri = offloadKqvTri;
        OpOffloadTri = opOffloadTri;
        SwaFullTri = swaFullTri;
        KvUnifiedTri = kvUnifiedTri;
        NoPerfTri = noPerfTri;
        Embeddings = embeddings ? (byte)1 : (byte)0;
        AttentionType = attentionType;
    }
}