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

    public PvxContextParams(uint contextSize, uint batchSize, uint seqMax,
        PvxPoolingType poolingType, uint nRsSeq = 0)
    {
        ContextSize = contextSize;
        BatchSize = batchSize;
        SeqMax = seqMax;
        PoolingType = poolingType;
        NRsSeq = nRsSeq;
    }
}