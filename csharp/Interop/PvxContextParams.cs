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

    public PvxContextParams(uint contextSize, uint batchSize, uint seqMax, PvxPoolingType poolingType)
    {
        ContextSize = contextSize;
        BatchSize = batchSize;
        SeqMax = seqMax;
        PoolingType = poolingType;
    }
}
