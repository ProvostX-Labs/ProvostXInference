using System.Runtime.InteropServices;

namespace ProvostXInference.Interop;

/// <summary>
/// Blittable mirror of pvx_model_params_t. The C string is pinned by the caller
/// as an HGlobal Ansi pointer; ownership of the pointer stays with the caller.
/// bool is marshalled as a 1-byte field to match C99 stdbool.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
internal readonly struct PvxModelParams
{
    public readonly IntPtr ModelPath;
    public readonly int GpuLayers;
    public readonly int Threads;
    public readonly byte FlashAttn;
    public readonly PvxKvType KvCacheType;
    public readonly IntPtr ReasoningMarkers;   // pointer to pinned const char* array; IntPtr.Zero = built-in default
    public readonly int NReasoningMarkers;     // <= 0 = built-in default

    public PvxModelParams(IntPtr modelPath, int gpuLayers, int threads, bool flashAttn, PvxKvType kvCacheType,
        IntPtr reasoningMarkers = default, int nReasoningMarkers = 0)
    {
        ModelPath = modelPath;
        GpuLayers = gpuLayers;
        Threads = threads;
        FlashAttn = flashAttn ? (byte)1 : (byte)0;
        KvCacheType = kvCacheType;
        ReasoningMarkers = reasoningMarkers;
        NReasoningMarkers = nReasoningMarkers;
    }
}
