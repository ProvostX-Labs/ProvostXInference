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
    // ── ABI v8 wave-5 (BL-21b) appended load controls — zeroed = llama defaults ──
    public readonly IntPtr KvOverrides;       // pointer to pinned PvxKvOverride array; IntPtr.Zero = none
    public readonly int NKvOverrides;         // <= 0 = none
    public readonly int LazyMode;              // 0 = llama default (AUTO)
    public readonly int UseExtraBuftsTri;      // 0 = llama default (true); 1 on; 2 off
    public readonly byte VocabOnly;            // false = llama default
    public readonly byte CheckTensors;         // false = llama default
    public readonly byte NoHost;                // false = llama default
    public readonly byte NoAlloc;               // false = llama default
    public readonly byte LoadMtp;              // false = llama default (model's own MTP layers stay unloaded)
    // ── ABI v9 (per-engine audit/telemetry, owner order 2026-10-22) — appended;
    // zeroed = audit disabled, bit-identical to pre-v9 loads ──
    public readonly IntPtr AuditPath;          // pinned UTF-8 path; IntPtr.Zero = audit disabled
    public readonly int AuditIncludePrompts;   // 0 = events only; 1 = also record prompt token counts

    public PvxModelParams(IntPtr modelPath, int gpuLayers, int threads, bool flashAttn, PvxKvType kvCacheType,
        IntPtr reasoningMarkers = default, int nReasoningMarkers = 0,
        IntPtr kvOverrides = default, int nKvOverrides = 0,
        int lazyMode = 0, int useExtraBuftsTri = 0,
        bool vocabOnly = false, bool checkTensors = false,
        bool noHost = false, bool noAlloc = false, bool loadMtp = false,
        IntPtr auditPath = default, int auditIncludePrompts = 0)
    {
        ModelPath = modelPath;
        GpuLayers = gpuLayers;
        Threads = threads;
        FlashAttn = flashAttn ? (byte)1 : (byte)0;
        KvCacheType = kvCacheType;
        ReasoningMarkers = reasoningMarkers;
        NReasoningMarkers = nReasoningMarkers;
        KvOverrides = kvOverrides;
        NKvOverrides = nKvOverrides;
        LazyMode = lazyMode;
        UseExtraBuftsTri = useExtraBuftsTri;
        VocabOnly = vocabOnly ? (byte)1 : (byte)0;
        CheckTensors = checkTensors ? (byte)1 : (byte)0;
        NoHost = noHost ? (byte)1 : (byte)0;
        NoAlloc = noAlloc ? (byte)1 : (byte)0;
        LoadMtp = loadMtp ? (byte)1 : (byte)0;
        AuditPath = auditPath;
        AuditIncludePrompts = auditIncludePrompts;
    }
}

/// <summary>ABI v8 wave-5 (BL-21b): blittable pvx_kv_override_t mirror —
/// the GGUF metadata override escape hatch. Explicit fixed layout (no union):
/// Tag: 0 = bool, 1 = int64, 2 = float64, 3 = string.</summary>
[StructLayout(LayoutKind.Sequential)]
internal readonly struct PvxKvOverride
{
    [MarshalAs(UnmanagedType.ByValArray, SizeConst = 128)]
    public readonly byte[] Key;              // ASCII (NUL-terminated)
    public readonly int Tag;
    public readonly long ValI64;
    public readonly double ValF64;
    public readonly byte ValBool;
    [MarshalAs(UnmanagedType.ByValArray, SizeConst = 128)]
    public readonly byte[] ValStr;           // ASCII (NUL-terminated)

    public PvxKvOverride(string key, int tag, long valI64 = 0,
        double valF64 = 0d, bool valBool = false, string? valStr = null)
    {
        var kb = new byte[128];
        var src = System.Text.Encoding.ASCII.GetBytes(key);
        System.Array.Copy(src, kb, System.Math.Min(src.Length, 127));
        var vs = new byte[128];
        if (valStr != null) {
            var vsb = System.Text.Encoding.ASCII.GetBytes(valStr);
            System.Array.Copy(vsb, vs, System.Math.Min(vsb.Length, 127));
        }
        Key = kb;
        Tag = tag;
        ValI64 = valI64;
        ValF64 = valF64;
        ValBool = valBool ? (byte)1 : (byte)0;
        ValStr = vs;
    }
}
