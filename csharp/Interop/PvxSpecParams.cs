using System.Runtime.InteropServices;
using ProvostXInference.Models;

namespace ProvostXInference.Interop;

/// <summary>Blittable mirror of pvx_spec_params_t. The draft path and the
/// n-gram cache path are pinned by the caller as HGlobal UTF-8 pointers (same
/// rule as PvxModelParams.ModelPath); ownership of the pointers stays with the
/// caller. Field order must match the C struct exactly (ABI v6: the n-gram
/// struct is appended — zero-sized defaults keep v5-shaped configs valid).</summary>
[StructLayout(LayoutKind.Sequential)]
internal readonly struct PvxSpecParams
{
    public readonly int Mode;
    public readonly IntPtr DraftModelPath;
    public readonly int DraftMax;
    public readonly int DraftMin;
    public readonly float PMin;
    public readonly int DraftGpuLayers;
    // pvx_ngram_params_t (appended, ABI v6)
    public readonly int NgramNMin;
    public readonly int NgramNMax;
    public readonly int NgramMinHits;
    public readonly int NgramNMatch;
    public readonly IntPtr NgramCachePath;

    /// <summary>Map a user config onto the native struct. SpeculativeMode values
    /// mirror pvx_spec_mode_t exactly; draftPathPtr/cachePathPtr must be pinned
    /// UTF-8 copies (IntPtr.Zero when null). <=0 / <0 sentinels mirror the
    /// native defaults (upstream draft-simple).</summary>
    public PvxSpecParams(SpeculativeConfig config, IntPtr draftPathPtr, IntPtr cachePathPtr)
    {
        if (config.Mode is not (SpeculativeMode.Off or SpeculativeMode.Mtp or SpeculativeMode.Draft
            or SpeculativeMode.NgramSimple or SpeculativeMode.NgramMapK
            or SpeculativeMode.NgramMod or SpeculativeMode.NgramCache))
            throw new ArgumentException($"unknown speculative mode {config.Mode}");
        Mode = (int) config.Mode;
        DraftModelPath = draftPathPtr;
        DraftMax = config.DraftMax;
        DraftMin = config.DraftMin;
        PMin = config.PMin;
        DraftGpuLayers = config.DraftGpuLayers;
        var ng = config.Ngram;
        NgramNMin = ng?.NMin ?? 0;
        NgramNMax = ng?.NMax ?? 0;
        NgramMinHits = ng?.MinHits ?? 0;
        NgramNMatch = ng?.NMatch ?? 0;
        NgramCachePath = cachePathPtr;
    }
}
