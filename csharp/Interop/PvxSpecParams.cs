using System.Runtime.InteropServices;
using ProvostXInference.Models;

namespace ProvostXInference.Interop;

/// <summary>Blittable mirror of pvx_spec_params_t. The draft path is pinned by
/// the caller as an HGlobal UTF-8 pointer (same rule as PvxModelParams.ModelPath);
/// ownership of the pointer stays with the caller. Field order must match the
/// C struct exactly.</summary>
[StructLayout(LayoutKind.Sequential)]
internal readonly struct PvxSpecParams
{
    public readonly int Mode;
    public readonly IntPtr DraftModelPath;
    public readonly int DraftMax;
    public readonly int DraftMin;
    public readonly float PMin;
    public readonly int DraftGpuLayers;

    /// <summary>Map a user config onto the native struct. SpeculativeMode values
    /// mirror pvx_spec_mode_t exactly; draftPathPtr must be a pinned UTF-8 copy
    /// of config.DraftModelPath (IntPtr.Zero when null). <=0 / <0 sentinels
    /// mirror the native defaults (upstream draft-simple).</summary>
    public PvxSpecParams(SpeculativeConfig config, IntPtr draftPathPtr)
    {
        if (config.Mode is not (SpeculativeMode.Off or SpeculativeMode.Mtp or SpeculativeMode.Draft))
            throw new ArgumentException($"unknown speculative mode {config.Mode}");
        Mode = (int) config.Mode;
        DraftModelPath = draftPathPtr;
        DraftMax = config.DraftMax;
        DraftMin = config.DraftMin;
        PMin = config.PMin;
        DraftGpuLayers = config.DraftGpuLayers;
    }
}
