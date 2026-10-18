using System.Runtime.InteropServices;

namespace ProvostXInference.Interop;

/// <summary>Blittable mirror of pvx_spec_capabilities_t — the engine's
/// self-reported supported-types list (PVX_SPEC_TYPE_* bitset) plus the
/// convenience flags and the ABI v6 spec-vision bit. Field order must match
/// the C struct exactly.</summary>
[StructLayout(LayoutKind.Sequential)]
internal readonly struct PvxSpecCapabilities
{
    public readonly uint Types;
    public readonly byte Mtp;
    public readonly byte Draft;
    /// <summary>ABI v6: spec sessions can draft after multimodal (image)
    /// inputs (embd-aware mirror).</summary>
    public readonly byte SpecVision;

    public PvxSpecCapabilities(uint types, byte mtp, byte draft, byte specVision)
    {
        Types = types;
        Mtp = mtp;
        Draft = draft;
        SpecVision = specVision;
    }
}
