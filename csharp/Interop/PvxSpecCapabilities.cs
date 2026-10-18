using System.Runtime.InteropServices;

namespace ProvostXInference.Interop;

/// <summary>Blittable mirror of pvx_spec_capabilities_t — the engine's
/// self-reported supported-types list (PVX_SPEC_TYPE_* bitset) plus the two
/// convenience flags. Field order must match the C struct exactly.</summary>
[StructLayout(LayoutKind.Sequential)]
internal readonly struct PvxSpecCapabilities
{
    public readonly uint Types;
    public readonly byte Mtp;
    public readonly byte Draft;

    public PvxSpecCapabilities(uint types, byte mtp, byte draft)
    {
        Types = types;
        Mtp = mtp;
        Draft = draft;
    }
}
