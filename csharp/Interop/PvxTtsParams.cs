using System.Runtime.InteropServices;

namespace ProvostXInference.Interop;

/// <summary>
/// Managed mirror of pvx_tts_params_t (ABI v8, wave 3 feature 7). Blittable —
/// in-parameter marshalling. Zero values select the per-model upstream
/// defaults (mtmd_gen_inp_default); seed == uint.MaxValue = random draw.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
internal struct PvxTtsParams
{
    public IntPtr Lang;   // UTF-8 string pointer when set, else Zero
    public int TopK;
    public float TopP;
    public float Temp;
    public uint Seed;
    public int MaxFrames;
}
