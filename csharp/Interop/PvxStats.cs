using System.Runtime.InteropServices;

namespace ProvostXInference.Interop;

/// <summary>
/// Managed mirror of pvx_stats_t (ABI v8, wave 3 feature 12). Blittable
/// copy-out snapshot of a lane's additive counters (spec telemetry mirrors
/// pvx_*_spec_stats: 0/0 when no session is active).
/// </summary>
[StructLayout(LayoutKind.Sequential)]
internal struct PvxStats
{
    public long NTokens;
    public long NKvPos;
    public long NPromptTokens;
    public long NDecodeSteps;
    public long SpecDrafted;
    public long SpecAccepted;
}
