using System.Runtime.InteropServices;

namespace ProvostXInference.Interop;

/// <summary>
/// Managed mirror of pvx_engine_telemetry_t (ABI v9, per-engine telemetry,
/// owner order 2026-10-22). Blittable copy-out snapshot of MODEL-aggregate
/// counters (roll-up of all executor/conversation lanes on this engine).
/// Rates (tok/s etc.) are derived by the caller between two snapshots — the
/// native side stays stateless.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
internal struct PvxEngineTelemetry
{
    public long NPromptTokens;
    public long NDecodeSteps;
    public long NCommittedTokens;
    public long SpecDrafted;
    public long SpecAccepted;
    public long NErrors;
    public long NContextsAlive;
}
