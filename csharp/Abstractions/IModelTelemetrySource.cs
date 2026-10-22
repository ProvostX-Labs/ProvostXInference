using ProvostXInference.Models;

namespace ProvostXInference.Abstractions;

/// <summary>
/// ABI v9 internal seam (per-engine telemetry, owner order 2026-10-22):
/// implemented by the concrete model; the facade ops view probes for it so
/// the PUBLIC IInferenceModel abstraction stays free of native concerns.
/// </summary>
internal interface IModelTelemetrySource
{
    /// <summary>Model-wide telemetry snapshot (per-engine, never process-wide).</summary>
    EngineTelemetry GetTelemetry();
}
