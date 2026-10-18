namespace ProvostXInference.Abstractions;

/// <summary>Runtime suspension of the speculative helper (draft/ngram) on the
/// owning executor stack — the real native wiring for the stall-safe prefill
/// ladder's DraftDisabledRetry rung (INF BL-33b). Implementations must be
/// idempotent (suspending with no active session is a no-op, mirroring the
/// native pvx_executor_spec_free contract) and must not throw when the
/// helper is already disabled.</summary>
public interface ISpeculationSuspender
{
    /// <summary>Disable the speculative session at runtime (native flag
    /// teardown: pvx_executor_spec_init with PVX_SPEC_OFF / spec_free).
    /// Idempotent.</summary>
    void Suspend();
}