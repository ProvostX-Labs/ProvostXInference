using ProvostXInference.Abstractions;

namespace ProvostXInference.Implementation;

/// <summary>Runtime draft/ngram suspension backed by the executor's native
/// speculative-session teardown (pvx_executor_spec_free; INF BL-33b).
/// Idempotent by construction — the native teardown is already idempotent and
/// the suspension is recorded so a second Suspend is a managed no-op.</summary>
public sealed class ExecutorSpeculationSuspender : ISpeculationSuspender
{
    private readonly IStandardExecutor _executor;
    private bool _suspended;

    public ExecutorSpeculationSuspender(IStandardExecutor executor)
    {
        _executor = executor ?? throw new ArgumentNullException(nameof(executor));
    }

    public void Suspend()
    {
        if (_suspended) return;
        _executor.DisableSpeculation(); // native spec_free — idempotent, no error path
        _suspended = true;
    }
}