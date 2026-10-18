namespace ProvostXInference.Logging;

/// <summary>
/// Minimal logging seam for Inference-side subsystems (decision judge execution,
/// build-on-first-use helpers). The host application injects an adapter over its
/// own logger; the default sink drops everything (non-fatal diagnostics only).
/// </summary>
public interface IInferenceLogger
{
    void Info(string tag, string message);
    void Warn(string tag, string message);
    void Error(string tag, string message);
    void Debug(string tag, string message);
}

/// <summary>Silent logger fallback (diagnostics are non-fatal — when no logger is
/// injected they are dropped rather than crashing the caller).</summary>
public sealed class QuietInferenceLogger : IInferenceLogger
{
    public void Info(string tag, string message) { }
    public void Warn(string tag, string message) { }
    public void Error(string tag, string message) { }
    public void Debug(string tag, string message) { }
}
