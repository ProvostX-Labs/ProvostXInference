namespace ProvostXInference.Decision.Engine;

/// <summary>
/// Typed fail-closed judge error: every judge endpoint maps this to HTTP 503 with
/// a machine-readable code — NEVER to a 200/allow verdict. Codes:
/// "model_unloaded" (judge engine present but not loaded) and "model_missing"
/// (no judge model configured/registered for the requested id).
/// </summary>
public sealed class JudgeModelNotReadyException : Exception
{
    /// <summary>Machine-readable fail-closed code ("model_unloaded" | "model_missing").</summary>
    public string Code { get; }

    public JudgeModelNotReadyException(string code, string message)
        : base(message)
    {
        if (code != "model_unloaded" && code != "model_missing" && code != "model_invalid")
            throw new ArgumentException($"Unknown fail-closed judge code '{code}'", nameof(code));
        Code = code;
    }
}