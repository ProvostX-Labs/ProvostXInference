namespace ProvostXInference.Exceptions;

/// <summary>
/// Fail-closed typed error for accessing an ops view of the wrong kind
/// (e.g. <c>engine.Decision</c> on a Gguf engine). The facade never hands
/// out a null view — it refuses.
/// </summary>
public sealed class EngineKindMismatchException : InferenceException
{
    public EngineKindMismatchException(string message) : base(-2, message) { }
}