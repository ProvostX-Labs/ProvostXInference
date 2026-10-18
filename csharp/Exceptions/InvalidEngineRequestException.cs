namespace ProvostXInference.Exceptions;

/// <summary>
/// Fail-closed typed error for an invalid engine contract (bad/missing path,
/// out-of-range knob, wrong-kind field). Thrown BEFORE any backend is touched.
/// </summary>
public sealed class InvalidEngineRequestException : InferenceException
{
    public InvalidEngineRequestException(string message) : base(-1, message) { }
}