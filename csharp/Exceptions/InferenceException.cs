using System.Runtime.InteropServices;

namespace ProvostXInference.Exceptions;

/// <summary>Thrown when a native ProvostXInference call returns an error code.
/// Base of the typed-error family (engine contract, LDC audit 2026-10-17).</summary>
public class InferenceException : Exception
{
    public int ErrorCode { get; }

    internal InferenceException(int code, string message) : base(message)
    {
        ErrorCode = code;
    }

    /// <summary>Derived typed-error ctor (engine contract family).</summary>
    protected InferenceException(int code, string message, bool derived) : base(message)
    {
        ErrorCode = code;
    }
}