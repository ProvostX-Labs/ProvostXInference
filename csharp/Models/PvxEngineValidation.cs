namespace ProvostXInference.Models;

/// <summary>Outcome of engine-selection validation: exactly one of a validated
/// spec or a typed fail-closed error (never both, never a guessed default).</summary>
public sealed record PvxEngineValidation
{
    public PvxEngineSpec? Spec { get; }
    public PvxEngineSelectionError? Error { get; }
    public string? Message { get; }
    public bool IsValid => Error is null;

    private PvxEngineValidation(PvxEngineSpec? spec, PvxEngineSelectionError? error, string? message)
    {
        Spec = spec;
        Error = error;
        Message = message;
    }

    public static PvxEngineValidation Success(PvxEngineSpec spec) => new(spec, null, null);
    public static PvxEngineValidation Fail(PvxEngineSelectionError error, string message) => new(null, error, message);
}