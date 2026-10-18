namespace ProvostXInference.Models;

/// <summary>
/// A selected per-model engine: a validated (engine, OS) specialization from
/// the 5-product matrix — the file-name contract lives exclusively in the
/// Interop policy/resolver layer (names in ONE place). The ModelConfig engine
/// field selects WHICH specialist dylib runs for this model; the resolver
/// rejects unknown engines and wrong-OS selections fail-closed.
/// </summary>
public sealed record PvxEngineSpec
{
    /// <summary>Engine id — one of the specialist ids in the engine matrix.</summary>
    public required string Engine { get; init; }

    /// <summary>OS family the specialist targets ("macos" | "linux" | "windows").</summary>
    public required string OS { get; init; }
}