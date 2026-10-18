namespace ProvostXInference.Models;

/// <summary>Typed failure of engine-selection validation
/// (fail-closed contract — every invalid pair or unknown id resolves to one of
/// these, never to a guessed default).</summary>
public enum PvxEngineSelectionError
{
    /// <summary>The engine id is not one of the known specialists.</summary>
    UnknownEngine = 1,

    /// <summary>The engine id is known but does not ship on this OS
    /// (metal = macOS only; cuda/vulkan = Windows/Linux only).</summary>
    EngineNotAvailableOnThisOs = 2,
}