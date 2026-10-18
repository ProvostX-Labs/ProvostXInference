using ProvostXInference.Interop;

namespace ProvostXInference.Models;

/// <summary>
/// Validates the ModelConfig engine field against the 5-product engine matrix:
/// result object over (engine id, OS probe) — unknown ids and wrong-OS
/// selections fail closed with a typed error; no default engine guessing here.
/// Stateless utility — no mutable state.
/// </summary>
public sealed class PvxEngineSelector
{
    // Stateless utility — no mutable state (matrix contract injected by ctor).
    private readonly IPvxEnginePolicy _policy;
    private readonly IOSProbe _os;

    internal PvxEngineSelector(IPvxEnginePolicy policy, IOSProbe os)
    {
        _policy = policy ?? throw new ArgumentNullException(nameof(policy));
        _os = os ?? throw new ArgumentNullException(nameof(os));
    }

    /// <summary>Validate an engine id against this machine's OS.
    /// Success → the resolved spec; failure → the typed error. Empty/whitespace
    /// input is treated as unknown (fail-closed; callers may treat null as
    /// "accept the platform default" BEFORE calling this).</summary>
    public PvxEngineValidation Validate(string? engine)
    {
        var id = engine?.Trim();
        if (string.IsNullOrEmpty(id) || !_policy.IsKnownEngine(id))
            return PvxEngineValidation.Fail(PvxEngineSelectionError.UnknownEngine,
                string.IsNullOrEmpty(id)
                    ? "engine must be one of the known specialists (metal | cuda | vulkan)"
                    : $"unknown engine '{engine}' (known specialists: {string.Join(", ", _policy.KnownEngines)})");

        if (!_policy.IsSupportedOnOs(id, _os))
            return PvxEngineValidation.Fail(PvxEngineSelectionError.EngineNotAvailableOnThisOs,
                id switch
                {
                    "metal" => "the metal specialist is macOS-only — cuda/vulkan are the Windows/Linux specialists",
                    _ => $"the {id} specialist ships on Windows/Linux only — macOS runs the metal specialist"
                });

        return PvxEngineValidation.Success(new PvxEngineSpec
        {
            Engine = id,
            OS = _os.IsMacOS() ? "macos" : _os.IsWindows() ? "windows" : "linux"
        });
    }
}