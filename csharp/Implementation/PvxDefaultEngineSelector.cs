using ProvostXInference.Interop;
using ProvostXInference.Models;

namespace ProvostXInference.Implementation;

/// <summary>
/// Platform default engine selection — used when the model config leaves the
/// engine field unset: macOS resolves to the metal specialist; Windows/Linux
/// prefer cuda when its specialist file ships, else vulkan; fail-closed
/// (typed error, no silent fallback) when neither is installed. The OS identity
/// is constructor-injected (same probe family the selector matrix consults —
/// no hidden environment dependence). Stateless utility — no mutable state.
/// </summary>
public sealed class PvxDefaultEngineSelector
{
    // Stateless utility — no mutable state (immutable candidate order, not state).
    private static readonly string[] PreferOrder = { "cuda", "vulkan" };

    private readonly IOSProbe _os;

    internal PvxDefaultEngineSelector(IOSProbe os)
    {
        _os = os ?? throw new ArgumentNullException(nameof(os));
    }

    /// <summary>The default engine id for the injected OS.
    /// Throws fail-closed when no specialist file exists for this OS.</summary>
    internal string DefaultEngine(IPvxEngineFileResolver resolver)
    {
        ArgumentNullException.ThrowIfNull(resolver);
        if (_os.IsMacOS()) return "metal";
        var available = resolver.AvailableEngines();
        foreach (var candidate in PreferOrder)
        {
            if (available.Contains(candidate)) return candidate;
        }
        throw new DllNotFoundException(
            "No inference specialist found (expected pvx-cuda or pvx-vulkan in runtimes/<rid>/native/). " +
            "Reinstall the runtime package for this platform.");
    }
}