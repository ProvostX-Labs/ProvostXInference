namespace ProvostXInference.Interop;

/// <summary>Immutable implementation of the specialist matrix. The owner design
/// (2026-10-06, extended 2026-10-22 owner order): Windows Vulkan, Windows
/// CUDA, Windows CPU, Linux Vulkan, Linux CUDA, Linux CPU, Apple Metal —
/// no mixing, no generic fallback.
/// Stateless utility — no mutable state.</summary>
internal sealed class PvxEnginePolicy : IPvxEnginePolicy
{
    public IReadOnlyList<string> KnownEngines { get; } = new[] { "metal", "cuda", "vulkan", "cpu" };

    public bool IsKnownEngine(string engineId) => KnownEngines.Contains(engineId);

    public bool IsSupportedOnOs(string engineId, IOSProbe os)
    {
        if (!IsKnownEngine(engineId)) return false;
        return engineId switch
        {
            "metal" => os.IsMacOS(),
            "cuda" => os.IsWindows() || os.IsLinux(),
            "vulkan" => os.IsWindows() || os.IsLinux(),
            "cpu" => os.IsWindows() || os.IsLinux(),
            _ => false
        };
    }
}