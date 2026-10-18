namespace ProvostXInference.Interop;

/// <summary>Immutable implementation of the specialist matrix. The owner design
/// (2026-10-06): exactly five specialists — Windows Vulkan, Windows CUDA,
/// Linux Vulkan, Linux CUDA, Apple Metal — no mixing, no generic fallback.
/// Stateless utility — no mutable state.</summary>
internal sealed class PvxEnginePolicy : IPvxEnginePolicy
{
    public IReadOnlyList<string> KnownEngines { get; } = new[] { "metal", "cuda", "vulkan" };

    public bool IsKnownEngine(string engineId) => KnownEngines.Contains(engineId);

    public bool IsSupportedOnOs(string engineId, IOSProbe os)
    {
        if (!IsKnownEngine(engineId)) return false;
        return engineId switch
        {
            "metal" => os.IsMacOS(),
            "cuda" => os.IsWindows() || os.IsLinux(),
            "vulkan" => os.IsWindows() || os.IsLinux(),
            _ => false
        };
    }
}