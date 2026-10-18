namespace ProvostXInference.Interop;

/// <summary>
/// The 5-product specialist matrix as data: which engine ids exist and which
/// OS they are valid on. metal = macOS only; cuda/vulkan = Windows and Linux.
/// Fail-closed by construction — nothing outside the matrix validates.
/// </summary>
internal interface IPvxEnginePolicy
{
    /// <summary>The known engine ids ("metal", "cuda", "vulkan").</summary>
    IReadOnlyList<string> KnownEngines { get; }

    /// <summary>Whether the engine id is one of the known specialists.</summary>
    bool IsKnownEngine(string engineId);

    /// <summary>Whether the (engine, OS) pair is valid — failing closed on
    /// wrong-OS selections (e.g. metal on Linux, vulkan on macOS).</summary>
    bool IsSupportedOnOs(string engineId, IOSProbe os);
}