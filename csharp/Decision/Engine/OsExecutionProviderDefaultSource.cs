using System.Runtime.InteropServices;

using ProvostXInference.Decision.Abstractions;
using ProvostXInference.Logging;

namespace ProvostXInference.Decision.Engine;

/// <summary>
/// Per-OS default EP chains (LLM-10): macOS coreml→cpu, Windows directml→cpu,
/// Linux cuda→rocm→openvino→cpu — first available wins at session init; cpu is
/// always the terminal fallback. An injected osOverride keeps the mapping
/// testable without touching the real host OS.
/// </summary>
public sealed class OsExecutionProviderDefaultSource : IExecutionProviderDefaultSource
{
    private readonly string? _osOverride;

    /// <param name="osOverride">
    /// Test pin: "macos" | "windows" | "linux" | "other". Null → real runtime OS.
    /// </param>
    public OsExecutionProviderDefaultSource(string? osOverride = null)
    {
        _osOverride = osOverride;
    }

    public ExecutionProviderChain DefaultChain()
    {
        // Unknown EP entries in a default chain are probed at session init and
        // degrade down the chain — Parse only fails on UNKNOWN names, and every
        // name used here is known. Pure construction, never throws.
        var os = _osOverride ?? OsName();
        return os switch
        {
            "macos" => ExecutionProviderChain.Parse("coreml->cpu", "default"),
            "windows" => ExecutionProviderChain.Parse("directml->cpu", "default"),
            "linux" => ExecutionProviderChain.Parse("cuda->rocm->openvino->cpu", "default"),
            _ => ExecutionProviderChain.Parse("cpu", "default"),
        };
    }

    private static string OsName()
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX)) return "macos";
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows)) return "windows";
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux)) return "linux";
        return "other";
    }
}
