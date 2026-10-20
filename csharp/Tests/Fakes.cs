using ProvostXInference.Interop;
using ProvostXInference.Models;
using Xunit;

namespace ProvostXInference.Tests;

/// <summary>Fake OS probe: scriptable OS identity for matrix tests.
/// Pure test double — no dispatch behavior.</summary>
internal sealed class FakeOsProbe : IOSProbe
{
    private readonly bool _macos;
    private readonly bool _linux;
    private readonly bool _windows;

    public FakeOsProbe(bool macos = false, bool linux = false, bool windows = false, string? architecture = null)
    {
        _macos = macos;
        _linux = linux;
        _windows = windows;
        _architecture = architecture ?? "x64";
    }

    public static FakeOsProbe MacOS() => new(macos: true);
    public static FakeOsProbe Linux(string architecture = "x64") => new(linux: true, architecture: architecture);
    public static FakeOsProbe Windows() => new(windows: true);

    public bool IsMacOS() => _macos;
    public bool IsLinux() => _linux;
    public bool IsWindows() => _windows;

    // BL-43: the RID seam — test doubles override the process architecture.
    public string Architecture() => _architecture;

    private readonly string _architecture;
}

/// <summary>Fake engine policy: records calls and returns scripted answers
/// (default = real matrix semantics so matrix tests exercise the actual rules
/// through the seam).</summary>
internal sealed class FakeEnginePolicy : IPvxEnginePolicy
{
    private readonly PvxEnginePolicy _real = new();

    public IReadOnlyList<string> KnownEngines => _real.KnownEngines;
    public bool IsKnownEngine(string engineId) => _real.IsKnownEngine(engineId);
    public bool IsSupportedOnOs(string engineId, IOSProbe os) => _real.IsSupportedOnOs(engineId, os);
}

/// <summary>Fake file resolver: scripted existence + fixed runtime directory.
/// Records the FileName calls it received (fail-closed pair coverage).</summary>
internal sealed class FakeFileResolver : IPvxEngineFileResolver
{
    private readonly HashSet<string> _existing;
    public List<string> FileNameCalls { get; } = new();

    public FakeFileResolver(IEnumerable<string>? existing = null)
    {
        _existing = existing?.ToHashSet() ?? new HashSet<string>();
        RuntimeDirectory = "/fake/runtimes/fake-rid/native";
    }

    public string RuntimeDirectory { get; }

    public string FileName(string validatedEngine)
    {
        FileNameCalls.Add(validatedEngine);
        return validatedEngine switch
        {
            "metal" => "libpvx-metal.dylib",
            "cuda" => "libpvx-cuda.so",
            "vulkan" => "libpvx-vulkan.so",
            _ => throw new DllNotFoundException($"unknown engine '{validatedEngine}'")
        };
    }

    public bool FileExists(string validatedEngine) => _existing.Contains(validatedEngine);
    public IReadOnlyList<string> AvailableEngines() => _existing.OrderBy(e => e).ToList();
}