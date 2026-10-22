using ProvostXInference.Implementation;
using ProvostXInference.Interop;
using ProvostXInference.Models;
using Xunit;

namespace ProvostXInference.Tests;

/// <summary>
/// Model-config engine field (BL-02 stage 1): ModelConfig selects WHICH
/// specialist runs, validated fail-closed at load time through the activator
/// — wrong-OS selections and unknown engines NEVER reach the filesystem.
/// </summary>
public class ModelEngineSelectionTests
{
    [Theory]
    [InlineData("vulkan", "vulkan-on-macos error")]
    [InlineData("cpu", "unknown engine error")]
    public void InvalidConfigEngine_FailClosed_ThroughActivator(string engine, string _)
    {
        var activator = new StandardEngineActivator(
            policy: new PvxEnginePolicy(),
            os: FakeOsProbe.MacOS(),
            defaults: new PvxDefaultEngineSelector(FakeOsProbe.MacOS()),
            resolver: new FakeFileResolver(existing: new[] { "metal" }),
            loader: new FakeModuleLoader(), // never reached on invalid selection
            files: new FakeFileExistence(exists: true));
        var config = new ModelConfig { Path = "model.gguf", Engine = engine };
        Assert.Throws<DllNotFoundException>(() => activator.Open(config));
    }

    [Fact]
    public void MissingSpecialistFile_FailClosed_WithClearError()
    {
        // Matrix-valid (metal on macOS) but the dylib is absent on disk.
        var activator = new StandardEngineActivator(
            new PvxEnginePolicy(),
            FakeOsProbe.MacOS(),
            new PvxDefaultEngineSelector(FakeOsProbe.MacOS()),
            new FakeFileResolver(existing: Array.Empty<string>()),
            new FakeModuleLoader(),
            new FakeFileExistence(exists: false));
        var config = new ModelConfig { Path = "model.gguf", Engine = "metal" };
        var ex = Assert.Throws<DllNotFoundException>(() => activator.Open(config));
        Assert.Contains("not available", ex.Message);
        Assert.Contains("libpvx-metal.dylib", ex.Message);
    }

    [MacOSOnlyFact]
    public void UnconfiguredEngine_MacOS_DefaultsToMetal()
    {
        var activator = new StandardEngineActivator(
            new PvxEnginePolicy(),
            FakeOsProbe.MacOS(),
            new PvxDefaultEngineSelector(FakeOsProbe.MacOS()),
            new FakeFileResolver(existing: new[] { "metal" }),
            EngineTestSupport.MetalLoader(),
            new FakeFileExistence(exists: true));
        var handle = activator.Open(new ModelConfig { Path = "model.gguf" });
        Assert.Equal("metal", handle.EngineId);
        Assert.NotNull(handle.EntryPoints);
        handle.Dispose();
    }

    [Fact]
    public void UnconfiguredEngine_Linux_UnexpectedlyMissingSpecialist_FailClosed()
    {
        var activator = new StandardEngineActivator(
            new PvxEnginePolicy(),
            FakeOsProbe.Linux(),
            new PvxDefaultEngineSelector(FakeOsProbe.Linux()),
            new FakeFileResolver(existing: Array.Empty<string>()),
            new FakeModuleLoader(),
            new FakeFileExistence(exists: false));
        var ex = Assert.Throws<DllNotFoundException>(
            () => activator.Open(new ModelConfig { Path = "model.gguf" }));
        Assert.Contains("No inference specialist found", ex.Message);
    }

    [MacOSOnlyFact]
    public void LoadedModel_CarriesItsOwnEngineHandle()
    {
        // Two models → two independently constructed handle instances from the
        // SAME factory path (per-handle ownership, no shared/global handle).
        var activator = new StandardEngineActivator(
            new PvxEnginePolicy(),
            FakeOsProbe.MacOS(),
            new PvxDefaultEngineSelector(FakeOsProbe.MacOS()),
            new FakeFileResolver(existing: new[] { "metal" }),
            EngineTestSupport.MetalLoader(),
            new FakeFileExistence(exists: true));
        var first = activator.Open(new ModelConfig { Path = "a.gguf" });
        var second = activator.Open(new ModelConfig { Path = "b.gguf" });
        Assert.NotSame(first, second);
        Assert.Equal("metal", first.EngineId);
        Assert.Equal("metal", second.EngineId);
        first.Dispose();
        second.Dispose();
    }
}

/// <summary>Fake file-existence seam.</summary>
internal sealed class FakeFileExistence : IPvxFileExistence
{
    private readonly bool _exists;
    public FakeFileExistence(bool exists) => _exists = exists;
    public bool Exists(string path) => _exists;
}