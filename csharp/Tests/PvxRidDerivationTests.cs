using ProvostXInference.Models;
using ProvostXInference.Interop;
using Xunit;

namespace ProvostXInference.Tests;

/// <summary>BL-43: the native-dir RID follows the process architecture on
/// Linux (linux-arm64 vs linux-x64) — was hardcoded linux-x64 for all Linux.
/// macOS/Windows RIDs are unchanged.</summary>
[Trait("Tier", "Unit")]
public sealed class PvxRidDerivationTests
{
    private static PvxEngineFileResolver Resolver(IOSProbe os) => new(os);

    [Fact]
    public void Linux_Arm64_Host_Derives_Linux_Arm64_Rid()
    {
        var dir = Resolver(new FakeOsProbe(linux: true, architecture: "arm64")).RuntimeDirectory;
        Assert.EndsWith(Path.Combine("runtimes", "linux-arm64", "native"), dir);
    }

    [Fact]
    public void Linux_X64_Host_Derives_Linux_X64_Rid()
    {
        var dir = Resolver(new FakeOsProbe(linux: true, architecture: "x64")).RuntimeDirectory;
        Assert.EndsWith(Path.Combine("runtimes", "linux-x64", "native"), dir);
    }

    [Fact]
    public void Mac_And_Windows_Rids_Are_Unchanged()
    {
        Assert.EndsWith(Path.Combine("runtimes", "osx-arm64", "native"),
            Resolver(new FakeOsProbe(macos: true)).RuntimeDirectory);
        Assert.EndsWith(Path.Combine("runtimes", "win-x64", "native"),
            Resolver(new FakeOsProbe(windows: true)).RuntimeDirectory);
    }

    [Fact]
    public void SystemOsProbe_Architecture_Matches_Host()
    {
        // The real probe reads RuntimeInformation.ProcessArchitecture — on an
        // arm64 dev machine this is "arm64", on x64 "x64". Never empty.
        var probe = new SystemOsProbe();
        Assert.Contains(probe.Architecture(), new[] { "arm64", "x64", "arm", "x86" });
    }
}