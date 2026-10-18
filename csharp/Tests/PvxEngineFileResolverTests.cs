using ProvostXInference.Interop;
using ProvostXInference.Models;
using Xunit;

namespace ProvostXInference.Tests;

/// <summary>File-name resolver (names live in ONE place): per-OS specialist
/// file names, wrong-OS pairs throw fail-closed, existence checks never
/// fall back to another engine's file.</summary>
public class PvxEngineFileResolverTests
{
    [Theory]
    [InlineData("metal", "libpvx-metal.dylib")]
    public void MetalFileName_OnMacOS(string engine, string expected)
    {
        var resolver = new PvxEngineFileResolver(FakeOsProbe.MacOS());
        Assert.Equal(expected, resolver.FileName(engine));
    }

    [Theory]
    [InlineData("cuda", "pvx-cuda.dll")]
    [InlineData("vulkan", "pvx-vulkan.dll")]
    public void Cuda_VulkanFileNames_OnWindows(string engine, string expected)
    {
        var resolver = new PvxEngineFileResolver(FakeOsProbe.Windows());
        Assert.Equal(expected, resolver.FileName(engine));
    }

    [Theory]
    [InlineData("cuda", "libpvx-cuda.so")]
    [InlineData("vulkan", "libpvx-vulkan.so")]
    public void Cuda_VulkanFileNames_OnLinux(string engine, string expected)
    {
        var resolver = new PvxEngineFileResolver(FakeOsProbe.Linux());
        Assert.Equal(expected, resolver.FileName(engine));
    }

    [Theory]
    [InlineData("vulkan")]
    [InlineData("cuda")]
    public void Cuda_VulkanFileNames_OnMacOS_Throw_FailClosed(string engine)
    {
        var resolver = new PvxEngineFileResolver(FakeOsProbe.MacOS());
        Assert.Throws<DllNotFoundException>(() => resolver.FileName(engine));
        Assert.False(resolver.FileExists(engine));
    }

    [Fact]
    public void MetalFileName_OnLinux_Throws_FailClosed()
    {
        var resolver = new PvxEngineFileResolver(FakeOsProbe.Linux());
        Assert.Throws<DllNotFoundException>(() => resolver.FileName("metal"));
        Assert.False(resolver.FileExists("metal"));
    }

    [Fact]
    public void MetalFileName_OnWindows_Throws_FailClosed()
    {
        var resolver = new PvxEngineFileResolver(FakeOsProbe.Windows());
        Assert.Throws<DllNotFoundException>(() => resolver.FileName("metal"));
        Assert.False(resolver.FileExists("metal"));
    }

    [Fact]
    public void AvailableEngines_MapsOnlyKnownIdsWithExistingFiles()
    {
        // The availability list is matrix-driven: known ids whose specialist
        // file exists — never the filesystem's raw directory listing.
        var resolver = new ScriptedResolver(FakeOsProbe.Linux(), onDisk: new[] { "libpvx-cuda.so", "something-unrelated.so" });
        Assert.Equal(new[] { "cuda" }, resolver.AvailableEngines());
    }

    [Fact]
    public void AvailableEngines_EmptyWhenNoSpecialistsOnDisk()
    {
        var resolver = new ScriptedResolver(FakeOsProbe.Linux(), onDisk: Array.Empty<string>());
        Assert.Empty(resolver.AvailableEngines());
    }

    /// <summary>Scriptable-filesystem resolver over the injected OS probe:
    /// file names come from the real matrix mapping, existence from a script —
    /// no real filesystem involved.</summary>
    private sealed class ScriptedResolver(IOSProbe os, string[] onDisk) : IPvxEngineFileResolver
    {
        public string FileName(string validatedEngine) =>
            new PvxEngineFileResolver(os).FileName(validatedEngine);

        public bool FileExists(string validatedEngine)
        {
            // Real mapping check, but existence is scripted per FILE NAME.
            string fileName;
            try { fileName = new PvxEngineFileResolver(os).FileName(validatedEngine); }
            catch (DllNotFoundException) { return false; }
            return onDisk.Contains(fileName);
        }

        public IReadOnlyList<string> AvailableEngines() =>
            new[] { "metal", "cuda", "vulkan" }.Where(FileExists).ToList();

        public string RuntimeDirectory => new PvxEngineFileResolver(os).RuntimeDirectory;
    }
}