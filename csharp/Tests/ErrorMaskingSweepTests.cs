using System.Runtime.InteropServices;
using ProvostXInference.Interop;
using ProvostXInference.Models;
using Xunit;

namespace ProvostXInference.Tests;

/// <summary>
/// BL-03 error-masking sweep: prove the missing-dylib / missing-export error
/// paths surface typed exceptions carrying the real cause (file name, export
/// name) instead of a masked generic failure.
/// </summary>
public class ErrorMaskingSweepTests
{
    /// <summary>Loader seam double that simulates the OS failing to load the
    /// dylib file (missing file → DllNotFoundException naming the path).</summary>
    private sealed class ThrowingModuleLoader : IPvxModuleLoader
    {
        public IntPtr Load(string path) =>
            throw new DllNotFoundException($"native library load failed: {path} (dlopen: image not found)");
        public IntPtr GetExport(IntPtr image, string exportName) =>
            throw new EntryPointNotFoundException($"native export not found: {exportName}");
        public bool Free(IntPtr image) => true;
    }

    [Fact]
    public void RouterLoad_MissingDylib_TypedErrorNamesEngineAndPath()
    {
        var ex = Assert.Throws<DllNotFoundException>(
            () => PvxDispatchRouter.Load("metal", "/fake/runtimes/osx-arm64/native/libpvx-metal.dylib", new ThrowingModuleLoader()));
        // The loader seam exception propagates UNCHANGED — the router adds no
        // masking layer; the OS-level path detail survives to the caller.
        Assert.Contains("/fake/runtimes/osx-arm64/native/libpvx-metal.dylib", ex.Message);
        Assert.Contains("dlopen: image not found", ex.Message);
    }

    [Fact]
    public void EngineHandleOpen_MissingFile_TypedErrorNamesFile()
    {
        var spec = new PvxEngineSpec { Engine = "metal", OS = "macos" };
        var resolver = new FakeFileResolver(existing: Array.Empty<string>());
        var ex = Assert.Throws<DllNotFoundException>(
            () => PvxEngineHandle.Open(spec, resolver, new FakeModuleLoader(), new FakeFileExistence(exists: false)));
        Assert.Contains("libpvx-metal.dylib", ex.Message);
        Assert.Contains("metal", ex.Message);
    }

    [Fact]
    public void RouterLoad_MissingExport_TypedErrorNamesExport()
    {
        // Image loads but pvx_abi_version is absent → IntPtr.Zero per the
        // loader seam normalization → table builder must name the export.
        var loader = new FakeModuleLoader();
        var image = loader.AddImage("no-abi", hasAbiVersion: false);
        var ex = Assert.Throws<EntryPointNotFoundException>(
            () => PvxDispatchRouter.Load("metal", "fake-no-abi", loader));
        Assert.Contains("pvx_abi_version", ex.Message);
        Assert.Equal(image.Id, loader.LastExportImage(image.Handle)); // provenance: image was probed
    }
}