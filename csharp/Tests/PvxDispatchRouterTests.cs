using ProvostXInference.Interop;
using Xunit;

namespace ProvostXInference.Tests;

/// <summary>
/// Per-handle dispatch (BL-02 stage 1): two routers on DISTINCT fake images
/// build THEIR OWN function tables — table construction binds each export
/// from the OWNING image (fake export addresses encode image provenance),
/// and per-handle routing keeps the two sets independent. Invocation of the
/// bound pointers is deliberately NOT exercised here (fake addresses are not
/// executable); call-path correctness is proven by the real engine in
/// integration, and table construction/provenance here.
/// </summary>
public class PvxDispatchRouterTests
{
    [Fact]
    public void TwoRouters_DistinctImages_BuildIndependentTables()
    {
        var loader = new FakeModuleLoader();
        loader.AddImage("A", loadPath: "fake-a");
        loader.AddImage("B", loadPath: "fake-b");

        var routerA = PvxDispatchRouter.Load("metal", "fake-a", loader);
        var routerB = PvxDispatchRouter.Load("vulkan", "fake-b", loader);

        Assert.NotEqual(routerA.ImageHandle, routerB.ImageHandle);

        // Each router's tables resolved exports from ITS OWN image
        // (provenance recorded by the fake loader at bind time).
        Assert.Equal("A", loader.LastExportImage(routerA.ImageHandle));
        Assert.Equal("B", loader.LastExportImage(routerB.ImageHandle));

        // Tables are distinct objects — no shared dispatch state.
        Assert.NotSame(routerA.EntryPoints, routerB.EntryPoints);
    }

    [Fact]
    public void RouteLibrary_RoutesLogicalNameOntoOwnImage_AndRejectsForeignNames()
    {
        var loader = new FakeModuleLoader();
        var image = loader.AddImage("X", loadPath: "fake-x");
        var router = PvxDispatchRouter.Load("metal", "fake-x", loader);

        Assert.Equal(image.Handle, router.RouteLibrary("pvxinference"));
        Assert.Equal(IntPtr.Zero, router.RouteLibrary("otherlib"));
        Assert.Equal(IntPtr.Zero, router.RouteLibrary("libc"));
    }

    [Fact]
    public void Load_AbiMismatch_FailsClosed_AndFreesImage()
    {
        var loader = new FakeModuleLoader();
        var image = loader.AddImage("BAD", abiVersion: 99, loadPath: "fake-bad");
        var ex = Assert.Throws<DllNotFoundException>(
            () => PvxDispatchRouter.Load("metal", "fake-bad", loader));
        Assert.Contains("ABI mismatch", ex.Message);
        Assert.True(loader.WasFreed(image.Handle), "failed router load must free the image");
    }

    [Fact]
    public void Load_MissingAbiExport_FailsClosed_AndFreesImage()
    {
        var loader = new FakeModuleLoader();
        var image = loader.AddImage("NOABI", hasAbiVersion: false, loadPath: "fake-noabi");
        var ex = Assert.Throws<EntryPointNotFoundException>(
            () => PvxDispatchRouter.Load("metal", "fake-noabi", loader));
        Assert.Contains("pvx_abi_version", ex.Message);
        Assert.True(loader.WasFreed(image.Handle));
    }

    [Fact]
    public void Load_AllExportsBound_FromOwnImage()
    {
        // Table construction binds EVERY export from the owning image —
        // count the provenance records: > 1 export bound, all from "X".
        var loader = new FakeModuleLoader();
        var image = loader.AddImage("X", loadPath: "fake-x");
        var router = PvxDispatchRouter.Load("metal", "fake-x", loader);
        Assert.NotNull(router.EntryPoints);
        var bound = loader.ExportProvenance(image.Handle);
        Assert.True(bound.Count > 80, $"expected the full ABI table bound, got {bound.Count}");
        Assert.All(bound, id => Assert.Equal("X", id));
    }
}