using ProvostXInference.Interop;
using Xunit;

namespace ProvostXInference.Tests;

/// <summary>
/// Dispatch registry: activation swaps the logical-name route (consulted only
/// during engine activation); non-pvxinference names never resolve.
/// </summary>
public class PvxDispatchRegistryTests : IDisposable
{
    public void Dispose() => PvxDispatchRegistry.ResetForTests();

    private sealed class FakeRouter : IPvxDispatchRouter
    {
        private readonly IntPtr _image;
        public FakeRouter(long handle) => _image = new IntPtr(handle);
        public string EngineId => "fake";
        public IntPtr ImageHandle => _image;
        public IPvxEntryPoints EntryPoints => null!; // registry tests never touch tables
        public IntPtr RouteLibrary(string logicalLibraryName) =>
            logicalLibraryName == "pvxinference" ? _image : IntPtr.Zero;
    }

    [Fact]
    public void WithoutActivation_NoRouteIsAvailable()
    {
        Assert.Null(PvxDispatchRegistry.Current);
        Assert.Throws<InvalidOperationException>(() => _ = PvxDispatchRegistry.ActiveEntryPoints);
    }

    [Fact]
    public void AfterActivation_LogicalNameRoutesOntoActiveImage()
    {
        var router = new FakeRouter(0x5150);
        PvxDispatchRegistry.Activate(router);
        Assert.Same(router, PvxDispatchRegistry.Current);
    }

    [Fact]
    public void Activation_IsAtomicUnderReplace()
    {
        PvxDispatchRegistry.Activate(new FakeRouter(0x1));
        PvxDispatchRegistry.Activate(new FakeRouter(0x2));
        Assert.Equal(0x2, PvxDispatchRegistry.Current!.ImageHandle.ToInt64());
    }
}