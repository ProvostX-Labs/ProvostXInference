using ProvostXInference.Models;

namespace ProvostXInference.Interop;

/// <summary>
/// One loaded specialist engine and its per-handle dispatch route: owns the
/// native image (load → validate → per-handle tables → dispatch registration),
/// hands its OWN function tables to every resource it creates, and routes the
/// logical "pvxinference" DllImport name onto itself during activation.
/// Two routers (two specialists, even two copies of one specialist) coexist
/// collision-free — ggml registry/symbol state is per-image (BL-01 spike GREEN).
/// </summary>
internal sealed class PvxDispatchRouter : IPvxDispatchRouter
{
    private const string LogicalLibraryName = "pvxinference";

    public string EngineId { get; }
    public IntPtr ImageHandle { get; }
    public IPvxEntryPoints EntryPoints { get; }

    private PvxDispatchRouter(string engineId, IntPtr imageHandle, IPvxEntryPoints entryPoints)
    {
        EngineId = engineId;
        ImageHandle = imageHandle;
        EntryPoints = entryPoints;
    }

    /// <summary>Load + validate a specialist image and build its per-handle
    /// tables. Throws fail-closed on load failure, missing pvx_abi_version,
    /// or ABI mismatch (expected = PvxNative.ExpectedAbiVersion, in lockstep
    /// with include/pvxinference.h).</summary>
    internal static PvxDispatchRouter Load(string engineId, string specialistPath, IPvxModuleLoader loader)
    {
        var image = loader.Load(specialistPath);
        if (image == IntPtr.Zero)
            throw new DllNotFoundException($"failed to load inference specialist '{engineId}' ({specialistPath}).");
        try
        {
            var entryPoints = new PvxHandleEntryPoints(image, loader);
            var actual = entryPoints.AbiVersion();
            const int expected = PvxNative.ExpectedAbiVersion;
            if (actual != expected)
                throw new DllNotFoundException(
                    $"inference specialist '{engineId}' ABI mismatch: image reports {actual}, binding expects {expected}.");
            return new PvxDispatchRouter(engineId, image, entryPoints);
        }
        catch
        {
            loader.Free(image);
            throw;
        }
    }

    /// <summary>RouteLibrary routes the logical library name onto this image (any other name
    /// is NOT ours — resolution fails over to the runtime's default probing).</summary>
    public IntPtr RouteLibrary(string logicalLibraryName) =>
        logicalLibraryName == LogicalLibraryName ? ImageHandle : IntPtr.Zero;
}