using System.Reflection;
using System.Runtime.InteropServices;

namespace ProvostXInference.Interop;

/// <summary>
/// Routes the logical "pvxinference" DllImport name onto the ACTIVE specialist
/// image. One resolver delegate is registered on the binding assembly for the
/// whole process (the runtime allows exactly one per assembly — verified by the
/// BL-01 spike, evidence E3); per-handle dispatch happens through each engine's
/// own function tables, so this registry is consulted ONLY during engine
/// activation (late-bound DllImports such as the ABI handshake), never per call.
/// Sanctioned interop state (same class of process-lifetime exception as
/// NativeLogBridge's pinned log callback).
/// </summary>
internal static class PvxDispatchRegistry
{
    private const string LogicalName = "pvxinference";

    private static readonly object Gate = new();
    private static IPvxDispatchRouter? _active;
    private static bool _resolverRegistered;

    /// <summary>The currently active specialist (null = legacy default probe).</summary>
    internal static IPvxDispatchRouter? Current => _active;

    /// <summary>Convenience view of the active engine's tables (used only by
    /// process-wide APIs that have no owning resource — e.g. backend queries).</summary>
    internal static IPvxEntryPoints ActiveEntryPoints =>
        _active?.EntryPoints ?? throw new InvalidOperationException(
            "no inference engine is active — load a model first (per-handle dispatch is model-scoped)");

    /// <summary>Register the process-wide trampoline once, then make the router
    /// active. Registration must precede any engine-specific P/Invoke on the
    /// binding assembly (callers activate before the ABI handshake).</summary>
    internal static void Activate(IPvxDispatchRouter router)
    {
        lock (Gate)
        {
            if (!_resolverRegistered)
            {
                NativeLibrary.SetDllImportResolver(typeof(PvxNative).Assembly, ResolverTrampoline);
                _resolverRegistered = true;
            }
            _active = router;
        }
    }

    /// <summary>Trampoline: routes the logical name to the active image, all
    /// other names fail resolution (default probe never silently wins).</summary>
    private static IntPtr ResolverTrampoline(string libraryName, Assembly assembly, DllImportSearchPath? searchPath)
    {
        if (libraryName == LogicalName)
        {
            var router = _active;
            if (router is not null)
                return router.RouteLibrary(libraryName);
        }
        return IntPtr.Zero;
    }

    /// <summary>Test seam: drop the active router (the once-per-process resolver
    /// registration cannot be undone — .NET allows exactly one per assembly).</summary>
    internal static void ResetForTests()
    {
        lock (Gate)
        {
            _active = null;
            _resolverRegistered = true;
        }
    }
}