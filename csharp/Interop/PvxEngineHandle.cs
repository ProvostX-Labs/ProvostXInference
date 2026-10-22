using System.IO;
using ProvostXInference.Models;

namespace ProvostXInference.Interop;

/// <summary>
/// Selects + loads the specialist named by the model config and owns its
/// per-handle dispatch route. Each instance = one engine handle (one per
/// loaded model); every resource created through
/// <see cref="IPvxEngineHandle.EntryPoints"/> is dispatched onto THAT engine's
/// own image — concurrent different-specialist models in one process are
/// collision-free (BL-01: per-image registries/symbols). OS matrix compliance
/// lives in PvxEngineSelector (fail-closed, Models layer); specialist file
/// names live in PvxEngineFileResolver (ONE place); ABI validation lives in
/// PvxDispatchRouter.
/// </summary>
internal interface IPvxEngineHandle : IDisposable
{
    /// <summary>Validated engine id of the loaded specialist.</summary>
    string EngineId { get; }

    /// <summary>Raw native image handle of the loaded specialist module.</summary>
    IntPtr ImageHandle { get; }

    /// <summary>Per-handle function tables onto this engine's image
    /// (the constructor-injected dispatch seam for everything downstream).</summary>
    IPvxEntryPoints EntryPoints { get; }
}

/// <summary>Implementation of the per-model engine handle.</summary>
internal sealed class PvxEngineHandle : IPvxEngineHandle
{
    private readonly PvxDispatchRouter _router;
    private bool _disposed;

    private PvxEngineHandle(PvxDispatchRouter router)
    {
        _router = router;
    }

    public string EngineId => _router.EngineId;
    public IntPtr ImageHandle => _router.ImageHandle;
    public IPvxEntryPoints EntryPoints => _router.EntryPoints;

    /// <summary>Spec → file name → loaded router: the single construction path
    /// (fail-closed; the selector matrix has already validated the (engine, OS)
    /// pair before any load — unknown engines/wrong-OS pairs never reach here).</summary>
    internal static IPvxEngineHandle Open(PvxEngineSpec spec, IPvxEngineFileResolver resolver, IPvxModuleLoader loader, IPvxFileExistence fileExistence)
    {
        var path = Path.Combine(resolver.RuntimeDirectory, resolver.EffectiveFileName(spec.Engine));
        if (!fileExistence.Exists(path))
            throw new DllNotFoundException(
                $"inference specialist '{spec.Engine}' is not available on this machine " +
                $"(missing {Path.GetFileName(path)} in runtimes/<rid>/native/; available: {string.Join(", ", resolver.AvailableEngines())}).");
        return new PvxEngineHandle(PvxDispatchRouter.Load(spec.Engine, path, loader));
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
    }
}