using ProvostXInference.Interop;
using ProvostXInference.Models;

namespace ProvostXInference.Implementation;

/// <summary>
/// Composition root of the per-handle engine pipeline: validates the model
/// config's engine selection against the OS matrix (fail-closed), defaults
/// unconfigured selections to the platform specialist, opens the specialist
/// module, and registers its dispatch route for the logical DllImport name.
/// Stateless utility — no mutable state (immutable contracts injected by ctor;
/// the Default instance is the sanctioned immutable composition point).
/// </summary>
internal sealed class StandardEngineActivator : IPvxEngineActivator
{
    /// <summary>Immutable composition of the system-backed pipeline.</summary>
    public static readonly StandardEngineActivator Default = new(
        new PvxEnginePolicy(),
        new SystemOsProbe(),
        new PvxDefaultEngineSelector(new SystemOsProbe()),
        new PvxEngineFileResolver(new SystemOsProbe()),
        new NativeModuleLoader(),
        new SystemFileExistence());

    private readonly IPvxEnginePolicy _policy;
    private readonly IOSProbe _os;
    private readonly PvxDefaultEngineSelector _defaults;
    private readonly IPvxEngineFileResolver _resolver;
    private readonly IPvxModuleLoader _loader;
    private readonly IPvxFileExistence _files;

    public StandardEngineActivator(IPvxEnginePolicy policy, IOSProbe os, PvxDefaultEngineSelector defaults,
        IPvxEngineFileResolver resolver, IPvxModuleLoader loader, IPvxFileExistence files)
    {
        _policy = policy ?? throw new ArgumentNullException(nameof(policy));
        _os = os ?? throw new ArgumentNullException(nameof(os));
        _defaults = defaults ?? throw new ArgumentNullException(nameof(defaults));
        _resolver = resolver ?? throw new ArgumentNullException(nameof(resolver));
        _loader = loader ?? throw new ArgumentNullException(nameof(loader));
        _files = files ?? throw new ArgumentNullException(nameof(files));
    }

    public IPvxEngineHandle Open(ModelConfig config)
    {
        ArgumentNullException.ThrowIfNull(config);
        var requested = config.Engine ?? _defaults.DefaultEngine(_resolver);
        var selection = new PvxEngineSelector(_policy, _os).Validate(requested);
        if (!selection.IsValid)
            throw new DllNotFoundException(
                $"invalid engine selection '{requested}': {selection.Message}");
        var handle = PvxEngineHandle.Open(selection.Spec!, _resolver, _loader, _files);
        PvxDispatchRegistry.Activate(new PvxEngineRouteAdapter(handle));
        return handle;
    }

    /// <summary>Bridges the engine handle onto the dispatch-registry contract
    /// (the route is consulted only during engine activation, e.g. the ABI
    /// handshake window and process-wide safety-net APIs).</summary>
    private sealed class PvxEngineRouteAdapter : IPvxDispatchRouter
    {
        private readonly IPvxEngineHandle _handle;
        public PvxEngineRouteAdapter(IPvxEngineHandle handle) => _handle = handle;
        public string EngineId => _handle.EngineId;
        public IntPtr ImageHandle => _handle.ImageHandle;
        public IPvxEntryPoints EntryPoints => _handle.EntryPoints;
        public IntPtr RouteLibrary(string logicalLibraryName) =>
            logicalLibraryName == "pvxinference" ? _handle.ImageHandle : IntPtr.Zero;
    }
}