namespace ProvostXInference.Interop;

/// <summary>
/// Per-engine handle bundle: which specialist is loaded, its raw image handle,
/// and its own function tables. DllImport resolution (the late-bound path used
/// during engine activation, e.g. the ABI handshake) routes through
/// RouteLibrary() onto THIS image — steady-state calls never consult it.
/// </summary>
internal interface IPvxDispatchRouter
{
    /// <summary>Engine id of the loaded specialist ("metal" | "cuda" | "vulkan").</summary>
    string EngineId { get; }

    /// <summary>Raw native image handle of the loaded specialist module.</summary>
    IntPtr ImageHandle { get; }

    /// <summary>Per-handle function tables onto this image.</summary>
    IPvxEntryPoints EntryPoints { get; }

    /// <summary>Route a logical library name onto this image
    /// (non-pvxinference names are explicitly NOT resolved).</summary>
    IntPtr RouteLibrary(string logicalLibraryName);
}