using System.Runtime.InteropServices;

namespace ProvostXInference.Interop;

/// <summary>
/// Native module seam over the OS loader — resolves pvx_* exports from a
/// loaded specialist image and loads/loads-idempotently/unloads by path.
/// Injectable so every dispatch type is testable with in-memory fakes.
/// Stateless utility — no mutable state.
/// </summary>
internal interface IPvxModuleLoader
{
    IntPtr GetExport(IntPtr image, string exportName);
    IntPtr Load(string path);
    bool Free(IntPtr image);
}