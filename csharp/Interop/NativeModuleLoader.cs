using System.Runtime.InteropServices;

namespace ProvostXInference.Interop;

/// <summary>System implementation of the module seam (NativeLibrary).
/// Stateless utility — no mutable state.</summary>
internal sealed class NativeModuleLoader : IPvxModuleLoader
{
    public IntPtr GetExport(IntPtr image, string exportName)
    {
        if (image == IntPtr.Zero)
            throw new ArgumentException("cannot resolve exports of a null module image", nameof(image));
        try
        {
            return NativeLibrary.GetExport(image, exportName);
        }
        catch (EntryPointNotFoundException)
        {
            // Normalize per the loader seam: a missing export IS IntPtr.Zero
            // (the table builder turns that into a typed, export-naming error).
            return IntPtr.Zero;
        }
    }

    /// <summary>Load the module image (already-loaded paths are de-duped by the
    /// OS loader and return the same handle — BL-01 spike evidence E1a).</summary>
    public IntPtr Load(string path) => NativeLibrary.Load(path);

    /// <summary>Decrement the OS refcount of a previously loaded image. A
    /// bool seam (always true — the runtime signals unload failure by
    /// exception, not return value). Stateless per call.</summary>
    public bool Free(IntPtr image)
    {
        NativeLibrary.Free(image);
        return true;
    }
}