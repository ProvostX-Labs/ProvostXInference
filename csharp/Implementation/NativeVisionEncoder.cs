using System.Runtime.InteropServices;
using ProvostXInference.Abstractions;
using ProvostXInference.Interop;
using ProvostXInference.SafeHandles;

namespace ProvostXInference.Implementation;

/// <summary>Wraps an mmproj context for vision/image encoding. All calls
/// dispatch on the OWNING ENGINE's per-handle tables (constructor-injected).</summary>
public sealed class NativeVisionEncoder : IVisionEncoder
{
    private readonly MmprojSafeHandle _handle;
    private readonly IPvxEntryPoints _entryPoints;
    private bool _disposed;

    internal NativeVisionEncoder(MmprojSafeHandle handle) : this(handle, handle.EntryPoints) { }

    internal NativeVisionEncoder(MmprojSafeHandle handle, IPvxEntryPoints entryPoints)
    {
        _handle = handle;
        _entryPoints = entryPoints ?? throw new ArgumentNullException(nameof(entryPoints));
    }

    internal IntPtr RawHandle => _handle.DangerousGetHandle();

    private IPvxEntryPoints EP => _entryPoints;

    public string Marker
    {
        get
        {
            var ptr = EP.MtmdMarkerStatic();
            return ptr != IntPtr.Zero ? Marshal.PtrToStringUTF8(ptr) ?? "<image>" : "<image>";
        }
    }

    public bool NeedsNonCausalAttention => EP.MtmdNeedsNonCausal(_handle.DangerousGetHandle());

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _handle.Dispose();
    }
}