using ProvostXInference.Abstractions;
using ProvostXInference.Interop;
using ProvostXInference.SafeHandles;

namespace ProvostXInference.Implementation;

/// <summary>Wraps a pvx_state_t (pvx_state_save). Token count queries dispatch
/// on the OWNING ENGINE's per-handle tables (constructor-injected).</summary>
public sealed class NativeInferenceState : IInferenceState
{
    private readonly StateSafeHandle _handle;
    private readonly IPvxEntryPoints _entryPoints;
    private bool _disposed;

    internal NativeInferenceState(StateSafeHandle handle) : this(handle, handle.EntryPoints) { }

    internal NativeInferenceState(StateSafeHandle handle, IPvxEntryPoints entryPoints)
    {
        _handle = handle;
        _entryPoints = entryPoints ?? throw new ArgumentNullException(nameof(entryPoints));
    }

    private IPvxEntryPoints EP => _entryPoints;
    internal IntPtr RawHandle => _handle.DangerousGetHandle();

    public int TokenCount => EP.StateTokenCount(_handle.DangerousGetHandle());

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _handle.Dispose();
    }
}