using ProvostXInference.Abstractions;
using ProvostXInference.Interop;
using ProvostXInference.SafeHandles;

namespace ProvostXInference.Implementation;

/// <summary>
/// ABI v8 (wave 3, feature 9): wraps a pvx_adapter_t (LoRA). Loaded via
/// NativeInferenceModel.CreateLoraAdapter (arch-matched, typed on mismatch);
/// all dispatch goes through the OWNING ENGINE's per-handle tables. Attached
/// to contexts via NativeInferenceContext.SetAdapters; adapters are freed on
/// Dispose (and by the model at its teardown — native contract).
/// </summary>
public sealed class NativeLoraAdapter : ILoraAdapter
{
    private readonly AdapterSafeHandle _handle;
    private readonly IPvxEntryPoints _entryPoints;
    private bool _disposed;

    internal NativeLoraAdapter(AdapterSafeHandle handle, IPvxEntryPoints entryPoints)
    {
        _handle = handle;
        _entryPoints = entryPoints;
    }

    internal IntPtr RawHandle => _handle.DangerousGetHandle();

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _handle.Dispose();
    }
}