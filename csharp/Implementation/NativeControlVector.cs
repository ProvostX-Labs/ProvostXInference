using ProvostXInference.Abstractions;
using ProvostXInference.Interop;
using ProvostXInference.SafeHandles;

namespace ProvostXInference.Implementation;

/// <summary>
/// ABI v8 (wave 3, feature 10): wraps a pvx_cvec_t (control-vector direction
/// file). Loaded via NativeInferenceModel.CreateControlVector (n_embd-matched,
/// typed refusal naming both dims on mismatch); all dispatch goes through the
/// OWNING ENGINE's per-handle tables. Applied to contexts via
/// NativeInferenceContext.SetControlVector; weights-surface modifier
/// (KV-neutral, live attach/detach).
/// </summary>
public sealed class NativeControlVector : IControlVector
{
    private readonly ControlVectorSafeHandle _handle;
    private readonly IPvxEntryPoints _entryPoints;
    private bool _disposed;

    internal NativeControlVector(ControlVectorSafeHandle handle, IPvxEntryPoints entryPoints)
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