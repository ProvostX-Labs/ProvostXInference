using System.Runtime.InteropServices;
using ProvostXInference.Interop;

namespace ProvostXInference.SafeHandles;

/// <summary>Owning SafeHandle for pvx_cvec_t (ABI v8, wave 3). Releases via the
/// OWNING ENGINE's per-handle dispatch tables (constructor-injected) — no
/// global P/Invoke routing.</summary>
internal sealed class ControlVectorSafeHandle : SafeHandle
{
    private readonly IPvxEntryPoints _entryPoints;

    public ControlVectorSafeHandle(IPvxEntryPoints entryPoints) : base(IntPtr.Zero, true)
    {
        _entryPoints = entryPoints ?? throw new ArgumentNullException(nameof(entryPoints));
    }

    public override bool IsInvalid => handle == IntPtr.Zero;
    internal void SetHandleSafe(IntPtr value) => SetHandle(value);
    protected override bool ReleaseHandle()
    {
        if (handle != IntPtr.Zero) _entryPoints.ControlVectorFree(handle);
        return true;
    }
}
