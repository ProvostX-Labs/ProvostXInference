using System.Runtime.InteropServices;
using ProvostXInference.Interop;

namespace ProvostXInference.SafeHandles;

/// <summary>Owning SafeHandle for pvx_executor_t. Releases via the OWNING ENGINE's
/// per-handle dispatch tables (constructor-injected) — no global P/Invoke
/// routing, so each executor frees through the engine module that created it.</summary>
internal sealed class ExecutorSafeHandle : SafeHandle
{
    private readonly IPvxEntryPoints _entryPoints;

    public ExecutorSafeHandle(IPvxEntryPoints entryPoints) : base(IntPtr.Zero, true)
    {
        _entryPoints = entryPoints ?? throw new ArgumentNullException(nameof(entryPoints));
    }

    public override bool IsInvalid => handle == IntPtr.Zero;
    internal void SetHandleSafe(IntPtr value) => SetHandle(value);
    protected override bool ReleaseHandle()
    {
        if (handle != IntPtr.Zero) _entryPoints.ExecutorFree(handle);
        return true;
    }
}