using System.Runtime.InteropServices;
using ProvostXInference.Interop;

namespace ProvostXInference.SafeHandles;

/// <summary>Owning SafeHandle for pvx_state_t (executor state). Releases via the
/// OWNING ENGINE's per-handle dispatch tables (constructor-injected) — no
/// global P/Invoke routing.</summary>
internal sealed class StateSafeHandle : SafeHandle
{
    private readonly IPvxEntryPoints _entryPoints;

    public StateSafeHandle(IPvxEntryPoints entryPoints) : base(IntPtr.Zero, true)
    {
        _entryPoints = entryPoints ?? throw new ArgumentNullException(nameof(entryPoints));
    }

    /// <summary>Tables this handle was created with (for wrappers built from
    /// the handle alone — e.g. the executor state wrapper).</summary>
    internal IPvxEntryPoints EntryPoints => _entryPoints;

    public override bool IsInvalid => handle == IntPtr.Zero;
    internal void SetHandleSafe(IntPtr value) => SetHandle(value);
    protected override bool ReleaseHandle()
    {
        if (handle != IntPtr.Zero) _entryPoints.StateFree(handle);
        return true;
    }
}