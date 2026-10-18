using System.Runtime.InteropServices;
using ProvostXInference.Interop;

namespace ProvostXInference.SafeHandles;

/// <summary>Owning SafeHandle for pvx_mmproj_t. Releases via the OWNING ENGINE's
/// per-handle dispatch tables (constructor-injected) — no global P/Invoke
/// routing.</summary>
internal sealed class MmprojSafeHandle : SafeHandle
{
    private readonly IPvxEntryPoints _entryPoints;

    public MmprojSafeHandle(IPvxEntryPoints entryPoints) : base(IntPtr.Zero, true)
    {
        _entryPoints = entryPoints ?? throw new ArgumentNullException(nameof(entryPoints));
    }

    /// <summary>Tables this handle was created with (for wrappers built from the
    /// handle alone — e.g. the vision encoder).</summary>
    internal IPvxEntryPoints EntryPoints => _entryPoints;

    public override bool IsInvalid => handle == IntPtr.Zero;
    internal void SetHandleSafe(IntPtr value) => SetHandle(value);
    protected override bool ReleaseHandle()
    {
        if (handle != IntPtr.Zero) _entryPoints.FreeMmproj(handle);
        return true;
    }
}