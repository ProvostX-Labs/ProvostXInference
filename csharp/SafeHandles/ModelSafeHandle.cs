using System.Runtime.InteropServices;
using ProvostXInference.Interop;

namespace ProvostXInference.SafeHandles;

/// <summary>Owning SafeHandle for pvx_model_t. Releases via the OWNING ENGINE's
/// per-handle dispatch tables (constructor-injected) — no global P/Invoke
/// routing, so each model frees through the engine module that created it.</summary>
internal sealed class ModelSafeHandle : SafeHandle
{
    private readonly IPvxEntryPoints _entryPoints;

    public ModelSafeHandle(IPvxEntryPoints entryPoints) : base(IntPtr.Zero, true)
    {
        _entryPoints = entryPoints ?? throw new ArgumentNullException(nameof(entryPoints));
    }

    public override bool IsInvalid => handle == IntPtr.Zero;
    internal void SetHandleSafe(IntPtr value) => SetHandle(value);
    protected override bool ReleaseHandle()
    {
        if (handle != IntPtr.Zero) _entryPoints.FreeModel(handle);
        return true;
    }
}