using ProvostXInference.Abstractions;
using ProvostXInference.Exceptions;
using ProvostXInference.Interop;
using ProvostXInference.SafeHandles;

namespace ProvostXInference.Implementation;

/// <summary>Wraps a native pvx_pool_t. All calls dispatch on the OWNING
/// ENGINE's per-handle tables (flowed from the context's model); a leased
/// conversation carries the same tables for its own calls.</summary>
public sealed class NativeConversationPool : IConversationPool
{
    private readonly PoolSafeHandle _handle;
    private readonly NativeInferenceContext _context;
    private bool _disposed;

    internal NativeConversationPool(PoolSafeHandle handle, NativeInferenceContext context)
    {
        _handle = handle;
        _context = context;
    }

    private IPvxEntryPoints EP => _context.ModelEntryPoints;

    internal IntPtr RawHandle => _handle.DangerousGetHandle();
    internal NativeInferenceContext Context => _context;

    public IConversation Lease()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        EP.PoolLease(_handle.DangerousGetHandle(), out var raw).ThrowIfError(EP, _context.RawHandle); // F16
        return new NativeConversation(raw, this, EP);
    }

    public void Return(IConversation conversation)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var conv = (NativeConversation)conversation;
        EP.PoolReturn(_handle.DangerousGetHandle(), conv.RawHandle).ThrowIfError(EP);
        conv.MarkReturned();
    }

    public int AvailableCount => EP.PoolAvailable(_handle.DangerousGetHandle());
    public uint Size => EP.PoolSize(_handle.DangerousGetHandle());

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _handle.Dispose();
    }
}