using ProvostXInference.Abstractions;
using ProvostXInference.Interop;

namespace ProvostXInference.Implementation;

/// <summary>Wraps a conversation state (pvx_conversation_save). Token count
/// queries dispatch on the OWNING ENGINE's per-handle tables
/// (constructor-injected, carried from the conversation at save time).</summary>
public sealed class NativeConversationState : IConversationState
{
    private readonly IntPtr _handle;
    private readonly IPvxEntryPoints _entryPoints;
    private bool _disposed;

    internal NativeConversationState(IntPtr handle, IPvxEntryPoints entryPoints)
    {
        _handle = handle;
        _entryPoints = entryPoints ?? throw new ArgumentNullException(nameof(entryPoints));
    }

    internal IntPtr RawHandle => _handle;

    public int TokenCount => _entryPoints.StateTokenCount(_handle);

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        if (_handle != IntPtr.Zero) _entryPoints.ConversationStateFree(_handle);
    }
}