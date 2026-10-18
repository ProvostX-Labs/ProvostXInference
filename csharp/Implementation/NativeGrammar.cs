using ProvostXInference.Abstractions;
using ProvostXInference.Exceptions;
using ProvostXInference.Interop;
using ProvostXInference.Models;
using ProvostXInference.SafeHandles;

namespace ProvostXInference.Implementation;

/// <summary>Wraps a pvx_grammar_t. Created through — and dispatching on — the
/// OWNING MODEL ENGINE's per-handle tables.</summary>
public sealed class NativeGrammar : IGrammar
{
    private readonly GrammarSafeHandle _handle;
    private bool _disposed;

    internal NativeGrammar(GrammarSafeHandle handle) { _handle = handle; }
    internal IntPtr RawHandle => _handle.DangerousGetHandle();

    public static NativeGrammar Create(IInferenceModel model, string grammarStr, string grammarRoot)
    {
        var nativeModel = (NativeInferenceModel)model;
        var raw = nativeModel.EntryPoints.GrammarCreate(nativeModel.RawHandle, grammarStr, grammarRoot);
        if (raw == IntPtr.Zero)
            throw new InferenceException(1, "Failed to parse grammar");
        var handle = new GrammarSafeHandle(nativeModel.EntryPoints);
        handle.SetHandleSafe(raw);
        return new NativeGrammar(handle);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _handle.Dispose();
    }
}