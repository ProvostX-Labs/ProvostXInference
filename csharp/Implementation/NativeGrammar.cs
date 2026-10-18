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
    private readonly GrammarKind _kind;
    private bool _disposed;

    internal NativeGrammar(GrammarSafeHandle handle, GrammarKind kind = GrammarKind.Gbnf)
    {
        _handle = handle;
        _kind = kind;
    }
    internal IntPtr RawHandle => _handle.DangerousGetHandle();

    /// <summary>Provenance label (ABI v7): Gbnf for hand-written GBNF,
    /// JsonSchema for schema-converted grammars. Pure label.</summary>
    public GrammarKind Kind => _kind;

    public static NativeGrammar Create(IInferenceModel model, string grammarStr, string grammarRoot,
        GrammarKind kind = GrammarKind.Gbnf)
    {
        var nativeModel = (NativeInferenceModel)model;
        var raw = nativeModel.EntryPoints.GrammarCreate(nativeModel.RawHandle, grammarStr, grammarRoot);
        if (raw == IntPtr.Zero)
            throw new InferenceException(1, "Failed to parse grammar");
        var handle = new GrammarSafeHandle(nativeModel.EntryPoints);
        handle.SetHandleSafe(raw);
        return new NativeGrammar(handle, kind);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _handle.Dispose();
    }
}