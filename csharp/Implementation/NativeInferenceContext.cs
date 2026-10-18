using System.Runtime.InteropServices;
using ProvostXInference.Abstractions;
using ProvostXInference.Exceptions;
using ProvostXInference.Interop;
using ProvostXInference.Models;
using ProvostXInference.SafeHandles;

namespace ProvostXInference.Implementation;

/// <summary>Wraps a native pvx_context_t. All calls dispatch on the OWNING
/// MODEL ENGINE's per-handle tables (injected via the model).</summary>
public sealed class NativeInferenceContext : IInferenceContext
{
    private readonly ContextSafeHandle _handle;
    private readonly NativeInferenceModel _model;
    private bool _disposed;

    internal NativeInferenceContext(ContextSafeHandle handle, NativeInferenceModel model)
    {
        _handle = handle;
        _model = model;
    }

    internal IPvxEntryPoints ModelEntryPoints => _model.EntryPoints;
    internal IntPtr RawHandle => _handle.DangerousGetHandle();

    /// <summary>F14 (audit 2026-10-04): detail of the most recent failed decode
/// (pvx_last_error for this context), null after a clean cycle. InferAll still
/// returns InferResult.Failed — callers (LLM batch loop) rely on that control
/// flow — but the failure is no longer detail-free.</summary>
    public string? LastDecodeError { get; internal set; }

    public uint ContextSize => _model.EntryPoints.ContextSize(_handle.DangerousGetHandle());
    public uint SeqMax => _model.EntryPoints.ContextSeqMax(_handle.DangerousGetHandle());

    public IStandardExecutor CreateExecutor()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var ep = _model.EntryPoints;
        ep.ExecutorCreate(_handle.DangerousGetHandle(), out var raw).ThrowIfError(ep, RawHandle); // F16: context-aware detail
        var execHandle = new ExecutorSafeHandle(ep);
        execHandle.SetHandleSafe(raw);
        return new NativeStandardExecutor(execHandle, this);
    }

    public IConversationPool CreatePool()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var ep = _model.EntryPoints;
        ep.PoolCreate(_handle.DangerousGetHandle(), out var raw).ThrowIfError(ep, RawHandle); // F16
        var poolHandle = new PoolSafeHandle(ep);
        poolHandle.SetHandleSafe(raw);
        return new NativeConversationPool(poolHandle, this);
    }

    public InferResult InferAll()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var ep = _model.EntryPoints;
        var result = ep.Infer(_handle.DangerousGetHandle());
        switch (result)
        {
            case PvxDecodeResult.Ok:
                LastDecodeError = null;
                return InferResult.Ok;
            case PvxDecodeResult.NoWork:
                return InferResult.NoWork;
            default:
                // F14: capture pvx_last_error for the owning context so decode
                // failures are traceable (KV overflow vs backend crash vs bad state).
                var errPtr = ep.LastError(RawHandle);
                LastDecodeError = errPtr != IntPtr.Zero ? Marshal.PtrToStringUTF8(errPtr) : null;
                return InferResult.Failed;
        }
    }

    public void KvCopy(int srcSeqId, int dstSeqId)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var ep = _model.EntryPoints;
        ep.KvCopy(_handle.DangerousGetHandle(), srcSeqId, dstSeqId).ThrowIfError(ep, RawHandle); // F16
    }

    public void KvClearSeq(int seqId)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var ep = _model.EntryPoints;
        ep.KvClearSeq(_handle.DangerousGetHandle(), seqId).ThrowIfError(ep, RawHandle); // F16
    }

    public void SetCausalAttention(bool causal)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var ep = _model.EntryPoints;
        ep.SetCausalAttn(_handle.DangerousGetHandle(), causal).ThrowIfError(ep, RawHandle); // F16
    }

    public int[] Tokenize(string text, bool addBos = false, bool parseSpecial = true)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var ep = _model.EntryPoints;
        ep.Tokenize(_handle.DangerousGetHandle(), text, addBos, parseSpecial,
            out var tokensPtr, out var count).ThrowIfError(ep, RawHandle); // F16
        try
        {
            if (count == 0) return Array.Empty<int>();
            var result = new int[count];
            Marshal.Copy(tokensPtr, result, 0, count);
            return result;
        }
        finally { ep.FreeTokens(tokensPtr); }
    }

    public string Detokenize(int[] tokens, bool removeSpecial = false)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var ep = _model.EntryPoints;
        ep.Detokenize(_handle.DangerousGetHandle(), tokens, tokens.Length,
            removeSpecial, out var textPtr).ThrowIfError(ep, RawHandle); // F16
        try
        {
            // F15: detokenized text is UTF-8 — ANSI decode truncated/mojibaked
            // every non-ASCII piece.
            return Marshal.PtrToStringUTF8(textPtr) ?? string.Empty;
        }
        finally { ep.FreeString(textPtr); }
    }

    public string TokenToPiece(int token)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var ep = _model.EntryPoints;
        var buffer = new byte[256];
        ep.TokenToPiece(_handle.DangerousGetHandle(), token, buffer, buffer.Length).ThrowIfError(ep, RawHandle); // F16
        var end = Array.IndexOf(buffer, (byte)0);
        if (end < 0) end = buffer.Length;
        // F15: vocab pieces are UTF-8 (CJK/emoji are multi-byte) — ASCII decoding
        // corrupted them. UTF-8 never splits on the NUL scan because a UTF-8
        // continuation byte is never 0x00.
        return System.Text.Encoding.UTF8.GetString(buffer, 0, end);
    }

    public bool IsEos(int token)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        return _model.EntryPoints.TokenIsEos(_handle.DangerousGetHandle(), token);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _handle.Dispose();
    }
}