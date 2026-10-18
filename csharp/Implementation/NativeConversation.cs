using System.Runtime.InteropServices;
using ProvostXInference.Abstractions;
using ProvostXInference.Exceptions;
using ProvostXInference.Interop;
using ProvostXInference.Models;
using ProvostXInference.SafeHandles;

namespace ProvostXInference.Implementation;

/// <summary>
/// Wraps a leased pvx_conversation_t. Does NOT own the handle — the pool does.
/// Dispose() returns it to the pool if not already returned. All calls dispatch
/// on the OWNING ENGINE's per-handle tables (carried from the pool at lease).
/// </summary>
public sealed class NativeConversation : IConversation
{
    private readonly IntPtr _handle;
    private readonly NativeConversationPool _pool;
    private readonly IPvxEntryPoints _entryPoints;
    private IPvxEntryPoints EP => _entryPoints;
    private bool _returned;
    private bool _disposed;
    private NativeConversationSpecSession? _specSession; // ABI v7 (lazy — created on first EnableSpeculation)

    internal NativeConversation(IntPtr handle, NativeConversationPool pool, IPvxEntryPoints entryPoints)
    {
        _handle = handle;
        _pool = pool;
        _entryPoints = entryPoints;
    }

    internal IntPtr RawHandle => _handle;

    internal void MarkReturned()
    {
        _returned = true;
        // ABI v7: pool return tears the native spec session down in the
        // existing teardown order — drop the managed mirror with it.
        _specSession?.OnReturned();
        _specSession = null;
    }

    public int TokenCount => EP.ConversationTokenCount(_handle);

    public void Prompt(string text)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        EP.ConversationPrompt(_handle, text).ThrowIfError(EP, _pool.Context.RawHandle); // F16
    }

    public void PromptTokens(int[] tokens)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        EP.ConversationPromptTokens(_handle, tokens, tokens.Length).ThrowIfError(EP, _pool.Context.RawHandle); // F16
    }

    public int Sample(SamplingConfig? config = null)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var cfg = config ?? new SamplingConfig();
        var nativeParams = NativeStandardExecutor.ToNativeSampling(cfg);
        EP.ConversationSample(_handle, in nativeParams, out var token).ThrowIfError(EP, _pool.Context.RawHandle); // F16
        return token;
    }

    public int SampleWithGrammar(SamplingConfig? config, IGrammar grammar)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var cfg = config ?? new SamplingConfig();
        var nativeParams = NativeStandardExecutor.ToNativeSampling(cfg);
        var nativeGrammar = (NativeGrammar)grammar;
        EP.ConversationSampleGrammar(_handle, in nativeParams, nativeGrammar.RawHandle, out var token).ThrowIfError(EP, _pool.Context.RawHandle); // F16
        return token;
    }

    public void Rewind(int tokenCount)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        EP.ConversationRewind(_handle, tokenCount).ThrowIfError(EP, _pool.Context.RawHandle); // F16
    }

    public void Reset()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        EP.ConversationReset(_handle).ThrowIfError(EP, _pool.Context.RawHandle); // F16
    }

    public void ResetGrammarState()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        EP.ConversationGrammarReset(_handle).ThrowIfError(EP, _pool.Context.RawHandle); // F16
    }

    public void ShiftLeft(int tokenCount)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        EP.ConversationShiftLeft(_handle, tokenCount).ThrowIfError(EP, _pool.Context.RawHandle); // F16
    }

    public IConversationState SaveState()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        EP.ConversationSave(_handle, out var raw).ThrowIfError(EP, _pool.Context.RawHandle); // F16
        return new NativeConversationState(raw, EP);
    }

    public void RestoreState(IConversationPool pool, IConversationState state)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var nativeState = (NativeConversationState)state;
        var nativePool = (NativeConversationPool)pool;
        EP.ConversationRestore(nativePool.RawHandle, _handle, nativeState.RawHandle).ThrowIfError(EP, _pool.Context.RawHandle); // F16
    }

    public byte[] ExportCheckpoint()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        EP.ConversationExport(_handle, out var buffer, out var size).ThrowIfError(EP, _pool.Context.RawHandle); // F16
        try
        {
            var blob = new byte[size];
            Marshal.Copy(buffer, blob, 0, (int)size);
            return blob;
        }
        finally
        {
            EP.ConversationExportFree(buffer);
        }
    }

    public void ImportCheckpoint(byte[] data)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(data);
        EP.ConversationImport(_handle, data, (ulong)data.Length).ThrowIfError(EP, _pool.Context.RawHandle); // F16
    }

    public void PromptWithImages(string text, IVisionEncoder vision, params byte[][] images)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var nativeVision = (NativeVisionEncoder)vision;
        var nImages = images.Length;
        var ptrs = new IntPtr[nImages];
        var sizes = new int[nImages];
        try
        {
            for (int i = 0; i < nImages; i++)
            {
                sizes[i] = images[i].Length;
                ptrs[i] = Marshal.AllocHGlobal(images[i].Length);
                Marshal.Copy(images[i], 0, ptrs[i], images[i].Length);
            }
            EP.ConversationPromptWithImages(_handle, nativeVision.RawHandle,
                text, ptrs, sizes, nImages).ThrowIfError(EP, _pool.Context.RawHandle); // F16
        }
        finally
        {
            foreach (var p in ptrs) if (p != IntPtr.Zero) Marshal.FreeHGlobal(p);
        }
    }

    // ── ABI v7: batched/pool-lane speculative decoding (one session per conversation) ──

    public SpeculativeConfig? ActiveSpeculation => _specSession?.ActiveConfig;

    public void EnableSpeculation(SpeculativeConfig config)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_returned) throw new InvalidOperationException(
            "speculation cannot be enabled on a conversation already returned to the pool");
        _specSession ??= new NativeConversationSpecSession(_handle, EP, _pool.Context.RawHandle);
        _specSession.Enable(config); // typed native refusal propagates, mirror untouched
    }

    public void DisableSpeculation()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        _specSession?.Disable(); // idempotent natively; no session = no-op
    }

    public IReadOnlyList<int> GenerateSpeculative(SamplingConfig? config, int maxTokens, Action<int>? onToken = null)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var nativeParams = NativeStandardExecutor.ToNativeSampling(config ?? new SamplingConfig());
        var tokens = new List<int>();
        // Reverse pinvoke: synchronous on the calling thread — the local
        // delegate reference keeps the stub alive for the call's duration.
        // Byte return: the C ABI expects a 1-byte C99 bool ("continue").
        PvxAbiDelegates.PvxSpecTokenCallbackDelegate cb = (token, _) =>
        {
            tokens.Add(token);
            onToken?.Invoke(token);
            return 1;
        };
        var cbPtr = System.Runtime.InteropServices.Marshal.GetFunctionPointerForDelegate(cb);
        // Typed refusal when no spec session is active on this conversation —
        // never a silent plain-sample fallback.
        EP.ConversationGenerateSpec(_handle, in nativeParams, maxTokens,
            cbPtr, IntPtr.Zero, out var generated).ThrowIfError(EP, _pool.Context.RawHandle); // F16
        System.GC.KeepAlive(cb);
        return tokens;
    }

    public SpeculationStats SpeculationStats()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        EP.ConversationSpecStats(_handle, out var drafted, out var accepted)
            .ThrowIfError(EP, _pool.Context.RawHandle); // F16
        return new SpeculationStats(drafted, accepted);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        if (_returned) return;
        // F17 (audit 2026-10-04): if the pool was disposed before this in-flight
        // conversation (context teardown ordering), Return throws from inside
        // Dispose — leaking the native slot AND breaking using/teardown chains.
        // Best-effort return; the failure is traced, never propagated.
        try
        {
            _pool.Return(this);
        }
        catch (Exception ex)
        {
            // BL-03 sweep: Debug.WriteLine compiles out in Release builds, which
            // would hide a leaked native slot entirely. The failure is still
            // best-effort (never propagated — F17), but the trace now goes to
            // stderr, which is observable in Release too.
            Console.Error.WriteLine($"[NativeConversation] Pool return failed during Dispose: {ex.GetType().Name}: {ex.Message}");
        }
    }
}