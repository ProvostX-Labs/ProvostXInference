using ProvostXInference.Abstractions;
using ProvostXInference.Exceptions;
using ProvostXInference.Interop;
using ProvostXInference.Models;
using ProvostXInference.SafeHandles;

namespace ProvostXInference.Implementation;

/// <summary>Wraps a pvx_executor_t (standard, non-batched path). All calls
/// dispatch on the OWNING ENGINE's per-handle tables (flowed from the context).</summary>
public sealed class NativeStandardExecutor : IStandardExecutor
{
    private readonly ExecutorSafeHandle _handle;
    private readonly NativeInferenceContext _context;
    private bool _disposed;

    internal NativeStandardExecutor(ExecutorSafeHandle handle, NativeInferenceContext context)
    {
        _handle = handle;
        _context = context;
    }

    private IPvxEntryPoints EP => _context.ModelEntryPoints;

    internal IntPtr RawHandle => _handle.DangerousGetHandle();

    /// <summary>F14 (audit 2026-10-04): detail of the most recent failed decode
    /// (pvx_last_error for the owning context), null after a clean cycle. Infer
    /// still returns InferResult.Failed — callers rely on that control flow —
    /// but the failure is no longer detail-free.</summary>
    public string? LastDecodeError => _context.LastDecodeError;

    public int TokenCount => EP.ExecutorTokenCount(_handle.DangerousGetHandle());

    public void Prompt(string text)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        EP.ExecutorPrompt(_handle.DangerousGetHandle(), text).ThrowIfError(EP, _context.RawHandle); // F16
    }

    public void PromptTokens(int[] tokens)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        EP.ExecutorPromptTokens(_handle.DangerousGetHandle(), tokens, tokens.Length).ThrowIfError(EP, _context.RawHandle); // F16
    }

    public void PromptWithImages(string text, IInferenceModel model, params byte[][] images)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var nativeModel = (NativeInferenceModel)model;
        var nImages = images.Length;
        var ptrs = new IntPtr[nImages];
        var sizes = new int[nImages];
        try
        {
            for (int i = 0; i < nImages; i++)
            {
                sizes[i] = images[i].Length;
                ptrs[i] = System.Runtime.InteropServices.Marshal.AllocHGlobal(images[i].Length);
                System.Runtime.InteropServices.Marshal.Copy(images[i], 0, ptrs[i], images[i].Length);
            }
            EP.ExecutorPromptWithImages(_handle.DangerousGetHandle(), nativeModel.RawHandle,
                text, ptrs, sizes, nImages).ThrowIfError(EP, _context.RawHandle); // F16
        }
        finally
        {
            foreach (var p in ptrs) if (p != IntPtr.Zero) System.Runtime.InteropServices.Marshal.FreeHGlobal(p);
        }
    }

    public int SampleWithGrammar(SamplingConfig? config, IGrammar grammar)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var cfg = config ?? new SamplingConfig();
        var nativeParams = ToNativeSampling(cfg);
        var nativeGrammar = (NativeGrammar)grammar;
        EP.ExecutorSampleGrammar(_handle.DangerousGetHandle(), in nativeParams, nativeGrammar.RawHandle, out var token).ThrowIfError(EP, _context.RawHandle); // F16
        return token;
    }

    public InferResult Infer()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var result = EP.ExecutorInfer(_handle.DangerousGetHandle());
        switch (result)
        {
            case PvxDecodeResult.Ok:
                _context.LastDecodeError = null;
                return InferResult.Ok;
            case PvxDecodeResult.NoWork:
                return InferResult.NoWork;
            default:
                // F14: capture pvx_last_error for the owning context so decode
                // failures are traceable instead of a bare InferResult.Failed.
                var errPtr = EP.LastError(_context.RawHandle);
                _context.LastDecodeError = errPtr != IntPtr.Zero
                    ? System.Runtime.InteropServices.Marshal.PtrToStringUTF8(errPtr)
                    : null;
                return InferResult.Failed;
        }
    }

    public int Sample(SamplingConfig? config = null)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var cfg = config ?? new SamplingConfig();
        var nativeParams = ToNativeSampling(cfg);
        EP.ExecutorSample(_handle.DangerousGetHandle(), in nativeParams, out var token).ThrowIfError(EP, _context.RawHandle); // F16
        return token;
    }

    public void Rewind(int tokenCount)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        EP.ExecutorRewind(_handle.DangerousGetHandle(), tokenCount).ThrowIfError(EP, _context.RawHandle); // F16
    }

    public void Reset()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        EP.ExecutorReset(_handle.DangerousGetHandle()).ThrowIfError(EP, _context.RawHandle); // F16
    }

    public void ShiftLeft(int tokenCount)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        EP.ExecutorShiftLeft(_handle.DangerousGetHandle(), tokenCount).ThrowIfError(EP, _context.RawHandle); // F16
    }

    public IInferenceState SaveState()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        EP.StateSave(_handle.DangerousGetHandle(), out var raw).ThrowIfError(EP, _context.RawHandle); // F16
        var stateHandle = new StateSafeHandle(EP);
        stateHandle.SetHandleSafe(raw);
        return new NativeInferenceState(stateHandle);
    }

    public void RestoreState(IInferenceState state)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var nativeState = (NativeInferenceState)state;
        EP.StateRestore(_handle.DangerousGetHandle(), nativeState.RawHandle).ThrowIfError(EP, _context.RawHandle); // F16
    }

    // ── Speculative decoding (INF BL-05, sequential mode) ──

    // Read-only live state mirror: all mutations flow through Enable/Disable
    // below, so a managed mirror can never diverge from the native session.
    private SpeculativeConfig? _activeSpeculation;

    /// <inheritdoc/>
    public SpeculativeConfig? ActiveSpeculation => _activeSpeculation;

    public void EnableSpeculation(SpeculativeConfig config)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var pathPtr = config.DraftModelPath is null ? IntPtr.Zero : PvxNative.AllocUTF8(config.DraftModelPath);
        try
        {
            var nativeParams = new PvxSpecParams(config, pathPtr);
            EP.ExecutorSpecInit(_handle.DangerousGetHandle(), in nativeParams)
                .ThrowIfError(EP, _context.RawHandle); // F16
            _activeSpeculation = config; // set only after native success
        }
        finally { if (pathPtr != IntPtr.Zero) System.Runtime.InteropServices.Marshal.FreeHGlobal(pathPtr); }
    }

    public void DisableSpeculation()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        EP.ExecutorSpecFree(_handle.DangerousGetHandle()); // idempotent, void
        _activeSpeculation = null;
    }

    public IReadOnlyList<int> GenerateSpeculative(SamplingConfig? config, int maxTokens, Action<int>? onToken = null)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var nativeParams = ToNativeSampling(config ?? new SamplingConfig());
        var tokens = new List<int>();
        // Reverse pinvoke: the collector runs on the SAME thread synchronously
        // inside the native call, so the local delegate reference keeps the
        // marshaling stub alive for the call's duration. Byte return: the C ABI
        // expects a 1-byte C99 bool ("continue").
        PvxAbiDelegates.PvxSpecTokenCallbackDelegate cb = (token, _) =>
        {
            tokens.Add(token);
            onToken?.Invoke(token);
            return 1;
        };
        var cbPtr = System.Runtime.InteropServices.Marshal.GetFunctionPointerForDelegate(cb);
        EP.ExecutorGenerateSpec(_handle.DangerousGetHandle(), in nativeParams, maxTokens,
            cbPtr, IntPtr.Zero, out var generated).ThrowIfError(EP, _context.RawHandle); // F16
        System.GC.KeepAlive(cb);
        return tokens;
    }

    internal static PvxSamplingParams ToNativeSampling(SamplingConfig cfg)
    {
        return new PvxSamplingParams(cfg.Temperature, cfg.TopP, cfg.TopK, cfg.MinP,
            cfg.RepeatPenalty, cfg.RepeatLastN, cfg.PenaltyPresent,
            cfg.MaxTokens, cfg.IgnoreEos);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _activeSpeculation = null; // Dispose tears the speculative session down
        _handle.Dispose();
    }
}