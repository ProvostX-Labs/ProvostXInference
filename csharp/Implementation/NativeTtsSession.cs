using System.Runtime.InteropServices;
using ProvostXInference.Abstractions;
using ProvostXInference.Exceptions;
using ProvostXInference.Interop;
using ProvostXInference.Models;
using ProvostXInference.SafeHandles;

namespace ProvostXInference.Implementation;

/// <summary>
/// ABI v8 (wave 3, feature 7): wraps a pvx_tts_session_t — a SEPARATE decode
/// context on the backbone speech model plus a gen-audio mmproj (never a
/// conversation context; KV-neutral by construction). All calls dispatch on
/// the OWNING ENGINE's per-handle tables.
/// </summary>
public sealed class NativeTtsSession : ITtsSession
{
    private readonly TtsSessionSafeHandle _handle;
    private readonly IPvxEntryPoints _entryPoints;
    private readonly IntPtr _contextErrorHandle;  // model handle for typed-error text
    private bool _disposed;

    internal NativeTtsSession(TtsSessionSafeHandle handle, IPvxEntryPoints entryPoints, IntPtr modelHandle)
    {
        _handle = handle;
        _entryPoints = entryPoints;
        _contextErrorHandle = modelHandle;
    }

    public TtsResult Generate(TtsRequest request, string text)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(request);
        ArgumentException.ThrowIfNullOrEmpty(text);

        // Pin the optional language tag for the duration of the call.
        var langPtr = IntPtr.Zero;
        try
        {
            if (!string.IsNullOrEmpty(request.Lang))
            {
                langPtr = PvxNative.AllocUTF8(request.Lang);
            }
            var nativeParams = new PvxTtsParams
            {
                Lang = langPtr,
                TopK = request.TopK,
                TopP = request.TopP,
                Temp = request.Temp,
                Seed = request.Seed,
                MaxFrames = request.MaxFrames,
            };
            _entryPoints.TtsGenerate(_handle.DangerousGetHandle(), in nativeParams, text,
                out var pcmPtr, out var nSamples, out var sampleRate).ThrowIfError(_entryPoints, _contextErrorHandle);
            try
            {
                var pcm = new float[nSamples];
                Marshal.Copy(pcmPtr, pcm, 0, nSamples);
                return new TtsResult(pcm, sampleRate);
            }
            finally { _entryPoints.FreeFloatArray(pcmPtr); }
        }
        finally
        {
            if (langPtr != IntPtr.Zero) Marshal.FreeHGlobal(langPtr);
        }
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _handle.Dispose();
    }
}
