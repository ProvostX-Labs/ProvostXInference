using System.Runtime.InteropServices;
using ProvostXInference.Abstractions;
using ProvostXInference.Exceptions;
using ProvostXInference.Interop;
using ProvostXInference.Models;
using ProvostXInference.SafeHandles;

namespace ProvostXInference.Implementation;

/// <summary>
/// Wraps a native pvx_model_t. Factory: Load(). BL-02: EVERY pvx_* call this
/// model (and everything created from it) makes is dispatched through the
/// OWNING ENGINE's per-handle dispatch tables — resolved once at engine open
/// and injected down the whole object graph; no global P/Invoke routing.
/// </summary>
public sealed class NativeInferenceModel : IInferenceModel
{
    private readonly ModelSafeHandle _handle;
    private readonly ModelConfig _config;
    private readonly IPvxEngineHandle _engine;
    private bool _disposed;

    private NativeInferenceModel(ModelSafeHandle handle, ModelConfig config, IPvxEngineHandle engine)
    {
        _handle = handle;
        _config = config;
        _engine = engine;
    }

    /// <summary>Factory: loads a model from disk on the config-selected
    /// specialist engine (fail-closed selection, per-handle dispatch).</summary>
    public static NativeInferenceModel Load(ModelConfig config)
    {
        // F18: explicit ABI handshake happens inside engine activation
        // (PvxDispatchRouter validates pvx_abi_version == expected at load time,
        // BEFORE any model P/Invoke can run).
        var engine = StandardEngineActivator.Default.Open(config);
        var entryPoints = engine.EntryPoints;
        var pathPtr = PvxNative.AllocUTF8(config.Path);
        // Reasoning-marker list (data-driven): pin the strings for the duration
        // of the native call — the native side copies them at load time.
        var markerPtrs = new List<IntPtr>();
        var markerArrayPtr = IntPtr.Zero;
        try
        {
            if (config.ReasoningMarkers is { Length: > 0 })
            {
                foreach (var m in config.ReasoningMarkers)
                    markerPtrs.Add(PvxNative.AllocUTF8(m));
                markerArrayPtr = Marshal.AllocHGlobal(IntPtr.Size * markerPtrs.Count);
                for (int i = 0; i < markerPtrs.Count; i++)
                    Marshal.WriteIntPtr(markerArrayPtr, i * IntPtr.Size, markerPtrs[i]);
            }
            var nativeParams = new PvxModelParams(pathPtr, config.GpuLayers, config.Threads,
                config.FlashAttention, (PvxKvType)config.KvCacheType,
                markerArrayPtr, markerPtrs.Count);
            try
            {
                entryPoints.LoadModel(in nativeParams, out var raw).ThrowIfError(entryPoints);
                var handle = new ModelSafeHandle(entryPoints);
                handle.SetHandleSafe(raw);
                return new NativeInferenceModel(handle, config, engine);
            }
            catch
            {
                // Load failure must not leak the opened specialist engine handle
                // (the module stays mapped and the dispatch route stays active).
                engine.Dispose();
                throw;
            }
        }
        finally
        {
            foreach (var p in markerPtrs) Marshal.FreeHGlobal(p);
            if (markerArrayPtr != IntPtr.Zero) Marshal.FreeHGlobal(markerArrayPtr);
            Marshal.FreeHGlobal(pathPtr);
        }
    }

    /// <summary>The dispatch tables this model's engine loaded (per-handle).</summary>
    internal IPvxEntryPoints EntryPoints => _engine.EntryPoints;

    public IInferenceContext CreateContext(ContextConfig config)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var nativeParams = new PvxContextParams(config.ContextSize, config.BatchSize,
            config.SeqMax, (PvxPoolingType)config.PoolingType);
        EntryPoints.CreateContext(_handle.DangerousGetHandle(), in nativeParams, out var raw).ThrowIfError(EntryPoints, _handle.DangerousGetHandle());
        var ctxHandle = new ContextSafeHandle(EntryPoints);
        ctxHandle.SetHandleSafe(raw);
        return new NativeInferenceContext(ctxHandle, this);
    }

    public float[] GetEmbeddings(IInferenceContext context, string text)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var ctx = (NativeInferenceContext)context;
        EntryPoints.GetEmbeddings(_handle.DangerousGetHandle(), ctx.RawHandle, text,
            out var embPtr, out var dim).ThrowIfError(EntryPoints, ctx.RawHandle);
        try
        {
            var result = new float[dim];
            Marshal.Copy(embPtr, result, 0, dim);
            return result;
        }
        finally { EntryPoints.FreeEmbeddings(embPtr); }
    }

    public int EmbeddingDimension
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return EntryPoints.EmbeddingDim(_handle.DangerousGetHandle());
        }
    }

    public IVisionEncoder LoadVisionEncoder(string mmprojPath)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        EntryPoints.LoadMmproj(_handle.DangerousGetHandle(), mmprojPath, out var raw).ThrowIfError(EntryPoints, _handle.DangerousGetHandle());
        var handle = new MmprojSafeHandle(EntryPoints);
        handle.SetHandleSafe(raw);
        return new NativeVisionEncoder(handle);
    }

    internal IntPtr RawHandle => _handle.DangerousGetHandle();

    internal IPvxEngineHandle Engine => _engine;

    /// <summary>Native capability probe (arch-based): hybrid/recurrent models
    /// cannot partially rewind their memory — see interface doc.</summary>
    public bool PartialRewindSupported =>
        EntryPoints.ModelPartialRewindSupported(_handle.DangerousGetHandle());

    /// <summary>Speculative-decoding self-report (INF BL-05): the engine's
    /// supported-types list for THIS loaded model (GGUF metadata-derived:
    /// arch + nextn/MTP tensor presence). Offer exactly these types for
    /// sequential mode — never assume.</summary>
    public SpeculativeCapabilities SpeculativeCapabilities
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            EntryPoints.ModelSpecCapabilities(_handle.DangerousGetHandle(), out var caps)
                .ThrowIfError(EntryPoints, _handle.DangerousGetHandle()); // F16
            return new SpeculativeCapabilities((SpeculativeType) caps.Types,
                                                caps.SpecVision != 0); // ABI v6
        }
    }

    /// <summary>Typed pre-validation of a speculative config against this model
    /// (mode/path/capability/vocab shape) without enabling anything — enabling
    /// re-validates fail-closed anyway.</summary>
    public void ValidateSpeculation(SpeculativeConfig config)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var pathPtr = config.DraftModelPath is null ? IntPtr.Zero : PvxNative.AllocUTF8(config.DraftModelPath);
        var cachePtr = config.Ngram?.CachePath is null ? IntPtr.Zero : PvxNative.AllocUTF8(config.Ngram.CachePath);
        try
        {
            var nativeParams = new PvxSpecParams(config, pathPtr, cachePtr);
            EntryPoints.ModelSpecValidate(_handle.DangerousGetHandle(), in nativeParams)
                .ThrowIfModelError(EntryPoints, _handle.DangerousGetHandle()); // F16 + model-scoped reason
        }
        finally
        {
            if (pathPtr != IntPtr.Zero) Marshal.FreeHGlobal(pathPtr);
            if (cachePtr != IntPtr.Zero) Marshal.FreeHGlobal(cachePtr);
        }
    }
    internal ModelConfig Config => _config;

    public string ApplyChatTemplate(string? template, IReadOnlyList<(string role, string content)> messages, bool addAssistant = true, int enableThinking = -1)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        // Marshal messages to pvx_chat_message_t struct array
        // pvx_chat_message_t is { const char* role; const char* content; } — blittable
        var nMsg = messages.Count;
        var msgSize = System.Runtime.InteropServices.Marshal.SizeOf<PvxChatMessage>();
        var msgPtr = System.Runtime.InteropServices.Marshal.AllocHGlobal(nMsg * msgSize);
        try
        {
            var allocatedStrings = new List<IntPtr>();
            try
            {
                for (int i = 0; i < nMsg; i++)
                {
                    var (role, content) = messages[i];
                    var rolePtr = PvxNative.AllocUTF8(role);
                    var contentPtr = PvxNative.AllocUTF8(content);
                    allocatedStrings.Add(rolePtr);
                    allocatedStrings.Add(contentPtr);
                    var msg = new PvxChatMessage { Role = rolePtr, Content = contentPtr };
                    System.Runtime.InteropServices.Marshal.StructureToPtr(msg, msgPtr + i * msgSize, false);
                }

                EntryPoints.ApplyChatTemplate(_handle.DangerousGetHandle(), template,
                    msgPtr, nMsg, addAssistant, enableThinking, out var textPtr).ThrowIfError(EntryPoints);
                try
                {
                    return System.Runtime.InteropServices.Marshal.PtrToStringUTF8(textPtr) ?? string.Empty; /* F15 */
                }
                finally { EntryPoints.FreeString(textPtr); }
            }
            finally
            {
                foreach (var p in allocatedStrings) System.Runtime.InteropServices.Marshal.FreeHGlobal(p);
            }
        }
        finally
        {
            System.Runtime.InteropServices.Marshal.FreeHGlobal(msgPtr);
        }
    }

    public IGrammar CreateGrammar(string grammarStr, string grammarRoot)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        return NativeGrammar.Create(this, grammarStr, grammarRoot);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _handle.Dispose();
    }
}