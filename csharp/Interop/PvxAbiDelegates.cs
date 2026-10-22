using System.Runtime.InteropServices;

namespace ProvostXInference.Interop;

/// <summary>
/// Per-handle ABI delegate set: unmanaged-function-pointer delegates for every
/// pvx_* export. These are resolved PER SPECIALIST IMAGE (one delegate instance
/// per loaded engine) instead of global DllImport binding — the BL-02 per-handle
/// dispatch tables. MarshalAs attributes mirror the P/Invoke declarations in
/// PvxNative exactly (UTF-8 strings, 1-byte bools, blittable structs).
/// All delegate types are nested in this one file on purpose: they form a single
/// cohesive ABI artifact — the managed mirror of the pvxinference export table —
/// and share no meaning outside the per-handle entry-point set.
/// Stateless utility — no mutable state.
/// </summary>
internal static class PvxAbiDelegates
{
    internal delegate PvxResult PvxResultNoArgsFromHandleDelegate(IntPtr handle);
    internal delegate bool PvxBoolHandleDelegate(IntPtr handle);
    internal delegate PvxResult PvxResultHandleIntIntDelegate(IntPtr handle, int first, int second);
    internal delegate int PvxIntHandleIntDelegate(IntPtr handle, int seqId);
    /// <summary>Delegate with the (pool, conversation, state) → PvxResult shape.</summary>
    internal delegate PvxResult PvxConversationRestoreDelegate(IntPtr pool, IntPtr conversation, IntPtr state);

    internal delegate int PvxAbiVersionDelegate();
    internal delegate IntPtr PvxResultStrDelegate(PvxResult code);
    internal delegate void PvxFreeHandleDelegate(IntPtr handle);
    internal delegate IntPtr PvxLastErrorDelegate(IntPtr context);
    internal delegate void PvxSetLogCallbackDelegate(IntPtr callback);
    internal delegate void PvxSetLogLevelDelegate(int level);
    internal delegate void PvxSetLogFileDelegate([MarshalAs(UnmanagedType.LPUTF8Str)] string path);
    internal delegate PvxResult PvxLoadModelDelegate(in PvxModelParams parameters, out IntPtr model);
    internal delegate PvxResult PvxLoadMmprojDelegate(IntPtr model, [MarshalAs(UnmanagedType.LPUTF8Str)] string mmprojPath, out IntPtr mmproj);
    internal delegate PvxResult PvxCreateContextDelegate(IntPtr model, in PvxContextParams parameters, out IntPtr context);
    internal delegate uint PvxUintFromHandleDelegate(IntPtr handle);
    internal delegate int PvxIntTwoHandlesDelegate(IntPtr first, IntPtr second);
    internal delegate PvxResult PvxTokenizeDelegate(IntPtr context, [MarshalAs(UnmanagedType.LPUTF8Str)] string text,
        [MarshalAs(UnmanagedType.U1)] bool addBos, [MarshalAs(UnmanagedType.U1)] bool parseSpecial,
        out IntPtr tokens, out int count);
    internal delegate PvxResult PvxDetokenizeDelegate(IntPtr context, int[] tokens, int count,
        [MarshalAs(UnmanagedType.U1)] bool removeSpecial, out IntPtr outText);
    internal delegate PvxResult PvxTokenToPieceDelegate(IntPtr context, int token, byte[] buffer, int bufferSize);
    internal delegate bool PvxBoolFromHandleAndIntDelegate(IntPtr handle, int token);
    internal delegate PvxResult PvxRepairEogDelegate([MarshalAs(UnmanagedType.LPStr)] string inPath,
        [MarshalAs(UnmanagedType.LPStr)] string outPath, byte[] report, int reportSize);
    internal delegate PvxResult PvxCreateFromHandleDelegate(IntPtr parent, out IntPtr handle);
    /// <summary>Delegate with the (handle, out IntPtr) → PvxResult shape (state save).</summary>
    internal delegate PvxResult PvxSaveStateDelegate(IntPtr handle, out IntPtr state);
    /// <summary>Delegate with the (handle, out IntPtr, out ulong) → PvxResult shape (export blob).</summary>
    internal delegate PvxResult PvxExportStateDelegate(IntPtr conversation, out IntPtr buffer, out ulong size);
    internal delegate PvxDecodeResult PvxDecodeDelegate(IntPtr handle);
    internal delegate PvxResult PvxSampleDelegate(IntPtr handle, in PvxSamplingParams parameters, out int token);
    internal delegate int PvxIntFromHandleDelegate(IntPtr handle);
    internal delegate PvxResult PvxPromptDelegate(IntPtr handle, [MarshalAs(UnmanagedType.LPUTF8Str)] string text);
    internal delegate PvxResult PvxPromptTokensDelegate(IntPtr handle, int[] tokens, int count);
    internal delegate PvxResult PvxPromptWithImagesDelegate(IntPtr handle, IntPtr companion,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string text, IntPtr[] imageData, int[] imageSizes, int nImages);
    internal delegate PvxResult PvxSampleGrammarDelegate(IntPtr handle, in PvxSamplingParams parameters, IntPtr grammar, out int token);
    internal delegate PvxResult PvxResultIntDelegate(IntPtr handle, int count);
    internal delegate PvxResult PvxResultTwoHandlesDelegate(IntPtr first, IntPtr second);
    internal delegate PvxResult PvxLoadImageDelegate(IntPtr conversation, byte[] data, int size);
    internal delegate PvxResult PvxImportDelegate(IntPtr conversation, byte[] data, ulong size);
    internal delegate PvxResult PvxGetEmbeddingsDelegate(IntPtr model, IntPtr context,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string text, out IntPtr embeddings, out int dim);
    internal delegate PvxResult PvxMtmdMarkerDelegate(out IntPtr outMarker);
    internal delegate IntPtr PvxIntPtrNoArgsDelegate();
    internal delegate bool PvxBoolNoArgsDelegate();
    internal delegate int PvxIntNoArgsDelegate();
    internal delegate IntPtr PvxIntPtrHandleDelegate(IntPtr handle);
    internal delegate PvxResult PvxSetCausalAttnDelegate(IntPtr context, [MarshalAs(UnmanagedType.U1)] bool causal);
    internal delegate int PvxCheckAntiPromptsDelegate([MarshalAs(UnmanagedType.LPUTF8Str)] string text,
        IntPtr[] antiPrompts, int count);
    internal delegate PvxResult PvxProbeBackendDelegate(IntPtr context);
    internal delegate PvxResult PvxCreateContextSafeDelegate(ref IntPtr model, in PvxModelParams modelParams,
        in PvxContextParams ctxParams, int requestedGpuLayers, out IntPtr context);
    internal delegate bool PvxVramAvailableDelegate(long neededBytes);
    internal delegate IntPtr PvxGrammarCreateDelegate(IntPtr model, [MarshalAs(UnmanagedType.LPUTF8Str)] string grammarStr,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string grammarRoot);
    internal delegate PvxResult PvxApplyChatTemplateDelegate(IntPtr model, [MarshalAs(UnmanagedType.LPUTF8Str)] string? tmpl,
        IntPtr messages, int nMessages, [MarshalAs(UnmanagedType.U1)] bool addAssistant, int enableThinking, out IntPtr outText);
    /// <summary>ABI v9: tools-aware chat template application (null tools = legacy parity).</summary>
    internal delegate PvxResult PvxApplyChatTemplateV2Delegate(IntPtr model, [MarshalAs(UnmanagedType.LPUTF8Str)] string? tmpl,
        IntPtr messages, int nMessages, IntPtr tools, int nTools,
        [MarshalAs(UnmanagedType.U1)] bool addAssistant, int enableThinking, out IntPtr outText);

    // ── Speculative decoding (INF BL-05) ──
    internal delegate PvxResult PvxModelSpecCapabilitiesDelegate(IntPtr model, out PvxSpecCapabilities capabilities);
    internal delegate PvxResult PvxModelSpecValidateDelegate(IntPtr model, in PvxSpecParams parameters);
    internal delegate PvxResult PvxExecutorSpecInitDelegate(IntPtr executor, in PvxSpecParams parameters);
    internal delegate void PvxExecutorSpecFreeDelegate(IntPtr executor);
    /// <summary>Managed mirror of pvx_spec_token_callback_t. Byte return: the
    /// C ABI expects a 1-byte C99 bool — byte marshals byte-exact in both
    /// directions (plain bool would marshal 4-byte).</summary>
    internal delegate byte PvxSpecTokenCallbackDelegate(int token, IntPtr userData);
    internal delegate PvxResult PvxExecutorGenerateSpecDelegate(IntPtr executor, in PvxSamplingParams parameters,
        int outMax, IntPtr onToken, IntPtr userData, out int generated);

    // ── ABI v7: batched/pool-lane speculative decoding ──
    internal delegate PvxResult PvxConversationSpecInitDelegate(IntPtr conversation, in PvxSpecParams parameters);
    internal delegate PvxResult PvxConversationGenerateSpecDelegate(IntPtr conversation, in PvxSamplingParams parameters,
        int outMax, IntPtr onToken, IntPtr userData, out int generated);
    /// <summary>Spec-session acceptance telemetry: cumulative drafted/accepted
    /// counters (native types 0/0 when no session is active).</summary>
    internal delegate PvxResult PvxConversationSpecStatsDelegate(IntPtr conversation, out long drafted, out long accepted);

    // ── ABI v7: JSON-schema → GBNF converter (pure string→string, KV-neutral) ──
    /// <summary>pvx_grammar_from_json_schema: schema + optional root rule name
    /// in, malloc'd GBNF + malloc'd typed boundary message out (both freed
    /// with pvx_free_string; out_error is set only on failure).</summary>
    internal delegate PvxResult PvxGrammarFromJsonSchemaDelegate(
        [MarshalAs(UnmanagedType.LPUTF8Str)] string jsonSchema,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string rootName,
        out IntPtr gbnf, out IntPtr error);
    // ── ABI v8: audio input ──
    internal delegate bool PvxMtmdSupportsAudioDelegate(IntPtr model);
    internal delegate int PvxMtmdAudioSampleRateDelegate(IntPtr model);
    /// <summary>Executor + conversation prompt-with-audio (PCM F32 mono).</summary>
    internal delegate PvxResult PvxPromptWithAudioDelegate(IntPtr handle, IntPtr model,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string text, float[] pcm, int nSamples);
    // ── ABI v8: TTS ──
    internal delegate PvxResult PvxTtsSessionCreateDelegate(IntPtr model,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string mmprojPath, uint contextSize, out IntPtr session);
    internal delegate void PvxTtsSessionFreeDelegate(IntPtr session);
    internal delegate PvxResult PvxTtsGenerateDelegate(IntPtr session, in PvxTtsParams parameters,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string text, out IntPtr pcm, out int nSamples, out int sampleRate);
    internal delegate void PvxFreeFloatArrayDelegate(IntPtr pcm);
    // ── ABI v8: rerank ──
    internal delegate PvxResult PvxRerankDelegate(IntPtr context,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string query, IntPtr[] documents, int nDocuments, out IntPtr scores);
    // ── ABI v8: LoRA adapters ──
    internal delegate PvxResult PvxAdapterLoadDelegate(IntPtr model,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string path, out IntPtr adapter);
    internal delegate void PvxAdapterFreeDelegate(IntPtr adapter);
    internal delegate PvxResult PvxContextSetLoraDelegate(IntPtr context, IntPtr[] adapters, float[] scales, int n);
    // ── ABI v8: control vectors ──
    internal delegate PvxResult PvxControlVectorLoadDelegate(IntPtr model,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string path, out IntPtr cvec);
    internal delegate void PvxControlVectorFreeDelegate(IntPtr cvec);
    internal delegate PvxResult PvxContextSetControlVectorDelegate(IntPtr context, IntPtr cvec, int layerStart, int layerEnd);
    // ── ABI v8: stats export ──
    internal delegate PvxResult PvxStatsDelegate(IntPtr handle, out PvxStats stats);
    // ── ABI v9: per-engine telemetry export ──
    internal delegate PvxResult PvxEngineTelemetryDelegate(IntPtr model, out PvxEngineTelemetry telemetry);
}
