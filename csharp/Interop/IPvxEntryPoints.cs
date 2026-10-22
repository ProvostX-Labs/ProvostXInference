namespace ProvostXInference.Interop;

/// <summary>
/// Per-handle function table: typed managed entry points onto ONE loaded
/// specialist image. Every method routes through this image's own function
/// pointers (resolved at load time from its exports) — no global DllImport
/// state is consulted per call. This interface is the dispatch seam the whole
/// Implementation layer depends on (injected from the owning engine handle).
/// Error translation (PvxResult → InferenceException) lives in the shared
/// PvxResultExtensions.ThrowIfError(EntryPoints, ...) overload.
/// </summary>
internal interface IPvxEntryPoints
{
    // ── Result / errors ──
    IntPtr ResultStr(PvxResult code);
    void FreeString(IntPtr str);
    IntPtr LastError(IntPtr context);
    IntPtr ModelLastError(IntPtr model); // ABI v5: model-scoped typed-error reason

    // ── Structured logging ──
    void SetLogCallback(IntPtr callback);
    void SetLogLevel(int level);
    void SetLogFile(string path);

    // ── Model ──
    PvxResult LoadModel(in PvxModelParams parameters, out IntPtr model);
    void FreeModel(IntPtr model);
    PvxResult LoadMmproj(IntPtr model, string mmprojPath, out IntPtr mmproj);
    void FreeMmproj(IntPtr mmproj);
    bool ModelPartialRewindSupported(IntPtr model);

    // ── Context ──
    PvxResult CreateContext(IntPtr model, in PvxContextParams parameters, out IntPtr context);
    void FreeContext(IntPtr context);
    uint ContextSize(IntPtr context);
    uint ContextSeqMax(IntPtr context);
    int ContextNCtxUsed(IntPtr context, IntPtr conversation);

    // ── Tokenization ──
    PvxResult Tokenize(IntPtr context, string text, bool addBos, bool parseSpecial, out IntPtr tokens, out int count);
    void FreeTokens(IntPtr tokens);
    PvxResult Detokenize(IntPtr context, int[] tokens, int count, bool removeSpecial, out IntPtr outText);
    PvxResult TokenToPiece(IntPtr context, int token, byte[] buffer, int bufferSize);
    bool TokenIsEos(IntPtr context, int token);

    // ── Metadata tooling ──
    PvxResult RepairEogMetadata(string inPath, string outPath, byte[] report, int reportSize);

    // ── Standard executor ──
    PvxResult ExecutorCreate(IntPtr context, out IntPtr executor);
    void ExecutorFree(IntPtr executor);
    PvxResult ExecutorPrompt(IntPtr executor, string text);
    PvxResult ExecutorPromptTokens(IntPtr executor, int[] tokens, int count);
    PvxDecodeResult ExecutorInfer(IntPtr executor);
    PvxResult ExecutorSample(IntPtr executor, in PvxSamplingParams parameters, out int token);
    int ExecutorTokenCount(IntPtr executor);
    PvxResult ExecutorRewind(IntPtr executor, int tokenCount);
    PvxResult ExecutorReset(IntPtr executor);
    PvxResult ExecutorShiftLeft(IntPtr executor, int tokenCount);

    // ── State save/restore ──
    PvxResult StateSave(IntPtr executor, out IntPtr state);
    PvxResult StateRestore(IntPtr executor, IntPtr state);
    void StateFree(IntPtr state);
    int StateTokenCount(IntPtr state);

    // ── Conversation pool ──
    PvxResult PoolCreate(IntPtr context, out IntPtr pool);
    void PoolFree(IntPtr pool);
    PvxResult PoolLease(IntPtr pool, out IntPtr conversation);
    PvxResult PoolReturn(IntPtr pool, IntPtr conversation);
    int PoolAvailable(IntPtr pool);
    uint PoolSize(IntPtr pool);

    // ── Conversation ──
    PvxResult ConversationPrompt(IntPtr conversation, string text);
    PvxResult ConversationPromptTokens(IntPtr conversation, int[] tokens, int count);
    PvxResult ConversationSample(IntPtr conversation, in PvxSamplingParams parameters, out int token);
    int ConversationTokenCount(IntPtr conversation);
    PvxResult ConversationRewind(IntPtr conversation, int tokenCount);
    PvxResult ConversationReset(IntPtr conversation);
    PvxResult ConversationGrammarReset(IntPtr conversation);
    PvxResult ConversationSave(IntPtr conversation, out IntPtr state);
    PvxResult ConversationRestore(IntPtr pool, IntPtr conversation, IntPtr state);
    void ConversationStateFree(IntPtr state);
    PvxResult ConversationExport(IntPtr conversation, out IntPtr buffer, out ulong size);
    void ConversationExportFree(IntPtr buffer);
    PvxResult ConversationImport(IntPtr conversation, byte[] data, ulong size);
    PvxResult ConversationShiftLeft(IntPtr conversation, int tokenCount);
    PvxResult ConversationLoadImage(IntPtr conversation, byte[] data, int size);
    PvxResult ConversationPromptWithImages(IntPtr conversation, IntPtr mmproj, string text, IntPtr[] imageData, int[] imageSizes, int nImages);

    // ── Batched inference ──
    PvxDecodeResult Infer(IntPtr context);

    // ── Embeddings ──
    PvxResult GetEmbeddings(IntPtr model, IntPtr context, string text, out IntPtr embeddings, out int dim);
    void FreeEmbeddings(IntPtr embeddings);
    int EmbeddingDim(IntPtr model);

    // ── Vision / MTMD ──
    PvxResult MtmdMarker(out IntPtr outMarker);
    IntPtr MtmdMarkerStatic();
    bool MtmdNeedsNonCausal(IntPtr mmproj);

    // ── KV cache manipulation ──
    PvxResult KvCopy(IntPtr context, int srcSeqId, int dstSeqId);
    PvxResult KvClearSeq(IntPtr context, int seqId);
    int KvSeqTokens(IntPtr context, int seqId);
    PvxResult SetCausalAttn(IntPtr context, bool causal);
    PvxResult KvFlush(IntPtr context, int seqId);

    // ── Anti-prompt detection ──
    int CheckAntiPrompts(string text, string[] antiPrompts, int count);

    // ── Backend safety ──
    PvxResult ProbeBackend(IntPtr context);
    PvxResult CreateContextSafe(ref IntPtr model, in PvxModelParams modelParams, in PvxContextParams ctxParams, int requestedGpuLayers, out IntPtr context);
    bool VramAvailable(long neededBytes);
    // ABI v4 (audit 2026-10-17): backend probe state is PER-CONTEXT — queries
    // take the owning context; no process-global "last probe wins" state.
    bool BackendProbePassed(IntPtr context);
    IntPtr BackendName(IntPtr context);
    IntPtr BackendProbeError(IntPtr context);
    bool BackendNeedsKvFlush(IntPtr context);
    bool BackendGpuLayersReduced(IntPtr context);
    int BackendEffectiveGpuLayers(IntPtr context);
    int BackendOriginalGpuLayers(IntPtr context);

    // ── Grammar (GBNF) ──
    IntPtr GrammarCreate(IntPtr model, string grammarStr, string grammarRoot);
    void GrammarFree(IntPtr grammar);
    PvxResult ExecutorSampleGrammar(IntPtr executor, in PvxSamplingParams parameters, IntPtr grammar, out int token);
    PvxResult ConversationSampleGrammar(IntPtr conversation, in PvxSamplingParams parameters, IntPtr grammar, out int token);

    // ── Chat template ──
    PvxResult ApplyChatTemplate(IntPtr model, string? tmpl, IntPtr messages, int nMessages, bool addAssistant, int enableThinking, out IntPtr outText);

    // ── Vision on standard executor ──
    PvxResult ExecutorPromptWithImages(IntPtr executor, IntPtr model, string text, IntPtr[] imageData, int[] imageSizes, int nImages);

    // ── Speculative decoding (INF BL-05) ──
    PvxResult ModelSpecCapabilities(IntPtr model, out PvxSpecCapabilities capabilities);
    PvxResult ModelSpecValidate(IntPtr model, in PvxSpecParams parameters);
    PvxResult ExecutorSpecInit(IntPtr executor, in PvxSpecParams parameters);
    void ExecutorSpecFree(IntPtr executor);
    /// <summary>onToken/userData: raw pvx_spec_token_callback_t pointer (may be
    /// IntPtr.Zero); the caller owns the callback's lifetime for the call.</summary>
    PvxResult ExecutorGenerateSpec(IntPtr executor, in PvxSamplingParams parameters,
        int outMax, IntPtr onToken, IntPtr userData, out int generated);

    // ── ABI v7: batched/pool-lane speculative decoding ──
    PvxResult ConversationSpecInit(IntPtr conversation, in PvxSpecParams parameters);
    void ConversationSpecFree(IntPtr conversation);
    PvxResult ConversationGenerateSpec(IntPtr conversation, in PvxSamplingParams parameters,
        int outMax, IntPtr onToken, IntPtr userData, out int generated);
    PvxResult ConversationSpecStats(IntPtr conversation, out long drafted, out long accepted);

    // ── ABI v7: JSON-schema → GBNF converter (no handles — string→string) ──
    PvxResult GrammarFromJsonSchema(string jsonSchema, string rootName, out IntPtr gbnf, out IntPtr error);

    // ── ABI v8: audio input ──
    bool MtmdSupportsAudio(IntPtr model);
    int MtmdAudioSampleRate(IntPtr model);
    PvxResult ExecutorPromptWithAudio(IntPtr executor, IntPtr model, string text, float[] pcm, int nSamples);
    PvxResult ConversationPromptWithAudio(IntPtr conversation, IntPtr model, string text, float[] pcm, int nSamples);
    // ── ABI v8: TTS ──
    PvxResult TtsSessionCreate(IntPtr model, string ttsMmprojPath, uint contextSize, out IntPtr session);
    void TtsSessionFree(IntPtr session);
    PvxResult TtsGenerate(IntPtr session, in PvxTtsParams parameters, string text, out IntPtr pcm, out int nSamples, out int sampleRate);
    void FreeFloatArray(IntPtr pcm);
    // ── ABI v8: rerank ──
    PvxResult Rerank(IntPtr context, string query, IntPtr[] documents, int nDocuments, out IntPtr scores);
    void FreeRerankScores(IntPtr scores);
    // ── ABI v8: LoRA adapters ──
    PvxResult AdapterLoad(IntPtr model, string path, out IntPtr adapter);
    void AdapterFree(IntPtr adapter);
    PvxResult ContextSetLora(IntPtr context, IntPtr[] adapters, float[] scales, int n);
    // ── ABI v8: control vectors ──
    PvxResult ControlVectorLoad(IntPtr model, string path, out IntPtr cvec);
    void ControlVectorFree(IntPtr cvec);
    PvxResult ContextSetControlVector(IntPtr context, IntPtr cvec, int layerStart, int layerEnd);
    // ── ABI v8: stats export ──
    PvxResult ExecutorStats(IntPtr executor, out PvxStats stats);
    PvxResult ConversationStats(IntPtr conversation, out PvxStats stats);
    // ABI v9: model-wide telemetry snapshot (per-engine, never process-wide)
    PvxResult EngineTelemetry(IntPtr model, out PvxEngineTelemetry telemetry);

    // ── Handshake ──
    int AbiVersion();
}
