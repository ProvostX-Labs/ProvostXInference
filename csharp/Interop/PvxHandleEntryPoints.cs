using System.Runtime.InteropServices;

namespace ProvostXInference.Interop;

/// <summary>
/// Per-handle dispatch tables: every pvx_* export of ONE loaded specialist
/// image resolved ONCE at construction into typed unmanaged-function-pointer
/// delegates; each method invokes its own pointer — no global DllImport state
/// is consulted per call (per-handle routing replaces name-based DllImport
/// binding; concurrent specialists in one process are collision-free because
/// ggml registries/symbols are per-image — BL-01 spike verdict GREEN).
/// Thread-safe: resolved delegate fields are read-only after construction.
/// </summary>
internal sealed class PvxHandleEntryPoints : IPvxEntryPoints
{
    private readonly PvxAbiDelegates.PvxResultStrDelegate _resultStr;
    private readonly PvxAbiDelegates.PvxFreeHandleDelegate _freeString;
    private readonly PvxAbiDelegates.PvxLastErrorDelegate _lastError;
    private readonly PvxAbiDelegates.PvxIntPtrHandleDelegate _modelLastError; // ABI v5
    private readonly PvxAbiDelegates.PvxSetLogCallbackDelegate _setLogCallback;
    private readonly PvxAbiDelegates.PvxSetLogLevelDelegate _setLogLevel;
    private readonly PvxAbiDelegates.PvxSetLogFileDelegate _setLogFile;
    private readonly PvxAbiDelegates.PvxLoadModelDelegate _loadModel;
    private readonly PvxAbiDelegates.PvxFreeHandleDelegate _freeModel;
    private readonly PvxAbiDelegates.PvxLoadMmprojDelegate _loadMmproj;
    private readonly PvxAbiDelegates.PvxFreeHandleDelegate _freeMmproj;
    private readonly PvxAbiDelegates.PvxBoolHandleDelegate _modelPartialRewindSupported;
    private readonly PvxAbiDelegates.PvxCreateContextDelegate _createContext;
    private readonly PvxAbiDelegates.PvxFreeHandleDelegate _freeContext;
    private readonly PvxAbiDelegates.PvxUintFromHandleDelegate _contextSize;
    private readonly PvxAbiDelegates.PvxUintFromHandleDelegate _contextSeqMax;
    private readonly PvxAbiDelegates.PvxIntTwoHandlesDelegate _contextNCtxUsed;
    private readonly PvxAbiDelegates.PvxTokenizeDelegate _tokenize;
    private readonly PvxAbiDelegates.PvxFreeHandleDelegate _freeTokens;
    private readonly PvxAbiDelegates.PvxDetokenizeDelegate _detokenize;
    private readonly PvxAbiDelegates.PvxTokenToPieceDelegate _tokenToPiece;
    private readonly PvxAbiDelegates.PvxBoolFromHandleAndIntDelegate _tokenIsEos;
    private readonly PvxAbiDelegates.PvxRepairEogDelegate _repairEogMetadata;
    private readonly PvxAbiDelegates.PvxCreateFromHandleDelegate _executorCreate;
    private readonly PvxAbiDelegates.PvxFreeHandleDelegate _executorFree;
    private readonly PvxAbiDelegates.PvxPromptDelegate _executorPrompt;
    private readonly PvxAbiDelegates.PvxPromptTokensDelegate _executorPromptTokens;
    private readonly PvxAbiDelegates.PvxDecodeDelegate _executorInfer;
    private readonly PvxAbiDelegates.PvxSampleDelegate _executorSample;
    private readonly PvxAbiDelegates.PvxIntFromHandleDelegate _executorTokenCount;
    private readonly PvxAbiDelegates.PvxResultIntDelegate _executorRewind;
    private readonly PvxAbiDelegates.PvxResultNoArgsFromHandleDelegate _executorReset;
    private readonly PvxAbiDelegates.PvxResultIntDelegate _executorShiftLeft;
    private readonly PvxAbiDelegates.PvxSaveStateDelegate _stateSave;
    private readonly PvxAbiDelegates.PvxResultTwoHandlesDelegate _stateRestore;
    private readonly PvxAbiDelegates.PvxFreeHandleDelegate _stateFree;
    private readonly PvxAbiDelegates.PvxIntFromHandleDelegate _stateTokenCount;
    private readonly PvxAbiDelegates.PvxCreateFromHandleDelegate _poolCreate;
    private readonly PvxAbiDelegates.PvxFreeHandleDelegate _poolFree;
    private readonly PvxAbiDelegates.PvxCreateFromHandleDelegate _poolLease;
    private readonly PvxAbiDelegates.PvxResultTwoHandlesDelegate _poolReturn;
    private readonly PvxAbiDelegates.PvxIntFromHandleDelegate _poolAvailable;
    private readonly PvxAbiDelegates.PvxUintFromHandleDelegate _poolSize;
    private readonly PvxAbiDelegates.PvxPromptDelegate _conversationPrompt;
    private readonly PvxAbiDelegates.PvxPromptTokensDelegate _conversationPromptTokens;
    private readonly PvxAbiDelegates.PvxSampleDelegate _conversationSample;
    private readonly PvxAbiDelegates.PvxIntFromHandleDelegate _conversationTokenCount;
    private readonly PvxAbiDelegates.PvxResultIntDelegate _conversationRewind;
    private readonly PvxAbiDelegates.PvxResultNoArgsFromHandleDelegate _conversationReset;
    private readonly PvxAbiDelegates.PvxResultNoArgsFromHandleDelegate _conversationGrammarReset;
    private readonly PvxAbiDelegates.PvxSaveStateDelegate _conversationSave;
    private readonly PvxAbiDelegates.PvxConversationRestoreDelegate _conversationRestore;
    private readonly PvxAbiDelegates.PvxFreeHandleDelegate _conversationStateFree;
    private readonly PvxAbiDelegates.PvxExportStateDelegate _conversationExport;
    private readonly PvxAbiDelegates.PvxFreeHandleDelegate _conversationExportFree;
    private readonly PvxAbiDelegates.PvxImportDelegate _conversationImport;
    private readonly PvxAbiDelegates.PvxResultIntDelegate _conversationShiftLeft;
    private readonly PvxAbiDelegates.PvxLoadImageDelegate _conversationLoadImage;
    private readonly PvxAbiDelegates.PvxPromptWithImagesDelegate _conversationPromptWithImages;
    private readonly PvxAbiDelegates.PvxDecodeDelegate _infer;
    private readonly PvxAbiDelegates.PvxGetEmbeddingsDelegate _getEmbeddings;
    private readonly PvxAbiDelegates.PvxFreeHandleDelegate _freeEmbeddings;
    private readonly PvxAbiDelegates.PvxIntFromHandleDelegate _embeddingDim;
    private readonly PvxAbiDelegates.PvxMtmdMarkerDelegate _mtmdMarker;
    private readonly PvxAbiDelegates.PvxIntPtrNoArgsDelegate _mtmdMarkerStatic;
    private readonly PvxAbiDelegates.PvxBoolHandleDelegate _mtmdNeedsNonCausal;
    private readonly PvxAbiDelegates.PvxResultHandleIntIntDelegate _kvCopy;
    private readonly PvxAbiDelegates.PvxResultIntDelegate _kvClearSeq;
    private readonly PvxAbiDelegates.PvxIntHandleIntDelegate _kvSeqTokens;
    private readonly PvxAbiDelegates.PvxSetCausalAttnDelegate _setCausalAttn;
    private readonly PvxAbiDelegates.PvxResultIntDelegate _kvFlush;
    private readonly PvxAbiDelegates.PvxCheckAntiPromptsDelegate _checkAntiPrompts;
    private readonly PvxAbiDelegates.PvxProbeBackendDelegate _probeBackend;
    private readonly PvxAbiDelegates.PvxCreateContextSafeDelegate _createContextSafe;
    private readonly PvxAbiDelegates.PvxVramAvailableDelegate _vramAvailable;
    private readonly PvxAbiDelegates.PvxBoolHandleDelegate _backendProbePassed;
    private readonly PvxAbiDelegates.PvxIntPtrHandleDelegate _backendName;
    private readonly PvxAbiDelegates.PvxIntPtrHandleDelegate _backendProbeError;
    private readonly PvxAbiDelegates.PvxBoolHandleDelegate _backendNeedsKvFlush;
    private readonly PvxAbiDelegates.PvxBoolHandleDelegate _backendGpuLayersReduced;
    private readonly PvxAbiDelegates.PvxIntFromHandleDelegate _backendEffectiveGpuLayers;
    private readonly PvxAbiDelegates.PvxIntFromHandleDelegate _backendOriginalGpuLayers;
    private readonly PvxAbiDelegates.PvxGrammarCreateDelegate _grammarCreate;
    private readonly PvxAbiDelegates.PvxFreeHandleDelegate _grammarFree;
    private readonly PvxAbiDelegates.PvxSampleGrammarDelegate _executorSampleGrammar;
    private readonly PvxAbiDelegates.PvxSampleGrammarDelegate _conversationSampleGrammar;
    private readonly PvxAbiDelegates.PvxApplyChatTemplateDelegate _applyChatTemplate;
    private readonly PvxAbiDelegates.PvxPromptWithImagesDelegate _executorPromptWithImages;
    private readonly PvxAbiDelegates.PvxModelSpecCapabilitiesDelegate _modelSpecCapabilities;
    private readonly PvxAbiDelegates.PvxModelSpecValidateDelegate _modelSpecValidate;
    private readonly PvxAbiDelegates.PvxExecutorSpecInitDelegate _executorSpecInit;
    private readonly PvxAbiDelegates.PvxExecutorSpecFreeDelegate _executorSpecFree;
    private readonly PvxAbiDelegates.PvxExecutorGenerateSpecDelegate _executorGenerateSpec;
    private readonly PvxAbiDelegates.PvxConversationSpecInitDelegate _conversationSpecInit; // ABI v7
    private readonly PvxAbiDelegates.PvxFreeHandleDelegate _conversationSpecFree; // ABI v7 (void)
    private readonly PvxAbiDelegates.PvxConversationGenerateSpecDelegate _conversationGenerateSpec; // ABI v7
    private readonly PvxAbiDelegates.PvxConversationSpecStatsDelegate _conversationSpecStats; // ABI v7
    private readonly PvxAbiDelegates.PvxGrammarFromJsonSchemaDelegate _grammarFromJsonSchema; // ABI v7
    private readonly PvxAbiDelegates.PvxAbiVersionDelegate _abiVersion;

    public PvxHandleEntryPoints(IntPtr image, IPvxModuleLoader loader)
    {
        if (image == IntPtr.Zero) throw new ArgumentNullException(nameof(image));
        _resultStr = Bind<PvxAbiDelegates.PvxResultStrDelegate>(loader, image, "pvx_result_str");
        _freeString = Bind<PvxAbiDelegates.PvxFreeHandleDelegate>(loader, image, "pvx_free_string");
        _lastError = Bind<PvxAbiDelegates.PvxLastErrorDelegate>(loader, image, "pvx_last_error");
        _modelLastError = Bind<PvxAbiDelegates.PvxIntPtrHandleDelegate>(loader, image, "pvx_model_last_error");
        _setLogCallback = Bind<PvxAbiDelegates.PvxSetLogCallbackDelegate>(loader, image, "pvx_set_log_callback");
        _setLogLevel = Bind<PvxAbiDelegates.PvxSetLogLevelDelegate>(loader, image, "pvx_set_log_level");
        _setLogFile = Bind<PvxAbiDelegates.PvxSetLogFileDelegate>(loader, image, "pvx_set_log_file");
        _loadModel = Bind<PvxAbiDelegates.PvxLoadModelDelegate>(loader, image, "pvx_load_model");
        _freeModel = Bind<PvxAbiDelegates.PvxFreeHandleDelegate>(loader, image, "pvx_free_model");
        _loadMmproj = Bind<PvxAbiDelegates.PvxLoadMmprojDelegate>(loader, image, "pvx_load_mmproj");
        _freeMmproj = Bind<PvxAbiDelegates.PvxFreeHandleDelegate>(loader, image, "pvx_free_mmproj");
        _modelPartialRewindSupported = Bind<PvxAbiDelegates.PvxBoolHandleDelegate>(loader, image, "pvx_model_partial_rewind_supported");
        _createContext = Bind<PvxAbiDelegates.PvxCreateContextDelegate>(loader, image, "pvx_create_context");
        _freeContext = Bind<PvxAbiDelegates.PvxFreeHandleDelegate>(loader, image, "pvx_free_context");
        _contextSize = Bind<PvxAbiDelegates.PvxUintFromHandleDelegate>(loader, image, "pvx_context_size");
        _contextSeqMax = Bind<PvxAbiDelegates.PvxUintFromHandleDelegate>(loader, image, "pvx_context_seq_max");
        _contextNCtxUsed = Bind<PvxAbiDelegates.PvxIntTwoHandlesDelegate>(loader, image, "pvx_context_n_ctx_used");
        _tokenize = Bind<PvxAbiDelegates.PvxTokenizeDelegate>(loader, image, "pvx_tokenize");
        _freeTokens = Bind<PvxAbiDelegates.PvxFreeHandleDelegate>(loader, image, "pvx_free_tokens");
        _detokenize = Bind<PvxAbiDelegates.PvxDetokenizeDelegate>(loader, image, "pvx_detokenize");
        _tokenToPiece = Bind<PvxAbiDelegates.PvxTokenToPieceDelegate>(loader, image, "pvx_token_to_piece");
        _tokenIsEos = Bind<PvxAbiDelegates.PvxBoolFromHandleAndIntDelegate>(loader, image, "pvx_token_is_eos");
        _repairEogMetadata = Bind<PvxAbiDelegates.PvxRepairEogDelegate>(loader, image, "pvx_repair_eog_metadata");
        _executorCreate = Bind<PvxAbiDelegates.PvxCreateFromHandleDelegate>(loader, image, "pvx_executor_create");
        _executorFree = Bind<PvxAbiDelegates.PvxFreeHandleDelegate>(loader, image, "pvx_executor_free");
        _executorPrompt = Bind<PvxAbiDelegates.PvxPromptDelegate>(loader, image, "pvx_executor_prompt");
        _executorPromptTokens = Bind<PvxAbiDelegates.PvxPromptTokensDelegate>(loader, image, "pvx_executor_prompt_tokens");
        _executorInfer = Bind<PvxAbiDelegates.PvxDecodeDelegate>(loader, image, "pvx_executor_infer");
        _executorSample = Bind<PvxAbiDelegates.PvxSampleDelegate>(loader, image, "pvx_executor_sample");
        _executorTokenCount = Bind<PvxAbiDelegates.PvxIntFromHandleDelegate>(loader, image, "pvx_executor_token_count");
        _executorRewind = Bind<PvxAbiDelegates.PvxResultIntDelegate>(loader, image, "pvx_executor_rewind");
        _executorReset = Bind<PvxAbiDelegates.PvxResultNoArgsFromHandleDelegate>(loader, image, "pvx_executor_reset");
        _executorShiftLeft = Bind<PvxAbiDelegates.PvxResultIntDelegate>(loader, image, "pvx_executor_shift_left");
        _stateSave = Bind<PvxAbiDelegates.PvxSaveStateDelegate>(loader, image, "pvx_state_save");
        _stateRestore = Bind<PvxAbiDelegates.PvxResultTwoHandlesDelegate>(loader, image, "pvx_state_restore");
        _stateFree = Bind<PvxAbiDelegates.PvxFreeHandleDelegate>(loader, image, "pvx_state_free");
        _stateTokenCount = Bind<PvxAbiDelegates.PvxIntFromHandleDelegate>(loader, image, "pvx_state_token_count");
        _poolCreate = Bind<PvxAbiDelegates.PvxCreateFromHandleDelegate>(loader, image, "pvx_pool_create");
        _poolFree = Bind<PvxAbiDelegates.PvxFreeHandleDelegate>(loader, image, "pvx_pool_free");
        _poolLease = Bind<PvxAbiDelegates.PvxCreateFromHandleDelegate>(loader, image, "pvx_pool_lease");
        _poolReturn = Bind<PvxAbiDelegates.PvxResultTwoHandlesDelegate>(loader, image, "pvx_pool_return");
        _poolAvailable = Bind<PvxAbiDelegates.PvxIntFromHandleDelegate>(loader, image, "pvx_pool_available");
        _poolSize = Bind<PvxAbiDelegates.PvxUintFromHandleDelegate>(loader, image, "pvx_pool_size");
        _conversationPrompt = Bind<PvxAbiDelegates.PvxPromptDelegate>(loader, image, "pvx_conversation_prompt");
        _conversationPromptTokens = Bind<PvxAbiDelegates.PvxPromptTokensDelegate>(loader, image, "pvx_conversation_prompt_tokens");
        _conversationSample = Bind<PvxAbiDelegates.PvxSampleDelegate>(loader, image, "pvx_conversation_sample");
        _conversationTokenCount = Bind<PvxAbiDelegates.PvxIntFromHandleDelegate>(loader, image, "pvx_conversation_token_count");
        _conversationRewind = Bind<PvxAbiDelegates.PvxResultIntDelegate>(loader, image, "pvx_conversation_rewind");
        _conversationReset = Bind<PvxAbiDelegates.PvxResultNoArgsFromHandleDelegate>(loader, image, "pvx_conversation_reset");
        _conversationGrammarReset = Bind<PvxAbiDelegates.PvxResultNoArgsFromHandleDelegate>(loader, image, "pvx_conversation_grammar_reset");
        _conversationSave = Bind<PvxAbiDelegates.PvxSaveStateDelegate>(loader, image, "pvx_conversation_save");
        _conversationRestore = Bind<PvxAbiDelegates.PvxConversationRestoreDelegate>(loader, image, "pvx_conversation_restore");
        _conversationStateFree = Bind<PvxAbiDelegates.PvxFreeHandleDelegate>(loader, image, "pvx_conversation_state_free");
        _conversationExport = Bind<PvxAbiDelegates.PvxExportStateDelegate>(loader, image, "pvx_conversation_export");
        _conversationExportFree = Bind<PvxAbiDelegates.PvxFreeHandleDelegate>(loader, image, "pvx_conversation_export_free");
        _conversationImport = Bind<PvxAbiDelegates.PvxImportDelegate>(loader, image, "pvx_conversation_import");
        _conversationShiftLeft = Bind<PvxAbiDelegates.PvxResultIntDelegate>(loader, image, "pvx_conversation_shift_left");
        _conversationLoadImage = Bind<PvxAbiDelegates.PvxLoadImageDelegate>(loader, image, "pvx_conversation_load_image");
        _conversationPromptWithImages = Bind<PvxAbiDelegates.PvxPromptWithImagesDelegate>(loader, image, "pvx_conversation_prompt_with_images");
        _infer = Bind<PvxAbiDelegates.PvxDecodeDelegate>(loader, image, "pvx_infer");
        _getEmbeddings = Bind<PvxAbiDelegates.PvxGetEmbeddingsDelegate>(loader, image, "pvx_get_embeddings");
        _freeEmbeddings = Bind<PvxAbiDelegates.PvxFreeHandleDelegate>(loader, image, "pvx_free_embeddings");
        _embeddingDim = Bind<PvxAbiDelegates.PvxIntFromHandleDelegate>(loader, image, "pvx_embedding_dim");
        _mtmdMarker = Bind<PvxAbiDelegates.PvxMtmdMarkerDelegate>(loader, image, "pvx_mtmd_marker");
        _mtmdMarkerStatic = Bind<PvxAbiDelegates.PvxIntPtrNoArgsDelegate>(loader, image, "pvx_mtmd_marker_static");
        _mtmdNeedsNonCausal = Bind<PvxAbiDelegates.PvxBoolHandleDelegate>(loader, image, "pvx_mtmd_needs_non_causal");
        _kvCopy = Bind<PvxAbiDelegates.PvxResultHandleIntIntDelegate>(loader, image, "pvx_kv_copy");
        _kvClearSeq = Bind<PvxAbiDelegates.PvxResultIntDelegate>(loader, image, "pvx_kv_clear_seq");
        _kvSeqTokens = Bind<PvxAbiDelegates.PvxIntHandleIntDelegate>(loader, image, "pvx_kv_seq_tokens");
        _setCausalAttn = Bind<PvxAbiDelegates.PvxSetCausalAttnDelegate>(loader, image, "pvx_set_causal_attn");
        _kvFlush = Bind<PvxAbiDelegates.PvxResultIntDelegate>(loader, image, "pvx_kv_flush");
        _checkAntiPrompts = Bind<PvxAbiDelegates.PvxCheckAntiPromptsDelegate>(loader, image, "pvx_check_anti_prompts");
        _probeBackend = Bind<PvxAbiDelegates.PvxProbeBackendDelegate>(loader, image, "pvx_probe_backend");
        _createContextSafe = Bind<PvxAbiDelegates.PvxCreateContextSafeDelegate>(loader, image, "pvx_create_context_safe");
        _vramAvailable = Bind<PvxAbiDelegates.PvxVramAvailableDelegate>(loader, image, "pvx_vram_available");
        _backendProbePassed = Bind<PvxAbiDelegates.PvxBoolHandleDelegate>(loader, image, "pvx_backend_probe_passed");
        _backendName = Bind<PvxAbiDelegates.PvxIntPtrHandleDelegate>(loader, image, "pvx_backend_name");
        _backendProbeError = Bind<PvxAbiDelegates.PvxIntPtrHandleDelegate>(loader, image, "pvx_backend_probe_error");
        _backendNeedsKvFlush = Bind<PvxAbiDelegates.PvxBoolHandleDelegate>(loader, image, "pvx_backend_needs_kv_flush");
        _backendGpuLayersReduced = Bind<PvxAbiDelegates.PvxBoolHandleDelegate>(loader, image, "pvx_backend_gpu_layers_reduced");
        _backendEffectiveGpuLayers = Bind<PvxAbiDelegates.PvxIntFromHandleDelegate>(loader, image, "pvx_backend_effective_gpu_layers");
        _backendOriginalGpuLayers = Bind<PvxAbiDelegates.PvxIntFromHandleDelegate>(loader, image, "pvx_backend_original_gpu_layers");
        _grammarCreate = Bind<PvxAbiDelegates.PvxGrammarCreateDelegate>(loader, image, "pvx_grammar_create");
        _grammarFree = Bind<PvxAbiDelegates.PvxFreeHandleDelegate>(loader, image, "pvx_grammar_free");
        _executorSampleGrammar = Bind<PvxAbiDelegates.PvxSampleGrammarDelegate>(loader, image, "pvx_executor_sample_grammar");
        _conversationSampleGrammar = Bind<PvxAbiDelegates.PvxSampleGrammarDelegate>(loader, image, "pvx_conversation_sample_grammar");
        _applyChatTemplate = Bind<PvxAbiDelegates.PvxApplyChatTemplateDelegate>(loader, image, "pvx_apply_chat_template");
        _executorPromptWithImages = Bind<PvxAbiDelegates.PvxPromptWithImagesDelegate>(loader, image, "pvx_executor_prompt_with_images");
        _modelSpecCapabilities = Bind<PvxAbiDelegates.PvxModelSpecCapabilitiesDelegate>(loader, image, "pvx_model_spec_capabilities");
        _modelSpecValidate = Bind<PvxAbiDelegates.PvxModelSpecValidateDelegate>(loader, image, "pvx_model_spec_validate");
        _executorSpecInit = Bind<PvxAbiDelegates.PvxExecutorSpecInitDelegate>(loader, image, "pvx_executor_spec_init");
        _executorSpecFree = Bind<PvxAbiDelegates.PvxExecutorSpecFreeDelegate>(loader, image, "pvx_executor_spec_free");
        _executorGenerateSpec = Bind<PvxAbiDelegates.PvxExecutorGenerateSpecDelegate>(loader, image, "pvx_executor_generate_spec");
        _conversationSpecInit = Bind<PvxAbiDelegates.PvxConversationSpecInitDelegate>(loader, image, "pvx_conversation_spec_init");
        _conversationSpecFree = Bind<PvxAbiDelegates.PvxFreeHandleDelegate>(loader, image, "pvx_conversation_spec_free");
        _conversationGenerateSpec = Bind<PvxAbiDelegates.PvxConversationGenerateSpecDelegate>(loader, image, "pvx_conversation_generate_spec");
        _conversationSpecStats = Bind<PvxAbiDelegates.PvxConversationSpecStatsDelegate>(loader, image, "pvx_conversation_spec_stats");
        _grammarFromJsonSchema = Bind<PvxAbiDelegates.PvxGrammarFromJsonSchemaDelegate>(loader, image, "pvx_grammar_from_json_schema");
        _abiVersion = Bind<PvxAbiDelegates.PvxAbiVersionDelegate>(loader, image, "pvx_abi_version");
    }

    /// <summary>Bind delegate-to-export (pin order in the ctor body keeps the
    /// resolver one-liner uniform; a missing export surfaces here as a clear
    /// EntryPointNotFoundException naming the export and not the whole module).</summary>
    private static T Bind<T>(IPvxModuleLoader loader, IntPtr image, string export) where T : Delegate
    {
        var ptr = loader.GetExport(image, export);
        if (ptr == IntPtr.Zero)
            throw new EntryPointNotFoundException($"pvxinference export not found: {export}");
        return Marshal.GetDelegateForFunctionPointer<T>(ptr);
    }

    // ── Result / errors ──

    public IntPtr ResultStr(PvxResult code) => _resultStr(code);
    public void FreeString(IntPtr str) => _freeString(str);
    public IntPtr LastError(IntPtr context) => _lastError(context);
    public IntPtr ModelLastError(IntPtr model) => _modelLastError(model); // ABI v5

    // ── Structured logging ──

    public void SetLogCallback(IntPtr callback) => _setLogCallback(callback);
    public void SetLogLevel(int level) => _setLogLevel(level);
    public void SetLogFile(string path) => _setLogFile(path);

    // ── Model ──

    public PvxResult LoadModel(in PvxModelParams parameters, out IntPtr model) => _loadModel(parameters, out model);
    public void FreeModel(IntPtr model) => _freeModel(model);
    public PvxResult LoadMmproj(IntPtr model, string mmprojPath, out IntPtr mmproj) => _loadMmproj(model, mmprojPath, out mmproj);
    public void FreeMmproj(IntPtr mmproj) => _freeMmproj(mmproj);
    public bool ModelPartialRewindSupported(IntPtr model) => _modelPartialRewindSupported(model);

    // ── Context ──

    public PvxResult CreateContext(IntPtr model, in PvxContextParams parameters, out IntPtr context) => _createContext(model, parameters, out context);
    public void FreeContext(IntPtr context) => _freeContext(context);
    public uint ContextSize(IntPtr context) => _contextSize(context);
    public uint ContextSeqMax(IntPtr context) => _contextSeqMax(context);
    public int ContextNCtxUsed(IntPtr context, IntPtr conversation) => _contextNCtxUsed(context, conversation);

    // ── Tokenization ──

    public PvxResult Tokenize(IntPtr context, string text, bool addBos, bool parseSpecial, out IntPtr tokens, out int count) => _tokenize(context, text, addBos, parseSpecial, out tokens, out count);
    public void FreeTokens(IntPtr tokens) => _freeTokens(tokens);
    public PvxResult Detokenize(IntPtr context, int[] tokens, int count, bool removeSpecial, out IntPtr outText) => _detokenize(context, tokens, count, removeSpecial, out outText);
    public PvxResult TokenToPiece(IntPtr context, int token, byte[] buffer, int bufferSize) => _tokenToPiece(context, token, buffer, bufferSize);
    public bool TokenIsEos(IntPtr context, int token) => _tokenIsEos(context, token);

    // ── Metadata tooling ──

    public PvxResult RepairEogMetadata(string inPath, string outPath, byte[] report, int reportSize) => _repairEogMetadata(inPath, outPath, report, reportSize);

    // ── Standard executor ──

    public PvxResult ExecutorCreate(IntPtr context, out IntPtr executor) => _executorCreate(context, out executor);
    public void ExecutorFree(IntPtr executor) => _executorFree(executor);
    public PvxResult ExecutorPrompt(IntPtr executor, string text) => _executorPrompt(executor, text);
    public PvxResult ExecutorPromptTokens(IntPtr executor, int[] tokens, int count) => _executorPromptTokens(executor, tokens, count);
    public PvxDecodeResult ExecutorInfer(IntPtr executor) => _executorInfer(executor);
    public PvxResult ExecutorSample(IntPtr executor, in PvxSamplingParams parameters, out int token) => _executorSample(executor, parameters, out token);
    public int ExecutorTokenCount(IntPtr executor) => _executorTokenCount(executor);
    public PvxResult ExecutorRewind(IntPtr executor, int tokenCount) => _executorRewind(executor, tokenCount);
    public PvxResult ExecutorReset(IntPtr executor) => _executorReset(executor);
    public PvxResult ExecutorShiftLeft(IntPtr executor, int tokenCount) => _executorShiftLeft(executor, tokenCount);

    // ── State save/restore ──

    public PvxResult StateSave(IntPtr executor, out IntPtr state) => _stateSave(executor, out state);
    public PvxResult StateRestore(IntPtr executor, IntPtr state) => _stateRestore(executor, state);
    public void StateFree(IntPtr state) => _stateFree(state);
    public int StateTokenCount(IntPtr state) => _stateTokenCount(state);

    // ── Conversation pool ──

    public PvxResult PoolCreate(IntPtr context, out IntPtr pool) => _poolCreate(context, out pool);
    public void PoolFree(IntPtr pool) => _poolFree(pool);
    public PvxResult PoolLease(IntPtr pool, out IntPtr conversation) => _poolLease(pool, out conversation);
    public PvxResult PoolReturn(IntPtr pool, IntPtr conversation) => _poolReturn(pool, conversation);
    public int PoolAvailable(IntPtr pool) => _poolAvailable(pool);
    public uint PoolSize(IntPtr pool) => _poolSize(pool);

    // ── Conversation ──

    public PvxResult ConversationPrompt(IntPtr conversation, string text) => _conversationPrompt(conversation, text);
    public PvxResult ConversationPromptTokens(IntPtr conversation, int[] tokens, int count) => _conversationPromptTokens(conversation, tokens, count);
    public PvxResult ConversationSample(IntPtr conversation, in PvxSamplingParams parameters, out int token) => _conversationSample(conversation, parameters, out token);
    public int ConversationTokenCount(IntPtr conversation) => _conversationTokenCount(conversation);
    public PvxResult ConversationRewind(IntPtr conversation, int tokenCount) => _conversationRewind(conversation, tokenCount);
    public PvxResult ConversationReset(IntPtr conversation) => _conversationReset(conversation);
    public PvxResult ConversationGrammarReset(IntPtr conversation) => _conversationGrammarReset(conversation);
    public PvxResult ConversationSave(IntPtr conversation, out IntPtr state) => _conversationSave(conversation, out state);
    public PvxResult ConversationRestore(IntPtr pool, IntPtr conversation, IntPtr state) => _conversationRestore(pool, conversation, state);
    public void ConversationStateFree(IntPtr state) => _conversationStateFree(state);
    public PvxResult ConversationExport(IntPtr conversation, out IntPtr buffer, out ulong size) => _conversationExport(conversation, out buffer, out size);
    public void ConversationExportFree(IntPtr buffer) => _conversationExportFree(buffer);
    public PvxResult ConversationImport(IntPtr conversation, byte[] data, ulong size) => _conversationImport(conversation, data, size);
    public PvxResult ConversationShiftLeft(IntPtr conversation, int tokenCount) => _conversationShiftLeft(conversation, tokenCount);
    public PvxResult ConversationLoadImage(IntPtr conversation, byte[] data, int size) => _conversationLoadImage(conversation, data, size);
    public PvxResult ConversationPromptWithImages(IntPtr conversation, IntPtr mmproj, string text, IntPtr[] imageData, int[] imageSizes, int nImages) => _conversationPromptWithImages(conversation, mmproj, text, imageData, imageSizes, nImages);

    // ── Batched inference ──

    public PvxDecodeResult Infer(IntPtr context) => _infer(context);

    // ── Embeddings ──

    public PvxResult GetEmbeddings(IntPtr model, IntPtr context, string text, out IntPtr embeddings, out int dim) => _getEmbeddings(model, context, text, out embeddings, out dim);
    public void FreeEmbeddings(IntPtr embeddings) => _freeEmbeddings(embeddings);
    public int EmbeddingDim(IntPtr model) => _embeddingDim(model);

    // ── Vision / MTMD ──

    public PvxResult MtmdMarker(out IntPtr outMarker) => _mtmdMarker(out outMarker);
    public IntPtr MtmdMarkerStatic() => _mtmdMarkerStatic();
    public bool MtmdNeedsNonCausal(IntPtr mmproj) => _mtmdNeedsNonCausal(mmproj);

    // ── KV cache manipulation ──

    public PvxResult KvCopy(IntPtr context, int srcSeqId, int dstSeqId) => _kvCopy(context, srcSeqId, dstSeqId);
    public PvxResult KvClearSeq(IntPtr context, int seqId) => _kvClearSeq(context, seqId);
    public int KvSeqTokens(IntPtr context, int seqId) => _kvSeqTokens(context, seqId);
    public PvxResult SetCausalAttn(IntPtr context, bool causal) => _setCausalAttn(context, causal);
    public PvxResult KvFlush(IntPtr context, int seqId) => _kvFlush(context, seqId);

    // ── Anti-prompt detection ──

    public int CheckAntiPrompts(string text, string[] antiPrompts, int count)
    {
        // string[] → UTF-8 ptr array marshaling is manual: GetDelegateForFunctionPointer
        // rejects LPArray ArraySubType=LPUTF8Str (LPStr/LPWStr/LPTStr only), and the ABI
        // contract is UTF-8 (see PvxNative.AllocUTF8).
        var ptrs = new IntPtr[antiPrompts.Length];
        try
        {
            for (var i = 0; i < antiPrompts.Length; i++) ptrs[i] = PvxNative.AllocUTF8(antiPrompts[i]);
            return _checkAntiPrompts(text, ptrs, count);
        }
        finally
        {
            foreach (var p in ptrs) if (p != IntPtr.Zero) Marshal.FreeHGlobal(p);
        }
    }

    // ── Backend safety ──

    public PvxResult ProbeBackend(IntPtr context) => _probeBackend(context);
    public PvxResult CreateContextSafe(ref IntPtr model, in PvxModelParams modelParams, in PvxContextParams ctxParams, int requestedGpuLayers, out IntPtr context) => _createContextSafe(ref model, modelParams, ctxParams, requestedGpuLayers, out context);
    public bool VramAvailable(long neededBytes) => _vramAvailable(neededBytes);
    public bool BackendProbePassed(IntPtr context) => _backendProbePassed(context);
    public IntPtr BackendName(IntPtr context) => _backendName(context);
    public IntPtr BackendProbeError(IntPtr context) => _backendProbeError(context);
    public bool BackendNeedsKvFlush(IntPtr context) => _backendNeedsKvFlush(context);
    public bool BackendGpuLayersReduced(IntPtr context) => _backendGpuLayersReduced(context);
    public int BackendEffectiveGpuLayers(IntPtr context) => _backendEffectiveGpuLayers(context);
    public int BackendOriginalGpuLayers(IntPtr context) => _backendOriginalGpuLayers(context);

    // ── Grammar (GBNF) ──

    public IntPtr GrammarCreate(IntPtr model, string grammarStr, string grammarRoot) => _grammarCreate(model, grammarStr, grammarRoot);
    public void GrammarFree(IntPtr grammar) => _grammarFree(grammar);
    public PvxResult ExecutorSampleGrammar(IntPtr executor, in PvxSamplingParams parameters, IntPtr grammar, out int token) => _executorSampleGrammar(executor, parameters, grammar, out token);
    public PvxResult ConversationSampleGrammar(IntPtr conversation, in PvxSamplingParams parameters, IntPtr grammar, out int token) => _conversationSampleGrammar(conversation, parameters, grammar, out token);

    // ── Chat template ──

    public PvxResult ApplyChatTemplate(IntPtr model, string? tmpl, IntPtr messages, int nMessages, bool addAssistant, int enableThinking, out IntPtr outText) => _applyChatTemplate(model, tmpl, messages, nMessages, addAssistant, enableThinking, out outText);

    // ── Vision on standard executor ──

    public PvxResult ExecutorPromptWithImages(IntPtr executor, IntPtr model, string text, IntPtr[] imageData, int[] imageSizes, int nImages) => _executorPromptWithImages(executor, model, text, imageData, imageSizes, nImages);

    // ── Speculative decoding (INF BL-05) ──

    public PvxResult ModelSpecCapabilities(IntPtr model, out PvxSpecCapabilities capabilities) => _modelSpecCapabilities(model, out capabilities);
    public PvxResult ModelSpecValidate(IntPtr model, in PvxSpecParams parameters) => _modelSpecValidate(model, parameters);
    public PvxResult ExecutorSpecInit(IntPtr executor, in PvxSpecParams parameters) => _executorSpecInit(executor, parameters);
    public void ExecutorSpecFree(IntPtr executor) => _executorSpecFree(executor);
    public PvxResult ExecutorGenerateSpec(IntPtr executor, in PvxSamplingParams parameters, int outMax, IntPtr onToken, IntPtr userData, out int generated) => _executorGenerateSpec(executor, parameters, outMax, onToken, userData, out generated);

    // ── ABI v7: batched/pool lane + JSON-schema converter ──
    public PvxResult ConversationSpecInit(IntPtr conversation, in PvxSpecParams parameters) => _conversationSpecInit(conversation, parameters);
    public void ConversationSpecFree(IntPtr conversation) => _conversationSpecFree(conversation);
    public PvxResult ConversationGenerateSpec(IntPtr conversation, in PvxSamplingParams parameters, int outMax, IntPtr onToken, IntPtr userData, out int generated) => _conversationGenerateSpec(conversation, parameters, outMax, onToken, userData, out generated);
    public PvxResult ConversationSpecStats(IntPtr conversation, out long drafted, out long accepted) => _conversationSpecStats(conversation, out drafted, out accepted);
    public PvxResult GrammarFromJsonSchema(string jsonSchema, string rootName, out IntPtr gbnf, out IntPtr error) => _grammarFromJsonSchema(jsonSchema, rootName, out gbnf, out error);

    // ── Handshake ──

    public int AbiVersion() => _abiVersion();
}