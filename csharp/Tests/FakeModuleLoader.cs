using System.Collections.ObjectModel;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.InteropServices;
using ProvostXInference.Interop;

namespace ProvostXInference.Tests;

/// <summary>
/// In-memory module loader: fake images hand out REAL executable function
/// pointers — one genuine managed stub PER DELEGATE TYPE (generated with
/// default-return bodies, cached per type) so GetDelegateForFunctionPointer
/// in PvxHandleEntryPoints binds exactly as it does against a real native
/// image. Probing pvx_abi_version runs managed stub code and returns the
// image's scripted value. Export provenance is recorded per image.
/// </summary>
internal sealed class FakeModuleLoader : IPvxModuleLoader
{
    /// <summary>One fake image: id + scripted ABI values + registration path.</summary>
    internal sealed class FakeImage
    {
        public string Id = "";
        public int AbiVersionValue = PvxNative.ExpectedAbiVersion; // lockstep with the binding
        public bool HasAbiVersion = true;
        public IntPtr Handle;
        internal int ScriptedAbi;
    }

    private readonly List<FakeImage> _images = new();
    private readonly Dictionary<string, FakeImage> _byPath = new();
    private readonly Dictionary<IntPtr, string> _lastExportImage = new();
    private readonly Dictionary<IntPtr, List<string>> _provenanceByImage = new();
    private readonly Dictionary<IntPtr, Dictionary<string, IntPtr>> _fpByExport = new();
    private IntPtr _next = new(0x50000);

    public IReadOnlyList<FakeImage> Images => _images;

    public FakeImage AddImage(string id, int abiVersion = PvxNative.ExpectedAbiVersion, bool hasAbiVersion = true, string? loadPath = null)
    {
        var image = new FakeImage
        {
            Id = id,
            AbiVersionValue = abiVersion,
            HasAbiVersion = hasAbiVersion,
            Handle = _next,
            ScriptedAbi = abiVersion
        };
        _next = IntPtr.Add(_next, 0x1000);
        _images.Add(image);
        _byPath[loadPath ?? $"fake-{id}"] = image;
        return image;
    }

    public bool WasFreed(IntPtr image) => _freed.Contains(image);
    private readonly HashSet<IntPtr> _freed = new();

    public string LastExportImage(IntPtr image) => _lastExportImage[image];

    /// <summary>All image ids that served an export for THIS image
    /// (provenance of every bound table entry).</summary>
    public IReadOnlyList<string> ExportProvenance(IntPtr image) =>
        _provenanceByImage.TryGetValue(image, out var ids) ? ids.ToList() : new List<string>();

    public IntPtr GetExport(IntPtr image, string exportName)
    {
        var fake = _images.FirstOrDefault(i => i.Handle.Equals(image))
            ?? throw new InvalidOperationException($"unknown fake image {image}");
        if (!_provenanceByImage.TryGetValue(image, out var ids))
        {
            ids = new List<string>();
            _provenanceByImage[image] = ids;
        }
        ids.Add(fake.Id);
        _lastExportImage[image] = fake.Id;
        if (exportName == "pvx_abi_version" && !fake.HasAbiVersion)
            return IntPtr.Zero;

        var table = _fpByExport.TryGetValue(image, out var t) ? t : _fpByExport[image] = new();
        if (!table.TryGetValue(exportName, out var fp))
        {
            fp = exportName == "pvx_abi_version"
                ? Marshal.GetFunctionPointerForDelegate(
                    (PvxAbiDelegates.PvxAbiVersionDelegate)(() => fake.ScriptedAbi))
                : StubFactory.PointerFor(exportName);
            table[exportName] = fp;
        }
        return fp;
    }

    public IntPtr Load(string path)
    {
        // Fake images are LOOKED UP BY PATH (the OS loader contract): each fake
        // registers under its load path at AddImage time.
        return _byPath.TryGetValue(path, out var image)
            ? image.Handle
            : IntPtr.Zero;
    }

    public bool Free(IntPtr image)
    {
        _freed.Add(image);
        return true;
    }
}

/// <summary>
/// Generates ONE real executable stub per delegate type (default-return
/// DynamicMethods, cached per type) — Bind&lt;T&gt; converts these pointers to T
/// exactly as it does with native export addresses. The ABI stub is special:
/// it is a rooted real delegate returning the scripted value.
/// Stateless utility — no mutable state beyond the process-wide type cache.
/// </summary>
internal static class StubFactory
{
    private static readonly object Gate = new();
    private static readonly Dictionary<Type, IntPtr> PointersByDelegateType = new();

    /// <summary>A pointer for the given export: resolved against the known
    /// export→delegate-type map of the ABI table (first matching delegate
    /// type whose Invoke signature the table entry uses).</summary>
    public static IntPtr PointerFor(string exportName)
    {
        // Any export binds to a REAL pointer of ITS delegate type; the table
        // in PvxHandleEntryPoints is bound with per-export types — serve a
        // per-type default stub keyed by the export's delegate type.
        var delegateType = PvxExportTypeMap.DelegateTypeOf(exportName);
        lock (Gate)
        {
            if (PointersByDelegateType.TryGetValue(delegateType, out var fp)) return fp;
            fp = Marshal.GetFunctionPointerForDelegate((Delegate)CreateStub(delegateType));
            PointersByDelegateType[delegateType] = fp;
            return fp;
        }
    }

    private static Delegate CreateStub(Type delegateType)
    {
        var invoke = delegateType.GetMethod("Invoke")!;
        var parameters = invoke.GetParameters().Select(p => p.ParameterType).ToArray();
        var returnType = invoke.ReturnType;
        var dm = new DynamicMethod($"pvx_stub_{delegateType.Name}", returnType, parameters,
            typeof(StubFactory).Module, true);
        var il = dm.GetILGenerator();
        if (returnType != typeof(void))
            EmitDefault(il, returnType);
        il.Emit(OpCodes.Ret);
        return dm.CreateDelegate(delegateType);
    }

    private static void EmitDefault(ILGenerator il, Type returnType)
    {
        if (returnType == typeof(bool)) { il.Emit(OpCodes.Ldc_I4_0); return; }
        if (returnType == typeof(int) || returnType.IsEnum) { il.Emit(OpCodes.Ldc_I4_0); return; }
        if (returnType == typeof(uint)) { il.Emit(OpCodes.Ldc_I4_0); return; }
        if (returnType == typeof(IntPtr)) { il.Emit(OpCodes.Ldc_I4_0); il.Emit(OpCodes.Conv_I); return; }
        if (returnType == typeof(float)) { il.Emit(OpCodes.Ldc_R4, 0f); return; }
        if (returnType == typeof(double)) { il.Emit(OpCodes.Ldc_R8, 0d); return; }
        if (returnType == typeof(long) || returnType == typeof(ulong)) { il.Emit(OpCodes.Ldc_I4_0); il.Emit(OpCodes.Conv_I8); return; }
        if (returnType.IsValueType) { var loc = il.DeclareLocal(returnType); il.Emit(OpCodes.Ldloca, loc); il.Emit(OpCodes.Initobj, returnType); il.Emit(OpCodes.Ldloc, loc); return; }
        il.Emit(OpCodes.Ldnull);
    }
}

/// <summary>Export name → delegate type map (mirrors the bind table in
/// PvxHandleEntryPoints). Stateless utility — no mutable state.</summary>
internal static class PvxExportTypeMap
{
    // Stateless utility — no mutable state.
    public static Type DelegateTypeOf(string exportName) => exportName switch
    {
        "pvx_abi_version" => typeof(PvxAbiDelegates.PvxAbiVersionDelegate),
        "pvx_result_str" => typeof(PvxAbiDelegates.PvxResultStrDelegate),
        "pvx_free_string" => typeof(PvxAbiDelegates.PvxFreeHandleDelegate),
        "pvx_last_error" => typeof(PvxAbiDelegates.PvxLastErrorDelegate),
        "pvx_model_last_error" => typeof(PvxAbiDelegates.PvxIntPtrHandleDelegate), // ABI v5
        "pvx_set_log_callback" => typeof(PvxAbiDelegates.PvxSetLogCallbackDelegate),
        "pvx_set_log_level" => typeof(PvxAbiDelegates.PvxSetLogLevelDelegate),
        "pvx_set_log_file" => typeof(PvxAbiDelegates.PvxSetLogFileDelegate),
        "pvx_load_model" => typeof(PvxAbiDelegates.PvxLoadModelDelegate),
        "pvx_free_model" => typeof(PvxAbiDelegates.PvxFreeHandleDelegate),
        "pvx_load_mmproj" => typeof(PvxAbiDelegates.PvxLoadMmprojDelegate),
        "pvx_free_mmproj" => typeof(PvxAbiDelegates.PvxFreeHandleDelegate),
        "pvx_model_partial_rewind_supported" => typeof(PvxAbiDelegates.PvxBoolHandleDelegate),
        "pvx_create_context" => typeof(PvxAbiDelegates.PvxCreateContextDelegate),
        "pvx_free_context" => typeof(PvxAbiDelegates.PvxFreeHandleDelegate),
        "pvx_context_size" => typeof(PvxAbiDelegates.PvxUintFromHandleDelegate),
        "pvx_context_seq_max" => typeof(PvxAbiDelegates.PvxUintFromHandleDelegate),
        "pvx_context_n_ctx_used" => typeof(PvxAbiDelegates.PvxIntTwoHandlesDelegate),
        "pvx_tokenize" => typeof(PvxAbiDelegates.PvxTokenizeDelegate),
        "pvx_free_tokens" => typeof(PvxAbiDelegates.PvxFreeHandleDelegate),
        "pvx_detokenize" => typeof(PvxAbiDelegates.PvxDetokenizeDelegate),
        "pvx_token_to_piece" => typeof(PvxAbiDelegates.PvxTokenToPieceDelegate),
        "pvx_token_is_eos" => typeof(PvxAbiDelegates.PvxBoolFromHandleAndIntDelegate),
        "pvx_repair_eog_metadata" => typeof(PvxAbiDelegates.PvxRepairEogDelegate),
        "pvx_executor_create" => typeof(PvxAbiDelegates.PvxCreateFromHandleDelegate),
        "pvx_executor_free" => typeof(PvxAbiDelegates.PvxFreeHandleDelegate),
        "pvx_executor_prompt" => typeof(PvxAbiDelegates.PvxPromptDelegate),
        "pvx_executor_prompt_tokens" => typeof(PvxAbiDelegates.PvxPromptTokensDelegate),
        "pvx_executor_infer" => typeof(PvxAbiDelegates.PvxDecodeDelegate),
        "pvx_executor_sample" => typeof(PvxAbiDelegates.PvxSampleDelegate),
        "pvx_executor_token_count" => typeof(PvxAbiDelegates.PvxIntFromHandleDelegate),
        "pvx_executor_rewind" => typeof(PvxAbiDelegates.PvxResultIntDelegate),
        "pvx_executor_reset" => typeof(PvxAbiDelegates.PvxResultNoArgsFromHandleDelegate),
        "pvx_executor_shift_left" => typeof(PvxAbiDelegates.PvxResultIntDelegate),
        "pvx_state_save" => typeof(PvxAbiDelegates.PvxSaveStateDelegate),
        "pvx_state_restore" => typeof(PvxAbiDelegates.PvxResultTwoHandlesDelegate),
        "pvx_state_free" => typeof(PvxAbiDelegates.PvxFreeHandleDelegate),
        "pvx_state_token_count" => typeof(PvxAbiDelegates.PvxIntFromHandleDelegate),
        "pvx_pool_create" => typeof(PvxAbiDelegates.PvxCreateFromHandleDelegate),
        "pvx_pool_free" => typeof(PvxAbiDelegates.PvxFreeHandleDelegate),
        "pvx_pool_lease" => typeof(PvxAbiDelegates.PvxCreateFromHandleDelegate),
        "pvx_pool_return" => typeof(PvxAbiDelegates.PvxResultTwoHandlesDelegate),
        "pvx_pool_available" => typeof(PvxAbiDelegates.PvxIntFromHandleDelegate),
        "pvx_pool_size" => typeof(PvxAbiDelegates.PvxUintFromHandleDelegate),
        "pvx_conversation_prompt" => typeof(PvxAbiDelegates.PvxPromptDelegate),
        "pvx_conversation_prompt_tokens" => typeof(PvxAbiDelegates.PvxPromptTokensDelegate),
        "pvx_conversation_sample" => typeof(PvxAbiDelegates.PvxSampleDelegate),
        "pvx_conversation_token_count" => typeof(PvxAbiDelegates.PvxIntFromHandleDelegate),
        "pvx_conversation_rewind" => typeof(PvxAbiDelegates.PvxResultIntDelegate),
        "pvx_conversation_reset" => typeof(PvxAbiDelegates.PvxResultNoArgsFromHandleDelegate),
        "pvx_conversation_grammar_reset" => typeof(PvxAbiDelegates.PvxResultNoArgsFromHandleDelegate),
        "pvx_conversation_save" => typeof(PvxAbiDelegates.PvxSaveStateDelegate),
        "pvx_conversation_restore" => typeof(PvxAbiDelegates.PvxConversationRestoreDelegate),
        "pvx_conversation_state_free" => typeof(PvxAbiDelegates.PvxFreeHandleDelegate),
        "pvx_conversation_export" => typeof(PvxAbiDelegates.PvxExportStateDelegate),
        "pvx_conversation_export_free" => typeof(PvxAbiDelegates.PvxFreeHandleDelegate),
        "pvx_conversation_import" => typeof(PvxAbiDelegates.PvxImportDelegate),
        "pvx_conversation_shift_left" => typeof(PvxAbiDelegates.PvxResultIntDelegate),
        "pvx_conversation_load_image" => typeof(PvxAbiDelegates.PvxLoadImageDelegate),
        "pvx_conversation_prompt_with_images" => typeof(PvxAbiDelegates.PvxPromptWithImagesDelegate),
        "pvx_infer" => typeof(PvxAbiDelegates.PvxDecodeDelegate),
        "pvx_get_embeddings" => typeof(PvxAbiDelegates.PvxGetEmbeddingsDelegate),
        "pvx_free_embeddings" => typeof(PvxAbiDelegates.PvxFreeHandleDelegate),
        "pvx_embedding_dim" => typeof(PvxAbiDelegates.PvxIntFromHandleDelegate),
        "pvx_mtmd_marker" => typeof(PvxAbiDelegates.PvxMtmdMarkerDelegate),
        "pvx_mtmd_marker_static" => typeof(PvxAbiDelegates.PvxIntPtrNoArgsDelegate),
        "pvx_mtmd_needs_non_causal" => typeof(PvxAbiDelegates.PvxBoolHandleDelegate),
        "pvx_kv_copy" => typeof(PvxAbiDelegates.PvxResultHandleIntIntDelegate),
        "pvx_kv_clear_seq" => typeof(PvxAbiDelegates.PvxResultIntDelegate),
        "pvx_kv_seq_tokens" => typeof(PvxAbiDelegates.PvxIntHandleIntDelegate),
        "pvx_set_causal_attn" => typeof(PvxAbiDelegates.PvxSetCausalAttnDelegate),
        "pvx_kv_flush" => typeof(PvxAbiDelegates.PvxResultIntDelegate),
        "pvx_check_anti_prompts" => typeof(PvxAbiDelegates.PvxCheckAntiPromptsDelegate),
        "pvx_probe_backend" => typeof(PvxAbiDelegates.PvxProbeBackendDelegate),
        "pvx_create_context_safe" => typeof(PvxAbiDelegates.PvxCreateContextSafeDelegate),
        "pvx_vram_available" => typeof(PvxAbiDelegates.PvxVramAvailableDelegate),
        "pvx_backend_probe_passed" => typeof(PvxAbiDelegates.PvxBoolHandleDelegate),
        "pvx_backend_name" => typeof(PvxAbiDelegates.PvxIntPtrHandleDelegate),
        "pvx_backend_probe_error" => typeof(PvxAbiDelegates.PvxIntPtrHandleDelegate),
        "pvx_backend_needs_kv_flush" => typeof(PvxAbiDelegates.PvxBoolHandleDelegate),
        "pvx_backend_gpu_layers_reduced" => typeof(PvxAbiDelegates.PvxBoolHandleDelegate),
        "pvx_backend_effective_gpu_layers" => typeof(PvxAbiDelegates.PvxIntFromHandleDelegate),
        "pvx_backend_original_gpu_layers" => typeof(PvxAbiDelegates.PvxIntFromHandleDelegate),
        "pvx_grammar_create" => typeof(PvxAbiDelegates.PvxGrammarCreateDelegate),
        "pvx_grammar_free" => typeof(PvxAbiDelegates.PvxFreeHandleDelegate),
        "pvx_executor_sample_grammar" => typeof(PvxAbiDelegates.PvxSampleGrammarDelegate),
        "pvx_conversation_sample_grammar" => typeof(PvxAbiDelegates.PvxSampleGrammarDelegate),
        "pvx_apply_chat_template" => typeof(PvxAbiDelegates.PvxApplyChatTemplateDelegate),
        "pvx_executor_prompt_with_images" => typeof(PvxAbiDelegates.PvxPromptWithImagesDelegate),
        "pvx_model_spec_capabilities" => typeof(PvxAbiDelegates.PvxModelSpecCapabilitiesDelegate),
        "pvx_model_spec_validate" => typeof(PvxAbiDelegates.PvxModelSpecValidateDelegate),
        "pvx_executor_spec_init" => typeof(PvxAbiDelegates.PvxExecutorSpecInitDelegate),
        "pvx_executor_spec_free" => typeof(PvxAbiDelegates.PvxExecutorSpecFreeDelegate),
        "pvx_executor_generate_spec" => typeof(PvxAbiDelegates.PvxExecutorGenerateSpecDelegate),
        "pvx_conversation_spec_init" => typeof(PvxAbiDelegates.PvxConversationSpecInitDelegate),
        "pvx_conversation_spec_free" => typeof(PvxAbiDelegates.PvxFreeHandleDelegate),
        "pvx_conversation_generate_spec" => typeof(PvxAbiDelegates.PvxConversationGenerateSpecDelegate),
        "pvx_conversation_spec_stats" => typeof(PvxAbiDelegates.PvxConversationSpecStatsDelegate),
        "pvx_grammar_from_json_schema" => typeof(PvxAbiDelegates.PvxGrammarFromJsonSchemaDelegate),
        // ── ABI v8 (wave 3): audio / TTS / rerank / lora / cvec / stats ──
        "pvx_mtmd_supports_audio" => typeof(PvxAbiDelegates.PvxMtmdSupportsAudioDelegate),
        "pvx_mtmd_audio_sample_rate" => typeof(PvxAbiDelegates.PvxMtmdAudioSampleRateDelegate),
        "pvx_executor_prompt_with_audio" => typeof(PvxAbiDelegates.PvxPromptWithAudioDelegate),
        "pvx_conversation_prompt_with_audio" => typeof(PvxAbiDelegates.PvxPromptWithAudioDelegate),
        "pvx_tts_session_create" => typeof(PvxAbiDelegates.PvxTtsSessionCreateDelegate),
        "pvx_tts_session_free" => typeof(PvxAbiDelegates.PvxFreeHandleDelegate),
        "pvx_tts_generate" => typeof(PvxAbiDelegates.PvxTtsGenerateDelegate),
        "pvx_free_audio_samples" => typeof(PvxAbiDelegates.PvxFreeHandleDelegate),
        "pvx_rerank" => typeof(PvxAbiDelegates.PvxRerankDelegate),
        "pvx_free_rerank_scores" => typeof(PvxAbiDelegates.PvxFreeHandleDelegate),
        "pvx_adapter_load" => typeof(PvxAbiDelegates.PvxAdapterLoadDelegate),
        "pvx_adapter_free" => typeof(PvxAbiDelegates.PvxFreeHandleDelegate),
        "pvx_context_set_lora" => typeof(PvxAbiDelegates.PvxContextSetLoraDelegate),
        "pvx_control_vector_load" => typeof(PvxAbiDelegates.PvxControlVectorLoadDelegate),
        "pvx_control_vector_free" => typeof(PvxAbiDelegates.PvxFreeHandleDelegate),
        "pvx_context_set_control_vector" => typeof(PvxAbiDelegates.PvxContextSetControlVectorDelegate),
        "pvx_executor_stats" => typeof(PvxAbiDelegates.PvxStatsDelegate),
        "pvx_conversation_stats" => typeof(PvxAbiDelegates.PvxStatsDelegate),
        "pvx_engine_telemetry" => typeof(PvxAbiDelegates.PvxEngineTelemetryDelegate),
        _ => throw new InvalidOperationException($"unknown export '{exportName}'")
    };
}
/// <summary>Shared builder: a fake loader with a metal-specialist image
/// registered under the scripted resolver's metal path.</summary>
internal static class EngineTestSupport
{
    public static FakeModuleLoader MetalLoader()
    {
        var loader = new FakeModuleLoader();
        loader.AddImage("fake-metal", loadPath: "/fake/runtimes/fake-rid/native/libpvx-metal.dylib");
        return loader;
    }
}
