using ProvostXInference.Abstractions;
using ProvostXInference.Models;

namespace ProvostXInference.Engines;

/// <summary>
/// GGUF lane ops view: the full native llama.cpp surface as properties/methods —
/// context management, KV-cache operations, sampling knobs, speculative knobs.
/// Wraps the existing wrapper-layer graph (model → context → pool → conversation);
/// the raw abstractions stay reachable for advanced callers.
/// </summary>
public interface IGgufEngineOps
{
    /// <summary>The loaded native model (raw abstraction escape hatch).</summary>
    IInferenceModel Model { get; }

    /// <summary>The live inference context (rebuilt by <see cref="RebuildContext"/>).</summary>
    IInferenceContext Context { get; }

    /// <summary>Sampling knobs — mutable, applied on every sample.</summary>
    SamplingConfig Sampling { get; set; }

    /// <summary>
    /// Context sizing knobs — mutable; changes take effect on the next
    /// <see cref="RebuildContext"/> (existing conversations keep the old context).
    /// </summary>
    ContextConfig ContextSettings { get; set; }

    /// <summary>
    /// Read-only live backend introspection (BL-02 seam surfaced as a
    /// property): per-context probe profile — backend name, probe pass/fail,
    /// probe error, KV-flush need, effective vs requested GPU layers. Each get
    /// re-probes the live context (cheap; state lives ON the context, ABI v4).
    /// Visibility, not control: mutation stays on the load-time knobs.
    /// </summary>
    BackendInfo Backend { get; }

    /// <summary>Rebuild the live context from <see cref="ContextSettings"/> (disposes the old).</summary>
    void RebuildContext();

    /// <summary>Lease a conversation from the context pool (Return via pool).</summary>
    IConversation LeaseConversation();

    /// <summary>The conversation pool (raw pool ops: Return/AvailableCount).</summary>
    IConversationPool Pool { get; }

    /// <summary>Create a standard executor on the live context.</summary>
    IStandardExecutor CreateExecutor();

    /// <summary>Native context ops: copy KV src→dst sequence.</summary>
    void KvCopy(int srcSeqId, int dstSeqId);

    /// <summary>Native context ops: clear one KV sequence.</summary>
    void KvClearSeq(int seqId);

    /// <summary>Toggle causal attention for the context (vision prefill needs non-causal).</summary>
    void SetCausalAttention(bool causal);

    /// <summary>Model template: prompt → templated text (native ApplyChatTemplate).</summary>
    string ApplyChatTemplate(string? template, IReadOnlyList<(string role, string content)> messages,
        bool addAssistant = true, int enableThinking = -1);

    /// <summary>ABI v9: tools-aware template application — the model's OWN embedded template
    /// renders tool docs in its trained format. Null/empty tools = legacy behavior.</summary>
    string ApplyChatTemplate(string? template, IReadOnlyList<(string role, string content)> messages,
        IReadOnlyList<InferenceToolSpec> tools, bool addAssistant = true, int enableThinking = -1);

    /// <summary>Tokenize on the context.</summary>
    int[] Tokenize(string text, bool addBos = false, bool parseSpecial = true);

    /// <summary>Detokenize on the context.</summary>
    string Detokenize(int[] tokens, bool removeSpecial = false);

    /// <summary>KV-cache save: export the conversation state (token count included).</summary>
    IConversationState SaveConversationState(IConversation conversation);

    /// <summary>KV-cache restore: bring a saved state back into a pool conversation.</summary>
    void RestoreConversationState(IConversationPool pool, IConversationState state);

    /// <summary>KV-cache portable export: raw checkpoint bytes.</summary>
    byte[] ExportConversationCheckpoint(IConversation conversation);

    /// <summary>KV-cache portable import: replace conversation memory from bytes.</summary>
    void ImportConversationCheckpoint(IConversation conversation, byte[] data);

    /// <summary>Whether this model's memory supports partial-sequence rewind (arch probed).</summary>
    bool PartialRewindSupported { get; }

    /// <summary>Speculative self-report: types supported for THIS loaded model.</summary>
    SpeculativeCapabilities SpeculativeCapabilities { get; }

    /// <summary>ABI v7: JSON Schema → GBNF grammar convenience (see
    /// IInferenceModel.CreateGrammarFromJsonSchema).</summary>
    IGrammar CreateGrammarFromJsonSchema(string jsonSchema, string? rootName = null);

    // ── ABI v7: speculative decoding on the batched/pool lane ──
    /// <summary>Capability-gated enable of speculation on ONE pooled
    /// conversation: the model self-report is validated first
    /// (ValidateSpeculation), then the conversation session is enabled (which
    /// re-validates fail-closed natively). Typed refusal — never a silent
    /// non-spec run.</summary>
    void EnableConversationSpeculation(IConversation conversation, SpeculativeConfig config);
    /// <summary>Teardown of a conversation's spec session (idempotent).</summary>
    void DisableConversationSpeculation(IConversation conversation);
    /// <summary>Per-conversation live mirror of the active speculative config
    /// (null when none).</summary>
    SpeculativeConfig? ActiveConversationSpeculation(IConversation conversation);
    /// <summary>Per-conversation spec acceptance telemetry (0/0 when none).</summary>
    SpeculationStats ConversationSpeculationStats(IConversation conversation);

    /// <summary>Typed pre-validation of a speculative config (mode/path/capability).</summary>
    void ValidateSpeculation(SpeculativeConfig config);

    // ── ABI v8 (wave 3): audio input / TTS / rerank / lora / cvec / stats ──

    /// <summary>Capability probe: loaded mmproj carries an audio adapter
    /// (false until a mmproj is loaded; vision-only projectors report
    /// false).</summary>
    bool SupportsAudioInput { get; }

    /// <summary>Native PCM rate the audio projector expects (-1 when
    /// unsupported). Product code never resamples.</summary>
    int AudioSampleRate { get; }

    /// <summary>Executor-lane prompt with ONE raw audio clip (PCM F32 mono at
    /// the projector's native rate; media marker in text). Typed refusal when
    /// the projector lacks an audio adapter.</summary>
    void PromptWithAudio(IStandardExecutor executor, string text, AudioChunk audio);

    /// <summary>Conversation-lane prompt with ONE raw audio clip.</summary>
    void PromptWithAudio(IConversation conversation, string text, AudioChunk audio);

    /// <summary>Single-shot rerank scores (RANK-pooling context required —
    /// set ContextSettings.PoolingType = Rank; typed refusal otherwise).
    /// KV-neutral (fresh seq per pair).</summary>
    float[] Rerank(string query, IReadOnlyList<string> documents);

    /// <summary>Load a LoRA adapter (arch-matched; typed refusal on
    /// file/arch mismatch).</summary>
    ILoraAdapter LoadLora(string path);

    /// <summary>Scoped attach: loads + attaches the adapter, returns a scope
    /// that detaches AND disposes on Dispose (weights-surface: KV-neutral,
    /// attach mid-conversation without rewind).</summary>
    IDisposable WithLora(string path, float scale);

    /// <summary>Detach all adapters on the live context (baseline logits
    /// return; KV untouched).</summary>
    void ClearLora();

    /// <summary>Load a control-vector direction file (n_embd-matched;
    /// typed refusal naming both dims).</summary>
    IControlVector LoadControlVector(string path);

    /// <summary>Scoped control-vector apply (layer range 0/0 = full);
    /// detaches on Dispose (KV-neutral).</summary>
    IDisposable WithControlVector(IControlVector vector, int layerStart = 0, int layerEnd = 0);

    /// <summary>Clear the applied control vector (baseline logits return).</summary>
    void ClearControlVector();

    /// <summary>Create a TTS session: a SEPARATE decode context on the
    /// backbone model + the gen-audio mmproj (KV-neutral by construction).
    /// Typed refusal (tts/gen-audio-capability) on a mmproj without a
    /// gen-audio pipeline.</summary>
    ITtsSession CreateTtsSession(string ttsMmprojPath, uint contextSize = 0);

    /// <summary>Stats snapshot of one executor lane.</summary>
    EngineStats GetExecutorStats(IStandardExecutor executor);

    /// <summary>Stats snapshot of one conversation lane.</summary>
    EngineStats GetConversationStats(IConversation conversation);

    /// <summary>ABI v9: model-wide telemetry snapshot for THIS engine instance
    /// (per-engine, never process-wide). Pull-based; derive rates between two
    /// snapshots in the caller.</summary>
    EngineTelemetry GetEngineTelemetry();
}
