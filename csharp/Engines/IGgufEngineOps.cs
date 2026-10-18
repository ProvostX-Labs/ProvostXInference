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

    /// <summary>Typed pre-validation of a speculative config (mode/path/capability).</summary>
    void ValidateSpeculation(SpeculativeConfig config);
}