using ProvostXInference.Models;

namespace ProvostXInference.Abstractions;

public interface IConversation : IDisposable
{
    int TokenCount { get; }
    void Prompt(string text);
    void PromptTokens(int[] tokens);
    int Sample(SamplingConfig? config = null);
    int SampleWithGrammar(SamplingConfig? config, IGrammar grammar);
    void Rewind(int tokenCount);
    void Reset();
    /// <summary>Resets grammar sampler state — call once per generation before a grammar-constrained sample loop.</summary>
    void ResetGrammarState();
    void ShiftLeft(int tokenCount);
    IConversationState SaveState();
    void RestoreState(IConversationPool pool, IConversationState state);
    /// <summary>
    /// Serializes the conversation's FULL inference state (KV cells, counters,
    /// pending tokens, penalty window) into an opaque blob for suspend-to-disk.
    /// </summary>
    byte[] ExportCheckpoint();
    /// <summary>
    /// Restores a previously exported checkpoint INTO this (empty) conversation.
    /// </summary>
    void ImportCheckpoint(byte[] data);
    void PromptWithImages(string text, IVisionEncoder vision, params byte[][] images);

    // ── ABI v7: batched/pool-lane speculative decoding (INF BL-05 lane 2) ──
    /// <summary>Enable speculative decoding on THIS pooled conversation (one
    /// spec session per conversation; requires ≥2 seq slots on the owning
    /// context — the native side types the refusal). Capability-gated: the
    /// facade validates against the model self-report first; enabling still
    /// re-validates fail-closed natively (mode/path/capability/vocab).</summary>
    void EnableSpeculation(SpeculativeConfig config);
    /// <summary>Teardown of the conversation's spec session (idempotent; pool
    /// return and context teardown also tear it down natively).</summary>
    void DisableSpeculation();
    /// <summary>Live mirror of the active speculative config (null when no
    /// session is enabled on this conversation).</summary>
    SpeculativeConfig? ActiveSpeculation { get; }
    /// <summary>Speculative token loop over this conversation: identical
    /// contract to the executor trio — draft → target-verify → accept into the
    /// conversation's normal state; token-exact with the plain lane for
    /// identical params. Typed refusal (never a silent plain run) when no spec
    /// session is active.</summary>
    IReadOnlyList<int> GenerateSpeculative(SamplingConfig? config, int maxTokens, Action<int>? onToken = null);
    /// <summary>Spec-session acceptance telemetry for this conversation.</summary>
    SpeculationStats SpeculationStats();
}
