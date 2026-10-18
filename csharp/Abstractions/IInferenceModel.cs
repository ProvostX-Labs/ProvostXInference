using ProvostXInference.Models;

namespace ProvostXInference.Abstractions;

public interface IInferenceModel : IDisposable
{
    IInferenceContext CreateContext(ContextConfig config);
    float[] GetEmbeddings(IInferenceContext context, string text);
    int EmbeddingDimension { get; }
    IVisionEncoder LoadVisionEncoder(string mmprojPath);
    /// <summary>
/// enableThinking: -1 = auto (model/template default), 0 = off, 1 = on.
/// When 0 and the vocabulary structurally contains a <think> special token
/// (single-token match), an empty think block is prefilled after the assistant
/// header (Qwen3 enable_thinking=false mechanism). Structural detection only.
/// </summary>
string ApplyChatTemplate(string? template, IReadOnlyList<(string role, string content)> messages, bool addAssistant = true, int enableThinking = -1);
    IGrammar CreateGrammar(string grammarStr, string grammarRoot);

    /// <summary>ABI v7: convert a JSON Schema document into a grammar
    /// consumable by the grammar-constrained samplers (pattern-port of the
    /// vendored JSON-schema→GBNF converter — never linked from product code).
    /// Fail-closed: invalid JSON → typed InferenceException naming the parse
    /// position; an unsupported schema keyword → typed InferenceException
    /// naming the keyword — never a loose best-effort grammar. KV-neutral
    /// (pure string→string conversion).</summary>
    IGrammar CreateGrammarFromJsonSchema(string jsonSchema, string? rootName = null);

    /// <summary>
    /// Whether the model's memory supports partial-sequence truncation (rewind to
    /// an arbitrary token boundary). Hybrid architectures (attention + recurrent
    /// layers, e.g. qwen35/qwen3next) and purely recurrent architectures CANNOT:
    /// their recurrent state is not preserved per token, so the native memory
    /// refuses partial seq_rm ranges (PVX_ERR_REWIND_UNSUPPORTED). KV retention /
    /// prefix-cache reuse via partial rewinds must be disabled for such models.
    /// Dense attention memories — including M-RoPE ones — support it.
    /// </summary>
    bool PartialRewindSupported => true;

    /// <summary>
    /// Speculative-decoding self-report (INF BL-05): the engine's supported-types
    /// list for THIS loaded model (SpeculativeType bitset derived from GGUF
    /// metadata: arch + nextn/MTP tensor presence). The LLM/dashboard offers
    /// exactly the reported types for sequential mode. No default: a fake or
    /// adapter must report honestly, never silently overpromise.
    /// </summary>
    SpeculativeCapabilities SpeculativeCapabilities { get; }

    /// <summary>
    /// Typed pre-validation of a speculative config against this model
    /// (mode/path/capability/vocab shape) without enabling anything. Enabling
    /// re-validates fail-closed anyway.
    /// </summary>
    void ValidateSpeculation(SpeculativeConfig config);

    /// <summary>ABI v8: does the loaded (shared) mmproj carry an audio
    /// adapter? Capability-truthful (false until a mmproj is loaded via
    /// LoadVisionEncoder; a vision-only projector reports false).</summary>
    bool SupportsAudioInput { get; }

    /// <summary>Native PCM sample rate the audio projector expects, or -1
    /// when audio is unsupported (product code never resamples).</summary>
    int AudioSampleRate { get; }

    /// <summary>ABI v8: load a LoRA adapter for THIS model (arch-matched;
    /// typed InferenceException on file/arch/rank mismatch naming the
    /// boundary). One adapter per record; the caller owns the disposable.</summary>
    ILoraAdapter CreateLoraAdapter(string path);

    /// <summary>ABI v8: load a control-vector direction file (direction.&lt;L&gt;
    /// F32 tensors) for THIS model (n_embd-matched, typed refusal naming both
    /// dims on mismatch).</summary>
    IControlVector CreateControlVector(string path);

    /// <summary>ABI v8: create a TTS session (separate decode context on this
    /// backbone model + the given gen-audio mmproj). Typed refusal when the
    /// mmproj has no gen-audio pipeline.</summary>
    ITtsSession CreateTtsSession(string ttsMmprojPath, uint contextSize = 0);
}
