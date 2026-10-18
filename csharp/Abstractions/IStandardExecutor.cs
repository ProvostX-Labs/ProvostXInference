using ProvostXInference.Models;

namespace ProvostXInference.Abstractions;

public interface IStandardExecutor : IDisposable
{
    int TokenCount { get; }
    void Prompt(string text);
    void PromptTokens(int[] tokens);
    void PromptWithImages(string text, IInferenceModel model, params byte[][] images);
    InferResult Infer();
    int Sample(SamplingConfig? config = null);
    int SampleWithGrammar(SamplingConfig? config, IGrammar grammar);
    void Rewind(int tokenCount);
    void Reset();
    void ShiftLeft(int tokenCount);
    IInferenceState SaveState();
    void RestoreState(IInferenceState state);

    // ── Speculative decoding (INF BL-05, sequential mode) ──

    /// <summary>Read-only live state: the speculative session this executor is
    /// currently running (as last enabled), or null when speculation is off.
    /// Mutations happen only via <see cref="EnableSpeculation"/>/
    /// <see cref="DisableSpeculation"/> — this is visibility, not control.</summary>
    SpeculativeConfig? ActiveSpeculation { get; }

    /// <summary>Enable a speculative session on this executor. Fail-closed: a
    /// config that cannot be honored (missing/unloadable draft model, missing
    /// capability, wrong vocab family) throws InferenceException naming the
    /// boundary — never a silent fallback to non-spec generation. Mode Off
    /// clears any existing session (disable).</summary>
    void EnableSpeculation(SpeculativeConfig config);

    /// <summary>Disable + tear down the speculative session (idempotent; Dispose
    /// also tears it down).</summary>
    void DisableSpeculation();

    /// <summary>Speculative generation over the executor's current prompt+history
    /// (draft → verify → commit until EOG or maxTokens). Returns the generated
    /// token ids in order; each token is also streamed to onToken when given.
    /// Requires an enabled session — typed error otherwise (never a silent
    /// non-spec run). Token-exact vs the plain Prompt/Infer/Sample loop for
    /// identical sampling params.</summary>
    IReadOnlyList<int> GenerateSpeculative(SamplingConfig? config, int maxTokens, Action<int>? onToken = null);

    /// <summary>ABI v8: prompt with ONE raw audio clip (PCM F32 mono at the
    /// projector's native rate — see IInferenceModel.AudioSampleRate). Same
    /// embd-row commit mechanics as images; typed refusal when the model's
    /// projector lacks an audio adapter.</summary>
    void PromptWithAudio(string text, IInferenceModel modelWithMmproj, AudioChunk audio);

    /// <summary>ABI v8: stats snapshot of this executor lane (see
    /// EngineStats).</summary>
    EngineStats GetStats();
}
