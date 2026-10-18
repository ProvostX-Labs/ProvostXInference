using ProvostXInference.Models;
using ProvostXInference.Abstractions;

namespace ProvostXInference.Abstractions;

public interface IInferenceContext : IDisposable
{
    uint ContextSize { get; }
    uint SeqMax { get; }
    IStandardExecutor CreateExecutor();
    IConversationPool CreatePool();
    InferResult InferAll();
    void KvCopy(int srcSeqId, int dstSeqId);
    void KvClearSeq(int seqId);
    void SetCausalAttention(bool causal);
    int[] Tokenize(string text, bool addBos = false, bool parseSpecial = true);
    string Detokenize(int[] tokens, bool removeSpecial = false);
    string TokenToPiece(int token);
    bool IsEos(int token);

    /// <summary>ABI v8: single-shot rerank scores for one query against N
    /// documents (RANK-pooling context required — typed refusal otherwise).
    /// Each pair scores on a fresh seq; KV-neutral.</summary>
    float[] Rerank(string query, IReadOnlyList<string> documents);

    /// <summary>ABI v8: attach (or detach — empty list) LoRA adapters on THIS
    /// live context. Weights-surface: KV-neutral, all-or-nothing, scales
    /// &gt; 0 (typed otherwise).</summary>
    void SetAdapters(IReadOnlyList<(ILoraAdapter adapter, float scale)> adapters);

    /// <summary>ABI v8: apply (or clear — null) a control vector on THIS live
    /// context. Weights-surface: KV-neutral. Layer range 0/0 = full 1..N;
    /// explicit out-of-range ends throw typed.</summary>
    void SetControlVector(IControlVector? vector, int layerStart = 0, int layerEnd = 0);
}
