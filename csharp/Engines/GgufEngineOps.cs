using ProvostXInference.Abstractions;
using ProvostXInference.Models;

namespace ProvostXInference.Engines;

/// <summary>
/// GGUF lane ops implementation over the wrapper-layer graph. Holds the model,
/// the live context (created from the request's context settings), and the pool
/// (lazily created on first <see cref="LeaseConversation"/>). Disposal flows
/// outward-in via the owning <see cref="InferenceEngine"/>.
/// </summary>
public sealed class GgufEngineOps : IGgufEngineOps, IDisposable
{
    private readonly IInferenceModel _model;
    private readonly IInferenceBackend _backend;
    private IInferenceContext _context;
    private IConversationPool? _pool;

    public GgufEngineOps(IInferenceModel model, ContextConfig contextSettings, SamplingConfig sampling,
        IInferenceBackend backend)
    {
        _model = model;
        _backend = backend;
        ContextSettings = contextSettings;
        Sampling = sampling;
        _context = model.CreateContext(contextSettings);
    }

    public IInferenceModel Model => _model;
    public IInferenceContext Context => _context;
    public SamplingConfig Sampling { get; set; }
    public ContextConfig ContextSettings { get; set; }

    public void RebuildContext()
    {
        var oldPool = _pool;
        _pool = null;
        oldPool?.Dispose();
        _context.Dispose();
        _context = _model.CreateContext(ContextSettings);
    }

    public IConversationPool Pool => _pool ??= _context.CreatePool();

    public IConversation LeaseConversation() => Pool.Lease();

    public IStandardExecutor CreateExecutor() => _context.CreateExecutor();

    public void KvCopy(int srcSeqId, int dstSeqId) => _context.KvCopy(srcSeqId, dstSeqId);
    public void KvClearSeq(int seqId) => _context.KvClearSeq(seqId);
    public void SetCausalAttention(bool causal) => _context.SetCausalAttention(causal);

    public string ApplyChatTemplate(string? template, IReadOnlyList<(string role, string content)> messages,
        bool addAssistant = true, int enableThinking = -1) =>
        _model.ApplyChatTemplate(template, messages, addAssistant, enableThinking);

    public int[] Tokenize(string text, bool addBos = false, bool parseSpecial = true) =>
        _context.Tokenize(text, addBos, parseSpecial);

    public string Detokenize(int[] tokens, bool removeSpecial = false) =>
        _context.Detokenize(tokens, removeSpecial);

    public IConversationState SaveConversationState(IConversation conversation)
    {
        ArgumentNullException.ThrowIfNull(conversation);
        return conversation.SaveState();
    }

    public void RestoreConversationState(IConversationPool pool, IConversationState state)
    {
        ArgumentNullException.ThrowIfNull(pool);
        ArgumentNullException.ThrowIfNull(state);
        var conversation = pool.Lease();
        try { conversation.RestoreState(pool, state); }
        finally { pool.Return(conversation); }
    }

    public byte[] ExportConversationCheckpoint(IConversation conversation)
    {
        ArgumentNullException.ThrowIfNull(conversation);
        return conversation.ExportCheckpoint();
    }

    public void ImportConversationCheckpoint(IConversation conversation, byte[] data)
    {
        ArgumentNullException.ThrowIfNull(conversation);
        ArgumentNullException.ThrowIfNull(data);
        conversation.ImportCheckpoint(data);
    }

    /// <inheritdoc/>
    public BackendInfo Backend => _backend.Probe(_context);

    public bool PartialRewindSupported => _model.PartialRewindSupported;
    public SpeculativeCapabilities SpeculativeCapabilities => _model.SpeculativeCapabilities;
    public void ValidateSpeculation(SpeculativeConfig config) => _model.ValidateSpeculation(config);

    /// <summary>ABI v7: JSON Schema → GBNF through the model converter
    /// (typed refusals: parse position / unsupported keyword).</summary>
    public IGrammar CreateGrammarFromJsonSchema(string jsonSchema, string? rootName = null) =>
        _model.CreateGrammarFromJsonSchema(jsonSchema, rootName);

    // ── ABI v7: batched/pool-lane speculation (capability-gated facade) ──

    public void EnableConversationSpeculation(IConversation conversation, SpeculativeConfig config)
    {
        ArgumentNullException.ThrowIfNull(conversation);
        ArgumentNullException.ThrowIfNull(config);
        // Capability self-report first (typed refusal naming the boundary);
        // the conversation enable re-validates fail-closed natively anyway.
        _model.ValidateSpeculation(config);
        conversation.EnableSpeculation(config);
    }

    public void DisableConversationSpeculation(IConversation conversation)
    {
        ArgumentNullException.ThrowIfNull(conversation);
        conversation.DisableSpeculation();
    }

    public SpeculativeConfig? ActiveConversationSpeculation(IConversation conversation)
    {
        ArgumentNullException.ThrowIfNull(conversation);
        return conversation.ActiveSpeculation;
    }

    public SpeculationStats ConversationSpeculationStats(IConversation conversation)
    {
        ArgumentNullException.ThrowIfNull(conversation);
        return conversation.SpeculationStats();
    }

    /// <summary>Teardown owned by the engine: pool first, then context; model is NOT disposed here.</summary>
    public void Dispose()
    {
        _pool?.Dispose();
        _context.Dispose();
    }
}