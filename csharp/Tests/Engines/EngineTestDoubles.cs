using ProvostXInference.Abstractions;
using ProvostXInference.Models;
using ProvostXInference.Engines;

namespace ProvostXInference.Tests.Engines;

/// <summary>
/// Test doubles for the facade layer: in-memory IInferenceModel/context/
/// pool/conversation/vision-encoder + a recording IGgufModelLoader. No native
/// deps — unit-test the ENGINE OBJECT wiring, not llama.cpp.
/// </summary>

internal sealed class FakeEngineVisionEncoder : IVisionEncoder
{
    public bool Disposed { get; private set; }
    public string Marker => "[img]";
    public bool NeedsNonCausalAttention => false;
    public void Dispose() => Disposed = true;
}

internal sealed class FakeEngineGrammar : IGrammar
{
    public GrammarKind Kind { get; init; } = GrammarKind.Gbnf;
    public void Dispose() { }
}

internal sealed class FakeEngineConversationState : IConversationState
{
    public int TokenCount => 0;
    public void Dispose() { }
}

internal sealed class FakeEngineConversation : IConversation
{
    public List<string> Prompts { get; } = new();
    public List<byte[]> Imports { get; } = new();
    public bool Disposed { get; private set; }
    public int TokenCount => Prompts.Sum(p => p.Length);
    public void Prompt(string text) => Prompts.Add(text);
    public void PromptTokens(int[] tokens) => Prompts.Add(string.Join(',', tokens));
    public int Sample(SamplingConfig? config = null) => 7;
    public int SampleWithGrammar(SamplingConfig? config, IGrammar grammar) => 7;
    public void Rewind(int tokenCount) { }
    public void Reset() => Prompts.Clear();
    public void ResetGrammarState() { }
    public void ShiftLeft(int tokenCount) { }
    public IConversationState SaveState() => new FakeEngineConversationState();
    public void RestoreState(IConversationPool pool, IConversationState state) { }
    public byte[] ExportCheckpoint() => new byte[] { 1, 2, 3 };
    public void ImportCheckpoint(byte[] data) => Imports.Add(data);
    public void PromptWithImages(string text, IVisionEncoder vision, params byte[][] images) { }

    // ABI v7: batched/pool-lane spec doubles
    public SpeculativeConfig? ActiveSpeculation { get; private set; }
    public void EnableSpeculation(SpeculativeConfig config) => ActiveSpeculation = config;
    public void DisableSpeculation() => ActiveSpeculation = null;
    public IReadOnlyList<int> GenerateSpeculative(SamplingConfig? config, int maxTokens, Action<int>? onToken = null) =>
        new List<int>();
    public ProvostXInference.Models.SpeculationStats SpeculationStats() => ProvostXInference.Models.SpeculationStats.None;
    public void Dispose() => Disposed = true;
}

internal sealed class FakeEnginePool : IConversationPool
{
    public int LeaseCount { get; private set; }
    public int ReturnCount { get; private set; }
    public bool Disposed { get; private set; }
    public IConversation Lease() { LeaseCount++; return new FakeEngineConversation(); }
    public void Return(IConversation conversation) => ReturnCount++;
    public int AvailableCount => 0;
    public uint Size => 1;
    public void Dispose() => Disposed = true;
}

internal sealed class FakeEngineContext : IInferenceContext
{
    public List<ContextConfig> CreatedWith { get; } = new();
    public List<(int, int)> KvCopies { get; } = new();
    public List<int> ClearedSeqs { get; } = new();
    public bool Causal { get; private set; } = true;
    public bool Disposed { get; private set; }
    public IConversationPool PoolToCreate { get; set; } = new FakeEnginePool();

    public uint ContextSize => 0;
    public uint SeqMax => 4;
    public IStandardExecutor CreateExecutor() => throw new NotSupportedException();
    public IConversationPool CreatePool() => PoolToCreate;
    public InferResult InferAll() => InferResult.Ok;
    public void KvCopy(int srcSeqId, int dstSeqId) => KvCopies.Add((srcSeqId, dstSeqId));
    public void KvClearSeq(int seqId) => ClearedSeqs.Add(seqId);
    public void SetCausalAttention(bool causal) => Causal = causal;
    public int[] Tokenize(string text, bool addBos = false, bool parseSpecial = true) => new[] { 1, 2 };
    public string Detokenize(int[] tokens, bool removeSpecial = false) => "fake";
    public string TokenToPiece(int token) => "x";
    public bool IsEos(int token) => false;
    public void Dispose() => Disposed = true;
}

internal sealed class FakeEngineModel : IInferenceModel
{
    public List<ContextConfig> Contexts { get; } = new();
    public List<IVisionEncoder> VisionEncoders { get; } = new();
    public bool Disposed { get; private set; }
    public FakeEngineContext LastContext { get; private set; } = new();

    public IInferenceContext CreateContext(ContextConfig config)
    {
        Contexts.Add(config);
        LastContext = new FakeEngineContext { CreatedWith = { config } };
        return LastContext;
    }

    public float[] GetEmbeddings(IInferenceContext context, string text) => new[] { 0.5f, 0.25f };
    public int EmbeddingDimension => 2;
    public IVisionEncoder LoadVisionEncoder(string mmprojPath) =>
        new FakeEngineVisionEncoder();

    public string ApplyChatTemplate(string? template, IReadOnlyList<(string role, string content)> messages,
        bool addAssistant = true, int enableThinking = -1) =>
        string.Concat(messages.Select(m => $"[{m.role}] {m.content}"));

    public IGrammar CreateGrammar(string grammarStr, string grammarRoot) => new FakeEngineGrammar();
    public IGrammar CreateGrammarFromJsonSchema(string jsonSchema, string? rootName = null) =>
        new FakeEngineGrammar { Kind = GrammarKind.JsonSchema };
    public bool PartialRewindSupported => true;
    public SpeculativeCapabilities SpeculativeCapabilities => SpeculativeCapabilities.None;
    public void ValidateSpeculation(SpeculativeConfig config) { }
    public void Dispose() => Disposed = true;
}

/// <summary>Recording backend probe: serves BackendInfo to the Gguf view tests.</summary>
internal sealed class FakeEngineBackend : IInferenceBackend
{
    public List<IInferenceContext> ProbedContexts { get; } = new();
    public BackendInfo NextInfo { get; set; } = new BackendInfo
    {
        Name = "fake-metal", ProbePassed = true, ProbeError = null,
        NeedsKvFlush = false, GpuLayersReduced = false,
        EffectiveGpuLayers = 99, OriginalGpuLayers = 99,
    };

    public BackendInfo Probe(IInferenceContext context)
    {
        ProbedContexts.Add(context);
        return NextInfo;
    }

    public bool VramAvailable(long neededBytes) => true;
}

/// <summary>Recording GGUF loader: captures ModelConfigs, hands out fake models.</summary>
internal sealed class FakeGgufModelLoader : IGgufModelLoader
{
    public List<ModelConfig> Loaded { get; } = new();
    public List<(IInferenceModel, string)> Vision { get; } = new();
    public Func<IInferenceModel> ModelFactory { get; set; } = () => new FakeEngineModel();
    public Func<IInferenceModel, string, IVisionEncoder> VisionFactory { get; set; } =
        (m, p) => new FakeEngineVisionEncoder();

    public IInferenceModel Load(ModelConfig config)
    {
        Loaded.Add(config);
        return ModelFactory();
    }

    public IVisionEncoder LoadVisionEncoder(IInferenceModel model, string mmprojPath)
    {
        var vision = VisionFactory(model, mmprojPath);
        Vision.Add((model, mmprojPath));
        return vision;
    }
}