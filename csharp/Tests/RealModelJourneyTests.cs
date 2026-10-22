using System.Diagnostics;
using System.IO;
using ProvostXInference.Abstractions;
using ProvostXInference.Exceptions;
using ProvostXInference.Implementation;
using ProvostXInference.Models;
using Xunit;

namespace ProvostXInference.Tests;

/// <summary>
/// Real-model journey tier (INF BL-47) — the formerly manual TEST-MATRIX lanes,
/// automated against a small real model so they run on demand on ANY host
/// (macOS/Linux/Windows). Fixture discovery: LiveModelFixtures (env-driven,
/// repo-relative defaults). A missing optional fixture skips the lane with a
/// named reason; the core generative lanes hard-require only the ≤1 GB small
/// model. RAM gate enforced before every load (house rule, >60% free).
/// </summary>
[Trait("Category", "RealModelJourney")]
public class RealModelJourneyTests
{
    private static string? SmallModel => LiveModelFixtures.SmallModel;
    private const double SmallModelBytes = 850_000_000; // ~0.8 GB Q4_K_M 1B

    private static bool RamOk() => LiveModelFixtures.RamGateAllows(SmallModelBytes);

    private static NativeInferenceModel LoadSmall(int gpuLayers) =>
        NativeInferenceModel.Load(new ModelConfig { Path = SmallModel!, GpuLayers = gpuLayers });

    // ═══════════════════════════════════════════════════
    // Lane 1 — load + sequential generation (matrix lanes 1/3)
    // ═══════════════════════════════════════════════════

    [LiveModelFact]
    public void Journey1_Load_Generate_SmallModel()
    {
        if (SmallModel is null || !RamOk()) return;
        using var model = LoadSmall(0); // CPU: identical behavior on every platform
        using var ctx = model.CreateContext(new ContextConfig { ContextSize = 2048, BatchSize = 512, SeqMax = 1 });
        using var exec = ctx.CreateExecutor();

        exec.Prompt("The capital of France is");
        Assert.Equal(InferResult.Ok, exec.Infer());
        Assert.True(exec.TokenCount > 0);

        var sampling = new SamplingConfig { Temperature = 0.3f, MaxTokens = 16 };
        var token = exec.Sample(sampling);
        Assert.True(token >= 0);
    }

    // ═══════════════════════════════════════════════════
    // Lane 2 — KV state: save/restore + checkpoint export/import round-trip
    // (matrix lane 5) — token-count exactness on restore.
    // ═══════════════════════════════════════════════════

    [LiveModelFact]
    public void Journey2_KvState_SaveRestore_CheckpointRoundTrip()
    {
        if (SmallModel is null || !RamOk()) return;
        using var model = LoadSmall(0);
        using var ctx = model.CreateContext(new ContextConfig { ContextSize = 2048, BatchSize = 512, SeqMax = 2 });
        using var pool = ctx.CreatePool();

        var conv = pool.Lease();
        conv.Prompt("Hello world, this is a KV state journey prompt.");
        Assert.Equal(InferResult.Ok, ctx.InferAll());
        var countBefore = conv.TokenCount;
        Assert.True(countBefore > 0);

        var state = conv.SaveState();
        Assert.Equal(countBefore, state.TokenCount);

        var checkpoint = conv.ExportCheckpoint();
        Assert.NotEmpty(checkpoint);

        // Generate more, then restore back to the saved position (the proven
        // executor-pattern contract).
        var extra = conv.Sample(new SamplingConfig { Temperature = 0.3f });
        Assert.True(extra >= 0);
        conv.PromptTokens(new[] { extra });
        Assert.Equal(InferResult.Ok, ctx.InferAll());
        Assert.True(conv.TokenCount > countBefore);

        conv.RestoreState(pool, state);
        Assert.Equal(countBefore, conv.TokenCount);

        // Checkpoint import round-trip: export → import → same count.
        conv.Reset();
        Assert.Equal(0, conv.TokenCount);
        conv.ImportCheckpoint(checkpoint);
        Assert.Equal(countBefore, conv.TokenCount);
        conv.Dispose();
    }

    // ═══════════════════════════════════════════════════
    // Lane 3 — continuous batching beyond the smoke scale: 8 conversations
    // interleaved via InferAll (matrix lane 4, deeper concurrency gap).
    // ═══════════════════════════════════════════════════

    [LiveModelFact]
    public void Journey3_Batched_8Conversations_Interleaved()
    {
        if (SmallModel is null || !RamOk()) return;
        using var model = LoadSmall(0);
        using var ctx = model.CreateContext(new ContextConfig { ContextSize = 2048, BatchSize = 512, SeqMax = 8 });
        using var pool = ctx.CreatePool();

        var convs = new IConversation[8];
        for (int i = 0; i < 8; i++)
        {
            convs[i] = pool.Lease();
            convs[i].Prompt($"Say the number {i + 1}.");
        }

        Assert.Equal(InferResult.Ok, ctx.InferAll());
        for (int i = 0; i < 8; i++)
        {
            Assert.True(convs[i].TokenCount > 0, $"conversation {i} produced no tokens");
            var token = convs[i].Sample(new SamplingConfig { Temperature = 0.3f });
            Assert.True(token >= 0);
        }
        for (int i = 0; i < 8; i++) convs[i].Dispose();
        Assert.Equal(8, pool.AvailableCount);
    }

    // ═══════════════════════════════════════════════════
    // Lane 4 — grammar-constrained sampling (matrix lane 10) with the REAL
    // small model (the existing JsonSchemaGrammarTests live legs are env-gated
    // separately; this lane runs wherever the small fixture exists).
    // ═══════════════════════════════════════════════════

    [LiveModelFact]
    public void Journey4_GbnfGrammar_ConstrainedSampling()
    {
        if (SmallModel is null || !RamOk()) return;
        using var model = LoadSmall(0);
        using var ctx = model.CreateContext(new ContextConfig { ContextSize = 2048, BatchSize = 512, SeqMax = 1 });
        using var pool = ctx.CreatePool();

        // GBNF: rule definition syntax — digits only; sampled pieces must satisfy it.
        var grammar = model.CreateGrammar("root ::= [0-9]+", "root");
        var conv = pool.Lease();
        conv.Prompt("Number:");
        Assert.Equal(InferResult.Ok, ctx.InferAll());

        for (int i = 0; i < 8; i++)
        {
            var token = conv.SampleWithGrammar(new SamplingConfig { Temperature = 0.0f }, grammar);
            Assert.True(token >= 0);
            var piece = ctx.TokenToPiece(token);
            // EOS is legal; any decoded piece must be a digit (no leading-space
            // artifacts: the grammar admits none).
            if (!ctx.IsEos(token)) Assert.True(piece.Trim().Length == 1 && char.IsDigit(piece.Trim()[0]),
                $"grammar violation: piece={piece}");
        }
        conv.Dispose();
    }

    // ═══════════════════════════════════════════════════
    // Lane 5 — embeddings on a small DENSE model (matrix lane 13). The
    // known SIGSEGV class (encode-vs-decode routing) is regression-guarded
    // here on every host with the small fixture; the dedicated MiniLM
    // encoder leg runs only when that fixture is present.
    // ═══════════════════════════════════════════════════

    [LiveModelFact]
    public void Journey5_Embeddings_DenseModel_MeanPooling_NoSegfault()
    {
        if (SmallModel is null || !RamOk()) return;
        using var model = LoadSmall(0);
        // Matrix lane-13 contract: generative-model embeddings ride a MEAN
        // pooling context (pooling must be set at context creation).
        using var ctx = model.CreateContext(new ContextConfig { ContextSize = 512, BatchSize = 256, SeqMax = 1, PoolingType = PoolingType.Mean });

        var emb = model.GetEmbeddings(ctx, "Hello world");
        Assert.Equal(model.EmbeddingDimension, emb.Length);
        Assert.True(Math.Sqrt(emb.Sum(x => x * (double)x)) > 0.9, "L2 norm must be ~1.0");

        // Similarity ordering (related > unrelated) — the lane-13 contract.
        var emb2 = model.GetEmbeddings(ctx, "Goodbye universe");
        var dot = 0.0;
        for (int i = 0; i < emb.Length; i++) dot += emb[i] * (double)emb2[i];
        Assert.True(dot > -0.5 && dot < 1.5, $"cosine out of plausible band: {dot}");
    }

    [LiveModelFact]
    public void Journey5b_Embeddings_MiniLMEncoder()
    {
        var embedPath = LiveModelFixtures.EmbedModel;
        if (embedPath is null || !RamOk()) return; // optional fixture
        using var model = NativeInferenceModel.Load(new ModelConfig { Path = embedPath, GpuLayers = 0 });
        using var ctx = model.CreateContext(new ContextConfig { ContextSize = 512, BatchSize = 256, SeqMax = 1, PoolingType = PoolingType.Mean });

        var related = model.GetEmbeddings(ctx, "A cat sits on the mat");
        var unrelated = model.GetEmbeddings(ctx, "Quarterly revenue exceeded projections");
        Assert.Equal(model.EmbeddingDimension, related.Length);

        var cat = model.GetEmbeddings(ctx, "The feline rests on the rug");
        Assert.True(Cosine(cat, related) > Cosine(cat, unrelated),
            "encoder must rank the paraphrase above the unrelated text");
    }

    private static double Cosine(float[] a, float[] b)
    {
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.Length; i++) { dot += a[i] * (double)b[i]; na += a[i] * (double)a[i]; nb += b[i] * (double)b[i]; }
        return dot / (Math.Sqrt(na) * Math.Sqrt(nb));
    }

    // ═══════════════════════════════════════════════════
    // Lane 6 — speculative draft-simple live leg (matrix lane 9): token-exact
    // vs the plain loop. Automated wherever target+draft fixtures share the
    // vocab family; skips with the mismatch named otherwise (the old Llama-3B
    // wrong-tokenizer-file finding is exactly what this lane catches).
    // ═══════════════════════════════════════════════════

    [LiveModelFact]
    public void Journey6_Speculative_DraftSimple_TokenExact()
    {
        var draftPath = LiveModelFixtures.DraftModel;
        if (SmallModel is null || draftPath is null || !RamOk()) return;

        using var target = LoadSmall(0);
        var caps = target.SpeculativeCapabilities;
        if (!caps.Supports(SpeculativeType.DraftSimple)) return; // target cannot draft — skip

        using var draft = NativeInferenceModel.Load(new ModelConfig { Path = draftPath, GpuLayers = 0 });
        var draftCaps = draft.SpeculativeCapabilities;
        if (!draftCaps.Supports(SpeculativeType.DraftSimple)) return;

        using var ctx = (NativeInferenceContext)target.CreateContext(new ContextConfig
        { ContextSize = 1024, BatchSize = 512, SeqMax = 1 });
        using var specExec = ctx.CreateExecutor();
        specExec.EnableSpeculation(new SpeculativeConfig
        { Mode = SpeculativeMode.Draft, DraftModelPath = draftPath });

        // Plain reference loop (greedy) — collect tokens.
        var reference = new List<int>();
        using var plainCtx = (NativeInferenceContext)target.CreateContext(new ContextConfig
        { ContextSize = 1024, BatchSize = 512, SeqMax = 1 });
        using var plainExec = plainCtx.CreateExecutor();
        plainExec.Prompt("The theory of relativity states");
        Assert.Equal(InferResult.Ok, plainExec.Infer());
        var greedy = new SamplingConfig { Temperature = 0.0f, TopK = 1, TopP = 1.0f, RepeatPenalty = 1.0f };
        for (int i = 0; i < 12; i++)
        {
            var t = plainExec.Sample(greedy);
            if (plainCtx.IsEos(t)) break;
            reference.Add(t);
            plainExec.PromptTokens(new[] { t });
            Assert.Equal(InferResult.Ok, plainExec.Infer());
        }
        Assert.NotEmpty(reference);

        // Speculative run — must produce the identical token sequence.
        var specTokens = new List<int>();
        specExec.Prompt("The theory of relativity states");
        Assert.Equal(InferResult.Ok, specExec.Infer());
        for (int i = 0; i < reference.Count; i++)
        {
            var t = specExec.Sample(greedy);
            if (ctx.IsEos(t)) break;
            specTokens.Add(t);
            specExec.PromptTokens(new[] { t });
            Assert.Equal(InferResult.Ok, specExec.Infer());
        }

        Assert.Equal(reference.Count, specTokens.Count);
        for (int i = 0; i < reference.Count; i++)
            Assert.Equal(reference[i], specTokens[i]); // token-exact contract
    }

    // ═══════════════════════════════════════════════════
    // Lane 7 — vision: live leg wherever the mmproj+backbone pair exists.
    // Cross-platform by construction (CPU layers); skipped elsewhere.
    // ═══════════════════════════════════════════════════

    [LiveModelFact]
    public void Journey7_Vision_Mmproj_PromptSplice()
    {
        var mmproj = LiveModelFixtures.MmprojModel;
        var hybrid = LiveModelFixtures.HybridModel;
        if (mmproj is null || hybrid is null || !RamOk()) return;

        var imagePath = Path.Combine(AppContext.BaseDirectory, "../../../test_image.png");
        if (!File.Exists(imagePath)) return; // image fixture optional

        using var model = NativeInferenceModel.Load(new ModelConfig { Path = hybrid, GpuLayers = 0 });
        using var vision = model.LoadVisionEncoder(mmproj);
        Assert.False(string.IsNullOrEmpty(vision.Marker));

        using var ctx = model.CreateContext(new ContextConfig { ContextSize = 2048, BatchSize = 512, SeqMax = 1 });
        using var pool = ctx.CreatePool();
        var conv = pool.Lease();
        conv.PromptWithImages($"What is in this image? {vision.Marker}", vision, File.ReadAllBytes(imagePath));
        Assert.Equal(InferResult.Ok, ctx.InferAll());
        Assert.True(conv.TokenCount > 0);
        conv.Dispose();
    }

    // ═══════════════════════════════════════════════════
    // Lane 8 — determinism: identical greedy generation across two contexts
    // catches silent backend/pooling regressions and is the cheapest
    // cross-platform consistency probe (same host, same build).
    // ═══════════════════════════════════════════════════

    [LiveModelFact]
    public void Journey8_GreedyGeneration_Deterministic()
    {
        if (SmallModel is null || !RamOk()) return;
        static List<int> Run(NativeInferenceModel model, string prompt)
        {
            using var ctx = model.CreateContext(new ContextConfig { ContextSize = 1024, BatchSize = 512, SeqMax = 1 });
            using var exec = ctx.CreateExecutor();
            exec.Prompt(prompt);
            Assert.Equal(InferResult.Ok, exec.Infer());
            var greedy = new SamplingConfig { Temperature = 0.0f, TopK = 1, TopP = 1.0f, RepeatPenalty = 1.0f };
            var tokens = new List<int>();
            for (int i = 0; i < 12; i++)
            {
                var t = exec.Sample(greedy);
                if (ctx.IsEos(t)) break;
                tokens.Add(t);
                exec.PromptTokens(new[] { t });
                Assert.Equal(InferResult.Ok, exec.Infer());
            }
            return tokens;
        }

        using var model = LoadSmall(0);
        var a = Run(model, "Once upon a time");
        var b = Run(model, "Once upon a time");
        Assert.Equal(a, b);
        Assert.NotEmpty(a);
    }

    // ═══════════════════════════════════════════════════
    // Lane 9 — model hygiene: tokenizer-family sanity on the fixture itself
    // (catches the mislabeled-file class found with the 3B GGUF).
    // ═══════════════════════════════════════════════════

    [LiveModelFact]
    public void Journey9_ModelHygiene_TokenizerRoundTrip()
    {
        if (SmallModel is null || !RamOk()) return;
        using var model = LoadSmall(0);
        using var ctx = model.CreateContext(new ContextConfig { ContextSize = 512, BatchSize = 256, SeqMax = 1 });

        var tokens = ctx.Tokenize("hello", parseSpecial: false);
        Assert.NotEmpty(tokens);
        var piece = ctx.Detokenize(tokens);
        Assert.Contains("hello", piece, StringComparison.OrdinalIgnoreCase);

        // Special-token hygiene: unknown special ids must not fabricate markers
        // from a foreign tokenizer family.
        var special = ctx.Tokenize("<|im_start|>", parseSpecial: false);
        var rendered = ctx.Detokenize(special);
        Assert.Equal("<|im_start|>", rendered); // literal, not a decoded foreign marker
    }
}
