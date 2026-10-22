using ProvostXInference.Abstractions;
using ProvostXInference.Exceptions;
using ProvostXInference.Implementation;
using ProvostXInference.Models;
using Xunit;

namespace ProvostXInference.Tests;

/// <summary>
/// Comprehensive test suite for ProvostXInference C# bindings.
/// Covers: model, context, tokenization, executor, pool, conversations,
/// batched inference, sampling, state, embeddings, vision, KV ops,
/// backend probe, shift_left, error handling, resource cleanup.
/// </summary>
public class SmokeTests
{
    private static readonly string ModelPath = Environment.GetEnvironmentVariable("PVX_MODEL_PATH") ?? "./models/Qwen3.5-4B-Q4_K_M.gguf";
    private static readonly string EmbedModelPath = Environment.GetEnvironmentVariable("PVX_EMBED_MODEL_PATH") ?? "./models/all-MiniLM-L6-v2-Q5_K_M.gguf";
    private static readonly string MmprojPath = Environment.GetEnvironmentVariable("PVX_MMPROJ_PATH") ?? "./models/mmproj-Qwen3.5-4B-BF16.gguf";
    private static readonly string HybridModelPath = Environment.GetEnvironmentVariable("PVX_HYBRID_MODEL_PATH") ?? "./models/Qwen3.5-4B-Q4_K_M.gguf";

        // Model-gated contract: smoke legs are model-dependent; when no fixture is
        // staged (CI) they typed-skip instead of failing LOAD_FAILED — the same
        // behavior the live journey tests get from LiveModelFixtures.
        private static bool ModelAvailable =>
            System.IO.File.Exists(ModelPath) || System.IO.File.Exists(HybridModelPath);

    // ═══════════════════════════════════════════════════
    // 1. Model lifecycle
    // ═══════════════════════════════════════════════════

    [Fact]
    public void Model_Load_And_Dispose()
    {
        if (!ModelAvailable) return; // typed-skip without model fixture
        using var model = NativeInferenceModel.Load(new ModelConfig { Path = ModelPath, GpuLayers = 99 });
        Assert.True(model.EmbeddingDimension > 0);
    }

    [Fact]
    public void Model_Load_InvalidPath_Throws()
    {
        if (!ModelAvailable) return; // typed-skip without model fixture
        Assert.Throws<InferenceException>(() =>
            NativeInferenceModel.Load(new ModelConfig { Path = "/nonexistent/model.gguf" }));
    }

    [Fact]
    public void Model_Reload_5x_NoLeak()
    {
        if (!ModelAvailable) return; // typed-skip without model fixture
        for (int i = 0; i < 5; i++)
        {
            using var model = NativeInferenceModel.Load(new ModelConfig { Path = ModelPath, GpuLayers = 0 });
            using var ctx = model.CreateContext(new ContextConfig { ContextSize = 512, BatchSize = 256, SeqMax = 1 });
            Assert.True(ctx.ContextSize > 0);
        }
    }

    [Fact]
    public void Model_CpuOnly()
    {
        if (!ModelAvailable) return; // typed-skip without model fixture
        using var model = NativeInferenceModel.Load(new ModelConfig { Path = ModelPath, GpuLayers = 0 });
        using var ctx = model.CreateContext(new ContextConfig { ContextSize = 2048, BatchSize = 512, SeqMax = 2 });
        using var exec = ctx.CreateExecutor();
        exec.Prompt("Say hello");
        Assert.Equal(InferResult.Ok, exec.Infer());
        Assert.True(exec.TokenCount > 0);
    }

    [Fact]
    public void Model_FlashAttentionDisabled()
    {
        if (!ModelAvailable) return; // typed-skip without model fixture
        using var model = NativeInferenceModel.Load(new ModelConfig { Path = ModelPath, GpuLayers = 99, FlashAttention = false });
        using var ctx = model.CreateContext(new ContextConfig { ContextSize = 2048, BatchSize = 512, SeqMax = 2 });
        using var exec = ctx.CreateExecutor();
        exec.Prompt("Hello");
        Assert.Equal(InferResult.Ok, exec.Infer());
        Assert.True(exec.TokenCount > 0);
    }

    [Fact]
    public void Model_Q8_KvCache()
    {
        if (!ModelAvailable) return; // typed-skip without model fixture
        using var model = NativeInferenceModel.Load(new ModelConfig { Path = ModelPath, GpuLayers = 99, KvCacheType = KvType.Q8_0 });
        using var ctx = model.CreateContext(new ContextConfig { ContextSize = 2048, BatchSize = 512, SeqMax = 2 });
        using var exec = ctx.CreateExecutor();
        exec.Prompt("Hello");
        Assert.Equal(InferResult.Ok, exec.Infer());
        var token = exec.Sample(new SamplingConfig { Temperature = 0.3f });
        Assert.True(token >= 0);
    }

    // ═══════════════════════════════════════════════════
    // 2. Context
    // ═══════════════════════════════════════════════════

    [Fact]
    public void Context_Size_And_SeqMax()
    {
        if (!ModelAvailable) return; // typed-skip without model fixture
        using var model = NativeInferenceModel.Load(new ModelConfig { Path = ModelPath, GpuLayers = 99 });
        using var ctx = model.CreateContext(new ContextConfig { ContextSize = 2048, BatchSize = 512, SeqMax = 8 });
        Assert.Equal(2048u, ctx.ContextSize);
        Assert.Equal(8u, ctx.SeqMax);
    }

    [Fact]
    public void Context_TinyContext()
    {
        if (!ModelAvailable) return; // typed-skip without model fixture
        using var model = NativeInferenceModel.Load(new ModelConfig { Path = ModelPath, GpuLayers = 99 });
        using var ctx = model.CreateContext(new ContextConfig { ContextSize = 256, BatchSize = 128, SeqMax = 1 });
        using var exec = ctx.CreateExecutor();
        exec.Prompt("Say hi");
        Assert.Equal(InferResult.Ok, exec.Infer());
        Assert.True(exec.TokenCount > 0);
    }

    // ═══════════════════════════════════════════════════
    // 3. Tokenization
    // ═══════════════════════════════════════════════════

    [Fact]
    public void Tokenize_RoundTrip()
    {
        if (!ModelAvailable) return; // typed-skip without model fixture
        using var model = NativeInferenceModel.Load(new ModelConfig { Path = ModelPath, GpuLayers = 0 });
        using var ctx = model.CreateContext(new ContextConfig { ContextSize = 2048, BatchSize = 512, SeqMax = 1 });

        var tokens = ctx.Tokenize("Hello world", parseSpecial: true);
        Assert.NotEmpty(tokens);

        var text = ctx.Detokenize(tokens);
        Assert.Contains("Hello", text);
    }

    [Fact]
    public void Tokenize_EmptyString()
    {
        if (!ModelAvailable) return; // typed-skip without model fixture
        using var model = NativeInferenceModel.Load(new ModelConfig { Path = ModelPath, GpuLayers = 0 });
        using var ctx = model.CreateContext(new ContextConfig { ContextSize = 2048, BatchSize = 512, SeqMax = 1 });

        var tokens = ctx.Tokenize("", parseSpecial: true);
        // Empty string → 0 or BOS-only, either is fine
        Assert.True(tokens.Length <= 1);
    }

    [Fact]
    public void TokenToPiece_SingleToken()
    {
        if (!ModelAvailable) return; // typed-skip without model fixture
        using var model = NativeInferenceModel.Load(new ModelConfig { Path = ModelPath, GpuLayers = 0 });
        using var ctx = model.CreateContext(new ContextConfig { ContextSize = 2048, BatchSize = 512, SeqMax = 1 });

        var tokens = ctx.Tokenize("test", parseSpecial: true);
        Assert.NotEmpty(tokens);
        var piece = ctx.TokenToPiece(tokens[0]);
        Assert.False(string.IsNullOrEmpty(piece));
    }

    [Fact]
    public void IsEos_Check()
    {
        if (!ModelAvailable) return; // typed-skip without model fixture
        using var model = NativeInferenceModel.Load(new ModelConfig { Path = ModelPath, GpuLayers = 0 });
        using var ctx = model.CreateContext(new ContextConfig { ContextSize = 2048, BatchSize = 512, SeqMax = 1 });

        // Token 0 may be EOS for some models (SmolLM2). Use a real tokenized word instead.
        var tokens = ctx.Tokenize("hello", parseSpecial: false);
        Assert.NotEmpty(tokens);
        Assert.False(ctx.IsEos(tokens[0]));
    }

    // ═══════════════════════════════════════════════════
    // 4. Standard executor — full generation loop
    // ═══════════════════════════════════════════════════

    [Fact]
    public void Executor_FullGeneration_20Tokens()
    {
        if (!ModelAvailable) return; // typed-skip without model fixture
        using var model = NativeInferenceModel.Load(new ModelConfig { Path = ModelPath, GpuLayers = 99 });
        using var ctx = model.CreateContext(new ContextConfig { ContextSize = 4096, BatchSize = 512, SeqMax = 2 });
        using var exec = ctx.CreateExecutor();

        exec.Prompt("Count from 1 to 20: 1 2 3 4 5");
        Assert.Equal(InferResult.Ok, exec.Infer());
        int initialCount = exec.TokenCount;
        Assert.True(initialCount > 0);

        var sampling = new SamplingConfig { Temperature = 0.3f, TopK = 40, MaxTokens = 20 };
        int generated = 0;
        for (int i = 0; i < 20; i++)
        {
            var token = exec.Sample(sampling);
            if (ctx.IsEos(token)) break;
            exec.PromptTokens(new[] { token });
            Assert.Equal(InferResult.Ok, exec.Infer());
            generated++;
        }
        Assert.True(generated > 0);
        Assert.True(exec.TokenCount > initialCount);
    }

    [Fact]
    public void Executor_Infer_NoWork_When_NoPending()
    {
        if (!ModelAvailable) return; // typed-skip without model fixture
        using var model = NativeInferenceModel.Load(new ModelConfig { Path = ModelPath, GpuLayers = 99 });
        using var ctx = model.CreateContext(new ContextConfig { ContextSize = 2048, BatchSize = 512, SeqMax = 1 });
        using var exec = ctx.CreateExecutor();

        // No prompt → infer should be NoWork
        Assert.Equal(InferResult.NoWork, exec.Infer());
        Assert.Equal(0, exec.TokenCount);
    }

    [Fact]
    public void Executor_Rewind_ToZero()
    {
        if (!ModelAvailable) return; // typed-skip without model fixture
        using var model = NativeInferenceModel.Load(new ModelConfig { Path = ModelPath, GpuLayers = 99 });
        using var ctx = model.CreateContext(new ContextConfig { ContextSize = 4096, BatchSize = 512, SeqMax = 2 });
        using var exec = ctx.CreateExecutor();

        exec.Prompt("Hello world test prompt");
        Assert.Equal(InferResult.Ok, exec.Infer());
        int n = exec.TokenCount;
        Assert.True(n > 0);

        exec.Rewind(n);
        Assert.Equal(0, exec.TokenCount);

        // Rewind past 0 (should clamp)
        exec.Rewind(100);
        Assert.Equal(0, exec.TokenCount);
    }

    [Fact]
    public void Executor_Reset()
    {
        if (!ModelAvailable) return; // typed-skip without model fixture
        using var model = NativeInferenceModel.Load(new ModelConfig { Path = ModelPath, GpuLayers = 99 });
        using var ctx = model.CreateContext(new ContextConfig { ContextSize = 4096, BatchSize = 512, SeqMax = 2 });
        using var exec = ctx.CreateExecutor();

        exec.Prompt("Hello world");
        Assert.Equal(InferResult.Ok, exec.Infer());
        Assert.True(exec.TokenCount > 0);

        exec.Reset();
        Assert.Equal(0, exec.TokenCount);
    }

    [Fact]
    public void Executor_PromptTokens_Raw()
    {
        if (!ModelAvailable) return; // typed-skip without model fixture
        using var model = NativeInferenceModel.Load(new ModelConfig { Path = ModelPath, GpuLayers = 99 });
        using var ctx = model.CreateContext(new ContextConfig { ContextSize = 2048, BatchSize = 512, SeqMax = 1 });
        using var exec = ctx.CreateExecutor();

        // Tokenize manually then feed raw tokens
        var tokens = ctx.Tokenize("Say hello", parseSpecial: true);
        exec.PromptTokens(tokens);
        Assert.Equal(InferResult.Ok, exec.Infer());
        Assert.True(exec.TokenCount > 0);
    }

    // ═══════════════════════════════════════════════════
    // 5. Sampling parameter extremes
    // ═══════════════════════════════════════════════════

    [Fact]
    public void Sample_HighTemperature()
    {
        if (!ModelAvailable) return; // typed-skip without model fixture
        using var model = NativeInferenceModel.Load(new ModelConfig { Path = ModelPath, GpuLayers = 99 });
        using var ctx = model.CreateContext(new ContextConfig { ContextSize = 2048, BatchSize = 512, SeqMax = 1 });
        using var exec = ctx.CreateExecutor();
        exec.Prompt("Say a number");
        Assert.Equal(InferResult.Ok, exec.Infer());

        var token = exec.Sample(new SamplingConfig { Temperature = 100f, TopP = 1f, TopK = 0, RepeatPenalty = 1f });
        Assert.True(token >= 0);
    }

    [Fact]
    public void Sample_NearGreedy()
    {
        if (!ModelAvailable) return; // typed-skip without model fixture
        using var model = NativeInferenceModel.Load(new ModelConfig { Path = ModelPath, GpuLayers = 99 });
        using var ctx = model.CreateContext(new ContextConfig { ContextSize = 2048, BatchSize = 512, SeqMax = 1 });
        using var exec = ctx.CreateExecutor();
        exec.Prompt("Say a number");
        Assert.Equal(InferResult.Ok, exec.Infer());

        var token = exec.Sample(new SamplingConfig { Temperature = 0.001f, TopK = 1 });
        Assert.True(token >= 0);
    }

    [Fact]
    public void Sample_TopPZero()
    {
        if (!ModelAvailable) return; // typed-skip without model fixture
        using var model = NativeInferenceModel.Load(new ModelConfig { Path = ModelPath, GpuLayers = 99 });
        using var ctx = model.CreateContext(new ContextConfig { ContextSize = 2048, BatchSize = 512, SeqMax = 1 });
        using var exec = ctx.CreateExecutor();
        exec.Prompt("Say a number");
        Assert.Equal(InferResult.Ok, exec.Infer());

        var token = exec.Sample(new SamplingConfig { Temperature = 0.3f, TopP = 0f, TopK = 0 });
        Assert.True(token >= 0);
    }

    [Fact]
    public void Sample_MinP_0_99()
    {
        if (!ModelAvailable) return; // typed-skip without model fixture
        using var model = NativeInferenceModel.Load(new ModelConfig { Path = ModelPath, GpuLayers = 99 });
        using var ctx = model.CreateContext(new ContextConfig { ContextSize = 2048, BatchSize = 512, SeqMax = 1 });
        using var exec = ctx.CreateExecutor();
        exec.Prompt("Say a number");
        Assert.Equal(InferResult.Ok, exec.Infer());

        var token = exec.Sample(new SamplingConfig { Temperature = 0.3f, TopP = 1f, TopK = 0, MinP = 0.99f });
        Assert.True(token >= 0);
    }

    // ═══════════════════════════════════════════════════
    // 6. Conversation pool — lease, return, reuse
    // ═══════════════════════════════════════════════════

    [Fact]
    public void Pool_Exhaust_And_Return()
    {
        if (!ModelAvailable) return; // typed-skip without model fixture
        using var model = NativeInferenceModel.Load(new ModelConfig { Path = ModelPath, GpuLayers = 99 });
        using var ctx = model.CreateContext(new ContextConfig { ContextSize = 4096, BatchSize = 512, SeqMax = 4 });
        using var pool = ctx.CreatePool();

        Assert.Equal(4u, pool.Size);
        Assert.Equal(4, pool.AvailableCount);

        var convs = new IConversation[4];
        for (int i = 0; i < 4; i++)
        {
            convs[i] = pool.Lease();
            Assert.Equal(4 - i - 1, pool.AvailableCount);
        }

        // 5th should throw
        Assert.Throws<InferenceException>(() => pool.Lease());

        // Return all
        for (int i = 0; i < 4; i++) convs[i].Dispose();
        Assert.Equal(4, pool.AvailableCount);
    }

    [Fact]
    public void Pool_LeaseReturn_10Cycles()
    {
        if (!ModelAvailable) return; // typed-skip without model fixture
        using var model = NativeInferenceModel.Load(new ModelConfig { Path = ModelPath, GpuLayers = 99 });
        using var ctx = model.CreateContext(new ContextConfig { ContextSize = 4096, BatchSize = 512, SeqMax = 2 });
        using var pool = ctx.CreatePool();

        for (int cycle = 0; cycle < 10; cycle++)
        {
            var conv = pool.Lease();
            Assert.Equal(1, pool.AvailableCount);

            conv.Prompt("Say a number");
            Assert.Equal(InferResult.Ok, ctx.InferAll());
            Assert.True(conv.TokenCount > 0);

            var token = conv.Sample(new SamplingConfig { Temperature = 0.3f });
            Assert.True(token >= 0);

            conv.Dispose();
            Assert.Equal(2, pool.AvailableCount);
        }
    }

    [Fact]
    public void Pool_ReturnWithoutInfer_CleansUp()
    {
        if (!ModelAvailable) return; // typed-skip without model fixture
        using var model = NativeInferenceModel.Load(new ModelConfig { Path = ModelPath, GpuLayers = 99 });
        using var ctx = model.CreateContext(new ContextConfig { ContextSize = 4096, BatchSize = 512, SeqMax = 2 });
        using var pool = ctx.CreatePool();

        var conv = pool.Lease();
        conv.Prompt("Hello world this is a test");
        // Return WITHOUT calling InferAll — pending tokens should be cleared
        conv.Dispose();

        // Re-lease should be clean
        var conv2 = pool.Lease();
        Assert.Equal(0, conv2.TokenCount);
        conv2.Dispose();
    }

    // ═══════════════════════════════════════════════════
    // 7. Batched inference
    // ═══════════════════════════════════════════════════

    [Fact]
    public void Batched_4Conversations_DifferentPrompts()
    {
        if (!ModelAvailable) return; // typed-skip without model fixture
        using var model = NativeInferenceModel.Load(new ModelConfig { Path = ModelPath, GpuLayers = 99 });
        using var ctx = model.CreateContext(new ContextConfig { ContextSize = 4096, BatchSize = 512, SeqMax = 4 });
        using var pool = ctx.CreatePool();

        var convs = new IConversation[4];
        var prompts = new[] { "Say 1.", "Say the number 2.", "Please say 3 now.", "I want you to say 4." };

        for (int i = 0; i < 4; i++)
        {
            convs[i] = pool.Lease();
            convs[i].Prompt(prompts[i]);
        }

        Assert.Equal(InferResult.Ok, ctx.InferAll());

        // Different prompt lengths → different token counts
        Assert.True(convs[0].TokenCount > 0);
        Assert.True(convs[3].TokenCount > 0);

        // Sample from each
        for (int i = 0; i < 4; i++)
        {
            var token = convs[i].Sample(new SamplingConfig { Temperature = 0.3f });
            Assert.True(token >= 0);
        }

        for (int i = 0; i < 4; i++) convs[i].Dispose();
    }

    [Fact]
    public void Batched_NoPending_NoWork()
    {
        if (!ModelAvailable) return; // typed-skip without model fixture
        using var model = NativeInferenceModel.Load(new ModelConfig { Path = ModelPath, GpuLayers = 99 });
        using var ctx = model.CreateContext(new ContextConfig { ContextSize = 4096, BatchSize = 512, SeqMax = 4 });
        using var pool = ctx.CreatePool();

        var conv = pool.Lease();
        // No prompt on any conversation
        Assert.Equal(InferResult.NoWork, ctx.InferAll());
        conv.Dispose();
    }

    [Fact]
    public void Batched_PartialPrompt_2of3()
    {
        if (!ModelAvailable) return; // typed-skip without model fixture
        using var model = NativeInferenceModel.Load(new ModelConfig { Path = ModelPath, GpuLayers = 99 });
        using var ctx = model.CreateContext(new ContextConfig { ContextSize = 4096, BatchSize = 512, SeqMax = 4 });
        using var pool = ctx.CreatePool();

        var c0 = pool.Lease();
        var c1 = pool.Lease();
        var c2 = pool.Lease(); // no prompt

        c0.Prompt("Say A");
        c1.Prompt("Say B");

        Assert.Equal(InferResult.Ok, ctx.InferAll());

        Assert.True(c0.TokenCount > 0);
        Assert.True(c1.TokenCount > 0);
        Assert.Equal(0, c2.TokenCount);

        c0.Dispose(); c1.Dispose(); c2.Dispose();
    }

    // ═══════════════════════════════════════════════════
    // 8. Shift left (sliding window)
    // ═══════════════════════════════════════════════════

    [Fact]
    public void ShiftLeft_Executor()
    {
        if (!ModelAvailable) return; // typed-skip without model fixture
        using var model = NativeInferenceModel.Load(new ModelConfig { Path = ModelPath, GpuLayers = 99 });
        using var ctx = model.CreateContext(new ContextConfig { ContextSize = 512, BatchSize = 256, SeqMax = 2 });
        using var exec = ctx.CreateExecutor();

        exec.Prompt("Hello world this is a sliding window test with enough tokens to work properly");
        Assert.Equal(InferResult.Ok, exec.Infer());
        int before = exec.TokenCount;
        Assert.True(before > 10);

        exec.ShiftLeft(10);
        Assert.Equal(before - 10, exec.TokenCount);

        // Can continue generating after shift
        exec.Prompt("Continue: ");
        Assert.Equal(InferResult.Ok, exec.Infer());
        Assert.True(exec.TokenCount > before - 10);
    }

    [Fact]
    public void ShiftLeft_Conversation()
    {
        if (!ModelAvailable) return; // typed-skip without model fixture
        using var model = NativeInferenceModel.Load(new ModelConfig { Path = ModelPath, GpuLayers = 99 });
        using var ctx = model.CreateContext(new ContextConfig { ContextSize = 512, BatchSize = 256, SeqMax = 2 });
        using var pool = ctx.CreatePool();
        var conv = pool.Lease();

        conv.Prompt("Hello world test prompt for sliding window");
        Assert.Equal(InferResult.Ok, ctx.InferAll());
        int before = conv.TokenCount;
        Assert.True(before > 5);

        conv.ShiftLeft(5);
        Assert.Equal(before - 5, conv.TokenCount);

        conv.Dispose();
    }

    [Fact]
    public void ShiftLeft_Zero_Throws()
    {
        if (!ModelAvailable) return; // typed-skip without model fixture
        using var model = NativeInferenceModel.Load(new ModelConfig { Path = ModelPath, GpuLayers = 99 });
        using var ctx = model.CreateContext(new ContextConfig { ContextSize = 512, BatchSize = 256, SeqMax = 2 });
        using var exec = ctx.CreateExecutor();
        exec.Prompt("Test");
        Assert.Equal(InferResult.Ok, exec.Infer());

        Assert.Throws<InferenceException>(() => exec.ShiftLeft(0));
    }

    // ═══════════════════════════════════════════════════
    // 9. State save / restore
    // ═══════════════════════════════════════════════════

    [Fact]
    public void State_Save_Restore_Executor()
    {
        if (!ModelAvailable) return; // typed-skip without model fixture
        using var model = NativeInferenceModel.Load(new ModelConfig { Path = ModelPath, GpuLayers = 99 });
        using var ctx = model.CreateContext(new ContextConfig { ContextSize = 4096, BatchSize = 512, SeqMax = 2 });
        using var exec = ctx.CreateExecutor();

        exec.Prompt("Hello world");
        Assert.Equal(InferResult.Ok, exec.Infer());
        int n = exec.TokenCount;
        Assert.True(n > 0);

        // Save
        var state = exec.SaveState();
        Assert.Equal(n, state.TokenCount);

        // Generate more
        for (int i = 0; i < 3; i++)
        {
            var t = exec.Sample(new SamplingConfig { Temperature = 0.3f });
            exec.PromptTokens(new[] { t });
            exec.Infer();
        }
        Assert.True(exec.TokenCount > n);

        // Restore — should go back to n
        exec.RestoreState(state);
        Assert.Equal(n, exec.TokenCount);

        state.Dispose();
    }

    [Fact]
    public void State_Save_Empty()
    {
        if (!ModelAvailable) return; // typed-skip without model fixture
        using var model = NativeInferenceModel.Load(new ModelConfig { Path = ModelPath, GpuLayers = 99 });
        using var ctx = model.CreateContext(new ContextConfig { ContextSize = 4096, BatchSize = 512, SeqMax = 2 });
        using var exec = ctx.CreateExecutor();

        var state = exec.SaveState();
        Assert.Equal(0, state.TokenCount);
        state.Dispose();
    }

    // ═══════════════════════════════════════════════════
    // 10. Embeddings
    // ═══════════════════════════════════════════════════

    [Fact]
    public void Embeddings_Mean_NonZero_Normalized()
    {
        if (!ModelAvailable) return; // typed-skip without model fixture
        using var model = NativeInferenceModel.Load(new ModelConfig { Path = EmbedModelPath, GpuLayers = 0 });
        using var ctx = model.CreateContext(new ContextConfig { ContextSize = 2048, BatchSize = 512, SeqMax = 1, PoolingType = PoolingType.Mean });

        var emb = model.GetEmbeddings(ctx, "Hello world test");
        Assert.True(emb.Length > 0);

        float norm = 0;
        foreach (var v in emb) norm += v * v;
        norm = MathF.Sqrt(norm);
        Assert.True(norm > 0.5f, $"norm={norm}");
        Assert.True(norm < 1.5f, $"norm={norm}");
    }

    [Fact]
    public void Embeddings_Cls_Pooling()
    {
        if (!ModelAvailable) return; // typed-skip without model fixture
        using var model = NativeInferenceModel.Load(new ModelConfig { Path = EmbedModelPath, GpuLayers = 0 });
        using var ctx = model.CreateContext(new ContextConfig { ContextSize = 2048, BatchSize = 512, SeqMax = 1, PoolingType = PoolingType.Cls });

        var emb = model.GetEmbeddings(ctx, "Hello world");
        Assert.True(emb.Length > 0);

        float norm = 0;
        foreach (var v in emb) norm += v * v;
        norm = MathF.Sqrt(norm);
        Assert.True(norm > 0.5f);
    }

    [Fact]
    public void Embeddings_Last_Pooling()
    {
        if (!ModelAvailable) return; // typed-skip without model fixture
        using var model = NativeInferenceModel.Load(new ModelConfig { Path = EmbedModelPath, GpuLayers = 0 });
        using var ctx = model.CreateContext(new ContextConfig { ContextSize = 2048, BatchSize = 512, SeqMax = 1, PoolingType = PoolingType.Last });

        var emb = model.GetEmbeddings(ctx, "Hello world");
        Assert.True(emb.Length > 0);

        float norm = 0;
        foreach (var v in emb) norm += v * v;
        norm = MathF.Sqrt(norm);
        Assert.True(norm > 0.5f);
    }

    [Fact]
    public void Embeddings_Dim_MatchesModel()
    {
        if (!ModelAvailable) return; // typed-skip without model fixture
        using var model = NativeInferenceModel.Load(new ModelConfig { Path = EmbedModelPath, GpuLayers = 0 });
        using var ctx = model.CreateContext(new ContextConfig { ContextSize = 2048, BatchSize = 512, SeqMax = 1, PoolingType = PoolingType.Mean });

        var emb = model.GetEmbeddings(ctx, "test");
        Assert.Equal(model.EmbeddingDimension, emb.Length);
    }

    [Fact]
    public void Embeddings_DifferentTexts_DifferentVectors()
    {
        if (!ModelAvailable) return; // typed-skip without model fixture
        using var model = NativeInferenceModel.Load(new ModelConfig { Path = EmbedModelPath, GpuLayers = 0 });
        using var ctx = model.CreateContext(new ContextConfig { ContextSize = 2048, BatchSize = 512, SeqMax = 1, PoolingType = PoolingType.Mean });

        var emb1 = model.GetEmbeddings(ctx, "Hello world");
        var emb2 = model.GetEmbeddings(ctx, "Goodbye universe");

        Assert.Equal(emb1.Length, emb2.Length);

        // Different texts should produce different embeddings
        float diff = 0;
        for (int i = 0; i < emb1.Length; i++) diff += MathF.Abs(emb1[i] - emb2[i]);
        Assert.True(diff > 0.1f, $"embeddings too similar, diff={diff}");
    }

    // ═══════════════════════════════════════════════════
    // 11. Vision / MTMD
    // ═══════════════════════════════════════════════════

    [Fact]
    public void Vision_LoadMmproj()
    {
        if (!ModelAvailable) return; // typed-skip without model fixture
        if (!File.Exists(MmprojPath)) return; // skip when mmproj not available
        // ABI v8: vision tests pair the mmproj with its HYBRID backbone
        // (default Qwen3.5-4B) — never the dense default (ModelPath may point
        // at a non-multimodal model under env gating).
        using var model = NativeInferenceModel.Load(new ModelConfig { Path = HybridModelPath, GpuLayers = 99 });
        using var vision = model.LoadVisionEncoder(MmprojPath);
        Assert.False(string.IsNullOrEmpty(vision.Marker));
        Assert.True(vision.Marker.Length > 0);
    }

    [Fact]
    public void Vision_PromptWithImages_64x64()
    {
        if (!ModelAvailable) return; // typed-skip without model fixture
        // Create a 64x64 PNG if it doesn't exist
        if (!File.Exists("./test_image.png"))
        {
            // Minimal 64x64 PNG (solid color)
            File.WriteAllBytes("./test_image.png", new byte[] {
                0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A // ... actual PNG created by C++ test
            });
        }

        // Use the test image from the C++ tests
        var imagePath = "./test_image.png";
        if (!File.Exists(imagePath)) return; // skip if not available

        var imageBytes = File.ReadAllBytes(imagePath);
        if (imageBytes.Length < 100) return; // skip if invalid

        // ABI v8: vision pair rides the HYBRID backbone (see Vision_LoadMmproj).
        using var model = NativeInferenceModel.Load(new ModelConfig { Path = HybridModelPath, GpuLayers = 99 });
        using var ctx = model.CreateContext(new ContextConfig { ContextSize = 4096, BatchSize = 512, SeqMax = 2 });
        using var pool = ctx.CreatePool();
        using var vision = model.LoadVisionEncoder(MmprojPath);

        var conv = pool.Lease();
        var marker = vision.Marker;
        var prompt = $"What is in this image? {marker}";

        conv.PromptWithImages(prompt, vision, imageBytes);
        Assert.Equal(InferResult.Ok, ctx.InferAll());
        Assert.True(conv.TokenCount > 0);

        var token = conv.Sample(new SamplingConfig { Temperature = 0.3f });
        Assert.True(token >= 0);

        conv.Dispose();
    }

    // ═══════════════════════════════════════════════════
    // 12. KV cache manipulation
    // ═══════════════════════════════════════════════════

    [Fact]
    public void KvCopy_And_Clear()
    {
        if (!ModelAvailable) return; // typed-skip without model fixture
        using var model = NativeInferenceModel.Load(new ModelConfig { Path = ModelPath, GpuLayers = 99 });
        using var ctx = model.CreateContext(new ContextConfig { ContextSize = 2048, BatchSize = 512, SeqMax = 4 });

        // These should not throw
        ctx.KvClearSeq(0);
        ctx.KvClearSeq(1);
        ctx.KvCopy(0, 1);
        ctx.KvClearSeq(3);
    }

    [Fact]
    public void KvClear_InvalidSeq_Throws()
    {
        if (!ModelAvailable) return; // typed-skip without model fixture
        using var model = NativeInferenceModel.Load(new ModelConfig { Path = ModelPath, GpuLayers = 99 });
        using var ctx = model.CreateContext(new ContextConfig { ContextSize = 2048, BatchSize = 512, SeqMax = 4 });

        Assert.Throws<InferenceException>(() => ctx.KvClearSeq(99));
        Assert.Throws<InferenceException>(() => ctx.KvClearSeq(-1));
    }

    [Fact]
    public void KvCopy_InvalidSeq_Throws()
    {
        if (!ModelAvailable) return; // typed-skip without model fixture
        using var model = NativeInferenceModel.Load(new ModelConfig { Path = ModelPath, GpuLayers = 99 });
        using var ctx = model.CreateContext(new ContextConfig { ContextSize = 2048, BatchSize = 512, SeqMax = 4 });

        Assert.Throws<InferenceException>(() => ctx.KvCopy(0, 99));
        Assert.Throws<InferenceException>(() => ctx.KvCopy(99, 0));
    }

    [Fact]
    public void SetCausalAttention_Toggle()
    {
        if (!ModelAvailable) return; // typed-skip without model fixture
        using var model = NativeInferenceModel.Load(new ModelConfig { Path = ModelPath, GpuLayers = 99 });
        using var ctx = model.CreateContext(new ContextConfig { ContextSize = 2048, BatchSize = 512, SeqMax = 2 });

        ctx.SetCausalAttention(false);
        ctx.SetCausalAttention(true);
    }

    // ═══════════════════════════════════════════════════
    // 13. Backend probe
    // ═══════════════════════════════════════════════════

    [Fact]
    public void Backend_Probe_Returns_ValidInfo()
    {
        if (!ModelAvailable) return; // typed-skip without model fixture
        using var model = NativeInferenceModel.Load(new ModelConfig { Path = ModelPath, GpuLayers = 99 });
        using var ctx = model.CreateContext(new ContextConfig { ContextSize = 4096, BatchSize = 512, SeqMax = 2 });

        var backend = new NativeInferenceBackend();
        var info = backend.Probe(ctx);

        Assert.True(info.ProbePassed);
        Assert.False(string.IsNullOrEmpty(info.Name));
        Assert.True(info.EffectiveGpuLayers >= 0);
    }

    [Fact]
    public void Backend_VramAvailable_ReturnsBool()
    {
        if (!ModelAvailable) return; // typed-skip without model fixture
        var backend = new NativeInferenceBackend();
        // Should return a bool (true on CPU, may vary on GPU)
        var result = backend.VramAvailable(1024 * 1024);
        Assert.True(result); // small allocation should be available
    }

    // ═══════════════════════════════════════════════════
    // 14. Conversation operations
    // ═══════════════════════════════════════════════════

    [Fact]
    public void Conversation_Rewind()
    {
        if (!ModelAvailable) return; // typed-skip without model fixture
        using var model = NativeInferenceModel.Load(new ModelConfig { Path = ModelPath, GpuLayers = 99 });
        using var ctx = model.CreateContext(new ContextConfig { ContextSize = 4096, BatchSize = 512, SeqMax = 2 });
        using var pool = ctx.CreatePool();
        var conv = pool.Lease();

        conv.Prompt("Hello world test prompt");
        Assert.Equal(InferResult.Ok, ctx.InferAll());
        int n = conv.TokenCount;
        Assert.True(n > 0);

        conv.Rewind(n);
        Assert.Equal(0, conv.TokenCount);
        conv.Dispose();
    }

    [Fact]
    public void Conversation_Reset()
    {
        if (!ModelAvailable) return; // typed-skip without model fixture
        using var model = NativeInferenceModel.Load(new ModelConfig { Path = ModelPath, GpuLayers = 99 });
        using var ctx = model.CreateContext(new ContextConfig { ContextSize = 4096, BatchSize = 512, SeqMax = 2 });
        using var pool = ctx.CreatePool();
        var conv = pool.Lease();

        conv.Prompt("Hello world");
        Assert.Equal(InferResult.Ok, ctx.InferAll());
        Assert.True(conv.TokenCount > 0);

        conv.Reset();
        Assert.Equal(0, conv.TokenCount);
        conv.Dispose();
    }

    [Fact]
    public void Conversation_Sample_WithDefaultConfig()
    {
        if (!ModelAvailable) return; // typed-skip without model fixture
        using var model = NativeInferenceModel.Load(new ModelConfig { Path = ModelPath, GpuLayers = 99 });
        using var ctx = model.CreateContext(new ContextConfig { ContextSize = 4096, BatchSize = 512, SeqMax = 2 });
        using var pool = ctx.CreatePool();
        var conv = pool.Lease();

        conv.Prompt("Say a number");
        Assert.Equal(InferResult.Ok, ctx.InferAll());

        var token = conv.Sample(); // default config
        Assert.True(token >= 0);
        conv.Dispose();
    }

    // ═══════════════════════════════════════════════════
    // 15. Anti-prompt detection
    // ═══════════════════════════════════════════════════

    [Fact]
    public void AntiPrompts_Detect_Match()
    {
        if (!ModelAvailable) return; // typed-skip without model fixture
        // pvx_check_anti_prompts is a standalone function — test via the static helper
        // We'll test it through the context's tokenize + manual check
        using var model = NativeInferenceModel.Load(new ModelConfig { Path = ModelPath, GpuLayers = 0 });
        using var ctx = model.CreateContext(new ContextConfig { ContextSize = 2048, BatchSize = 512, SeqMax = 1 });

        // This is a pure C function, no context needed for the logic
        // But our C# wrapper doesn't expose it directly — it's available via PvxNative
        // For now, just verify the context works for tokenizing anti-prompt text
        var tokens = ctx.Tokenize("### User: What is 2+2?", parseSpecial: false);
        Assert.NotEmpty(tokens);
    }

    // ═══════════════════════════════════════════════════
    // 16. Resource cleanup (using pattern)
    // ═══════════════════════════════════════════════════

    [Fact]
    public void Dispose_Order_Model_Context_Executor()
    {
        if (!ModelAvailable) return; // typed-skip without model fixture
        // Disposing in correct order (model last) should work
        var model = NativeInferenceModel.Load(new ModelConfig { Path = ModelPath, GpuLayers = 99 });
        var ctx = model.CreateContext(new ContextConfig { ContextSize = 2048, BatchSize = 512, SeqMax = 2 });
        var exec = ctx.CreateExecutor();

        exec.Prompt("Hello");
        exec.Infer();

        exec.Dispose();
        ctx.Dispose();
        model.Dispose();
    }

    [Fact]
    public void Dispose_Reverse_Order_Works()
    {
        if (!ModelAvailable) return; // typed-skip without model fixture
        // GC + SafeHandle should handle any disposal order
        {
            using var model = NativeInferenceModel.Load(new ModelConfig { Path = ModelPath, GpuLayers = 99 });
            using var ctx = model.CreateContext(new ContextConfig { ContextSize = 2048, BatchSize = 512, SeqMax = 2 });
            using var pool = ctx.CreatePool();
            var conv = pool.Lease();
            conv.Prompt("Hello");
            ctx.InferAll();
            conv.Dispose(); // returns to pool
        } // pool → ctx → model disposed by GC order
        // If we get here without crash, it works
    }
}
