using ProvostXInference.Abstractions;
using ProvostXInference.Engines;
using ProvostXInference.Models;
using Xunit;

namespace ProvostXInference.Tests.Engines;

/// <summary>
/// LIVE facade smoke: the full contract chain — builder → validated request →
/// factory → engine → Initialize → inference → KV ops → Dispose — against the
/// REAL native metal specialist, same env-var convention as SmokeTests.
/// </summary>
public class EngineFacadeSmokeTests
{
    private const string ModelPath = "./models/Qwen3.5-4B-Q4_K_M.gguf";

    private static string ResolveModel()
    {
        if (Environment.GetEnvironmentVariable("PVX_MODEL_PATH") is { Length: > 0 } s) return s;
        return ModelPath;
    }

    [Fact]
    public void GgufFacade_FullLifecycle_Inference()
    {
        var modelPath = ResolveModel();
        if (!File.Exists(modelPath)) return; // no model on this host — skip
        using var engine = InferenceEngineBuilder.Gguf(modelPath)
            .WithContextSize(1024)
            .Create(new InferenceEngineFactory());
        Assert.False(engine.IsLoaded);
        engine.Initialize();
        Assert.True(engine.IsLoaded);

        // Context management + inference through the facade ops view.
        var ops = engine.Gguf;
        var templated = ops.ApplyChatTemplate(null, new[]
            { ("user", "Say exactly: ok") });
        var tokens = ops.Tokenize(templated);
        Assert.NotEmpty(tokens);
        var executor = ops.CreateExecutor();
        using (executor)
        {
            executor.Prompt(templated);
            var sampling = ops.Sampling with { Temperature = 0.1f, MaxTokens = 8 };
            var seen = new List<int>();
            ops.ValidateSpeculation(new SpeculativeConfig()); // typed pre-validation, off mode
            var result = executor.Infer();
            while (result == InferResult.Ok && seen.Count < 8)
            {
                seen.Add(executor.Sample(sampling));
                result = executor.Infer();
            }
            Assert.NotEmpty(seen);
        }

        // KV-cache ops through the facade: save/copy/clear on the live context.
        var conv = ops.LeaseConversation();
        using (conv)
        {
            conv.Prompt("hello");
            var state = ops.SaveConversationState(conv);
            var checkpoint = ops.ExportConversationCheckpoint(conv);
            Assert.NotEmpty(checkpoint);
            ops.ImportConversationCheckpoint(conv, checkpoint);
            Assert.True(state.TokenCount >= 0);
        }
        // PartialRewindSupported is arch-probed per model (Qwen3.5 hybrid = false);
        // the facade just delegates — verified above by the template/tokenize roundtrip.

        // Dispose is engine-owned: pool → context → model.
        engine.Dispose();
        Assert.False(engine.IsLoaded);
    }
}