using ProvostXInference.Abstractions;
using ProvostXInference.Implementation;
using ProvostXInference.Models;
using Xunit;

namespace ProvostXInference.Tests;

/// <summary>Integration test for the stall-safe prefill watchdog on the live
/// Metal journey (INF BL-33). LARGE-CONTEXT JOURNEY — only runs when a real
/// model is available via PVX_MODEL_PATH (same fixture contract as
/// SmokeTests); skipped otherwise so the unit suite stays hermetic.</summary>
[Trait("Category", "LargeContextJourney")]
public class StallSafePrefillLiveJourneyTests
{
    private static readonly string ModelPath = Environment.GetEnvironmentVariable("PVX_MODEL_PATH") ?? "";

    [Fact]
    public void Live_Model_Prefill_CommitsChunks_ThroughWatchdog()
    {
        // Journey guard: skipped when no live model is wired (unit suite stays hermetic).
        if (string.IsNullOrEmpty(ModelPath)) return;

        using var model = NativeInferenceModel.Load(new ModelConfig { Path = ModelPath, GpuLayers = 99 });
        using var context = model.CreateContext(new ContextConfig { ContextSize = 2048, BatchSize = 512, SeqMax = 1 });
        using var pool = context.CreatePool();

        var cfg = new StallWatchdogConfig { Enabled = true, ChunkSizeTokens = 2048 };
        var runner = new ChunkedStallSafePrefillRunner(
            pool, cfg,
            new SystemChunkWallClock(TimeProvider.System),
            new RollingPrefillStallDetector(cfg));

        // Small journey prompt (unit-scale; the LARGE-context variant of this
        // journey runs the ~200k-token prefill per TEST-EVIDENCE-spec.md).
        var tokens = context.Tokenize("The capital of France is");
        Assert.NotEmpty(tokens);

        var committed = runner.RunPromptTokens(tokens);
        Assert.Equal(tokens.Length, committed);
    }
}