using Microsoft.ML.OnnxRuntime;
using Xunit;
using ProvostXInference.Decision.Engine;

namespace ProvostXInference.Tests.Decision;

/// <summary>
/// Real ONNX judge session factory tests (wave 2): CPU chains initialize the tiny
/// synthetic model; a GPU-first chain whose provider is unavailable in this
/// runtime falls back to CPU (the EP fallback rule — GPU attempts never fail the
/// load); the selected provider is recorded honestly.
/// </summary>
[Trait("Tier","Unit")]
public class DecisionSessionFactoryTests
{
    private static (DecisionSessionFactory Factory, JudgeTokenizer Tokenizer, JudgeSequenceSpec Spec, string ModelPath) NewFixture()
    {
        var assets = new DecisionTestAssets();
        return (new DecisionSessionFactory(), assets.NewTokenizer(), assets.NewShellSpec(), assets.TinyModelPath);
    }

    [Fact]
    public void Create_CpuChain_InitializesAndReportsCpu()
    {
        var (factory, tokenizer, spec, model) = NewFixture();
        var chain = ExecutionProviderChain.Parse("cpu", "test");

        using var session = factory.Create(model, tokenizer, spec, chain);

        Assert.Equal("cpu", session.ActiveProvider);
        Assert.True(session.IsCpu);
        Assert.NotNull(session.Spec);
        Assert.Equal(JudgeRecordKind.Shell, session.Spec.Kind);
    }

    [Fact]
    public void Create_GpuFirstChainUnavailable_FallsBackToCpu()
    {
        var (factory, tokenizer, spec, model) = NewFixture();
        var chain = ExecutionProviderChain.Parse("cuda->cpu", "test");
        var cudaAvailable = OrtEnv.Instance().GetAvailableProviders()
            .Any(p => p.Contains("CUDA", StringComparison.OrdinalIgnoreCase));
        if (cudaAvailable)
            return; // runtime bundles CUDA — the unavailable-fallback path is covered by the engine-level stub tests

        using var session = factory.Create(model, tokenizer, spec, chain);

        Assert.Equal("cpu", session.ActiveProvider);
        Assert.True(session.IsCpu);
    }

    [Fact]
    public void Create_GpuOnlyChain_WhenUnavailable_Throws()
    {
        var (factory, tokenizer, spec, model) = NewFixture();
        var chain = ExecutionProviderChain.Parse("cuda", "test");
        var cudaAvailable = OrtEnv.Instance().GetAvailableProviders()
            .Any(p => p.Contains("CUDA", StringComparison.OrdinalIgnoreCase));
        if (cudaAvailable)
            return; // runtime bundles CUDA — a GPU-only chain legitimately succeeds there

        Assert.ThrowsAny<Exception>(() => factory.Create(model, tokenizer, spec, chain));
    }

    [Fact]
    public void Create_CoreMlFallbackChain_OnThisRuntime_ResolvesWithoutCrash()
    {
        // "coreml->cpu" — CoreML IS bundled in the CPU package on Apple platforms and
        // may or may not initialize for this graph; either way the call must succeed
        // (coreml selected OR the documented CPU fallback) and never throw.
        var (factory, tokenizer, spec, model) = NewFixture();
        var chain = ExecutionProviderChain.Parse("coreml->cpu", "test");

        using var session = factory.Create(model, tokenizer, spec, chain);
        Assert.False(string.IsNullOrEmpty(session.ActiveProvider));
    }
}