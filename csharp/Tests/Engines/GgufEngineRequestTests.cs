using ProvostXInference.Exceptions;
using ProvostXInference.Models;
using ProvostXInference.Engines;
using Xunit;

namespace ProvostXInference.Tests.Engines;

/// <summary><see cref="GgufEngineRequest"/>: knob validation + native config mapping.</summary>
public class GgufEngineRequestTests
{
    private static GgufEngineRequest Sample() => InferenceEngineBuilder.Gguf("/m/m.gguf")
        .WithMmproj("/m/mmproj.gguf").WithDraftModel("/m/draft.gguf").WithGpuLayers(10)
        .WithThreads(4).WithFlashAttention(false).WithKvCacheType(KvType.Q8_0)
        .WithReasoningMarkers("think", "reason").WithContextSize(2048)
        .WithBatchSize(64).WithSeqMax(2).WithPooling(PoolingType.Mean).Build();

    [Fact]
    public void Builder_MapsEveryKnob()
    {
        var r = Sample();
        Assert.Equal("/m/m.gguf", r.ModelPath);
        Assert.Equal("/m/mmproj.gguf", r.MmprojPath);
        Assert.Equal("/m/draft.gguf", r.DraftModelPath);
        Assert.Equal(10, r.GpuLayers);
        Assert.Equal(4, r.Threads);
        Assert.False(r.FlashAttention);
        Assert.Equal(KvType.Q8_0, r.KvCacheType);
        Assert.Equal(new[] { "think", "reason" }, r.ReasoningMarkers);
        Assert.Equal(2048u, r.ContextSize);
        Assert.Equal(64u, r.BatchSize);
        Assert.Equal(2u, r.SeqMax);
        Assert.Equal(PoolingType.Mean, r.PoolingType);
    }

    [Theory]
    [InlineData(-2)]
    public void GpuLayers_BelowMinusOne_Fails(int gpuLayers) =>
        Assert.Throws<InvalidEngineRequestException>(() =>
            InferenceEngineBuilder.Gguf("/m/m.gguf").WithGpuLayers(gpuLayers).Build());

    [Fact]
    public void NegativeThreads_Fails() =>
        Assert.Throws<InvalidEngineRequestException>(() =>
            InferenceEngineBuilder.Gguf("/m/m.gguf").WithThreads(-1).Build());

    [Fact]
    public void ZeroContextKnobs_FailClosed()
    {
        Assert.Throws<InvalidEngineRequestException>(() =>
            InferenceEngineBuilder.Gguf("/m/m.gguf").WithContextSize(0).Build());
        Assert.Throws<InvalidEngineRequestException>(() =>
            InferenceEngineBuilder.Gguf("/m/m.gguf").WithBatchSize(0).Build());
        Assert.Throws<InvalidEngineRequestException>(() =>
            InferenceEngineBuilder.Gguf("/m/m.gguf").WithSeqMax(0).Build());
    }

    [Fact]
    public void EmptyMmprojPath_Fails() =>
        Assert.Throws<InvalidEngineRequestException>(() =>
            InferenceEngineBuilder.Gguf("/m/m.gguf").WithMmproj("  ").Build());

    [Fact]
    public void ToModelConfig_MapsTheNativeKnobs()
    {
        var r = Sample();
        var c = r.ToModelConfig();
        Assert.Equal(r.ModelPath, c.Path);
        Assert.Equal(r.EngineId, c.Engine);
        Assert.Equal(r.GpuLayers, c.GpuLayers);
        Assert.Equal(r.Threads, c.Threads);
        Assert.Equal(r.FlashAttention, c.FlashAttention);
        Assert.Equal(r.KvCacheType, c.KvCacheType);
        Assert.Equal(r.ReasoningMarkers, c.ReasoningMarkers);
    }

    [Fact]
    public void ToContextConfig_MapsTheContextKnobs()
    {
        var c = Sample().ToContextConfig();
        Assert.Equal(2048u, c.ContextSize);
        Assert.Equal(64u, c.BatchSize);
        Assert.Equal(2u, c.SeqMax);
        Assert.Equal(PoolingType.Mean, c.PoolingType);
    }
}
