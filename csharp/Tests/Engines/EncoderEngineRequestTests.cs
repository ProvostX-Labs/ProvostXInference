using ProvostXInference.Exceptions;
using ProvostXInference.Models;
using ProvostXInference.Engines;
using Xunit;

namespace ProvostXInference.Tests.Engines;

/// <summary><see cref="EncoderEngineRequest"/>: mmproj is REQUIRED (encoder without one is meaningless).</summary>
public class EncoderEngineRequestTests
{
    [Fact]
    public void MissingMmproj_FailsClosed() =>
        Assert.Throws<InvalidEngineRequestException>(() =>
            InferenceEngineBuilder.Encoder("/m/m.gguf").Build());

    [Fact]
    public void FullRequest_Maps()
    {
        var r = InferenceEngineBuilder.Encoder("/m/m.gguf").WithMmproj("/m/mm.gguf")
            .WithEngineId("metal").WithGpuLayers(5).Build();
        Assert.Equal("/m/mm.gguf", r.MmprojPath);
        Assert.Equal("metal", r.EngineId);
        Assert.Equal(5, r.GpuLayers);
        var c = r.ToModelConfig();
        Assert.Equal("/m/m.gguf", c.Path);
        Assert.Equal("metal", c.Engine);
    }

    [Fact]
    public void GpuLayers_StillValidated() =>
        Assert.Throws<InvalidEngineRequestException>(() =>
            InferenceEngineBuilder.Encoder("/m/m.gguf").WithMmproj("/m/mm.gguf").WithGpuLayers(-5).Build());
}
