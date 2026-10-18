using ProvostXInference.Exceptions;
using ProvostXInference.Models;
using ProvostXInference.Engines;
using Xunit;

namespace ProvostXInference.Tests.Engines;

/// <summary>Base <see cref="EngineRequest"/> contract: required ModelPath,
/// fail-closed validation, EngineId pass-through. Exercised through all three
/// concrete requests.</summary>
public class EngineRequestTests
{
    [Fact]
    public void NullModelPath_IsFailClosed_OnAllKinds()
    {
        EngineRequest Make(string path) => new GgufEngineRequest { ModelPath = path };
        Assert.Throws<InvalidEngineRequestException>(() =>
            InferenceEngineBuilder.Gguf("  ").Build());
        Assert.Throws<InvalidEngineRequestException>(() => Make("").Validate());
        Assert.Throws<InvalidEngineRequestException>(() => Make(" \t ").Validate());
    }

    [Fact]
    public void ValidGguf_PassesValidation()
    {
        var r = InferenceEngineBuilder.Gguf("/m/model.gguf").Build();
        r.Validate(); // must not throw
        Assert.Equal("/m/model.gguf", r.ModelPath);
    }

    [Fact]
    public void EngineId_DefaultsNull_AndPassesThrough()
    {
        var r = InferenceEngineBuilder.Gguf("/m/a.gguf").WithEngineId("metal").Build();
        Assert.Equal("metal", r.EngineId);
        Assert.Null(InferenceEngineBuilder.Gguf("/m/a.gguf").Build().EngineId);
    }
}
