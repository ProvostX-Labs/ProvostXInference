using ProvostXInference.Decision.Abstractions;
using ProvostXInference.Decision.Engine;
using ProvostXInference.Engines;
using Xunit;

namespace ProvostXInference.Tests.Engines;

/// <summary><see cref="InferenceEngineFactory"/>: validation boundary + composition root.</summary>
public class InferenceEngineFactoryTests
{
    private static InferenceEngineFactory NewFactory(FakeGgufModelLoader? loader = null) =>
        new(loader ?? new FakeGgufModelLoader(), new DecisionSessionFactory(),
            new OsExecutionProviderDefaultSource());

    [Fact]
    public void NullRequest_Throws() =>
        Assert.Throws<ArgumentNullException>(() => NewFactory().Create(null!));

    [Fact]
    public void InvalidRequest_FailsClosed_BeforeAnyLoad()
    {
        var loader = new FakeGgufModelLoader();
        Assert.Throws<Exceptions.InvalidEngineRequestException>(() =>
            NewFactory(loader).Create(InferenceEngineBuilder.Gguf(" ").Build()));
        Assert.Empty(loader.Loaded);
    }

    [Fact]
    public void Create_ReturnsUnloadedEngine_WithRequestIdentity()
    {
        var loader = new FakeGgufModelLoader();
        var engine = NewFactory(loader).Create(InferenceEngineBuilder.Gguf("/m/m.gguf")
            .WithEngineId("metal").Build());
        Assert.False(engine.IsLoaded);
        Assert.Equal(EngineKind.Gguf, engine.Kind);
        Assert.Equal("/m/m.gguf", engine.ModelPath);
        Assert.Equal("metal", engine.ActiveSpecialist);
        Assert.Empty(loader.Loaded); // factory must NOT eagerly load
    }
}