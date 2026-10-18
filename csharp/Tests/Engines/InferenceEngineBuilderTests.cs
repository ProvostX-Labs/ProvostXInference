using ProvostXInference.Decision.Abstractions;
using ProvostXInference.Decision.Engine;
using ProvostXInference.Engines;
using Xunit;

namespace ProvostXInference.Tests.Engines;

/// <summary><see cref="InferenceEngineBuilder"/>: entry points + Build/Create plumbing.</summary>
public class InferenceEngineBuilderTests
{
    private sealed class RecordingFactory : IInferenceEngineFactory
    {
        public List<EngineRequest> Received { get; } = new();
        public IInferenceEngine Create(EngineRequest request)
        {
            Received.Add(request);
            return new InferenceEngine(request, new FakeGgufModelLoader(),
                new DecisionSessionFactory(), new OsExecutionProviderDefaultSource(), new FakeEngineBackend());
        }
    }

    [Fact]
    public void EntryPoints_ChooseTheRightKind()
    {
        Assert.IsType<GgufEngineRequest>(InferenceEngineBuilder.Gguf("/a.gguf").Build());
        Assert.IsType<DecisionEngineRequest>(InferenceEngineBuilder.Decision("/a.onnx")
            .WithTokenizer("/a.tok.json").WithSpec("/a.spec.json").Build());
        Assert.Throws<Exceptions.InvalidEngineRequestException>(() =>
            InferenceEngineBuilder.Encoder("/a.gguf").Build()); // no mmproj yet
        Assert.IsType<EncoderEngineRequest>(
            InferenceEngineBuilder.Encoder("/a.gguf").WithMmproj("/a.mm").Build());
    }

    [Fact]
    public void Create_PlumbsThroughTheFactory()
    {
        var factory = new RecordingFactory();
        var engine = InferenceEngineBuilder.Gguf("/m/m.gguf").Create(factory);
        Assert.Single(factory.Received);
        Assert.Equal("/m/m.gguf", factory.Received[0].ModelPath);
        Assert.False(engine.IsLoaded);
        Assert.Equal(EngineKind.Gguf, engine.Kind);
    }

    [Fact]
    public void DecisionCreate_PlumbsSpecAndTokenizer()
    {
        var factory = new RecordingFactory();
        var engine = InferenceEngineBuilder.Decision("/m/j.onnx")
            .WithTokenizer("/m/t.json").WithSpec("/m/s.json").Create(factory);
        var req = Assert.IsType<DecisionEngineRequest>(factory.Received[0]);
        Assert.Equal("/m/t.json", req.TokenizerJsonPath);
        Assert.Equal("/m/s.json", req.SpecJsonPath);
        Assert.Equal(EngineKind.Decision, engine.Kind);
    }
}