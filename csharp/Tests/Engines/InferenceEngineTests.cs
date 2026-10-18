using ProvostXInference.Decision.Abstractions;
using ProvostXInference.Decision.Engine;
using ProvostXInference.Engines;
using ProvostXInference.Exceptions;
using ProvostXInference.Tests.Decision;
using Xunit;

namespace ProvostXInference.Tests.Engines;

/// <summary>
/// <see cref="InferenceEngine"/> facade: lifecycle, kind routing, fail-closed
/// mismatch views, disposal order. GGUF/Encoder lanes run on the recording
/// fake loader; the Decision lane runs the REAL tiny ONNX asset end-to-end.
/// </summary>
public class InferenceEngineTests
{
    private static (InferenceEngine Engine, FakeGgufModelLoader Loader) NewGgufEngine()
    {
        var loader = new FakeGgufModelLoader();
        var engine = (InferenceEngine)new InferenceEngineFactory(loader,
            new DecisionSessionFactory(), new OsExecutionProviderDefaultSource())
            .Create(InferenceEngineBuilder.Gguf("/m/m.gguf").WithContextSize(512).Build());
        return (engine, loader);
    }

    private static InferenceEngine NewDecisionEngine(DecisionTestAssets assets)
    {
        var tmp = Path.Combine(Path.GetTempPath(), $"facade-{Guid.NewGuid():N}.onnx");
        assets.WriteTinyModel(tmp);
        var factory = new InferenceEngineFactory(new FakeGgufModelLoader(),
            new DecisionSessionFactory(), new OsExecutionProviderDefaultSource());
        return (InferenceEngine)factory.Create(InferenceEngineBuilder.Decision(tmp)
            .WithTokenizer(assets.TokenizerPath).WithSpec(assets.ShellSpecPath).Build());
    }

    [Fact]
    public void GgufInitialize_LoadsModelAndBuildsOps()
    {
        var (engine, loader) = NewGgufEngine();
        using (engine)
        {
            engine.Initialize();
            Assert.True(engine.IsLoaded);
            Assert.Single(loader.Loaded);
            Assert.Equal("/m/m.gguf", loader.Loaded[0].Path);
            Assert.Equal(512u, engine.Gguf.ContextSettings.ContextSize);
            engine.Initialize(); // idempotent
            Assert.Single(loader.Loaded);
        }
    }

    [Fact]
    public void GgufInitialize_DisposesModelOnFailure()
    {
        var loader = new FakeGgufModelLoader
        {
            ModelFactory = () => throw new InvalidOperationException("native load exploded"),
        };
        var engine = new InferenceEngineFactory(loader, new DecisionSessionFactory(),
            new OsExecutionProviderDefaultSource())
            .Create(InferenceEngineBuilder.Gguf("/m/m.gguf").Build());
        Assert.Throws<InvalidOperationException>(engine.Initialize);
        Assert.False(engine.IsLoaded);
    }

    [Fact]
    public void Views_FailClosed_OnKindMismatch()
    {
        var (engine, _) = NewGgufEngine();
        using (engine)
        {
            engine.Initialize();
            Assert.Throws<EngineKindMismatchException>(() => engine.Decision);
            Assert.Throws<EngineKindMismatchException>(() => engine.Encoder);
        }
    }

    [Fact]
    public void UnloadedEngine_ViewsThrowMismatch()
    {
        var (engine, _) = NewGgufEngine();
        using (engine)
        {
            Assert.False(engine.IsLoaded);
            Assert.Throws<EngineKindMismatchException>(() => engine.Gguf);
        }
    }

    [Fact]
    public void EncoderLane_LoadsVisionEncoder()
    {
        var loader = new FakeGgufModelLoader();
        var engine = new InferenceEngineFactory(loader, new DecisionSessionFactory(),
            new OsExecutionProviderDefaultSource())
            .Create(InferenceEngineBuilder.Encoder("/m/m.gguf").WithMmproj("/m/mm.gguf").Build());
        using (engine)
        {
            engine.Initialize();
            Assert.True(engine.IsLoaded);
            Assert.Equal(EngineKind.Encoder, engine.Kind);
            Assert.Equal("[img]", engine.Encoder.ImageMarker);
            Assert.Single(loader.Vision);
            Assert.Equal("/m/mm.gguf", loader.Vision[0].Item2);
            Assert.Throws<EngineKindMismatchException>(() => engine.Gguf);
        }
    }

    [Fact]
    public void DecisionLane_LiveTinyOnnx()
    {
        var assets = new DecisionTestAssets();
        var engine = NewDecisionEngine(assets);
        using (engine)
        {
            engine.Initialize();
            Assert.True(engine.IsLoaded);
            Assert.Equal(EngineKind.Decision, engine.Kind);
            var active = engine.Decision.ActiveProvider;
            Assert.False(string.IsNullOrWhiteSpace(active)); // truthful provider (cpu or coreml here)
            Assert.Contains(active, new[] { "cpu", "coreml" }, StringComparer.OrdinalIgnoreCase);
            Assert.NotNull(engine.Decision.Tokenizer);
            Assert.NotNull(engine.Decision.Spec);
            var score = engine.Decision.Judge(new[]
            {
                new DecisionRecord { Kind = JudgeRecordKind.Shell, Command = "ls -la", Cwd = "/tmp" },
            });
            Assert.Single(score);
            Assert.Throws<EngineKindMismatchException>(() => engine.Gguf);
        }
    }

    [Fact]
    public void Dispose_IsIdempotent_AndDisposesBackend()
    {
        var model = new FakeEngineModel();
        var loader = new FakeGgufModelLoader { ModelFactory = () => model };
        var engine = new InferenceEngineFactory(loader, new DecisionSessionFactory(),
            new OsExecutionProviderDefaultSource())
            .Create(InferenceEngineBuilder.Gguf("/m/m.gguf").Build());
        engine.Initialize();
        engine.Dispose();
        engine.Dispose(); // idempotent
        Assert.False(engine.IsLoaded);
        Assert.True(model.Disposed);
        Assert.Throws<ObjectDisposedException>(() => engine.Initialize());
    }
}