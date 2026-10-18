using ProvostXInference.Decision.Engine;
using ProvostXInference.Engines;
using ProvostXInference.Tests.Decision;
using Xunit;

namespace ProvostXInference.Tests.Engines;

/// <summary><see cref="DecisionEngineOps"/>: wrapping seams over a REAL
/// tiny-ONNX session (parity with OnnxJudgeSessionFactoryTests' asset path).</summary>
public class DecisionEngineOpsTests
{
    private static DecisionEngineOps NewOps(DecisionTestAssets assets)
    {
        var tmp = Path.Combine(Path.GetTempPath(), $"ops-{Guid.NewGuid():N}.onnx");
        assets.WriteTinyModel(tmp);
        var session = new DecisionSessionFactory().Create(tmp, assets.NewTokenizer(),
            assets.NewShellSpec(), ExecutionProviderChain.Parse("cpu", "ops-test"));
        return new DecisionEngineOps(session);
    }

    [Fact]
    public void Wraps_Session_WithTruthfulProvider()
    {
        var assets = new DecisionTestAssets();
        using var ops = NewOps(assets);
        Assert.Equal("cpu", ops.ActiveProvider, ignoreCase: true);
        Assert.Same(ops.Tokenizer, ops.Tokenizer); // stable
        Assert.NotNull(ops.Spec);
    }

    [Fact]
    public void Judge_DelegatesToSession()
    {
        var assets = new DecisionTestAssets();
        using var ops = NewOps(assets);
        var scores = ops.Judge(new[]
        {
            new DecisionRecord { Kind = JudgeRecordKind.Shell, Command = "rm -rf /", Cwd = "/" },
            new DecisionRecord { Kind = JudgeRecordKind.Shell, Command = "ls", Cwd = "/tmp" },
        });
        Assert.Equal(2, scores.Count);
    }

    [Fact]
    public void JudgeQuestionSet_DelegatesToSession()
    {
        var assets = new DecisionTestAssets();
        using var ops = NewOps(assets);
        var answers = ops.JudgeQuestionSet(Array.Empty<JudgeStateQuestions>());
        Assert.Empty(answers);
    }
}