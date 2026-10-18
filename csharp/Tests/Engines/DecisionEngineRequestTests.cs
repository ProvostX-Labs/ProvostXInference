using ProvostXInference.Exceptions;
using ProvostXInference.Models;
using ProvostXInference.Engines;
using Xunit;

namespace ProvostXInference.Tests.Engines;

/// <summary><see cref="DecisionEngineRequest"/>: required tokenizer/spec, EP chain, gate.</summary>
public class DecisionEngineRequestTests
{
    [Fact]
    public void MissingTokenizer_FailsClosed() =>
        Assert.Throws<InvalidEngineRequestException>(() =>
            InferenceEngineBuilder.Decision("/m/judge.onnx").Build());

    [Fact]
    public void MissingSpec_FailsClosed()
    {
        Assert.Throws<InvalidEngineRequestException>(() =>
            InferenceEngineBuilder.Decision("/m/judge.onnx").WithTokenizer("/m/tok.json").Build());
        Assert.Throws<InvalidEngineRequestException>(() =>
            InferenceEngineBuilder.Decision("/m/judge.onnx")
                .WithTokenizer(" ").WithSpec("/m/spec.json").Build());
    }

    [Fact]
    public void FullRequest_Maps()
    {
        var r = InferenceEngineBuilder.Decision("/m/judge.onnx").WithTokenizer("/m/tok.json")
            .WithSpec("/m/spec.json").WithExecutionProviders("coreml->cpu").Build();
        Assert.Equal("/m/judge.onnx", r.ModelPath);
        Assert.Equal("/m/tok.json", r.TokenizerJsonPath);
        Assert.Equal("/m/spec.json", r.SpecJsonPath);
        Assert.Equal("coreml->cpu", r.ExecutionProviders);
        Assert.Null(r.NumericsGate);
    }
}
