using ProvostXInference.Interop;
using ProvostXInference.Models;
using Xunit;

namespace ProvostXInference.Tests;

/// <summary>
/// Selector matrix (BL-02 stage 1): every (engine, OS) pair of the 5-product
/// matrix, INCLUDING all fail-closed pairs — unknown engine, wrong-OS selection,
/// null/empty engine. Fail-closed = typed error, never a guessed default.
/// </summary>
public class PvxEngineSelectorTests
{
    private static PvxEngineSelector SelectorFor(IOSProbe os) => new(new PvxEnginePolicy(), os);

    // ── Valid pairs: exactly the 5-product matrix ──

    [Theory]
    [InlineData("metal")]
    public void Metal_Valid_OnMacOS(string engine)
    {
        var result = SelectorFor(FakeOsProbe.MacOS()).Validate(engine);
        Assert.True(result.IsValid);
        Assert.Equal("metal", result.Spec!.Engine);
        Assert.Equal("macos", result.Spec.OS);
    }

    [Theory]
    [InlineData("cuda", "windows")]
    [InlineData("vulkan", "windows")]
    public void Cuda_Vulkan_Valid_OnWindows(string engine, string expectedOs)
    {
        var result = SelectorFor(FakeOsProbe.Windows()).Validate(engine);
        Assert.True(result.IsValid);
        Assert.Equal(engine, result.Spec!.Engine);
        Assert.Equal(expectedOs, result.Spec.OS);
    }

    [Theory]
    [InlineData("cuda", "linux")]
    [InlineData("vulkan", "linux")]
    public void Cuda_Vulkan_Valid_OnLinux(string engine, string expectedOs)
    {
        var result = SelectorFor(FakeOsProbe.Linux()).Validate(engine);
        Assert.True(result.IsValid);
        Assert.Equal(engine, result.Spec!.Engine);
        Assert.Equal(expectedOs, result.Spec.OS);
    }

    // ── Fail-closed pairs: wrong-OS selections MUST NOT resolve ──

    [Theory]
    [InlineData("vulkan")]
    [InlineData("cuda")]
    public void Cuda_Vulkan_OnMacOS_FailClosed(string engine)
    {
        var result = SelectorFor(FakeOsProbe.MacOS()).Validate(engine);
        Assert.False(result.IsValid);
        Assert.Equal(PvxEngineSelectionError.EngineNotAvailableOnThisOs, result.Error);
        Assert.NotNull(result.Message);
        Assert.Null(result.Spec);
    }

    [Fact]
    public void Metal_OnLinux_FailClosed()
    {
        var result = SelectorFor(FakeOsProbe.Linux()).Validate("metal");
        Assert.False(result.IsValid);
        Assert.Equal(PvxEngineSelectionError.EngineNotAvailableOnThisOs, result.Error);
        Assert.Null(result.Spec);
    }

    [Fact]
    public void Metal_OnWindows_FailClosed()
    {
        var result = SelectorFor(FakeOsProbe.Windows()).Validate("metal");
        Assert.False(result.IsValid);
        Assert.Equal(PvxEngineSelectionError.EngineNotAvailableOnThisOs, result.Error);
        Assert.Null(result.Spec);
    }

    // ── Fail-closed: unknown / empty / null ──

    [Theory]
    [InlineData("opencl")]
    [InlineData("METAL")]
    [InlineData("ggml")]
    [InlineData("Vulkan")]
    public void UnknownEngine_FailClosed(string engine)
    {
        var result = SelectorFor(FakeOsProbe.MacOS()).Validate(engine);
        Assert.False(result.IsValid);
        Assert.Equal(PvxEngineSelectionError.UnknownEngine, result.Error);
        Assert.Null(result.Spec);
    }

    [Fact]
    public void CpuEngine_IsKnown_SinceSpecialistMatrixAddition()
    {
        // caba7d1 added the cpu engine id to the specialist matrix — it must
        // no longer be treated as unknown (regression guard for the matrix
        // addition being reflected in validation).
        var policy = new PvxEnginePolicy();
        Assert.Contains("cpu", policy.KnownEngines);
        Assert.True(policy.IsKnownEngine("cpu"));
    }

    [Fact]
    public void NullEngine_FailClosed_AsUnknown()
    {
        var result = SelectorFor(FakeOsProbe.MacOS()).Validate(null);
        Assert.False(result.IsValid);
        Assert.Equal(PvxEngineSelectionError.UnknownEngine, result.Error);
    }

    [Fact]
    public void EmptyEngine_FailClosed_AsUnknown()
    {
        var result = SelectorFor(FakeOsProbe.Linux()).Validate("");
        Assert.False(result.IsValid);
        Assert.Equal(PvxEngineSelectionError.UnknownEngine, result.Error);
    }

    [Fact]
    public void WhitespaceEngine_IsTrimmedAndValidated()
    {
        var result = SelectorFor(FakeOsProbe.MacOS()).Validate("  metal  ");
        Assert.True(result.IsValid);
        Assert.Equal("metal", result.Spec!.Engine);
    }

    // ── OS probe identity (no OS overlap — exactly one is true) ──

    [Fact]
    public void SystemOsProbe_IdentifiesExactlyOneOs()
    {
        var probe = new SystemOsProbe();
        var count = (probe.IsMacOS() ? 1 : 0) + (probe.IsLinux() ? 1 : 0) + (probe.IsWindows() ? 1 : 0);
        Assert.Equal(1, count);
    }
}