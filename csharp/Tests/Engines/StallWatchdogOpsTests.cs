using ProvostXInference.Abstractions;
using ProvostXInference.Engines;
using ProvostXInference.Models;
using Xunit;

namespace ProvostXInference.Tests.Engines;

/// <summary>Facade tests (owner black-box directive 2026-10-19):
/// StallWatchdogOps is the sole construction seam for the stall-watchdog
/// helper family — callers hold the Abstractions interfaces and never touch
/// the Implementation namespace. These tests pin the delegation: the
/// factories return working detector/ladder/wall-clock instances with the
/// same semantics as the underlying helpers. No native library involved.</summary>
public class StallWatchdogOpsTests
{
    private static StallWatchdogConfig Cfg(double mult = 5.0, int retries = 3, int chunk = 2, int window = 8) =>
        new() { GapMultiplier = mult, MaxRetries = retries, ChunkSizeTokens = chunk, MinChunkSizeTokens = 1, BaselineWindow = window };

    [Fact]
    public void CreateDetector_Returns_Working_RollingDetector()
    {
        IPrefillStallDetector d = StallWatchdogOps.CreateDetector(Cfg());
        Assert.False(d.Observe(0, 2, 1.0).IsStall);
        Assert.False(d.Observe(1, 2, 1.0).IsStall);
        Assert.False(d.Observe(2, 2, 1.0).IsStall);
        Assert.True(d.Observe(3, 2, 100.0).IsStall);
    }

    [Fact]
    public void CreateRecoveryLadder_Steps_Through_The_Configured_Order()
    {
        IStallRecoveryLadder ladder = StallWatchdogOps.CreateRecoveryLadder(Cfg(retries: 2));
        Assert.Equal(2, ladder.MaxRetries);
        Assert.NotNull(ladder.StepFor(0));
        Assert.NotNull(ladder.StepFor(1));
        Assert.Null(ladder.StepFor(2));
    }

    [Fact]
    public void CreateWallClock_Measures_Real_Elapsed_Time()
    {
        IChunkWallClock clock = StallWatchdogOps.CreateWallClock(TimeProvider.System);
        var wall = clock.Measure(() => Thread.Sleep(20));
        Assert.True(wall >= 0.015, $"measured {wall}s, expected >= 0.015s");
    }

    [Fact]
    public void Factories_Throw_On_Null_Config()
    {
        Assert.Throws<ArgumentNullException>(() => StallWatchdogOps.CreateDetector(null!));
        Assert.Throws<ArgumentNullException>(() => StallWatchdogOps.CreateRecoveryLadder(null!));
        Assert.Throws<ArgumentNullException>(() => StallWatchdogOps.CreateWallClock(null!));
    }
}
