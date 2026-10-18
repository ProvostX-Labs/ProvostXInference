using ProvostXInference.Abstractions;
using ProvostXInference.Models;

namespace ProvostXInference.Implementation;

/// <summary>Rolling-median stall detector (INF BL-33). A chunk is a stall when
/// its wall exceeds GapMultiplier × the rolling median of the recent baseline
/// window. The detector is silent (never a stall) until the baseline holds at
/// least MinBaselineSamples clean observations — the first chunks of a prefill
/// cannot be judged against an empty baseline.</summary>
public sealed class RollingPrefillStallDetector : IPrefillStallDetector
{
    /// <summary>Minimum baseline observations before stall judgment activates.</summary>
    public const int MinBaselineSamples = 3;

    private readonly double _gapMultiplier;
    private readonly int _window;
    private readonly Queue<double> _recent;

    public RollingPrefillStallDetector(StallWatchdogConfig config)
    {
        _gapMultiplier = config.GapMultiplier;
        _window = config.BaselineWindow;
        _recent = new Queue<double>(_window);
    }

    public PrefillStallVerdict Observe(int chunkIndex, int chunkTokens, double wallSeconds)
    {
        if (_recent.Count < MinBaselineSamples)
        {
            Record(wallSeconds);
            return PrefillStallVerdict.Clean(chunkIndex, chunkTokens, wallSeconds);
        }

        var median = Median(_recent);
        var ratio = median > 0 ? wallSeconds / median : 0.0;
        var isStall = wallSeconds > _gapMultiplier * median;
        if (!isStall) Record(wallSeconds);
        // A stalled chunk's wall is deliberately NOT recorded into the baseline:
        // the baseline must describe healthy chunk throughput, not the stall.

        return new PrefillStallVerdict
        {
            IsStall = isStall,
            ChunkIndex = chunkIndex,
            ChunkTokens = chunkTokens,
            ChunkWallSeconds = wallSeconds,
            BaselineMedianSeconds = median,
            Ratio = ratio,
        };
    }

    private void Record(double wallSeconds)
    {
        if (_recent.Count == _window) _recent.Dequeue();
        _recent.Enqueue(wallSeconds);
    }

    private static double Median(Queue<double> source)
    {
        var sorted = source.ToArray();
        Array.Sort(sorted);
        int n = sorted.Length;
        return n % 2 == 1 ? sorted[n / 2] : (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0;
    }
}