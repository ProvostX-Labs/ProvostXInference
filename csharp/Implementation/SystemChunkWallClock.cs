using ProvostXInference.Abstractions;

namespace ProvostXInference.Implementation;

/// <summary>Production wall clock backed by the injected <see cref="TimeProvider"/>
/// (constructor injection, no statics — INF BL-33).</summary>
public sealed class SystemChunkWallClock : IChunkWallClock
{
    private readonly TimeProvider _timeProvider;

    public SystemChunkWallClock(TimeProvider timeProvider)
    {
        _timeProvider = timeProvider ?? throw new ArgumentNullException(nameof(timeProvider));
    }

    public double Measure(Action work)
    {
        ArgumentNullException.ThrowIfNull(work);
        var start = _timeProvider.GetTimestamp();
        work();
        return _timeProvider.GetElapsedTime(start).TotalSeconds;
    }
}