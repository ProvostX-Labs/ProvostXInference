namespace ProvostXInference.Abstractions;

/// <summary>Wall-clock seam for per-chunk prefill timing (INF BL-33).
/// The native prompt commit is synchronous, so the wall measured around it IS
/// the chunk wall on whatever backend is compiled in (metal / cuda / vulkan /
/// cpu — backend-independent by construction); the seam exists so tests can
/// script timings instead of sleeping.</summary>
public interface IChunkWallClock
{
    /// <summary>Execute <paramref name="work"/> once and return its wall time
    /// in seconds (fractional).</summary>
    double Measure(Action work);
}