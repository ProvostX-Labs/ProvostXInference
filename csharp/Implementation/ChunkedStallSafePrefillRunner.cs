using ProvostXInference.Abstractions;
using ProvostXInference.Exceptions;
using ProvostXInference.Models;

namespace ProvostXInference.Implementation;

/// <summary>Engine-level stall-safe chunked prefill layer (INF BL-33).
/// Independent of downstream orchestration AND backend-independent by
/// construction: it leases conversations from the injected pool, commits the
/// prompt in chunks via PromptTokens, and times each chunk commit with pure
/// wall-clock around the synchronous native prompt call — the pvxinference
/// substrate dispatches to whatever backend is compiled in (metal / cuda /
/// vulkan / cpu), and this layer never knows or cares which. No backend
/// timing hook exists or is required; wall-clock is the sole detection
/// mechanism, so detection behaves identically on every backend. On stall it
/// walks the configured recovery ladder (fresh-slot retry → smaller-chunk
/// retry → draft/ngram disabled → fail loud).
///
/// Honesty notes:
/// - A fresh-slot retry re-prefills the committed prefix on the new slot (the
///   new slot starts with an empty KV), so it is correct, not cheap.
/// - The draft/ngram-disabled step suspends the speculative session through a
///   constructor-injected <see cref="ISpeculationSuspender"/> (real native
///   wiring: pvx_executor_spec_free via ExecutorSpeculationSuspender). When
///   no suspender is wired, that step fails closed with the typed unrecovered
///   error naming the missing seam — never a silent skip.
/// - With <see cref="StallWatchdogConfig.Enabled"/> = false the runner performs
///   a single plain prompt commit (watchdog fully inert).</summary>
public sealed class ChunkedStallSafePrefillRunner
{
    private readonly IConversationPool _pool;
    private readonly StallWatchdogConfig _config;
    private readonly IChunkWallClock _clock;
    private readonly IPrefillStallDetector _detector;
    private readonly StallRecoveryLadder _ladder;
    private readonly ISpeculationSuspender? _draftSuspender;

    public ChunkedStallSafePrefillRunner(
        IConversationPool pool,
        StallWatchdogConfig config,
        IChunkWallClock clock,
        IPrefillStallDetector detector,
        ISpeculationSuspender? draftSuspender = null)
    {
        _pool = pool ?? throw new ArgumentNullException(nameof(pool));
        _config = config ?? throw new ArgumentNullException(nameof(config));
        _clock = clock ?? throw new ArgumentNullException(nameof(clock));
        _detector = detector ?? throw new ArgumentNullException(nameof(detector));
        _draftSuspender = draftSuspender;
        _config.Validate();
        _ladder = new StallRecoveryLadder(_config);
    }

    /// <summary>Commit <paramref name="tokens"/> with stall-safe chunked
    /// prefill. Returns the total tokens committed (== tokens.Length on
    /// success). Throws <see cref="PrefillStallUnrecoveredException"/> when the
    /// ladder is exhausted.</summary>
    public int RunPromptTokens(int[] tokens)
    {
        ArgumentNullException.ThrowIfNull(tokens);
        if (tokens.Length == 0) return 0;

        if (!_config.Enabled)
        {
            var conv = _pool.Lease();
            try { _clock.Measure(() => conv.PromptTokens(tokens)); }
            finally { _pool.Return(conv); }
            return tokens.Length;
        }

        var conversation = _pool.Lease();
        try
        {
            var cursor = 0;                 // tokens resident in the CURRENT conversation's KV
            var chunkSize = _config.ChunkSizeTokens;
            var attempt = 0;
            var attemptedSteps = new List<PrefillRecoveryStep>();
            var chunkOrdinal = 0;

            while (cursor < tokens.Length)
            {
                var count = Math.Min(chunkSize, tokens.Length - cursor);
                var chunk = new int[count];
                Array.Copy(tokens, cursor, chunk, 0, count);

                var wall = _clock.Measure(() => conversation.PromptTokens(chunk));
                var verdict = _detector.Observe(chunkOrdinal++, count, wall);
                if (!verdict.IsStall)
                {
                    cursor += count;
                    continue;
                }

                var step = _ladder.StepFor(attempt);
                if (step is null || step == PrefillRecoveryStep.FailLoud)
                {
                    attemptedSteps.Add(PrefillRecoveryStep.FailLoud);
                    throw new PrefillStallUnrecoveredException(
                        verdict.ChunkIndex, verdict.ChunkTokens,
                        verdict.ChunkWallSeconds, verdict.BaselineMedianSeconds, attemptedSteps);
                }

                attemptedSteps.Add(step.Value);
                attempt++;

                switch (step.Value)
                {
                    case PrefillRecoveryStep.FreshSlotRetry:
                        _pool.Return(conversation);
                        conversation = _pool.Lease();
                        cursor = 0;         // fresh slot: empty KV, prefix must re-prefill
                        chunkOrdinal = 0;
                        break;

                    case PrefillRecoveryStep.SmallerChunkRetry:
                        chunkSize = Math.Max(_config.MinChunkSizeTokens, chunkSize / 2);
                        break;

                    case PrefillRecoveryStep.DraftDisabledRetry:
                        if (_draftSuspender is null)
                        {
                            // Fail closed: the hook is unwired, this step cannot
                            // be honored — surface it, never silently skip.
                            attemptedSteps.Add(PrefillRecoveryStep.FailLoud);
                            throw new PrefillStallUnrecoveredException(
                                verdict.ChunkIndex, verdict.ChunkTokens,
                                verdict.ChunkWallSeconds, verdict.BaselineMedianSeconds, attemptedSteps,
                                "draft/ngram suspension hook not wired into the stall-safe prefill runner");
                        }
                        _draftSuspender.Suspend();
                        break;
                }
            }

            return tokens.Length;
        }
        finally
        {
            _pool.Return(conversation);
        }
    }
}