using ProvostXInference.Abstractions;
using ProvostXInference.Exceptions;
using ProvostXInference.Implementation;
using ProvostXInference.Models;
using Xunit;

namespace ProvostXInference.Tests;

/// <summary>Unit tests for the stall-safe prefill watchdog (INF BL-33):
/// rolling-median detector over synthetic timings, recovery-ladder ordering,
/// and runner integration over fakes. No native library involved.</summary>
public class StallWatchdogTests
{
    private static StallWatchdogConfig Cfg(double mult = 5.0, int retries = 3, int chunk = 2, int minChunk = 1, int window = 8) =>
        new() { GapMultiplier = mult, MaxRetries = retries, ChunkSizeTokens = chunk, MinChunkSizeTokens = minChunk, BaselineWindow = window };

    // ── RollingPrefillStallDetector ──────────────────────────

    [Fact]
    public void Detector_WarmingUp_NeverStalls()
    {
        var d = new RollingPrefillStallDetector(Cfg());
        Assert.False(d.Observe(0, 2, 1.0).IsStall);
        Assert.False(d.Observe(1, 2, 1.0).IsStall);
        // 3rd observation completes the baseline; even a huge wall before the
        // 4th observation is judged against a 3-sample median.
        Assert.False(d.Observe(2, 2, 1.0).IsStall);
        Assert.True(d.Observe(3, 2, 100.0).IsStall);
    }

    [Fact]
    public void Detector_NormalChunk_BelowMultiplier_IsClean()
    {
        var d = new RollingPrefillStallDetector(Cfg(mult: 5.0));
        for (int i = 0; i < 5; i++) d.Observe(i, 2, 1.0);
        var v = d.Observe(5, 2, 4.9);
        Assert.False(v.IsStall);
        Assert.Equal(1.0, v.BaselineMedianSeconds, precision: 5);
        Assert.Equal(4.9, v.ChunkWallSeconds, precision: 5);
    }

    [Fact]
    public void Detector_AtExactlyMultiplier_IsNotAStall()
    {
        var d = new RollingPrefillStallDetector(Cfg(mult: 5.0));
        for (int i = 0; i < 5; i++) d.Observe(i, 2, 1.0);
        Assert.False(d.Observe(5, 2, 5.0).IsStall);
    }

    [Fact]
    public void Detector_StallWall_NotFedIntoBaseline()
    {
        var d = new RollingPrefillStallDetector(Cfg(mult: 5.0));
        for (int i = 0; i < 5; i++) d.Observe(i, 2, 1.0);
        Assert.True(d.Observe(5, 2, 50.0).IsStall);
        // baseline median still 1.0 (stall wall excluded), next clean chunk judged vs 1.0
        var v = d.Observe(6, 2, 1.0);
        Assert.False(v.IsStall);
        Assert.Equal(1.0, v.BaselineMedianSeconds, precision: 5);
    }

    [Fact]
    public void Detector_ThroughputPhaseChange_KeepsFlagging_FailLoudByDesign()
    {
        // A sustained throughput phase change within one prefill is, by the
        // owner-specified semantics, indistinguishable from a pathological
        // stall chain at the detector level — the ladder owns the recovery
        // decision, so the detector keeps flagging (fail-loud mirrors the
        // RAM-gate error style). Stalled walls are never folded into the
        // baseline.
        var d = new RollingPrefillStallDetector(Cfg(mult: 5.0, window: 4));
        for (int i = 0; i < 6; i++) d.Observe(i, 2, 1.0);   // healthy ~1s chunks
        for (int i = 6; i < 12; i++)
        {
            var v = d.Observe(i, 2, 10.0);
            Assert.True(v.IsStall);
            Assert.Equal(1.0, v.BaselineMedianSeconds, precision: 5); // unchanged baseline
        }
    }

    // ── StallRecoveryLadder ──────────────────────────

    [Fact]
    public void Ladder_DefaultOrder_FreshSlot_SmallerChunk_DraftDisabled_FailLoud()
    {
        var ladder = new StallRecoveryLadder(Cfg());
        Assert.Equal(PrefillRecoveryStep.FreshSlotRetry, ladder.StepFor(0));
        Assert.Equal(PrefillRecoveryStep.SmallerChunkRetry, ladder.StepFor(1));
        Assert.Equal(PrefillRecoveryStep.DraftDisabledRetry, ladder.StepFor(2));
        Assert.Null(ladder.StepFor(3)); // budget exhausted → caller fails loud
    }

    [Fact]
    public void Ladder_ExhaustedBudget_ReturnsNull_FailClosed()
    {
        var ladder = new StallRecoveryLadder(Cfg(retries: 1));
        Assert.Equal(PrefillRecoveryStep.FreshSlotRetry, ladder.StepFor(0));
        Assert.Null(ladder.StepFor(1));
    }

    [Fact]
    public void Ladder_ShortStepList_FailsLoudBeyondList()
    {
        var cfg = Cfg(retries: 3) with { LadderSteps = new[] { PrefillRecoveryStep.SmallerChunkRetry, PrefillRecoveryStep.FailLoud } };
        var ladder = new StallRecoveryLadder(cfg);
        Assert.Equal(PrefillRecoveryStep.SmallerChunkRetry, ladder.StepFor(0));
        Assert.Equal(PrefillRecoveryStep.FailLoud, ladder.StepFor(1));
        Assert.Equal(PrefillRecoveryStep.FailLoud, ladder.StepFor(2));
    }

    // ── Config validation (fail-closed) ──────────────────────────

    [Fact]
    public void Config_ZeroMultiplier_Throws()
    {
        Assert.Throws<ArgumentException>(() => Cfg(mult: 0).Validate());
    }

    [Fact]
    public void Config_NegativeRetries_Throws()
    {
        var cfg = Cfg(retries: -1);
        Assert.Throws<ArgumentException>(() => cfg.Validate());
    }

    [Fact]
    public void Config_MinChunkAboveChunk_Throws()
    {
        var cfg = Cfg(chunk: 2, minChunk: 4);
        Assert.Throws<ArgumentException>(() => cfg.Validate());
    }

    [Fact]
    public void Config_LadderNotTerminatingInFailLoud_Throws()
    {
        var cfg = Cfg() with { LadderSteps = new[] { PrefillRecoveryStep.FreshSlotRetry } };
        Assert.Throws<ArgumentException>(() => cfg.Validate());
    }

    // ── ChunkedStallSafePrefillRunner (integration over fakes) ──

    [Fact]
    public void Runner_CleanPath_CommitsAllChunks_InOrder()
    {
        var pool = new FakeConversationPool();
        var clock = new FakeChunkWallClock(new[] { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 });
        var runner = new ChunkedStallSafePrefillRunner(pool, Cfg(), clock, new RollingPrefillStallDetector(Cfg()));

        var committed = runner.RunPromptTokens(new[] { 1, 2, 3, 4, 5, 6 });

        Assert.Equal(6, committed);
        Assert.Equal(1, pool.LeaseCount);
        var conv = Assert.IsType<FakeConversation>(pool.LastReturned);
        Assert.Equal(new[] { new[] { 1, 2 }, new[] { 3, 4 }, new[] { 5, 6 } }, conv.PromptedChunks);
    }

    [Fact]
    public void Runner_FreshSlotRetry_ReplaysPrefixOnNewSlot()
    {
        var pool = new FakeConversationPool();
        // chunks 0-2 clean on slot A (detector warming); chunk3 stalls →
        // fresh slot B (attempt 0) replays chunks 0-3, all clean.
        var clock = new FakeChunkWallClock(new[] { 1.0, 1.0, 1.0, 50.0, 1.0, 1.0, 1.0, 1.0 });
        var runner = new ChunkedStallSafePrefillRunner(pool, Cfg(), clock, new RollingPrefillStallDetector(Cfg()));

        var committed = runner.RunPromptTokens(new[] { 1, 2, 3, 4, 5, 6, 7, 8 });

        Assert.Equal(8, committed);
        Assert.Equal(2, pool.LeaseCount);
        Assert.Equal(2, pool.ReturnedFromLease.Count);
        var fresh = Assert.IsType<FakeConversation>(pool.LastReturned);
        // Fresh slot replays the full prefix + the stalled chunk: 4 chunks of 2
        Assert.Equal(4, fresh.PromptedChunks.Count);
        Assert.Equal(new[] { 1, 2 }, fresh.PromptedChunks[0]);
        Assert.Equal(new[] { 7, 8 }, fresh.PromptedChunks[^1]);
    }

    [Fact]
    public void Runner_SmallerChunkRetry_HalvesChunkAndRecovers()
    {
        var pool = new FakeConversationPool();
        // A: chunks 0-2 clean, chunk3 stalls → attempt0 FreshSlotRetry.
        // B: replay 0-2 clean, chunk3 stalls again → attempt1
        // SmallerChunkRetry halves 2 → 1; the retry commits as 1-token chunks.
        var clock = new FakeChunkWallClock(new[] { 1.0, 1.0, 1.0, 50.0, 1.0, 1.0, 1.0, 50.0, 1.0, 1.0 });
        var runner = new ChunkedStallSafePrefillRunner(pool, Cfg(), clock, new RollingPrefillStallDetector(Cfg()));

        var tokens = new[] { 1, 2, 3, 4, 5, 6, 7, 8 };
        var committed = runner.RunPromptTokens(tokens);

        Assert.Equal(tokens.Length, committed);
        Assert.Equal(2, pool.LeaseCount);
        var conv = Assert.IsType<FakeConversation>(pool.LastReturned);
        // B: 2+2+2 replay, stalled 2 (chunk [7,8]), then halved 1+1 retry of
        // the stalled chunk
        Assert.Equal(new[] { 2, 2, 2, 2, 1, 1 }, conv.PromptedChunks.Select(c => c.Length).ToArray());
    }

    [Fact]
    public void Runner_DraftDisabledRetry_InvokesSuspender()
    {
        var pool = new FakeConversationPool();
        var suspender = new FakeSpeculationSuspender();
        // A: chunks 0-2 clean, chunk3 stalls → FreshSlotRetry. B: replay 0-2
        // clean, chunk3 stalls → SmallerChunkRetry (size 1). Halved retry
        // chunk stalls → DraftDisabledRetry suspends the speculative session,
        // then all clean.
        var walls = new List<double>(Enumerable.Repeat(1.0, 20));
        walls[3] = 50.0;
        walls[7] = 50.0;
        walls[8] = 50.0; // first halved retry chunk stalls
        var clock = new FakeChunkWallClock(walls);
        var runner = new ChunkedStallSafePrefillRunner(pool, Cfg(), clock,
            new RollingPrefillStallDetector(Cfg()), suspender);

        var tokens = new[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
        var committed = runner.RunPromptTokens(tokens);

        Assert.Equal(tokens.Length, committed);
        Assert.Equal(1, suspender.SuspendCalls); // real rung activation
    }

    [Fact]
    public void ExecutorSpeculationSuspender_DisablesSpeculation_Idempotent()
    {
        var executor = new FakeStandardExecutor();
        var suspender = new ExecutorSpeculationSuspender(executor);

        suspender.Suspend();
        suspender.Suspend(); // second call = managed no-op (recorded)

        Assert.Equal(1, executor.DisableSpeculationCalls); // native teardown hit exactly once
    }

    [Fact]
    public void ExecutorSpeculationSuspender_NullExecutor_Throws()
    {
        Assert.Throws<ArgumentNullException>(() => new ExecutorSpeculationSuspender(null!));
    }

    [Fact]
    public void Runner_UnwiredDraftHook_FailsLoud_Typed()
    {
        var pool = new FakeConversationPool();
        // A: chunks 0-2 healthy (1s), chunk3 stalls → FreshSlotRetry.
        // B: replay healthy, chunk3 stalls → SmallerChunkRetry (size 1);
        // halved retry chunk still stalls → DraftDisabledRetry with NO hook
        // → fail closed.
        var walls = new List<double>(Enumerable.Repeat(1.0, 20));
        walls[3] = 50.0;
        walls[7] = 50.0;
        walls[8] = 50.0;
        var clock = new FakeChunkWallClock(walls);
        var runner = new ChunkedStallSafePrefillRunner(pool, Cfg(), clock, new RollingPrefillStallDetector(Cfg()));

        var ex = Assert.Throws<PrefillStallUnrecoveredException>(() => runner.RunPromptTokens(new[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 }));
        Assert.Contains("ladder exhausted", ex.Message);
        Assert.Contains("suspension hook not wired", ex.Message); // hook-miss surfaced, not skipped
        Assert.Equal(PrefillRecoveryStep.FreshSlotRetry, ex.AttemptedSteps[0]);
        Assert.Equal(PrefillRecoveryStep.SmallerChunkRetry, ex.AttemptedSteps[1]);
        Assert.Equal(PrefillRecoveryStep.FailLoud, ex.AttemptedSteps[^1]);
    }

    [Fact]
    public void Runner_PersistentStalls_ExhaustLadder_ThrowsTyped()
    {
        var pool = new FakeConversationPool();
        // Healthy chunks 1s, every post-warmup chunk a 50s stall: the ladder
        // cannot recover → typed fail-loud.
        var walls = new List<double>(Enumerable.Repeat(1.0, 20));
        walls[3] = 50.0;
        walls[7] = 50.0;
        walls[8] = 50.0;
        var clock = new FakeChunkWallClock(walls);
        var runner = new ChunkedStallSafePrefillRunner(pool, Cfg(), clock, new RollingPrefillStallDetector(Cfg()));

        var ex = Assert.Throws<PrefillStallUnrecoveredException>(() => runner.RunPromptTokens(new[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 }));
        // warming-up chunks pass; the first judged chunk (ordinal 3) triggers the ladder
        Assert.Equal(4, ex.ChunkIndex);
        Assert.Equal(PrefillRecoveryStep.FreshSlotRetry, ex.AttemptedSteps[0]);
        Assert.Equal(PrefillRecoveryStep.SmallerChunkRetry, ex.AttemptedSteps[1]);
        Assert.Equal(PrefillRecoveryStep.DraftDisabledRetry, ex.AttemptedSteps[2]); // hook unwired → fail closed
        Assert.Equal(PrefillRecoveryStep.FailLoud, ex.AttemptedSteps[^1]);
    }

    [Fact]
    public void Runner_Disabled_SingleCommit_NoDetector()
    {
        var pool = new FakeConversationPool();
        var clock = new FakeChunkWallClock(new[] { 1.0 });
        var cfg = Cfg() with { Enabled = false };
        var runner = new ChunkedStallSafePrefillRunner(pool, cfg, clock, new RollingPrefillStallDetector(cfg));

        var committed = runner.RunPromptTokens(new[] { 1, 2, 3, 4 });

        Assert.Equal(4, committed);
        var conv = Assert.IsType<FakeConversation>(pool.LastReturned);
        Assert.Single(conv.PromptedChunks); // one whole-prompt commit, no chunking
    }

    [Fact]
    public void Runner_EmptyPrompt_NoWork()
    {
        var pool = new FakeConversationPool();
        var clock = new FakeChunkWallClock(new List<double>());
        var runner = new ChunkedStallSafePrefillRunner(pool, Cfg(), clock, new RollingPrefillStallDetector(Cfg()));
        Assert.Equal(0, runner.RunPromptTokens(Array.Empty<int>()));
        Assert.Equal(0, pool.LeaseCount);
    }
}

// ── Fakes (test doubles, no dispatch behavior) ──────────────

/// <summary>Scripted wall clock: executes the work and returns the next
/// scripted duration — lets tests drive stall detection without sleeping.
/// Falls back to 1.0 once the script is exhausted.</summary>
internal sealed class FakeChunkWallClock : IChunkWallClock
{
    private readonly IEnumerator<double> _scripted;

    public FakeChunkWallClock(IEnumerable<double> scripted)
    {
        _scripted = scripted.GetEnumerator();
    }

    public double Measure(Action work)
    {
        work();
        return _scripted.MoveNext() ? _scripted.Current : 1.0;
    }
}

/// <summary>In-memory conversation: records PromptTokens chunk boundaries.
/// The stall-safe runner only needs the prompt-commit seam.</summary>
internal sealed class FakeConversation : IConversation
{
    public List<int[]> PromptedChunks { get; } = new();

    public int TokenCount => PromptedChunks.Sum(c => c.Length);
    public void Prompt(string text) => PromptedChunks.Add(new[] { text.Length });
    public void PromptTokens(int[] tokens) => PromptedChunks.Add((int[])tokens.Clone());
    public int Sample(SamplingConfig? config = null) => 0;
    public int SampleWithGrammar(SamplingConfig? config, IGrammar grammar) => 0;
    public void Rewind(int tokenCount) { }
    public void Reset() => PromptedChunks.Clear();
    public void ResetGrammarState() { }
    public void ShiftLeft(int tokenCount) { }
    public IConversationState SaveState() => null!;
    public void RestoreState(IConversationPool pool, IConversationState state) { }
    public byte[] ExportCheckpoint() => Array.Empty<byte>();
    public void ImportCheckpoint(byte[] data) { }
    public void PromptWithImages(string text, IVisionEncoder vision, params byte[][] images) { }
    public SpeculativeConfig? ActiveSpeculation => null;
    public void EnableSpeculation(SpeculativeConfig config) { }
    public void DisableSpeculation() { }
    public IReadOnlyList<int> GenerateSpeculative(SamplingConfig? config, int maxTokens, Action<int>? onToken = null) => Array.Empty<int>();
    public ProvostXInference.Models.SpeculationStats SpeculationStats() => ProvostXInference.Models.SpeculationStats.None;
    public void Dispose() { }
}

/// <summary>Suspender fake: counts Suspend calls for rung-activation
/// assertions.</summary>
internal sealed class FakeSpeculationSuspender : ISpeculationSuspender
{
    public int SuspendCalls { get; private set; }
    public void Suspend() => SuspendCalls++;
}

/// <summary>Executor fake: records the native speculative-session teardown
/// seam (the only member the suspender needs); all other members are unused
/// by the suspender seam.</summary>
internal sealed class FakeStandardExecutor : IStandardExecutor
{
    public int DisableSpeculationCalls { get; private set; }
    public SpeculativeConfig? ActiveSpeculation { get; private set; }
    public void DisableSpeculation() { DisableSpeculationCalls++; ActiveSpeculation = null; }
    public void EnableSpeculation(SpeculativeConfig config) => throw new NotSupportedException();
    public IReadOnlyList<int> GenerateSpeculative(SamplingConfig? config, int maxTokens, Action<int>? onToken = null) => throw new NotSupportedException();
    public int TokenCount => throw new NotSupportedException();
    public void Prompt(string text) => throw new NotSupportedException();
    public void PromptTokens(int[] tokens) => throw new NotSupportedException();
    public void PromptWithImages(string text, IInferenceModel model, params byte[][] images) => throw new NotSupportedException();
    public InferResult Infer() => throw new NotSupportedException();
    public int Sample(SamplingConfig? config = null) => throw new NotSupportedException();
    public int SampleWithGrammar(SamplingConfig? config, IGrammar grammar) => throw new NotSupportedException();
    public void Rewind(int tokenCount) => throw new NotSupportedException();
    public void Reset() => throw new NotSupportedException();
    public void ShiftLeft(int tokenCount) => throw new NotSupportedException();
    public IInferenceState SaveState() => throw new NotSupportedException();
    public void RestoreState(IInferenceState state) => throw new NotSupportedException();
    public void Dispose() { }
}

/// <summary>Pool fake: hands out fresh FakeConversations and records
/// lease/return bookkeeping for fresh-slot-retry assertions.</summary>
internal sealed class FakeConversationPool : IConversationPool
{
    private readonly List<FakeConversation> _live = new();

    public int LeaseCount { get; private set; }
    public List<IConversation> ReturnedFromLease { get; } = new();
    public FakeConversation LastReturned { get; private set; } = null!;

    public IConversation Lease()
    {
        LeaseCount++;
        var conv = new FakeConversation();
        _live.Add(conv);
        LastReturned = conv;
        return conv;
    }

    public void Return(IConversation conversation)
    {
        ReturnedFromLease.Add(conversation);
        var conv = (FakeConversation)conversation;
        _live.Remove(conv);
        LastReturned = conv;
    }

    public int AvailableCount => _live.Count;
    public uint Size => 4;
    public void Dispose() { }
}