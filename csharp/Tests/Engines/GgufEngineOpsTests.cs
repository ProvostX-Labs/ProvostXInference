using ProvostXInference.Engines;
using ProvostXInference.Models;
using Xunit;

namespace ProvostXInference.Tests.Engines;

/// <summary><see cref="GgufEngineOps"/>: delegation wiring over the fake
/// model/context/pool graph — context settings, sampling knobs, pool laziness,
/// KV ops, state/checkpoint plumbing.</summary>
public class GgufEngineOpsTests
{
    private static GgufEngineOps NewOps(FakeEngineModel? model = null, ContextConfig? settings = null,
        FakeEngineBackend? backend = null)
    {
        model ??= new FakeEngineModel();
        settings ??= new ContextConfig { ContextSize = 256, BatchSize = 32, SeqMax = 2 };
        return new GgufEngineOps(model, settings, new SamplingConfig(), backend ?? new FakeEngineBackend());
    }

    [Fact]
    public void Ctor_CreatesContext_FromSettings()
    {
        var model = new FakeEngineModel();
        var ops = NewOps(model, new ContextConfig { ContextSize = 99 });
        Assert.Same(model, ops.Model);
        Assert.Equal(99u, ops.ContextSettings.ContextSize);
        // FakeEngineModel.CreateContext ignores the config beyond recording.
    }

    [Fact]
    public void Sampling_Knob_IsMutable()
    {
        var ops = NewOps();
        Assert.Equal(0.3f, ops.Sampling.Temperature); // lane default
        ops.Sampling = new SamplingConfig { Temperature = 0.9f, TopK = 7 };
        Assert.Equal(0.9f, ops.Sampling.Temperature);
        Assert.Equal(7, ops.Sampling.TopK);
    }

    [Fact]
    public void Backend_ReadOnlyProbe_ServesLiveContext()
    {
        var backend = new FakeEngineBackend { NextInfo = new BackendInfo { Name = "probe-mock", ProbePassed = true } };
        var ops = NewOps(backend: backend);
        var info = ops.Backend;
        Assert.Equal("probe-mock", info.Name);
        Assert.True(info.ProbePassed);
        Assert.Same(ops.Context, backend.ProbedContexts.Single()); // live context, per-context probe
    }

    [Fact]
    public void Backend_AfterRebuildContext_ProbesTheNewContext()
    {
        var backend = new FakeEngineBackend();
        var ops = NewOps(backend: backend);
        var oldContext = ops.Context;
        ops.RebuildContext();
        _ = ops.Backend;
        Assert.NotSame(oldContext, backend.ProbedContexts[^1]); // stale context never probed again
    }

    [Fact]
    public void Pool_IsLazy_ThenCached()
    {
        var ops = NewOps();
        var pool1 = ops.Pool;
        Assert.NotNull(pool1);
        Assert.Same(pool1, ops.Pool);
    }

    [Fact]
    public void RebuildContext_ReplacesContext_AndDisposesOldPool()
    {
        var ops = NewOps();
        var oldPool = ops.Pool;
        var oldContext = ops.Context;
        ops.ContextSettings = new ContextConfig { ContextSize = 512 };
        ops.RebuildContext();
        Assert.NotSame(oldContext, ops.Context);
        Assert.True((oldPool as FakeEnginePool)!.Disposed);
        Assert.NotSame(oldPool, ops.Pool); // new lazy pool
        Assert.Equal(512u, ops.ContextSettings.ContextSize);
    }

    [Fact]
    public void LeaseConversation_ComesFromPool()
    {
        var ops = NewOps();
        var conv = ops.LeaseConversation();
        Assert.IsType<FakeEngineConversation>(conv);
        Assert.Equal(1, ((FakeEnginePool)ops.Pool).LeaseCount);
    }

    [Fact]
    public void KV_Ops_DelegateToContext()
    {
        var ops = NewOps();
        ops.KvCopy(2, 3);
        ops.KvClearSeq(3);
        var ctx = (FakeEngineContext)ops.Context;
        Assert.Contains((2, 3), ctx.KvCopies);
        Assert.Contains(3, ctx.ClearedSeqs);
        ops.SetCausalAttention(false);
        Assert.False(ctx.Causal);
    }

    [Fact]
    public void SaveRestore_Checkpoint_PlumbThrough()
    {
        var ops = NewOps();
        var pool = ops.Pool;
        var conv = ops.LeaseConversation();
        var state = ops.SaveConversationState(conv);
        Assert.IsType<FakeEngineConversationState>(state);
        ops.RestoreConversationState(pool, state);
        Assert.Equal(2, ((FakeEnginePool)pool).LeaseCount); // test lease + restore lease
        Assert.Equal(1, ((FakeEnginePool)pool).ReturnCount); // restore path only
        var bytes = ops.ExportConversationCheckpoint(conv);
        Assert.Equal(new byte[] { 1, 2, 3 }, bytes);
        ops.ImportConversationCheckpoint(conv, new byte[] { 9 });
        Assert.Contains(new byte[] { 9 }, ((FakeEngineConversation)conv).Imports);
    }

    [Fact]
    public void NullGuards_FailClosed()
    {
        var ops = NewOps();
        Assert.Throws<ArgumentNullException>(() => ops.SaveConversationState(null!));
        Assert.Throws<ArgumentNullException>(() => ops.RestoreConversationState(null!, null!));
        Assert.Throws<ArgumentNullException>(() => ops.ExportConversationCheckpoint(null!));
        Assert.Throws<ArgumentNullException>(() => ops.ImportConversationCheckpoint(null!, new byte[1]));
        Assert.Throws<ArgumentNullException>(() => ops.ImportConversationCheckpoint(ops.LeaseConversation(), null!));
    }

    [Fact]
    public void Template_Tokenize_Speculation_Delegate()
    {
        var ops = NewOps();
        Assert.Equal("[user] hi", ops.ApplyChatTemplate(null, new[] { ("user", "hi") }));
        Assert.Equal(new[] { 1, 2 }, ops.Tokenize("hello"));
        Assert.Equal("fake", ops.Detokenize(new[] { 1 }));
        Assert.True(ops.PartialRewindSupported);
        Assert.Equal(SpeculativeCapabilities.None, ops.SpeculativeCapabilities);
        ops.ValidateSpeculation(new SpeculativeConfig());
    }
}