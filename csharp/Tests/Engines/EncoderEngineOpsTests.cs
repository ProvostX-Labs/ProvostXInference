using ProvostXInference.Engines;
using ProvostXInference.Models;
using Xunit;

namespace ProvostXInference.Tests.Engines;

/// <summary><see cref="EncoderEngineOps"/>: vision-encoder wiring over fakes.</summary>
public class EncoderEngineOpsTests
{
    private static (EncoderEngineOps Ops, FakeEngineModel Model, FakeEngineVisionEncoder Vision) NewOps()
    {
        var model = new FakeEngineModel();
        var vision = new FakeEngineVisionEncoder();
        return (new EncoderEngineOps(model, vision), model, vision);
    }

    [Fact]
    public void WrapsVision_AndModel()
    {
        var (ops, model, vision) = NewOps();
        Assert.Same(model, ops.Model);
        Assert.Same(vision, ops.VisionEncoder);
        Assert.Equal("[img]", ops.ImageMarker);
        Assert.False(ops.NeedsNonCausalAttention);
    }

    [Fact]
    public void EmbedText_DelegatesToModel()
    {
        var (ops, model, _) = NewOps();
        var context = model.CreateContext(new ContextConfig());
        var emb = ops.EmbedText(context, "hello");
        Assert.Equal(new[] { 0.5f, 0.25f }, emb);
    }

    [Fact]
    public void Ctor_FailsClosed_OnNulls()
    {
        Assert.Throws<ArgumentNullException>(() => new EncoderEngineOps(null!, new FakeEngineVisionEncoder()));
        Assert.Throws<ArgumentNullException>(() => new EncoderEngineOps(new FakeEngineModel(), null!));
    }

    [Fact]
    public void Dispose_DisposesVisionOnly()
    {
        var (ops, model, vision) = NewOps();
        ops.Dispose();
        Assert.True(vision.Disposed);
        Assert.False(model.Disposed); // model belongs to the engine, not the ops view
    }
}