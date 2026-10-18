using ProvostXInference.Abstractions;

namespace ProvostXInference.Engines;

/// <summary>Encoder lane ops: wraps the vision encoder + backing model (see interface).</summary>
public sealed class EncoderEngineOps : IEncoderEngineOps, IDisposable
{
    private readonly IVisionEncoder _vision;
    private readonly IInferenceModel _model;

    public EncoderEngineOps(IInferenceModel model, IVisionEncoder vision)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(vision);
        _model = model;
        _vision = vision;
    }

    public IVisionEncoder VisionEncoder => _vision;
    public IInferenceModel Model => _model;
    public float[] EmbedText(IInferenceContext context, string text) => _model.GetEmbeddings(context, text);
    public string ImageMarker => _vision.Marker;
    public bool NeedsNonCausalAttention => _vision.NeedsNonCausalAttention;

    public void Dispose() => _vision.Dispose();
}