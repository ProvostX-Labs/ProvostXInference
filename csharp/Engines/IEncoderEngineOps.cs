using ProvostXInference.Abstractions;

namespace ProvostXInference.Engines;

/// <summary>
/// Encoder lane ops view: image embedding through the mmproj projector mounted
/// on the GGUF model's context. Text embeddings are reachable via the Gguf-style
/// model context; this view keeps the vision lane honest and minimal.
/// </summary>
public interface IEncoderEngineOps
{
    /// <summary>The loaded mmproj vision encoder (raw abstraction escape hatch).</summary>
    IVisionEncoder VisionEncoder { get; }

    /// <summary>The backing GGUF model (embedding dimension, template, tokenizer).</summary>
    IInferenceModel Model { get; }

    /// <summary>Text embeddings from the backing model's context.</summary>
    float[] EmbedText(IInferenceContext context, string text);

    /// <summary>The image marker tag the projector injects into prompts.</summary>
    string ImageMarker { get; }

    /// <summary>Whether the projector needs non-causal attention (set via the context).</summary>
    bool NeedsNonCausalAttention { get; }
}