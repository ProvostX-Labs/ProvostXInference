namespace ProvostXInference.Abstractions;

public interface IVisionEncoder : IDisposable
{
    string Marker { get; }
    bool NeedsNonCausalAttention { get; }
}
