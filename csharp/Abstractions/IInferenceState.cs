namespace ProvostXInference.Abstractions;

public interface IInferenceState : IDisposable
{
    int TokenCount { get; }
}
