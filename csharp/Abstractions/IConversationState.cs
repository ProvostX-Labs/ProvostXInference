namespace ProvostXInference.Abstractions;

public interface IConversationState : IDisposable
{
    int TokenCount { get; }
}
