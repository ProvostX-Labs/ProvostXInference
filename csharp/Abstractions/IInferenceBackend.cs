using ProvostXInference.Models;

namespace ProvostXInference.Abstractions;

public interface IInferenceBackend
{
    BackendInfo Probe(IInferenceContext context);
    bool VramAvailable(long neededBytes);
}
