namespace ProvostXInference.Engines;

/// <summary>
/// Resolves a validated <see cref="EngineRequest"/> into an UNLOADED
/// <see cref="IInferenceEngine"/>. Constructor-injected backend seams keep this
/// unit-testable without native images.
/// </summary>
public interface IInferenceEngineFactory
{
    /// <summary>Create an engine for a request (validates fail-closed first).</summary>
    IInferenceEngine Create(EngineRequest request);
}
