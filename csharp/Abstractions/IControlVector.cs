namespace ProvostXInference.Abstractions;

/// <summary>
/// ABI v8 (wave 3, feature 10): a loaded control-vector direction file
/// (direction.<layer> F32 tensors, n_embd-matched to the model). Applied to
/// contexts via IInferenceContext.SetControlVector; weights-surface modifier
/// (KV-neutral, live attach/detach).
/// </summary>
public interface IControlVector : IDisposable;
