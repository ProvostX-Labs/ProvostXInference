namespace ProvostXInference.Abstractions;

/// <summary>
/// ABI v8 (wave 3, feature 9): a loaded LoRA adapter (opaque native handle).
/// Loaded against ONE model (arch-matched, typed on mismatch); attached to
/// contexts via IInferenceContext.SetAdapters. Weights-surface modifier:
/// KV-neutral, attach/detach mid-conversation without rewind.
/// </summary>
public interface ILoraAdapter : IDisposable;
