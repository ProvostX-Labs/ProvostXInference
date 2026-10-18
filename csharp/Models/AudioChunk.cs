namespace ProvostXInference.Models;

/// <summary>
/// ABI v8 (wave 3, feature 6): one raw PCM audio clip for prompt-with-audio.
/// Mono F32 samples at the projector's NATIVE sample rate (probe via
/// IInferenceModel.AudioSampleRate; product code never resamples — a rate
/// mismatch is the caller's typed failure). One clip per record.
/// </summary>
public sealed record AudioChunk(float[] PcmMono);
