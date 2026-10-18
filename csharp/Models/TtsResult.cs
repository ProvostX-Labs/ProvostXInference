namespace ProvostXInference.Models;

/// <summary>TTS generation result: raw PCM F32 mono at the pipeline rate.</summary>
public sealed record TtsResult(float[] PcmMono, int SampleRate);
