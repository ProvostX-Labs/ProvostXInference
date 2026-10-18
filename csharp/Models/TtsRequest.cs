namespace ProvostXInference.Models;

/// <summary>
/// ABI v8 (wave 3, feature 7): TTS knobs. Zero values select the per-model
/// upstream defaults (native mtmd_gen_inp_default); Seed == uint.MaxValue
/// selects a random draw (fixed seeds replay byte-identically).
/// </summary>
public sealed record TtsRequest
{
    public string? Lang { get; init; }
    public int TopK { get; init; }
    public float TopP { get; init; }
    public float Temp { get; init; }
    public uint Seed { get; init; } = uint.MaxValue;
    public int MaxFrames { get; init; } = 512;
}
