using System.Runtime.InteropServices;

namespace ProvostXInference.Interop;

/// <summary>Blittable mirror of pvx_sampling_params_t. Field order must match the C struct exactly.</summary>
[StructLayout(LayoutKind.Sequential)]
internal readonly struct PvxSamplingParams
{
    public readonly float Temperature;
    public readonly float TopP;
    public readonly int TopK;
    public readonly float MinP;
    public readonly float RepeatPenalty;
    public readonly int RepeatLastN;
    public readonly float PenaltyPresent;
    public readonly int MaxTokens;
    public readonly byte IgnoreEos;

    public PvxSamplingParams(
        float temperature, float topP, int topK, float minP,
        float repeatPenalty, int repeatLastN, float penaltyPresent,
        int maxTokens, bool ignoreEos)
    {
        Temperature = temperature;
        TopP = topP;
        TopK = topK;
        MinP = minP;
        RepeatPenalty = repeatPenalty;
        RepeatLastN = repeatLastN;
        PenaltyPresent = penaltyPresent;
        MaxTokens = maxTokens;
        IgnoreEos = ignoreEos ? (byte)1 : (byte)0;
    }
}
