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
    /// <summary>ABI v8 wave-5 (BL-21a): appended sampler knobs —
    /// zeroed = pre-wave-5 chain. Embedded struct (blittable).</summary>
    public readonly PvxSamplingParamsWave5 Wave5;

    public PvxSamplingParams(
        float temperature, float topP, int topK, float minP,
        float repeatPenalty, int repeatLastN, float penaltyPresent,
        int maxTokens, bool ignoreEos,
        in PvxSamplingParamsWave5 wave5 = default)
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
        Wave5 = wave5;
    }
}

/// <summary>ABI v8 wave-5 (BL-21a): blittable pvx_sampling_params_wave5_t mirror.
/// Zero-value = pre-wave-5 sampling behavior everywhere.</summary>
[StructLayout(LayoutKind.Sequential)]
internal readonly struct PvxSamplingParamsWave5
{
    public readonly long Seed;
    public readonly float FrequencyPenalty;
    public readonly float TypicalP;
    public readonly float XtcP;
    public readonly float XtcT;
    public readonly float TopNSigma;
    public readonly int MirostatMode;
    public readonly float MirostatTau;
    public readonly float MirostatEta;
    public readonly float DryMultiplier;
    public readonly float DryBase;
    public readonly int DryAllowedLength;
    public readonly int DryPenaltyLastN;
    public readonly IntPtr LogitBias;
    public readonly int NLogitBias;

    public PvxSamplingParamsWave5(
        long seed = 0, float frequencyPenalty = 0f, float typicalP = 0f,
        float xtcP = 0f, float xtcT = 0f, float topNSigma = 0f,
        int mirostatMode = 0, float mirostatTau = 0f, float mirostatEta = 0f,
        float dryMultiplier = 0f, float dryBase = 0f,
        int dryAllowedLength = 0, int dryPenaltyLastN = 0,
        IntPtr logitBias = default, int nLogitBias = 0)
    {
        Seed = seed;
        FrequencyPenalty = frequencyPenalty;
        TypicalP = typicalP;
        XtcP = xtcP;
        XtcT = xtcT;
        TopNSigma = topNSigma;
        MirostatMode = mirostatMode;
        MirostatTau = mirostatTau;
        MirostatEta = mirostatEta;
        DryMultiplier = dryMultiplier;
        DryBase = dryBase;
        DryAllowedLength = dryAllowedLength;
        DryPenaltyLastN = dryPenaltyLastN;
        LogitBias = logitBias;
        NLogitBias = nLogitBias;
    }
}

/// <summary>ABI v8 wave-5 (BL-21a): blittable pvx_logit_bias_t mirror
/// (llama_logit_bias parity — token → added logit bias).</summary>
[StructLayout(LayoutKind.Sequential)]
internal readonly struct PvxLogitBias
{
    public readonly int Token;
    public readonly float Bias;

    public PvxLogitBias(int token, float bias)
    {
        Token = token;
        Bias = bias;
    }
}
