
using Microsoft.ML.OnnxRuntime;

namespace ProvostXInference.Decision.Engine;

/// <summary>One candidate configuration for an EP session init: a label for
/// diagnostics and the options mutation it applies. Ordered — first that
/// successfully creates a session wins.</summary>
public sealed record EpAttempt(string Label, Action<SessionOptions> Apply);

/// <summary>
/// Per-EP session-option presets (LLM-10): each configured chain entry may need
/// provider-specific session options (CUDA device id, DirectML device id, ROCm
/// device id, OpenVINO device type, CoreML flags). Every preset list ends with a
/// PLAIN name-append fallback so a quirk mismatch never fails an otherwise
/// loadable provider — remaining failures fall down the chain to cpu (never fail
/// when cpu works). Instance type, no state — cheap to construct, injectable.
/// </summary>
public sealed class ExecutionProviderOptionsApplier
{
    /// <summary>Attempts for a chain entry, in order. CPU has exactly one (plain).</summary>
    public IReadOnlyList<EpAttempt> Presets(OnnxExecutionProvider provider) => provider switch
    {
        OnnxExecutionProvider.Cpu => new[]
        {
            new EpAttempt("cpu-plain", _ => { /* CPU is the implicit default EP */ }),
        },
        OnnxExecutionProvider.CoreMl => new[]
        {
            // CoreML quirk: explicit compute-unit scope (CPU+GPU compute units,
            // ML-program conversion on) via the typed flags API; native rejects
            // incompatible flag sets at session creation — the plain append below
            // is the safety net.
            new EpAttempt("coreml-flags", o => o.AppendExecutionProvider_CoreML(
                CoreMLFlags.COREML_FLAG_USE_CPU_AND_GPU | CoreMLFlags.COREML_FLAG_CREATE_MLPROGRAM)),
            new EpAttempt("coreml-plain", o => o.AppendExecutionProvider("CoreMLExecutionProvider")),
        },
        OnnxExecutionProvider.DirectMl => new[]
        {
            // DirectML quirk: explicit device id 0.
            new EpAttempt("directml-device0", o => o.AppendExecutionProvider_DML(0)),
            new EpAttempt("directml-plain", o => o.AppendExecutionProvider("DmlExecutionProvider")),
        },
        OnnxExecutionProvider.Cuda => new[]
        {
            // CUDA quirk: explicit device id 0 (HEURISTIC cudnn algo is the default
            // the provider applies when no explicit options struct is passed).
            new EpAttempt("cuda-device0", o => o.AppendExecutionProvider_CUDA(0)),
            new EpAttempt("cuda-plain", o => o.AppendExecutionProvider("CUDAExecutionProvider")),
        },
        OnnxExecutionProvider.Rocm => new[]
        {
            // ROCm quirk: explicit device id 0.
            new EpAttempt("rocm-device0", o => o.AppendExecutionProvider_ROCm(0)),
            new EpAttempt("rocm-plain", o => o.AppendExecutionProvider("ROCExecutionProvider")),
        },
        OnnxExecutionProvider.OpenVino => new[]
        {
            // OpenVINO quirk: empty device type → auto-select the best device.
            new EpAttempt("openvino-auto", o => o.AppendExecutionProvider_OpenVINO(string.Empty)),
            new EpAttempt("openvino-plain", o => o.AppendExecutionProvider("OpenVINOExecutionProvider")),
        },
        _ => new[]
        {
            new EpAttempt("cpu-plain", _ => { }),
        },
    };
}
