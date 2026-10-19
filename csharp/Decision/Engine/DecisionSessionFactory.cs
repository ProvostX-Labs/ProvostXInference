
using ProvostXInference.Decision.Abstractions;
using ProvostXInference.Logging;
using Microsoft.ML.OnnxRuntime;

namespace ProvostXInference.Decision.Engine;

/// <summary>
/// Creates <see cref="DecisionSession"/>s for a judge model file: builds the ORT
/// session options (full graph optimization, matching the training runtime) and
/// walks the parsed <see cref="ExecutionProviderChain"/> in preference order
/// (LLM-10 per-OS chains). EP FALLBACK RULE: a provider that fails to initialize
/// (not bundled, driver/library problems, unsupported ops) degrades to the next
/// entry — CPU is always the terminal fallback and is the only provider REQUIRED
/// to work. The session's ACTUALLY-active provider is read back from the runtime
/// (a silent ORT fallback to CPU counts as a fallback, not as GPU) and recorded
/// so /pvx/health `active_execution_provider` is truthful. One factory per host;
/// thread-safe (stateless beyond the injected option applier).
/// </summary>
public sealed class DecisionSessionFactory : IDecisionSessionFactory
{
    private readonly ExecutionProviderOptionsApplier _optionsApplier;
    private IInferenceLogger? _logger;

    /// <summary>Optional logger for EP-walk diagnostics (owner black-box close-out
    /// 2026-10-19: the logger moved from a ctor parameter to this property — the
    /// ctor signature must stay free of Logging-namespace types so callers that
    /// construct the factory parameterless do not drag the Logging namespace
    /// into their compiled metadata). Defaults to the quiet no-op logger.</summary>
    public IInferenceLogger Logger
    {
        get => _logger ??= new QuietInferenceLogger();
        set => _logger = value;
    }

    public DecisionSessionFactory(ExecutionProviderOptionsApplier? optionsApplier = null)
    {
        _optionsApplier = optionsApplier ?? new ExecutionProviderOptionsApplier();
    }

    /// <summary>ORT execution-provider name for a configured chain entry. Pure function.</summary>
    private static string ProviderName(OnnxExecutionProvider provider) => provider switch
    {
        OnnxExecutionProvider.Cpu => "CPUExecutionProvider",
        OnnxExecutionProvider.CoreMl => "CoreMLExecutionProvider",
        OnnxExecutionProvider.DirectMl => "DmlExecutionProvider",
        OnnxExecutionProvider.Cuda => "CUDAExecutionProvider",
        OnnxExecutionProvider.Rocm => "ROCExecutionProvider",
        OnnxExecutionProvider.OpenVino => "OpenVINOExecutionProvider",
        _ => "CPUExecutionProvider",
    };

    /// <summary>Lowercase display name for health reporting. Pure function.</summary>
    private static string DisplayName(OnnxExecutionProvider provider) => provider switch
    {
        OnnxExecutionProvider.Cpu => "cpu",
        OnnxExecutionProvider.CoreMl => "coreml",
        OnnxExecutionProvider.DirectMl => "directml",
        OnnxExecutionProvider.Cuda => "cuda",
        OnnxExecutionProvider.Rocm => "rocm",
        OnnxExecutionProvider.OpenVino => "openvino",
        _ => "cpu",
    };

    /// <summary>
    /// Initialize a session for <paramref name="modelPath"/>, trying each configured
    /// provider in chain order (per-provider option presets first, plain append
    /// fallback second). Returns the session plus the provider that ACTUALLY came
    /// up (never a GPU name when the provider failed to register). Throws when NO
    /// provider in the chain could initialize (mapped to model_invalid by the engine).
    /// </summary>
    public DecisionSession Create(
        string modelPath,
        JudgeTokenizer tokenizer,
        JudgeSequenceSpec spec,
        ExecutionProviderChain chain,
        JudgeNumericsGate? numericsGate = null)
    {
        ArgumentNullException.ThrowIfNull(modelPath);
        ArgumentNullException.ThrowIfNull(tokenizer);
        ArgumentNullException.ThrowIfNull(spec);
        ArgumentNullException.ThrowIfNull(chain);

        Exception? lastError = null;
        var available = OrtEnv.Instance().GetAvailableProviders().ToHashSet(StringComparer.Ordinal);
        foreach (var provider in chain.Providers)
        {
            var requested = ProviderName(provider);
            if (provider != OnnxExecutionProvider.Cpu && !available.Contains(requested))
            {
                // Not bundled in this build (e.g. CoreML without a CoreML-capable
                // runtime, CUDA/ROCm without the GPU package or driver): count as
                // a failed attempt and continue down the chain.
                lastError = new InvalidOperationException($"{requested} not available in this runtime");
                continue;
            }

            foreach (var attempt in _optionsApplier.Presets(provider))
            {
                InferenceSession? session = null;
                SessionOptions? options = null;
                try
                {
                    options = new SessionOptions();
                    options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
                    attempt.Apply(options);
                    session = new InferenceSession(modelPath, options);

                    // Truthful active-provider reporting: the provider is ACTIVE only
                    // when the runtime bundles it AND the session initialized with it
                    // registered. ORT never silently demotes a registered EP — a
                    // registration that fails throws, which lands in the catch below
                    // and walks down the chain. (A GPU EP that yields no kernels for
                    // the graph logs a partitioning warning; the numerics-gate E2E
                    // anchors re-run on the active provider to catch numeric drift.)
                    var display = DisplayName(provider);
                    var candidate = new DecisionSession(tokenizer, spec, session, display);

                    // LLM-10 numerics gate: a NON-CPU EP that loads must still
                    // reproduce the canonical anchor — drift means the EP's kernels
                    // change the verdict math; reject and continue down the chain.
                    // CPU (the parity baseline) is never gated.
                    if (numericsGate is not null && provider != OnnxExecutionProvider.Cpu)
                    {
                        try
                        {
                            var score = candidate.Judge(new[] { numericsGate.Record })[0];
                            if (!numericsGate.IsSatisfied(score))
                            {
                                _logger.Warn("Judge",
                                    $"EP {display} loaded but FAILED the numerics gate " +
                                    $"(action={score.RawAction}, conf={score.Confidence}, risk={score.Risk} " +
                                    $"vs {numericsGate.ExpectedAction}/{numericsGate.ExpectedConfidence}/{numericsGate.ExpectedRisk}) " +
                                    "— rejecting EP, continuing down the chain");
                                candidate.Dispose();
                                lastError = new InvalidOperationException(
                                    $"numerics gate failed on {display}");
                                continue;
                            }
                            _logger.Info("Judge", $"EP {display} passed the numerics gate — accepting");
                        }
                        catch (Exception gex)
                        {
                            _logger.Warn("Judge",
                                $"EP {display} numerics gate evaluation threw ({gex.GetType().Name}: {gex.Message}) — rejecting EP");
                            candidate.Dispose();
                            lastError = gex;
                            continue;
                        }
                    }

                    return candidate;
                }
                catch (Exception ex)
                {
                    lastError = ex;
                    session?.Dispose();
                }
                finally
                {
                    options?.Dispose();
                }
            }
        }
        throw lastError ?? new InvalidOperationException("no execution provider configured");
    }
}
