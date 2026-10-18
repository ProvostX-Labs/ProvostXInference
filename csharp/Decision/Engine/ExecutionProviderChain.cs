namespace ProvostXInference.Decision.Engine;

/// <summary>
/// Execution providers for decision (judge) models. The chain is parsed and
/// validated at config time so unknown names fail loudly. Per-OS: CoreML
/// (macOS), DirectML (Windows), CUDA/ROCm/OpenVINO (Linux) — cpu is always the
/// terminal fallback.
/// </summary>
public enum OnnxExecutionProvider
{
    /// <summary>ONNX Runtime CPU EP (default).</summary>
    Cpu = 0,

    /// <summary>Apple Core ML EP (macOS).</summary>
    CoreMl = 1,

    /// <summary>Windows DirectML EP.</summary>
    DirectMl = 2,

    /// <summary>NVIDIA CUDA EP.</summary>
    Cuda = 3,

    /// <summary>AMD ROCm EP.</summary>
    Rocm = 4,

    /// <summary>Intel OpenVINO EP.</summary>
    OpenVino = 5,
}

/// <summary>
/// Parsed execution-provider chain for a decision (judge) model (e.g. "cpu",
/// "coreml-&gt;cpu", "cuda->cpu"). First entry is the preferred provider; later
/// entries are fallbacks. Unknown names fail config validation loudly (the
/// parser throws) — silent CPU fallback for typos is a misconfiguration hazard.
/// </summary>
public sealed class ExecutionProviderChain
{
    /// <summary>Known EP names (lowercase). Per-OS: CoreML/DirectML/CUDA/RoCM/OpenVINO.</summary>
    private static readonly Dictionary<string, OnnxExecutionProvider> Known = new(StringComparer.Ordinal)
    {
        ["cpu"] = OnnxExecutionProvider.Cpu,
        ["coreml"] = OnnxExecutionProvider.CoreMl,
        ["directml"] = OnnxExecutionProvider.DirectMl,
        ["cuda"] = OnnxExecutionProvider.Cuda,
        ["rocm"] = OnnxExecutionProvider.Rocm,
        ["openvino"] = OnnxExecutionProvider.OpenVino,
    };

    /// <summary>Ordered provider preference (first = preferred).</summary>
    public IReadOnlyList<OnnxExecutionProvider> Providers { get; }

    /// <summary>True when the chain configures NO GPU provider (cpu only). Such
    /// models register 0 into a VRAM budget; GPU EP values register their
    /// resident cost (model-file-size estimate).</summary>
    public bool IsCpuOnly => Providers.All(p => p == OnnxExecutionProvider.Cpu);

    /// <summary>True when the chain configures at least one GPU provider.</summary>
    public bool RegistersGpuVram => !IsCpuOnly;

    private ExecutionProviderChain(IReadOnlyList<OnnxExecutionProvider> providers)
    {
        Providers = providers;
    }

    /// <summary>
    /// Parse a per-OS chain string ("cpu", "coreml-&gt;cpu", "cuda-&gt;cpu").
    /// Null/empty → CPU default. Unknown names throw (loud config validation).
    /// Pure factory — no state, no side effects.
    /// </summary>
    public static ExecutionProviderChain Parse(string? raw, string modelId)
    {
        if (string.IsNullOrWhiteSpace(raw))
            return new ExecutionProviderChain(new[] { OnnxExecutionProvider.Cpu });

        var tokens = raw.Split("->", StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries);
        if (tokens.Length == 0)
            throw new InvalidOperationException(
                $"model '{modelId}': execution_provider '{raw}' is empty after parsing (expected e.g. 'cpu' or 'coreml->cpu')");

        var providers = new List<OnnxExecutionProvider>(tokens.Length);
        foreach (var token in tokens)
        {
            var key = token.ToLowerInvariant();
            if (!Known.TryGetValue(key, out var provider))
                throw new InvalidOperationException(
                    $"model '{modelId}': unknown execution provider '{token}' " +
                    "(known: cpu, coreml, directml, cuda, rocm, openvino)");
            providers.Add(provider);
        }

        return new ExecutionProviderChain(providers);
    }

    /// <summary>Canonical display string (lowercase, '-&gt;'-joined). Pure function.</summary>
    public string ToDisplayString() => string.Join("->", Providers.Select(p => p.ToString().ToLowerInvariant()));
}
