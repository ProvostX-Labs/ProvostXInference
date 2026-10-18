using ProvostXInference.Decision.Engine;

namespace ProvostXInference.Decision.Abstractions;

/// <summary>
/// Decision-model file validation seam (load state machine step). A REAL
/// protobuf scan of the ONNX ModelProto wire (ir_version, graph IO signature
/// names, opset_import presence) — long initializers are skipped by seek, so a
/// 500MB+ model validates in milliseconds. Any structural violation throws —
/// the load state machine maps that to fail-closed model_invalid before any
/// session init is attempted.
/// </summary>
public interface IDecisionHeaderReader
{
    /// <summary>
    /// Validate that the file at <paramref name="path"/> is a plausibly loadable
    /// decision model. Throws when the header is invalid (caller maps to fail-closed).
    /// </summary>
    void Validate(string path);
}

/// <summary>
/// Per-OS DEFAULT execution-provider chain for decision models: applied when a
/// model configures NO explicit execution_provider. Injected so tests can pin
/// the OS.
/// </summary>
public interface IExecutionProviderDefaultSource
{
    /// <summary>The default EP chain for the current OS (first available wins, cpu terminal).</summary>
    ExecutionProviderChain DefaultChain();
}
