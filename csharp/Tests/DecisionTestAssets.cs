using ProvostXInference.Decision.Engine;

namespace ProvostXInference.Tests.Decision;

/// <summary>
/// Shared decision-engine test assets: the tiny synthetic ONNX model
/// (Tests/Assets/onnx), the tokenizer/question-spec assets (the Inference repo's
/// assets/models/judge copied next to the test assembly), and per-test accessors.
/// Mirrors the LLM-side JudgeTestAssets fixture (which delegates ORT work here).
/// </summary>
public sealed class DecisionTestAssets
{
    public string TinyModelPath { get; } = Path.Combine(AppContext.BaseDirectory, "Assets", "onnx", "tiny-judge.onnx");
    public string TokenizerPath { get; } = Path.Combine(AppContext.BaseDirectory, "assets", "models", "judge", "tokenizer.json");
    public string ShellSpecPath { get; } = Path.Combine(AppContext.BaseDirectory, "assets", "models", "judge", "judge-spec-shell.json");
    public string ProcessSpecPath { get; } = Path.Combine(AppContext.BaseDirectory, "assets", "models", "judge", "judge-spec-process.json");

    /// <summary>Copy the tiny synthetic model to a path (fresh bytes per use).</summary>
    public void WriteTinyModel(string destination) => File.Copy(TinyModelPath, destination, overwrite: true);

    /// <summary>A tokenizer loaded from the shared asset (fresh instance per use).</summary>
    public JudgeTokenizer NewTokenizer() => new(TokenizerPath);

    /// <summary>The shell question spec (fresh instance per use).</summary>
    public JudgeSequenceSpec NewShellSpec() => DecisionAssetCatalog.LoadSpec(ShellSpecPath);

    /// <summary>The process question spec (fresh instance per use).</summary>
    public JudgeSequenceSpec NewProcessSpec() => DecisionAssetCatalog.LoadSpec(ProcessSpecPath);
}
