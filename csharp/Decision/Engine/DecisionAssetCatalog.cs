using System.Text.Json;

namespace ProvostXInference.Decision.Engine;

/// <summary>
/// Resolves the judge ASSET files (tokenizer.json, judge-spec-*.json) for one
/// configured judge model. Resolution order (first hit wins):
///   1. model config "file" SIBLING directory (a bundled model dir ships both),
///   2. the server root's assets/models/judge/ directory,
///   3. next to the server binary (Assets/models/judge copied at build/pack time).
/// Spec file per record kind: judge-spec-shell.json / judge-spec-process.json.
/// Pure resolver — performs no IO beyond existence checks at Resolve time.
/// </summary>
public sealed class DecisionAssetCatalog
{
    private const string TokenizerFileName = "tokenizer.json";
    private const string AssetsDirRelative = "assets/models/judge";

    private readonly string _rootDir;

    public DecisionAssetCatalog(string rootDir)
    {
        _rootDir = string.IsNullOrWhiteSpace(rootDir)
            ? throw new ArgumentNullException(nameof(rootDir))
            : Path.GetFullPath(rootDir);
    }

    /// <summary>Resolve the tokenizer file for a model whose .onnx lives at <paramref name="modelPath"/>.</summary>
    public string ResolveTokenizer(string modelPath)
        => ResolveAsset(modelPath, TokenizerFileName)
           ?? throw new JudgeModelNotReadyException("model_invalid",
               $"judge tokenizer '{TokenizerFileName}' not found for model '{Path.GetFileName(modelPath)}'");

    /// <summary>Resolve the question-spec file for a judge record kind.</summary>
    public string ResolveSpec(string modelPath, JudgeRecordKind kind)
    {
        var fileName = kind == JudgeRecordKind.Process ? "judge-spec-process.json" : "judge-spec-shell.json";
        return ResolveAsset(modelPath, fileName)
            ?? throw new JudgeModelNotReadyException("model_invalid",
                $"judge question spec '{fileName}' not found for model '{Path.GetFileName(modelPath)}'");
    }

    /// <summary>First existing candidate for an asset file. Pure function.</summary>
    private string? ResolveAsset(string modelPath, string fileName)
    {
        var modelDir = Path.GetDirectoryName(Path.GetFullPath(modelPath));
        if (!string.IsNullOrEmpty(modelDir))
        {
            var sibling = Path.Combine(modelDir, fileName);
            if (File.Exists(sibling)) return sibling;
        }

        var inRoot = Path.Combine(_rootDir, "assets", "models", "judge", fileName);
        if (File.Exists(inRoot)) return inRoot;

        var besideBinary = Path.Combine(AppContext.BaseDirectory, "assets", "models", "judge", fileName);
        if (File.Exists(besideBinary)) return besideBinary;

        var binaryRoot = Path.Combine(AppContext.BaseDirectory, fileName);
        return File.Exists(binaryRoot) ? binaryRoot : null;
    }

    /// <summary>Load + validate a judge spec JSON into a <see cref="JudgeSequenceSpec"/>. Pure per call.</summary>
    public static JudgeSequenceSpec LoadSpec(string path)
    {
        using var stream = File.OpenRead(path);
        using var doc = JsonDocument.Parse(stream);
        var root = doc.RootElement;

        var kindText = root.TryGetProperty("kind", out var kindEl) ? kindEl.GetString() : null;
        var kind = string.Equals(kindText, "process", StringComparison.OrdinalIgnoreCase)
            ? JudgeRecordKind.Process
            : JudgeRecordKind.Shell;

        var maxLen = root.TryGetProperty("max_len", out var maxLenEl) ? maxLenEl.GetInt32() : 512;
        var headMaxLen = root.TryGetProperty("head_max_len", out var headEl) ? headEl.GetInt32() : 192;
        if (maxLen < 64 || headMaxLen < 32 || headMaxLen >= maxLen)
            throw new JudgeModelNotReadyException("model_invalid",
                $"judge spec '{Path.GetFileName(path)}' has an invalid sequence budget (max_len={maxLen}, head_max_len={headMaxLen})");

        var temperatures = new List<double>();
        if (root.TryGetProperty("temperature", out var tempEl) && tempEl.ValueKind == JsonValueKind.Array)
        {
            foreach (var t in tempEl.EnumerateArray())
            {
                // Clamped to the fitted range the training runtime accepts (out-of-range
                // temperatures distort confidence; clamp + flag is the training contract).
                temperatures.Add(Math.Clamp(t.GetDouble(), 0.5, 5.0));
            }
        }
        while (temperatures.Count < 3) temperatures.Add(1.0);

        if (!root.TryGetProperty("questions", out var questionsEl) || questionsEl.ValueKind != JsonValueKind.Array)
            throw new JudgeModelNotReadyException("model_invalid",
                $"judge spec '{Path.GetFileName(path)}' is missing the questions array");
        if (questionsEl.GetArrayLength() == 0)
            throw new JudgeModelNotReadyException("model_invalid",
                $"judge spec '{Path.GetFileName(path)}' declares no questions");

        var questions = new List<JudgeQuestionDef>();
        foreach (var q in questionsEl.EnumerateArray())
        {
            var id = q.GetProperty("id").GetString() ?? throw new JudgeModelNotReadyException("model_invalid",
                $"judge spec '{Path.GetFileName(path)}' has a question without an id");
            var typeText = q.GetProperty("type").GetString();
            var type = typeText switch
            {
                "choice" => JudgeQuestionType.Choice,
                "score" => JudgeQuestionType.Score,
                "noul" => JudgeQuestionType.Noul,
                _ => throw new JudgeModelNotReadyException("model_invalid",
                    $"judge spec '{Path.GetFileName(path)}': unknown question type '{typeText}'"),
            };
            var instructions = q.GetProperty("instructions").GetString() ?? string.Empty;
            var options = new List<JudgeOption>();
            foreach (var opt in q.GetProperty("options").EnumerateArray())
            {
                var label = opt.GetProperty("label").GetString() ?? string.Empty;
                var description = opt.TryGetProperty("description", out var descEl) && descEl.ValueKind == JsonValueKind.String
                    ? descEl.GetString()
                    : null;
                options.Add(new JudgeOption(label, description));
            }
            if (options.Count == 0)
                throw new JudgeModelNotReadyException("model_invalid",
                    $"judge spec '{Path.GetFileName(path)}': question '{id}' has no options");
            questions.Add(new JudgeQuestionDef(id, type, instructions, options));
        }

        return new JudgeSequenceSpec(kind, maxLen, headMaxLen, temperatures, questions);
    }
}
