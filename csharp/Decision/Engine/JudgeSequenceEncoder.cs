using System.Text;
using System.Text.Json;

namespace ProvostXInference.Decision.Engine;

/// <summary>
/// Builds judge model input rows from validated <see cref="DecisionRecord"/>s:
/// serializes the record EXACTLY as the training pipeline did (compact JSON with
/// the mask token stripped), then assembles
/// [CLS] {type} question: {instructions} [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] {state} [SEP]
/// per question, with the state tail capped to the sequence budget. Mirrors the
/// training sequence builder (head budget 256, options capped at 48 tokens each)
/// against the committed fixtures (Tests/Assets/judge/seq-fixtures.json).
/// One encoder per judge model spec; thread-safe (stateless + immutable fields).
/// </summary>
public sealed class JudgeSequenceEncoder
{
    private const int OptionTokenCap = 48;
    private const int HeadFloorTokens = 8;

    /// <summary>
    /// Fixed neutral purpose fed to the shell judge input. The record's untrusted
    /// purpose is deliberately NOT decision input (audit echo only — see
    /// DecisionRecord); the model was trained purpose-invariant, so the verdict
    /// must not depend on this value.
    /// </summary>
    private const string FixedShellPurpose = "inspect";

    private readonly JudgeTokenizer _tokenizer;
    private readonly JudgeSequenceSpec _spec;

    public JudgeSequenceSpec Spec => _spec;

    public JudgeSequenceEncoder(JudgeTokenizer tokenizer, JudgeSequenceSpec spec)
    {
        _tokenizer = tokenizer ?? throw new ArgumentNullException(nameof(tokenizer));
        _spec = spec ?? throw new ArgumentNullException(nameof(spec));
    }

    /// <summary>Encode one validated record into one row per question, in spec question order.</summary>
    public IReadOnlyList<JudgeRowEncoding> Encode(DecisionRecord record)
    {
        ArgumentNullException.ThrowIfNull(record);
        var state = SerializeRecord(record);
        return EncodeQuestions(state, _spec.Questions).Rows;
    }

    /// <summary>
    /// Encode ONE state string against ARBITRARY typed questions (the Jev wire —
    /// caller-supplied questions or the judge's canonical default set). The state
    /// text is used AS GIVEN (the server layer owns the Jev state serialization
    /// rule), mask tokens stripped exactly as the training pipeline, then one row
    /// per question. Returns the rows + the PRE-truncation state token count
    /// (usage.input_tokens). Thread-safe (stateless).</summary>
    public JudgeQuestionRowSet EncodeQuestions(string stateText, IReadOnlyList<JudgeQuestionDef> questions)
    {
        ArgumentException.ThrowIfNullOrEmpty(stateText);
        ArgumentNullException.ThrowIfNull(questions);
        var stateIds = _tokenizer.Encode(stateText.Replace(_tokenizer.MaskToken, " "));
        var stateTokens = stateIds.Length;
        var rows = new JudgeRowEncoding[questions.Count];
        for (var i = 0; i < questions.Count; i++)
            rows[i] = EncodeRow(questions[i], stateIds);
        return new JudgeQuestionRowSet(rows, stateTokens);
    }

    /// <summary>Option rendering, EXACTLY as the training renderer (laya common.py):
    ///   choice → "label: description" (bare label when no description),
    ///   score  → "level {i}: {description-or-label}",
    ///   noul   → "false|true: {description-or-default}" (defaults are trained strings).
    /// Pure static.</summary>
    private static string RenderOption(JudgeQuestionDef question, int index, JudgeOption option)
    {
        switch (question.Type)
        {
            case JudgeQuestionType.Score:
                return $"level {index}: {option.Description ?? option.Label}";
            case JudgeQuestionType.Noul:
            {
                var description = option.Description ?? (string.Equals(option.Label, "true", StringComparison.OrdinalIgnoreCase)
                    ? "yes, the statement holds"
                    : "no, the statement does not hold");
                return $"{option.Label}: {description}";
            }
            default:
                return option.Description is null ? option.Label : $"{option.Label}: {option.Description}";
        }
    }

    /// <summary>Serialize a validated record to the exact training input format.
    /// SHELL: {"command": ..., "cwd": ..., "purpose": ...} — purpose is part of the
    /// training input, NOT an untrusted decision signal (the model was trained
    /// purpose-invariant). PROCESS: {"command": "PROCESS os=|exe=|chain=|cmd=|net=|writes=...",
    /// "cwd": "", "purpose": "sentinel"} — the raw record rides in "command"
    /// (constant cwd/purpose in the process training set). Pure function.</summary>
    public static string SerializeRecord(DecisionRecord record)
    {
        if (record.Kind == JudgeRecordKind.Process)
        {
            // Training format: the raw PROCESS record rides in the "command" field of
            // the state JSON with a fixed cwd/purpose (both constant in the training set).
            var sb = new StringBuilder(192);
            sb.Append("{\"command\": ");
            AppendJsonString(sb, SerializeProcessCommand(record));
            sb.Append(", \"cwd\": \"\", \"purpose\": \"sentinel\"}");
            return sb.ToString();
        }

        var shell = new StringBuilder(128);
        shell.Append("{\"command\": ");
        AppendJsonString(shell, record.Command);
        shell.Append(", \"cwd\": ");
        AppendJsonString(shell, record.Cwd);
        shell.Append(", \"purpose\": ");
        AppendJsonString(shell, FixedShellPurpose);
        shell.Append('}');
        return shell.ToString();
    }

    /// <summary>Rebuild the canonical raw PROCESS record string (fixed segment order,
    /// empty fields omitted) — mirrors the training dataset exactly. Also the
    /// Jev state rule for process-shaped state objects (the server layer's state
    /// serializer calls this). Pure static.</summary>
    public static string SerializeProcessCommand(DecisionRecord record)
    {
        var text = $"PROCESS os={record.Os}";
        if (record.Exe.Length > 0) text += $" | exe={record.Exe}";
        if (record.Chain.Count > 0) text += $" | chain={string.Join("->", record.Chain)}";
        if (record.Cmd.Length > 0) text += $" | cmd={record.Cmd}";
        if (record.Net.Length > 0) text += $" | net={record.Net}";
        if (record.Writes.Length > 0) text += $" | writes={record.Writes}";
        return text;
    }

    /// <summary>Build one question row: [CLS] head [SEP] (MASK option)* [SEP] state-tail [SEP].</summary>
    private JudgeRowEncoding EncodeRow(JudgeQuestionDef question, int[] stateIds)
    {
        // Internal type name EXACTLY as trained (lowercase — QTYPES keys).
        var typeName = question.Type switch
        {
            JudgeQuestionType.Choice => "choice",
            JudgeQuestionType.Score => "score",
            JudgeQuestionType.Noul => "noul",
            _ => question.Type.ToString(),
        };
        var headIds = _tokenizer.Encode(
            $"{typeName} question: {question.Instructions}".Replace(_tokenizer.MaskToken, " "));
        var optionIds = new List<int[]>(question.Options.Count);
        for (var oi = 0; oi < question.Options.Count; oi++)
        {
            var option = question.Options[oi];
            var optionText = " " + RenderOption(question, oi, option);
            var tokens = _tokenizer.Encode(optionText.Replace(_tokenizer.MaskToken, " "));
            if (tokens.Length > OptionTokenCap)
                tokens = tokens[..OptionTokenCap];
            var withMask = new int[tokens.Length + 1];
            withMask[0] = _tokenizer.MaskTokenId;
            Array.Copy(tokens, 0, withMask, 1, tokens.Length);
            optionIds.Add(withMask);
        }

        // Head budget: whatever max_len leaves after the options, floored at 8.
        var optionTotal = optionIds.Sum(o => o.Length);
        var headBudget = _spec.HeadMaxLen - optionTotal;
        if (headBudget < 16)
        {
            var per = Math.Max(4, (_spec.HeadMaxLen - 16) / Math.Max(1, optionIds.Count));
            optionIds = optionIds.Select(o => o.Length > per ? o[..per] : o).ToList();
            optionTotal = optionIds.Sum(o => o.Length);
            headBudget = _spec.HeadMaxLen - optionTotal;
        }
        if (headIds.Length > Math.Max(HeadFloorTokens, headBudget))
            headIds = headIds[..Math.Max(HeadFloorTokens, headBudget)];

        var ids = new List<int>(2 + headIds.Length + optionTotal + 2 + stateIds.Length + 1)
        {
            _tokenizer.ClsTokenId,
        };
        ids.AddRange(headIds);
        ids.Add(_tokenizer.SepTokenId);
        var markers = new List<int>(optionIds.Count);
        foreach (var option in optionIds)
        {
            markers.Add(ids.Count);
            ids.AddRange(option);
        }
        ids.Add(_tokenizer.SepTokenId);

        var room = Math.Max(0, _spec.MaxLen - ids.Count - 1);
        if (stateIds.Length > room)
            stateIds = room == 0 ? Array.Empty<int>() : stateIds[^room..];
        if (room > 0)
        {
            ids.AddRange(stateIds);
            ids.Add(_tokenizer.SepTokenId);
        }

        if (ids.Count > _spec.MaxLen)
            ids = ids.GetRange(0, _spec.MaxLen);
        var markerArray = markers.Where(m => m < _spec.MaxLen).ToArray();
        return new JudgeRowEncoding(question.Type, ids.ToArray(), markerArray);
    }

    /// <summary>Compact JSON string literal (ensure_ascii=False semantics — raw chars pass through). Pure static.</summary>
    private static void AppendJsonString(StringBuilder sb, string value)
    {
        sb.Append('"');
        foreach (var ch in value)
        {
            switch (ch)
            {
                case '"': sb.Append("\\\""); break;
                case '\\': sb.Append("\\\\"); break;
                case '\n': sb.Append("\\n"); break;
                case '\r': sb.Append("\\r"); break;
                case '\t': sb.Append("\\t"); break;
                default:
                    if (ch < ' ') sb.Append("\\u").Append(((int)ch).ToString("x4"));
                    else sb.Append(ch);
                    break;
            }
        }
        sb.Append('"');
    }
}
