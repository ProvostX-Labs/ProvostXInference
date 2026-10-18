using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace ProvostXInference.Decision.Engine;

/// <summary>
/// ONE loaded ONNX judge model ready for batch inference: the ORT session
/// (selected execution provider recorded), the tokenizer, the sequence spec,
/// and the row collation + decode that mirrors the training evaluation exactly.
/// Disposal releases the session. NOT thread-safe per instance beyond the
/// session's own safety — the executor serializes inference per engine.
///
/// Decode (training-eval semantics, per question row k = option count):
///   choice  → argmax over the k class logits → JudgeAction (allow/sandbox/deny);
///   noul    → softmax over 2, P(true) = NeedsSandbox;
///   score   → softmax over 5, expected value = Risk (0..4);
///   confidence → max class probability after temperature scaling (0..1).
/// The Jev raw wire (JudgeQuestionSet) decodes ARBITRARY caller questions instead:
/// choice → argmax option label + per-option probabilities, score → expected level
/// value + level-index probabilities, noul → P(true) — same softmax/temperature
/// machinery, no escalation.
/// </summary>
public sealed class DecisionSession : IDisposable
{
    /// <summary>Argument of the softmax over choice logits: 0=allow, 1=sandbox, 2=deny.</summary>
    private static readonly JudgeAction[] ChoiceActions =
        { JudgeAction.Allow, JudgeAction.Sandbox, JudgeAction.Deny };

    private readonly JudgeTokenizer _tokenizer;
    private readonly JudgeSequenceSpec _spec;
    private readonly JudgeSequenceEncoder _encoder;
    private readonly InferenceSession _session;
    private readonly object _inferGate = new();
    private bool _disposed;

    /// <summary>Execution provider the session actually initialized with
    /// ("cpu" or the ORT provider name; a GPU EP failure falls back to CPU).</summary>
    public string ActiveProvider { get; }

    /// <summary>True when the session runs on the CPU execution provider.</summary>
    public bool IsCpu => string.Equals(ActiveProvider, "cpu", StringComparison.OrdinalIgnoreCase);

    public JudgeSequenceSpec Spec => _spec;
    public JudgeTokenizer Tokenizer => _tokenizer;

    public DecisionSession(JudgeTokenizer tokenizer, JudgeSequenceSpec spec, InferenceSession session, string activeProvider)
    {
        _tokenizer = tokenizer ?? throw new ArgumentNullException(nameof(tokenizer));
        _spec = spec ?? throw new ArgumentNullException(nameof(spec));
        _encoder = new JudgeSequenceEncoder(tokenizer, spec);
        _session = session ?? throw new ArgumentNullException(nameof(session));
        ActiveProvider = activeProvider;
        ValidateSessionSignature(session);
    }

    /// <summary>
    /// Fail-fast IO contract check at session init: the judge graph must expose
    /// EXACTLY the five trained inputs and the logits output — anything else is a
    /// foreign/corrupt model (mapped to model_invalid, never silently mis-fed).
    /// Pure per call.
    /// </summary>
    private static void ValidateSessionSignature(InferenceSession session)
    {
        var requiredInputs = new HashSet<string>(StringComparer.Ordinal)
        { "input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype" };
        var inputNames = session.InputMetadata?.Keys.ToHashSet(StringComparer.Ordinal)
            ?? new HashSet<string>(StringComparer.Ordinal);
        if (!requiredInputs.SetEquals(inputNames))
        {
            var expected = string.Join(", ", requiredInputs.OrderBy(n => n));
            var actual = string.Join(", ", inputNames.OrderBy(n => n));
            throw new JudgeModelNotReadyException("model_invalid",
                $"judge model input signature mismatch (expected: {expected}; got: {actual})");
        }
        if (session.OutputMetadata == null ||
            !session.OutputMetadata.Keys.Contains("logits", StringComparer.Ordinal))
            throw new JudgeModelNotReadyException("model_invalid",
                "judge model output signature mismatch (expected at least: logits)");
    }

    /// <summary>
    /// Score a batch of records (predict_batch semantics): all rows are collated
    /// into ONE padded session run, logits are sliced back per row in order, and
    /// one <see cref="JudgeScore"/> per record is returned (records with several
    /// questions fold action/needs_sandbox/risk from their rows).
    /// </summary>
    public IReadOnlyList<JudgeScore> Judge(IReadOnlyList<DecisionRecord> records)
    {
        ArgumentNullException.ThrowIfNull(records);
        if (records.Count == 0) return Array.Empty<JudgeScore>();
        ThrowIfDisposed();

        var recordRows = new IReadOnlyList<JudgeRowEncoding>[records.Count];
        var allRows = new List<JudgeRowEncoding>(records.Count * _spec.Questions.Count);
        for (var i = 0; i < records.Count; i++)
        {
            recordRows[i] = _encoder.Encode(records[i]);
            allRows.AddRange(recordRows[i]);
        }

        var (logits, kMax) = RunCollated(allRows);
        var scores = new JudgeScore[records.Count];
        var rowOffset = 0;
        for (var i = 0; i < records.Count; i++)
        {
            scores[i] = DecodeRecord(recordRows[i], logits, rowOffset, kMax);
            rowOffset += recordRows[i].Count;
        }
        return scores;
    }

    /// <summary>
    /// Score a batch of Jev question sets (raw wire): every state/question row is
    /// collated into ONE padded session run, then each row decodes INDEPENDENTLY
    /// into its question's raw answer — choice → argmax option label + per-option
    /// probabilities, score → expected level value + level-index probabilities +
    /// legend, noul → P(true). RAW semantics: NO risk-escalation rule (the
    /// escalation owner is JudgeVerdictMapper on the enriched arms).
    /// </summary>
    public IReadOnlyList<JudgeQuestionAnswerSet> JudgeQuestionSet(IReadOnlyList<JudgeStateQuestions> requests)
    {
        ArgumentNullException.ThrowIfNull(requests);
        if (requests.Count == 0) return Array.Empty<JudgeQuestionAnswerSet>();
        ThrowIfDisposed();

        var allRows = new List<JudgeRowEncoding>(requests.Count * 3);
        var questionSets = new IReadOnlyList<JudgeQuestionDef>[requests.Count];
        var rowSets = new IReadOnlyList<JudgeRowEncoding>[requests.Count];
        var stateTokens = new int[requests.Count];
        for (var i = 0; i < requests.Count; i++)
        {
            var request = requests[i];
            ArgumentNullException.ThrowIfNull(request);
            // Questions omitted on the wire → the judge's CANONICAL trained question
            // set (the spec this session was initialized from — our additive default).
            var questions = request.Questions ?? _spec.Questions;
            var rowSet = _encoder.EncodeQuestions(request.StateText, questions);
            rowSets[i] = rowSet.Rows;
            questionSets[i] = questions;
            stateTokens[i] = rowSet.StateTokens;
            allRows.AddRange(rowSet.Rows);
        }

        var (logits, kMax) = RunCollated(allRows);

        var results = new JudgeQuestionAnswerSet[requests.Count];
        var rowOffset = 0;
        for (var i = 0; i < requests.Count; i++)
        {
            var rows = rowSets[i];
            var answers = new JudgeQuestionAnswer[rows.Count];
            for (var r = 0; r < rows.Count; r++)
                answers[r] = DecodeQuestionRow(questionSets[i][r], rows[r], logits, (rowOffset + r) * kMax, kMax);
            results[i] = new JudgeQuestionAnswerSet(answers, stateTokens[i]);
            rowOffset += rows.Count;
        }
        return results;
    }

    /// <summary>Pad-collate rows, run ONE padded session pass, return the flat logits
    /// + the collated marker width k_max (row i's logits start at i·k_max).</summary>
    private (float[] Logits, int KMax) RunCollated(IReadOnlyList<JudgeRowEncoding> rows)
    {
        var (inputIds, attentionMask, markerPos, markerMask, qtypes) = Collate(rows);
        lock (_inferGate)
        {
            ThrowIfDisposed();
            var inputIdsT = TensorFrom(inputIds, rows.Count);
            var attentionT = TensorFrom(attentionMask, rows.Count);
            var markerPosT = TensorFrom(markerPos, rows.Count);
            var markerMaskT = TensorFrom(markerMask, rows.Count);
            var qtypesT = TensorFrom1D(qtypes);
            using var results = _session.Run(new[]
            {
                NamedOnnxValue.CreateFromTensor("input_ids", inputIdsT),
                NamedOnnxValue.CreateFromTensor("attention_mask", attentionT),
                NamedOnnxValue.CreateFromTensor("marker_pos", markerPosT),
                NamedOnnxValue.CreateFromTensor("marker_mask", markerMaskT),
                NamedOnnxValue.CreateFromTensor("qtype", qtypesT),
            });
            var logits = ToFloatArray(results);
            return (logits, markerPos.Length / Math.Max(1, rows.Count));
        }
    }

    /// <summary>Pad-collate rows into batch tensors (training collate semantics:
    /// pad id fills the tail, attention 1 only over real tokens, markers zero-padded). Pure.</summary>
    private static (long[] Ids, long[] Att, long[] Pos, bool[] MMask, long[] Qt) Collate(
        IReadOnlyList<JudgeRowEncoding> rows)
    {
        var seqMax = 0;
        var kMax = 0;
        foreach (var row in rows)
        {
            if (row.InputIds.Length > seqMax) seqMax = row.InputIds.Length;
            if (row.MarkerPositions.Length > kMax) kMax = row.MarkerPositions.Length;
        }
        var ids = new long[rows.Count * seqMax];
        var att = new long[rows.Count * seqMax];
        var pos = new long[rows.Count * kMax];
        var mmask = new bool[rows.Count * kMax];
        var qt = new long[rows.Count];
        for (var i = 0; i < rows.Count; i++)
        {
            var row = rows[i];
            qt[i] = (long)row.QuestionType;
            for (var t = 0; t < row.InputIds.Length; t++)
            {
                ids[i * seqMax + t] = row.InputIds[t];
                att[i * seqMax + t] = 1;
            }
            for (var m = 0; m < row.MarkerPositions.Length; m++)
                pos[i * kMax + m] = row.MarkerPositions[m];
            for (var m = 0; m < row.MarkerPositions.Length; m++)
                mmask[i * kMax + m] = true;
        }
        return (ids, att, pos, mmask, qt);
    }

    /// <summary>Fold one record's question-row logits into a JudgeScore. Pure (per call).</summary>
    private JudgeScore DecodeRecord(IReadOnlyList<JudgeRowEncoding> rows, float[] logits, int rowOffset, int kMax)
    {
        JudgeAction action = JudgeAction.Allow;
        double confidence = 0, risk = 0, needsSandbox = 0;
        for (var r = 0; r < rows.Count; r++)
        {
            var row = rows[r];
            var k = row.MarkerPositions.Length;
            var baseIndex = (rowOffset + r) * kMax;
            if (k <= 0) continue;

            var temperature = TemperatureFor(row.QuestionType, k);
            switch (row.QuestionType)
            {
                case JudgeQuestionType.Choice:
                {
                    var best = 0;
                    var bestLogit = float.NegativeInfinity;
                    for (var c = 0; c < k && c < ChoiceActions.Length; c++)
                    {
                        if (logits[baseIndex + c] > bestLogit)
                        {
                            bestLogit = logits[baseIndex + c];
                            best = c;
                        }
                    }
                    action = ChoiceActions[best];
                    confidence = SoftmaxProb(logits, baseIndex, k, temperature, best);
                    break;
                }
                case JudgeQuestionType.Noul:
                {
                    // Trained noul order: [false, true]. P(true) = sandbox recommendation.
                    needsSandbox = SoftmaxProb(logits, baseIndex, k, temperature, 1);
                    break;
                }
                case JudgeQuestionType.Score:
                {
                    var expected = 0.0;
                    for (var c = 0; c < k; c++)
                        expected += SoftmaxProb(logits, baseIndex, k, temperature, c) * c;
                    risk = expected;
                    break;
                }
            }
        }
        return new JudgeScore(action, Math.Round(confidence, 4), Math.Round(risk, 4),
            Math.Round(needsSandbox, 4), string.Empty); // ModelVersion stamped by the executor
    }

    /// <summary>Max-softmax probability for class <paramref name="classIndex"/> after temperature scaling. Pure.</summary>
    private static double SoftmaxProb(float[] logits, int baseIndex, int k, double temperature, int classIndex)
    {
        var max = float.NegativeInfinity;
        for (var c = 0; c < k; c++)
            if (logits[baseIndex + c] > max) max = logits[baseIndex + c];
        double sum = 0, target = 0;
        for (var c = 0; c < k; c++)
        {
            var e = Math.Exp((logits[baseIndex + c] - max) / temperature);
            sum += e;
            if (c == classIndex) target = e;
        }
        return sum > 0 ? target / sum : 0;
    }

    /// <summary>Temperature for a question type (bucket "3-5" covers 3..10 in the
    /// fitted config; option counts here are 2, 3, 5 — bucket "3-5"). Pure.</summary>
    private double TemperatureFor(JudgeQuestionType type, int optionCount)
    {
        var temperatures = _spec.Temperature;
        var index = (int)type;
        if (temperatures.Count == 0) return 1.0;
        return index < temperatures.Count ? temperatures[index] : temperatures[^1];
    }

    /// <summary>Decode ONE Jev question row into its raw answer (general decode —
    /// any option count, labels from the question def). Pure per call.</summary>
    private JudgeQuestionAnswer DecodeQuestionRow(
        JudgeQuestionDef question, JudgeRowEncoding row, float[] logits, int baseIndex, int kMax)
    {
        var k = row.MarkerPositions.Length;
        if (k <= 0)
            throw new JudgeModelNotReadyException("model_invalid",
                $"question '{question.Id}' encoded with no option markers");
        var temperature = TemperatureFor(question.Type, k);
        switch (question.Type)
        {
            case JudgeQuestionType.Choice:
            {
                var best = 0;
                var bestLogit = float.NegativeInfinity;
                for (var c = 0; c < k; c++)
                {
                    if (logits[baseIndex + c] > bestLogit)
                    {
                        bestLogit = logits[baseIndex + c];
                        best = c;
                    }
                }
                var probabilities = new Dictionary<string, double>(k);
                for (var c = 0; c < k; c++)
                    probabilities[OptionLabel(question, c)] = Math.Round(SoftmaxProb(logits, baseIndex, k, temperature, c), 4);
                return new JudgeQuestionAnswer(question.Id, JudgeQuestionType.Choice,
                    Choice: OptionLabel(question, best),
                    Confidence: Math.Round(SoftmaxProb(logits, baseIndex, k, temperature, best), 4),
                    Probabilities: probabilities);
            }
            case JudgeQuestionType.Score:
            {
                var probabilities = new Dictionary<string, double>(k);
                var legend = new Dictionary<string, string>(k);
                double expected = 0, confidence = 0;
                for (var c = 0; c < k; c++)
                {
                    var p = Math.Round(SoftmaxProb(logits, baseIndex, k, temperature, c), 4);
                    probabilities[c.ToString()] = p;
                    legend[c.ToString()] = OptionDescription(question, c);
                    expected += p * c;
                    if (p > confidence) confidence = p;
                }
                return new JudgeQuestionAnswer(question.Id, JudgeQuestionType.Score,
                    Score: Math.Round(expected, 4),
                    Confidence: confidence,
                    Probabilities: probabilities,
                    Legend: legend);
            }
            case JudgeQuestionType.Noul:
            {
                // Trained noul order: [false, true]. P(true) IS the noul answer.
                return new JudgeQuestionAnswer(question.Id, JudgeQuestionType.Noul,
                    Noul: Math.Round(SoftmaxProb(logits, baseIndex, k, temperature, 1), 4));
            }
            default:
                throw new JudgeModelNotReadyException("model_invalid",
                    $"question '{question.Id}' has an unsupported type {question.Type}");
        }
    }

    /// <summary>Option label at an argmax index (falls back to the index string when
    /// the row carries more classes than declared options). Pure.</summary>
    private static string OptionLabel(JudgeQuestionDef question, int index)
        => index < question.Options.Count && !string.IsNullOrEmpty(question.Options[index].Label)
            ? question.Options[index].Label
            : index.ToString();

    /// <summary>Level description for a score legend entry (training render rule:
    /// description-or-label). Pure.</summary>
    private static string OptionDescription(JudgeQuestionDef question, int index)
    {
        var option = index < question.Options.Count ? question.Options[index] : null;
        return option?.Description ?? option?.Label ?? index.ToString();
    }

    /// <summary>Wrap a flat buffer as a rank-1 DenseTensor. Pure static.</summary>
    private static DenseTensor<T> TensorFrom1D<T>(T[] buffer)
    {
        var tensor = new DenseTensor<T>(new[] { buffer.Length });
        buffer.AsSpan().CopyTo(tensor.Buffer.Span);
        return tensor;
    }

    /// <summary>Wrap a flat buffer as a DenseTensor with shape [rows, cols]. Pure static.</summary>
    private static DenseTensor<T> TensorFrom<T>(T[] buffer, int rows)
    {
        var tensor = new DenseTensor<T>(new[] { rows, buffer.Length / Math.Max(1, rows) });
        buffer.AsSpan().CopyTo(tensor.Buffer.Span);
        return tensor;
    }

    private static float[] ToFloatArray(IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results)
    {
        foreach (var value in results)
        {
            if (value.Value is DenseTensor<float> tensor)
            {
                var array = new float[tensor.Length];
                tensor.Buffer.CopyTo(array);
                return array;
            }
        }
        throw new JudgeModelNotReadyException("model_invalid",
            "judge model output 'logits' is missing or not a float tensor");
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(nameof(DecisionSession));
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _session.Dispose();
    }
}
