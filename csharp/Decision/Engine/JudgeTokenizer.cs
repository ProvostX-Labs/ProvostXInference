using System.Collections.Concurrent;
using System.Text;
using System.Text.Json;

namespace ProvostXInference.Decision.Engine;

/// <summary>
/// Deterministic BPE tokenizer for the Laya judge models (mmBERT vocabulary,
/// HuggingFace tokenizer.json format: BPE model + Replace(" " → "▁") normalizer
/// + Metaspace pre-tokenizer, add_special_tokens=False semantics — no BOS/EOS
/// post-processing; the judge sequence builder places CLS/SEP itself).
///
/// Pipeline (replicates the training tokenizer byte-for-byte):
///   1. added-token extraction (longest match, pre-normalization, per segment),
///   2. normalization: " " → "▁" per non-added segment,
///   3. Metaspace pre-tokenization: prepend "▁" when the segment does not start
///      with it, then split BEFORE every "▁" (each "▁" begins a piece),
///   4. greedy lowest-rank BPE merges per piece (pairs of in-vocab symbols only),
///   5. unknown symbols: byte fallback ("&lt;0xXX&gt;" tokens), then &lt;unk&gt;
///      (consecutive unknowns fused).
///
/// Thread-safe: all mutable state is ConcurrentDictionary caches; encoding is a
/// pure function of the input text. Deterministic: no RNG, no time, no IO after
/// construction. Verified against the training pipeline on the committed vector
/// fixture (Tests/Assets/judge/tok-vectors.json).
/// </summary>
public sealed class JudgeTokenizer
{
    private const string MetaSpace = "▁";
    private const string UnknownToken = "<unk>";
    private const int MergePairSeparator = '\u0000';

    private readonly Dictionary<string, int> _vocab;
    private readonly Dictionary<long, int> _mergeRanks;
    private readonly Dictionary<string, int> _addedTokens;
    private readonly Dictionary<char, List<string>> _addedByFirstChar;
    private readonly int _maxAddedTokenLength;
    private readonly Dictionary<byte, int> _byteTokens;
    private readonly ConcurrentDictionary<string, int[]> _textCache = new();
    private readonly ConcurrentDictionary<string, int[]> _pieceCache = new();

    /// <summary>Token id used as the CLS/sequence-start token ("&lt;bos&gt;").</summary>
    public int ClsTokenId { get; }
    /// <summary>Token id used as the SEP token ("&lt;eos&gt;").</summary>
    public int SepTokenId { get; }
    /// <summary>Token id of the MASK token ("&lt;mask&gt;") — option placeholder in judge rows.</summary>
    public int MaskTokenId { get; }
    /// <summary>Token id used for batch padding ("&lt;pad&gt;").</summary>
    public int PadTokenId { get; }
    /// <summary>The mask token string (stripped from decision input before tokenization).</summary>
    public string MaskToken { get; }

    public JudgeTokenizer(string tokenizerJsonPath)
    {
        using var stream = File.OpenRead(tokenizerJsonPath);
        using var doc = JsonDocument.Parse(stream);
        var root = doc.RootElement;

        (_vocab, _mergeRanks, _addedTokens, _byteTokens) = LoadModel(root);        MaskToken = "<mask>";

        // Added tokens are extracted BEFORE normalization (verbatim, never normalized).
        _addedTokens = new Dictionary<string, int>(StringComparer.Ordinal);
        _maxAddedTokenLength = 0;
        foreach (var added in root.GetProperty("added_tokens").EnumerateArray())
        {
            var content = added.GetProperty("content").GetString() ?? string.Empty;
            if (content.Length == 0 || _addedTokens.ContainsKey(content)) continue;
            _addedTokens[content] = added.GetProperty("id").GetInt32();
            if (content.Length > _maxAddedTokenLength) _maxAddedTokenLength = content.Length;
        }
        _addedByFirstChar = new Dictionary<char, List<string>>();
        foreach (var content in _addedTokens.Keys)
        {
            if (!_addedByFirstChar.TryGetValue(content[0], out var bucket))
            {
                bucket = new List<string>();
                _addedByFirstChar[content[0]] = bucket;
            }
            bucket.Add(content);
        }

        // Special token ids come straight from the vocab (config special tokens may
        // name them, but the judge models pin <bos>/<eos>/<mask>/<pad> by content).
        ClsTokenId = RequireId(_vocab, "<bos>");
        SepTokenId = RequireId(_vocab, "<eos>");
        MaskTokenId = RequireId(_vocab, "<mask>");
        PadTokenId = RequireId(_vocab, "<pad>");
    }

    /// <summary>
    /// Encode text exactly like the training tokenizer with add_special_tokens=False:
    /// no added BOS/EOS, added-token strings in the text map to their ids.
    /// Pure function — same input always yields the same ids.
    /// </summary>
    public int[] Encode(string text)
    {
        if (string.IsNullOrEmpty(text)) return Array.Empty<int>();
        if (_textCache.TryGetValue(text, out var cached)) return cached;

        var ids = new List<int>();
        var index = 0;
        while (index < text.Length)
        {
            var addedLength = MatchAddedToken(text, index);
            if (addedLength > 0)
            {
                ids.Add(_addedTokens[text.Substring(index, addedLength)]);
                index += addedLength;
                continue;
            }
            var segmentEnd = NextAddedToken(text, index);
            var segment = text.Substring(index, segmentEnd - index);
            EncodeSegment(segment, ids);
            index = segmentEnd;
        }

        var result = ids.ToArray();
        _textCache[text] = result;
        return result;
    }

    // ── pipeline steps ──────────────────────────────────────────────────

    /// <summary>Normalize + pre-tokenize + BPE one non-added segment, appending ids.</summary>
    private void EncodeSegment(string segment, List<int> ids)
    {
        // Normalizer: Replace(" ", "▁").
        var normalized = segment.Replace(' ', '▁');
        if (normalized.Length == 0) return;
        // Metaspace prepend_scheme=always: prepend "▁" when the segment does not start with it.
        if (normalized[0] != '▁') normalized = "▁" + normalized;

        // Metaspace split: each "▁" begins a piece (piece = this "▁" through the char before the next one).
        var start = 0;
        for (var i = 1; i < normalized.Length; i++)
        {
            if (normalized[i] == '▁')
            {
                EncodePiece(normalized.Substring(start, i - start), ids);
                start = i;
            }
        }
        if (start < normalized.Length)
            EncodePiece(normalized.Substring(start), ids);
    }

    /// <summary>BPE-encode one pre-tokenized piece (cached — records repeat pieces heavily).</summary>
    private void EncodePiece(string piece, List<int> ids)
    {
        if (_pieceCache.TryGetValue(piece, out var cached))
        {
            ids.AddRange(cached);
            return;
        }

        var computed = BpePiece(piece);
        _pieceCache[piece] = computed;
        ids.AddRange(computed);
    }

    /// <summary>Greedy lowest-rank BPE over characters, byte fallback + fused unk at the end.</summary>
    private int[] BpePiece(string piece)
    {
        // Initial symbols: single CHARACTERS (Unicode code points — surrogate pairs
        // stay whole; the training tokenizer operates on code points, not UTF-16 units).
        var symbols = new List<string>(piece.Length);
        for (var i = 0; i < piece.Length; )
        {
            if (char.IsSurrogatePair(piece, i) && i + 1 < piece.Length)
            {
                symbols.Add(piece.Substring(i, 2));
                i += 2;
            }
            else
            {
                symbols.Add(piece[i].ToString());
                i += 1;
            }
        }

        while (symbols.Count > 1)
        {
            int bestRank = int.MaxValue, bestIndex = -1;
            for (var i = 0; i < symbols.Count - 1; i++)
            {
                var left = _vocab.TryGetValue(symbols[i], out var li) ? li : -1;
                var right = _vocab.TryGetValue(symbols[i + 1], out var ri) ? ri : -1;
                if (left < 0 || right < 0) continue;
                if (!_mergeRanks.TryGetValue(MergeKey(left, right), out var rank)) continue;
                if (rank < bestRank)
                {
                    bestRank = rank;
                    bestIndex = i;
                }
            }
            if (bestIndex < 0) break;
            symbols[bestIndex] = symbols[bestIndex] + symbols[bestIndex + 1];
            symbols.RemoveAt(bestIndex + 1);
        }

        var ids = new List<int>(symbols.Count);
        var pendingUnknown = new StringBuilder();
        foreach (var symbol in symbols)
        {
            if (_vocab.TryGetValue(symbol, out var id))
            {
                FlushUnknown(pendingUnknown, ids);
                ids.Add(id);
                continue;
            }
            pendingUnknown.Append(symbol);
        }
        FlushUnknown(pendingUnknown, ids);
        return ids.ToArray();
    }

    /// <summary>Emit accumulated unknown symbols: byte fallback per UTF-8 byte, fused unk otherwise.</summary>
    private void FlushUnknown(StringBuilder pending, List<int> ids)
    {
        if (pending.Length == 0) return;
        var bytes = Encoding.UTF8.GetBytes(pending.ToString());
        pending.Clear();
        var allBytesKnown = true;
        foreach (var b in bytes)
        {
            if (!_byteTokens.TryGetValue(b, out _)) { allBytesKnown = false; break; }
        }
        if (allBytesKnown)
        {
            foreach (var b in bytes) ids.Add(_byteTokens[b]);
            return;
        }
        ids.Add(_vocab[UnknownToken]);
    }

    /// <summary>Merge-rank dictionary key for a (leftId, rightId) pair. Pure.</summary>
    private static long MergeKey(int leftId, int rightId)
        => ((long)leftId << 32) | (uint)rightId;

    /// <summary>Length of the added token starting at <paramref name="index"/> (longest match), else 0.</summary>
    private int MatchAddedToken(string text, int index)
    {
        if (!_addedByFirstChar.TryGetValue(text[index], out var bucket)) return 0;
        var longest = 0;
        foreach (var content in bucket)
        {
            if (content.Length <= longest || index + content.Length > text.Length) continue;
            if (content.Length > _maxAddedTokenLength) continue;
            if (string.CompareOrdinal(text, index, content, 0, content.Length) == 0)
                longest = content.Length;
        }
        return longest;
    }

    /// <summary>Index of the next added-token start at or after <paramref name="index"/>, or text end.</summary>
    private int NextAddedToken(string text, int index)
    {
        for (var i = index; i < text.Length; i++)
        {
            if (_addedByFirstChar.ContainsKey(text[i]) && MatchAddedToken(text, i) > 0)
                return i;
        }
        return text.Length;
    }

    // ── tokenizer.json loading ──────────────────────────────────────────

    private static (Dictionary<string, int> Vocab, Dictionary<long, int> Ranks,
        Dictionary<string, int> Added, Dictionary<byte, int> ByteTokens) LoadModel(JsonElement root)
    {
        var model = root.GetProperty("model");
        if (!model.TryGetProperty("type", out var typeEl) ||
            !string.Equals(typeEl.GetString(), "BPE", StringComparison.Ordinal))
            throw new InvalidDataException("judge tokenizer: only a BPE model is supported");

        var vocab = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (var entry in model.GetProperty("vocab").EnumerateObject())
            vocab[entry.Name] = entry.Value.GetInt32();

        var ranks = new Dictionary<long, int>();
        var rank = 0;
        foreach (var merge in model.GetProperty("merges").EnumerateArray())
        {
            // tokenizer.json merges are [left, right] pairs (new format) or "left right" strings (old).
            string left, right;
            if (merge.ValueKind == JsonValueKind.Array)
            {
                var parts = merge.EnumerateArray().ToArray();
                if (parts.Length != 2)
                    throw new InvalidDataException("judge tokenizer: malformed merge entry");
                left = parts[0].GetString() ?? string.Empty;
                right = parts[1].GetString() ?? string.Empty;
            }
            else
            {
                var raw = merge.GetString() ?? string.Empty;
                var space = raw.IndexOf(' ');
                if (space <= 0)
                    throw new InvalidDataException("judge tokenizer: malformed merge entry");
                left = raw[..space];
                right = raw[(space + 1)..];
            }
            if (!vocab.TryGetValue(left, out var leftId) || !vocab.TryGetValue(right, out var rightId))
                throw new InvalidDataException("judge tokenizer: merge references a token outside the vocab");
            ranks[MergeKey(leftId, rightId)] = rank++;
        }

        var byteTokens = new Dictionary<byte, int>();
        for (var b = 0; b < 256; b++)
        {
            var name = $"<0x{b:X2}>";
            if (vocab.TryGetValue(name, out var id)) byteTokens[(byte)b] = id;
        }

        var added = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (var addedEntry in root.GetProperty("added_tokens").EnumerateArray())
        {
            var content = addedEntry.GetProperty("content").GetString() ?? string.Empty;
            if (content.Length > 0) added[content] = addedEntry.GetProperty("id").GetInt32();
        }
        return (vocab, ranks, added, byteTokens);
    }

    private static int RequireId(Dictionary<string, int> vocab, string token)
        => vocab.TryGetValue(token, out var id)
            ? id
            : throw new InvalidDataException($"judge tokenizer: required special token '{token}' missing from vocab");
}
