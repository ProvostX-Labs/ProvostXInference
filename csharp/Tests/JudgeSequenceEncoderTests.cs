using System.Text.Json;
using Xunit;
using ProvostXInference.Decision.Engine;

namespace ProvostXInference.Tests.Decision;

/// <summary>
/// Judge sequence encoder tests (wave 2): record → serialized training state →
/// question rows must match the TRAINING pipeline byte-for-byte on the committed
/// fixtures (Tests/Assets/judge/seq-fixtures.json), including truncation of the
/// state tail to the sequence budget and marker placement.
/// </summary>
[Trait("Tier","Unit")]
public class JudgeSequenceEncoderTests
{
    private static (string Text, int[] Ids)[] TokenVectors()
    {
        var path = Path.Combine(AppContext.BaseDirectory, "Assets", "judge", "tok-vectors.json");
        using var doc = JsonDocument.Parse(File.ReadAllText(path));
        return doc.RootElement.EnumerateArray()
            .Select(e => (e.GetProperty("text").GetString()!, e.GetProperty("ids").EnumerateArray().Select(i => i.GetInt32()).ToArray()))
            .ToArray();
    }

    private sealed record FixtureRow(string Id, int QType, int[] Ids, int[] Markers);
    private sealed record FixtureEntry(string Kind, string StateJson, List<FixtureRow> Rows);

    private static List<FixtureEntry> LoadSeqFixtures()
    {
        var path = Path.Combine(AppContext.BaseDirectory, "Assets", "judge", "seq-fixtures.json");
        using var doc = JsonDocument.Parse(File.ReadAllText(path));
        var entries = new List<FixtureEntry>();
        foreach (var e in doc.RootElement.EnumerateArray())
        {
            var rows = new List<FixtureRow>();
            foreach (var r in e.GetProperty("rows").EnumerateArray())
            {
                rows.Add(new FixtureRow(
                    r.GetProperty("id").GetString()!,
                    r.GetProperty("qtype").GetInt32(),
                    r.GetProperty("ids").EnumerateArray().Select(i => i.GetInt32()).ToArray(),
                    r.GetProperty("markers").EnumerateArray().Select(i => i.GetInt32()).ToArray()));
            }
            entries.Add(new FixtureEntry(
                e.GetProperty("kind").GetString()!,
                e.GetProperty("state_json").GetString()!,
                rows));
        }
        return entries;
    }

    private static DecisionRecord ShellRecord(string command, string cwd) => new()
    {
        Kind = JudgeRecordKind.Shell,
        Command = command,
        Cwd = cwd,
    };

    private static DecisionRecord ProcessRecord(string raw)
    {
        // Build the PROCESS record the same way the endpoint parser normalizes it:
        // fields are '|'-separated key=value pairs; the leading PROCESS token is
        // optional. (The parser itself is LLM server-domain; the encoder contract
        // under test is the canonical re-serialization.)
        var trimmed = raw.Trim();
        if (trimmed.StartsWith("PROCESS", StringComparison.OrdinalIgnoreCase))
            trimmed = trimmed.Length > 7 ? trimmed[7..].TrimStart() : string.Empty;
        var fields = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        foreach (var segment in trimmed.Split('|'))
        {
            var seg = segment.Trim();
            if (seg.Length == 0) continue;
            var eq = seg.IndexOf('=');
            if (eq <= 0) continue;
            fields[seg[..eq].Trim()] = seg[(eq + 1)..].Trim();
        }
        var chain = fields.TryGetValue("chain", out var ch) ? ch : string.Empty;
        return new DecisionRecord
        {
            Kind = JudgeRecordKind.Process,
            Os = fields.GetValueOrDefault("os", string.Empty).ToLowerInvariant(),
            Exe = fields.GetValueOrDefault("exe", string.Empty),
            Cmd = fields.GetValueOrDefault("cmd", string.Empty),
            Net = fields.GetValueOrDefault("net", string.Empty),
            Writes = fields.GetValueOrDefault("writes", string.Empty),
            Chain = string.IsNullOrWhiteSpace(chain)
                ? Array.Empty<string>()
                : chain.Split("->", StringSplitOptions.TrimEntries).Where(x => x.Length > 0).ToArray(),
        };
    }

    [Fact]
    public void SerializeRecord_Shell_MatchesTrainingJsonExactly()
    {
        // Python json.dumps({"command": ..., "cwd": ..., "purpose": "inspect"}, ensure_ascii=False)
        Assert.Equal("{\"command\": \"ls -la ~/Documents\", \"cwd\": \"/var/log\", \"purpose\": \"inspect\"}",
            JudgeSequenceEncoder.SerializeRecord(ShellRecord("ls -la ~/Documents", "/var/log")));
        // control characters / quotes escape like the training serializer
        Assert.Equal("{\"command\": \"echo \\\"hi\\\" && rm -rf /\", \"cwd\": \"/\", \"purpose\": \"inspect\"}",
            JudgeSequenceEncoder.SerializeRecord(ShellRecord("echo \"hi\" && rm -rf /", "/")));
    }

    [Fact]
    public void SerializeRecord_Process_RebuildsCanonicalTrainingFormat()
    {
        // Training state = JSON dict with the raw PROCESS record in "command"
        // and constant cwd/purpose (matches the process dataset exactly).
        var record = ProcessRecord("PROCESS os=linux | exe=svchost.exe | chain=services.exe->svchost.exe");
        Assert.Equal(
            "{\"command\": \"PROCESS os=linux | exe=svchost.exe | chain=services.exe->svchost.exe\", \"cwd\": \"\", \"purpose\": \"sentinel\"}",
            JudgeSequenceEncoder.SerializeRecord(record));

        var withAll = ProcessRecord("PROCESS os=windows | exe=rundll32.exe | chain=OUTLOOK.EXE->rundll32.exe | cmd=rundll32 comsvcs.dll MiniDump 1234 dump.bin full | writes=dump.bin");
        Assert.Contains("cmd=rundll32 comsvcs.dll MiniDump 1234 dump.bin full | writes=dump.bin",
            JudgeSequenceEncoder.SerializeRecord(withAll));
    }

    [Fact]
    public void Encode_ShellRecord_MatchesTrainingRows()
    {
        var assets = new DecisionTestAssets();
        var tokenizer = assets.NewTokenizer();
        var encoder = new JudgeSequenceEncoder(tokenizer, assets.NewShellSpec());
        var fixtures = LoadSeqFixtures();

        var shell = fixtures.First(f => f.Kind == "shell" && f.StateJson.Contains("ls -la"));
        var rows = encoder.Encode(ShellRecord("ls -la ~/Documents", "/home/user"));

        Assert.Equal(3, rows.Count);
        for (var i = 0; i < 3; i++)
        {
            Assert.Equal(shell.Rows[i].QType, (int)rows[i].QuestionType);
            Assert.Equal(shell.Rows[i].Ids, rows[i].InputIds);
            Assert.Equal(shell.Rows[i].Markers, rows[i].MarkerPositions);
        }
    }

    [Fact]
    public void Encode_ProcessRecord_MatchesTrainingRows()
    {
        var assets = new DecisionTestAssets();
        var tokenizer = assets.NewTokenizer();
        var encoder = new JudgeSequenceEncoder(tokenizer, assets.NewProcessSpec());
        var fixtures = LoadSeqFixtures();

        var process = fixtures.First(f => f.Kind == "process");
        var raw = "PROCESS os=linux | exe=svchost.exe | chain=services.exe->svchost.exe";
        var rows = encoder.Encode(ProcessRecord(raw));

        Assert.Equal(3, rows.Count);
        for (var i = 0; i < 3; i++)
        {
            Assert.Equal(process.Rows[i].QType, (int)rows[i].QuestionType);
            Assert.Equal(process.Rows[i].Ids, rows[i].InputIds);
            Assert.Equal(process.Rows[i].Markers, rows[i].MarkerPositions);
        }
    }

    [Fact]
    public void Encode_LongState_TruncatesToMaxLen_WithStateTailTruncation()
    {
        var assets = new DecisionTestAssets();
        var tokenizer = assets.NewTokenizer();
        var encoder = new JudgeSequenceEncoder(tokenizer, assets.NewShellSpec());
        var fixtures = LoadSeqFixtures();
        var longFixture = fixtures.First(f => f.StateJson.Contains("x/x/", StringComparison.Ordinal));

        var command = "rm " + string.Concat(Enumerable.Repeat("x/", 400));
        var rows = encoder.Encode(ShellRecord(command, "/var/log"));

        Assert.Equal(3, rows.Count);
        for (var i = 0; i < 3; i++)
        {
            Assert.Equal(longFixture.Rows[i].QType, (int)rows[i].QuestionType);
            Assert.Equal(longFixture.Rows[i].Ids, rows[i].InputIds);
            Assert.Equal(longFixture.Rows[i].Markers, rows[i].MarkerPositions);
            Assert.True(rows[i].InputIds.Length <= 1024);
        }
    }

    [Fact]
    public void Encode_RowStructure_HasClsSepAndMarkerMaskIds()
    {
        var assets = new DecisionTestAssets();
        var tokenizer = assets.NewTokenizer();
        var encoder = new JudgeSequenceEncoder(tokenizer, assets.NewShellSpec());

        var rows = encoder.Encode(ShellRecord("git status", "/home/user"));
        var actionRow = rows[0];
        Assert.Equal(JudgeQuestionType.Choice, actionRow.QuestionType);
        Assert.Equal(tokenizer.ClsTokenId, actionRow.InputIds[0]); // [CLS]
        Assert.Equal(tokenizer.SepTokenId, actionRow.InputIds[^1]); // final [SEP]
        // 3 options → 3 markers, each pointing at a MASK token
        Assert.Equal(3, actionRow.MarkerPositions.Length);
        foreach (var marker in actionRow.MarkerPositions)
            Assert.Equal(tokenizer.MaskTokenId, actionRow.InputIds[marker]);
    }

    [Fact]
    public void TokenizerVectorFixture_CoversTheDomain()
    {
        // Sanity: the committed fixture must exist and be substantial — the
        // tokenizer parity claim rests on it.
        Assert.True(TokenVectors().Length >= 100);
    }
}