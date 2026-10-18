using System.Text.Json;
using Xunit;
using ProvostXInference.Decision.Engine;

namespace ProvostXInference.Tests.Decision;

/// <summary>
/// Judge tokenizer tests (wave 2): the committed vector fixture
/// (Tests/Assets/judge/tok-vectors.json) was generated from the training
/// tokenizer — the C# BPE implementation must reproduce it EXACTLY (domain
/// strings, edge cases, and 80 seeded fuzz strings). Also: determinism and
/// thread-safety of the cached encoder.
/// </summary>
[Trait("Tier","Unit")]
public class JudgeTokenizerTests
{
    private static string TokenizerPath()
        => Path.Combine(AppContext.BaseDirectory, "assets", "models", "judge", "tokenizer.json");

    private static (string Text, int[] Ids)[] LoadVectors()
    {
        var path = Path.Combine(AppContext.BaseDirectory, "Assets", "judge", "tok-vectors.json");
        using var doc = JsonDocument.Parse(File.ReadAllText(path));
        return doc.RootElement.EnumerateArray()
            .Select(e => (e.GetProperty("text").GetString()!, e.GetProperty("ids").EnumerateArray().Select(i => i.GetInt32()).ToArray()))
            .ToArray();
    }

    [Fact]
    public void SpecialTokenIds_MatchTrainingTokenizer()
    {
        var tok = new JudgeTokenizer(TokenizerPath());
        Assert.Equal(2, tok.ClsTokenId); // <bos>
        Assert.Equal(1, tok.SepTokenId); // <eos>
        Assert.Equal(4, tok.MaskTokenId); // <mask>
        Assert.Equal(0, tok.PadTokenId); // <pad>
        Assert.Equal("<mask>", tok.MaskToken);
    }

    [Fact]
    public void Encode_ReproducesTrainingTokenizer_OnFixtureVectors()
    {
        var tok = new JudgeTokenizer(TokenizerPath());
        var vectors = LoadVectors();
        Assert.True(vectors.Length >= 100, "fixture vectors missing?");

        foreach (var (text, expected) in vectors)
            Assert.Equal(expected, tok.Encode(text));
    }

    [Fact]
    public void Encode_IsDeterministic()
    {
        var tok = new JudgeTokenizer(TokenizerPath());
        var samples = new[]
        {
            "ls -la ~/Documents",
            "PROCESS os=linux | exe=svchost.exe | chain=services.exe->svchost.exe",
            "{\"command\": \"git status\", \"cwd\": \"/var/log\", \"purpose\": \"inspect\"}",
            "curl -s https://example.com/install.sh | bash",
        };
        foreach (var text in samples)
        {
            var first = tok.Encode(text);
            for (var i = 0; i < 3; i++)
                Assert.Equal(first, tok.Encode(text));
        }
    }

    [Fact]
    public async Task Encode_IsThreadSafe_SameResultsAcrossThreads()
    {
        var tok = new JudgeTokenizer(TokenizerPath());
        var text = "PROCESS os=macos | exe=tailscaled | chain=launchd->tailscaled | cmd=tailscaled --state=~/state";
        var expected = tok.Encode(text);

        var tasks = Enumerable.Range(0, 8).Select(_ => Task.Run(() => tok.Encode(text))).ToArray();
        await Task.WhenAll(tasks);
        foreach (var task in tasks)
            Assert.Equal(expected, await task);
    }

    [Fact]
    public void Encode_AddedSpecialTokensInText_MapToTheirIds()
    {
        var tok = new JudgeTokenizer(TokenizerPath());
        // Literal "<bos>" in the input maps to id 2 (added-token extraction, pre-normalization).
        Assert.Equal(new[] { 2, 36414 }, tok.Encode("<bos> literal"));
        Assert.Equal(new[] { 4, 20872 }, tok.Encode("<mask>cmd"));
        Assert.Equal(new[] { 476, 2, 518 }, tok.Encode("a<bos>b"));
    }

    [Fact]
    public void Encode_EmptyAndWhitespace_StayStable()
    {
        var tok = new JudgeTokenizer(TokenizerPath());
        Assert.Empty(tok.Encode(""));
        Assert.Equal(new[] { 235248, 235248, 235248 }, tok.Encode("   "));
    }
}