using System.IO;
using ProvostXInference.Abstractions;
using ProvostXInference.Exceptions;
using ProvostXInference.Implementation;
using ProvostXInference.Models;
using Xunit;

namespace ProvostXInference.Tests;

/// <summary>
/// Speculative decoding (INF BL-05) — mirrors the native spec tests:
/// supported-types capability self-report, typed fail-closed boundaries
/// (NOT_SUPPORTED / INVALID_ARG / LOAD_FAILED with boundary names), config
/// round-trip, and a verify-loop smoke that also proves token-exactness vs
/// the plain Prompt/Infer/Sample loop. TARGETED tests only: real models via
/// env (PVX_MODEL_PATH / PVX_DRAFT_MODEL_PATH — draft and target must share a
/// vocab family), no slow full suites.
/// </summary>
public class SpecDecodeTests
{
    private static readonly string? ModelPath = Environment.GetEnvironmentVariable("PVX_MODEL_PATH");
    private static readonly string? DraftModelPath = Environment.GetEnvironmentVariable("PVX_DRAFT_MODEL_PATH");

    private static bool HasModel() => ModelPath is not null && File.Exists(ModelPath);
    private static bool HasDraft() => DraftModelPath is not null && File.Exists(DraftModelPath);

    private static NativeInferenceModel LoadTarget() =>
        NativeInferenceModel.Load(new ModelConfig { Path = ModelPath!, GpuLayers = 0 });

    private static NativeInferenceContext NewContext(NativeInferenceModel model) =>
        (NativeInferenceContext) model.CreateContext(new ContextConfig
        {
            ContextSize = 1024,
            BatchSize = 512,
            SeqMax = 1,
        });

    private static SamplingConfig Greedy() => new()
    {
        Temperature = 0.0f,
        TopK = 1,
        TopP = 1.0f,
        RepeatPenalty = 1.0f,
        MaxTokens = 24,
    };

    // ═══════════════════════════════════════════════════
    // 1. Capability self-report (supported-types list)
    // ═══════════════════════════════════════════════════

    [Fact]
    public void Capabilities_SupportedTypes_List_From_Metadata()
    {
        if (!HasModel()) return; // skip when no test model
        using var model = LoadTarget();
        var caps = model.SpeculativeCapabilities;

        // Dense llama-family target: draft_simple reported; no nextn tensors in
        // this GGUF -> no mtp bit; eagle3/dflash/dspark not reported. ABI v6:
        // the n-gram drafters are model-independent (standard vocab suffices)
        // and the spec engine's mirror is embd-aware (spec+vision).
        Assert.True(caps.Draft);
        Assert.False(caps.Mtp);
        Assert.Equal(new List<string> { "draft_simple", "ngram-simple", "ngram-map-k",
                                       "ngram-mod", "ngram-cache" }, caps.SupportedTypeNames);
        Assert.True(caps.Supports(SpeculativeType.DraftSimple));
        Assert.True(caps.Supports(SpeculativeType.Ngram));
        Assert.True(caps.Supports(SpeculativeType.NgramMapK));
        Assert.True(caps.Supports(SpeculativeType.NgramMod));
        Assert.True(caps.Supports(SpeculativeType.NgramCache));
        Assert.False(caps.Supports(SpeculativeType.Mtp));
        Assert.True(caps.SpecVision); // ABI v6 embd-aware mirror
    }

    // ═══════════════════════════════════════════════════
    // 2. Typed fail-closed boundaries
    // ═══════════════════════════════════════════════════

    [Fact]
    public void Validate_Off_Ok_Mtp_TypedNotSupported_DraftWithoutPath_Invalid()
    {
        if (!HasModel()) return; // skip when no test model
        using var model = LoadTarget();

        // OFF: valid configuration (disable).
        model.ValidateSpeculation(new SpeculativeConfig { Mode = SpeculativeMode.Off });

        // MTP capability bit is 0 for this model → typed NOT_SUPPORTED.
        var mtpEx = Assert.Throws<InferenceException>(() =>
            model.ValidateSpeculation(new SpeculativeConfig { Mode = SpeculativeMode.Mtp }));
        Assert.Equal(7, mtpEx.ErrorCode); // PVX_ERR_NOT_SUPPORTED
        // ABI v5: the typed error carries its reason (no more empty "NOT_SUPPORTED: ").
        Assert.Contains("capability", mtpEx.Message);

        // Draft without a path: typed invalid arg, never a silent disable.
        var noPathEx = Assert.Throws<InferenceException>(() =>
            model.ValidateSpeculation(new SpeculativeConfig { Mode = SpeculativeMode.Draft }));
        Assert.Equal(1, noPathEx.ErrorCode); // PVX_ERR_INVALID_ARG

        // Shaped-config draft with a (nonexistent) path: validation passes —
        // load failure happens at enable time, typed LOAD_FAILED there.
        model.ValidateSpeculation(new SpeculativeConfig
        {
            Mode = SpeculativeMode.Draft,
            DraftModelPath = "/nonexistent/spec-draft.gguf",
        });
    }

    [Fact]
    public void Enable_FailClosed_TypedErrors_NeverSilentFallback()
    {
        if (!HasModel()) return; // skip when no test model
        using var model = LoadTarget();
        using var ctx = NewContext(model);
        using var exec = (NativeStandardExecutor) ctx.CreateExecutor();

        // MTP: typed NOT_SUPPORTED with the wiring boundary named.
        var mtpEx = Assert.Throws<InferenceException>(() =>
            exec.EnableSpeculation(new SpeculativeConfig { Mode = SpeculativeMode.Mtp }));
        Assert.Equal(7, mtpEx.ErrorCode);
        Assert.Contains("mtp-wiring", mtpEx.Message);

        // Draft without path: typed invalid arg.
        var noPathEx = Assert.Throws<InferenceException>(() =>
            exec.EnableSpeculation(new SpeculativeConfig { Mode = SpeculativeMode.Draft }));
        Assert.Equal(1, noPathEx.ErrorCode);
        Assert.Contains("draft-config", noPathEx.Message);

        // Draft with unloadable path: typed LOAD_FAILED, boundary named, no fallback.
        var loadEx = Assert.Throws<InferenceException>(() =>
            exec.EnableSpeculation(new SpeculativeConfig
            {
                Mode = SpeculativeMode.Draft,
                DraftModelPath = "/nonexistent/spec-draft.gguf",
            }));
        Assert.Equal(3, loadEx.ErrorCode); // PVX_ERR_LOAD_FAILED
        Assert.Contains("draft-load", loadEx.Message);

        // No session was created by any of the failures — generate stays typed
        // NOT_SUPPORTED (never a silent non-spec run).
        var noSessionEx = Assert.Throws<InferenceException>(() =>
            exec.GenerateSpeculative(Greedy(), 8));
        Assert.Equal(7, noSessionEx.ErrorCode);
    }

    // ═══════════════════════════════════════════════════
    // 3. Verify-loop smoke + token-exactness (real draft)
    // ═══════════════════════════════════════════════════

    [Fact]
    public void VerifyLoop_Greedy_TokenExact_And_SessionLifecycle()
    {
        if (!HasModel() || !HasDraft()) return; // skip without a compatible pair
        var greedy = Greedy();
        string prompt = "List three colors:";

        // ── Plain reference loop: Prompt → Infer → [Sample → PromptTokens → Infer]×N ──
        IReadOnlyList<int> plainTokens;
        using (var model = LoadTarget())
        using (var ctx = NewContext(model))
        using (var exec = ctx.CreateExecutor())
        {
            exec.Prompt(prompt);
            Assert.Equal(InferResult.Ok, exec.Infer());
            var acc = new List<int>();
            for (int i = 0; i < 12; i++)
            {
                var t = exec.Sample(greedy);
                if (ctx.IsEos(t)) break;
                acc.Add(t);
                exec.PromptTokens(new[] { t });
                Assert.Equal(InferResult.Ok, exec.Infer());
            }
            plainTokens = acc;
        }

        // ── Spec session on a fresh executor: same prompt, same params ──
        using (var model = LoadTarget())
        using (var ctx = NewContext(model))
        using (var exec = (NativeStandardExecutor) ctx.CreateExecutor())
        {
            var streamed = new List<int>();
            exec.EnableSpeculation(new SpeculativeConfig
            {
                Mode = SpeculativeMode.Draft,
                DraftModelPath = DraftModelPath,
                DraftMax = 4,
            });
            exec.Prompt(prompt);
            Assert.Equal(InferResult.Ok, exec.Infer());

            var specTokens = exec.GenerateSpeculative(greedy, 12, t => streamed.Add(t));
            Assert.Equal(streamed, specTokens.ToList()); // stream == returned ids
            Assert.InRange(specTokens.Count, 1, 12);
            Assert.True(exec.TokenCount > 0);

            // TOKEN-EXACT vs the plain loop (identical params, greedy).
            var expected = plainTokens.Take(specTokens.Count).ToList();
            Assert.Equal(expected, specTokens.ToList());

            // Re-enable replaces the session cleanly; Off clears it; after
            // clearing, generation is typed NOT_SUPPORTED again.
            exec.EnableSpeculation(new SpeculativeConfig
            {
                Mode = SpeculativeMode.Draft,
                DraftModelPath = DraftModelPath,
                DraftMax = 4,
            });
            exec.DisableSpeculation();
            var offEx = Assert.Throws<InferenceException>(() =>
                exec.GenerateSpeculative(greedy, 4));
            Assert.Equal(7, offEx.ErrorCode);
        }
    }

    // ═════════════════════════════════════════════════
    // 3b. ABI v6: n-gram modes — typed boundaries + token-exactness
    // ═════════════════════════════════════════════════

    [Fact]
    public void Ngram_FailClosed_TypedErrors_NeverSilentFallback()
    {
        if (!HasModel()) return; // skip when no test model
        using var model = LoadTarget();
        using var ctx = NewContext(model);
        using var exec = (NativeStandardExecutor) ctx.CreateExecutor();

        // n_min > n_max: typed invalid arg (validate + enable agree).
        var bad = new SpeculativeConfig
        {
            Mode = SpeculativeMode.NgramSimple,
            Ngram = new NgramConfig { NMin = 8, NMax = 2 },
        };
        var badEx = Assert.Throws<InferenceException>(() => model.ValidateSpeculation(bad));
        Assert.Equal(1, badEx.ErrorCode); // PVX_ERR_INVALID_ARG
        Assert.Contains("ngram-config", badEx.Message);
        Assert.Throws<InferenceException>(() => exec.EnableSpeculation(bad));

        // Corrupt cache file: typed LOAD_FAILED naming the path — never a
        // silently empty cache.
        var dir = Path.Combine(Path.GetTempPath(), $"pvx-ngram-test-{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);
        var cachePath = Path.Combine(dir, "corrupt.bin");
        File.WriteAllBytes(cachePath, new byte[] { 1, 2, 3, 4, 5 });
        try
        {
            var cacheEx = Assert.Throws<InferenceException>(() =>
                exec.EnableSpeculation(new SpeculativeConfig
                {
                    Mode = SpeculativeMode.NgramCache,
                    Ngram = new NgramConfig { CachePath = cachePath },
                }));
            Assert.Equal(3, cacheEx.ErrorCode); // PVX_ERR_LOAD_FAILED
            Assert.Contains("ngram-cache", cacheEx.Message);
        }
        finally { Directory.Delete(dir, recursive: true); }
    }

    public static TheoryData<SpeculativeMode> NgramModes => new()
    {
        SpeculativeMode.NgramSimple,
        SpeculativeMode.NgramMapK,
        SpeculativeMode.NgramMod,
        SpeculativeMode.NgramCache,
    };

    [Theory]
    [MemberData(nameof(NgramModes))]
    public void Ngram_TokenExact_Vs_PlainLoop(SpeculativeMode mode)
    {
        if (!HasModel()) return; // skip without a test model
        var greedy = Greedy();
        const string prompt = "The colors of the rainbow, in order, are: red, orange, yellow, green, blue, indigo and";
        const int nGen = 16;

        // ── Plain reference loop ──
        IReadOnlyList<int> plainTokens;
        using (var model = LoadTarget())
        using (var ctx = NewContext(model))
        using (var exec = ctx.CreateExecutor())
        {
            exec.Prompt(prompt);
            Assert.Equal(InferResult.Ok, exec.Infer());
            var acc = new List<int>();
            for (int i = 0; i < nGen; i++)
            {
                var t = exec.Sample(greedy);
                if (ctx.IsEos(t)) break;
                acc.Add(t);
                exec.PromptTokens(new[] { t });
                Assert.Equal(InferResult.Ok, exec.Infer());
            }
            plainTokens = acc;
        }

        // ── n-gram spec on a fresh executor: same prompt, same params ──
        using (var model = LoadTarget())
        using (var ctx = NewContext(model))
        using (var exec = (NativeStandardExecutor) ctx.CreateExecutor())
        {
            exec.EnableSpeculation(new SpeculativeConfig { Mode = mode });
            exec.Prompt(prompt);
            Assert.Equal(InferResult.Ok, exec.Infer());

            var specTokens = exec.GenerateSpeculative(greedy, nGen);
            Assert.True(specTokens.Count >= 1);
            Assert.True(exec.TokenCount > 0);
            // TOKEN-EXACT vs the plain loop (identical params, greedy).
            Assert.Equal(plainTokens.Take(specTokens.Count).ToList(), specTokens.ToList());
        }
    }
}
