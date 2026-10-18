using System.IO;
using ProvostXInference.Abstractions;
using ProvostXInference.Exceptions;
using ProvostXInference.Implementation;
using ProvostXInference.Interop;
using ProvostXInference.Models;
using ProvostXInference.Engines;
using ProvostXInference.Tests.Engines;
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

    // ═══════════════════════════════════════════════════
    // ABI v7 (wave 2): companion-draft modes, batched/pool lane, JSON schema
    // ═══════════════════════════════════════════════════

    /// <summary>Blittable struct discipline: wave-2 modes map onto the native
    /// enum numerically and the pinned companion pointers ride the ABI v7
    /// field appends (pure — no model needed).</summary>
    [Fact]
    public void SpecParams_Wave2_Modes_And_Companion_Pointers_Map()
    {
        var eagle = new SpeculativeConfig { Mode = SpeculativeMode.Eagle3, EagleCompanionPath = "/x/eagle.gguf" };
        var dflash = new SpeculativeConfig { Mode = SpeculativeMode.Dflash, DflashCompanionPath = "/x/dflash.gguf" };
        var draft = new SpeculativeConfig { Mode = SpeculativeMode.Draft, DraftModelPath = "/x/draft.gguf" };

        var eaglePtr = PvxNative.AllocUTF8("/x/eagle.gguf");
        var dflashPtr = PvxNative.AllocUTF8("/x/dflash.gguf");
        var draftPtr = PvxNative.AllocUTF8("/x/draft.gguf");
        try
        {
            var p1 = new PvxSpecParams(eagle, IntPtr.Zero, IntPtr.Zero, eaglePtr, IntPtr.Zero);
            Assert.Equal((int) PvxSpecMode.Eagle3, p1.Mode);
            Assert.Equal(eaglePtr, p1.EagleCompanionPath);
            Assert.Equal(IntPtr.Zero, p1.DflashCompanionPath);

            var p2 = new PvxSpecParams(dflash, IntPtr.Zero, IntPtr.Zero, IntPtr.Zero, dflashPtr);
            Assert.Equal((int) PvxSpecMode.Dflash, p2.Mode);
            Assert.Equal(dflashPtr, p2.DflashCompanionPath);
            Assert.Equal(IntPtr.Zero, p2.EagleCompanionPath);

            var p3 = new PvxSpecParams(draft, draftPtr, IntPtr.Zero, IntPtr.Zero, IntPtr.Zero);
            Assert.Equal((int) PvxSpecMode.Draft, p3.Mode);
            Assert.Equal(draftPtr, p3.DraftModelPath);

            // field growth: ABI v7 appends AFTER ngram (layout order contract)
            var offEagle = System.Runtime.InteropServices.Marshal.OffsetOf<PvxSpecParams>(nameof(PvxSpecParams.EagleCompanionPath));
            var offCache = System.Runtime.InteropServices.Marshal.OffsetOf<PvxSpecParams>(nameof(PvxSpecParams.NgramCachePath));
            Assert.True((long) offEagle >= (long) offCache, "eagle path must append after the ngram block");
        }
        finally
        {
            System.Runtime.InteropServices.Marshal.FreeHGlobal(eaglePtr);
            System.Runtime.InteropServices.Marshal.FreeHGlobal(dflashPtr);
            System.Runtime.InteropServices.Marshal.FreeHGlobal(draftPtr);
        }
    }

    /// <summary>Unknown speculative modes are a typed managed refusal at the
    /// blit boundary (pure).</summary>
    [Fact]
    public void SpecParams_Unknown_Mode_Refuses_Typed()
    {
        var bad = new SpeculativeConfig { Mode = (SpeculativeMode) 99 };
        Assert.Throws<ArgumentException>(() => new PvxSpecParams(bad, IntPtr.Zero, IntPtr.Zero, IntPtr.Zero, IntPtr.Zero));
    }

    /// <summary>Capability-gated typed refusals for the companion modes on a
    /// model without companion metadata: missing companion path AND an
    /// unloadable companion file both type fail-closed naming the boundary —
    /// never a silent non-spec enable (env-gated: mirrors native legs 30/31).</summary>
    [Fact]
    public void Eagle3_Dflash_Refuse_Typed_Without_Companion()
    {
        if (!HasModel()) return;
        using var model = LoadTarget();

        // validate is SHAPE-level for companion modes: a path present = OK
        // (mirror of native leg 31: pair-level validation happens at spec_init)
               var named = new SpeculativeConfig { Mode = SpeculativeMode.Eagle3, EagleCompanionPath = "x.gguf" };
        model.ValidateSpeculation(named); // no throw

        // executor lane: enabling types the refusals (mirror of native legs 30/31)
        using var ctx = NewContext(model);
        using var exec = (NativeStandardExecutor) ctx.CreateExecutor();

        // missing companion path -> typed INVALID_ARG
        var noPath = new SpeculativeConfig { Mode = SpeculativeMode.Eagle3 };
        var ex1 = Assert.Throws<InferenceException>(() => exec.EnableSpeculation(noPath));
        Assert.Equal((int) PvxResult.ErrInvalidArg, ex1.ErrorCode);
        Assert.Null(exec.ActiveSpeculation); // mirror untouched on refusal

        // unloadable companion file -> typed refusal at the load boundary
        var badPath = new SpeculativeConfig { Mode = SpeculativeMode.Eagle3, EagleCompanionPath = "/nonexistent/eagle3.gguf" };
        var ex2 = Assert.Throws<InferenceException>(() => exec.EnableSpeculation(badPath));
        Assert.True(ex2.ErrorCode == (int) PvxResult.ErrLoadFailed || ex2.ErrorCode == (int) PvxResult.ErrNotSupported);

        // DFlash: missing path typed; metadata boundary typed
        var dfNo = new SpeculativeConfig { Mode = SpeculativeMode.Dflash };
        var ex3 = Assert.Throws<InferenceException>(() => exec.EnableSpeculation(dfNo));
        Assert.Equal((int) PvxResult.ErrInvalidArg, ex3.ErrorCode);
        var dfBad = new SpeculativeConfig { Mode = SpeculativeMode.Dflash, DflashCompanionPath = "/nonexistent/dflash.gguf" };
        Assert.Throws<InferenceException>(() => exec.EnableSpeculation(dfBad));
        Assert.Null(exec.ActiveSpeculation);
    }

    /// <summary>Batched/pool lane (ABI v7 feature 5): per-conversation n-gram
    /// spec session is TOKEN-EXACT with the plain pooled lane for identical
    /// greedy params, the mirror reports the active config, stats never lie,
    /// teardown is idempotent, and generation without a session is a typed
    /// refusal — never a silent plain run (env-gated; mirrors native legs
    /// 25/26/29).</summary>
    [Fact]
    public void Conversation_Spec_Lane_Token_Exact_And_Typed_Refusals()
    {
        if (!HasModel()) return;
        var greedy = Greedy();
        const string prompt = "The colors of the rainbow, in order, are: red, orange, yellow, green, blue, indigo and";
        const int nGen = 12;

        // plain pooled-lane reference (seq_max 2 context, no speculation)
        IReadOnlyList<int> plain;
        using (var model = LoadTarget())
        {
            using var ctx = (NativeInferenceContext) model.CreateContext(new ContextConfig
            {
                ContextSize = 1024, BatchSize = 512, SeqMax = 2,
            });
            using var pool = ctx.CreatePool();
            var conv = (NativeConversation) pool.Lease();
            try
            {
                conv.Prompt(prompt);
                Assert.Equal(InferResult.Ok, ctx.InferAll()); // decode the prompt first (pooled-lane contract)
                var acc = new List<int>();
                for (int i = 0; i < nGen; i++)
                {
                    var t = conv.Sample(greedy);
                    if (ctx.IsEos(t)) break;
                    acc.Add(t);
                    conv.PromptTokens(new[] { t });
                    Assert.Equal(InferResult.Ok, ctx.InferAll());
                }
                plain = acc;
            }
            finally { pool.Return(conv); }
        }

        // spec session on a fresh identical pair
        using (var model = LoadTarget())
        {
            using var ctx = (NativeInferenceContext) model.CreateContext(new ContextConfig
            {
                ContextSize = 1024, BatchSize = 512, SeqMax = 2,
            });
            using var pool = ctx.CreatePool();
            var conv = (NativeConversation) pool.Lease();
            try
            {
                // typed refusal BEFORE any session: never a silent plain run
                Assert.Throws<InferenceException>(() => conv.GenerateSpeculative(greedy, 4));

                conv.Prompt(prompt);
                Assert.Equal(InferResult.Ok, ctx.InferAll());
                conv.EnableSpeculation(new SpeculativeConfig { Mode = SpeculativeMode.NgramSimple });
                Assert.NotNull(conv.ActiveSpeculation);
                Assert.Equal(SpeculativeMode.NgramSimple, conv.ActiveSpeculation!.Mode);

                var specTokens = conv.GenerateSpeculative(greedy, nGen);
                Assert.True(specTokens.Count >= 1);
                // TOKEN-EXACT vs the plain pooled loop (identical greedy params)
                Assert.Equal(plain.Take(specTokens.Count).ToList(), specTokens.ToList());

                var stats = conv.SpeculationStats();
                Assert.True(stats.Drafted >= stats.Accepted && stats.Accepted >= 0,
                    $"stats never lie: drafted={stats.Drafted} accepted={stats.Accepted}");

                conv.DisableSpeculation();
                Assert.Null(conv.ActiveSpeculation);
                conv.DisableSpeculation(); // idempotent
                Assert.Throws<InferenceException>(() => conv.GenerateSpeculative(greedy, 2));
            }
            finally { pool.Return(conv); }
        }
    }

    /// <summary>Pool return tears the conversation spec session down and the
    /// managed mirror never reports a phantom session (env-gated).</summary>
    [Fact]
    public void Conversation_Spec_Session_Dies_With_Pool_Return()
    {
        if (!HasModel()) return;
        using var model = LoadTarget();
        using var ctx = (NativeInferenceContext) model.CreateContext(new ContextConfig
        {
            ContextSize = 1024, BatchSize = 512, SeqMax = 2,
        });
        using var pool = ctx.CreatePool();
        var conv = (NativeConversation) pool.Lease();
        conv.EnableSpeculation(new SpeculativeConfig { Mode = SpeculativeMode.NgramSimple });
        Assert.NotNull(conv.ActiveSpeculation);
        pool.Return(conv);
        Assert.Null(conv.ActiveSpeculation); // mirror dropped with the session
    }

    /// <summary>Facade view: EnableConversationSpeculation is capability-gated
    /// (model self-report validation first) and the convenience surface wires
    /// through to the conversation (pure doubles — mirrors the fake-based ops
    /// suite discipline).</summary>
    [Fact]
    public void Facade_Conversation_Spec_Is_Capability_Gated()
    {
        var model = new FakeEngineModel();
        var backend = new FakeEngineBackend();
        using var ops = new GgufEngineOps(model, new ContextConfig(), new SamplingConfig(), backend);
        var pool = ops.Pool;
        var conv = pool.Lease();
        Assert.Throws<ArgumentNullException>(() => { ops.EnableConversationSpeculation(conv, null!); });
        // FakeEngineModel.ValidateSpeculation is a no-op gate -> enable wires through
        var cfg = new SpeculativeConfig { Mode = SpeculativeMode.NgramSimple };
        ops.EnableConversationSpeculation(conv, cfg);
        Assert.Equal(cfg, conv.ActiveSpeculation);
        ops.DisableConversationSpeculation(conv);
        Assert.Null(conv.ActiveSpeculation);
        Assert.Equal(SpeculationStats.None, ops.ConversationSpeculationStats(conv));
    }
}
