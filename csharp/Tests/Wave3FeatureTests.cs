using ProvostXInference.Abstractions;
using Xunit;
using ProvostXInference.Exceptions;
using ProvostXInference.Implementation;
using ProvostXInference.Models;

namespace ProvostXInference.Tests;

/// <summary>
/// ABI v8 (wave 3) surface tests: audio input, TTS, rerank, LoRA attach/
/// detach, control vectors, stats export. Live legs are env-gated (typed
/// graceful skip when unset) — same discipline as the spec suite:
/// PVX_AUDIO_MODEL_PATH/PVX_AUDIO_MMPROJ_PATH (Qwen3-ASR pair),
/// PVX_TTS_MODEL_PATH/PVX_TTS_MMPROJ_PATH (Qwen3-TTS pair),
/// PVX_RERANK_MODEL_PATH (jina-bert-v2 RANK-head conversion),
/// PVX_LORA_ADAPTER_PATH (Llama-3.2-3B adapter, arch-matched to
/// PVX_MODEL_PATH), PVX_CVEC_TEST_FILE (synthetic direction.GGUF produced by
/// the native suite — direction.1..28 F32, ne0 = 3072).
/// </summary>
public class Wave3FeatureTests
{
    private static readonly string? ModelPath = Environment.GetEnvironmentVariable("PVX_MODEL_PATH");
    private static readonly string? HybridModelPath = Environment.GetEnvironmentVariable("PVX_HYBRID_MODEL_PATH");
    private static readonly string? MmprojPath = Environment.GetEnvironmentVariable("PVX_MMPROJ_PATH");
    private static readonly string? AudioModelPath = Environment.GetEnvironmentVariable("PVX_AUDIO_MODEL_PATH");
    private static readonly string? AudioMmprojPath = Environment.GetEnvironmentVariable("PVX_AUDIO_MMPROJ_PATH");
    private static readonly string? TtsModelPath = Environment.GetEnvironmentVariable("PVX_TTS_MODEL_PATH");
    private static readonly string? TtsMmprojPath = Environment.GetEnvironmentVariable("PVX_TTS_MMPROJ_PATH");
    private static readonly string? RerankModelPath = Environment.GetEnvironmentVariable("PVX_RERANK_MODEL_PATH");
    private static readonly string? LoraAdapterPath = Environment.GetEnvironmentVariable("PVX_LORA_ADAPTER_PATH");
    private static readonly string? CvecTestFile = Environment.GetEnvironmentVariable("PVX_CVEC_TEST_FILE");

    private static bool HasModel() => ModelPath is not null && File.Exists(ModelPath);
    private static bool Has(string? p) => p is not null && File.Exists(p);

    private static NativeInferenceModel LoadModel(string path, int gpuLayers = 99) =>
        NativeInferenceModel.Load(new ModelConfig { Path = path, GpuLayers = gpuLayers, FlashAttention = true });

    private static NativeInferenceContext MakeCtx(NativeInferenceModel m, uint ctxSize, uint seqMax = 1,
        PoolingType pooling = PoolingType.None) =>
        (NativeInferenceContext)m.CreateContext(new ContextConfig
        {
            ContextSize = ctxSize, BatchSize = 512, SeqMax = seqMax, PoolingType = pooling,
        });

    // ── rerank: typed pooling gate + live scoring ──

    [Fact]
    public void Rerank_Non_Rank_Context_Refuses_Typed()
    {
        if (!HasModel()) return;
        using var m = LoadModel(ModelPath!);
        using var ctx = MakeCtx(m, 512);
        var ex = Assert.Throws<InferenceException>(() =>
            ctx.Rerank("what color is an apple?", new[] { "a red apple", "a blue car" }));
        Assert.Contains("rerank/pooling", ex.Message);
    }

    [Fact]
    public void Rerank_Live_Scores_Ordered_And_Deterministic()
    {
        if (!Has(RerankModelPath)) return;
        using var m = LoadModel(RerankModelPath!);
        using var ctx = MakeCtx(m, 1024, pooling: PoolingType.Rank);
        var query = "What is the capital of France?";
        var docs = new[]
        {
            "The capital of France is Paris.",
            "Photosynthesis converts sunlight into chemical energy.",
            "The Eiffel Tower is located in Paris, France.",
        };
        var s1 = ctx.Rerank(query, docs);
        var s2 = ctx.Rerank(query, docs);
        Assert.Equal(3, s1.Length);
        Assert.Equal(s1, s2);                       // determinism: bit-stable replay
        Assert.All(s1, v => Assert.True(float.IsFinite(v)));
        Assert.True(s1[0] > s1[1]);                 // relevant doc outranks distractor
        Assert.True(s1[2] > s1[1]);
    }

    // ── audio input: capability gates + live decode ──

    [Fact]
    public void Audio_Vision_Only_Mmproj_Refuses_Typed()
    {
        if (!Has(HybridModelPath) || !Has(MmprojPath)) return;
        using var m = LoadModel(HybridModelPath!);
        using var vision = m.LoadVisionEncoder(MmprojPath!);
        using var ctx = MakeCtx(m, 1024);
        using var exec = ctx.CreateExecutor();
        Assert.False(m.SupportsAudioInput);          // vision-only projector: no audio adapter
        Assert.Equal(-1, m.AudioSampleRate);
        var pcm = new float[16000];
        Assert.Throws<InferenceException>(() => exec.PromptWithAudio("Transcribe: <__media__>", m, new AudioChunk(pcm)));
        Assert.Equal(0, exec.TokenCount);           // nothing committed
    }

    [Fact]
    public void Audio_Live_Decodes_Both_Lanes()
    {
        if (!Has(AudioModelPath) || !Has(AudioMmprojPath)) return;
        using var m = LoadModel(AudioModelPath!);
        using var vision = m.LoadVisionEncoder(AudioMmprojPath!);
        Assert.True(m.SupportsAudioInput);
        Assert.True(m.AudioSampleRate > 0);
        // 1 s of 440 Hz sine at the native rate — decode mechanics gate
        var rate = m.AudioSampleRate;
        var pcm = new float[rate];
        for (int i = 0; i < pcm.Length; i++)
            pcm[i] = 0.25f * MathF.Sin(2 * MathF.PI * 440f * i / pcm.Length);

        using var ctx = MakeCtx(m, 4096);
        using var exec = ctx.CreateExecutor();
        exec.PromptWithAudio("Transcribe this audio: <__media__>", m, new AudioChunk(pcm));
        Assert.True(exec.TokenCount > 3);
        var st = exec.GetStats();
        Assert.Equal((long)exec.TokenCount, st.Tokens);
        Assert.Equal(st.Tokens, st.KvPositions);
        Assert.True(st.DecodeSteps >= 1);
        Assert.Equal(0L, st.SpecDrafted);
        // greedy continuation works after the audio rows
        var tok = exec.Sample(new SamplingConfig { Temperature = 0f, TopK = 1 });
        Assert.NotEqual(0, tok);
    }

    // ── TTS: live generation, determinism, single-shot contract ──

    [Fact]
    public void Tts_Live_Deterministic_Fresh_Session()
    {
        if (!Has(TtsModelPath) || !Has(TtsMmprojPath)) return;
        using var m = LoadModel(TtsModelPath!);
        using var s1 = (NativeTtsSession)m.CreateTtsSession(TtsMmprojPath!, 4096);
        var req = new TtsRequest { Seed = 42, MaxFrames = 64 };
        var r1 = s1.Generate(req, "Hello world.");
        Assert.True(r1.PcmMono.Length > 0);
        Assert.Equal(24000, r1.SampleRate);
        Assert.All(r1.PcmMono.Take(1000), v => Assert.True(float.IsFinite(v)));

        // single-shot: second generate on the same session refuses typed
        Assert.Throws<InferenceException>(() => s1.Generate(req, "Hello again."));

        // fresh session: identical (text, seed) → byte-identical PCM
        using var s2 = (NativeTtsSession)m.CreateTtsSession(TtsMmprojPath!, 4096);
        var r2 = s2.Generate(req, "Hello world.");
        Assert.Equal(r1.PcmMono.Length, r2.PcmMono.Length);
        Assert.Equal(r1.PcmMono, r2.PcmMono);
    }

    [Fact]
    public void Tts_Missing_Mmproj_Refuses_Typed()
    {
        if (!Has(TtsModelPath)) return;
        using var m = LoadModel(TtsModelPath!);
        Assert.Throws<InferenceException>(() => m.CreateTtsSession("/nonexistent/nope.gguf", 4096));
        // empty text: typed arg error on a valid session
        if (!Has(TtsMmprojPath)) return;
        using var s = (NativeTtsSession)m.CreateTtsSession(TtsMmprojPath!, 4096);
        Assert.Throws<ArgumentException>(() => s.Generate(new TtsRequest { Seed = 7 }, ""));
    }

    // ── LoRA: typed gates + live attach/detach KV-neutrality ──

    [Fact]
    public void Lora_Missing_File_Refuses_Typed()
    {
        if (!HasModel()) return;
        using var m = LoadModel(ModelPath!);
        Assert.Throws<InferenceException>(() => m.CreateLoraAdapter("/nonexistent/nope.gguf"));
    }

    [Fact]
    public void Lora_Live_Attach_Detach_Restores_Baseline()
    {
        if (!HasModel() || !Has(LoraAdapterPath)) return;
        using var m = LoadModel(ModelPath!);
        using var adapter = (NativeLoraAdapter)m.CreateLoraAdapter(LoraAdapterPath!);
        using var ctx = MakeCtx(m, 2048, seqMax: 2);   // seq_max 2: state-save backup seq
        using var exec = (NativeStandardExecutor)ctx.CreateExecutor();
        exec.Prompt("The quick brown fox");
        Assert.Equal(InferResult.Ok, exec.Infer());
        using var anchor = exec.SaveState();
        var nAnchor = anchor.TokenCount;
        var sp = new SamplingConfig { Temperature = 0f, TopK = 1 };

        var tBase = exec.Sample(sp);                  // greedy from post-prompt logits

        // baseline continuation arm (decode tBase at the anchor position)
        exec.RestoreState(anchor);
        exec.PromptTokens(new[] { tBase });
        Assert.Equal(InferResult.Ok, exec.Infer());
        var tPre = exec.Sample(sp);

        // attach @1.0 — weights-surface: KV untouched, no rewind required
        exec.RestoreState(anchor);
        ctx.SetAdapters(new (ILoraAdapter, float)[] { (adapter, 1.0f) });
        exec.PromptTokens(new[] { tBase });
        Assert.Equal(InferResult.Ok, exec.Infer());
        var tLora = exec.Sample(sp);

        // detach — same decode → baseline logits RETURN
        exec.RestoreState(anchor);
        ctx.SetAdapters(Array.Empty<(ILoraAdapter, float)>());
        exec.PromptTokens(new[] { tBase });
        Assert.Equal(InferResult.Ok, exec.Infer());
        var tOff = exec.Sample(sp);

        Assert.Equal(tPre, tOff);                    // detach restores baseline (same pos, same KV)
        Assert.NotEqual(0, tLora);
    }

    // ── control vectors: typed gates + live apply/detach ──

    [Fact]
    public void ControlVector_Missing_File_Refuses_Typed()
    {
        if (!HasModel()) return;
        using var m = LoadModel(ModelPath!);
        Assert.Throws<InferenceException>(() => m.CreateControlVector("/nonexistent/nope.gguf"));
    }

    [Fact]
    public void ControlVector_Live_Apply_Detach_Restores_Baseline()
    {
        // PVX_CVEC_TEST_FILE: synthetic direction.GGUF from the native suite
        // (direction.1..28 F32, ne0 = 3072 — matches the Llama-3.2-3B fixture).
        if (!HasModel() || !Has(CvecTestFile)) return;
        using var m = LoadModel(ModelPath!);
        using var cv = (NativeControlVector)m.CreateControlVector(CvecTestFile!);
        using var ctx = MakeCtx(m, 2048, seqMax: 2);
        using var exec = (NativeStandardExecutor)ctx.CreateExecutor();
        exec.Prompt("The quick brown fox");
        Assert.Equal(InferResult.Ok, exec.Infer());

        // invalid layer range: typed (explicit out-of-range end, never clamped)
        Assert.Throws<InferenceException>(() => ctx.SetControlVector(cv, 0, 999));

        using var anchor = exec.SaveState();
        var sp = new SamplingConfig { Temperature = 0f, TopK = 1 };
        var tBase = exec.Sample(sp);
        exec.RestoreState(anchor);
        exec.PromptTokens(new[] { tBase });
        Assert.Equal(InferResult.Ok, exec.Infer());
        var tPre = exec.Sample(sp);

        exec.RestoreState(anchor);
        ctx.SetControlVector(cv, 0, 0);               // full range apply
        exec.PromptTokens(new[] { tBase });
        Assert.Equal(InferResult.Ok, exec.Infer());
        var tCv = exec.Sample(sp);

        exec.RestoreState(anchor);
        ctx.SetControlVector(null);                  // detach
        exec.PromptTokens(new[] { tBase });
        Assert.Equal(InferResult.Ok, exec.Infer());
        var tOff = exec.Sample(sp);

        Assert.Equal(tPre, tOff);
        Assert.NotEqual(0, tCv);
    }

    // ── stats export ──

    [Fact]
    public void Stats_Executor_Counters_Consistent()
    {
        if (!HasModel()) return;
        using var m = LoadModel(ModelPath!);
        using var ctx = MakeCtx(m, 1024);
        using var exec = ctx.CreateExecutor();
        exec.Prompt("Hello world");
        Assert.Equal(InferResult.Ok, exec.Infer());
        var s = exec.GetStats();
        Assert.Equal((long)exec.TokenCount, s.Tokens);
        Assert.Equal(s.Tokens, s.KvPositions);
        Assert.Equal(s.Tokens, s.PromptTokens);
        Assert.True(s.DecodeSteps >= 1);
        Assert.Equal(0L, s.SpecDrafted);
        Assert.Equal(0L, s.SpecAccepted);
    }
}