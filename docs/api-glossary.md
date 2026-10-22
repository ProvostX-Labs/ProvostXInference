# ProvostXInference — API Glossary

Every public type a consumer can touch, one purpose paragraph each, with the members
that matter. Types are listed by layer; the consumer-facing set is deliberately small.

## Engines layer (the consumer surface)

### `InferenceEngineBuilder` (static)
Entry point for every consumer. `InferenceEngineBuilder.Gguf(path)` / `.Decision(path)` /
`.Encoder(path)` starts a typed request builder; `With*` methods add parameters; `Build()`
returns a validated `EngineRequest`. Fail-closed: invalid combinations throw
`InvalidEngineRequestException` before any native work.

### `InferenceEngineFactory`
Constructs `IInferenceEngine` from a validated request. The default instance needs no
configuration (parameterless).

### `IInferenceEngine` (interface)
One loaded-or-unloaded engine object. `Initialize()` loads it (failure leaves it
unloaded and disposable); `Gguf` / `Decision` / `Encoder` return typed ops views;
`Dispose()` tears down in the fixed order pool → context → model.

### `IGgufEngineOps` (interface, via `GgufEngineOps`)
The GGUF lane view: chat template, tokenize/detokenize, sampling, `RebuildContext()`,
KV copy/clear, checkpoints, rewind/shift, speculation
(`ValidateSpeculation`/`EnableSpeculation`/`DisableSpeculation` + `ActiveSpeculation`),
vision encoder load, TTS session create, rerank, LoRA/cvec, `Backend` probe,
`GetExecutorStats` / `GetConversationStats` (per-lane `EngineStats`),
`GetEngineTelemetry()` (model-wide `EngineTelemetry`), `CreateExecutor()`,
`Pool` (lazy `IConversationPool`).

### `IDecisionEngineOps` / `IEncoderEngineOps`
Decision lane (ONNX judge session) and encoder lane (vision embedding route). Wrong-kind
access throws `EngineKindMismatchException`.

### `EngineLogging` (static facade)
Process-wide native DEBUG stream: `SetLogFile(path)` (native file sink),
`Enable(handler)` (managed `(level, tag, message)` callback), `SetLevel(0..3, 99)`.
Distinct from the per-engine audit trail (see `ModelConfig.AuditPath`).

### `EngineMaintenance` (static facade)
File-level engine-bound maintenance without a loaded engine: GGUF EOG/special-token
metadata repair.

### `StallWatchdogOps` (static facade)
Sole construction seam for the stall-safe prefill helpers
(`IPrefillStallDetector`, `IStallRecoveryLadder`, `IChunkWallClock`).

## Abstractions layer (implemented by the native wrappers)

### `IStandardExecutor`
Single-session generation: `Prompt(text)`, `PromptTokens(...)` (feed-back during
generation), `Infer()` → `InferResult`, `Sample(SamplingConfig)` → token,
`GenerateSpeculative(...)`, `SaveState()/RestoreState()`, `GetStats()`,
`EnableSpeculation(SpeculativeConfig)`, `PromptWithAudio(...)`.

### `IConversation` / `IConversationPool`
Pooled batched lane. Pool: `Lease()` / return-by-dispose, `AvailableCount`. Conversation:
prompt/sample/grammar (`SampleWithGrammar`), `SaveState()/RestoreState`,
`ExportCheckpoint()/ImportCheckpoint()` (byte-exact blobs), `Rewind`, `Reset`,
`GetStats()`, `SpeculationStats()`.

### `IInferenceContext`
Per-model decode context: `Tokenize/Detokenize/TokenToPiece/IsEos`, `CreatePool()`,
`CreateExecutor()`, KV ops (`KvCopy`, `KvClearSeq`), `ContextSize`, `SeqMax`.

### `IInferenceModel`
Model handle surface: `CreateContext(ContextConfig)`, `GetEmbeddings(ctx, text)`,
`EmbeddingDimension`, `LoadVisionEncoder(mmprojPath)`, `ApplyChatTemplate(...)`,
`CreateGrammar(...)` / `CreateGrammarFromJsonSchema(...)`,
`SpeculativeCapabilities`, `CreateLoraAdapter(path)`, `SupportsAudioInput`,
`AudioSampleRate`, `PartialRewindSupported`.

### `IVisionEncoder`
Vision pipeline handle: `Marker` (prompt splice token) — used with
`PromptWithImages(prompt, encoder, imageBytes)`.

### `ITtsSession`
Single-shot speech generation: `Generate(text, params)` → PCM F32 samples + rate;
second generate refuses typed (`tts/session-consumed`); fresh sessions replay
byte-identically per (text, seed).

### `IGrammar`
Grammar handle (GBNF or schema-derived; `Kind` reports which). Owned by the sampled
lane; dispose after the executor/conversation.

## Models layer (config records and snapshots)

| Record | Purpose |
|---|---|
| `ModelConfig` | Model load knobs: `Path`, `GpuLayers` (-1 all / 0 CPU), `Threads`, `FlashAttention`, `KvCacheType`, context-pointer fields (wave-5: kv overrides, lazy mode, tri-states, `LoadMtp`), **`AuditPath`/`AuditIncludePrompts` (ABI v9 audit)** |
| `ContextConfig` | Context knobs: `ContextSize`, `BatchSize`, `SeqMax`, `PoolingType`, recurrent-state ring (`NRsSeq`), wave-5 extensions |
| `SamplingConfig` | Full pipeline: temperature/top-k/top-p/min-p, penalties, dry/xtc/top-n-sigma/mirostat, `MaxTokens`, grammar pointer |
| `SpeculativeConfig` | `Mode` (draft-simple, n-gram family, EAGLE3, draft-MTP), draft/companion paths, n-gram params |
| `StallWatchdogConfig` | Watchdog knobs: enabled, chunk size, gap multiplier, ladder budget |
| `EngineStats` | Per-lane snapshot: `Tokens`, `KvPositions`, `PromptTokens`, `DecodeSteps`, `SpecDrafted`, `SpecAccepted` |
| `EngineTelemetry` | Model-wide snapshot (ABI v9): `PromptTokens`, `DecodeSteps`, `CommittedTokens`, `SpecDrafted`, `SpecAccepted`, `Errors`, `ContextsAlive` |
| `BackendInfo` | Probe truth: backend type/name, `EffectiveGpuLayers`/requested, `GpuLayersReduced`, `ProbePassed` |

## Exceptions layer

| Type | Meaning |
|---|---|
| `InferenceException` | every native-side failure; `ErrorCode` + message + named boundary |
| `InvalidEngineRequestException` | builder request validation failure |
| `EngineKindMismatchException` | ops-view/kind mismatch |
| `PrefillStallUnrecoveredException` | watchdog ladder exhausted (carries chunk index, walls, steps) |

## Enums & small types

`InferResult` (Ok/NoWork/DecodeFailed), `KvType` (F16/Q8_0/Q4_0/Q4_1),
`PoolingType` (None/Mean/Cls/Last/Rank), `SpeculativeType`/`SpeculativeMode`
(draft/ngram family/mtp/eagle3/dflash), `GrammarKind` (Gbnf/JsonSchema),
`AudioChunk` (PCM F32 + rate).
