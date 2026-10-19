# ProvostXInference C# — Architecture (as-is)

Object-oriented wrapper layer for the `pvxinference` C API (`libpvxinference.dylib` / `.so` /
`.dll`). The `Engines` facade is the sole caller-facing surface; everything below it is
internal composition. Replaces LLamaSharp as the .NET → llama.cpp bridge.

## Principles
- Strict OOP: encapsulation, interfaces, constructor injection, one type per file.
- No static mutable state. Sanctioned statics only: `Interop.PvxNative` (pure P/Invoke
  declarations), factory methods on immutable types (`NativeInferenceModel.Load`), and the
  process-wide `PvxDispatchRegistry` (DllImport-resolver registration is once-per-assembly by
  design; internal `ResetForTests` seam keeps it testable).
- No public mutable state: wrappers expose read-only properties and operations.
- Folder structure mirrors namespaces; all native dependencies injectable → testable by design.

## Layering
```
Caller (ProvostXLLM) — touches ONLY this:
Engines/          IInferenceEngine, InferenceEngine, InferenceEngineBuilder,
                 InferenceEngineFactory, EngineRequest records + Builders,
                 Gguf/Decision/EncoderEngineOps (+ I*EngineOps), EngineKind,
                 GgufModelLoader (IGgufModelLoader), plus the black-box helper
                 facades StallWatchdogOps / EngineMaintenance / EngineLogging
                 (pure delegation; the only construction seam for the stall
                 watchdog family and the only Logging-surface entry point)
   │
Decision/         DecisionSession + Factory, JudgeTokenizer, JudgeSequenceEncoder,
                 JudgeNumericsGate, ExecutionProviderChain (ONNX = implementation
                 detail). DecisionSessionFactory constructs parameterless — the
                 logger is a settable Logger property so a facade-only caller's
                 compiled metadata never drags the Logging namespace in
   │
Abstractions/     IInferenceModel, IInferenceContext, IConversationPool, IConversation,
                 IStandardExecutor, IInferenceBackend, IInferenceState, IVisionEncoder,
                 IInferenceModel, IPrefillStallDetector, IChunkWallClock,
                 ISpeculationSuspender, IStallRecoveryLadder
   │
Implementation/   NativeInferenceModel, NativeInferenceContext, NativeConversationPool,
                 NativeConversation, NativeStandardExecutor, NativeInferenceBackend,
                 NativeInferenceState, NativeVisionEncoder, NativeGrammar,
                 StandardEngineActivator, PvxDefaultEngineSelector,
                 ChunkedStallSafePrefillRunner, RollingPrefillStallDetector,
                 StallRecoveryLadder, ExecutorSpeculationSuspender, SystemChunkWallClock
   │
Models/           ModelConfig, ContextConfig, SamplingConfig, SpeculativeConfig,
                 NgramConfig, SpeculativeMode/Type, SpeculativeCapabilities,
                 StallWatchdogConfig, BackendInfo, PvxEngineSelector/Spec/Validation/Error,
                 PvxEngineFileResolver, PvxEngineSelection
   │
SafeHandles/      Model, Context, Pool, Conversation, Executor, State, Grammar, Mmproj
   │
Interop/          PvxNative, PvxResult, blittable param structs, PvxAbiDelegates,
                 per-handle dispatch (IPvxEntryPoints, PvxHandleEntryPoints, PvxDispatchRouter,
                 PvxDispatchRegistry), PvxEngineHandle, NativeModuleLoader, OS/file probes
```

## Engine facade (the caller-facing contract)
- **Requests:** abstract `EngineRequest` (required `ModelPath`, sealed `Validate()`) + sealed
  records `GgufEngineRequest` (mmproj/draft sidecars, GPU layers, threads, flash-attn, KV
  type, reasoning markers, context sizing), `DecisionEngineRequest` (ONNX path, tokenizer
  JSON, judge spec JSON, EP chain, numerics gate), `EncoderEngineRequest` (GGUF + required
  mmproj). `EngineKind` enum { Gguf, Decision, Encoder }. Fail-closed:
  `InvalidEngineRequestException`.
- **Builder:** `InferenceEngineBuilder` static `Gguf(path)`/`Decision(path)`/`Encoder(path)`
  → fluent builders → `Build()` = validated request; `Create(factory)` = ready engine.
- **Factory:** `IInferenceEngineFactory`/`InferenceEngineFactory` — injected `IGgufModelLoader`
  + `IDecisionSessionFactory`; `Create(EngineRequest)` → UNLOADED engine, exhaustive kind
  routing.
- **Facade:** `IInferenceEngine`/`InferenceEngine` — `Kind`, `Request`, `ModelPath`,
  `IsLoaded`, `ActiveSpecialist`, `Initialize()`, `Dispose()`, typed views `Gguf`/`Decision`/
  `Encoder`; wrong-kind view access throws `EngineKindMismatchException`.
- **Views wrap, never duplicate:** `GgufEngineOps` (chat template, tokenize, sampling knobs,
  live context rebuild, conversation convenience, KV copy/clear, checkpoint
  save/restore/export/import, speculation validation + `ActiveSpeculation` mirror, live
  `Backend` probe), `DecisionEngineOps` (judge session + truthful provider/tokenizer/spec),
  `EncoderEngineOps` (vision encoder + image embedding route).

## Key decisions
1. **SafeHandle for all native resources** — each ReleaseRelease calls the matching
   `pvx_free_*`; guaranteed cleanup even on exceptions/thread aborts. Ownership: contexts
   belong to the model, pool/executor to the context, conversations to the pool; disposal
   order enforced outermost-first. `PvxEngineHandle.Dispose` is a bookkeeping no-op —
   module unmap deliberately deferred (NativeLibrary.Free with live handles is unsafe).
2. **P/Invoke + per-handle dispatch** — steady-state calls route through the OWNING
   engine's `IPvxEntryPoints` function table (delegates resolved per loaded image), not
   global DllImports. The registry router covers only the activation window + process-wide
   safety-net APIs.
3. **UTF-8 marshalling** — all strings `LPUTF8Str`/`PtrToStringUTF8`/`AllocUTF8`;
   anti-prompt delegate marshalling rejects `LPArray`+`LPUTF8Str` and marshals elements
   manually, freed in `finally`.
4. **Error handling** — `PVX_ERR_*` → `InferenceException` with native detail
   (`pvx_last_error` context-scoped, `pvx_model_last_error` model-scoped — ABI v5; the
   caller passes the OWNING handle so reasons always surface). `InferResult` is a legitimate
   outcome, not an exception.
5. **Engine selection fail-closed** — `ModelConfig.Engine` validated against the per-OS
   matrix (`metal` macOS only; `cuda`/`vulkan` Windows/Linux only); invalid pair / missing
   specialist file → typed `DllNotFoundException`/`InferenceException` naming the file and
   engine, NO fallback to another backend. ABI mismatch between image and binding refuses
   the load.
6. **Thread safety** — the C++ core is internally mutex-protected; no C# locking here.
   ProvostXLLM adds `SemaphoreSlim` coordination above this layer.
7. **Watchdog layer** — stall-safe chunked prefill is backend-independent pure wall-clock
   over `IConversation` prompt commits; recovery ladder ends in FailLoud
   (`PrefillStallUnrecoveredException`); disabled = single plain commit, zero overhead.

## Native library resolution
Specialist modules load explicitly via `NativeModuleLoader` from
`runtimes/<rid>/native/libpvx-<engine>.*` (resolver probes the assembly directory);
`DllImport("pvxinference")` remains the fallback resolution path for the activation window.

## Tests
`csharp/Tests/` — one test class per type; all native seams faked (no dylib needed except
env-guarded live smoke/journey tests): engine request/builder/factory/ops/facade suites,
selector + resolver + router + registry fail-closed matrices, error-masking sweep, spec
decode, stall watchdog (detector/ladder/runner over synthetic timing + fakes), decision
domain (tokenizer, sequence encoder, session factory on real tiny ONNX fixtures), dispatch
tests, model-gated live journey tests. Full suite green (194/194 at last run), build
0 errors/0 warnings (Debug + Release), LDC artifacts regenerated + enforcement PASS with
every public-surface change.

## Speculative decoding surface (ABI v7)
`SpeculativeConfig` carries the mode (Off / Mtp / Draft / NgramSimple / NgramMapK /
NgramMod / NgramCache, plus ABI v7 companion modes Eagle3 / Dflash), the draft-model
knobs, the n-gram knobs (`NgramConfig`: lookup sizes, map-k min hits, mod window,
optional cache file — pinned with the same HGlobal rule as the draft path at the native
boundary), and the ABI v7 companion paths (`EagleCompanionPath` / `DflashCompanionPath`).
The engine self-report (`SpeculativeCapabilities`) is a types list (`draft_simple`, `mtp`,
ngram-simple/-map-k/-mod/-cache, plus the ABI v7 `eagle3`/`dflash` companion bits —
reported from companion metadata, typed refusals when absent) plus `SpecVision` —
spec sessions draft after multimodal inputs through the embd-aware mirror. ABI v7 adds
the batched/pool lane: `IConversation` grows EnableSpeculation/DisableSpeculation/
ActiveSpeculation/GenerateSpeculative/SpeculationStats (one spec session per pooled
conversation, `NativeConversationSpecSession` one-type-per-file; pool return drops the
managed mirror with the native teardown), and the ops facade gates the lane through the
model self-report (`EnableConversationSpeculation` validates first). Acceptance telemetry
(`SpeculationStats`: drafted/accepted counters) never hides rejections.

## JSON-schema → GBNF surface (ABI v7)
`IInferenceModel.CreateGrammarFromJsonSchema(schema, rootName?)` + the
`GgufEngineOps.CreateGrammarFromJsonSchema` facade convenience expose the native
converter (pure string→string, KV-neutral). `GrammarKind` (Gbnf / JsonSchema) labels
provenance on the grammar record. Fail-closed: invalid JSON → InferenceException naming
the parse position; unsupported schema keyword → InferenceException naming the keyword;
the converter's out buffers are freed on every path. Golden pin: the decision-envelope
schema's GBNF is byte-exact against `tests/fixtures/decision-envelope.gbnf` (the same
fixture the native suite pins).

Fail-closed everywhere: missing/unloadable draft, bad n-gram params, missing/unloadable
companion GGUFs, and corrupt cache files are typed errors naming the boundary; never
silent fallbacks.

## Wave 3 media / rerank / weights-surface (ABI v8)

ABI v8 (wave 3) is additive across the binding: `PvxNative.ExpectedAbiVersion = 8`
(lockstep with the native macro), `PvxPoolingType.Rank` / `PoolingType.Rank` (the
rerank lane — RANK pooling attaches the model's classification head to the graph;
`NativeInferenceContext.Rerank(query, documents)` scores pairs single-shot on a
fresh seq and throws typed naming `rerank/pooling` on every other pooling), audio
input (`NativeStandardExecutor.PromptWithAudio` / `NativeConversation.PromptWithAudio`
with `AudioChunk` (raw PCM F32 mono) + `IInferenceModel.SupportsAudioInput` /
`AudioSampleRate` probes; `EncoderEngineOps` stays images-only — audio is GGUF-lane
input), TTS (`NativeTtsSession` + `TtsRequest`/`TtsResult`; the session is
single-shot — `tts/session-consumed` — and fresh sessions replay byte-identically
per (text, seed)), LoRA (`NativeLoraAdapter` + `IInferenceContext.SetAdapters`,
weights-surface: KV-neutral live attach/detach), control vectors
(`NativeControlVector` + `SetControlVector`, same pattern-port discipline), and the
stats export (`EngineStats` snapshot via `IStandardExecutor.GetStats` /
`IConversation.GetStats`; spec telemetry mirrors the spec stats). The facade view:
`GgufEngineOps` grows `Rerank`, `PromptWithAudio` (both lanes), `LoadLora` /
`WithLora` (scoped disposable) / `ClearLora`, `LoadControlVector` /
`WithControlVector` / `ClearControlVector`, `CreateTtsSession`,
`SupportsAudioInput`/`AudioSampleRate`, and `GetExecutorStats` /
`GetConversationStats`. Fail-closed: every wave-3 surface throws typed
`InferenceException`s naming the boundary (`rerank/pooling`, `media/audio-adapter`,
`tts/gen-audio-capability`, `tts/session-consumed`, `adapter/*`, `cvector/*`).
