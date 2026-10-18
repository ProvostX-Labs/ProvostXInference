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
                 GgufModelLoader (IGgufModelLoader)
   │
Decision/         DecisionSession + Factory, JudgeTokenizer, JudgeSequenceEncoder,
                 JudgeNumericsGate, ExecutionProviderChain (ONNX = implementation detail)
   │
Abstractions/     IInferenceModel, IInferenceContext, IConversationPool, IConversation,
                 IStandardExecutor, IInferenceBackend, IInferenceState, IVisionEncoder,
                 IInferenceModel, IPrefillStallDetector, IChunkWallClock, ISpeculationSuspender
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

## Speculative decoding surface (ABI v6)
`SpeculativeConfig` carries the mode (Off / Mtp / Draft / NgramSimple / NgramMapK /
NgramMod / NgramCache), the draft-model knobs, and the n-gram knobs (`NgramConfig`:
lookup sizes, map-k min hits, mod window, optional cache file — pinned with the same
HGlobal rule as the draft path at the native boundary). The engine self-report
(`SpeculativeCapabilities`) is a types list (`draft_simple`, `mtp`, `ngram-simple/-map-k/
-mod/-cache`) plus `SpecVision` — spec sessions draft after multimodal inputs through the
embd-aware mirror. Fail-closed everywhere: missing/unloadable draft, bad n-gram params,
and corrupt cache files are typed errors naming the boundary; never silent fallbacks.
