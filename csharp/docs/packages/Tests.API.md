# Tests.API.md

Types: 27  |  LOC: 3652  |  ~1474 tokens

---

### Class: DecisionEngineOpsTests
> <see cref="DecisionEngineOps"/>: wrapping seams over a REAL
Cross-package deps: ProvostXInference.Decision.Engine, ProvostXInference.Engines, ProvostXInference.Tests.Decision, Xunit

### Class: DecisionEngineRequestTests
> <see cref="DecisionEngineRequest"/>: required tokenizer/spec, EP chain, gate.
Cross-package deps: ProvostXInference.Exceptions, ProvostXInference.Models, ProvostXInference.Engines, Xunit

### Class: DecisionSessionFactoryTests
> Real ONNX judge session factory tests (wave 2): CPU chains initialize the tiny
Cross-package deps: Xunit, ProvostXInference.Decision.Engine

### Class: DecisionTestAssets
> Shared decision-engine test assets: the tiny synthetic ONNX model
Cross-package deps: ProvostXInference.Decision.Engine

### Class: EncoderEngineOpsTests
> <see cref="EncoderEngineOps"/>: vision-encoder wiring over fakes.
Cross-package deps: ProvostXInference.Engines, ProvostXInference.Models, Xunit

### Class: EncoderEngineRequestTests
> <see cref="EncoderEngineRequest"/>: mmproj is REQUIRED (encoder without one is meaningless).
Cross-package deps: ProvostXInference.Exceptions, ProvostXInference.Models, ProvostXInference.Engines, Xunit

### Class: EngineFacadeSmokeTests
> LIVE facade smoke: the full contract chain — builder → validated request →
Cross-package deps: ProvostXInference.Abstractions, ProvostXInference.Engines, ProvostXInference.Models, Xunit

### Class: EngineRequestTests
> Base <see cref="EngineRequest"/> contract: required ModelPath,
Cross-package deps: ProvostXInference.Exceptions, ProvostXInference.Models, ProvostXInference.Engines, Xunit

### Class: ErrorMaskingSweepTests
> BL-03 error-masking sweep: prove the missing-dylib / missing-export error
Cross-package deps: ProvostXInference.Interop, ProvostXInference.Models, Xunit

### Class: GgufEngineOpsTests
> <see cref="GgufEngineOps"/>: delegation wiring over the fake
Cross-package deps: ProvostXInference.Engines, ProvostXInference.Models, Xunit

### Class: GgufEngineRequestTests
> <see cref="GgufEngineRequest"/>: knob validation + native config mapping.
Cross-package deps: ProvostXInference.Exceptions, ProvostXInference.Models, ProvostXInference.Engines, Xunit

### Class: InferenceEngineBuilderTests
> <see cref="InferenceEngineBuilder"/>: entry points + Build/Create plumbing.
Cross-package deps: ProvostXInference.Decision.Abstractions, ProvostXInference.Decision.Engine, ProvostXInference.Engines, Xunit

### Class: InferenceEngineFactoryTests
> <see cref="InferenceEngineFactory"/>: validation boundary + composition root.
Cross-package deps: ProvostXInference.Decision.Abstractions, ProvostXInference.Decision.Engine, ProvostXInference.Engines, Xunit

### Class: InferenceEngineTests
> <see cref="InferenceEngine"/> facade: lifecycle, kind routing, fail-closed
Cross-package deps: ProvostXInference.Decision.Abstractions, ProvostXInference.Decision.Engine, ProvostXInference.Engines, ProvostXInference.Exceptions, ProvostXInference.Tests.Decision, Xunit

### Class: JsonSchemaGrammarTests
> JSON-schema → GBNF converter surface (ABI v7, feature 11): the native
Cross-package deps: ProvostXInference.Abstractions, ProvostXInference.Exceptions, ProvostXInference.Implementation, ProvostXInference.Interop, ProvostXInference.Models, ProvostXInference.Engines, ProvostXInference.Tests.Engines, Xunit

### Class: JudgeSequenceEncoderTests
> Judge sequence encoder tests (wave 2): record → serialized training state →
Cross-package deps: Xunit, ProvostXInference.Decision.Engine

### Class: JudgeTokenizerTests
> Judge tokenizer tests (wave 2): the committed vector fixture
Cross-package deps: Xunit, ProvostXInference.Decision.Engine

### Class: ModelEngineSelectionTests
> Model-config engine field (BL-02 stage 1): ModelConfig selects WHICH
Cross-package deps: ProvostXInference.Implementation, ProvostXInference.Interop, ProvostXInference.Models, Xunit

### Class: PvxDispatchRegistryTests
> Dispatch registry: activation swaps the logical-name route (consulted only
Implements: IDisposable
Cross-package deps: ProvostXInference.Interop, Xunit

### Class: PvxDispatchRouterTests
> Per-handle dispatch (BL-02 stage 1): two routers on DISTINCT fake images
Cross-package deps: ProvostXInference.Interop, Xunit

### Class: PvxEngineFileResolverTests
> File-name resolver (names live in ONE place): per-OS specialist
Cross-package deps: ProvostXInference.Interop, ProvostXInference.Models, Xunit

### Class: PvxEngineSelectorTests
> Selector matrix (BL-02 stage 1): every (engine, OS) pair of the 5-product
Cross-package deps: ProvostXInference.Interop, ProvostXInference.Models, Xunit

### Class: SmokeTests
> Comprehensive test suite for ProvostXInference C# bindings.
Cross-package deps: ProvostXInference.Abstractions, ProvostXInference.Exceptions, ProvostXInference.Implementation, ProvostXInference.Models, Xunit

### Class: SpecDecodeTests
> Speculative decoding (INF BL-05) — mirrors the native spec tests:
Cross-package deps: ProvostXInference.Abstractions, ProvostXInference.Exceptions, ProvostXInference.Implementation, ProvostXInference.Interop, ProvostXInference.Models, ProvostXInference.Engines, ProvostXInference.Tests.Engines, Xunit

### Class: StallSafePrefillLiveJourneyTests
> Integration test for the stall-safe prefill watchdog on the live
Cross-package deps: ProvostXInference.Abstractions, ProvostXInference.Implementation, ProvostXInference.Models, Xunit

### Class: StallWatchdogTests
> Unit tests for the stall-safe prefill watchdog (INF BL-33):
Cross-package deps: ProvostXInference.Abstractions, ProvostXInference.Exceptions, ProvostXInference.Implementation, ProvostXInference.Models, Xunit

### Class: Wave3FeatureTests
> ABI v8 (wave 3) surface tests: audio input, TTS, rerank, LoRA attach/
Cross-package deps: ProvostXInference.Abstractions, Xunit, ProvostXInference.Exceptions, ProvostXInference.Implementation, ProvostXInference.Models
