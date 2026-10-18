# csharp.API.md

Types: 80  |  LOC: 4956  |  ~5714 tokens

---

### Interface: IChunkWallClock
> Wall-clock seam for per-chunk prefill timing (INF BL-33).
Methods:
  - double Measure(Action work)

### Interface: IConversation
> Resets grammar sampler state — call once per generation before a grammar-constrained sample loop.
Implements: IDisposable
Properties:
  - int TokenCount { get; set; }
  - SpeculativeConfig? ActiveSpeculation { get; set; }
Methods:
  - void Prompt(string text)
  - void PromptTokens(int[] tokens)
  - int Sample(SamplingConfig? config = null)
  - int SampleWithGrammar(SamplingConfig? config, IGrammar grammar)
  - void Rewind(int tokenCount)
  - void Reset()
  - void ResetGrammarState()
  - void ShiftLeft(int tokenCount)
  - IConversationState SaveState()
  - void RestoreState(IConversationPool pool, IConversationState state)
  - byte[] ExportCheckpoint()
  - void ImportCheckpoint(byte[] data)
  - void PromptWithImages(string text, IVisionEncoder vision, params byte[][] images)
  - void EnableSpeculation(SpeculativeConfig config)
  - void DisableSpeculation()
  - IReadOnlyList<int> GenerateSpeculative(SamplingConfig? config, int maxTokens, Action<int>? onToken = null)
  - SpeculationStats SpeculationStats()
Cross-package deps: ProvostXInference.Models

### Interface: IConversationPool
Implements: IDisposable
Properties:
  - int AvailableCount { get; set; }
  - uint Size { get; set; }
Methods:
  - IConversation Lease()
  - void Return(IConversation conversation)

### Interface: IConversationState
Implements: IDisposable
Properties:
  - int TokenCount { get; set; }

### Interface: IDecisionEngineOps
> Decision (ONNX judge) lane ops view: batch judging on the ORT session with
Properties:
  - string ActiveProvider { get; set; }
  - JudgeTokenizer Tokenizer { get; set; }
  - JudgeSequenceSpec Spec { get; set; }
Methods:
  - IReadOnlyList<JudgeScore> Judge(IReadOnlyList<DecisionRecord> records)
  - IReadOnlyList<JudgeQuestionAnswerSet> JudgeQuestionSet(IReadOnlyList<JudgeStateQuestions> requests)
Cross-package deps: ProvostXInference.Decision.Engine

### Interface: IDecisionHeaderReader
> Decision-model file validation seam (load state machine step). A REAL
Methods:
  - void Validate(string path)
Cross-package deps: ProvostXInference.Decision.Engine

### Interface: IDecisionSessionFactory
> Creates <see cref="DecisionSession"/>s for a decision (judge) model file:
Methods:
  - DecisionSession Create(string modelPath,
        JudgeTokenizer tokenizer,
        JudgeSequenceSpec spec,
        ExecutionProviderChain chain,
        JudgeNumericsGate? numericsGate = null)
Cross-package deps: ProvostXInference.Decision.Engine

### Interface: IEncoderEngineOps
> Encoder lane ops view: image embedding through the mmproj projector mounted
Properties:
  - IVisionEncoder VisionEncoder { get; set; }
  - IInferenceModel Model { get; set; }
  - string ImageMarker { get; set; }
  - bool NeedsNonCausalAttention { get; set; }
Methods:
  - float[] EmbedText(IInferenceContext context, string text)
Cross-package deps: ProvostXInference.Abstractions

### Interface: IExecutionProviderDefaultSource
> Decision-model file validation seam (load state machine step). A REAL
Methods:
  - ExecutionProviderChain DefaultChain()
Cross-package deps: ProvostXInference.Decision.Engine

### Interface: IGgufEngineOps
> GGUF lane ops view: the full native llama.cpp surface as properties/methods —
Properties:
  - IInferenceModel Model { get; set; }
  - IInferenceContext Context { get; set; }
  - SamplingConfig Sampling { get; set; }
  - ContextConfig ContextSettings { get; set; }
  - BackendInfo Backend { get; set; }
  - IConversationPool Pool { get; set; }
  - bool PartialRewindSupported { get; set; }
  - SpeculativeCapabilities SpeculativeCapabilities { get; set; }
Methods:
  - void RebuildContext()
  - IConversation LeaseConversation()
  - IStandardExecutor CreateExecutor()
  - void KvCopy(int srcSeqId, int dstSeqId)
  - void KvClearSeq(int seqId)
  - void SetCausalAttention(bool causal)
  - int[] Tokenize(string text, bool addBos = false, bool parseSpecial = true)
  - string Detokenize(int[] tokens, bool removeSpecial = false)
  - IConversationState SaveConversationState(IConversation conversation)
  - void RestoreConversationState(IConversationPool pool, IConversationState state)
  - byte[] ExportConversationCheckpoint(IConversation conversation)
  - void ImportConversationCheckpoint(IConversation conversation, byte[] data)
  - IGrammar CreateGrammarFromJsonSchema(string jsonSchema, string? rootName = null)
  - void EnableConversationSpeculation(IConversation conversation, SpeculativeConfig config)
  - void DisableConversationSpeculation(IConversation conversation)
  - SpeculativeConfig? ActiveConversationSpeculation(IConversation conversation)
  - SpeculationStats ConversationSpeculationStats(IConversation conversation)
  - void ValidateSpeculation(SpeculativeConfig config)
Cross-package deps: ProvostXInference.Abstractions, ProvostXInference.Models

### Interface: IGgufModelLoader
> Testability seam over the native GGUF loader: the engine factory depends on
Methods:
  - IInferenceModel Load(ModelConfig config)
  - IVisionEncoder LoadVisionEncoder(IInferenceModel model, string mmprojPath)
Cross-package deps: ProvostXInference.Abstractions, ProvostXInference.Models

### Interface: IGrammar
> Provenance of this grammar's GBNF text: hand-written GBNF
Implements: IDisposable
Properties:
  - GrammarKind Kind { get; set; }

### Interface: IInferenceBackend
Methods:
  - BackendInfo Probe(IInferenceContext context)
  - bool VramAvailable(long neededBytes)
Cross-package deps: ProvostXInference.Models

### Interface: IInferenceContext
Implements: IDisposable
Properties:
  - uint ContextSize { get; set; }
  - uint SeqMax { get; set; }
Methods:
  - IStandardExecutor CreateExecutor()
  - IConversationPool CreatePool()
  - InferResult InferAll()
  - void KvCopy(int srcSeqId, int dstSeqId)
  - void KvClearSeq(int seqId)
  - void SetCausalAttention(bool causal)
  - int[] Tokenize(string text, bool addBos = false, bool parseSpecial = true)
  - string Detokenize(int[] tokens, bool removeSpecial = false)
  - string TokenToPiece(int token)
  - bool IsEos(int token)
Cross-package deps: ProvostXInference.Models, ProvostXInference.Abstractions

### Interface: IInferenceEngine
> The SOLE caller-facing object of Inference's world: one loaded backend (GGUF
Implements: IDisposable
Properties:
  - EngineKind Kind { get; set; }
  - EngineRequest Request { get; set; }
  - string ModelPath { get; set; }
  - string? ActiveSpecialist { get; set; }
  - bool IsLoaded { get; set; }
  - IGgufEngineOps Gguf { get; set; }
  - IDecisionEngineOps Decision { get; set; }
  - IEncoderEngineOps Encoder { get; set; }
Methods:
  - void Initialize()

### Interface: IInferenceEngineFactory
> Resolves a validated <see cref="EngineRequest"/> into an UNLOADED
Methods:
  - IInferenceEngine Create(EngineRequest request)

### Interface: IInferenceLogger
> Minimal logging seam for Inference-side subsystems (decision judge execution,
Methods:
  - void Info(string tag, string message)
  - void Warn(string tag, string message)
  - void Error(string tag, string message)
  - void Debug(string tag, string message)

### Interface: IInferenceModel
> enableThinking: -1 = auto (model/template default), 0 = off, 1 = on.
Implements: IDisposable
Properties:
  - int EmbeddingDimension { get; set; }
  - SpeculativeCapabilities SpeculativeCapabilities { get; set; }
Methods:
  - IInferenceContext CreateContext(ContextConfig config)
  - float[] GetEmbeddings(IInferenceContext context, string text)
  - IVisionEncoder LoadVisionEncoder(string mmprojPath)
  - IGrammar CreateGrammar(string grammarStr, string grammarRoot)
  - IGrammar CreateGrammarFromJsonSchema(string jsonSchema, string? rootName = null)
  - void ValidateSpeculation(SpeculativeConfig config)
Cross-package deps: ProvostXInference.Models

### Interface: IInferenceState
Implements: IDisposable
Properties:
  - int TokenCount { get; set; }

### Interface: IPrefillStallDetector
> Typed stall detector over per-chunk prefill wall timings
Methods:
  - PrefillStallVerdict Observe(int chunkIndex, int chunkTokens, double wallSeconds)
Cross-package deps: ProvostXInference.Models

### Interface: ISpeculationSuspender
> Runtime suspension of the speculative helper (draft/ngram) on the
Methods:
  - void Suspend()

### Interface: IStandardExecutor
> Read-only live state: the speculative session this executor is
Implements: IDisposable
Properties:
  - int TokenCount { get; set; }
  - SpeculativeConfig? ActiveSpeculation { get; set; }
Methods:
  - void Prompt(string text)
  - void PromptTokens(int[] tokens)
  - void PromptWithImages(string text, IInferenceModel model, params byte[][] images)
  - InferResult Infer()
  - int Sample(SamplingConfig? config = null)
  - int SampleWithGrammar(SamplingConfig? config, IGrammar grammar)
  - void Rewind(int tokenCount)
  - void Reset()
  - void ShiftLeft(int tokenCount)
  - IInferenceState SaveState()
  - void RestoreState(IInferenceState state)
  - void EnableSpeculation(SpeculativeConfig config)
  - void DisableSpeculation()
  - IReadOnlyList<int> GenerateSpeculative(SamplingConfig? config, int maxTokens, Action<int>? onToken = null)
Cross-package deps: ProvostXInference.Models

### Interface: IVisionEncoder
Implements: IDisposable
Properties:
  - string Marker { get; set; }
  - bool NeedsNonCausalAttention { get; set; }

### Class: ChunkedStallSafePrefillRunner
> Engine-level stall-safe chunked prefill layer (INF BL-33).
Constructor:
  - ChunkedStallSafePrefillRunner(IConversationPool pool, StallWatchdogConfig config, IChunkWallClock clock, IPrefillStallDetector detector, ISpeculationSuspender? draftSuspender = null)
Cross-package deps: ProvostXInference.Abstractions, ProvostXInference.Exceptions, ProvostXInference.Models

### Class: DecisionAssetCatalog
> Resolves the judge ASSET files (tokenizer.json, judge-spec-*.json) for one
Constructor:
  - DecisionAssetCatalog(string rootDir)

### Class: DecisionEngineOps
> Decision lane ops over an existing <see cref="DecisionSession"/> (loaded by
Implements: IDecisionEngineOps, IDisposable
Constructor:
  - DecisionEngineOps(DecisionSession session)
Cross-package deps: ProvostXInference.Decision.Engine

### Class: DecisionEngineRequestBuilder
> Fluent builder for <see cref="DecisionEngineRequest"/> (ONNX judge lane).
Constructor:
  - DecisionEngineRequestBuilder(string modelPath)
Cross-package deps: ProvostXInference.Decision.Engine

### Class: DecisionHeaderReader
> Wave-2 ONNX header validation: a REAL protobuf scan of the ModelProto wire —
Implements: IDecisionHeaderReader
Cross-package deps: ProvostXInference.Decision.Abstractions, ProvostXInference.Logging

### Class: DecisionRecord
> One judged decision record, validated and normalized (parser → executor boundary).

### Class: DecisionSession
> ONE loaded ONNX judge model ready for batch inference: the ORT session
Implements: IDisposable
Constructor:
  - DecisionSession(JudgeTokenizer tokenizer, JudgeSequenceSpec spec, InferenceSession session, string activeProvider)

### Class: DecisionSessionFactory
> Creates <see cref="DecisionSession"/>s for a judge model file: builds the ORT
Implements: IDecisionSessionFactory
Constructor:
  - DecisionSessionFactory(ExecutionProviderOptionsApplier? optionsApplier = null, IInferenceLogger? logger = null)
Cross-package deps: ProvostXInference.Decision.Abstractions, ProvostXInference.Logging

### Class: DecisionTestSupport
> Test-support surface for the decision engine slice (Inference-side, consumed

### Class: EncoderEngineOps
> Encoder lane ops: wraps the vision encoder + backing model (see interface).
Implements: IEncoderEngineOps, IDisposable
Constructor:
  - EncoderEngineOps(IInferenceModel model, IVisionEncoder vision)
Cross-package deps: ProvostXInference.Abstractions

### Class: EncoderEngineRequestBuilder
> Fluent builder for <see cref="EncoderEngineRequest"/> (vision encoder lane:
Constructor:
  - EncoderEngineRequestBuilder(string modelPath)
Cross-package deps: ProvostXInference.Models

### Class: EngineKindMismatchException
> Fail-closed typed error for accessing an ops view of the wrong kind
Implements: InferenceException
Constructor:
  - EngineKindMismatchException(string message)

### Class: ExecutionProviderChain
> Execution providers for decision (judge) models. The chain is parsed and

### Class: ExecutionProviderOptionsApplier
> One candidate configuration for an EP session init: a label for

### Class: ExecutorSpeculationSuspender
> Runtime draft/ngram suspension backed by the executor's native
Implements: ISpeculationSuspender
Constructor:
  - ExecutorSpeculationSuspender(IStandardExecutor executor)
Cross-package deps: ProvostXInference.Abstractions

### Class: GgufEngineOps
> GGUF lane ops implementation over the wrapper-layer graph. Holds the model,
Implements: IGgufEngineOps, IDisposable
Constructor:
  - GgufEngineOps(IInferenceModel model, ContextConfig contextSettings, SamplingConfig sampling, IInferenceBackend backend)
Cross-package deps: ProvostXInference.Abstractions, ProvostXInference.Models

### Class: GgufEngineRequestBuilder
> Fluent builder for <see cref="GgufEngineRequest"/>. <see cref="Build"/> yields a
Constructor:
  - GgufEngineRequestBuilder(string modelPath)
Cross-package deps: ProvostXInference.Models

### Class: GgufMetadataRepair
> GGUF tokenizer-metadata repair (owner 2026-10-06): some HuggingFace GGUFs omit
Cross-package deps: ProvostXInference.Interop

### Class: GgufModelLoader
> Production <see cref="IGgufModelLoader"/>: thin wrapper over the native
Implements: IGgufModelLoader
Cross-package deps: ProvostXInference.Abstractions, ProvostXInference.Models, ProvostXInference.Implementation

### Class: InferenceEngine
> Facade implementation: created UNLOADED by <see cref="InferenceEngineFactory"/>,
Implements: IInferenceEngine
Cross-package deps: ProvostXInference.Abstractions, ProvostXInference.Decision.Abstractions, ProvostXInference.Decision.Engine, ProvostXInference.Exceptions, ProvostXInference.Models

### Class: InferenceEngineBuilder
> Fluent entry point of the engine-object contract:

### Class: InferenceEngineFactory
> Default factory (composition root of the facade): native GGUF loading via
Implements: IInferenceEngineFactory
Constructor:
  - InferenceEngineFactory(IGgufModelLoader? ggufLoader = null, IDecisionSessionFactory? decisionFactory = null, IExecutionProviderDefaultSource? epDefaultSource = null, IInferenceBackend? backend = null)
Cross-package deps: ProvostXInference.Abstractions, ProvostXInference.Decision.Abstractions, ProvostXInference.Decision.Engine, ProvostXInference.Implementation

### Class: InferenceException
> Thrown when a native ProvostXInference call returns an error code.
Implements: Exception

### Class: InvalidEngineRequestException
> Fail-closed typed error for an invalid engine contract (bad/missing path,
Implements: InferenceException
Constructor:
  - InvalidEngineRequestException(string message)

### Class: JudgeModelNotReadyException
> Typed fail-closed judge error: every judge endpoint maps this to HTTP 503 with
Implements: Exception
Constructor:
  - JudgeModelNotReadyException(string code, string message)

### Class: JudgeSequenceEncoder
> Builds judge model input rows from validated <see cref="DecisionRecord"/>s:
Constructor:
  - JudgeSequenceEncoder(JudgeTokenizer tokenizer, JudgeSequenceSpec spec)

### Class: JudgeTokenizer
> Deterministic BPE tokenizer for the Laya judge models (mmBERT vocabulary,
Constructor:
  - JudgeTokenizer(string tokenizerJsonPath)

### Class: NativeConversation
> Wraps a leased pvx_conversation_t. Does NOT own the handle — the pool does.
Implements: IConversation
Cross-package deps: ProvostXInference.Abstractions, ProvostXInference.Exceptions, ProvostXInference.Interop, ProvostXInference.Models, ProvostXInference.SafeHandles

### Class: NativeConversationPool
> Wraps a native pvx_pool_t. All calls dispatch on the OWNING
Implements: IConversationPool
Cross-package deps: ProvostXInference.Abstractions, ProvostXInference.Exceptions, ProvostXInference.Interop, ProvostXInference.SafeHandles

### Class: NativeConversationSpecSession
> Speculative session bound to ONE pooled conversation (ABI v7,
Cross-package deps: ProvostXInference.Abstractions, ProvostXInference.Exceptions, ProvostXInference.Interop, ProvostXInference.Models

### Class: NativeConversationState
> Wraps a conversation state (pvx_conversation_save). Token count
Implements: IConversationState
Cross-package deps: ProvostXInference.Abstractions, ProvostXInference.Interop

### Class: NativeGrammar
> Wraps a pvx_grammar_t. Created through — and dispatching on — the
Implements: IGrammar
Cross-package deps: ProvostXInference.Abstractions, ProvostXInference.Exceptions, ProvostXInference.Interop, ProvostXInference.Models, ProvostXInference.SafeHandles

### Class: NativeInferenceBackend
> Backend safety layer (pvx_probe_backend + queries). The probe is
Implements: IInferenceBackend
Cross-package deps: ProvostXInference.Abstractions, ProvostXInference.Exceptions, ProvostXInference.Interop, ProvostXInference.Models

### Class: NativeInferenceContext
> Wraps a native pvx_context_t. All calls dispatch on the OWNING
Implements: IInferenceContext
Cross-package deps: ProvostXInference.Abstractions, ProvostXInference.Exceptions, ProvostXInference.Interop, ProvostXInference.Models, ProvostXInference.SafeHandles

### Class: NativeInferenceModel
> Wraps a native pvx_model_t. Factory: Load(). BL-02: EVERY pvx_* call this
Implements: IInferenceModel
Cross-package deps: ProvostXInference.Abstractions, ProvostXInference.Exceptions, ProvostXInference.Interop, ProvostXInference.Models, ProvostXInference.SafeHandles

### Class: NativeInferenceState
> Wraps a pvx_state_t (pvx_state_save). Token count queries dispatch
Implements: IInferenceState
Cross-package deps: ProvostXInference.Abstractions, ProvostXInference.Interop, ProvostXInference.SafeHandles

### Class: NativeLogBridge
> Bridges native C-level logging (pvx_set_log_callback) to a managed callback.
Cross-package deps: ProvostXInference.Interop

### Class: NativeStandardExecutor
> Wraps a pvx_executor_t (standard, non-batched path). All calls
Implements: IStandardExecutor
Cross-package deps: ProvostXInference.Abstractions, ProvostXInference.Exceptions, ProvostXInference.Interop, ProvostXInference.Models, ProvostXInference.SafeHandles

### Class: NativeVisionEncoder
> Wraps an mmproj context for vision/image encoding. All calls
Implements: IVisionEncoder
Cross-package deps: ProvostXInference.Abstractions, ProvostXInference.Interop, ProvostXInference.SafeHandles

### Class: OsExecutionProviderDefaultSource
> Per-OS default EP chains (LLM-10): macOS coreml→cpu, Windows directml→cpu,
Implements: IExecutionProviderDefaultSource
Constructor:
  - OsExecutionProviderDefaultSource(string? osOverride = null)
Cross-package deps: ProvostXInference.Decision.Abstractions, ProvostXInference.Logging

### Class: PrefillStallUnrecoveredException
> Fail-loud terminal outcome of the stall-safe prefill recovery
Implements: Exception
Constructor:
  - PrefillStallUnrecoveredException(int chunkIndex, int chunkTokens, double lastStallWallSeconds, double baselineMedianSeconds, IReadOnlyList<PrefillRecoveryStep> attemptedSteps, string? reason = null)
Cross-package deps: ProvostXInference.Models

### Class: PvxDefaultEngineSelector
> Platform default engine selection — used when the model config leaves the
Cross-package deps: ProvostXInference.Interop, ProvostXInference.Models

### Class: PvxEngineSelector
> Validates the ModelConfig engine field against the 5-product engine matrix:
Cross-package deps: ProvostXInference.Interop

### Class: QuietInferenceLogger
> Minimal logging seam for Inference-side subsystems (decision judge execution,
Implements: IInferenceLogger

### Class: RollingPrefillStallDetector
> Rolling-median stall detector (INF BL-33). A chunk is a stall when
Implements: IPrefillStallDetector
Constructor:
  - RollingPrefillStallDetector(StallWatchdogConfig config)
Cross-package deps: ProvostXInference.Abstractions, ProvostXInference.Models

### Class: StallRecoveryLadder
> Recovery ladder for the stall-safe prefill layer (INF BL-33).
Constructor:
  - StallRecoveryLadder(StallWatchdogConfig config)
Cross-package deps: ProvostXInference.Models

### Class: SystemChunkWallClock
> Production wall clock backed by the injected <see cref="TimeProvider"/>
Implements: IChunkWallClock
Constructor:
  - SystemChunkWallClock(TimeProvider timeProvider)
Cross-package deps: ProvostXInference.Abstractions

### Record: EpAttempt
> One candidate configuration for an EP session init: a label for
Constructor:
  - EpAttempt(string Label, Action<SessionOptions> Apply)

### Record: JudgeOption
> One rendered option of a judge question: its answer label and its
Constructor:
  - JudgeOption(string Label, string? Description)

### Record: JudgeQuestionAnswer
> ONE raw model answer for one Jev question (session decode → response mapper
Constructor:
  - JudgeQuestionAnswer(string QuestionId, JudgeQuestionType Type, string? Choice = null, double? Score = null, double? Confidence = null, double? Noul = null, IReadOnlyDictionary<string, double>? Probabilities = null, IReadOnlyDictionary<string, string>? Legend = null)

### Record: JudgeQuestionAnswerSet
> The raw answers for ONE Jev request (one state), in question order, plus the
Constructor:
  - JudgeQuestionAnswerSet(IReadOnlyList<JudgeQuestionAnswer> Answers, int StateTokens)

### Record: JudgeQuestionDef
> One typed judge question as trained (id, type, instructions, ordered options).
Constructor:
  - JudgeQuestionDef(string Id, JudgeQuestionType Type, string Instructions, IReadOnlyList<JudgeOption> Options)

### Record: JudgeQuestionRowSet
> Encoded question rows for ONE Jev state + the pre-truncation token count of
Constructor:
  - JudgeQuestionRowSet(IReadOnlyList<JudgeRowEncoding> Rows, int StateTokens)

### Record: JudgeRowEncoding
> Encoded ONE question row of a judge record: token ids, marker (MASK) offsets,
Constructor:
  - JudgeRowEncoding(JudgeQuestionType QuestionType, int[] InputIds, int[] MarkerPositions)

### Record: JudgeScore
> Raw model output for one judged record (executor → response mapper boundary).
Constructor:
  - JudgeScore(JudgeAction RawAction, double Confidence, double Risk, double NeedsSandbox, string ModelVersion)

### Record: JudgeSequenceSpec
> The FULL serialized inference specification for one judge model: record kind,
Constructor:
  - JudgeSequenceSpec(JudgeRecordKind Kind, int MaxLen, int HeadMaxLen, IReadOnlyList<double> Temperature, IReadOnlyList<JudgeQuestionDef> Questions)

### Record: JudgeStateQuestions
> One decision-engine question batch on the Jev wire: the SERIALIZED state text
Constructor:
  - JudgeStateQuestions(string StateText, IReadOnlyList<JudgeQuestionDef>? Questions)
