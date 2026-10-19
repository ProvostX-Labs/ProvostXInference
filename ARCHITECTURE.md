# ProvostXInference — Architecture (as-is)

C/C++ inference engine + C# bindings for ProvostX. A **llama.cpp-only, GGUF-only** execution
substrate: the vendored llama.cpp tree is linked directly behind a flat C API, P/Invoked from a
strict-OOP C# layer. Replaces LLamaSharp entirely — full control over batching, KV cache,
vision, speculative decoding, and backend safety. ProvostXLLM consumes this library in-process
and knows **only** this API — never llama.cpp, ONNX, or any engine internals.

## Layers

```
┌─────────────────────────────────────────────┐
│ ProvostXLLM (C# — HTTP server)               │
│  OpenAI API, sessions, orchestration only    │
├─────────────────────────────────────────────┤
│ ProvostXInference C# (this repo)             │
│  Engines facade → Abstractions → Interop     │
│  Decision (ONNX judge), Encoder, watchdog     │
├─────────────────────────────────────────────┤
│ libpvxinference (C++ — this repo)            │
│  Model, context, pool, batched decode,        │
│  sampling, embeddings, vision, spec decode,   │
│  KV ops, backend safety                       │
├─────────────────────────────────────────────┤
│ llama.cpp (vendored submodule — FROZEN)       │
│  Native inference, Metal/CUDA/Vulkan/CPU     │
└─────────────────────────────────────────────┘
```

Dependency flow is strictly one-directional: callers touch only the Engines facade; the facade
composes backend adapters into one uniform engine object; nothing below leaks upward.

## Native core (src/, include/)

### C API (pvxinference.h) — ABI v8
Flat C API, 90+ exports. Opaque handles: `pvx_model_t`, `pvx_context_t`, `pvx_pool_t`,
`pvx_conversation_t`, `pvx_executor_t`, `pvx_state_t`, `pvx_grammar_t`, mmproj wrapper,
`pvx_adapter_t` (LoRA), `pvx_cvec_t` (control vectors), `pvx_tts_session_t` (gen-audio).
The ABI version is handshake-checked against the C# binding's `ExpectedAbiVersion` at
engine open — mismatch refuses to load (fail-closed, never best-effort). v6 (wave 1) was
append-only: `pvx_spec_mode_t` gained the n-gram modes, `pvx_spec_params_t` appends
`pvx_ngram_params_t`, `pvx_spec_capabilities_t` appends `spec_vision`. v7 (wave 2) is
append-only too: `pvx_spec_mode_t` gains `PVX_SPEC_EAGLE3`/`PVX_SPEC_DFLASH`,
`pvx_spec_params_t` appends the eagle/dflash companion paths, and the new surfaces are
the batched/pool-lane spec trio (`pvx_conversation_spec_init/_generate_spec/_spec_stats`),
the per-conversation spec-session teardown hook (`pvx_conversation_spec_free`), and the
pure JSON-schema→GBNF converter (`pvx_grammar_from_json_schema`, string→string, KV-neutral).
v8 (wave 3) is append-only as well: `pvx_pooling_type_t` gains `PVX_POOL_RANK` (the
rerank lane — classification head attached to the graph, upstream llama-server
`--reranking` contract; `pvx_rerank` scores (query, doc) pairs single-shot on a fresh
seq, refusal typed on any other pooling), audio input rides the mtmd surface
(`pvx_mtmd_supports_audio` / `pvx_mtmd_audio_sample_rate` /
`pvx_executor_prompt_with_audio` / `pvx_conversation_prompt_with_audio` — PCM F32 at
the projector's native rate, no resampling in product code), TTS is a SEPARATE decode
context (`pvx_tts_session_create`/`_free`/`pvx_tts_generate` — single-shot session,
fresh sessions replay byte-identically per (text, seed), typed refusal when the mmproj
has no gen-audio pipeline), LoRA adapters attach/detach on LIVE contexts
(`pvx_adapter_load`/`pvx_context_set_lora`, weights-surface modifier — KV-neutral),
control vectors port `common_control_vector_load` into src/
(`pvx_control_vector_load`/`pvx_context_set_control_vector`, same KV-neutrality),
and the stats export is a snapshot copy-out. v8 wave 4 (draft-mtp, owner order
2026-10-19 17:00) is append-only too: `pvx_spec_mode_t` gains
`PVX_SPEC_MTP_DRAFT` (9) and `pvx_spec_params_t` appends `mtp_draft_path` — the
SEPARATE self-contained MTP head GGUF (upstream `-md --spec-type draft-mtp`),
distinct from `draft_model_path` (draft-simple only) and from
`PVX_SPEC_MTP_AUTO` (embedded nextn, which stays a typed wiring boundary).
The stats export remains a snapshot (`pvx_executor_stats`/`pvx_conversation_stats`,
`pvx_stats_t` copy-out counters, spec telemetry mirrors the spec stats).

- **`pvxinference.cpp`** — core: model load, tokenize, chat template, context, conversation
  pool, batched + executor inference, sampling, KV ops (save/restore/copy/rewind/shift),
  embeddings, speculative decode, chunked decode (all decode paths split pending tokens into
  ≤ `n_batch` slices with per-slice position advance — `llama_decode` ABORTS beyond one batch).
- **`pvx_backend.cpp`** — backend safety layer (below).
- **`pvx_adapters.cpp`** — ABI v8 (wave 3): LoRA adapter load/attach/detach
  (`llama_set_adapters_lora` — weights-surface: live contexts, KV content untouched,
  serialized by the context `infer_mtx`) and the control-vector loader (pattern-port of
  the vendored `common_control_vector_load_one`: `direction.<L>` F32 1-D tensors
  accumulated into an n_embd × max_layer buffer; explicit out-of-range layer ends are a
  typed refusal, never a silent clamp).
- **`pvx_tts.cpp`** — ABI v8 (wave 3): speech output via the PUBLIC mtmd gen-audio
  helper surface (`mtmd_helper_gen_audio_*` — never the frozen common runtimes): a
  dedicated decode context on the backbone (qwen3tts — embeddings on, semantic codes
  through the classification-style output), our own deterministic top-k/top-p backbone
  sampler, raw-PCM output. The session is SINGLE-SHOT: the gen pipeline carries
  cross-call state inside the mtmd context, so a second generate refuses typed
  (`tts/session-consumed`) rather than degrade — fresh sessions replay byte-identically.
- **`pvx_spec_engine.*`** — speculative decoding: draft model (draft-simple / MTP per
  vendored llama.cpp support) AND n-gram modes (simple / map-k / mod / cache — no draft
  model, drafts from the committed stream). The draft-context mirror is EMBD-AWARE
  (ABI v6): the executor's committed-stream ring (`pvx_mirror_entry_t`) replays token
  rows by id and multimodal rows by projected embedding at pinned positions (draft-width
  mismatch zero-pads with a one-time warn — upstream pattern); the ring's invariant is
  enforced fail-closed by a consistency guard on every round. Image prompts land embd
  rows in the ring via the stable `llama_batch_ext` surface (helper semantics ported
  exactly, M-RoPE decoder positions included). Capability discovery on load inspects
  GGUF metadata (arch + nextn/MTP tensors) and self-reports a SUPPORTED-TYPES LIST
  (draft_simple + the n-gram types for every standard-vocab target), not booleans, plus
  the model-independent `spec_vision` bit; unwired types answer typed NOT_SUPPORTED with
  the boundary named. A second draft model is required for draft-simple — missing draft
  = typed error, no silent fallback. A corrupt n-gram cache file is a typed LOAD_FAILED
  naming the path — never a silently empty cache.
- **`pvx_ngram.*`** — pattern-port of the vendored n-gram drafters (simple, map-k with
  accept statistics, mod hashed-suffix table, static+dynamic token-prob caches with
  save/load/merge) against stable llama.h types only; the frozen tree is never included.
- **`pvx_spec_engine.*` (ABI v7 companion modes)** — EAGLE3 (upstream draft-eagle3
  pattern-port): the draft decoder consumes (token, g_embd) pairs, g rows are the
  EAGLE3 encoder output of the target's captured layer features
  (`eagle3.target_layers`, exactly 3 entries, plus compatible vocab family — validated
  typed at spec_init naming the metadata key); deferred-boundary `pending_g_last` state
  rides the existing executor state chain additively (versioned length-prefixed payload:
  an EAGLE3-enabled restore of a checkpoint without EAGLE3 state types the refusal and
  demands re-init). Live-verified against a genuine Qwen3-8B +
  community EAGLE3-speculator pair; three round invariants, kept as-is now: (a) the capture hook drops the drafted-but-unverified chain
  cells before the decoder training batch (draft-KV restore-to-committed-prefix
  pattern, upstream TAG_SPEC_AVOID_DRAFT_REEVAL — without the drop the training
  batch starts at pos_first while the draft KV still ends at the chain tail and
  llama's Y = X+1 consecutiveness check aborts); (b) the verify chain is bounded
  per ROUND by the remaining out_max budget (a call-wide cap alone let the last
  rounds overrun out_max); (c) a deferred-boundary-only capture (single-row
  commit, bridge skipped) adds no pairs and decodes NOTHING (an empty batch is
  llama's n_tokens == 0 refusal). Executor-lane acceptance telemetry:
  `pvx_executor_spec_stats` — its definition shipped without the header
  declaration (C++-mangled export, unreachable from the C ABI); the declaration
  is restored (extern "C" linkage, no ABI-version change). DFlash: companion
  metadata is validated at the metadata boundary,
  then the unwired drafting boundary refuses typed NOT_SUPPORTED (`dflash-wiring`),
  MTP_AUTO precedent — never faked (live-verified against a genuine
  LFM2.5-1.2B-Instruct target + DSpark sidecar pair: the real sidecar — arch
  dflash, dflash.target_layers i32[5] — passes the metadata precheck, the
  enable refuses typed naming spec-engine/dflash-wiring, no phantom session;
  block-diffusion drafting is NOT claimed). Both refuse typed on the
  batched/pool lane.
- **`pvx_json_schema_grammar.*`** (ABI v7) — pattern-port of the vendored
  `common/json-schema-to-grammar` converter (JSON Schema → GBNF, pure string→string,
  feeds the existing grammar chain; the frozen tree is never linked into product code).
  Fail-closed per keyword: invalid JSON → INVALID_ARG naming the parse position; an
  unsupported schema keyword (e.g. regex `pattern`) → NOT_SUPPORTED naming the keyword —
  never a best-effort loose grammar.
- **Batched/pool-lane speculative decoding (ABI v7)** — the executor trio mirrored
  per-conversation (`pvx_conversation_spec_init/generate_spec/spec_stats` + `spec_free`):
  per-seq draft state keyed by the conversation's seq_id, spec ops under the same
  `infer_mtx` discipline as sample ops, pool return tears the session down in the
  existing teardown order, pool `seq_max` budgeted for the draft-side KV. Fail-closed:
spec init on a pool with `seq_max < 2` types the refusal; generation without a session
  types NOT_SUPPORTED — never a silent plain run.

- **Separate-head draft-MTP (ABI v8 wave 4)** — `pvx_spec_engine.*` pattern-port of the
  vendored `common_speculative_impl_draft_mtp` (upstream llama.cpp PR #28243, evolved):
  the MTP head is a self-contained GGUF (own token embedding + output head + one nextn
  block) loaded as a SECOND draft model and run on a `LLAMA_CONTEXT_TYPE_MTP` context
  (nextn-layer-only KV, `n_rs_seq = 0`; `ctx_other` stays unset for self-contained heads
  → independent KV — gemma4-style shared memory and step35 multi-head chains are OUT
  of this slice). Pairing convention (DeepSeek-style): a draft row predicting stream
  position P consumes `(embd(t_{P-1}), h_{P-1})` — `h` = the target trunk's nextn
  embedding (`llama_set_embeddings_nextn(target, true, masked=false)`), row width =
  `llama_model_n_embd_out` (head must match target — typed refusal at spec_init, never
  an upstream-style assert); chain rows pair the DRAFT's own nextn output
  (`llama_set_embeddings_nextn(draft, true, masked=true)`). Round structure mirrors the
  EAGLE3 lane: per-committed-position `h_hist` capture hook after every target decode
  (`mtp_process_target_rows` — catch-up decodes the committed rows into the draft KV
  with the `(t_p, h_{p-1})` pairing, zero-h at position 0, dropping stale chain cells
  first), seed pair-decode `(t_last, h_last)` at the next position, greedy top-1 chain
  under `p_min`/EOG gates, ONE verify decode on the target, sample-and-match
  acceptance (greedy-exact), KV repair, `h_hist` truncated to the committed count.
  Sequential executor lane only — the batched/pool conversation lane and multimodal
  prompts refuse typed (capture-hook seams are executor-lane wiring, EAGLE3
  precedent); spec enable requires an EMPTY stream (per-position capture must observe
  every commit); state save/restore carries the `h_hist` rows additively (mismatch on
  restore invalidates the session typed, re-init required); kv `shift_left` refuses
  typed. `llama_model_n_layer_nextn(head) == 0` (head GGUF without nextn tensors) is a
  typed refusal naming the boundary — never faked.

### Embeddings — architecture-routed
Routing is by the model's own architecture (`llama_model_has_encoder/has_decoder`); typed
NOT_SUPPORTED when a model exposes neither. Encoder models (MiniLM) use `llama_encode`;
causal models decode and mean-pool. Decoder-only models whose output head differs from the
hidden state (merged multi-token heads, e.g. n_embd_out ≠ n_embd) fail closed with a typed
NOT_SUPPORTED before any decode. The extraction path fails closed if no embedding source
produces data — never zeros dressed as an embedding.

### Grammar (GBNF)
Persistent grammar chain on executor/conversation; grammar FIRST in the sampler chain;
sampled token accepted once at commit (EOG/control skipped). Prompt tokens are NOT
grammar-accepted (a per-token accept hook desyncs the stateless feedback loop — removed).
Known upstream llama.cpp issue: `llama_grammar_accept` crashes on multi-character tokens
spanning rules with optional whitespace; try/catch fallback yields unconstrained output;
callers bound array repetition in their grammars as mitigation.

### Rewind / hybrid-memory guard
All partial rewinds check `llama_memory_seq_rm`'s return and answer `PVX_ERR_REWIND_UNSUPPORTED`
on refusal (hybrid/recurrent archs cannot take partial erasure) — counters untouched, no
corruption. Full-range rewind still works everywhere. `pvx_model_partial_rewind_supported`
surfaces the capability (C#: `IInferenceModel.PartialRewindSupported`).

### Backend probe & safety (pvx_backend.cpp)
Per-CONTEXT probe state (`pvx_context_s::backend` — never process-global). Detection skips
CPU-class devices and matches stable vendor-tag device ids (`MTL`/`Metal`, `CUDA`, `Vulkan`);
an unrecognized accelerator reports its raw device name instead of lying "CPU". All seven
`pvx_backend_*` queries take `pvx_context_t*` (null-safe) and are re-run by `pvx_probe_backend`.
GPU-layer profile fields (Effective/Original/ReducedGpuLayers) are filled on every facade path.

Safety layer intervenes only on probe failure: minimal 1-token decode probe validates config;
progressive fallback gpu_layers N → N/2 → 0 (`pvx_create_context_safe`); Vulkan stale-KV
flush after pool return (upstream #26744). On total fallback failure the caller's original
gpu_layers is reloaded; if even that fails, `*model` is null and is not freed.

### Concurrency & lifetime
- `infer_mtx` on every context serializes all `llama_*` work; KV-mutating conversation ops
  (rewind/reset/shift/export) and pool-return hold the SAME lock in a fixed order
  (pool → infer) as lease — no inverted lock pairs.
- `pvx_state_save` clears the backup seq before copying (seq_cp MERGES).
- Model destruction ordering: per-model `active_context_count` + condition variable;
  `pvx_free_model` blocks until all its contexts are freed (prevents Metal device-before-buffers
  aborts); contexts of different models free concurrently.
- Vision: a second mmproj load is refused while a wrapper is alive (refcounted shared mtmd
  context — the model waits for wrapper release before freeing it). A failed pre-image text
  flush in `pvx_executor_prompt_with_images` is a typed decode failure with pending tokens
  kept for retry — image embeddings never land at wrong positions.

### Errors & logging
Every failure path sets a typed result + reason string: context-scoped `pvx_last_error`,
model-scoped `pvx_model_last_error` (ABI v5 — model-level calls must NOT pass a model handle
to the context-scoped function). `PVX_LOG` structured logging: zero-overhead when disabled
(null callback check), levels DEBUG/INFO/WARN/ERROR/NONE, set via `pvx_set_log_callback` +
`pvx_set_log_level`; C# bridge pins the delegate.

## C# layer (csharp/) — one type per file, constructor injection, no statics, no env vars

### Namespace map
| Namespace | Purpose |
|-----------|---------|
| `Interop` | P/Invoke + per-handle dispatch: `PvxNative`, `PvxAbiDelegates`, `PvxEngineHandle`, `PvxDispatchRouter`/`Registry`, `NativeModuleLoader`; injected seams `IPvxModuleLoader`, `IPvxEntryPoints`, `IPvxFileExistence`, `IOSProbe` |
| `SafeHandles` | One SafeHandle per native resource — guaranteed `pvx_free_*` cleanup |
| `Abstractions` | `IInferenceModel/Context/State`, `IConversation(Pool)`, `IStandardExecutor`, `IGrammar`, `IVisionEncoder`, `IInferenceBackend`, watchdog seams `IPrefillStallDetector`/`IChunkWallClock`/`ISpeculationSuspender`/`IStallRecoveryLadder` |
| `Implementation` | Native* wrappers; `StandardEngineActivator`, `PvxDefaultEngineSelector`, stall-safe prefill classes |
| `Engines` | Caller-facing facade: `IInferenceEngine`, `InferenceEngine(Builder/Factory)`, request records + ops views + the black-box helper facades `EngineLogging` (native log routing), `EngineMaintenance` (GGUF EOG metadata repair), `StallWatchdogOps` (sole construction seam for detector/ladder/wall-clock) |
| `Decision` | ONNX judge execution — `DecisionSession`, `IDecisionSessionFactory`, `JudgeTokenizer`, `JudgeSequenceEncoder`, `JudgeNumericsGate`, `ExecutionProviderChain` |
| `Models` | Config records: `ModelConfig`, `ContextConfig`, `SamplingConfig`, `SpeculativeConfig`, `StallWatchdogConfig`, `BackendInfo`, engine selection types |
| `Exceptions` | `InferenceException`, `InvalidEngineRequestException`, `EngineKindMismatchException`, `PrefillStallUnrecoveredException`, `PvxResultExtensions` |
| `Logging` | `IInferenceLogger`, `NativeLogBridge` (pinned delegate, UTF-8) |

### Engine object model (the ONLY caller-facing surface)
```
InferenceEngineBuilder.Gguf(path) / .Decision(path) / .Encoder(path)
  → validated EngineRequest (fail-closed) → InferenceEngineFactory.Create(request)
  → InferenceEngine (unloaded) → Initialize()
  → three typed views: IGgufEngineOps / IDecisionEngineOps / IEncoderEngineOps
```
- Kind routing is exhaustive + fail-closed; a view for the wrong kind throws
  `EngineKindMismatchException`. Views WRAP the underlying graph, never duplicate it.
- `IGgufEngineOps` exposes the model/context/pool graph: chat template, tokenize, sampling
  knobs, live context rebuild, KV copy/clear, checkpoints, rewind, speculation
  (validate/enable/disable, `ActiveSpeculation` mirror), live `Backend` probe.
- `IDecisionEngineOps` exposes the judge session; `IEncoderEngineOps` the vision/image
  embedding route.
- Load failure leaves the engine unloaded and disposable — no partial state leaks.

### Per-handle backend dispatch (engine selection)
`ModelConfig.Engine` picks the specialist (`metal` = macOS only; `cuda`/`vulkan` =
Windows/Linux only). `PvxEngineSelector` + `PvxEngineFileResolver` validate fail-closed —
unknown/empty/wrong-OS pairs refuse, never fall back. `PvxEngineHandle.Open`: resolve file →
load image → resolve per-image entry-point tables → ABI handshake → router bundle; the
process-wide `PvxDispatchRegistry` routes only the activation-window DllImports (steady-state
calls go through per-handle tables). Ownership chain is constructor-injected:
Model → Context → Pool/Executor → Conversation.

### Stall-safe chunked prefill layer
Engine-level, backend-independent, above the native seam (no ABI surface):
`ChunkedStallSafePrefillRunner` commits PromptTokens in chunks over `IConversationPool`,
timing each with `IChunkWallClock` (wall-clock is the sole mechanism — no backend hooks).
`RollingPrefillStallDetector`: stall = chunk wall > GapMultiplier × rolling median of the
baseline window; stalled walls never enter the baseline. `StallRecoveryLadder`: FreshSlotRetry
(smaller-chunk retry halves to the floor) → SmallerChunkRetry → DraftDisabledRetry (via
injected `ISpeculationSuspender`, idempotent native teardown) → FailLoud. Budget exhausted =
typed `PrefillStallUnrecoveredException` carrying chunk index, walls, baseline, steps.
Watchdog disabled = single plain commit, zero overhead. Config: `stall_watchdog` section
(fail-closed Validate).

## Test coverage & evidence

- **Native:** basic + stress C++ suites (`tests/`), Release build, `ctest` green —
  stress covers NULL args, empty inputs, pool exhaustion, double-return, KV boundaries,
  sampling extremes, race stress (grammar mutex, pool churn).
- **C#:** full suite green (189/189 at last run), one test class per type, all native seams
  faked (`FakeModuleLoader`, fakes) — no test touches a real dylib except env-guarded
  live smoke/journey tests. Decision tests run on tiny real ONNX fixtures.
- **Live capability matrix:** `TEST-MATRIX-2026-10-18.md` — 6 models (dense, hybrid-GDN,
  MoE/Flash-Next, 3B, MiniLM encoder, judge-ONNX) × all lanes (load, sequential gen,
  continuous batching, KV ops, rewind, speculative, grammar, tool calls, vision, embeddings),
  executed on the Release build.
- **LDC:** API-INDEX.md + RELATIONSHIP-GRAPH.md regenerated + enforcement PASS in the same
  commit as any public-surface change.

## CI / NuGet policy

- CI workflow builds native + C# (Debug + Release, 0 warnings) + runs tests; does NOT
  publish or tag. Planned 5-specialist matrix: osx-arm64 metal/cpu, linux-x64 cpu/cuda/vulkan,
  win-x64 cpu/cuda/vulkan.
- Package `ProvostXInference` prepared (native binaries as `runtimes/{RID}/native/`), not
  shipped. All work stays local — no push, no tags without explicit owner order.

## Files
- `include/pvxinference.h`, `include/pvx_internal.h` — public C API + internal structs
- `src/pvxinference.cpp`, `src/pvx_backend.cpp`, `src/pvx_spec_engine.{cpp,h}`,
  `src/pvx_ngram.{cpp,h}` — native core
- `csharp/` — bindings, facade, Decision domain, tests
- `assets/models/judge/` — ONNX judge assets (tokenizer.json, judge specs)
- `docs/CAPABILITY.md` — capability matrix; `TEST-EVIDENCE-spec.md` — evidence policy
- `CMakeLists.txt`, `.github/workflows/` — build + CI

**Status:** as-is — native ctest green (Release build), C# suite green, LDC PASS.

## Black-box consumption contract

The caller (ProvostXLLM) consumes this library ONLY through the engine/builder/factory
facade surface; everything else in this repo is a CLOSED BLACK BOX to it. To make that
consumable without leaking internals, three facade views live in `csharp/Engines`
(pure delegation, no logic added):

- `StallWatchdogOps` — sole construction seam for the stall-watchdog helper family;
  returns `IPrefillStallDetector` / `IStallRecoveryLadder` / `IChunkWallClock`
  (the ladder interface is `csharp/Abstractions/IStallRecoveryLadder.cs`;
  `Implementation.StallRecoveryLadder` implements it).
- `EngineMaintenance` — file-level engine-bound maintenance (GGUF EOG/special-token
  metadata repair) without a loaded engine object in hand.
- `EngineLogging` — process-wide native log routing (file sink / managed handler /
  level), previously reachable only via the `Logging` namespace.

Effect: callers never reference `Implementation`, `Interop`, `SafeHandles`, or `Logging`
namespaces at all. The native tree is untouched by this contract (the C API surface is
unchanged). `Decision.Engine.DecisionSessionFactory` therefore constructs parameterless —
the logger is a settable `Logger` property (quiet default) — so a facade-only caller's
compiled metadata never drags the `Logging` namespace in (a ctor parameter's member-ref
signature would do exactly that). No caller passes a logger through the ctor.
