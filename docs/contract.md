# ProvostXInference — Consumer Contract

**Audience:** developers of consumer services (ProvostXLLM, tools, tests). This is the
stability promise of the caller-facing surface: what you may touch, what is guaranteed,
and what is closed. Anything not listed here is internal and may change without notice.

## 1. The only entry point

```csharp
using ProvostXInference.Engines;        // builder, factory, ops views, helper facades
using ProvostXInference.Models;         // config records, EngineStats, EngineTelemetry

var engine = InferenceEngineBuilder.Gguf(modelPath)
    .WithGpuLayers(99)
    .Build();                           // validated EngineRequest, fail-closed
engine.Initialize();
var ops = engine.Gguf;                  // typed view; wrong kind => EngineKindMismatchException
```

You reference exactly two namespaces: `ProvostXInference.Engines` and
`ProvostXInference.Models`. Everything else (`Implementation`, `Interop`, `SafeHandles`,
`Logging`, `Abstractions`) is a closed black box.

## 2. Engine object lifecycle contract

| Step | Guarantee |
|---|---|
| `Builder.*` → `Build()` | Request validated fail-closed; bad paths/kinds/environments throw typed exceptions before any native work |
| `Initialize()` | Engine loads; failure leaves the engine **unloaded and disposable** — no partial state leaks |
| `engine.Gguf` / `.Decision` / `.Encoder` | Typed views over ONE graph; views wrap, never duplicate |
| `Dispose()` | Teardown in fixed order (pool → context → model); safe to call twice; safe after failed load |

## 3. What `IGgufEngineOps` guarantees

- **Model/context graph:** chat template, tokenize/detokenize, sampling knobs, live
  context rebuild (`RebuildContext()` — old pool is disposed, state is not preserved),
  KV copy/clear, checkpoints (export/import byte-exact), rewind, shift.
- **Backend probe:** `Backend` reports the TRUTH — active specialist, effective/requested
  GPU layers, reduced flag. It never reports CPU for an active accelerator.
- **Stats & telemetry (ABI v9):**
  - `GetExecutorStats(executor)` / `GetConversationStats(conv)` — per-lane counters
    (prompt tokens, decode steps, committed tokens, spec drafted/accepted). Snapshot
    copy-out; rejected drafts derive as drafted − accepted.
  - `GetEngineTelemetry()` — model-wide snapshot for THIS engine instance:
    `EngineTelemetry(PromptTokens, DecodeSteps, CommittedTokens, SpecDrafted,
    SpecAccepted, Errors, ContextsAlive)`. **Pull-based and per-engine** — derive rates
    between two snapshots in the caller; the engine keeps no rate state.
- **Speculation:** capability self-report (`SpeculativeCapabilities` — supported-types
  list, not booleans), validate/enable/disable. Unwired capability ⇒ typed
  `NOT_SUPPORTED` naming the boundary. Never a silent plain-generation fallback.
- **Grammar:** `CreateGrammar` (GBNF) / `CreateGrammarFromJsonSchema`. Every sampled
  token passed a grammar check (tiered rejection sampling) — output always conforms.

## 4. Per-engine audit trail (ABI v9)

Set on the builder via `ModelConfig`:

| Setting | Meaning |
|---|---|
| `AuditPath` (null/empty = off) | JSON-lines audit file for THIS engine instance only |
| `AuditIncludePrompts` | also record prompt token **counts** (never prompt text) |

Guarantees:
- **Per-engine isolation:** two engines in one process write two independent files and
  keep two independent telemetry streams. Nothing is shared (test-pinned).
- **Events:** `model_loaded`, `model_freed`, `context_created`, `context_freed`,
  `spec_enabled`, `spec_disabled`, `kv_cleared`, `kv_shifted` — each `{"ts", "seq",
  "event", "detail"?}`, monotonic seq, flushed per line (survives hard kill).
- **Content policy:** prompt TEXT is never recorded. Audit logs are not a content sink.
- **Zero cost when off:** unconfigured engines execute one branch per commit and
  nothing else — no counters, no I/O.
- **Best-effort by design:** audit is observability, never a correctness dependency;
  an unusable path logs a warning and disables the sink (load still succeeds).

## 5. Process-wide debug logging vs per-engine audit

- `EngineLogging.SetLogFile/Enable/SetLevel` routes the native DEBUG stream
  (llama.cpp/ggml/Metal + pvx internal logs). This is process-wide by nature — one
  specialist module owns the log state. Use it for diagnostics.
- Use `AuditPath` for per-engine operation trails. The two are independent channels.

## 6. Error contract

Every failure is a typed exception carrying an error code, message, and a named
boundary (e.g. `spec-engine/mtp-wiring`, `json-schema/parse`, `kv/rewind`):

| Exception | When |
|---|---|
| `InferenceException` | any native-side failure (load, decode, capability, marshaled refusals) |
| `InvalidEngineRequestException` | builder request failed validation |
| `EngineKindMismatchException` | wrong ops view for the engine kind |
| `PrefillStallUnrecoveredException` | stall watchdog exhausted its recovery ladder |

Refusals are always typed and named — never empty messages, never silent fallbacks,
never process aborts on supported paths.

## 7. Stability rules

- **ABI handshake:** the native library version must match the binding; mismatch refuses
  to load. Struct layouts are size-pinned by tests.
- **Append-only ABI:** new fields are appended; a zeroed struct reproduces previous
  behavior bit-for-bit. Consumers compiled against an older surface keep working.
- **No silent behavior changes:** every capability boundary refuses typed instead of
  degrading. If a feature reports supported, it is verified by tests.
- **Config validation is fail-closed:** invalid configs are rejected at build time,
  never clamped silently.

## 8. What is NOT in the contract

- Anything under `Implementation`, `Interop`, `SafeHandles`, `Logging`, `Abstractions`
- Internal counters not surfaced through the stats/telemetry records
- Native C API details (the flat `pvx_*` surface) — consumers go through the facade
- Prompt text handling inside audit trails (explicitly excluded by policy)

## 9. Version history

| ABI | Consumer-visible change |
|---|---|
| v8 | wave-5 config verbs, stats export, audio/TTS/rerank/LoRA/cvec |
| v9 | per-engine audit (`AuditPath`, `AuditIncludePrompts`) + `pvx_engine_telemetry` / `GetEngineTelemetry()` |
