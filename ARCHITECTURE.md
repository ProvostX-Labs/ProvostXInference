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
Embedded-nextn FILE CONVENTION (live-verified 2026-10-19): `load_mtp` reads the
nextn projections from the SAME GGUF, named at the LAST TRUNK layer index
(`blk.{n_layer-1}.nextn.*` — NOT upstream's `blk.{n_layer}.*`), and wants only the
6 projection tensors (eh_proj/enorm/hnorm/hc_head_norm/hc_head_down/hc_head_up);
`qwen4exp.attention.compress_ratios` must stay n_layer_all-long (do NOT extend for
the nextn layer). See memory/2026-10-19-mtp-merge-fix.md for the merge tooling.
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
  (nextn-layer-only KV, `n_rs_seq = 0`). BL-42: the head context ALWAYS links the
  target trunk via `ctx_other` (upstream ctor parity — self-contained heads ignore it);
  shared-memory heads (`gemma4-assistant`) are SUPPORTED (no catch-up decodes, no
  head seq_rm, all draft rows at the SAME position, state restore replays the h ring
  only) — BL-45 extends the lane to multi-block chain heads (step35 class,
  `n_layer_nextn > 1`, per-step `nextn_layer_offset` selection). Multi-block
  chain heads (BL-45): `chain_heads = n_mtp_layers > 1 && !is_mem_shared` (pure
  planner `pvx_mtp_chain_plan`, unit-tested); the chain budget is clamped to the
  trained block count (`n_max = min(n_max, n_mtp_layers)` — an untrained block is
  never addressed). Shared-memory multi-block heads (gemma4 class) run all blocks
  in one graph — NOT chain mode, BL-42 path byte-identical; single-block heads
  (`n_layer_nextn == 1`) are byte-identical to BL-42 (offset untouched everywhere).
  Catch-up/state-restore replay is per head: seq_rm `[pos_first, end)`,
  `llama_set_nextn_layer_offset(head)`, chunked `(t_p, h_{p-1})` replay
  (`mtp_replay_rows`), offset restored to 0; the head context KV holds every nextn
  block (MTP-context layer filter), so each head rebuilds its OWN layer's committed
  prefix. Chain drafting mirrors upstream draft(): step i selects head i via
  `llama_set_nextn_layer_offset(i)` and re-decodes the WHOLE prefix (seed row +
  drafted so far, pairing row t = `chain_h[t]`, then each head's own nextn output),
  per-step seq_rm `(base, -1)`, logits only on the last row (`pvx_mtp_chain_arm`
  planner, unit-tested); verification/acceptance/commit are unchanged — the
  target-side verify is head-count-agnostic. This family has NO typed refusal
  (upstream supports it); refusals stay only for non-MTP heads, width mismatch,
  vocab incompat. Fixture plumbing: the vision+MTP live leg accepts
  `PVX_MTP_MMPROJ_PATH` (family override) before falling back to
  `PVX_MMPROJ_PATH`/suite MMPROJ — the MTP pair may be a different family than the
  suite's vision fixtures. Pairing convention (DeepSeek-style): a draft row predicting stream
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
  Sequential executor lane only — the batched/pool conversation lane refuses typed
  (capture-hook seams are executor-lane wiring, EAGLE3 precedent); spec enable
  requires an EMPTY stream (per-position capture must observe every commit);
  vision prompts now coexist EXACTLY in the SAME session (BL-39): the capture
  hook takes mirror ENTRIES, so image embd rows ride the same pairing — the
  image embedding is the head INPUT (`add_embd` arm, upstream
  speculative.cpp:1630 "vision tokens carry an embedding instead of an id"),
  the `h_{p-1}` state pairing is unchanged, and every commit site
  (`pvx_executor_infer` slices, the vision-prompt text flush, and EACH embd
  decode chunk — h rows die at the next decode) fires the hook; the capture-
  order invariant (`h_hist.size() == n_tokens`) holds across vision chunks,
  and `mtp_state_restore` replays vision rows the same way (the old typed
  multimodal refusal is gone; pure pairing planner
  `pvx_mtp_catch_up_arm` is unit-tested without models). EAGLE3 still refuses
  vision prompts typed (its per-position feature capture cannot see embd
  rows). AUDIO prompts coexist EXACTLY too (BL-40, same embd-row class as
  BL-39): audio embd rows ride the same `add_embd` input arm with unchanged
  `h_{p-1}` pairing at FLAT pinned positions (upstream `mrope_1d` == plain),
  the hook fires at every audio commit site (the audio-prompt text flush —
  per n_batch slice — and EACH audio embd decode chunk), and the old typed
  audio-in-MTP refusal is gone. EAGLE3 still refuses audio prompts typed
  (same feature-capture seam as vision). State save/restore carries the `h_hist` rows additively (mismatch on
  restore invalidates the session typed, re-init required); kv `shift_left` refuses
  typed. `llama_model_n_layer_nextn(head) == 0` (head GGUF without nextn tensors) is a
  typed refusal naming the boundary — never faked. The TARGET context carries the
  recurrent-state snapshot ring via `pvx_context_params_t.n_rs_seq` (ABI v8 wave-4
  append; upstream `llama_context_params.n_rs_seq`) — 0 = llama default (no
  snapshots, hybrid targets then refuse partial rewind per the existing contract);
  the C# binding mirrors it (`PvxContextParams` 5th field, size-pinned by
  `PvxContextParamsMarshalTests`; `ContextConfig.NRsSeq` pass-through) and the LLM
  layer AUTO-sizes it (8 while `spec_mode draft_mtp` is active).

### Config-surface parity (ABI v8 wave 5) — full llama.cpp config verbs
Every llama.cpp configuration verb is reachable from C# (engine structs → P/Invoke →
facade configs). Wave-5 fields are APPENDED ONLY — a zeroed struct is bit-identical to
the pre-wave-5 load (proven: native suite 48/48, ngram 25/25, stress 31/31 with zeroed
wave-5 everywhere; C# 230/230). Zero-value semantics: model/context ints and floats at
0, bools false, and null kv-override lists mean "llama default" — the native side maps
them to upstream defaults, never to a literal zero override. Knobs whose llama default
is TRUE (use_extra_bufts, offload_kqv, op_offload, swa_full, kv_unified, no_perf) use
a tri-state int: 0 = llama default, 1 = force on, 2 = force off — a plain bool could
not express "unset". kv-override tag mapping: upstream `llama_model_kv_override_type`
order is INT/FLOAT/BOOL/STR; the pvx ABI uses 0=bool, 1=int64, 2=f64, 3=str — mapped
explicitly at the native boundary (never by enum order). Seed: <=0 = time entropy
(llama default), >=1 = deterministic sampling. Wave-5 samplers run inside
`pooled_selection_chain` in OpenAI-parity order: penalties (repeat/present/frequency)
→ logit_bias → dry → top_k → typical_p → top_p/min_p → top_n_sigma → xtc → mirostat
1/2 (mirostat bypasses top_p when active) → dist(seed). Facade pass-through:
`GgufEngineRequest(Builder).WithAdvanced(ModelConfig?, ContextConfig?)` carries the
extended knobs; classic With* fields always win in the merge, so existing callers are
unchanged. ABI sizeof pins (Release, osx-arm64): `pvx_model_params_t` = 72,
`pvx_context_params_t` = 112, `pvx_sampling_params_t` = 112 (incl. wave-5 sub-struct
pointer), `pvx_sampling_params_wave5_t` = 72, `pvx_logit_bias_t` = 8,
`pvx_kv_override_t` = 288 — pinned by `PvxAbiSizeTests` (C#) and probed at build time.

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
- **Backup mechanism (implemented 2026-10-20, seq1-free):** save-state arms by slot
  budget. With `n_seq_max >= 2` it copies seq0 → seq1 (`llama_memory_seq_cp`) and
  restore copies back after a seq-0 clear — cheap cell-level copy, but it spends a
  sequence slot and halves per-sequence capacity (`n_ctx_seq = n_ctx/n_seq_max`).
  With `n_seq_max == 1` (the LLM `seq_max` knob AUTO=1 full-context posture) the
  save now BYTE-SERIALIZES the seq-0 state instead (`llama_state_seq_get_size` /
  `get_data` — cells + K/V rows — held in `pvx_state_s::kv_bytes`) and restore
  replays it via `llama_state_seq_set_data` after the `seq_rm` clear. One full-state
  copy at save time buys save/rewind at full per-session context; the old typed
  `PVX_ERR_NOT_SUPPORTED` refusal is gone. (Verified 2026-10-20: llama-level
  roundtrip is byte-identical and logit-exact on CPU and Metal, q8_0/F16 KV;
  journey `SingleSlotSaveRewindJourneyTests` pins the recall.)
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
unknown/empty/wrong-OS pairs refuse, never fall back. The native-dir RID is ARCH-AWARE
on Linux and Windows: `IOSProbe.Architecture()` (injectable seam; `SystemOsProbe` reads
`RuntimeInformation.ProcessArchitecture`, test doubles override) drives the resolver
(`linux-arm64`/`linux-x64`, `win-arm64`/`win-x64`); the `PvxNative` static ctor derives
the same from `ProcessArchitecture` in the bootstrap context (no injectable seam exists
pre-activation); macOS stays `osx-arm64`. Per-RID payload story (BL-49): NO native
binaries are checked in — every RID slice (osx-arm64 metal + generic, win-arm64 cpu
generic, win-x64/linux-x64 cuda+vulkan) is built by the CI specialist matrix at
NuGet-pack time. This keeps compiler-baked build paths out of the repository
(anonymity rule — MSVC/ld embed absolute paths in debug strings); local payloads
remain gitignored working-tree artifacts for VM testing only. linux-arm64 has no
CI payload — source-build the native engine (cmake) and ONNX Runtime for arm64
(no official ORT arm64 payload exists); the resolver looks in
`runtimes/linux-arm64/native/` for the source-built artifacts. `PvxEngineHandle.Open`:
resolve file → load image → resolve per-image entry-point tables → ABI handshake →
router bundle; the process-wide `PvxDispatchRegistry` routes only the
activation-window DllImports (steady-state calls go through per-handle tables).
Ownership chain is constructor-injected: Model → Context → Pool/Executor → Conversation.

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
  sampling extremes, race stress (grammar mutex, pool churn). Test fixtures are
  model-agnostic — dims/path fixtures derive from the loaded GGUF at runtime (e.g. the
  control-vector test scans `<arch>.embedding_length`/`.block_count`), never hardcoded
  per-model constants; live legs are env-gated (typed skip when the fixture env is unset).
- **C#:** full suite green (251/251 at last run), one test class per type, all native seams
  faked (`FakeModuleLoader`, fakes) — no test touches a real dylib except env-guarded
  live smoke/journey tests. Decision tests run on tiny real ONNX fixtures.
- **Real-model journey tier (BL-47):** `csharp/Tests/RealModelJourneyTests.cs` +
  `LiveModelFixtures.cs` — the formerly manual TEST-MATRIX lanes, automated against a small
  real model (Llama-3.2-1B ≤1 GB: load/generate, KV save/restore + checkpoint round-trip,
  8-conv batching, GBNF grammar, MEAN-pooling embeddings (SIGSEGV guard) + MiniLM encoder,
  speculative draft-simple token-exact, vision mmproj splice, greedy determinism, tokenizer
  hygiene). Fixture discovery is env-driven with repo-relative defaults
  (`PVX_SMALL_*_PATH` / `Tests/models` symlinks); lanes run CPU-only (GpuLayers=0) so the
  same suite is cross-platform by construction; a missing fixture yields a named skip, and
  a RAM gate enforces the >60%-free house rule before every load. The tier is green both
  with fixtures and hermetically (no env, no models → all live lanes skip typed).
- **Live capability matrix:** `TEST-MATRIX-2026-10-18.md` — 6 models (dense, hybrid-GDN,
  MoE/Flash-Next, 3B, MiniLM encoder, judge-ONNX) × all lanes (load, sequential gen,
  continuous batching, KV ops, rewind, speculative, grammar, tool calls, vision, embeddings),
  executed on the Release build.
- **LDC:** API-INDEX.md + RELATIONSHIP-GRAPH.md regenerated + enforcement PASS in the same
  commit as any public-surface change.

## CI / NuGet policy

### Windows (MSVC) build portability
The native tree builds under MSVC (VS 2022, ARM64 and x64) with a fixed portability set:
explicit `<functional>` include (libc++ transitive, MSVC not), `localtime_s` under
`_WIN32` (POSIX `localtime_r`), no designated initializers at /std:c++17 (lambda-built
constant), no raw strings inside the ASSERT macro stringize path, `_USE_MATH_DEFINES`
for `M_PI`, `WINDOWS_EXPORT_ALL_SYMBOLS ON` (no dllexport in the extern "C" header —
restores the macOS-parity export surface + import lib), and WIN32 test executables
link llama/ggml/ggml-base directly (Windows import libs don't re-export transitive
static-lib symbols).

- CI workflow builds native + C# + runs tests; does NOT publish or tag. Specialist
  matrix (6 lanes): osx-arm64 metal, linux-x64 cuda + vulkan, win-x64 cuda + vulkan,
  win-arm64 cpu (BL-49 — the win-arm64 cpu lane replaces the former checked-in payload).
  A dedicated `windows` lane (ci.yml `windows` job, windows-latest) builds the native
  engine with MSVC (Release) + runs ctest (model-gated legs typed-skip without model
  files) + runs the C# suite with the fresh x64 dll staged into
  `csharp/runtimes/win-x64/native/` so the arch-aware resolver finds it.
- Package `ProvostXInference` packs all CI-built RID slices into `runtimes/{RID}/native/`.
  Pack verified end-to-end locally (private nupkg, consumer project resolved the
  package and ran a Metal-backed generation). All work stays local — no push, no tags
  without explicit owner order.

## Files
- `include/pvxinference.h`, `include/pvx_internal.h` — public C API + internal structs
- `src/pvxinference.cpp`, `src/pvx_backend.cpp`, `src/pvx_spec_engine.{cpp,h}`,
  `src/pvx_ngram.{cpp,h}` — native core
- `csharp/` — bindings, facade, Decision domain, tests
- `assets/models/judge/` — ONNX judge assets (tokenizer.json, judge specs)
- `docs/CAPABILITY.md` — capability matrix; root `TEST-EVIDENCE-spec.md` — evidence policy
- `CMakeLists.txt`, `.github/workflows/` — build + CI

**Status:** as-is — native ctest green (Release build), C# suite green, LDC PASS.
Latest gates: native 48/48 + ngram 27/27 + stress 31/31; C# 251/251 (also verified
249+2 skips on the ARM64 Linux and Windows lab VMs, BL-47/48); the real-model journey
tier (10 lanes) is green with and without fixtures; no native binaries in the repo
(BL-49, CI-built payloads).

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
