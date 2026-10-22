# BACKLOG-MASTER — ProvostXInference (per-repo work queue)

**Summary:** Active work queue. ONLY items owner-accepted from the global IDEAS-MASTER (~/Agent/ProvostX/docs/IDEAS-MASTER.md) enter here. Agents pick one item at a time.

## Item schema
```
### BL-<n> · <title>
- Source: <IDEAS-MASTER ID> (accepted by owner <date>)
- Spec: one-two sentences
- Scope: files/areas allowed to touch — NOTHING outside
- Acceptance: which tests prove it (unit / integration / journey)
- Status: ready / in-progress / in-review / done
```

## Rules
- One item per sandbox clone. No drive-by changes.
- Implementation: sandbox clone → build → tests → verify live → merge to main.
- Cross-repo items reference the shared contract doc; implement this repo's slice only.
- ⚠ DOC GATE (owner 2026-10-09 13:26, MANDATORY): ARCHITECTURE.md MUST be updated and LDC REGENERATED + PASS in the sandbox clone BEFORE merge (pre-merge check). THEN, after merge, a post-merge verification pass on main: re-run LDC + review ARCHITECTURE.md against the merged state; if anything fails, item stays NOT done until fixed on main. Both gates required.
- Finished items get a `done` line with commit + date, then stay for audit (trim quarterly).
- Owner is the only gate for new items AND for merging to main.

## Queue

### BL-01 · ggml multi-backend isolation spike (Phase A of INF-01)
- Source: INF-01 (accepted by owner 2026-10-09 13:36)
- Spec: prove or disprove that ggml's global backend registries allow two different-specialist dylibs loaded in ONE process without cross-contamination. Spike only — no product code.
- Scope: spike project under csharp/tests or scratch dir; PvxEngineLibrary, loader seam only. NOTHING else.
- Acceptance: written verdict doc (docs/) + repro test; green = Phase B unlocked, red = design rethink reported to owner.
- Status: **done** 2026-10-09 13:53 — verdict GREEN. Evidence: per-image ggml registries (image A registry identical after image B init; reg_get(0) pointers differ), zero symbol collisions via per-handle resolvers, dlopen path-keying confirmed. Verdict doc: `docs/spike-BL01-ggml-isolation.md`.

### BL-02 · Config-driven per-OS backend selection + per-handle dispatch + CI matrix (Phase B of INF-01)
- Source: INF-01 (accepted by owner 2026-10-09 13:36) — REQUIREMENT: model config selects WHICH backend dylib runs, depending on OS (metal on macOS; cuda/vulkan on Win/Linux; fail-closed on invalid pairs)
- Spec: per-handle dispatch tables replacing the global P/Invoke routing; model-config engine field resolved per-OS at load; CI matrix builds all 5 specialists.
- Scope: PvxEngineLibrary, engine pool/SafeHandle layer, model config schema + validator, csproj packaging, CI workflows. NOTHING outside.
- Acceptance: unit (selector matrix incl. fail-closed pairs), integration (live load per available specialist), journey (model starts on configured backend, wrong-OS config refuses); token-exactness re-proven per specialist; ARCHITECTURE.md + LDC per double doc gate.
- Status: **done** 2026-10-09 16:05 — merged to main. Live SmokeTests 51/51 (metal), Tier-2 exactness TOKEN_EXACT=YES against MAIN build; internal router method renamed Resolve→RouteLibrary in doc-gate commit; CI matrix build proven locally.

### BL-03 · Error-masking when specialist dylib is missing
- Source: owner order 2026-10-09 18:41 "fix all bugs" — bug surfaced during BL-01/BL-02 verification.
- Spec: audit remaining swallow-sites for the masking class; verify each intentional swallow is DOCUMENTED and logs/traces loudly; file/fix any real masking found.
- Scope: docs/ + csharp only where a genuine mask is found. NOTHING else.
- Acceptance: sweep table in this item (site → verdict intentional/real); missing-dylib error path test; ARCHITECTURE.md + LDC per double doc gate if code changes.
- Status: **done** 2026-10-09 19:02 — sweep verdicts: 3 intentional+documented (PvxNative static-ctor safety net, NativeModuleLoader.GetExport→typed export error, FileExists wrong-OS fail-closed), 1 real observability gap FIXED (pool-return trace now Release-observable). Net code change: one line. New tests: `ErrorMaskingSweepTests.cs` (3). Build 0/0; C# tests 97/97.

### BL-05 · Speculative decoding in the pvx_* ABI (sequential mode) — INF-03 Inference slice
- Source: INF-03 (accepted by owner 2026-10-11 22:12)
- Spec: surface speculative decoding in the engine matrix: (a) draft-model mode — second small draft model + prompt/verify loop exposed through pvx_* ABI; (b) MTP hooks for models with native multi-token-prediction heads. Engine self-report gains spec-decode capability flags; typed NOT_SUPPORTED where model/engine lacks the feature (fail-closed).
- Scope: include/pvxinference.h (ABI additions only where required), src/ (spec decode path), csharp/Interop + tests, docs/, CMakeLists. NOTHING else.
- ⚠ OWNER REFINEMENT 22:22: capability discovery is the core feature — on model load the ENGINE inspects the model (arch, MTP layers, draft structures) and self-reports a SUPPORTED-TYPES LIST from the llama.cpp speculative enum (not boolean flags). If the model supports NO native type, "second draft model" remains available — it works for any target model when a draft model is provided.
- Acceptance: live smoke — sequential generation with draft model shows measurable speedup + IDENTICAL output text vs non-spec path (token-exact); unit tests for verify loop + capability flags; C# tests green; doc gates.
- Status: **merged/done** 2026-10-12 01:30 — capability discovery (supported-types list from GGUF metadata), draft-simple wired, MTP typed per vendored support, C# bindings + SpecDecodeTests, vision fixture pairing fix; native 21/21 (0 skipped incl. vision); token-exact vs plain loop proven. Live speedup evidence: draft-mtp Qwen3.5-9B +31-43%.

### BL-12 · Engine-object blackbox refactor S0–S5 (ENGINE-OBJ-01)
- Spec: caller-facing engine object = the ONLY surface (implemented: see csharp/ARCHITECTURE.md Engines facade). Remaining stages: S2 LLM flip (LLM consumes the facade exclusively), S4 decision/vision fold refinements, S5 residue sweep. Arch-test: LLM must reference no backend types.
- Status: facade + factory + three ops views implemented and gated (2026-10-17); S2 LLM-side flip pending owner go.

### BL-33 · Stall-safe chunked prefill layer (engine-level)
- Spec: `stall_watchdog` config + `ChunkedStallSafePrefillRunner` over `IConversationPool` — wall-clock stall detection, recovery ladder (fresh-slot retry / smaller-chunk retry / draft-disable retry via `ISpeculationSuspender`), terminal FailLoud with typed `PrefillStallUnrecoveredException`. Backend-independent by construction.
- Status: **done** 2026-10-17 — full unit + runner integration coverage; wired into ProvostXLLM per-model config (see LLM-side evidence there). Remaining increment: live ~224k-token journey with watchdog active (large-memory host, RAM-gated); optional finer-granularity backend timing hook (wall-clock fallback mandatory, never a dependency).

### BL-35 · Upstream parity wave 1 (ABI v6): embd-aware spec mirror, spec+vision, n-gram modes
- Source: owner order 2026-10-18 15:04 (upstream-parity plan, docs/plans/UPSTREAM-PARITY-PLAN.md)
- Spec: embd-aware mirror ring in the standard executor (token id OR projected
  embedding, pinned positions — the vendored draft-simple batch-builder pattern, zero-pad
  + one-time warn on draft-width mismatch) so spec sessions draft after image prompts;
  fail-closed mirror-consistency guard per round. Spec+vision capability bit
  (`spec_vision`); the consumer's blanket image refusal flips to a capability check with
  sticky per-session auto-off until KV reset. n-gram speculative modes (simple / map-k /
  mod / cache) ported from the vendored common/ngram-* implementation behind
  `PVX_SPEC_*` params + knobs via ABI + SpeculativeConfig + the ops/facade speculation
  surface; capability self-report lists the n-gram types. ABI v6 append-only with C#
  lockstep in the same commit.
- Scope: src/pvx_spec_engine.{h,cpp}, src/pvx_ngram.{h,cpp} (new), src/pvxinference.cpp,
  include/pvxinference.h, include/pvx_internal.h, tests/, csharp/ (interop + models +
  implementation + tests), docs; consumer flip in ProvostXLLM Engine/SessionContext.
- Acceptance: native ctest green incl. new unit tests (ngram map/mod/cache logic, mirror
  embd entries, mirror guard) + n-gram & spec-vision token-exactness (spec ON == OFF);
  C# build 0/0 (Debug+Release) + full suite green incl. new n-gram exactness/fail-closed
  tests; live gates env-gated; LDC both roots PASS; docs as-is.
- Status: **done** 2026-10-18 — branch wave1-upstream-parity. Native ctest 3/3
  (basic 24/24 incl. new spec_ngram_token_exactness ×4 modes, spec_ngram_fail_closed,
  spec_vision_exactness; ngram unit suite 14/14; stress). C# 194/194 (Debug+Release,
  0 errors/0 warnings) incl. n-gram fail-closed + 4-mode token-exactness vs plain loop.
  Live evidence: n-gram ON == OFF byte-identical (Llama-3.2-3B native + Qwen3.5-4B C#);
  image-prompt spec ON == OFF prefix-identical (Qwen3.5-4B + mmproj, ngram mode).
  NOT verified live: draft-model spec after images (no vocab-compatible draft GGUF on
  this host — every candidate is rejected typed at spec_init, which is the fail-closed
  contract working). ProvostXLLM flip: spec/session tests 60/62 (2 pre-existing live-
  server-fixture failures, no loaded model). LDC PASS both roots.

- Audit evidence (2026-10-18 comprehensive audit, branch wave1-upstream-parity): mirror
  bookkeeping verified on all mutation paths (prompt commit per-slice, rewind, shift_left,
  kv_copy, checkpoint save/restore/import, pool return/lease); ONE regression found+fixed:
  pvx_executor_infer's mid-chunk decode-failure path re-added recent_tokens + mirror entries
  for already-committed slices (double-commit, breaks mirror.size()==n_tokens invariant;
  commit 2c521e2). NEW permanent leg: conversation_checkpoint_import_phantom_prefix_guard —
  checkpoint import restores KV+counters but NOT the ring → spec init must refuse typed
  naming mirror-consistency (fresh-lease control OK, imported stream continues after a fresh
  decode). Native suite 44/44 with standard fixtures (was 43/43), 0 skipped.

### BL-36 · Upstream parity wave 2 (ABI v7): EAGLE3/DFlash companion modes, batched-lane spec, JSON-schema→GBNF
- Source: owner order 2026-10-18 17:43 (upstream-parity plan wave 2, docs/plans/UPSTREAM-PARITY-PLAN.md)
- Spec: companion-draft modes behind `pvx_spec_mode_t` appends (PVX_SPEC_EAGLE3 /
  PVX_SPEC_DFLASH): EAGLE3 pattern-port of the vendored draft-eagle3 (draft decoder fed
  (token, g_embd) pairs from the target's captured layer features — `eagle3.target_layers`
  metadata validated typed at spec_init naming the key; deferred-boundary state rides the
  state chain additively); DFlash metadata precheck + typed NOT_SUPPORTED wiring boundary
  (never faked, MTP_AUTO precedent). Batched/pool-lane spec trio per conversation
  (`pvx_conversation_spec_init/_generate_spec/_stats/_free`, `infer_mtx` discipline, pool
  return teardown order, ≥2 seq slots typed refusal). JSON-schema→GBNF converter
  (`pvx_grammar_from_json_schema`, pure string→string pattern-port; invalid JSON →
  INVALID_ARG naming the parse position, unsupported keyword → NOT_SUPPORTED naming the
  keyword; golden pin vs tests/fixtures/decision-envelope.gbnf). ABI v7 append-only; C#
  `ExpectedAbiVersion=7` lockstep in the same commit.
- Scope: include/pvxinference.h, include/pvx_internal.h, src/pvx_spec_engine.{h,cpp},
  src/pvx_json_schema_grammar.cpp (new), src/pvxinference.cpp, CMakeLists.txt, tests/
  (test_basic.cpp legs 25–32, fixtures), csharp/ (interop delegates/structs/handle-table,
  models SpeculativeMode/Config/SpeculationStats/GrammarKind, abstractions IConversation/
  IInferenceModel/IGrammar, implementation NativeConversationSpecSession (new, one per
  file) + Native{Conversation,Grammar,InferenceModel,StandardExecutor}, engines
  IGgufEngineOps/GgufEngineOps facade, tests SpecDecode/JsonSchemaGrammar/StallWatchdog/
  EngineTestDoubles), docs (ARCHITECTURE ×2 as-is ABI v7).
- Acceptance: native ctest green incl. new legs (eagle3/dflash typed, conversation spec
  exactness ×4 modes + churn + rewind + fail-closed, json-schema roundtrip); C# build
  0/0 (Debug+Release) + full suite green incl. new capability-gating matrices + typed
  refusals + converter fixture pin; KV/context regression sweep green; live gates
  env-gated; LDC both roots PASS; docs as-is.
- Status: **done** 2026-10-18 — branch wave1-upstream-parity. Native ctest 2/2 (basic
  32/32: rewind/shift/state/copy/pool legs green — [3] partial_rewind_capability,
  [4] partial_rewind_refused_on_hybrid, [6] pool, [9] state_save_restore, [12] pool_reuse,
  [14] shift_left, [15] kv_copy, [28] conversation_spec_rewind_continue; new wave-2 legs
  [25]–[32]; ngram suite 14/14). C# build 0/0 (Debug+Release); full suite 206/206
  (194 baseline + 12 new) default-env + live targeted 21/21 (Llama-3.2-3B: ngram 4-mode
  token-exactness, conversation spec lane token-exact + typed refusals + stats, eagle3/
  dflash typed refusals, JSON-schema converter byte-exact vs fixture). Live harness:
  draft+vision BYTE-EXACT 24/24 (Qwen3.5-4B + mmproj + Qwen3.5-0.8B draft). NOT verified
  live: EAGLE3 drafting (no EAGLE3 companion GGUF on this machine — enabling types the
  refusal at the metadata/load boundary, the fail-closed contract working). LDC both
  roots regenerated + enforcement PASS.
- Live verification (2026-10-18 evening, owner order): BOTH companion legs verified
  with real HF pairs (downloaded ~7.96 GB ≤ 8 GB budget into the owner-sanctioned
  models dir; sizes verified byte-exact vs HF metadata; RAM-gated loads, 95% free).
  EAGLE3 REAL DRAFTING: Qwen/Qwen3-8B-GGUF (Qwen3-8B-Q4_K_M.gguf, 5.03 GB) +
  williamliao/Qwen3-8B-EAGLE3-Speculator-GGUF (Qwen3-8B-speculator.eagle3-F16.gguf,
  2.05 GB; eagle3.target_layers [2,18,33]) — spec ON vs plain greedy (N=24, temp 0,
  top_k 1, ignore_eos in both legs) BYTE-IDENTICAL 24/24 tokens; spec stats
  drafted=95 accepted=9 (accept-rate 9.5% — real drafting, no silent degradation).
  The live run surfaced FOUR native defects, all fixed + full rerun green:
  1) `pvx_executor_spec_stats` shipped without its header declaration — C++-mangled
     export unreachable from the C ABI (declaration restored, symbol extern "C");
  2) eagle3 capture did not drop the drafted-but-unverified chain cells before the
     decoder training batch — llama's Y = X+1 consecutiveness check aborted round 1
     ("training write failed": draft KV pos_max 30 vs batch start 24) — the
     restore-to-committed-prefix drop added (upstream TAG_SPEC_AVOID_DRAFT_REEVAL);
  3) the verify chain was bounded by a call-wide cap only — the last rounds overran
     out_max (26 emitted for N=24) — per-ROUND budget bound added;
  4) deferred-boundary-only captures (single-row plain-step commits, bridge skipped)
     decoded an EMPTY training batch — llama refuses n_tokens == 0 — batch.size() > 0
     guard added (upstream pattern).
  DFLASH leg: LiquidAI/LFM2.5-1.2B-Instruct-GGUF (LFM2.5-1.2B-Instruct-Q4_0.gguf,
  0.70 GB, arch lfm2) + LiquidAI/LFM2.5-1.2B-Instruct-DSpark-GGUF
  (LFM2.5-1.2B-Instruct-DSpark-Q4_K_M.gguf, 0.18 GB, arch dflash,
  dflash.target_layers i32[5]) — the honest typed boundary verified LIVE: the genuine
  sidecar PASSES the metadata precheck, then `pvx_model_spec_validate` AND
  `pvx_executor_spec_init` both refuse typed PVX_ERR_NOT_SUPPORTED naming
  spec-engine/dflash-wiring; no phantom session (generate_spec typed refusal); the
  real target generates plain greedy 8/8. Block-diffusion drafting is NOT claimed
  (ABI v7 typed wiring boundary — never faked; the finding stands: our vendored
  upstream dflash/DSpark wiring is not ported, and no GGUF pair can make pvx draft
  dflash this slice).
  Permanent env-gated tests: native `spec_eagle3_live_pair` / `spec_dflash_live_pair`
  and C# `Eagle3_Live_TokenExact_EnvGated` / `Dflash_Live_TypedWiringRefusal_EnvGated`
  (PVX_EAGLE3_TARGET_PATH / PVX_EAGLE3_DRAFT_PATH, PVX_DFLASH_TARGET_PATH /
  PVX_DFLASH_DRAFT_PATH; typed skip when unset; no absolute paths; exact HF repo+file
  names in the doc comments for re-provisioning). Full regression after the fixes:
  native ctest 3/3 — basic 34/34 passed 0 skipped (incl. both live legs) with standard
  fixtures (Llama-3.2-3B, Qwen3.5-4B hybrid + mmproj, Llama-1B draft, MiniLM embed) +
  ngram 14/14 + stress; C# build 0/0 + full suite 208/208 (206 baseline + 2 new)
  default-env + targeted live 2/2 env-set; LDC both roots regenerated + PASS.

- Audit evidence (2026-10-18 comprehensive audit): EAGLE3 session lifecycle gap found+fixed
  (2c521e2): the graceful eagle_on_stream_shrunk hook existed but was never invoked — a live
  EAGLE3 session died permanently (capture-order invariant typed INTERNAL) after ANY executor
  rewind/reset; now wired into rewind + reset under infer_mtx, and executor shift_left refuses
  typed NOT_SUPPORTED (front-drop cannot re-index g rows) while preserving the conversation-lane
  refusal symmetry; executor shift_left additionally now resets grammar state (conversation twin
  did). DFlash metadata-precheck vs draft-tensor contract re-verified fail-closed (no change).
  Batched-lane round lock verified: nothing inside a spec round re-enters infer_mtx (typed
  refusal paths are pre-round). Re-test: eagle3/dflash live pairs green, ngram 23/23, stress 31/31.

### BL-37 · Upstream parity wave 3 (ABI v8): audio in/out, rerank, LoRA, control vectors, stats
- Source: owner order 2026-10-18 20:16 (upstream-parity plan wave 3, docs/plans/UPSTREAM-PARITY-PLAN.md)
- Spec: six plan features, ABI v8 additive-only, PVX_ABI_VERSION + C# ExpectedAbiVersion bumped
  lockstep in the same commit: (6) audio input — `pvx_mtmd_supports_audio` /
  `pvx_mtmd_audio_sample_rate` probes + `pvx_executor_prompt_with_audio` /
  `pvx_conversation_prompt_with_audio` (MTMD_INPUT_CHUNK_TYPE_AUDIO through the same embd-row
  mirror mechanics as vision; PCM F32 at the projector's native rate, NO resampling in product
  code); (7) TTS — `pvx_tts_session_create`/`_free`/`pvx_tts_generate` via the PUBLIC
  mtmd-helper gen-audio surface, a SEPARATE decode context on the backbone (qwen3tts:
  embeddings on), our own deterministic top-k/top-p backbone sampler (frozen common runtimes
  untouched), SINGLE-SHOT session (`tts/session-consumed` — upstream gen pipeline carries
  cross-call state in the mtmd context; fresh sessions replay byte-identically per (text, seed));
  (8) rerank — `PVX_POOL_RANK` pooling append + `pvx_rerank` (upstream llama-server --reranking
  contract: RANK pooling attaches the classification head; score = pooled embd[0]; "rerank"
  chat template or BOS/query/EOS/SEP/doc/EOS token fallback; fresh seq per pair);
  (9) LoRA — `pvx_adapter_load` (arch precheck naming both) + `pvx_context_set_lora`
  (weights-surface: LIVE context attach/detach under infer_mtx, all-or-nothing, scale > 0 typed);
  (10) control vectors — `pvx_control_vector_load` (pattern-port of common_control_vector_load
  into src/: direction.<L> F32 accumulation, n_embd typed mismatch naming both dims, explicit
  out-of-range layer ends typed, never clamped) + `pvx_context_set_control_vector` (NULL clears);
  (12) stats export — `pvx_executor_stats`/`pvx_conversation_stats` snapshot (pvx_stats_t copy-out
  counters: tokens / kv-pos / prompt-tokens / decode-steps + spec telemetry mirroring spec stats).
- Scope: include/pvxinference.h, include/pvx_internal.h, src/pvx_adapters.cpp (new),
  src/pvx_tts.cpp (new), src/pvxinference.cpp, CMakeLists.txt, tests/test_basic.cpp (legs
  35–43), csharp/ (interop PvxAbiDelegates/PvxTtsParams/PvxStats/IPvxEntryPoints/PvxHandleEntryPoints/
  PvxNative v8 + PoolingType.Rank, SafeHandles Adapter/ControlVector/TtsSession, models
  AudioChunk/TtsRequest/TtsResult/EngineStats, abstractions ILoraAdapter/IControlVector/
  ITtsSession + IInferenceModel/IInferenceContext/IStandardExecutor/IConversation appends,
  implementation NativeLoraAdapter/NativeControlVector/NativeTtsSession (new) +
  NativeInferenceModel/NativeInferenceContext/NativeStandardExecutor/NativeConversation appends,
  engines IGgufEngineOps/GgufEngineOps facade view: Rerank/PromptWithAudio/WithLora/ClearLora/
  WithControlVector/CreateTtsSession/GetExecutorStats/GetConversationStats, tests
  Wave3FeatureTests (new) + EngineTestDoubles/StallWatchdog fake appends + SmokeTests vision
  hybrid-pair fix), docs (ARCHITECTURE ×2 → ABI v8).
- KV-neutrality (owner constraint, argued per feature): audio = image mechanics (positions
  advance, mirror rows added; no KV-op changes); TTS = separate decode context, zero
  conversation interaction; rerank = single-shot fresh seq per pair; LoRA + cvec =
  weights-surface modifiers (upstream semantics: existing KV content stays valid, no context
  recreation required — gated live: attach/detach mid-conversation on ONE context via state
  save/restore, detached continuation == baseline continuation); stats = read-only snapshot.
- Acceptance / Status: **done** 2026-10-18 — branch wave1-upstream-parity, HEAD child of 88f8c1a.
  Native Release build 0 errors; native suite **43/43 passed, 0 skipped** with standard fixtures
  (Llama-3.2-3B dense, Qwen3.5-4B hybrid + mmproj, Llama-1B draft, MiniLM embed) + all nine new
  legs (rerank_typed / rerank_live / audio_capability_typed / audio_live / tts_live / lora_typed
  / lora_live / control_vector / stats_export); ngram 23/23; stress 31/31; EAGLE3 live legs
  green (PVX_EAGLE3_TARGET_PATH/PVX_DRAFT_PATH: Qwen3-8B pair — "live: eagle3 exact 24/24
  tokens, drafted=95 accepted=9 (accept-rate 9.5%)"). KV/context regression sweep (all green,
  from the 43/43 run): [3] partial_rewind_capability, [4] partial_rewind_refused_on_hybrid,
  [6] pool, [9] state_save_restore, [12] pool_reuse, [14] shift_left, [15] kv_copy,
  [16] causal_attn, [28] conversation_spec_rewind_continue. Wave-1 draft+vision harness
  rebuilt + **BYTE-EXACT 24/24** (Qwen3.5-4B-PVXLLM + mmproj + Qwen3.5-0.8B draft). C# build
  Debug+Release 0 errors / suite **219/219** (208 baseline + 11 new) with ALL wave-3 live legs
  env-set; SmokeTests vision legs pinned to the hybrid pair (env-gating mismatch fix).
  LDC regenerated + enforcement **PASS both roots** (repo + csharp; 1888 types indexed).
  Provisioning (owner-sanctioned ≤ 8 GB; ~4.07 GB used; range-request header prechecks BEFORE
  download, byte-size verified after, RAM-gated loads): audio pair
  ggml-org/Qwen3-ASR-1.7B-GGUF (Qwen3-ASR-1.7B-Q8_0.gguf 2165034944 B + mmproj-...-Q8_0.gguf
  355709344 B; clip.has_audio_encoder=true, qwen3a projector), TTS pair
  ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF (Qwen3-TTS-12Hz-1.7B-Base-Q4_K_M.gguf 1035965280 B +
  mmproj-...-Q8_0.gguf 446422912 B; clip.has_gen_audio_encoder=true, QWEN3TTS), LoRA adapter
  Leejy0-0/Llama-3.2-3b-lora_model-F16-GGUF (Llama-3.2-3b-lora_model-f16.gguf 97282976 B;
  general.type=adapter, adapter.type=lora, arch llama — matches the Llama-3.2-3B fixture),
  reranker gpustack/jina-reranker-v1-tiny-en-GGUF (jina-reranker-v1-tiny-en-Q8_0.gguf 36676960 B;
  cls.weight 384x1 score head). Control vectors: NO download — synthetic direction.GGUF written
  by the test via the stable gguf writer API (tests-only; product loader is a pattern-port).
  Live evidence: rerank scores 0.1101 / 0.0029 / 0.0449 (relevant docs > distractor, replay
  bit-stable); ASR native rate 16000 Hz both lanes + post-audio greedy continuation;
  TTS 26880 samples @ 24000 Hz, seed-42 replay byte-identical on fresh session + random-seed leg;
  LoRA/cvec attach/detach: prompt-greedy=35308, baseline-cont=927, lora-cont=927,
  detached-cont=927 (detach == baseline at the same anchor KV, single context, no reset);
  stats counters consistent (n_tokens == n_kv_pos == n_prompt_tokens, decode_steps ≥ 1, spec 0/0).
- Honest findings (documented, no band-aids): (a) TTS single-shot contract — repeated generates
  on one live session shift the output (cross-call gen-pipeline state inside the mtmd context);
  typed `tts/session-consumed` refusal + fresh-session determinism is the shipped contract.
  (b) UPSTREAM LEAK (llama.cpp frozen, not fixed here): mtmd_init_from_file of a vision mmproj
  against the qwen3tts backbone fails deep and leaks a Metal buffer set → exit-time
  GGML_ASSERT([rsets->data count] == 0) abort; the suite deliberately avoids that combination
  (the negative leg uses the audio-only mmproj → clean typed NOT_SUPPORTED); repro is minimal
  (load pair, gen, exit). (c) rerank gate is the pooling TYPE (upstream --reranking contract);
  `llama_model_n_embd_out` is NOT a rerank capability signal (the jina conversion lacks
  embedding_length_out; score-head presence (cls.weight) is the GGUF conversion contract,
  verified at provisioning). (d) Plan deviation: shipped lane-specific prompt-with-audio
  exports instead of the generalized `pvx_media_chunk_t` union — the vision path stays
  ABI-stable untouched (additive-only wave constraint); EncoderEngineOps untouched (images-only,
  audio rides the Gguf ops view) per plan.

- Audit evidence (2026-10-18 comprehensive audit): wave-3 surfaces fixed in the same audit
  batch (commits 2c521e2 + 356a366): (1) pvx_rerank leaked llama_set_embeddings(true) into every
  subsequent decode on the shared context on ALL exit paths incl. success — RAII-balanced now
  (pvx_get_embeddings discipline); (2) conversation-lane vision errors were swallowed
  (set_error(nullptr)) — routed to the conversation's context scope; the lane also lacked the
  executor twin's fail-closed on a projector that yields no usable embd rows (silently advanced
  nothing) — now typed DECODE_FAILED; (3) pvx_load_mmproj's second-encoder refusal path leaked
  the freshly created mtmd context (no dtor) — freed explicitly; (4) pvx_pool_return kept
  stale recent_tokens across leases (cross-lease penalty contamination) — cleared (fresh lease =
  fresh penalty history); (5) pvx_tts raw stderr diagnostics routed through pvx_log. C# wrapper
  audit: interop delegate/signature table verified 1:1 vs native ABI v8 (PvxStats field order,
  PvxTtsParams layout, rerank/audio HGlobal pinning freed on every path, SafeHandle releases
  match frees, reverse-pinvoke GC.KeepAlive pattern present); engine facade verified
  fail-closed (builder→request validation, factory kind routing, mismatch views, disposal
  chain); no facade changes needed. Full regression: native 44/44 + ngram 23/23 + stress 31/31,
  C# build 0/0 Debug+Release, C# suite green (see run log), live eagle3/dflash pairs green,
  LDC enforcement unaffected (internal fixes only — public surface unchanged).

### Ongoing verification duty
- Every repo state keeps: native Release build 0 errors, ctest green, C# suite green, LDC regenerated + enforcement PASS. Any public-surface change regenerates LDC artifacts in the SAME commit.

**History note (2026-10-18):** the repo was re-initialized on this date to reflect the as-is
product state (llama.cpp/GGUF-only). Completed historical backlog entries not relevant to the
as-is product were retired; pre-reset history is archived outside every repo and is NOT
recoverable from this git.

### BL-38 · Separate-head draft-MTP spec mode (ABI v8 wave 4) — INF-04 Inference slice (owner order 2026-10-19 17:00)
- Source: INF-04 (owner order 2026-10-19 17:00 "I want mtp to work" — Qwen3.8-Flash-Next on the large-memory host serving).
- Spec: port the vendored `common_speculative_impl_draft_mtp` (upstream PR #28243) as `PVX_SPEC_MTP_DRAFT` (mode 9) + `mtp_draft_path` — the separate self-contained MTP head GGUF. EAGLE3-lane round structure: per-position `h_hist` capture hook (`mtp_process_target_rows`) after every target decode, `(t_{P-1}, h_{P-1})` pairing (zero-h at position 0), seed/chain pair-decodes, ONE verify decode + sample-and-match, KV repair, state save/restore additive h rows. Executor lane only (conversation lane refuses typed); empty-stream enable requirement; `n_layer_nextn(head)==0` and width/vocab mismatches are typed refusals. `PVX_SPEC_MTP_AUTO` (embedded) stays a typed wiring boundary — unchanged.
- Scope: include/pvxinference.h, include/pvx_internal.h, src/pvx_spec_engine.{h,cpp}, src/pvxinference.cpp, csharp bindings (PvxSpecMode, PvxSpecParams, SpeculativeMode, SpeculativeConfig, 3 marshal sites), tests/test_basic.cpp, ARCHITECTURE.md, LDC artifacts. NOTHING outside.
- Acceptance: native suite legs (typed fail-closed: missing path, metadata-only head, wrong-arch head, conversation-lane refusal, empty-stream refusal; env-gated live leg PVX_MTP_TARGET_PATH/PVX_MTP_DRAFT_PATH — typed skip off-pair) + C# binding/marshaling tests; Debug+Release 0 errors/0 warnings; LDC regen + enforcement same commit.
- Status: done 2026-10-19 (local commit, no push). Live the large-memory host verification (160GB pair) = orchestrator.

### BL-39 · Spec+vision same-session: MTP capture hook embedding-row pairing (owner order 2026-10-20 08:34)
- Source: 2026-10-19 session backlog (memory/2026-10-20.md); owner order "do the open backlog items now".
- Spec: make speculative decoding + vision work in the SAME session: the draft-MTP per-position `h_hist` capture hook (`mtp_process_target_rows`) runs after every target decode and must capture ONLY transformer rows — when a vision chunk (image embd rows) rode the batch, row indices shift and the hook must not pair image-embd positions as (t,h) history (zero-h at pos 0 only, never vision rows). Typed refusal for mixed spec+vision in one lane is acceptable ONLY if pairing cannot be made exact — but exactness is preferred (upstream speculative.cpp pairs on target text decode; vision tokens are inputs, not sampled positions).
- Scope: src/pvx_spec_engine.cpp, src/pvxinference.cpp, include/pvx_internal.h, tests (native vision+spec leg), ARCHITECTURE.md, LDC artifacts. NOTHING outside.
- Acceptance: native test proving spec ON == OFF token-exact on a vision-then-text session (env-gated live leg) or a typed fail-closed refusal with a clear native error; regression: existing native suite green, C# suite green, LDC regen + enforcement PASS.
- Status: done 2026-10-20 (local commit, no push). Vision+MTP live leg env-gated (PVX_MTP_TARGET_PATH/PVX_MTP_DRAFT_PATH/PVX_MMPROJ_PATH/PVX_TEST_IMAGE — typed skip off-pair; no vision+MTP pair on this host — the orchestrator runs it where one exists).

### BL-40 · Spec+audio same-session: MTP audio embd-row pairing (owner order 2026-10-20 09:13)
- Source: BL-39 follow-up (owner order 2026-10-20 09:13 "yes do also for audio"); same embd-row class as vision.
- Spec: audio prompts currently refuse typed in draft-MTP sessions. Apply the BL-39 mechanism (commit b08700d): audio embd rows must ride the upstream input arm feeding the MTP head, h-history capture hook pairing only on mirror/text commit sites, per-audio-chunk decode commits, and mtp_state_restore replaying audio rows — removing the typed audio refusal in MTP sessions. Exactness required, not refusal.
- Scope: src/pvx_spec_engine.cpp, src/pvxinference.cpp, include/pvx_internal.h, tests (native audio+spec leg, env-gated live leg), ARCHITECTURE.md, LDC artifacts. NOTHING outside.
- Acceptance: native audio+spec leg deterministic OR token-exact live (env-gated); typed refusal gone; regression native + C# suites green; LDC regen + enforcement PASS same commit.
- Status: done 2026-10-20 (local commit, no push). Audio+MTP live leg env-gated (PVX_MTP_TARGET_PATH/PVX_MTP_DRAFT_PATH/PVX_AUDIO_MMPROJ_PATH — typed skip off-pair/not audio-capable; no audio+MTP pair on this host — the orchestrator runs it where one exists).

### BL-41 · control-vector test fixture de-hardcoding (owner order 2026-10-20 09:50 "nothing should be hardcoded please fix")
- Source: BL-40 follow-up note (pre-existing failure surfaced by the 2026-10-20 runs); owner order.
- Spec: test_control_vector hardcoded Llama-3.2-3B dims (3072/28) against the default Qwen3-8B model path → deterministic load failure in build-metal-rel3. Fixture now derives BOTH dims at runtime from the model GGUF (<arch>.embedding_length/.block_count scanned by suffix), wrong-width file derived (n_embd/2), out-of-range layer end derived (n_layer+1) — no constants, model-agnostic.
- Acceptance: pvxinference_tests green against Qwen3-8B AND Llama-3.2-3B (48/48 each); ngram 25/25; stress 31/31; LDC regen + enforcement PASS on merged main.
- Status: **done** 2026-10-20 10:05 — main d8a122a (fixture) + LDC post-merge regen; local only, no push.

### BL-39 live verification follow-up (2026-10-20 10:50) — DONE, root-caused
- the large-memory host live run (134.7GB Q5_K_M-mtp pair) initially failed the vision+MTP leg: token-exact but accepted=0. Instrumented spike probe (per-round h-norm/chain/match prints) proved the pairing is CORRECT — root cause: the leg's RAW un-templated prompt degenerated the target into a near-tied argmax between ADJACENT end-marker specials (248044/248045); the head picks the sibling, greedy-exact verification rejects every draft (legitimate behavior, not corruption). Fix: both live legs (vision+MTP, audio+MTP) apply the model chat template (pvx_apply_chat_template, thinking=0) — commit 0d971f6.
- LIVE RESULT after fix (the large-memory host, prod pair): text 24/24 exact accepted 11/26 (42.3%); vision 12/12 exact accepted 5/12 (41.7%); audio leg typed-skips (mtmd_support_audio false — flashnext mmproj is vision-only; audio+MTP exactness stands on the deterministic unit leg until an audio-capable pair exists). Suite 48/48 on the large-memory host.
- Status: **done** (live-verified 2026-10-20). Local only, no push.

### BL-42 · Shared-memory draft-MTP heads — gemma4-assistant support (owner order 2026-10-20 12:00)
- Source: owner order "I dont get what the issue is with our engine and MTP. Can you adjust it so it works and supports it."
- Spec: gemma-4's MTP head (gemma4-assistant arch) SHARES memory with the target trunk (upstream ctx_other at head init; our engine created heads standalone → typed "mtp-head-context" refusal). Fix: always link cp.ctx_other = ctx_tgt at head-context creation (upstream-faithful); detect is_mem_shared via llama_get_ctx_other; shared heads skip catch-up decodes + head seq_rm entirely, draft rows all ride the same position (upstream is_mem_shared draft()), state restore replays only the h ring.
- Survey (owner question): three MTP behavioral classes in the vendored tree — (1) plain self-contained heads (qwen4exp, qwen35/moe, deepseek2/32/4, glm4-moe/glm5-next, qwen3next, nemotron-h-moe, mimo2, bailingmoe3, cohere2moe, hy-v3, granite-switch, glm-dsa: supported, live-proven); (2) shared-memory (gemma4-assistant: NOW supported, BL-42); (3) multi-block chain heads (step35 class, n_layer_nextn > 1: still typed-refused — needs nextn_layer_offset wiring, separate slice).
- Scope: src/pvx_spec_engine.h/.cpp, tests/test_basic.cpp (PVX_MTP_MMPROJ_PATH fixture override), ARCHITECTURE.md, LDC artifacts. NOTHING outside.
- Acceptance: full native suite green incl. the three env-gated live legs on the gemma-4 pair (text/vision/audio+MTP); ngram/stress/C# spec suites green; LDC regen + enforcement PASS.
- Status: **done** 2026-10-20 — LIVE: text 24/24 exact 16/16 accepted (100%); vision 12/12 exact 3/15 (20%); audio 12/12 exact 6/9 (66.7%, FIRST audio+MTP live proof). Local only, no push.

### BL-21 · Full llama.cpp config-surface parity through C# — umbrella (owner order 2026-10-20 13:12)
- Source: owner order "Register as backlog and start doing it we need to capture all" — config survey 2026-10-20 13:06 found ~70 llama.cpp verbs vs ~21 engine verbs (~35% coverage).
- Spec: NO hidden states — every llama.cpp configuration verb reachable from C# (engine structs → ProvostXInference P/Invoke → LLM ModelConfig/request). ABI v8 wave-5: APPENDED fields only (default = current behavior bit-identical).
- Sub-slices:
  - BL-21a sampling: seed (deterministic), frequency_penalty, typical_p, xtc (p,t), top_n_sigma, mirostat (mode,tau,eta), dry (multiplier/base/penalty/seq_last_n), logit_bias (token→bias pairs) — wired into the pooled selection chain in OpenAI order.
  - BL-21b model-load: load_mtp, vocab_only, check_tensors, use_extra_bufts, no_host, no_alloc, lazy_mode, kv_overrides (GGUF metadata override escape hatch, typed {key,type,value}).
  - BL-21c context: ubatch_size, n_outputs_max(_per_seq), threads_batch, rope_freq_base/scale, rope_scaling_type, yarn_* (ext/attn/beta_fast/beta_slow/orig_ctx), separate type_k/type_v, moe_cache_size, offload_kqv, op_offload, swa_full, kv_unified, embeddings, no_perf, attention_type, defrag_thold.
  - BL-21d C#/server: P/Invoke struct extensions, ModelConfig keys, request params (seed/frequency_penalty OpenAI-relevant), typed clamps, tests.
- Acceptance: defaults-equivalence (full native suite + C# suites green with no env changes), new-knob unit legs, LDC regen + enforcement PASS, live spot-check on prod pair.
- Status: **done** 2026-10-20 — native 48/48 + ngram 25/25 + stress 31/31 with zeroed wave-5 (defaults equivalence proven); C# 230/230; ABI pins 72/112/112/72/8/288; facade WithAdvanced pass-through; LLM layer + live checks under the LLM repo's BL-21.

### BL-43 · Linux ARM64 RID portability (verification finding 2026-10-20)
- Source: Linux ARM64 verification of the LLM repo (post BL-22) — native INF built clean on the VM (cmake Release, CPU backend, 0 errors), but the C# probe layer hardcodes `linux-x64` for ALL Linux: `Interop/PvxNative.cs` static ctor probes `runtimes/linux-x64/native/libpvxinference.so` and `Models/PvxEngineFileResolver.cs` derives RID `linux-x64` even on arm64 hosts (no RuntimeInformation.ProcessArchitecture check). On the arm64 VM the freshly built arm64 `.so` had to be dropped into the `linux-x64` native dir for the probe to find it — wrong-dir deployment, not a real x64 binary.
- Related: the packaged `runtimes/linux-x64/native` ORT payloads (libonnxruntime.so etc.) are prebuilt x64 ELF binaries — unusable on arm64 Linux; arm64 hosts need source-built ORT or a linux-arm64 package payload.
- Spec: derive the native dir RID from ProcessArchitecture (linux-arm64 + linux-x64) in the probe + resolver; ship or document an arm64 ORT payload story. No behavior change on existing x64 Linux or macOS/Windows paths.
- Scope: csharp/Interop/PvxNative.cs, csharp/Models/PvxEngineFileResolver.cs, csharp/Tests (RID expectations), ARCHITECTURE.md, LDC artifacts. NOTHING outside.
- Acceptance: unit tests proving arm64 RID derivation (host-overridable), native probe finds `.so` under `linux-arm64` on an arm64 Linux host, existing x64 path unchanged, LDC regen + enforcement PASS.
- Status: **done** 2026-10-20 — arch-aware Linux RID (IOSProbe.Architecture seam + PvxNative ProcessArchitecture derive); macOS/Windows unchanged; full C# suite 234/234; live-artifact: the Linux VM build tree (arch-correct .so dir) picks up natively on next sync. ORT arm64 payload story: documented (source-built ORT needed; resolver now finds it in linux-arm64).

### BL-44 · Windows ARM64 MSVC portability (verification finding 2026-10-20)
- Source: Windows 11 ARM64 VM verification of the native build (first-ever MSVC pass, VS 2022 BuildTools 14.44 / ARM64, CMake "Visual Studio 17 2022" generator).
- Findings & fixes (landed same day, this item records the set):
  (1) missing `<functional>` include (libc++ transitive, MSVC not); (2) POSIX `localtime_r` → `localtime_s` under `_WIN32`; (3) designated initializers under /std:c++17 → lambda-built constant; (4) raw strings inside ASSERT macro args (MSVC legacy preprocessor stringize limitation); (5) `M_PI` needs `_USE_MATH_DEFINES`; (6) DLL exported NOTHING (no dllexport in the extern "C" header) → `WINDOWS_EXPORT_ALL_SYMBOLS ON` restores macOS-parity export surface + import lib; (7) Windows import libs don't re-export transitive static-lib symbols → WIN32 test links to llama/ggml/ggml-base.
- Windows VM results: `pvxinference.dll` + all 3 test exes build; ngram 25/25, basic 48/48 (41 model-gated skips), stress built (needs model files). the primary host gate: 48/48 + 25/25 + 31/31 unchanged.
- Follow-ups (closed by BL-46 same day): win-arm64 runtime payload + Windows CI lane (see BL-46). The LLM repo's native-gated/model-gated E2E tiers remain unproven on Windows until a model is present on a Windows host (model-gated, by design).
- Status: **done** 2026-10-20 (fixes merged with this item); follow-ups closed by BL-46 (payload + CI lane).

### BL-45 · Multi-block chain draft-MTP heads — step35 class (owner order 2026-10-20 19:32)
- Source: BL-42 survey follow-up (owner order); last typed-refused MTP family — heads whose GGUF carries n_layer_nextn > 1 (llama.cpp drives them via llama_set_nextn_layer_offset).
- Spec: upstream SUPPORTS multi-block chain heads (speculative.cpp: chain_heads = n_mtp_layers > 1 && !is_mem_shared; draft() step i decodes the whole re-added prefix under nextn layer offset i with per-step seq_rm(pos0,-1) and logits only on the last row; process() replays the catch-up batch once per head; offset restored to 0 after both). Wire exactly that: chain detection at head init, draft_max clamped to the trained block count, catch-up/replay per head, chain drafting per head; n_nextn==1 and shared-memory paths byte-identical to BL-42. The typed refusal stays ONLY where upstream itself refuses (none found for this family — support replaces the refusal).
- Scope: src/pvx_spec_engine.h/.cpp, tests/test_ngram.cpp (pure planner legs), ARCHITECTURE.md, LDC artifacts. NOTHING outside.
- Acceptance: deterministic pure-planner unit legs (chain plan + per-step arm); env-gated live leg if a multi-block pair exists (none on this host — unit legs + source-proven semantics); n_nextn==1 regression paths unchanged; native 48/48 + ngram 25/25 + stress 31/31; C# suite green; LDC regen + enforcement PASS.
- Status: **done** 2026-10-20 — upstream SUPPORTS the family (speculative.cpp chain_heads path; no refusal exists to keep); chain detection + per-head catch-up + per-step offset drafting wired upstream-exact; n_nextn==1 and shared paths byte-identical; pure-planner legs ngram 27/27; native 48/48; stress 31/31; C# 234/234; LDC PASS. No multi-block pair on this host — live leg deferred to where a step35-class pair exists (env-gated fixtures unchanged). Local only, no push.

### BL-46 · Windows ARM64 runtime payload + Windows CI lane (BL-23/BL-44 follow-ups, 2026-10-20)
- Source: the two registered follow-ups from the Windows ARM64 VM verification — (a) no win-arm64 native runtime payload in the NuGet runtimes dir (resolver Windows path hardcoded win-x64); (b) no Windows CI lane (the MSVC pass was VM-manual).
- Landed:
  (1) Windows RID is now arch-aware in BOTH the resolver (`PvxEngineFileResolver`: `win-{Architecture()}`) and the bootstrap probe (`PvxNative` static ctor derives from ProcessArchitecture) — mirrors the BL-43 Linux rule; `PvxRidDerivationTests` covers win-arm64/win-x64 (C# suite 234→235).
  (2) Payload shipped: `csharp/runtimes/win-arm64/native/pvxinference.dll` + `pvxinference.lib` (MSVC import lib) — REAL binaries copied from the VM build (VS 2022 BuildTools 14.44, CMake VS generator -A ARM64, Release, CPU/default specialist, ABI v8; provenance in `csharp/runtimes/win-arm64/README.md`). The csproj `runtimes/**` glob packs them into the NuGet slice automatically.
  (3) Windows CI lane: ci.yml new `windows` job (windows-latest) — native cmake MSVC Release build + ctest (model-gated legs typed-skip without models — same as the VM run) + the C# suite, with the fresh x64 dll staged into `csharp/runtimes/win-x64/native/` first so the arch-aware resolver finds it. Existing specialist/csharp-build/nuget-pack lanes untouched. actionlint clean.
  (4) Test-infra fix found by the gate itself: `pvx_stress_tests` had no `WORKING_DIRECTORY` — it silently ran from the build dir under ctest and couldn't find `./models` (the binary worked when invoked directly, which is why earlier gates passed); now pinned to `CMAKE_SOURCE_DIR` like the basic suite.
- Per-RID payload story (ARCHITECTURE.md addendum): osx-arm64 = checked-in dylibs (inference + metal); win-arm64 = checked-in VM-built dll + import lib; linux-x64/win-x64 = CI specialist matrix outputs at pack time (not checked in); linux-arm64 = source-build native + ORT (documented, BL-43).
- Gates (the primary host, clone): native 48/48 (+6 typed live-skips) + ngram 27/27 (incl. BL-45 chain legs) + stress 31/31 via ctest (post WORKING_DIRECTORY fix); C# 235/235 (env-set model paths, matched Qwen3.5-4B pair); LLM 924/924. LDC regen on merged main. Local only, no push/tags.
- Still open (model-gated): Windows E2E tiers need a model on a Windows host — CI intentionally runs model-free only (no env vars, no model artifacts by design).
- Status: **done** 2026-10-20.

### BL-47 · Automated real-model journey tier + cross-platform fixture contract (owner order 2026-10-22 07:53)
- Source: owner review of test coverage — the TEST-MATRIX real-model lanes were manual-only; owner ordered: small real model as the standard test fixture, all testing gaps closed, tests cross-platform.
- Landed:
  (1) `csharp/Tests/LiveModelFixtures.cs` — env-driven fixture discovery with repo-relative defaults (small 1B dense model, draft pair, MiniLM encoder, mmproj+hybrid backbone); RAM gate helper (house >60% rule); `LiveModelFactAttribute` = named skip on every OS when the fixture is absent (never a silent pass, never a platform-only unconditional skip).
  (2) `csharp/Tests/RealModelJourneyTests.cs` — 10 automated real-model journey lanes: load+generate, KV save/restore + checkpoint export/import round-trip, 8-conversation continuous batching, GBNF grammar-constrained sampling, dense MEAN-pooling embeddings (SIGSEGV regression guard) + MiniLM encoder similarity, speculative draft-simple token-exact vs the plain loop, vision mmproj splice, greedy determinism, model/tokenizer hygiene (catches mislabeled-file vocab mismatches).
  (3) Fixture repair: `Tests/models/spec-draft-1b.gguf` symlink was DANGLING (target `~/models/spec-test/` deleted) — retargeted to the canonical 1B file; `Tests/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf` symlink added so the tier resolves without env vars.
  (4) Pre-existing test bug fixed: `PvxEngineSelectorTests.UnknownEngine_FailClosed` still listed "cpu" as unknown after caba7d1 added the cpu engine id to the specialist matrix — row removed, positive regression guard added (`CpuEngine_IsKnown_SinceSpecialistMatrixAddition`).
- Verification (the primary host, Debug, PvxUseProjectRefs): full journey tier 10/10 with real fixtures (Llama-3.2-1B target+draft, MiniLM); full C# suite 247/247 WITH fixtures; 247/247 HERMETIC with all env unset (fixture defaults resolve from repo). Journey lanes ride GpuLayers=0 (CPU) → identical behavior on macOS/Linux/Windows by construction; the vision lane pairs the checked-in mmproj with its hybrid backbone.
- Scope: csharp/Tests (2 new files, selector test fix, fixture symlinks), ARCHITECTURE.md, LDC artifacts. NOTHING outside.
- Acceptance: journey tier green with and without fixtures on this host; cross-platform contract = env-driven discovery + typed skips (CI lanes run the suite model-free, skipped-not-failed); native suites untouched and unchanged.
- Still open (model-gated, unchanged): native CUDA/Vulkan specialists on real models remain unproven until a Windows/Linux host carries a model; MTP end-to-end still awaits a step35-class GGUF pair (BL-45 note).
- Status: **done** 2026-10-22. Local only, no push/tags.
