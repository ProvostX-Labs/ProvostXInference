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

### Ongoing verification duty
- Every repo state keeps: native Release build 0 errors, ctest green, C# suite green, LDC regenerated + enforcement PASS. Any public-surface change regenerates LDC artifacts in the SAME commit.

**History note (2026-10-18):** the repo was re-initialized on this date to reflect the as-is
product state (llama.cpp/GGUF-only). Completed historical backlog entries not relevant to the
as-is product were retired; pre-reset history is archived outside every repo and is NOT
recoverable from this git.
