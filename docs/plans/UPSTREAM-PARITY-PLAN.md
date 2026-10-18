# UPSTREAM-PARITY-PLAN — ProvostXInference vs vendored llama.cpp

**Status:** planning document only — no product code changed. No commits, no tags, no pushes.
**Baseline:** 2026-10-18 as-is repo state (ABI v5, ctest green Release/metal, C# 189/189, LDC PASS).
**Added:** 2026-10-18

## Executive summary

Bring every missing upstream capability into ProvostXInference by **porting patterns from the
frozen vendored tree** (`llama.cpp/common`, `llama.cpp/tools/mtmd`) into `src/` behind **stable
llama.h/mtmd.h APIs** — never by running or linking the examples' runtime code. The native core
grows **per-feature spec implementations** that extend the existing `PvxBLSpecEngine`
(`src/pvx_spec_engine.{h,cpp}`) rather than replacing it. Every new capability enters through
three waves of **additive ABI bumps** (v6/v7/v8), each handshake-checked fail-closed against the
C# `ExpectedAbiVersion`. The **first and most important change is the embd-aware batch/mirror
refactor** — `common/speculative.cpp`'s token/embd row model (`LLAMA_TOKEN_NULL` + embd pointer,
zero rows + warning on draft-width mismatch, M-RoPE pinned positions) — because it is the shared
substrate for spec+vision, n-gram, DFlash/DExtract, and EAGLE. KV-cache and context-management
surfaces (`pvx_kv_*`, rewind/shift, save/restore/export) are **untouched**: every feature below is
argued to be KV-neutral or protected. Fail-closed philosophy everywhere: unsupported boundaries
return typed `PVX_ERR_NOT_SUPPORTED` naming the exact boundary; no env vars in product code;
facade (csharp/Engines) stays the only caller surface.

---

## Rules of engagement (binding for every phase)

1. **Port patterns, never runtimes.** Code in `llama.cpp/common`, `llama.cpp/tools`,
   `llama.cpp/examples` is *reference reading material*. Everything we ship lives in `src/`
   (native) or `csharp/` (wrapper). We never `#include "common/..."` from the frozen tree, never
   link its TUs, never call `common_*`/`params`-based runtimes.
2. **Frozen tree is read-only.** No edits inside `llama.cpp/` ever, including whitespace
   fixes. If a pattern is wrong upstream, we diverge in `src/` with a comment explaining why.
3. **Only stable API surface.** Native code calls only symbols exported from
   `llama.cpp/include/llama.h`, `llama.cpp/tools/mtmd/mtmd.h`, `mtmd-helper.h`, clip/mtmd
   headers that are part of the MTMD public API surface. Anything else (internal `llama-*`
   private structs) is off-limits; if a feature needs an internal API, it stays OUT of scope
   until upstream stabilizes it (documented in the risk register).
4. **Every generation-affecting feature ships a token-exactness gate.** A test that runs the
   same prompt/sampling through the new path and the plain greedy/non-spec path and asserts
   token-for-token equality (the existing spec-loop contract: "spec and non-spec generation
   agree token-for-token for identical params"). No gate, no merge.
5. **Every feature typed fail-closed.** New boundary → new `PVX_ERR_*` code + reason string via
   the existing error channels (`pvx_last_error` / `pvx_model_last_error`). Never silent
   fallback, never best-effort zeros.
6. **KV/context management is load-bearing — protect it.** `pvx_kv_*`, rewind/shift guards
   (`PVX_ERR_REWIND_UNSUPPORTED` on `llama_memory_seq_rm` refusal), save/restore/export state
   chains, and the lock order (pool → infer) are preserved verbatim. Any feature that cannot
   coexist with them is deferred, not hacked in.
7. **ABI is additive + handshake-bumped per wave.** No existing struct/layout change; new
   exports and new `PVX_*` version macro values per wave; C# `ExpectedAbiVersion` bumps in the
   same commit.
8. **C# rules unchanged:** one type per file, constructor injection, no statics (sanctioned
   `PvxNative`/`PvxDispatchRegistry`/factory statics only), no env vars, UTF-8 marshalling,
   SafeHandle per native resource, LDC regeneration in the same commit.
9. **Facade is the only caller surface.** ProvostXLLM never sees Interop/Abstractions/
   Implementation types; new capability = new ops/view method + request/config record, not a
   new assembly boundary.
10. **Docs are anonymous:** relative paths only; no hostnames, hardware names, or usernames.

---

## Current state (verified on disk, 2026-10-18)

- ABI v5 in `include/pvxinference.h` (584 lines, 70+ exports listed there); handshake against
  C# `ExpectedAbiVersion` at engine open.
- Spec: `src/pvx_spec_engine.{h,cpp}` (119/516 lines) — `PvxBLSpecEngine` supports
  `draft-simple` (second GGUF) + MTP via target model's own nextn tensors. Supported-types
  list (`pvx_spec_capabilities_t` bits, `PVX_SPEC_TYPE_DRAFT_SIMPLE 0x1`) — eagle3/dflash/
  dspark named in the header comment as per-target companion GGUFs, **not yet wired**.
- Spec loop mirrors target state by replaying plain token ids from `exec->recent_tokens`,
  assuming token-count == position-count (`pvx_spec_engine.cpp:185`).
- Vision: `pvx_executor_prompt_with_images` / `pvx_conversation_prompt_with_images`; a second
  mmproj load refused while a wrapper is alive (refcounted shared mtmd context). Pre-image text
  flush failure = typed decode failure, pending kept.
- Batched lane: `pvx_pool_*` conversation pool, `seq_max` config, `pvx_conversation_*` ops —
  spec decode exists **only on the standard executor** (all `pvx_executor_spec_*`).
- Context/KV: `pvx_kv_clear_seq/copy/flush`, `pvx_kv_seq_tokens`, rewind/shift, save/restore,
  export/import, `pvx_state_save` seq-clears backup before `seq_cp`.
- Vendored tree capabilities verified present: `llama.cpp/common/speculative.cpp` (3115 lines:
  draft-eagle3, draft-dflash, draft-dspark types + ngram-simple, ngram-map-k, ngram-map-k4v,
  ngram-mod, ngram-cache at lines 75–83), `common/ngram-cache.{h,cpp}`, `common/ngram-map.h`,
  `common/ngram-mod.{h,cpp}`, `common/json-schema-to-grammar.{h,cpp}` (1028 lines),
  `tools/mtmd/mtmd.h` (`MTMD_INPUT_CHUNK_TYPE_AUDIO`, `MTMD_GEN_AUDIO_TYPE_QWEN3TTS`,
  `MTMD_GEN_AUDIO_TYPE_POCKETTTS`, `mtmd_bitmap_init_from_audio`,
  `mtmd_get_audio_sample_rate`), `include/llama.h` LoRA adapters
  (`llama_adapter_lora_init*`, `llama_adapter_meta_*`, `llama_set_adapter_lora` line ~731,
  `llama_set_adapter_cvec` line 741), control vectors (`common_control_vector_load` in
  `common/common.cpp:1224`, applied via `llama_set_adapter_cvec` from `common.cpp:1451`),
  `llama_state_seq_get/set_data(_ext)` for per-seq state (EAGLE3 deferred-boundary
  serialize/restore), `ggml` RPC backend (`ggml/src/ggml-rpc`).
- Spike `vision-spec-decode` README was **not found on disk** (`.tmp/` paths absent in the
  workspace); its conclusions as captured in the task brief are treated as requirements input:
  image path advances `n_kv_pos` by `n_img_tokens` **without** appending to `recent_tokens`,
  so the plain-id mirror desyncs → spec+images must currently fail closed typed at the C#
  `SessionContext` level. Verify the refusal seam location during wave 1 design.

---

# Feature plans

Each feature: (a) upstream mechanism + verified vendored references, (b) native ABI, (c) C#
surface, (d) facade surface, (e) KV impact, (f) fail-closed boundaries, (g) tests, (h) size,
(i) tier. Sizes S ≤ ~300 lines native+wrapper, M ~300–900, L ≥ ~900 or multi-system.

---

## 1. Spec decode + vision combined (embd-aware mirror port) — Tier 1

**(a) Upstream mechanism.** `llama.cpp/common/speculative.cpp:313–331`: the target-side batch
builder tracks per-token `(id, embd)` — `t.id != LLAMA_TOKEN_NULL` decides token-vs-embd rows;
`batch.set_embd(idx, t.embd)` for image rows; for draft-width mismatch
(`n_embd != zeros.size()`, line 322–330) it **zero-pads the draft row** with a warning instead
of aborting — the draft consumes a zero row and the target still verifies the real embd. M-RoPE
models pin `t.pos` per-row (windowed drafts use pinned positions, `speculative.cpp` pos handling
in the batch add path).

**(b) Native ABI.** No new exports needed for the core fix — this is a `src/` refactor:
replace `exec->recent_tokens : std::vector<llama_token>` replay with a
`pvx_mirror_entry_t { llama_token id; bool has_embd; size_t embd_off; std::array<int32_t,4> pos; }`
ring (in `pvx_spec_engine.h`), fed by the same places that currently push tokens
(`pvx_spec_engine.cpp:364,456,492-493`) plus `prompt_with_images`'s embd append (the mtmd
decode output already lands via `mtmd` callback — we add the row bookkeeping there). The mirror
builds drafts with `llama_batch` embd rows via `llama_batch` + `llama_batch_add` — stable API.
Image tokens in the mirror carry `LLAMA_TOKEN_NULL` + stored embd slice. ABI stays v5; this
ships inside wave-1's build.

**(c) C#.** No interop change; `SpeculativeConfig` gains `AllowVision` flag (default false →
the existing typed refusal moves from "always" to "only when unset or model lacks vision"), and
`ActiveSpeculation` mirror record gains `HasVisionSupport` bit from capabilities.

**(d) Facade.** `IGgufEngineOps`: speculation validate/enable already exist; add
`SpeculationSupportsVision` to the `ActiveSpeculation` mirror record.

**(e) KV impact — neutral.** The mirror is draft-model-side state only; target KV positions
are unchanged (image embd already occupies `n_img_tokens` positions today). Draft context must
simply reproduce positions the target already committed — no new KV ops.

**(f) Fail-closed.** Draft model lacking vision (no mmproj wired for it) while target drafts
with images → `PVX_ERR_NOT_SUPPORTED` "draft-vision: target uses image embd rows but draft model
has no vision support". Draft-width mismatch is upstream-warned zero-pad — we keep the warning
(via `PVX_LOG WARN`) but never abort generation.

**(g) Tests.** Token-exactness gate: multimodal prompt (one image), greedy sampling, spec-on vs
spec-off, identical token stream. Desync regression test: image + n_img_tokens bookkeeping
asserts mirror length == n_kv_pos after every step. Negative tests: vision draft without
vision-capable draft model → typed error.

**(h)** M. **(i)** Tier 1 — **first**, shared substrate for #2/#3/#4.

---

## 2. n-gram speculative decoding — Tier 1

**(a) Upstream.** `common/ngram-cache.h/.cpp` (persistable static n-gram→next-token cache),
`common/ngram-map.h` (dynamic online cache built from recent generation), `common/ngram-mod.h/
.cpp` (MOD pattern — matches (n-1)-gram suffixes allowing one deviating token per prediction;
each strategy registers as `common_speculative_impl_*` per `speculative.cpp:79-83`). Drafts
are produced by cache lookup, not a draft model — no draft context at all.

**(b) Native ABI (wave v6).** New `PVX_SPEC_TYPE_*` bits (0x2 NGRAM_MAP, 0x4 NGRAM_MOD, 0x8
NGRAM_CACHE) merged into `pvx_spec_capabilities_t.types` (additive bitset); `pvx_spec_params_t`
grows a `pvx_ngram_params_t { int n_min; int n_max; float p_min_mod; const char* cache_path; }`
union-field `ngram` (mode `PVX_SPEC_NGRAM` added to `pvx_spec_mode_t`). Cache file IO is a
plain `FILE*` read/write in `src/` (no env vars). No second model → works on any target.

**(c) C#.** `SpeculativeType` gains `NgramMap/NgramMod/NgramCache` entries (already modeled as
enum), `SpeculativeConfig` gains `NgramConfig` record (Validate: n_min ≤ n_max ≥ 1, cache path
exists-or-creatable), `PvxSpecParams` struct extended (blittable, additive fields appended).

**(d) Facade.** `GgufEngineOps.Speculation` record + builder: `.WithNgram(nMin, nMax, cachePath)`.
`ActiveSpeculation` mirror reports active n-gram type + cache stats (hits/misses counters —
feeds #12).

**(e) KV — neutral.** n-gram drafts verify against the target's existing KV exactly like
draft-model drafts; cache lives in RAM/disk outside KV.

**(f) Fail-closed.** Cache file corrupt/mismatched vocab → `PVX_ERR_LOAD_FAILED` naming path;
no draft source for current n → skip-draft-step (existing draft_min behavior), not an error;
empty cache at enable → typed warning, not refusal (cache fills online).

**(g)** Token-exactness gate same as #1. Plus cache round-trip test (save→load→identical draft
decisions). **(h)** M. **(i)** Tier 1.

---

## 3. DFlash / DExtract fused decode — Tier 2

**(a) Upstream.** `speculative.cpp:77` `draft-dflash` (DExtract): instead of a full draft-model
decode, the draft runs **one fused decode** per verify round — a small custom graph injects
the target's hidden-state *feature rows* directly (DExtract: target last-layer features feed
the draft's input projection at the correct layer). Requires target-side feature capture
(`llama_state_seq_*` not needed; upstream uses `llama_batch` output flags + the draft's own
GGUF's `target_layer_ids` metadata, verified pattern in `speculative.cpp` draft-eagle3 section
lines 480–700: features buffer `[n_tokens × target_layer_ids_n × n_embd_tgt]`, EAGLE3 encoder
pass, deferred boundary state).

**(b) Native ABI (wave v7).** `PVX_SPEC_TYPE_DFLASH 0x10` capability bit (companion-GGUF
declared in GGUF metadata — capability discovery already inspects arch metadata, extends
naturally). `pvx_spec_params_t.dflash { const char* companion_path; }`. A **per-target-model
DExtract runner** in `src/` (new `src/pvx_dextract.cpp`, ~400 lines): owns the draft context +
feature-injection decode step. Draft width mismatch → zero-pad row + warning (shared with #1).

**(c) C#.** `SpeculativeType.DFlash`, `SpeculativeConfig.DFlashConfig { CompanionPath }`,
Interop struct append.

**(d) Facade.** `GgufEngineRequest.DraftCompanionPath` sidecar slot already exists as pattern
(draft sidecar) — extend with `SpeculationKind` discriminator; ops mirror reports fused mode.

**(e) KV — protected.** Draft consumes target *hidden features*, not KV; but feature capture
adds one extra target decode step per round — the mirror must keep `n_kv_pos` counting exactly
aligned (feature step advances positions only via the same batch rows). Uses #1's mirror.

**(f) Fail-closed.** Companion GGUF missing DExtract metadata (`target_layer_ids` /
`target_hidden_size` / 3-extract-layers check, upstream line 562) → `PVX_ERR_LOAD_FAILED`
naming the metadata key. Target layer id ≥ target n_layer → same.

**(g)** Token-exactness gate vs non-spec greedy. Companion-metadata negative tests. **(h)** L.
**(i)** Tier 2 (depends on #1).

---

## 4. EAGLE / EAGLE3 draft models — Tier 2

**(a) Upstream.** `speculative.cpp:480–700+`: `common_speculative_impl_draft_eagle3` — draft
decoder consumes `(token, g_embd)` pairs where `g_embd` is the target's concatenated
layer-feature encoder output (`n_embd_enc = target_layer_ids_n × n_embd_tgt`, line 525);
`pending_g_last` deferred-boundary state (lines 499–514) with **serialize/restore via
`llama_state_seq_get/set_data`** (upstream comment line 213) — this interacts directly with our
conversation save/restore! EAGLE1/2 (plain draft-decoder + target last-hidden-state) is the
same row model minus the encoder pass.

**(b) Native ABI (wave v7, shared with #3).** `PVX_SPEC_TYPE_EAGLE3 0x20` bit;
`pvx_spec_params_t.eagle { path }`. Native `src/pvx_spec_engine.cpp` grows an EAGLE3
specialization using #1's mirror entry ring (its `(token, g_embd)` rows are exactly
`pvx_mirror_entry_t` with `has_embd=true`, `embd` = encoder output row).

**(c) C#.** `SpeculativeType.Eagle3`, config record, params struct append.

**(d) Facade.** Speculation builder `.WithEagle3(companionPath)` + mirror fields.

**(e) KV — protected (the one real interaction).** EAGLE3 deferred-boundary state must ride
**inside** `pvx_conversation_save/restore` and `pvx_state_save`: add per-seq EAGLE3 state
ser/de appended after the llama state blob (length-prefixed, versioned), so restore of a
conversation with active EAGLE3 doesn't resume against a stale `pending_g_last`. This touches
the state chain **additively** (append, never reorder); existing checkpoints restore as
"EAGLE3 state absent → session must re-init spec → typed error if enabled without re-init".

**(f) Fail-closed.** Draft GGUF arch mismatch (`llama_model_is_recurrent`/arch family check)
→ `PVX_ERR_LOAD_FAILED`; EAGLE3 with a target that can't expose layer features →
`PVX_ERR_NOT_SUPPORTED` naming boundary.

**(g)** Token-exactness gate; save/restore round-trip WITH active EAGLE3 across restart;
rewind-with-EAGLE3 (see risk register) test.

**(h)** L. **(i)** Tier 2 (depends on #1; #3 shares the feature-capture runner).

---

## 5. Spec decode on the batched/pool lane — Tier 2

**(a) Upstream.** Not an upstream pattern port — upstream `llama-server` runs spec across
multi-seq contexts with per-seq `llama_state_seq_*`; our gap is architectural: spec state
lives on `pvx_executor_t` only.

**(b) Native ABI (wave v7).** `pvx_conversation_spec_init(conversation, params)` /
`pvx_conversation_generate_spec(conversation, ...)` + `pvx_conversation_spec_free` — mirror of
the executor trio, one spec session per conversation (per-seq draft state keyed by the
conversation's seq_id). Pool return frees spec session in the existing teardown order
(pool → infer lock discipline preserved: spec ops take `infer_mtx` same as sample ops).

**(c) C#.** `IConversation` grows spec enable/generate (capability-gated via caps),
`SpeculativeConfig` unchanged; new `NativeConversationSpec` types one-per-file.

**(d) Facade.** `GgufEngineOps` conversation convenience exposes `WithSpeculation` on pooled
conversations; `ActiveSpeculation` reports per-conversation.

**(e) KV — protected.** Draft contexts consume a **separate** KV (own `llama_memory`), target
KV untouched; pool `seq_max` budgeting must account for draft-side KV (context sizing config
gains optional `draft_ctx_size`, default derived from target `n_ctx`).

**(f) Fail-closed.** Spec on pool with `seq_max < 2` → typed NOT_SUPPORTED naming
"batched spec requires ≥2 seq slots"; pool exhaustion mid-draft → existing pool error path.

**(g)** Token-exactness gate per-seq: N parallel conversations, same prompts, spec vs non-spec
identical per-stream. Pool churn stress (existing race-stress suite extended).

**(h)** L. **(i)** Tier 2 (after #1; benefits from #2/#3 as draft sources).

---

## 6. Audio input (MTMD_INPUT_CHUNK_TYPE_AUDIO) — Tier 3

**(a) Upstream.** `tools/mtmd/mtmd.h`: chunk types TEXT/IMAGE/**AUDIO** (lines 55–58);
`mtmd_support_audio(ctx)` capability probe, `mtmd_get_audio_sample_rate` (line 154),
`mtmd_bitmap_init_from_audio(n_samples, float*)` (line 176), `mtmd-audio.{h,cpp}`
adapter/convert helpers, `mtmd_decode_use_mrope` (line 144) — audio rows participate in the
**same embd-row mirror problem** as #1 (M-RoPE pinned positions for audio).

**(b) Native ABI (wave v8).** `pvx_executor_prompt_with_media` generalization: input becomes
a chunk list `pvx_media_chunk_t { PVX_MEDIA_TEXT | PVX_MEDIA_IMAGE | PVX_MEDIA_AUDIO; union }`;
audio chunk = raw float PCM + sample rate. Extends the existing with-images path (image path
preserved as-is for ABI compat). `pvx_mtmd_supports_audio` probe export. All behind the same
refcounted shared-mtmd wrapper discipline (second mmproj load refused while wrapper alive).

**(c) C#.** New `MediaChunk` records (one per kind, one-per-file), `IStandardExecutor.Prompt`
overload with media chunk list, audio `MediaSample` (float[], sampleRate) with ownership rules
documented.

**(d) Facade.** `GgufEngineOps` prompt-with-media; `EncoderEngineOps` unchanged (vision stays
encoder-lane; audio is gguf-lane input).

**(e) KV — neutral (same mechanics as images: n_audio_tokens positions advanced, mirror rows
added per #1).

**(f) Fail-closed.** Model without audio-capable mmproj → `PVX_ERR_NOT_SUPPORTED` naming
"audio input requires a multimodal projector with audio adapter"; sample-rate mismatch with
`mtmd_get_audio_sample_rate` → typed error, no resampling in product code (caller resamples);
audio > context window → existing context-size typed error.

**(g)** Token-exactness gate (same transcript, audio dropped vs non-audio model path); format
negative tests (empty PCM, wrong rate).

**(h)** M–L. **(i)** Tier 3 (needs #1 mirror rows for spec+audio).

---

## 7. Speech output / TTS (mtmd_gen_audio, QWEN3TTS) — Tier 3

**(a) Upstream.** `tools/mtmd/mtmd.h:416–418`: `MTMD_GEN_AUDIO_TYPE_{NONE, QWEN3TTS,
POCKETTTS}` + `mtmd_gen_audio` API + `mtmd-helper-gen.*` helpers. This is an output-lane
capability of the mtmd context.

**(b) Native ABI (wave v8).** `pvx_mtmd_gen_audio_supported`, `pvx_generate_audio(model,
pvx_tts_params_t, text, out pcm float array + sample rate)` — a **separate decode context**
(speech model), never entangled with the conversation context.

**(c) C#.** `GgufEngineOps.Speak(text) → AudioSample` or a dedicated request record; TTS
config in `ModelConfig` only as a separate engine instance (TTS model = own GGUF load).

**(d) Facade.** Optional: keep TTS OUT of EngineKind (no new kind — a TTS GGUF is a Gguf-kind
model with gen-audio capability); ops view exposes `AudioGeneration` view record.

**(e) KV — neutral (separate context, no conversation KV interaction).

**(f) Fail-closed.** Model GGUF without TTS head / no `mtmd_gen_audio` support → typed
NOT_SUPPORTED naming "speech output requires a QWEN3TTS-capable model". No best-effort
text-to-audio hacks.

**(g)** Output is audio, not tokens — gate is **byte-exactness** where deterministic (same
text+seed → identical PCM), plus format sanity (non-empty, finite floats, expected sample
rate). **(h)** M. **(i)** Tier 3 (model availability risk — see register).

---

## 8. Reranking (embeddings + cross-scoring) — Tier 3

**(a) Upstream.** `examples/deepseek-r1`-style rerankers: decoder-encoder pairs where the
**decoder cross-attends the encoder KV** (`llama_model_has_encoder` + rerank arch, upstream
`llama.cpp/examples/` rerank demo). Stable API: `llama_encode` + `llama_decode` on a combined
batch — same architecture routing we already have (`pvx_backend`/embeddings section).

**(b) Native ABI (wave v8).** `pvx_rerank(model, query, doc[], out scores)` on a dedicated
rerank context: encoder pass (query+docs interleaved per upstream reranker layout), decoder
score head read, typed float score per doc. Reuses existing embeddings routing +
`pvx_get_embeddings` plumbing patterns.

**(c) C#.** `IRerankSession`-ish record via `Abstractions` + `Implementation/NativeRerank`;
request record `RerankRequest`.

**(d) Facade.** Stays Gguf-kind: `GgufEngineOps.Rerank(query, docs)`. Capability probed from
model arch (encoder+decoder present → rerank-capable).

**(e) KV — neutral (own context, single-shot, freed immediately).

**(f) Fail-closed.** Encoder-only or decoder-only model → `PVX_ERR_NOT_SUPPORTED` naming
"reranking requires an encoder+decoder architecture" (consistent with existing embeddings
routing refusal). Score head missing → typed error before any decode.

**(g)** Score-determinism gate (same inputs → bit-stable scores, greedy path); negative arch
tests; N-doc stress. **(h)** M. **(i)** Tier 3.

---

## 9. LoRA adapters runtime attach/detach — Tier 3

**(a) Upstream.** `include/llama.h:691–741`: `llama_adapter_lora_init(path)`,
`llama_adapter_lora_init_from_file_ptr`, `llama_adapter_meta_*` (GGUF metadata introspection),
`llama_set_adapter_lora(ctx, adapters[], n)` with per-adapter scale; removal = set with empty
array (0 = detach all). aLoRA variant: `llama_adapter_get_alora_*` (723–724).

**(b) Native ABI (wave v8).** `pvx_adapter_t` opaque handle: `pvx_adapter_load(model, path)`
(fails on arch/vocab mismatch vs model via metadata check before any context use),
`pvx_context_set_lora(context, adapters[], scales[], n)` (0 = detach),
`pvx_adapter_free`. Attaches/detaches on the **live context** — upstream supports it, and our
`infer_mtx` serializes it against decodes (adapter set is a memory-surface op).

**(c) C#.** `SafeHandles/Adapter`, `NativeLoraAdapter`, `IInferenceContext.SetAdapters(...)`
(IReadOnlyList record).

**(d) Facade.** `GgufEngineOps.WithLora(path, scale)` / `ClearLora()` — convenience returning
disposable scope.

**(e) KV — protected.** Adapters change *weights*, not KV; existing KV content remains valid
(LoRA changes future logits, same as a sampling change). Document + test: attach mid-
conversation continues generation without rewind (upstream semantics). Save/restore unaffected.

**(f) Fail-closed.** `llama_adapter_lora_init` failure → `PVX_ERR_LOAD_FAILED` naming path;
arch mismatch (adapter GGUF `arch` vs model) → typed error naming both; scale ≤ 0 → typed
arg error. Never apply partially on failure (set-all-or-nothing semantics).

**(g)** Token-exactness: same prompt+sampling, adapter on vs adapter off → both deterministic
and each stable; no cross-contamination test (attach A, detach, verify logits == baseline).

**(h)** S–M. **(i)** Tier 3 (independent, easy win).

---

## 10. Control vectors — Tier 3

**(a) Upstream.** `common/common.cpp:1224` `common_control_vector_load(load_infos)` →
`common_control_vector_data`; applied via **`llama_set_adapter_cvec(ctx, data, len, n_embd,
layer_start, layer_end)`** (`llama.h:741`, `common.cpp:1451–1457`). Layer range defaults 1..n_layer.

**(b) Native ABI (wave v8).** `pvx_control_vector_load(model, path, layer_start, layer_end)`
(port of `common_control_vector_load` into `src/` — GGUB/loRA-vector file parse, ~150 lines),
`pvx_context_set_control_vector(context, handle)` / detach on `pvx_control_vector_free`.
Layer-range validation against `llama_model_n_layer` before set.

**(c) C#.** `SafeHandles/ControlVector`, `NativeControlVector`, config record
`ControlVectorConfig { Path, LayerStart, LayerEnd }`.

**(d) Facade.** `GgufEngineOps.WithControlVector(...)` scoped disposable, same shape as #9.

**(e) KV — protected (identical argument to #9: weights-surface modifier).

**(f) Fail-closed.** Vector n_embd mismatch vs model → typed error naming both dims
(`common.cpp` returns `n_embd == -1` — we translate to typed error, never return res);
bad file → `PVX_ERR_LOAD_FAILED`.

**(g)** Token-exactness: deterministic-logit gate on/off; no-residual test after detach.
**(h)** S–M. **(i)** Tier 3 (ship with #9 — same attach/detach plumbing).

---

## 11. JSON-schema → GBNF grammar — Tier 2

**(a) Upstream.** `common/json-schema-to-grammar.{h,cpp}` (1028 lines): schema (JSON) → GBNF
grammar rules string, consumed by `llama_grammar_init` via the existing grammar chain
(schema→grammar→`common_speculative`-independent). Self-contained, JSON-only, no deps.

**(b) Native ABI (wave v7 or v8).** `pvx_grammar_from_json_schema(const char* json_schema,
const char* root_name?, bool compact_grammar, char** out_grammar_gbnf)` → feeds existing
`pvx_grammar_*`/persistent chain **or** a direct `pvx_executor_prompt_with_schema` variant
(choose in design review; lean: expose as pure converter + caller feeds existing grammar APIs).
Port into `src/pvx_json_schema_grammar.cpp` — the file is a near-mechanical port (JSON parse
via our existing light JSON reader or a vendored-port of `common/json.h` parsing subset).

**(c) C#.** `IGgufEngineOps.CreateGrammarFromJsonSchema(schema)` (facade-level convenience) +
`GrammarKind.JsonSchema` provenance on the grammar record.

**(d) Facade.** As above;ProvostXLLM's tool-calling can then bind schemas without hand-writing
GBNF.

**(e) KV — neutral (pure string→string conversion; grammar acceptance happens at commit only,
existing chain).

**(f) Fail-closed.** Invalid JSON schema → `PVX_ERR_INVALID_ARG` with the parse position;
unsupported schema features → typed `PVX_ERR_NOT_SUPPORTED` naming the JSON-schema keyword
(the port fails closed per keyword — no best-effort grammars).

**(g)** Golden corpus tests: upstream test schemas (vendored tree has none for this file —
build a corpus of ~30 schemas incl. OpenAI tool-call shapes) → expected GBNF, compared
token-exact after normalization. Runtime: grammar-constrained generation must equal the
hand-written-Grammar equivalent output for the same constraints (this is the strong gate).

**(h)** M. **(i)** Tier 2 — **highest value-per-line of the whole plan** (unblocks
structured/tool-call output today).

---

## 12. Metrics / stats export — Tier 3 (low priority)

**(a) Upstream.** `common/common.h` timing + `llama-server` metrics patterns; token-accept
stats exist in `speculative.cpp` (accepted/rejected per round, `draft_min` decisions).

**(b) Native ABI (wave v8).** `pvx_stats_get(context|executor, out pvx_stats_t*)`: counters
(tokens, tok/s windows, spec accepted/drafted ratio, prefill/decode wall splits, cache hit
counters from #2). Snapshot semantics (copy-out struct), zero new locks (counters already
atomics or behind `infer_mtx`).

**(c) C#.** `EngineStats` record + facade property views.

**(d) Facade.** `IInferenceEngine.Stats` per engine + per-conversation.

**(e) KV — neutral (read-only counters).

**(f) Fail-closed.** Counters never lie about failures: spec stats include rejected-draft
counters alongside accepted; prefill-stall counters surface (never hide watchdog events).

**(g)** Determinism-neutral: stats snapshot after known workload == fixture. **(h)** S.
**(i)** Tier 3.

---

## 13. ggml-RPC remote-offload lane — evaluation & recommendation only

**Upstream.** `ggml/src/ggml-rpc`: backend type where host offloads compute to a remote
ggml server (TCP), full or partial layers; `n_ctx`/KV stays local; weights + activations stream.

**Analysis.**
- **Pro:** uniform Metal/CUDA/remote handling inside ggml; would let the weaker Mac borrow the
  stronger one's GPU without any ProvostX-specific protocol.
- **Contra:** (1) KV traffic — with remote-offloaded compute, **every decode step streams
  activations both ways**; llama.cpp RPC is weights-offload-shaped, not KV-cache-shaped — our
  own cross-Mac path moves *state blobs* (existing `pvx_conversation_export` /
  `llama_state_seq_*`) at conversation boundaries, which is dramatically cheaper for our
  long-conversation workload. (2) Latency: RPC per-token round-trips over TCP vs local Metal
  would *regress* tokens/s for the common case. (3) Complexity: ggml-RPC is a second
  scheduling lane interacting with `pvx_backend`'s probe/fallback logic (`pvx_create_context_safe`
  progressive fallback would need RPC-specific failure taxonomy — an entire backend wave for
  a lane we may never use). (4) We already have a working SSH/TB5 transport and a
  state-export surface to build any future remote lane on.

**Recommendation:** **do not integrate ggml-RPC** in this parity program. Keep it in the
backlog as a spike-tier investigation (sizes: S spike — benchmark a hand-built ggml-rpc
server pair against the local-metal baseline before any design work). If a future use case
emerges ("run 70B on the big Mac from the small Mac"), prefer the **state-export +
remote-engine** shape (move the conversation to a full engine on the strong host) over
per-token RPC offload. No ABI surface reserved.

---

# Phased rollout

## Wave 1 — ABI v6 (foundation): Tier 1
- **F1. Embd-aware mirror refactor** (`src/pvx_spec_engine.{h,cpp}`: mirror entry ring
  `(id, embd, pos)`, embd rows via stable batch APIs, zero-pad+warn on width mismatch, M-RoPE
  pinned positions) — feature #1 core, unblocks everything below.
- **F2. Spec + vision enabled end-to-end** (#1) — removes the C# SessionContext-level typed
  refusal; token-exactness gate on the multimodal path.
- **F3. n-gram speculative** (#2): map/mod/cache types, params, cache IO.
- ABI: additive bits + `pvx_ngram_params_t` append → **ABI v6 handshake bump** (C#
  `ExpectedAbiVersion` in same commit).
- Exit criteria: token-exactness gates green for plain/vision/ngram lanes; LDC PASS; live
  capability matrix re-run for the three new lane cells.

## Wave 2 — ABI v7 (draft-graph + structured output): Tier 2
- **F4. DExtract/DFlash fused decode** (#3) — feature-capture runner `src/pvx_dextract.cpp`.
- **F5. EAGLE3** (#4) — encoder pass, deferred-boundary state serialized into the
  conversation save chain (append-only state format, version-tagged).
- **F6. Batched-lane spec** (#5) — `pvx_conversation_spec_*` trio, pool budgeting, lock-order
  extension test.
- **F7. JSON-schema→GBNF** (#11) — converter port + facade grammar convenience.
- ABI v7 handshake bump.

## Wave 3 — ABI v8 (media + weight-surface modifiers): Tier 3
- **F8. Audio input** (#6) — media chunk list ABI, audio embd mirror rows (rides F1).
- **F9. TTS gen-audio** (#7). **F10. Reranking** (#8). **F11. LoRA adapters** (#9).
  **F12. Control vectors** (#10). **F13. Stats export** (#12).
- ABI v8 handshake bump; reranker + TTS cells added to the live capability matrix.

**Dependency order (hard):** F1 → {F3, F4, F5, F6, F8}; F4 ↔ F5 share the feature-capture
runner (F4 first); F5 → F6 (per-seq spec state before pool); nothing blocks F7 (parallelizable
into Wave 2 anytime after design review); F9–F13 mutually independent (F11+F12 ship together).

**Per-wave definition of done:** all listed features token-exactness-gated; ABI handshake bump
committed atomically; `API-INDEX.md` + `RELATIONSHIP-GRAPH.md` (both roots) regenerated, LDC
enforcement PASS; native ctest (Release/metal) + full C# suite green; live capability matrix
updated in the same commit; zero edits inside `llama.cpp/` (CI adds a guard: diff against
submodule pinned SHA must be empty).

---

# Risk register (top 10)

| # | Risk | Impact | Mitigation |
|---|------|--------|-----------|
| 1 | **Mirror desync class (recurring):** image/audio rows advance `n_kv_pos` without plain-token mirror entries — the #1 bug class, already bit us on the image path | Spec drafts verified against wrong positions → accepted garbage tokens silently | F1's mirror ring is the single source of truth; invariant test `mirror.size() == n_kv_pos` asserted after EVERY batch commit + every spec round; caught by token-exactness gate (any desync shows as spec≠nonspec divergence) |
| 2 | **M-RoPE pinned positions vs rewind/shift:** windowed drafts pin absolute positions in mirror rows; `pvx_*_rewind`/`shift_left` change position semantics underneath | Stale pinned rows replayed at wrong positions after rewind | Mirror entries invalidated on rewind/shift (drop ring ≥ rewind point, re-pin on next draft); test: rewind mid-spec + continue generation, spec must equal nonspec; partial-rewind refusal path (hybrid archs) unchanged |
| 3 | **EAGLE3 deferred-boundary state vs conversation save/restore:** `pending_g_last` lives outside llama state; naive restore resumes with stale g | Corruption of draft quality (silent) or crash | Append versioned, length-prefixed EAGLE3 block to conversation save (additive, never reorder); restore without block + EAGLE3 enabled → typed error demanding spec re-init; round-trip test across restart |
| 4 | **Draft-vocab mismatch** (draft-simple / ngram cache from another model): ids valid in one vocab, garbage in another | Crash or silent nonsense drafts | `pvx_model_spec_validate` already checks vocab family — extend to ngram cache load (cache stores vocab hash; mismatch → `PVX_ERR_LOAD_FAILED`); same check on EAGLE3/DFlash companion GGUFs |
| 5 | **Draft-width mismatch zero-pad warns but doesn't abort** (upstream behavior ported in F1): a real architectural mismatch could hide behind warnings | Quiet quality loss | Warn once per pair (not spam), expose `DraftWidthPadded` flag in stats (#12) + facade `ActiveSpeculation` mirror so callers can alert; never abort generation (fail-closed philosophy governs *errors*, warnings govern *quality*) |
| 6 | **TTS model availability:** `MTMD_GEN_AUDIO_TYPE_QWEN3TTS` requires a specific model family we may not have locally | Feature shipped but unusable | Capability probe fails closed at load (typed NOT_SUPPORTED, names QWEN3TTS); ship #7 behind a facade flag default-off; live matrix only adds the cell when a model exists |
| 7 | **Batched-lane spec lock discipline:** per-conversation spec sessions add draft contexts freed at pool return — risk of inverted lock order vs `pool → infer` | Deadlock or use-after-free under pool churn | Spec ops take ONLY `infer_mtx` (same as sample ops); pool-return frees spec session inside existing teardown path (order unchanged); extend the existing race-stress suite with spec-on-pool churn; draft contexts tracked per-model for the `active_context_count` destruction wait (extends existing mechanism, no new condition vars) |
| 8 | **Feature-capture runner (DExtract) needs `llama_batch` output-flag APIs stable across vendored-tree upgrades:** we depend on hidden-state capture surface | Silent breakage on next submodule bump | Pin to current vendored SHA (already); add a `pvx_dextract` self-test at context init (one 1-token capture + shape check) that fails typed at spec-enable, not at generation time |
| 9 | **JSON-schema port drift** vs upstream converter behavior | Structured output regressions vs users' expectations from llama-server | Golden-corpus tests generated from upstream `json-schema-to-grammar` outputs at the pinned SHA (we run the vendored tree in TESTS ONLY — no product link); drift review at every submodule bump |
| 10 | **State-format growth (#4, #6) vs existing checkpoint compatibility** | Old ProvostXLLM checkpoints failing to restore | All new state blocks append-only + version-tagged; unknown/absent blocks → typed, named restore error (never partial restore); round-trip tests with pre-wave fixtures kept in `tests/fixtures/` |

---

**Review point:** this plan is the design input; each wave's feature gets its own short design
note in `docs/plans/` (linked from the backlog) before implementation. Nothing here licenses
edits to the frozen tree, pushes, or tags.

*Relative paths only; no hostnames, no usernames, no hardware names.*