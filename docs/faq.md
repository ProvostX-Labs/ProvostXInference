# ProvostXInference — FAQ

Real questions from development and integration. Answers are contracts, not marketing.

## 1. Why not just use LLamaSharp?

LLamaSharp wraps llama.cpp but owns the seams we need to control: batching discipline,
KV lifecycle, backend safety, and error typing. We link llama.cpp directly behind our own
flat C API so every one of those surfaces is explicit, fail-closed, and test-pinned. The
C# layer is strict OOP (one type per file, constructor injection, no statics in product
code) — that is a deliberate architectural stance, not style.

## 2. What happens if the native library version doesn't match the C# binding?

The engine refuses to load. The ABI version is handshake-checked at engine open
(`PVX_ABI_VERSION` vs `ExpectedAbiVersion`); a mismatch is a fail-closed error, never a
best-effort attempt. Struct layouts are size-pinned by tests (`PvxAbiSizeTests`) so a
drift cannot pass silently.

## 3. Does one process support multiple engines with different models?

Yes — that is a first-class scenario. Each engine instance owns its model, contexts,
pools, and telemetry. Two engines write two independent audit files (per-engine audit,
ABI v9) and expose two independent `GetEngineTelemetry()` streams. Nothing is shared
except the process-wide DEBUG log stream (`EngineLogging`), which is a single native
log sink by design.

## 4. Does the audit trail record prompt text?

No — by policy. The audit trail records lifecycle and operation events
(model/context lifecycle, spec enable/disable, destructive KV ops) and optionally prompt
token COUNTS (`AuditIncludePrompts`). Prompt text is never written: audit logs must not
become a content sink. If you need content, that belongs in your service layer.

## 5. What does the audit feature cost when I don't configure it?

One branch per commit. Every counter bump and every file write is gated behind
`audit.enabled`; an unconfigured engine executes no counters and no I/O. The feature is
dead when not actively set up.

## 6. A capability reports "supported" but I get a NOT_SUPPORTED exception — why?

You are likely outside a documented lane boundary. Speculation, for example,
self-reports capability per MODEL, but some modes are sequential-executor-lane only
(the batched/pool conversation lane refuses typed). Every refusal names its boundary
(e.g. `spec-engine/mtp-wiring`, `spec-engine/session`). The refusal is the contract:
the engine never silently falls back to plain generation.

## 7. Which specialist backend is actually running?

Ask the engine: `ops.Backend` reports the active specialist, effective vs requested GPU
layers, and whether the load was reduced. The probe never misreports an active
accelerator as CPU (a historic bug, fixed and pinned by tests). On macOS the metal
specialist is the default; Linux/Windows default to cuda → vulkan → cpu in order of
availability.

## 8. How do I get throughput numbers (tokens/second)?

Telemetry is pull-based and stateless: take two `GetEngineTelemetry()` snapshots around
your workload and derive rates in your code (`DecodeSteps` or `CommittedTokens` divided
by elapsed time). The native side deliberately keeps no rate state — snapshots stay
cheap and allocation-free.

## 9. Can I resume a conversation after a process restart?

Yes — `ExportCheckpoint()` produces a byte blob (cells + K/V rows) that
`ImportCheckpoint()` restores on a fresh engine, with token counts matching exactly.
Verified across dense and hybrid architectures up to the largest test models.

## 10. Why are model files not in the repository, and why do live tests "skip"?

Models are gigabyte-scale artifacts; the repo ships none (and no native binaries —
CI builds every platform slice). Live test lanes discover models via
`PVX_SMALL_*_PATH` env vars or repo-relative fixture locations; when a fixture is
absent the lane skips with a named reason instead of failing. The suite is fully green
with zero fixtures present (hermetic mode).

## 11. Is Windows / Linux really supported, or just macOS?

All three build and test in CI (specialist matrix: osx-arm64 metal, win-arm64 cpu,
win-x64/linux-x64 cuda+vulkan, plus a full Windows MSVC lane). The real-model journey
tier has been verified live on ARM64 Linux and Windows lab VMs. CUDA/Vulkan
specialists are CI-built but their real-model verification awaits a GPU host with
models — that is stated openly in the backlog, not hidden.

## 12. What does "fail-closed" mean in practice?

A failure or an unknown state produces a typed error that names what happened and
where — it never produces a guessed result. Examples: unknown engine id → refusal
listing known specialists; grammar with an unsupported JSON-schema keyword →
`NOT_SUPPORTED` naming the keyword; rewind on a hybrid architecture →
`REWIND_UNSUPPORTED`. We consider a silent wrong answer worse than a loud refusal.
