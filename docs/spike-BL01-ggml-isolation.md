# BL-01 · ggml multi-backend isolation spike — Verdict: GREEN (2026-10-09)

**Summary:** ggml backend registries are PER-IMAGE statics, not shared globals — two specialist dylib images coexist in one process with zero registry pollution; Phase B per-handle tables unlocked (macOS slice proven).

## Evidence (all PASS, see `spike/BL01Spike/spike-run.log`)
- **E1a dlopen dedupe:** `NativeLibrary.Load` same path twice → identical handle (7a7fd3d0) — de-duped to one image.
- **E1b identical content, different path** → DISTINCT handles (7a7fd3d0 vs 7a7e2af0) — dlopen keys on path, NOT content; specialist copies do not collapse.
- **E2 two independently built specialist images coexist** as distinct handles (7a7fd3d0 / 7a7e2d70).
- **E3 symbol routing:** ONE `DllImportResolver` on the C# assembly, name→handle dict (`pvxinferenceA`→A, `pvxinferenceB`→B); `pvx_abi_version` returned 2 via both routes — no symbol collision, no interference.
- **E4 registry isolation:** image A's registry snapshot `regs=[MTL,BLAS,CPU] devs=[MTL0,BLAS,CPU]` identical before/after image B's full metal init — NO pollution, NO duplication (A still reports 3 regs, not 6).
- **E5 registry pointer identity:** `ggml_backend_reg_get(0)` A=166d5f898 vs B=16d6db3d0 — SEPARATE registry instances.

## Repro
- `spike/BL01Spike/` (scratch .NET 10 console; NOT wired into product). Run: `spike/BL01Spike/run.sh` (env `SPIKE_A`, `SPIKE_B` = two built specialist dylib paths).
- Tested with `build-metal-rel` + `build-metal2` metal specialists (osx-arm64, Apple Silicon desktop-class); both fully initialized metal (35 embedded metal libs each, ~2.5s + cached).

## What Phase B CAN assume (macOS-proven)
- Two same-specialist dylib images coexist via dlopen path-keying; registry/symbol state is per-image.
- Per-handle function tables + per-assembly resolvers can route logical names to distinct images safely.
- Registry of image A is untouched by image B's load — fail-closed per-handle dispatch is sound.

## What remains UNPROVEN
- **macOS metal + cuda/vulkan cross-specialist pair:** no vulkan/cuda specialist built locally — cross-backend registry semantics per-OS (Windows/Linux) need the BL-02 CI matrix builds. dlopen path-keying and registry-per-image statics are ggml structural properties and are expected to hold, but only the metal+metal pair was executed.
- **Windows:** .NET on Windows uses `LoadLibrary`, not dlopen — same-path dedupe and copy-path distinctness should map, but is untested on Windows.
- No token-exactness check in this spike (that is BL-02 acceptance).

## Rules
- Spike only; no product source touched; no merge to main. No identifying data anywhere.
