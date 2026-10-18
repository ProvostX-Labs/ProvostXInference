# Speculative decoding test evidence (owner-requested, 2026-10-11 23:27)

All runs: llama-cli (vendored build-spec, metal), temp 0.

## draft_simple — WORKS
- Target Llama-3.2-3B-Q4_K_M + draft Llama-3.2-1B-Q4_K_M: clean generation.
- Baseline 168.9 t/s vs draft ~161–162 t/s → NO gain on a fast small target (draft overhead ≥ savings at 3B speeds). Expected physics; use spec on bigger/slower targets.

## draft_mtp — WORKS, real speedup
- Model: unsloth/Qwen3.5-9B-MTP-GGUF (Qwen3.5-9B-Q4_K_M.gguf, 5.9 GB, MTP tensors embedded).
- Short gen (96 tok): baseline 63.1 t/s → MTP 90.3 t/s (+43%)
- Long gen (256 tok): baseline 82.2 t/s → MTP 108.1 t/s (+31%)
- Conclusion: ~1.3–1.4× on Qwen3.5-9B via --spec-type draft-mtp. No second model needed.

## Notes for ABI wiring
- CLI flag: `--spec-type draft-mtp` (enum: none, draft-simple, draft-eagle3, draft-mtp, draft-dflash, draft-dspark, ngram-*).
- Our other local GGUFs (Qwen3-8B, Llama-3.2-3B, Qwen3.6-35B) have NO nextn tensors → engine capability discovery must read GGUF metadata (arch + nextn tensor presence), not assume.
- EAGLE3/DFLASH/DSpark drafts are per-target companions; drafts need GGUF conversion against the target (docs/speculative.md has exact commands; EAGLE3 pairs AngelSlim/Qwen3-8B_eagle3 with our local Qwen3-8B).
