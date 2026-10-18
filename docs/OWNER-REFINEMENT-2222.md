# Owner refinement — 22:22 (applies to BL-05)

The ENGINE decides what the loaded MODEL supports — capability discovery is the core feature.

- On model load, inspect the model (arch, MTP layers present, draft-compatible structures) and self-report a SUPPORTED-TYPES LIST from the llama.cpp speculative enum (e.g. mtp, draft_simple, eagle3, dflash) — not boolean flags.
- The LLM/dashboard then offers exactly the supported types as options for sequential mode.
- If the model supports NO native speculative type → the capability report says so, and "second draft model" (DRAFT_SIMPLE with a user-provided draft model) remains the available option — it works for any target model when a draft model is provided.
- MTP default-ON decision belongs to the LLM slice later; this slice delivers accurate capability reporting + working wiring for each type landed in this pass (DRAFT_SIMPLE first; MTP/EAGLE3/DFLASH only if the vendored llama.cpp makes them straightforward, else typed NOT_SUPPORTED with the boundary named).

Update BACKLOG-MASTER BL-05 spec text to reflect capability-type discovery (supported-types list, not booleans) when you update statuses.
