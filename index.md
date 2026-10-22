# ProvostXInference

C/C++ inference engine + C# bindings for ProvostX — a llama.cpp-only, GGUF-only
execution substrate behind a strict-OOP C# facade.

Start here:

- [Consumer contract](CONTRACT.md) — what you may touch, what is guaranteed, what is closed
- [Architecture](ARCHITECTURE.md) — as-is design, all layers, ABI waves
- [Getting started](GETTING-STARTED.md) — build native + C# from a fresh clone
- [Configuration](docs/CONFIGURATION.md) — configuration reference
- [Capability matrix](docs/CAPABILITY.md) — what works where, careful-handling areas
- [API glossary](docs/api-glossary.md) — every public type explained
- [FAQ](docs/faq.md) — real questions, real answers
- API reference — generated site under `api/` (nav via the sidebar; entry points below)

Key entry points for consumers:

- <xref:ProvostXInference.Engines.InferenceEngineBuilder> — the only entry point
- <xref:ProvostXInference.Engines.IGgufEngineOps> — the GGUF lane ops view
- <xref:ProvostXInference.Models.EngineTelemetry> — model-wide telemetry snapshot (ABI v9)

LLM (llama.cpp, GGUF), Decision (ONNX judge) and Encoder lanes live behind one
factory-built engine object. Fail-closed contracts throughout; per-engine audit
and telemetry since ABI v9.
