# runtimes/<rid>/native

NO native binaries are checked into the repository. Every RID slice is built
by the CI specialist matrix at NuGet-pack time:

- `osx-arm64` — metal specialist (`libpvx-metal.dylib`) + generic payload
- `win-arm64` — cpu/default specialist (generic `pvxinference.dll`)
- `win-x64` / `linux-x64` — cuda + vulkan specialists (and cpu where built)
- `linux-arm64` — source-build the native engine (cmake) + ONNX Runtime for
  arm64 (no official ORT arm64 payload exists)

Rationale: compilers bake absolute build paths into binaries (MSVC embeds
PDB/source paths, ld64 records DWARF dirs). Checking binaries in would leak
build-host paths; CI-built payloads carry only the CI runner's ephemeral
paths. Local (untracked) binaries in these directories are VM-test fixtures
only and never ship.

The managed resolver (`PvxEngineFileResolver`) loads specialist-named files
first; for the cpu specialist it falls back to the generic payload name
(BL-48) — which is what this directory receives from CI.
