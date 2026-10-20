# win-arm64 native payload

Provenance: built from the Windows 11 ARM64 VM (MSVC VS 2022 BuildTools 14.44,
CMake "Visual Studio 17 2022" -A ARM64, Release, backend set matching the
macOS Metal specialist's feature set minus Metal). This is the CPU/default
specialist — no CUDA/Vulkan. ABI v8, export surface via WINDOWS_EXPORT_ALL_SYMBOLS
(see INF BL-44 for the MSVC portability record).

Contents:
- `pvxinference.dll` — the native engine, loaded by the C# resolver on
  win-arm64 hosts (PvxNative derives the RID from ProcessArchitecture).
- `pvxinference.lib` — MSVC import lib for native consumers linking the DLL.

arm64 Windows users who need ORT-dependent features source-build ORT for
win-arm64 (no official ONNX Runtime win-arm64 payload ships here); the
resolver looks in `runtimes/win-arm64/native/` next to the assembly.
win-x64 payloads are produced by the CI specialist matrix (cuda/vulkan), not
checked in. See ARCHITECTURE.md "Native runtime payloads" for the full
per-RID story.
