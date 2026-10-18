using System.IO;
using System.Runtime.InteropServices;
using System.Runtime.Versioning;
using System.Security;

namespace ProvostXInference.Interop;

/// <summary>
/// Pure P/Invoke declarations for libpvxinference. This is the single static
/// class in the project (the sanctioned interop exception). EntryPoint matches
/// the C API snake_case names exactly. BL-02: the class carries ONLY the
/// sanctioned interop exception members — logical-name load + UTF-8 string
/// alloctor + ABI-version constant (the handshake itself moved into the
/// per-handle dispatch router). Steady-state calls never go through this
/// class; every model resource dispatches on its engine's own tables.
/// </summary>
[SuppressUnmanagedCodeSecurity]
internal static class PvxNative
{

    /// <summary>ABI version this C# binding expects — must equal PVX_ABI_VERSION
    /// in include/pvxinference.h. Bumped in LOCKSTEP with any native signature change;
    /// a mismatch fails fast with a clear error instead of a silent ABI drift
    /// (root cause of the 2026-09-27 INVALID_ARG/segfault: an orphaned app-root
    /// native copy was loaded instead of the current one). BL-05: 3 — speculative
    /// decoding entry points + supported-types capability report. v6 (wave 1):
    /// embd-aware spec mirror (spec+vision), n-gram spec modes + knobs,
    /// spec_vision capability bit. v7 (wave 2): companion-draft modes
    /// (eagle3/dflash params append), batched/pool-lane spec trio,
    /// JSON-schema→GBNF converter.</summary>
    internal const int ExpectedAbiVersion = 7;

    /// <summary>
    /// NATIVE RESOLUTION (2026-09-27 fix, retained): load the native library
    /// explicitly from runtimes/&lt;rid&gt;/native NEXT TO THIS ASSEMBLY before any
    /// DllImport that could still resolve through the default probe. With the
    /// per-handle router active this is only a safety net (the router pre-loads
    /// the exact specialist path; DllImports never fire while a router is
    /// registered), but it keeps app-root orphans from silently winning.
    /// NativeLibrary.Load is idempotent for an already-loaded path.
    /// </summary>
    static PvxNative()
    {
        try
        {
            var asmDir = Path.GetDirectoryName(typeof(PvxNative).Assembly.Location);
            if (!string.IsNullOrEmpty(asmDir))
            {
                string relative = OperatingSystem.IsMacOS()
                    ? Path.Combine("runtimes", "osx-arm64", "native", "libpvxinference.dylib")
                    : OperatingSystem.IsLinux()
                        ? Path.Combine("runtimes", "linux-x64", "native", "libpvxinference.so")
                        : Path.Combine("runtimes", "win-x64", "native", "pvxinference.dll");
                var candidate = Path.Combine(asmDir, relative);
                if (File.Exists(candidate))
                    NativeLibrary.Load(candidate);
            }
        }
        catch
        {
            // Resolution failure here is NOT fatal — the default probe gets its
            // chance on the first P/Invoke and reports its own error.
        }
    }

    /// <summary>F15: allocate a NUL-terminated UTF-8 copy of the string
    /// (there is no StringToHGlobalUTF8 in .NET). Free with Marshal.FreeHGlobal.
    /// Returns IntPtr.Zero for null/empty.</summary>
    internal static IntPtr AllocUTF8(string? s)
    {
        if (string.IsNullOrEmpty(s)) return IntPtr.Zero;
        var bytes = System.Text.Encoding.UTF8.GetBytes(s);
        var ptr = System.Runtime.InteropServices.Marshal.AllocHGlobal(bytes.Length + 1);
        System.Runtime.InteropServices.Marshal.Copy(bytes, 0, ptr, bytes.Length);
        System.Runtime.InteropServices.Marshal.WriteByte(ptr, bytes.Length, 0);
        return ptr;
    }
}