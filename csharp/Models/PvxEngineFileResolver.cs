using System.IO;
using ProvostXInference.Interop;

namespace ProvostXInference.Models;

/// <summary>
/// OS-aware resolver from validated engine id to the specialist module file
/// name under this assembly's runtimes/&lt;rid&gt;/native/ directory (BL-43:
/// Linux rid = linux-&lt;arch&gt; from the process architecture). metal =
/// macOS (.dylib); cuda/vulkan/cpu = Windows (.dll) / Linux (.so). Fail-closed:
/// any (engine, OS) pair outside the matrix throws rather than guessing.
/// Stateless utility — no mutable state.
/// </summary>
internal sealed class PvxEngineFileResolver : IPvxEngineFileResolver
{
    // Stateless utility — no mutable state (immutable matrix data, not state).
    private static readonly string[] KnownEngineIds = { "metal", "cuda", "vulkan", "cpu" };

    private readonly IOSProbe _os;

    public PvxEngineFileResolver(IOSProbe os)
    {
        _os = os ?? throw new ArgumentNullException(nameof(os));
    }

    public string FileName(string validatedEngine) => validatedEngine switch
    {
        "metal" when _os.IsMacOS() => "libpvx-metal.dylib",
        "cuda" when _os.IsWindows() => "pvx-cuda.dll",
        "cuda" when _os.IsLinux() => "libpvx-cuda.so",
        "vulkan" when _os.IsWindows() => "pvx-vulkan.dll",
        "vulkan" when _os.IsLinux() => "libpvx-vulkan.so",
        "cpu" when _os.IsWindows() => "pvx-cpu.dll",
        "cpu" when _os.IsLinux() => "libpvx-cpu.so",
        _ => throw new DllNotFoundException(
            $"engine '{validatedEngine}' is not a specialist on this OS " +
            "(metal = macOS only; cuda/vulkan/cpu = Windows/Linux only)")
    };

    public bool FileExists(string validatedEngine)
    {
        try
        {
            return System.IO.File.Exists(Path.Combine(RuntimeDirectory, FileName(validatedEngine)));
        }
        catch (DllNotFoundException)
        {
            // Wrong-OS pair: report as absent (fail-closed), never a fallback.
            return false;
        }
    }

    public IReadOnlyList<string> AvailableEngines() => KnownEngineIds
        .Where(FileExists)
        .ToList();

    public string RuntimeDirectory
    {
        get
        {
            var asmDir = Path.GetDirectoryName(typeof(PvxEngineFileResolver).Assembly.Location);
            // BL-43: the Linux RID follows the PROCESS ARCHITECTURE (linux-arm64 on
            // arm64 hosts — was hardcoded linux-x64). BL-46: Windows follows the
            // same rule (win-arm64 payload ships; win-x64 from the CI specialist
            // matrix). macOS stays osx-arm64.
            var rid = _os.IsMacOS() ? "osx-arm64"
                : _os.IsLinux() ? $"linux-{_os.Architecture()}"
                : $"win-{_os.Architecture()}";
            return Path.Combine(asmDir ?? ".", "runtimes", rid, "native");
        }
    }
}