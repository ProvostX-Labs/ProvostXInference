using System.IO;
using ProvostXInference.Interop;

namespace ProvostXInference.Models;

/// <summary>
/// OS-aware resolver from validated engine id to the specialist module file
/// name under this assembly's runtimes/&lt;rid&gt;/native/ directory. metal =
/// macOS (.dylib); cuda/vulkan = Windows (.dll) / Linux (.so). Fail-closed:
/// any (engine, OS) pair outside the matrix throws rather than guessing.
/// Stateless utility — no mutable state.
/// </summary>
internal sealed class PvxEngineFileResolver : IPvxEngineFileResolver
{
    // Stateless utility — no mutable state (immutable matrix data, not state).
    private static readonly string[] KnownEngineIds = { "metal", "cuda", "vulkan" };

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
        _ => throw new DllNotFoundException(
            $"engine '{validatedEngine}' is not a specialist on this OS " +
            "(metal = macOS only; cuda/vulkan = Windows/Linux only)")
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
            var rid = _os.IsMacOS() ? "osx-arm64" : _os.IsLinux() ? "linux-x64" : "win-x64";
            return Path.Combine(asmDir ?? ".", "runtimes", rid, "native");
        }
    }
}