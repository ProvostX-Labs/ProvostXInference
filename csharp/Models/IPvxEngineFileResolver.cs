namespace ProvostXInference.Models;

/// <summary>
/// Resolves a validated engine spec onto the specialist module file for this
/// OS — the ONLY place specialist file names live. The raw native image handle
/// flows downward through the Interop loader; a resolution miss is fail-closed
/// with a typed error (never a silent fallback to another engine's file).
/// </summary>
internal interface IPvxEngineFileResolver
{
    /// <summary>Specialist file name for an ALREADY-VALIDATED engine on this OS
    /// (throws fail-closed when the (engine, OS) pair is not in the matrix).</summary>
    string FileName(string validatedEngine);

    /// <summary>BL-48: the file name to actually LOAD — equals FileName for
    /// cuda/vulkan/metal; for the cpu specialist falls back to the generic
    /// payload name (pvxinference.dll/.so) when no specialist-named copy is
    /// staged (that is what the checked-in win-arm64 payload ships).</summary>
    string EffectiveFileName(string validatedEngine);

    /// <summary>Whether the specialist file exists next to this assembly.</summary>
    bool FileExists(string validatedEngine);

    /// <summary>Engine ids that have their specialist file on this machine.</summary>
    IReadOnlyList<string> AvailableEngines();

    /// <summary>The per-OS runtimes directory this resolver resolves against
    /// (runtimes/&lt;rid&gt;/native next to the binding assembly).</summary>
    string RuntimeDirectory { get; }
}