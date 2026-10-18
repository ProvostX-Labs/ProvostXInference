using System.IO;

namespace ProvostXInference.Interop;

/// <summary>Filesystem implementation of the existence seam.
/// Stateless utility — no mutable state.</summary>
internal sealed class SystemFileExistence : IPvxFileExistence
{
    public bool Exists(string path) => File.Exists(path);
}