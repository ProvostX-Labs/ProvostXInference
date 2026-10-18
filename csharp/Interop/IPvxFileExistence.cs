namespace ProvostXInference.Interop;

/// <summary>File existence seam (testable stand-in for File.Exists).
/// Stateless utility — no mutable state.</summary>
internal interface IPvxFileExistence
{
    bool Exists(string path);
}