namespace ProvostXInference.Interop;

/// <summary>OS identity seam — injected so the engine matrix logic is testable
/// on every OS from any host.</summary>
internal interface IOSProbe
{
    bool IsMacOS();
    bool IsLinux();
    bool IsWindows();
    /// <summary>BL-43: process architecture string ("arm64" / "x64" / ...) —
    /// the RID seam for the native dir derivation. Test doubles override.</summary>
    string Architecture();
}