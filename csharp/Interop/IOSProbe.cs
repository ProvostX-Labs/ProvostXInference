namespace ProvostXInference.Interop;

/// <summary>OS identity seam — injected so the engine matrix logic is testable
/// on every OS from any host.</summary>
internal interface IOSProbe
{
    bool IsMacOS();
    bool IsLinux();
    bool IsWindows();
}