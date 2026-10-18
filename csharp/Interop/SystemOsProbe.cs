namespace ProvostXInference.Interop;

/// <summary>OperatingSystem-based implementation of the OS probe seam.
/// Stateless utility — no mutable state.</summary>
internal sealed class SystemOsProbe : IOSProbe
{
    public bool IsMacOS() => OperatingSystem.IsMacOS();
    public bool IsLinux() => OperatingSystem.IsLinux();
    public bool IsWindows() => OperatingSystem.IsWindows();
}