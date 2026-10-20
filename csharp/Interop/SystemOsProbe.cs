namespace ProvostXInference.Interop;

/// <summary>OperatingSystem-based implementation of the OS probe seam.
/// Stateless utility — no mutable state.</summary>
internal sealed class SystemOsProbe : IOSProbe
{
    public bool IsMacOS() => OperatingSystem.IsMacOS();
    public bool IsLinux() => OperatingSystem.IsLinux();

    public string Architecture() => System.Runtime.InteropServices.RuntimeInformation.ProcessArchitecture switch
    {
        System.Runtime.InteropServices.Architecture.Arm64 => "arm64",
        System.Runtime.InteropServices.Architecture.Arm => "arm",
        System.Runtime.InteropServices.Architecture.X86 => "x86",
        _ => "x64",
    };
    public bool IsWindows() => OperatingSystem.IsWindows();
}