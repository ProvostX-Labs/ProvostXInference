using Xunit;

namespace ProvostXInference.Tests;

/// <summary>
/// Fact that runs only on macOS — for tests asserting macOS-specific
/// engine defaults (Metal). Skips elsewhere instead of failing.
/// </summary>
public sealed class MacOSOnlyFactAttribute : FactAttribute
{
    public MacOSOnlyFactAttribute()
    {
        if (!OperatingSystem.IsMacOS())
            Skip = "macOS-only: asserts Metal as the default engine selection.";
    }
}
