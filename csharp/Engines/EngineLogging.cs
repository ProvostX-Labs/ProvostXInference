namespace ProvostXInference.Engines;

/// <summary>
/// Facade view of the native logging bridge (owner black-box directive
/// 2026-10-19): native C-level log setup is a process-wide facade concern
/// (the pvx_set_log_* entry points dispatch on the ACTIVE engine's tables),
/// so callers configure it here instead of reaching into the Logging
/// namespace. Pure delegation — identical semantics, facade-only surface.
/// </summary>
public static class EngineLogging
{
    /// <summary>Route native engine logs into <paramref name="path"/>
    /// (native file sink). Pass null to disable the file sink.</summary>
    public static void SetLogFile(string? path)
        => Logging.NativeLogBridge.SetLogFile(path);

    /// <summary>Forward native log lines to a managed handler
    /// ((level, tag, message); pass null to disable with zero overhead).</summary>
    public static void Enable(Action<int, string, string> handler)
        => Logging.NativeLogBridge.Enable(handler);

    /// <summary>Set the minimum native log level. 0=Debug, 1=Info, 2=Warn,
    /// 3=Error, 99=None.</summary>
    public static void SetLevel(int level)
        => Logging.NativeLogBridge.SetLevel(level);
}
