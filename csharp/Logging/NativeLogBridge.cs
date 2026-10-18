using System.Runtime.InteropServices;
using ProvostXInference.Interop;

namespace ProvostXInference.Logging;

/// <summary>
/// Bridges native C-level logging (pvx_set_log_callback) to a managed callback.
/// Encapsulated in the Inference layer — no dependency on LLM or Core loggers.
/// The caller provides an Action&lt;int,string,string&gt; that forwards to whatever
/// logger they have (LLM's ServerLogger, Core's Logger, console, etc.).
/// Per-handle dispatch (BL-02): the pvx_set_log_* entry points are invoked on
/// the ENGINE'S OWN dispatch tables (resolved via the active per-handle
/// dispatch route) — never the global static route — so logging state lives in
/// the specialist module that owns the model. The delegate thunk itself must
/// still be pinned for the process lifetime (the native side stores the raw
/// function pointer).
/// </summary>
public static class NativeLogBridge
{
    // Sanctioned interop exception: static mutable state is required to root
    // the callback thunk for the process lifetime (GC-rooted, F13).
    private static PvxLogDelegate? _pinned;
    private static Action<int, string, string>? _handler;

    /// <summary>Enable native logging. Pass null to disable (zero-overhead).
    /// The pvx_set_log_* calls dispatch on the ACTIVE engine's per-handle
    /// tables (this API is process-wide by its nature — native logging state
    /// lives in exactly one specialist module).</summary>
    public static void Enable(Action<int, string, string> handler)
    {
        var entryPoints = PvxDispatchRegistry.ActiveEntryPoints;
        if (handler is null)
        {
            entryPoints.SetLogCallback(IntPtr.Zero);
            _pinned = null;
            _handler = null;
            return;
        }

        _handler = handler;
        // Native log strings are UTF-8 — decode explicitly (no ANSI mojibake).
        _pinned = (level, tagPtr, msgPtr) => handler(level,
            Marshal.PtrToStringUTF8(tagPtr) ?? "",
            Marshal.PtrToStringUTF8(msgPtr) ?? "");
        entryPoints.SetLogCallback(Marshal.GetFunctionPointerForDelegate(_pinned));
    }

    /// <summary>Set the minimum log level. 0=Debug, 1=Info, 2=Warn, 3=Error,
    /// 99=None. Dispatches on the ACTIVE engine's per-handle tables.</summary>
    public static void SetLevel(int level) =>
        PvxDispatchRegistry.ActiveEntryPoints.SetLogLevel(level);

    /// <summary>Config-driven native log file (inference.logpath): the dylib
    /// writes ALL native output (llama.cpp/ggml/Metal) to this path itself.
    /// Empty/unset disables the file sink. Dispatches on the ACTIVE engine's
    /// per-handle tables.</summary>
    public static void SetLogFile(string? path) =>
        PvxDispatchRegistry.ActiveEntryPoints.SetLogFile(path ?? "");
}