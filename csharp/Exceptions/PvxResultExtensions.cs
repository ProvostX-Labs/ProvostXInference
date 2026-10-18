using System.Runtime.InteropServices;
using ProvostXInference.Interop;

namespace ProvostXInference.Exceptions;

/// <summary>Extension methods for PvxResult error handling.</summary>
internal static class PvxResultExtensions
{
    /// <summary>Per-handle path (default): errors resolved through the OWNING
    /// ENGINE's dispatch tables (constructor-injected entry points) — never a
    /// global static route, so message strings come from the engine module that
    /// produced the result.</summary>
    internal static void ThrowIfError(this PvxResult result, IPvxEntryPoints entryPoints, IntPtr contextHandle = default)
    {
        if (result == PvxResult.Ok) return;
        throw new InferenceException((int)result, Describe(result, contextHandle, entryPoints.ResultStr, entryPoints.LastError));
    }

    /// <summary>Model-scoped error translation (ABI v5): the reason string
    /// comes from pvx_model_last_error — never from pvx_last_error, whose
    /// context-handle parameter would read the wrong struct layout when handed
    /// a model handle (the 2026-10-18 empty "NOT_SUPPORTED: " message bug).
    /// Stateless per call.</summary>
    internal static void ThrowIfModelError(this PvxResult result, IPvxEntryPoints entryPoints, IntPtr modelHandle)
    {
        if (result == PvxResult.Ok) return;
        var strPtr = entryPoints.ResultStr(result);
        var baseMsg = strPtr != IntPtr.Zero ? Marshal.PtrToStringUTF8(strPtr) ?? "unknown" : "unknown";
        var errPtr = entryPoints.ModelLastError(modelHandle);
        if (errPtr != IntPtr.Zero)
        {
            var detail = Marshal.PtrToStringUTF8(errPtr);
            if (!string.IsNullOrEmpty(detail))
                baseMsg = $"{baseMsg}: {detail}";
        }
        throw new InferenceException((int)result, baseMsg);
    }

    /// <summary>Formats a native failure message from the result string plus
    /// the context-scoped last-error detail. Stateless per call.</summary>
    private static string Describe(PvxResult result, IntPtr contextHandle,
        Func<PvxResult, IntPtr> resultStr, Func<IntPtr, IntPtr> lastError)
    {
        var strPtr = resultStr(result);
        var baseMsg = strPtr != IntPtr.Zero ? Marshal.PtrToStringUTF8(strPtr) ?? "unknown" : "unknown";
        if (contextHandle != default)
        {
            var errPtr = lastError(contextHandle);
            if (errPtr != IntPtr.Zero)
            {
                var detail = Marshal.PtrToStringUTF8(errPtr);
                if (!string.IsNullOrEmpty(detail))
                    baseMsg = $"{baseMsg}: {detail}";
            }
        }
        return baseMsg;
    }
}

/// <summary>Formats a native failure message from the result string plus the
/// context-scoped last-error detail. Shared by the per-handle dispatch path.
/// Stateless utility — no mutable state.</summary>
internal static class PvxErrorDescriber
{
    // Stateless utility — no mutable state.
    internal static string Describe(PvxResult result, IntPtr contextHandle,
        Func<PvxResult, IntPtr> resultStr, Func<IntPtr, IntPtr> lastError)
    {
        var strPtr = resultStr(result);
        var baseMsg = strPtr != IntPtr.Zero ? Marshal.PtrToStringUTF8(strPtr) ?? "unknown" : "unknown";
        if (contextHandle != default)
        {
            var errPtr = lastError(contextHandle);
            if (errPtr != IntPtr.Zero)
            {
                var detail = Marshal.PtrToStringUTF8(errPtr);
                if (!string.IsNullOrEmpty(detail))
                    baseMsg = $"{baseMsg}: {detail}";
            }
        }
        return baseMsg;
    }
}