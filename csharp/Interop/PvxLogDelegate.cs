using System.Runtime.InteropServices;

namespace ProvostXInference.Interop;

/// <summary>
/// Log callback delegate (pvx_set_log_callback). Must be rooted by the caller
/// for the process lifetime — the native side stores the raw function pointer.
/// </summary>
[UnmanagedFunctionPointer(CallingConvention.Cdecl)]
public delegate void PvxLogDelegate(int level, IntPtr tag, IntPtr message);