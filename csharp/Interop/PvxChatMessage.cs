using System.Runtime.InteropServices;

namespace ProvostXInference.Interop;

/// <summary>Blittable struct for pvx_chat_message_t { const char* role; const char* content; }</summary>
[StructLayout(LayoutKind.Sequential)]
internal struct PvxChatMessage
{
    public IntPtr Role;
    public IntPtr Content;
}
