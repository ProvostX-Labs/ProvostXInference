using System.Runtime.InteropServices;

namespace ProvostXInference.Interop;

/// <summary>Blittable struct for pvx_chat_message_t { const char* role; const char* content; }</summary>
[StructLayout(LayoutKind.Sequential)]
internal struct PvxChatMessage
{
    public IntPtr Role;
    public IntPtr Content;
}

/// <summary>Blittable struct for pvx_chat_tool_t { name, description, parameters_json } — ABI v9.</summary>
[StructLayout(LayoutKind.Sequential)]
internal struct PvxChatTool
{
    public IntPtr Name;
    public IntPtr Description;
    public IntPtr ParametersJson;
}
