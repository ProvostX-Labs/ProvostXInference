using ProvostXInference.Abstractions;
using ProvostXInference.Exceptions;
using ProvostXInference.Interop;
using ProvostXInference.Models;

namespace ProvostXInference.Implementation;

/// <summary>Speculative session bound to ONE pooled conversation (ABI v7,
/// feature 5): enable/generate/stats over pvx_conversation_spec_*, with the
/// active-config mirror kept honest by routing every mutation through
/// Enable/Disable below (same discipline as the executor lane). The native
/// pool-return/teardown paths free the session natively; <see cref="OnReturned"/>
/// clears the managed mirror so it can never report a phantom session.</summary>
public sealed class NativeConversationSpecSession
{
    private readonly IntPtr _conversation;
    private readonly IPvxEntryPoints _entryPoints;
    private readonly IntPtr _contextRawHandle;
    private SpeculativeConfig? _active;
    private bool _enabled;

    internal NativeConversationSpecSession(IntPtr conversation, IPvxEntryPoints entryPoints,
        IntPtr contextRawHandle)
    {
        _conversation = conversation;
        _entryPoints = entryPoints;
        _contextRawHandle = contextRawHandle;
    }

    /// <summary>Live mirror: the config the session was enabled with (null
    /// when not enabled). Set only after native success.</summary>
    internal SpeculativeConfig? ActiveConfig => _enabled ? _active : null;
    internal bool Enabled => _enabled;

    internal void Enable(SpeculativeConfig config)
    {
        ArgumentNullException.ThrowIfNull(config);
        // Same pinned-HGlobal discipline as the executor lane: the companion /
        // draft / cache paths outlive the native call only.
        var draftPtr = config.DraftModelPath is null ? IntPtr.Zero : PvxNative.AllocUTF8(config.DraftModelPath);
        var cachePtr = config.Ngram?.CachePath is null ? IntPtr.Zero : PvxNative.AllocUTF8(config.Ngram.CachePath);
        var eaglePtr = config.EagleCompanionPath is null ? IntPtr.Zero : PvxNative.AllocUTF8(config.EagleCompanionPath);
        var dflashPtr = config.DflashCompanionPath is null ? IntPtr.Zero : PvxNative.AllocUTF8(config.DflashCompanionPath);
        try
        {
            var nativeParams = new PvxSpecParams(config, draftPtr, cachePtr, eaglePtr, dflashPtr);
            _entryPoints.ConversationSpecInit(_conversation, in nativeParams)
                .ThrowIfError(_entryPoints, _contextRawHandle); // F16
            _active = config; // set only after native success
            _enabled = true;
        }
        finally
        {
            if (draftPtr != IntPtr.Zero) System.Runtime.InteropServices.Marshal.FreeHGlobal(draftPtr);
            if (cachePtr != IntPtr.Zero) System.Runtime.InteropServices.Marshal.FreeHGlobal(cachePtr);
            if (eaglePtr != IntPtr.Zero) System.Runtime.InteropServices.Marshal.FreeHGlobal(eaglePtr);
            if (dflashPtr != IntPtr.Zero) System.Runtime.InteropServices.Marshal.FreeHGlobal(dflashPtr);
        }
    }

    internal void Disable()
    {
        _entryPoints.ConversationSpecFree(_conversation); // idempotent, void
        _active = null;
        _enabled = false;
    }

    /// <summary>Pool return / context teardown: the native side frees the
    /// session in the existing teardown order — drop the managed mirror so it
    /// can never diverge.</summary>
    internal void OnReturned()
    {
        _active = null;
        _enabled = false;
    }
}