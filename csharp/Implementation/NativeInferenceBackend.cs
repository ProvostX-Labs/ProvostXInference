using System.Runtime.InteropServices;
using ProvostXInference.Abstractions;
using ProvostXInference.Exceptions;
using ProvostXInference.Interop;
using ProvostXInference.Models;

namespace ProvostXInference.Implementation;

/// <summary>Backend safety layer (pvx_probe_backend + queries). The probe is
/// context-anchored and dispatches on the OWNING CONTEXT ENGINE's per-handle
/// tables; the process-wide queries dispatch on the ACTIVE engine — a no-op
/// passthrough until a model load activates it.</summary>
public sealed class NativeInferenceBackend : IInferenceBackend
{
    public BackendInfo Probe(IInferenceContext context)
    {
        var ctx = (NativeInferenceContext)context;
        var ep = ctx.ModelEntryPoints;
        ep.ProbeBackend(ctx.RawHandle).ThrowIfError(ep, ctx.RawHandle); // F16

        var namePtr = ep.BackendName(ctx.RawHandle);
        var name = namePtr != IntPtr.Zero ? Marshal.PtrToStringUTF8(namePtr) ?? "Unknown" : "Unknown";

        var errPtr = ep.BackendProbeError(ctx.RawHandle);
        var error = errPtr != IntPtr.Zero ? Marshal.PtrToStringUTF8(errPtr) : null;

        return new BackendInfo
        {
            Name = name,
            ProbePassed = ep.BackendProbePassed(ctx.RawHandle),
            ProbeError = error,
            NeedsKvFlush = ep.BackendNeedsKvFlush(ctx.RawHandle),
            GpuLayersReduced = ep.BackendGpuLayersReduced(ctx.RawHandle),
            EffectiveGpuLayers = ep.BackendEffectiveGpuLayers(ctx.RawHandle),
            OriginalGpuLayers = ep.BackendOriginalGpuLayers(ctx.RawHandle),
        };
    }

    public bool VramAvailable(long neededBytes)
    {
        return PvxDispatchRegistry.ActiveEntryPoints.VramAvailable(neededBytes);
    }
}