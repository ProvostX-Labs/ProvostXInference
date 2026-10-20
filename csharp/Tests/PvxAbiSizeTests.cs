using System.Runtime.InteropServices;
using ProvostXInference.Interop;
using Xunit;

namespace ProvostXInference.Tests;

/// <summary>ABI v8 wave-5 (BL-21): C# mirrors must byte-match the NATIVE
/// sizeof — native sizes pinned by the C-side probe (see ARCHITECTURE.md
/// addendum): model=72 ctx=112 smp=112 w5=72 lb=8 kv=288. Any field change
/// in either tree must bump BOTH.</summary>
[Trait("Tier","Unit")]
public sealed class PvxAbiSizeTests
{
    [Fact]
    public void Mirrors_Match_Native_Sizeof()
    {
        Assert.Equal(72,  Marshal.SizeOf<PvxModelParams>());
        Assert.Equal(112, Marshal.SizeOf<PvxContextParams>());
        Assert.Equal(112, Marshal.SizeOf<PvxSamplingParams>());
        Assert.Equal(72,  Marshal.SizeOf<PvxSamplingParamsWave5>());
        Assert.Equal(8,   Marshal.SizeOf<PvxLogitBias>());
        Assert.Equal(288, Marshal.SizeOf<PvxKvOverride>());
    }

    [Fact]
    public void Wave5_Defaults_Are_Zeroed()
    {
        // Zero-value semantics: default construction = pre-wave-5 behavior.
        var w = new PvxSamplingParamsWave5();
        Assert.Equal(0L, w.Seed);
        Assert.Equal(0f, w.FrequencyPenalty);
        Assert.Equal(0, w.NLogitBias);
        Assert.Equal(IntPtr.Zero, w.LogitBias);
    }

    [Fact]
    public void KvOverride_String_RoundTrip()
    {
        var o = new PvxKvOverride("<arch>.rope.freq_base", 2, valF64: 10000d);
        Assert.Equal(128, o.Key.Length);
        Assert.Equal((byte)'<', o.Key[0]);
    }
}
