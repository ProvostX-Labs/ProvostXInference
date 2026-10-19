using System.Runtime.InteropServices;
using ProvostXInference.Interop;
using Xunit;

namespace ProvostXInference.Tests;

/// <summary>Pins the blittable C# mirror of pvx_context_params_t to the
/// native ABI (ABI v8 wave-4: 5 uint32 fields = 20 bytes). ABI drift here
/// would read out of bounds natively — the size and field offsets are the
/// contract.</summary>
public class PvxContextParamsMarshalTests
{
    [Fact]
    public void MarshalSizeMatchesNativeStruct()
    {
        // pvx_context_params_t = 5 x uint32 = 20 bytes.
        Assert.Equal(20, Marshal.SizeOf<PvxContextParams>());
    }

    [Fact]
    public void FieldsAreSequentialAtNativeOffsets()
    {
        Assert.Equal(0, Marshal.OffsetOf<PvxContextParams>(nameof(PvxContextParams.ContextSize)));
        Assert.Equal(4, Marshal.OffsetOf<PvxContextParams>(nameof(PvxContextParams.BatchSize)));
        Assert.Equal(8, Marshal.OffsetOf<PvxContextParams>(nameof(PvxContextParams.SeqMax)));
        Assert.Equal(12, Marshal.OffsetOf<PvxContextParams>(nameof(PvxContextParams.PoolingType)));
        Assert.Equal(16, Marshal.OffsetOf<PvxContextParams>(nameof(PvxContextParams.NRsSeq)));
    }

    [Fact]
    public void ConstructorCarriesNRsSeq()
    {
        var p = new PvxContextParams(4096, 512, 4, PvxPoolingType.None, 8);
        Assert.Equal(8u, p.NRsSeq);
        Assert.Equal(0u, new PvxContextParams(1, 1, 1, PvxPoolingType.None).NRsSeq);
    }

    [Fact]
    public void ContextConfigDefaultsNRsSeqToZeroAndCarriesIt()
    {
        Assert.Equal(0u, new ProvostXInference.Models.ContextConfig().NRsSeq);
        Assert.Equal(16u, new ProvostXInference.Models.ContextConfig { NRsSeq = 16 }.NRsSeq);
    }
}