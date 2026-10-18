using System.Text;
using ProvostXInference.Interop;

namespace ProvostXInference.Implementation;

/// <summary>
/// GGUF tokenizer-metadata repair (owner 2026-10-06): some HuggingFace GGUFs omit
/// the EOS token from tokenizer.ggml.special_eog_ids — the engine's native EOS
/// stop then never fires and generation runs past end-of-turn. The native
/// pvx_repair_eog_metadata rewrites the METADATA ONLY (weights stream-copied
/// byte-for-byte); this is the public managed facade over it. Dispatches on
/// the ACTIVE engine's per-handle dispatch tables (BL-02).
/// </summary>
public static class GgufMetadataRepair
{
    /// <summary>Repair (or verify) a GGUF's EOG metadata. Returns (ok, report).</summary>
    public static (bool Ok, string Report) Repair(string inPath, string outPath)
    {
        var buf = new byte[1024];
        var rc = PvxDispatchRegistry.ActiveEntryPoints.RepairEogMetadata(inPath, outPath, buf, buf.Length);
        var report = Encoding.UTF8.GetString(buf).TrimEnd('\0');
        return (rc == PvxResult.Ok, report);
    }
}