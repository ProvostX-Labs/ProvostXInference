namespace ProvostXInference.Engines;

/// <summary>
/// Facade view of the offline engine-maintenance utilities (owner black-box
/// directive 2026-10-19): file-level operations that need NO loaded engine
/// object but ARE engine-bound (they dispatch on the ACTIVE engine's
/// per-handle tables). Currently: GGUF tokenizer-metadata repair (EOG/special
/// token fixup — weights stream-copied byte-for-byte, metadata rewritten).
/// Pure delegation to the Implementation layer — callers never leave the
/// facade surface.
/// </summary>
public static class EngineMaintenance
{
    /// <summary>Repair (or verify) a GGUF's EOG/special-token metadata.
    /// Returns (ok, report). Requires an active engine (per-handle dispatch
    /// is model-scoped — throws when no engine has been activated).</summary>
    public static (bool Ok, string Report) RepairGgufEogMetadata(string inPath, string outPath)
        => Implementation.GgufMetadataRepair.Repair(inPath, outPath);
}
