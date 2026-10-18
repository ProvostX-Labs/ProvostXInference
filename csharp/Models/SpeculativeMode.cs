namespace ProvostXInference.Models;

/// <summary>Speculative-decoding mode, values mirror pvx_spec_mode_t exactly.
/// Fail-closed rule: a configured path that cannot be honored (missing/unloadable
/// draft model, missing capability, wrong vocab family) is a typed error naming
/// the boundary — never a silent fallback to non-spec generation.</summary>
public enum SpeculativeMode
{
    Off = 0,
    /// <summary>Model-native MTP drafting — requires an mtp capability; enabling
    /// currently fails typed NOT_SUPPORTED while the MTP wiring boundary stands.</summary>
    Mtp = 1,
    /// <summary>Second draft model (upstream draft-simple) — requires a
    /// draft_simple capability plus a user-provided DraftModelPath.</summary>
    Draft = 2,
}
