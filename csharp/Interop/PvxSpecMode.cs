namespace ProvostXInference.Interop;

/// <summary>Mirrors pvx_spec_mode_t. C enums are int-sized; values in lockstep
/// with include/pvxinference.h.</summary>
internal enum PvxSpecMode : int
{
    Off = 0,
    /// <summary>Model-native MTP drafting — capability is reported per model,
    /// enabling fails typed NOT_SUPPORTED while the MTP wiring boundary stands.</summary>
    MtpAuto = 1,
    /// <summary>Second draft model (upstream draft-simple); requires a
    /// draft_model_path and a draft_simple capability.</summary>
    Draft = 2,
    // ABI v6 appends — n-gram drafting (no draft model; drafts from the
    // committed stream + online caches).
    NgramSimple = 3,
    NgramMapK = 4,
    NgramMod = 5,
    NgramCache = 6,
    // ABI v7 appends — companion-GGUF draft-graph modes (upstream
    // draft-eagle3 / draft-dflash). Eagle3 wires the sequential executor
    // lane; the batched/pool lane refuses typed. DFlash validates companion
    // metadata and fails typed at the wiring boundary — never faked.
    Eagle3 = 7,
    Dflash = 8,
    /// <summary>ABI v8 wave 4 append — separate self-contained draft-MTP head
    /// GGUF (MtpDraftPath); embedded-nextn drafting (MtpAuto) stays a typed
    /// wiring boundary.</summary>
    MtpDraft = 9,
}
