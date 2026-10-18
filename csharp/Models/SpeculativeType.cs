namespace ProvostXInference.Models;

/// <summary>Supported speculative-decoding types, mirrored from the llama.cpp
/// speculative enum via the engine's self-report (PVX_SPEC_TYPE_* bits).
/// The ENGINE decides what the loaded MODEL supports — query
/// IInferenceModel.SpeculativeCapabilities per model load and offer exactly
/// these types for sequential mode; never assume.</summary>
[Flags]
public enum SpeculativeType
{
    None = 0,
    /// <summary>Second draft model (upstream draft-simple) — works for any
    /// dense, standard-vocab target when a user-provided draft model is given.</summary>
    DraftSimple = 0x1,
    /// <summary>Model carries its own MTP (nextn) layers. Reported per model;
    /// enabling currently fails typed NOT_SUPPORTED (MTP wiring boundary).</summary>
    Mtp = 0x2,
    /// <summary>Per-target companion drafts — not detectable from the target
    /// GGUF alone and not wired in this slice.</summary>
    Eagle3 = 0x4,
    Dflash = 0x8,
    Dspark = 0x10,
    /// <summary>ABI v6 — n-gram simple: history-pattern lookup. Reported for
    /// every standard-vocab target (model-independent drafter).</summary>
    Ngram = 0x20,
    /// <summary>ABI v6 — n-gram map-k: n-gram -> m-gram map with accept stats.</summary>
    NgramMapK = 0x40,
    /// <summary>ABI v6 — n-gram MOD: hashed (n-1)-suffix match, one deviation.</summary>
    NgramMod = 0x80,
    /// <summary>ABI v6 — n-gram cache: static + dynamic token-probability caches.</summary>
    NgramCache = 0x100,
}
