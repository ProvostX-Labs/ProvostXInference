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
    /// <summary>ABI v6 — n-gram drafting from the committed stream: no draft
    /// model. History-pattern lookup (upstream ngram-simple).</summary>
    NgramSimple = 3,
    /// <summary>ABI v6 — n-gram -> m-gram map with per-value accept statistics
    /// (upstream ngram-map-k).</summary>
    NgramMapK = 4,
    /// <summary>ABI v6 — MOD: hashed (n-1)-suffix match, one deviation
    /// (upstream ngram-mod).</summary>
    NgramMod = 5,
    /// <summary>ABI v6 — static (file) + dynamic token-probability caches
    /// (upstream ngram-cache).</summary>
    NgramCache = 6,
    /// <summary>ABI v7 — EAGLE3 companion draft decoder fed (token, g_embd)
    /// pairs from target layer features (upstream draft-eagle3). Requires an
    /// eagle3 companion GGUF (EagleCompanionPath); sequential executor lane
    /// only — the batched/pool lane refuses typed.</summary>
    Eagle3 = 7,
    /// <summary>ABI v7 — DFlash block-diffusion fused drafting (upstream
    /// draft-dflash). Requires a dflash companion GGUF (DflashCompanionPath);
    /// metadata validated, drafting unwired — typed boundary.</summary>
    Dflash = 8,
    /// <summary>ABI v8 wave 4 — SEPARATE self-contained draft-MTP head GGUF
    /// (upstream draft-mtp, llama.cpp PR #28243; qwen4exp-family targets).
    /// Requires an MTP head GGUF (MtpDraftPath — never DraftModelPath);
    /// drafts on the target trunk's nextn embeddings, verified greedy-exact by
    /// the target. Sequential executor lane only — the batched/pool lane
    /// refuses typed.</summary>
    MtpDraft = 9,
}
