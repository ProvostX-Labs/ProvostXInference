namespace ProvostXInference.Decision.Engine;

/// <summary>
/// One judged decision record, validated and normalized (parser → executor boundary).
/// SHELL records carry command+cwd; PROCESS records carry the os/exe/chain/cmd/net/writes
/// tuple. The untrusted SHELL "purpose" field deliberately has NO field here — it is
/// echoed in audit logs only and never reaches decision input.
/// </summary>
public sealed class DecisionRecord
{
    public JudgeRecordKind Kind { get; init; }

    // ── SHELL kind ──
    public string Command { get; init; } = string.Empty;
    public string Cwd { get; init; } = string.Empty;

    // ── PROCESS kind ──
    public string Os { get; init; } = string.Empty;
    public string Exe { get; init; } = string.Empty;
    public string Cmd { get; init; } = string.Empty;
    public string Net { get; init; } = string.Empty;
    public string Writes { get; init; } = string.Empty;

    /// <summary>Process ancestry, grandparent → parent → child (validated: no empty segments).</summary>
    public IReadOnlyList<string> Chain { get; init; } = Array.Empty<string>();
}