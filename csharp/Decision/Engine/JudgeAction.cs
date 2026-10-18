namespace ProvostXInference.Decision.Engine;

/// <summary>
/// Judge verdict actions. The FINAL action in the wire response is produced by
/// <c>JudgeVerdictMapper</c>, which owns the risk-escalation rule — the executor's
/// raw action may be upgraded (allow → require_approval) but never downgraded.
/// </summary>
public enum JudgeAction
{
    Allow = 0,
    Sandbox = 1,
    Deny = 2,
    RequireApproval = 3,
}