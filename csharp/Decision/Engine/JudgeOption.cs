namespace ProvostXInference.Decision.Engine;

/// <summary>
/// One rendered option of a judge question: its answer label and its
/// criterion description (null = no description — rendered as the bare label).
/// Option ORDER is decision-meaningful (argmax index → label). Pure data.
/// </summary>
public sealed record JudgeOption(string Label, string? Description);