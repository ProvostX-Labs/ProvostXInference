namespace ProvostXInference.Abstractions;

public interface IGrammar : IDisposable
{
    /// <summary>Provenance of this grammar's GBNF text: hand-written GBNF
    /// (Gbnf) or converted from a JSON Schema document (JsonSchema, ABI v7 —
    /// the converter validated the schema at creation). Pure label — no
    /// behavioral branch on this value.</summary>
    GrammarKind Kind { get; }
}

/// <summary>Provenance label of an IGrammar (ABI v7).</summary>
public enum GrammarKind
{
    Gbnf = 0,
    JsonSchema = 1,
}
