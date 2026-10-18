using ProvostXInference.Models;

namespace ProvostXInference.Models;

/// <summary>The engine's self-reported speculative capabilities for ONE loaded
/// model: a supported-types list (SpeculativeType bitset), not booleans.
/// A report of None means the model supports no native speculative type — the
/// second draft model (SpeculativeMode.Draft with a user-provided draft model)
/// remains the available option whenever DraftSimple is reported.</summary>
public sealed record SpeculativeCapabilities(SpeculativeType Types)
{
    public static readonly SpeculativeCapabilities None = new(SpeculativeType.None);

    public bool Supports(SpeculativeType type) =>
        type != SpeculativeType.None && (Types & type) == type;

    /// <summary>Convenience mirror of the draft_simple bit.</summary>
    public bool Draft => Supports(SpeculativeType.DraftSimple);
    /// <summary>Convenience mirror of the mtp bit.</summary>
    public bool Mtp => Supports(SpeculativeType.Mtp);

    /// <summary>The supported types as the llama.cpp speculative enum names the
    /// LLM/dashboard offers for sequential mode ("draft_simple", "mtp", …).</summary>
    public IReadOnlyList<string> SupportedTypeNames
    {
        get
        {
            var names = new List<string>();
            foreach (var t in new[] { SpeculativeType.DraftSimple, SpeculativeType.Mtp,
                                      SpeculativeType.Eagle3, SpeculativeType.Dflash,
                                      SpeculativeType.Dspark, SpeculativeType.Ngram })
            {
                if (Supports(t)) names.Add(TypeName(t));
            }
            return names;
        }
    }

    /// <summary>llama.cpp speculative enum name for a reported type.</summary>
    public static string TypeName(SpeculativeType type) => type switch
    {
        SpeculativeType.DraftSimple => "draft_simple",
        SpeculativeType.Mtp => "mtp",
        SpeculativeType.Eagle3 => "eagle3",
        SpeculativeType.Dflash => "dflash",
        SpeculativeType.Dspark => "dspark",
        SpeculativeType.Ngram => "ngram",
        _ => "none",
    };
}
