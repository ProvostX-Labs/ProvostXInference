namespace ProvostXInference.Abstractions;

/// <summary>
/// ABI v9: one function tool definition passed to the model's chat template.
/// OpenAI function shape; ParametersJson is the raw JSON Schema string
/// (null/empty = schema-less). Pure data — the model's embedded template owns
/// the rendering format. Stateless immutable record.
/// </summary>
public sealed record InferenceToolSpec(string Name, string Description, string ParametersJson);
