using System.IO;
using ProvostXInference.Abstractions;
using ProvostXInference.Exceptions;
using ProvostXInference.Implementation;
using ProvostXInference.Interop;
using ProvostXInference.Models;
using ProvostXInference.Engines;
using ProvostXInference.Tests.Engines;
using Xunit;

namespace ProvostXInference.Tests;

/// <summary>
/// JSON-schema → GBNF converter surface (ABI v7, feature 11): the native
/// pattern-port of the vendored converter exposed through the model/facade
/// with a GrammarKind.JsonSchema provenance label. Golden-corpus discipline:
/// the decision-envelope schema's GBNF is pinned byte-exact against
/// tests/fixtures/decision-envelope.gbnf (the same fixture the native suite
/// pins). Fail-closed: invalid JSON and unsupported keywords are typed
/// refusals naming the boundary — never a loose best-effort grammar.
/// Live legs are env-gated (PVX_MODEL_PATH) like the spec suite.
/// </summary>
public class JsonSchemaGrammarTests
{
    private static readonly string? ModelPath = Environment.GetEnvironmentVariable("PVX_MODEL_PATH");
    private static bool HasModel() => ModelPath is not null && File.Exists(ModelPath);

    private static NativeInferenceModel LoadTarget() =>
        NativeInferenceModel.Load(new ModelConfig { Path = ModelPath!, GpuLayers = 0 });

    /// <summary>Locates the repo-pinned native fixture by walking up from the
    /// test bin directory (relative paths only — no absolute repo roots in
    /// product code or tests).</summary>
    private static string? FindFixture(string relative)
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null)
        {
            var candidate = Path.Combine(dir.FullName, relative);
            if (File.Exists(candidate)) return candidate;
            if (dir.Name == "csharp" && dir.Parent is not null)
            {
                var sibling = Path.Combine(dir.Parent.FullName, relative);
                if (File.Exists(sibling)) return sibling;
            }
            dir = dir.Parent!;
        }
        return null;
    }

    private const string DecisionEnvelopeSchema = """
        {
            "type": "object",
            "properties": {
                "thinking": { "type": "string" },
                "answer": { "type": ["string", "null"] },
                "toolcalls": { "type": ["array", "null"],
                    "items": { "type": "object",
                        "properties": {
                            "name": { "type": "string" },
                            "args": { "type": "object", "additionalProperties": { "type": "string" } }
                        },
                        "required": ["name", "args"] } }
            },
            "required": ["thinking"]
        }
        """;

    [Fact]
    public void Converter_Output_Is_Pinned_Byte_Exact_Against_Fixture()
    {
        if (!HasModel()) return; // skip when no test model
        var fixture = FindFixture("tests/fixtures/decision-envelope.gbnf");
        if (fixture is null) return; // fixture not present (out-of-repo run)
        using var model = LoadTarget();
        // BYTE-EXACT against the same fixture the native suite pins
        // (tests/fixtures/decision-envelope.gbnf).
        var want = File.ReadAllText(fixture);
        var gbnf = model.JsonSchemaToGbnf(DecisionEnvelopeSchema, null);
        Assert.Equal(want, gbnf);
        // and the compile path labels the grammar schema-derived
        using var grammar = (NativeGrammar) model.CreateGrammarFromJsonSchema(DecisionEnvelopeSchema);
        Assert.Equal(GrammarKind.JsonSchema, grammar.Kind);
    }

    [Fact]
    public void Converter_Is_Consumable_By_The_Plain_Grammar_Chain()
    {
        if (!HasModel()) return;
        using var model = LoadTarget();
        var gbnf = model.CreateGrammarFromJsonSchema(DecisionEnvelopeSchema);
        Assert.Equal(GrammarKind.JsonSchema, ((NativeGrammar) gbnf).Kind);
        gbnf.Dispose();
    }

    [Fact]
    public void Invalid_Json_Types_Refusal_Naming_Parse_Boundary()
    {
        if (!HasModel()) return;
        using var model = LoadTarget();
        var ex = Assert.Throws<InferenceException>(
            () => model.CreateGrammarFromJsonSchema("{ not json at all"));
        Assert.Equal((int) PvxResult.ErrInvalidArg, ex.ErrorCode);
        Assert.NotEmpty(ex.Message);
    }

    [Fact]
    public void Unsupported_Keyword_Types_Refusal_Naming_The_Keyword()
    {
        if (!HasModel()) return;
        using var model = LoadTarget();
        // "pattern" (regex) is per-keyword ported: unsupported -> typed refusal
        var schema = """{ "type": "string", "pattern": "^a+$" }""";
        var ex = Assert.Throws<InferenceException>(() => model.CreateGrammarFromJsonSchema(schema));
        Assert.Equal((int) PvxResult.ErrNotSupported, ex.ErrorCode);
        Assert.Contains("pattern", ex.Message);
    }

    [Fact]
    public void Null_Schema_Refuses_Typed()
    {
        if (!HasModel()) return;
        using var model = LoadTarget();
        Assert.Throws<ArgumentNullException>(() => model.CreateGrammarFromJsonSchema(null!));
    }

    [Fact]
    public void Facade_Wires_The_Converter_With_JsonSchema_Provenance()
    {
        // pure doubles: the facade routes through the model converter verbatim
        var model = new FakeEngineModel();
        var backend = new FakeEngineBackend();
        using var ops = new GgufEngineOps(model, new ContextConfig(), new SamplingConfig(), backend);
        using var grammar = ops.CreateGrammarFromJsonSchema(DecisionEnvelopeSchema);
        Assert.Equal(GrammarKind.JsonSchema, grammar.Kind);
    }
}