using System.IO;
using Xunit;

namespace ProvostXInference.Tests;

/// <summary>
/// Cross-platform live-model test tier (INF BL-47). A small real GGUF model is
/// the standard fixture for live journey lanes: discovery is env-driven with
/// platform-neutral defaults, every lane runs on macOS/Linux/Windows, and a
/// missing fixture yields a named skip — never a silent pass and never a
/// platform-specific unconditional skip. RAM gate: the default fixture is a
/// ≤1 GB model, so the >60% free-RAM rule is satisfied on CI-class runners.
/// </summary>
public static class LiveModelFixtures
{
    private static string? Resolve(string envVar, params string[] relativeDefaults)
    {
        // Env wins when set AND the file exists (CI mounts fixtures explicitly).
        var env = Environment.GetEnvironmentVariable(envVar);
        if (!string.IsNullOrEmpty(env) && File.Exists(env)) return env;

        // BL-48: try each default in order — Tests/models symlinks first
        // (host-canonical store), then the repo-root models/ dir (how the
        // Linux/Windows lab VMs are provisioned: real files, no symlinks).
        // Walk upward from the test assembly to the repo root so the same
        // relative defaults work regardless of the build layout.
        var dir = AppContext.BaseDirectory;
        for (int i = 0; i < 10 && dir is not null; i++)
        {
            foreach (var rel in relativeDefaults)
            {
                var candidate = Path.Combine(dir, rel);
                if (File.Exists(candidate)) return candidate;
            }
            dir = Path.GetDirectoryName(Path.TrimEndingDirectorySeparator(dir));
        }
        return null;
    }

    /// <summary>Small dense generative model (~1B, ≤1 GB) — the standard live lane fixture.</summary>
    public static string? SmallModel =>
        Resolve("PVX_SMALL_MODEL_PATH",
            Path.Combine("models", "Llama-3.2-1B-Instruct-Q4_K_M.gguf"),
            Path.Combine("models", "Qwen3.5-4B-Q4_K_M.gguf"))
        ?? Resolve("PVX_MODEL_PATH",
            Path.Combine("models", "Llama-3.2-1B-Instruct-Q4_K_M.gguf"),
            Path.Combine("models", "Qwen3.5-4B-Q4_K_M.gguf"));

    /// <summary>Speculative draft model — must share the target's vocab family.</summary>
    public static string? DraftModel =>
        Resolve("PVX_SMALL_DRAFT_MODEL_PATH", Path.Combine("models", "spec-draft-1b.gguf"), Path.Combine("models", "Llama-3.2-1B-Instruct-Q4_K_M.gguf"))
        ?? Resolve("PVX_DRAFT_MODEL_PATH", Path.Combine("models", "spec-draft-1b.gguf"), Path.Combine("models", "Llama-3.2-1B-Instruct-Q4_K_M.gguf"));

    /// <summary>Embedding encoder (MiniLM-class).</summary>
    public static string? EmbedModel =>
        Resolve("PVX_SMALL_EMBED_MODEL_PATH", Path.Combine("models", "all-MiniLM-L6-v2-Q5_K_M.gguf"));

    /// <summary>Vision projector (mmproj) — pairs with a multimodal backbone.</summary>
    public static string? MmprojModel =>
        Resolve("PVX_SMALL_MMPROJ_PATH", Path.Combine("models", "mmproj-Qwen3.5-4B-BF16.gguf"));

    /// <summary>Multimodal backbone for the vision projector (LLM repo fixture contract).</summary>
    public static string? HybridModel =>
        Resolve("PVX_SMALL_HYBRID_MODEL_PATH", Path.Combine("models", "Qwen3.5-4B-Q4_K_M.gguf"));

    /// <summary>
    /// RAM safety gate (house rule): refuse to load when free memory cannot
    /// comfortably hold the model. Unknown limits allow (fixture is ≤1 GB by
    /// contract); the gate itself must never break a lane.
    /// </summary>
    public static bool RamGateAllows(double modelBytes)
    {
        try
        {
            var gcMem = GC.GetGCMemoryInfo();
            var total = gcMem.TotalAvailableMemoryBytes;
            if (total <= 0) return true;
            var freeEstimate = total - gcMem.MemoryLoadBytes;
            return modelBytes * 1.5 < freeEstimate * 0.6;
        }
        catch
        {
            return true;
        }
    }
}

/// <summary>
/// Fact whose lane requires a small real model. Skips with a named reason when
/// the fixture is absent — works identically on every OS. Model discovery is
/// shared with <see cref="LiveModelFixtures"/>.
/// </summary>
public sealed class LiveModelFactAttribute : FactAttribute
{
    public LiveModelFactAttribute()
    {
        if (LiveModelFixtures.SmallModel is null)
            Skip = "live-lane: no small model fixture on this host (set PVX_SMALL_MODEL_PATH)";
    }
}

/// <summary>
/// Shared helpers for live lanes: fixture require-with-skip and a
/// generation-loop kernel used by several journey tests.
/// </summary>
public static class LiveLane
{
    /// <summary>Returns the fixture path, or null when the lane must skip.</summary>
    public static string? Optional(string? path)
    {
        return path is not null && File.Exists(path) ? path : null;
    }
}
