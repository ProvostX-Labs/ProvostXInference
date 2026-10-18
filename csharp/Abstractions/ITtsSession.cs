using ProvostXInference.Models;

namespace ProvostXInference.Abstractions;

/// <summary>
/// ABI v8 (wave 3, feature 7): a TTS session — a SEPARATE decode context on
/// the backbone speech model plus a gen-audio mmproj. Never entangled with a
/// conversation context (KV-neutral by construction). Capability-truthful:
/// creation against a mmproj without a gen-audio pipeline throws typed.
/// Determinism: identical (text, request.Seed) pairs replay byte-identically.
/// </summary>
public interface ITtsSession : IDisposable
{
    TtsResult Generate(TtsRequest request, string text);
}
