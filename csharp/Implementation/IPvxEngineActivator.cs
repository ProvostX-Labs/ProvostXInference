using ProvostXInference.Interop;
using ProvostXInference.Models;

namespace ProvostXInference.Implementation;

/// <summary>Engine-activation seam for the model factory: validates the config's
/// engine selection against the OS matrix (fail-closed, platform default when
/// unconfigured) and opens the selected specialist module with its per-handle
/// dispatch tables. Injectable so NativeInferenceModel is unit-testable with
/// in-memory fakes (no native library required).</summary>
internal interface IPvxEngineActivator
{
    IPvxEngineHandle Open(ModelConfig config);
}