namespace ProvostXInference.Interop;

/// <summary>Maps pvx_result_t. C enums are int-sized; underlying type is explicit for clarity.</summary>
internal enum PvxResult : int
{
    Ok = 0,
    ErrInvalidArg = 1,
    ErrInvalidHandle = 2,
    ErrLoadFailed = 3,
    ErrNoSlot = 4,
    ErrDecodeFailed = 5,
    ErrOverflow = 6,
    ErrNotSupported = 7,
    /// <summary>Partial rewind refused: hybrid/recurrent memories cannot
    /// partially erase state (see IInferenceModel.PartialRewindSupported).</summary>
    ErrRewindUnsupported = 8,
    ErrInternal = 99,
}
