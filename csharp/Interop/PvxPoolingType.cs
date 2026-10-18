namespace ProvostXInference.Interop;

/// <summary>Maps pvx_pooling_type_t.</summary>
internal enum PvxPoolingType : int
{
    None = 0,
    Mean = 1,
    Cls = 2,
    Last = 3,
    /// <summary>ABI v8: RANK pooling attaches the model's classification head
    /// (rerank lane). pvx_rerank refuses typed on any other pooling.</summary>
    Rank = 4,
}
