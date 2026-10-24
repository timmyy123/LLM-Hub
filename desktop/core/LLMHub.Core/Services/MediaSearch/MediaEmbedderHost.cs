namespace LLMHub.Services.MediaSearch;

/// <summary>
/// The one EmbeddingGemma 2 (.litertlm) engine in memory, shared by Studio › Search and /v1/embeddings image and audio
/// inputs, so switching between them never loads a second copy.
/// </summary>
public sealed class MediaEmbedderHost
{
    private readonly SemaphoreSlim _gate = new(1, 1);
    private LiteRtEngine.LiteRtEmbedder? _embedder;

    public LiteRtEngine.LiteRtEmbedder? Loaded => _embedder;

    public async Task<LiteRtEngine.LiteRtEmbedder> GetAsync(InstalledModel model, Action<string> status, CancellationToken ct)
    {
        await _gate.WaitAsync(ct);
        try
        {
            if (_embedder?.Model.Id == model.Id)
                return _embedder;
            _embedder?.Dispose();
            _embedder = null;
            status(Loc.F("LoadingModel", model.Name));
            _embedder = await Ioc.LiteRt.OpenEmbedderAsync(model, status, ct);
            return _embedder;
        }
        finally
        {
            _gate.Release();
        }
    }

    public async Task UnloadAsync()
    {
        await _gate.WaitAsync();
        try
        {
            _embedder?.Dispose();
            _embedder = null;
        }
        finally
        {
            _gate.Release();
        }
    }
}
