using System.ComponentModel;
using CommunityToolkit.Mvvm.ComponentModel;

namespace LLMHub.Services;

/// <summary>
/// One place that loads a model on the right engine (llama.cpp for GGUF, LiteRT-LM for .litertlm) and streams replies.
/// Chat and the Local API both go through here, so only one model is in memory at a time.
/// </summary>
public sealed partial class EngineHub : ObservableObject
{
    private readonly LlamaEngine _llama;
    private readonly LiteRtEngine _liteRt;
    private readonly SemaphoreSlim _gate = new(1, 1);
    private bool _liteRtActive;

    public EngineHub(LlamaEngine llama, LiteRtEngine liteRt)
    {
        _llama = llama;
        _liteRt = liteRt;
        StatusText = _llama.StatusText;
        _llama.PropertyChanged += OnLlamaChanged;
    }

    [ObservableProperty] public partial EngineState State { get; set; }
    [ObservableProperty] public partial string StatusText { get; set; }
    [ObservableProperty] public partial InstalledModel? LoadedModel { get; set; }

    public LlamaEngine Llama => _llama;

    /// <summary>"win-cuda-13.4-x64", "LiteRT-LM gpu", ...</summary>
    public string? Backend => _liteRtActive ? $"LiteRT-LM {_liteRt.Backend}" : _llama.Variant;

    public bool IsReady(InstalledModel model) =>
        LoadedModel?.Id == model.Id && (_liteRtActive
            ? _liteRt.LoadedModel is not null && _liteRt.LoadedWithMtp == model.MtpEnabled
            : _llama.BaseUrl is not null);

    public async Task LoadAsync(InstalledModel model, CancellationToken ct)
    {
        await _gate.WaitAsync(ct);
        try
        {
            if (IsReady(model))
                return;
            if (model.Format == "litertlm")
            {
                _llama.Stop();
                _liteRtActive = true;
                try
                {
                    await _liteRt.LoadAsync(model, text => Set(EngineState.Loading, text), ct);
                    Set(EngineState.Ready, Loc.F("EngineReady", model.Name), model);
                }
                catch (Exception e) when (e is not OperationCanceledException)
                {
                    Log.Error("LiteRT-LM load failed", e);
                    Set(EngineState.Error, Loc.F("EngineError", e.Message));
                    throw;
                }
            }
            else
            {
                _liteRt.Unload();
                _liteRtActive = false;
                await _llama.LoadAsync(model, ct);
            }
        }
        finally
        {
            _gate.Release();
        }
    }

    public IAsyncEnumerable<ChatDelta> StreamAsync(InstalledModel model, IReadOnlyList<ChatTurn> turns, CancellationToken ct, ChatOptions? options = null) =>
        model.Format == "litertlm"
            ? _liteRt.StreamAsync(turns, ct, options)
            : ChatClient.StreamAsync(_llama.BaseUrl!, model.Id, turns, options, ct);

    public void Stop()
    {
        _liteRt.Unload();
        _llama.Stop();
        Set(EngineState.Idle, Loc.S("EngineIdle"));
    }

    private void OnLlamaChanged(object? sender, PropertyChangedEventArgs e)
    {
        if (_liteRtActive)
            return;
        UiThread.Run(() =>
        {
            State = _llama.State;
            StatusText = _llama.StatusText;
            LoadedModel = _llama.LoadedModel;
        });
    }

    private void Set(EngineState state, string text, InstalledModel? model = null) => UiThread.Run(() =>
    {
        State = state;
        StatusText = text;
        LoadedModel = model;
    });
}
