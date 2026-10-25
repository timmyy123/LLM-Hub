using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using LLMHub.Services;
using LLMHub.Services.MediaSearch;

namespace LLMHub.ViewModels;

public sealed record PhotoMatch(string Path, float Score)
{
    public string Name => System.IO.Path.GetFileName(Path);
    public string ScoreText => $"{Score:0.00}";
}

public sealed record AudioMoment(int StartMs, int EndMs, float Score)
{
    public string Label => $"{StartMs / 60000}:{StartMs / 1000 % 60:00}";
}

public sealed record AudioMatch(string Path, float Score, IReadOnlyList<AudioMoment> Moments)
{
    public string Name => System.IO.Path.GetFileName(Path);
    public string ScoreText => $"{Score:0.00}";
}

/// <summary>
/// Studio › Search, after the Android app's Photo Search and Audio Search: EmbeddingGemma 2 (440M for photos, 740M for
/// photos and audio) embeds a folder's photos, or 5-second moments of its audio, on this computer; then a description,
/// or a photo, finds the closest ones. Indexes are kept per mode and redone only for new or changed files.
/// </summary>
public sealed partial class MediaSearchViewModel : ObservableObject
{
    private const int PhotoEdge = 512;
    private const int AudioWindowMs = 5_000;
    private const int MaxAudioSeconds = 600;
    private const float SilenceRms = 0.003f;
    private const int MomentsPerFile = 3;
    private const int MaxAudioResults = 40;
    private const int MaxPhotoResults = 60;
    private const int MaxFiles = 20_000;

    private readonly MediaIndex _photos = new(Path.Combine(AppPaths.Root, "media_search", "photos.idx"));
    private readonly MediaIndex _audio = new(Path.Combine(AppPaths.Root, "media_search", "audio.idx"));
    private CancellationTokenSource? _indexing;

    private readonly bool _ready;

    public MediaSearchViewModel()
    {
        Mode = "photos";
        Query = "";
        Status = "";
        var s = AppSettings.Current;
        PhotoFolder = s.MediaPhotoFolder is { Length: > 0 } p ? p : Environment.GetFolderPath(Environment.SpecialFolder.MyPictures);
        AudioFolder = s.MediaAudioFolder is { Length: > 0 } a ? a : Environment.GetFolderPath(Environment.SpecialFolder.MyMusic);
        _photos.Load();
        _audio.Load();
        Ioc.Models.Installed.CollectionChanged += (_, _) => UiThread.Post(RefreshModels);
        RefreshModels();
        _ready = true;
    }

    /// <summary>Downloaded EmbeddingGemma 2 builds that understand images (440M) or images and audio (740M).</summary>
    public ObservableCollection<InstalledModel> Models { get; } = [];
    public ObservableCollection<PhotoMatch> Photos { get; } = [];
    public ObservableCollection<AudioMatch> Audio { get; } = [];

    /// <summary>photos or audio.</summary>
    [ObservableProperty] public partial string Mode { get; set; }
    [ObservableProperty] public partial InstalledModel? SelectedModel { get; set; }
    [ObservableProperty] public partial string PhotoFolder { get; set; }
    [ObservableProperty] public partial string AudioFolder { get; set; }
    [ObservableProperty] public partial string Query { get; set; }
    [ObservableProperty] public partial string Status { get; set; }
    [ObservableProperty] public partial bool IsIndexing { get; set; }
    [ObservableProperty] public partial bool IsSearching { get; set; }
    [ObservableProperty] public partial double Progress { get; set; }

    public bool IsPhotos => Mode == "photos";
    public bool IsAudio => Mode == "audio";
    public string Folder => IsPhotos ? PhotoFolder : AudioFolder;
    public bool HasModel => SelectedModel is not null;
    /// <summary>Audio needs the 740M build; photos work with either.</summary>
    public bool ModelFitsMode => SelectedModel is { } m && (IsPhotos ? m.Tags.Contains("vision") : m.Tags.Contains("audio"));
    public string IndexedText => Loc.F(IsPhotos ? "SearchIndexedPhotos" : "SearchIndexedAudio", Index.FileCount.ToString("N0"));
    private MediaIndex Index => IsPhotos ? _photos : _audio;

    partial void OnModeChanged(string value)
    {
        OnPropertyChanged(nameof(IsPhotos));
        OnPropertyChanged(nameof(IsAudio));
        OnPropertyChanged(nameof(Folder));
        OnPropertyChanged(nameof(IndexedText));
        OnPropertyChanged(nameof(ModelFitsMode));
        // Pick a build that can do this mode.
        if (!ModelFitsMode && Models.FirstOrDefault(m => IsPhotos ? m.Tags.Contains("vision") : m.Tags.Contains("audio")) is { } better)
            SelectedModel = better;
        Status = "";
    }

    partial void OnSelectedModelChanged(InstalledModel? value)
    {
        OnPropertyChanged(nameof(HasModel));
        OnPropertyChanged(nameof(ModelFitsMode));
        // Choosing a model is the user's; loading the view model must not rewrite settings.
        if (value is not null && _ready)
        {
            AppSettings.Current.MediaModelId = value.Id;
            AppSettings.Save();
        }
    }

    partial void OnPhotoFolderChanged(string value)
    {
        OnPropertyChanged(nameof(Folder));
        if (!_ready)
            return;
        AppSettings.Current.MediaPhotoFolder = value;
        AppSettings.Save();
    }

    partial void OnAudioFolderChanged(string value)
    {
        OnPropertyChanged(nameof(Folder));
        if (!_ready)
            return;
        AppSettings.Current.MediaAudioFolder = value;
        AppSettings.Save();
    }

    public void SetFolder(string folder)
    {
        if (IsPhotos)
            PhotoFolder = folder;
        else
            AudioFolder = folder;
    }

    private void RefreshModels()
    {
        var models = Ioc.Models.Installed.Where(m => m.IsEmbedding && m.Format == "litertlm" && m.Tags.Contains("vision"))
            .OrderByDescending(m => m.Tags.Contains("audio")).ToList();
        Models.Clear();
        foreach (var m in models)
            Models.Add(m);
        SelectedModel = models.FirstOrDefault(m => m.Id == SelectedModel?.Id)
                        ?? models.FirstOrDefault(m => m.Id == AppSettings.Current.MediaModelId)
                        ?? models.FirstOrDefault();
    }

    private Task<LiteRtEngine.LiteRtEmbedder> EmbedderAsync(CancellationToken ct) =>
        Ioc.MediaEmbedder.GetAsync(SelectedModel ?? throw new InvalidOperationException(Loc.S("SearchNoModel")), text => UiThread.Post(() => Status = text), ct);

    public Task UnloadAsync() => Ioc.MediaEmbedder.UnloadAsync();

    // ---------------------------------------------------------------- indexing

    [RelayCommand]
    private async Task IndexAsync()
    {
        if (IsIndexing || SelectedModel is not { } model)
            return;
        if (!ModelFitsMode)
        {
            Status = Loc.S("SearchAudioNeeds740");
            return;
        }
        var folder = Folder;
        if (!Directory.Exists(folder))
        {
            Status = Loc.F("SearchNoFolder", folder);
            return;
        }
        var photos = IsPhotos;
        var index = Index;
        _indexing = new CancellationTokenSource();
        var ct = _indexing.Token;
        IsIndexing = true;
        Progress = 0;
        var clock = System.Diagnostics.Stopwatch.StartNew();
        var added = 0;
        try
        {
            var embedder = await EmbedderAsync(ct);
            if (index.ModelId != model.Id)
                index.Reset(model.Id);
            Status = Loc.S("SearchScanning");
            var extensions = photos ? MediaDecoders.ImageExtensions : MediaDecoders.AudioExtensions;
            var files = await Task.Run(() => Scan(folder, extensions), ct);
            // Files that were deleted or are no longer under the folder leave the index.
            foreach (var gone in index.Files.Where(f => !files.Contains(f)).ToList())
                index.Remove(gone);
            var todo = files.Where(f => !index.IsCurrent(f, Stamp(f))).ToList();
            for (var i = 0; i < todo.Count; i++)
            {
                ct.ThrowIfCancellationRequested();
                var file = todo[i];
                var vectors = await Task.Run(() => photos ? EmbedPhotoAsync(embedder, file, ct) : EmbedAudioAsync(embedder, file, ct), ct);
                index.Put(file, Stamp(file), vectors);
                added++;
                if (added % 20 == 0)
                    index.SaveIfDirty();
                var done = i + 1;
                UiThread.Post(() =>
                {
                    // Updates still queued when indexing ends must not cover the final message.
                    if (!IsIndexing)
                        return;
                    Progress = (double)done / todo.Count;
                    Status = Loc.F("SearchIndexing", done.ToString("N0"), todo.Count.ToString("N0"));
                });
            }
            index.SaveIfDirty();
            Status = Loc.F("SearchIndexDone", files.Count.ToString("N0"), clock.Elapsed.TotalSeconds.ToString("F0"));
            Log.Info($"Media search {(photos ? "photos" : "audio")}: {added} new of {files.Count} in {folder} with {model.Id} on {embedder.Backend} in {clock.Elapsed.TotalSeconds:F0}s");
        }
        catch (OperationCanceledException)
        {
            index.SaveIfDirty();
            Status = Loc.S("SearchStopped");
        }
        catch (Exception e)
        {
            Log.Error("Media indexing failed", e);
            index.SaveIfDirty();
            Status = Loc.F("ErrorPrefix", e.Message);
        }
        finally
        {
            IsIndexing = false;
            Progress = 0;
            OnPropertyChanged(nameof(IndexedText));
            _indexing.Dispose();
            _indexing = null;
        }
    }

    [RelayCommand]
    private void StopIndexing() => _indexing?.Cancel();

    private static HashSet<string> Scan(string folder, string[] extensions)
    {
        var options = new EnumerationOptions { RecurseSubdirectories = true, IgnoreInaccessible = true, AttributesToSkip = FileAttributes.Hidden | FileAttributes.System };
        return Directory.EnumerateFiles(folder, "*", options)
            .Where(f => extensions.Contains(Path.GetExtension(f).ToLowerInvariant()))
            .Take(MaxFiles)
            .ToHashSet();
    }

    private static long Stamp(string file)
    {
        var info = new FileInfo(file);
        return info.Exists ? info.LastWriteTimeUtc.Ticks ^ info.Length : 0;
    }

    private static async Task<IEnumerable<MediaVector>> EmbedPhotoAsync(LiteRtEngine.LiteRtEmbedder embedder, string file, CancellationToken ct)
    {
        var jpeg = await MediaDecoders.Image(file, PhotoEdge, ct);
        var vector = jpeg is null ? null : embedder.EmbedImages([jpeg]);
        return vector is null ? [] : [new MediaVector(file, 0, 0, vector)];
    }

    /// <summary>One vector per 5-second moment, skipping silence, for the first 10 minutes of the file.</summary>
    private static async Task<IEnumerable<MediaVector>> EmbedAudioAsync(LiteRtEngine.LiteRtEmbedder embedder, string file, CancellationToken ct)
    {
        var samples = await MediaDecoders.Audio(file, MaxAudioSeconds, ct);
        if (samples is null || samples.Length < MediaDecoders.SampleRate)
            return [];
        var window = MediaDecoders.SampleRate * AudioWindowMs / 1000;
        var vectors = new List<MediaVector>();
        for (var start = 0; start < samples.Length; start += window)
        {
            ct.ThrowIfCancellationRequested();
            var end = Math.Min(start + window, samples.Length);
            if (end - start < MediaDecoders.SampleRate || MediaDecoders.Rms(samples, start, end) < SilenceRms)
                continue;
            if (embedder.EmbedAudio(MediaDecoders.Wav(samples, start, end)) is { } vector)
                vectors.Add(new MediaVector(file, start * 1000 / MediaDecoders.SampleRate, end * 1000 / MediaDecoders.SampleRate, vector));
        }
        return vectors;
    }

    // ---------------------------------------------------------------- search

    [RelayCommand]
    private async Task SearchAsync()
    {
        var query = Query.Trim();
        if (query.Length == 0 || IsSearching)
            return;
        await RunSearchAsync(async (embedder, ct) => await Task.Run(() => embedder.EmbedText(query, query: true), ct));
    }

    /// <summary>Photos that look like this one.</summary>
    public async Task SearchByPhotoAsync(string path)
    {
        if (IsSearching)
            return;
        Mode = "photos";
        Query = "";
        await RunSearchAsync(async (embedder, ct) =>
            await MediaDecoders.Image(path, PhotoEdge, ct) is { } jpeg ? await Task.Run(() => embedder.EmbedImages([jpeg]), ct) : null);
    }

    private async Task RunSearchAsync(Func<LiteRtEngine.LiteRtEmbedder, CancellationToken, Task<float[]?>> embed)
    {
        if (SelectedModel is not { } model)
        {
            Status = Loc.S("SearchNoModel");
            return;
        }
        var index = Index;
        if (index.FileCount == 0 || index.ModelId != model.Id)
        {
            Status = Loc.S("SearchIndexFirst");
            return;
        }
        IsSearching = true;
        try
        {
            var clock = System.Diagnostics.Stopwatch.StartNew();
            var vector = await embed(await EmbedderAsync(CancellationToken.None), CancellationToken.None);
            if (vector is null)
            {
                Status = Loc.S("SearchNoVector");
                return;
            }
            if (IsPhotos)
            {
                var matches = index.All.Select(v => new PhotoMatch(v.Id, MediaIndex.Dot(vector, v.Vector)))
                    .Where(m => File.Exists(m.Path)).OrderByDescending(m => m.Score).Take(MaxPhotoResults).ToList();
                Photos.Clear();
                foreach (var m in matches)
                    Photos.Add(m);
            }
            else
            {
                var matches = index.All.GroupBy(v => v.Id).Select(g =>
                {
                    var moments = g.Select(v => new AudioMoment(v.StartMs, v.EndMs, MediaIndex.Dot(vector, v.Vector))).OrderByDescending(m => m.Score).ToList();
                    return new AudioMatch(g.Key, moments[0].Score, moments.Take(MomentsPerFile).ToList());
                }).Where(m => File.Exists(m.Path)).OrderByDescending(m => m.Score).Take(MaxAudioResults).ToList();
                Audio.Clear();
                foreach (var m in matches)
                    Audio.Add(m);
            }
            Status = Loc.F("SearchFound", (IsPhotos ? Photos.Count : Audio.Count).ToString(), clock.Elapsed.TotalSeconds.ToString("F1"));
        }
        catch (Exception e) when (e is not OperationCanceledException)
        {
            Log.Error("Media search failed", e);
            Status = Loc.F("ErrorPrefix", e.Message);
        }
        finally
        {
            IsSearching = false;
        }
    }

    /// <summary>Plays the matching moment (decoded again and written as a short WAV), so any format plays on any platform.</summary>
    public async Task PlayMomentAsync(AudioMatch match, AudioMoment moment)
    {
        try
        {
            var samples = await MediaDecoders.Audio(match.Path, moment.EndMs / 1000 + 1, CancellationToken.None);
            if (samples is null)
                return;
            var from = Math.Min(samples.Length, moment.StartMs * MediaDecoders.SampleRate / 1000);
            var to = Math.Min(samples.Length, moment.EndMs * MediaDecoders.SampleRate / 1000);
            var wav = Path.Combine(Path.GetTempPath(), "llmhub-moment.wav");
            await File.WriteAllBytesAsync(wav, MediaDecoders.Wav(samples, from, to));
            await Services.Speech.AudioIO.PlayAsync(wav, (moment.EndMs - moment.StartMs) / 1000.0, CancellationToken.None);
        }
        catch (Exception e) when (e is IOException or InvalidOperationException)
        {
            Status = Loc.F("ErrorPrefix", e.Message);
        }
    }
}
