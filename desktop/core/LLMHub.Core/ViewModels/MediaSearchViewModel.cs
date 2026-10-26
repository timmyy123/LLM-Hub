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

/// <summary>A 2-second moment of a video that matched, by where it starts.</summary>
public sealed record VideoMomentMatch(string Path, int StartMs, int EndMs, float Score)
{
    public string Name => System.IO.Path.GetFileName(Path);
    public string Label => $"{StartMs / 60000}:{StartMs / 1000 % 60:00}";
    public string ScoreText => $"{Score:0.00}";
}

/// <summary>
/// Studio › Search, after the Android app's Photo Search, Audio Search and Video Moment Finder: EmbeddingGemma 2 (440M
/// for photos and video frames, 740M also for audio) embeds a folder's photos, 5-second moments of its audio, or
/// 2-second moments of its videos on this computer; then a description, or a photo, finds the closest ones. Indexes are
/// kept per mode and redone only for new or changed files.
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
    // Video Moment Finder, as on Android: 2-second windows of two 512 px frames and the sound around them (1024 input
    // tokens), the first 10 minutes of each video, top 5 moments.
    private const int VideoWindowMs = 2_000;
    private const int MinVideoWindowMs = 500;
    private const int MinVideoMs = 1_000;
    private const int MaxVideoSeconds = 600;
    private const int MaxVideoResults = 5;
    private const int WindowsPerBatch = 16;

    private readonly MediaIndex _photos = new(Path.Combine(AppPaths.Root, "media_search", "photos.idx"));
    private readonly MediaIndex _audio = new(Path.Combine(AppPaths.Root, "media_search", "audio.idx"));
    private readonly MediaIndex _videos = new(Path.Combine(AppPaths.Root, "media_search", "videos.idx"));
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
        VideoFolder = s.MediaVideoFolder is { Length: > 0 } v ? v : Environment.GetFolderPath(Environment.SpecialFolder.MyVideos);
        _photos.Load();
        _audio.Load();
        _videos.Load();
        Ioc.Models.Installed.CollectionChanged += (_, _) => UiThread.Post(RefreshModels);
        RefreshModels();
        _ready = true;
    }

    /// <summary>Downloaded EmbeddingGemma 2 builds that understand images (440M) or images and audio (740M).</summary>
    public ObservableCollection<InstalledModel> Models { get; } = [];
    public ObservableCollection<PhotoMatch> Photos { get; } = [];
    public ObservableCollection<AudioMatch> Audio { get; } = [];
    public ObservableCollection<VideoMomentMatch> Videos { get; } = [];

    /// <summary>photos, audio or videos.</summary>
    [ObservableProperty] public partial string Mode { get; set; }
    [ObservableProperty] public partial InstalledModel? SelectedModel { get; set; }
    [ObservableProperty] public partial string PhotoFolder { get; set; }
    [ObservableProperty] public partial string AudioFolder { get; set; }
    [ObservableProperty] public partial string VideoFolder { get; set; }
    [ObservableProperty] public partial string Query { get; set; }
    [ObservableProperty] public partial string Status { get; set; }
    [ObservableProperty] public partial bool IsIndexing { get; set; }
    [ObservableProperty] public partial bool IsSearching { get; set; }
    [ObservableProperty] public partial double Progress { get; set; }

    public bool IsPhotos => Mode == "photos";
    public bool IsAudio => Mode == "audio";
    public bool IsVideos => Mode == "videos";
    public string Folder => Mode switch { "photos" => PhotoFolder, "audio" => AudioFolder, _ => VideoFolder };
    public bool HasModel => SelectedModel is not null;
    /// <summary>EmbeddingGemma 2's image and audio encoders run on LiteRT-LM, which has no Windows on Arm build.</summary>
    public bool Supported => LiteRtEngine.IsSupported;
    public bool ShowGetModel => Supported && !HasModel;
    /// <summary>Audio needs the 740M build; photos and videos work with either.</summary>
    public bool ModelFitsMode => SelectedModel is { } m && m.Tags.Contains(IsAudio ? "audio" : "vision");
    /// <summary>Videos with 440M, which has no audio encoder: moments are matched from their frames only.</summary>
    public bool VideoFramesOnly => IsVideos && SelectedModel is { } m && !m.Tags.Contains("audio");
    public string IndexedText => Loc.F(Mode switch { "photos" => "SearchIndexedPhotos", "audio" => "SearchIndexedAudio", _ => "SearchIndexedVideos" },
        Index.FileCount.ToString("N0"));
    public string Placeholder => Loc.S(Mode switch { "photos" => "SearchPlaceholderPhotos", "audio" => "SearchPlaceholderAudio", _ => "SearchPlaceholderVideos" });
    private MediaIndex Index => Mode switch { "photos" => _photos, "audio" => _audio, _ => _videos };
    /// <summary>Video windows hold frames, audio and timestamps, so their engine takes longer inputs (as on Android).</summary>
    private int MaxInputTokens => IsVideos ? LiteRtEngine.LiteRtEmbedder.VideoMaxInputTokens : LiteRtEngine.LiteRtEmbedder.MaxInputTokens;

    partial void OnModeChanged(string value)
    {
        OnPropertyChanged(nameof(IsPhotos));
        OnPropertyChanged(nameof(IsAudio));
        OnPropertyChanged(nameof(IsVideos));
        OnPropertyChanged(nameof(Folder));
        OnPropertyChanged(nameof(IndexedText));
        OnPropertyChanged(nameof(Placeholder));
        OnPropertyChanged(nameof(ModelFitsMode));
        // Pick a build that can do this mode; videos prefer 740M, which also hears them.
        if ((!ModelFitsMode || IsVideos) && Models.FirstOrDefault(m => m.Tags.Contains(IsPhotos ? "vision" : "audio")) is { } better)
            SelectedModel = better;
        OnPropertyChanged(nameof(VideoFramesOnly));
        Status = "";
    }

    partial void OnSelectedModelChanged(InstalledModel? value)
    {
        OnPropertyChanged(nameof(HasModel));
        OnPropertyChanged(nameof(ShowGetModel));
        OnPropertyChanged(nameof(ModelFitsMode));
        OnPropertyChanged(nameof(VideoFramesOnly));
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

    partial void OnVideoFolderChanged(string value)
    {
        OnPropertyChanged(nameof(Folder));
        if (!_ready)
            return;
        AppSettings.Current.MediaVideoFolder = value;
        AppSettings.Save();
    }

    public void SetFolder(string folder)
    {
        if (IsPhotos)
            PhotoFolder = folder;
        else if (IsAudio)
            AudioFolder = folder;
        else
            VideoFolder = folder;
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

    private Task<LiteRtEngine.LiteRtEmbedder> EmbedderAsync(int maxInputTokens, CancellationToken ct) =>
        Ioc.MediaEmbedder.GetAsync(SelectedModel ?? throw new InvalidOperationException(Loc.S("SearchNoModel")), text => UiThread.Post(() => Status = text), ct,
            maxInputTokens);

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
        if (IsVideos && MediaDecoders.VideoUnavailable() is { } missing)
        {
            Status = missing;
            return;
        }
        var mode = Mode;
        var index = Index;
        _indexing = new CancellationTokenSource();
        var ct = _indexing.Token;
        IsIndexing = true;
        Progress = 0;
        var clock = System.Diagnostics.Stopwatch.StartNew();
        var added = 0;
        var unreadable = 0;
        var moments = 0;
        try
        {
            var embedder = await EmbedderAsync(MaxInputTokens, ct);
            if (index.ModelId != model.Id)
                index.Reset(model.Id);
            Status = Loc.S("SearchScanning");
            var extensions = mode switch { "photos" => MediaDecoders.ImageExtensions, "audio" => MediaDecoders.AudioExtensions, _ => MediaDecoders.VideoExtensions };
            var files = await Task.Run(() => Scan(folder, extensions), ct);
            // Files that were deleted or are no longer under the folder leave the index.
            foreach (var gone in index.Files.Where(f => !files.Contains(f)).ToList())
                index.Remove(gone);
            var todo = files.Where(f => !index.IsCurrent(f, Stamp(f))).ToList();
            for (var i = 0; i < todo.Count; i++)
            {
                ct.ThrowIfCancellationRequested();
                var file = todo[i];
                var position = i;
                // Videos report their progress moment by moment; photos and audio file by file.
                void Report(double part) => UiThread.Post(() =>
                {
                    // Updates still queued when indexing ends must not cover the final message.
                    if (!IsIndexing)
                        return;
                    Progress = (position + part) / todo.Count;
                    Status = Loc.F("SearchIndexing", (position + 1).ToString("N0"), todo.Count.ToString("N0"));
                });
                var vectors = await Task.Run(() => mode switch
                {
                    "photos" => EmbedPhotoAsync(embedder, file, ct),
                    "audio" => EmbedAudioAsync(embedder, file, ct),
                    _ => EmbedVideoAsync(embedder, file, Report, ct),
                }, ct);
                // A video that cannot be opened is not marked as done, so it is tried again (say, once a decoder is installed).
                if (vectors is null)
                {
                    unreadable++;
                    continue;
                }
                index.Put(file, Stamp(file), vectors);
                added++;
                moments += vectors.Count;
                if (added % 20 == 0 || mode == "videos")
                    index.SaveIfDirty();
                Report(1);
            }
            index.SaveIfDirty();
            Status = Loc.F("SearchIndexDone", files.Count.ToString("N0"), clock.Elapsed.TotalSeconds.ToString("F0"))
                     + (unreadable > 0 ? " " + Loc.F("SearchUnreadable", unreadable.ToString("N0")) : "");
            Log.Info($"Media search {mode}: {added} new of {files.Count} ({moments} vectors, {unreadable} unreadable) in {folder} with {model.Id} on {embedder.Backend} in {clock.Elapsed.TotalSeconds:F0}s");
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

    private static async Task<IReadOnlyList<MediaVector>?> EmbedPhotoAsync(LiteRtEngine.LiteRtEmbedder embedder, string file, CancellationToken ct)
    {
        var jpeg = await MediaDecoders.Image(file, PhotoEdge, ct);
        var vector = jpeg is null ? null : embedder.EmbedImages([jpeg]);
        return vector is null ? [] : [new MediaVector(file, 0, 0, vector)];
    }

    /// <summary>One vector per 5-second moment, skipping silence, for the first 10 minutes of the file.</summary>
    private static async Task<IReadOnlyList<MediaVector>?> EmbedAudioAsync(LiteRtEngine.LiteRtEmbedder embedder, string file, CancellationToken ct)
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

    /// <summary>
    /// The Android app's Video Moment Finder: one vector per 2-second window of the first 10 minutes, from the window's
    /// first and last frames and (with 740M) the sound around each. Null when the video cannot be opened.
    /// </summary>
    private static async Task<IReadOnlyList<MediaVector>?> EmbedVideoAsync(LiteRtEngine.LiteRtEmbedder embedder, string file, Action<double> progress,
        CancellationToken ct)
    {
        using var video = await MediaDecoders.Video(file, PhotoEdge, MaxVideoSeconds, ct);
        if (video is null)
            return null;
        var limit = Math.Min(video.DurationMs, MaxVideoSeconds * 1000);
        if (limit < MinVideoMs)
            return [];
        var samples = embedder.SupportsAudio ? await MediaDecoders.Audio(file, MaxVideoSeconds, ct) : null;
        var windows = new List<(int Start, int End)>();
        for (var start = 0; start < limit; start += VideoWindowMs)
        {
            var end = Math.Min(start + VideoWindowMs, limit);
            if (end - start >= MinVideoWindowMs)
                windows.Add((start, end));
        }
        var vectors = new List<MediaVector>();
        for (var first = 0; first < windows.Count; first += WindowsPerBatch)
        {
            var batch = windows.Skip(first).Take(WindowsPerBatch).ToList();
            var times = batch.SelectMany(w => new[] { w.Start, Math.Max(w.Start, w.End - 1) }).ToList();
            var frames = await video.FramesAsync(times, ct);
            for (var j = 0; j < batch.Count; j++)
            {
                ct.ThrowIfCancellationRequested();
                var shots = Enumerable.Range(2 * j, 2).Where(k => frames[k] is not null).Select(k => (times[k], frames[k]!)).ToList();
                var parts = WindowParts(shots, samples, batch[j].Start, batch[j].End);
                if (parts.Count > 0 && embedder.Embed(parts) is { } vector)
                    vectors.Add(new MediaVector(file, batch[j].Start, batch[j].End, vector));
                progress((double)(first + j + 1) / windows.Count);
            }
        }
        return vectors;
    }

    /// <summary>
    /// One window as AI Edge Gallery and the Android app build it: for each frame, its timestamp and the audio slice
    /// around it, then its timestamp and the frame. Frames alone when there is no sound to add.
    /// </summary>
    private static List<(LiteRtEngine.EmbedKind Kind, byte[] Data)> WindowParts(List<(int TimeMs, byte[] Jpeg)> frames, float[]? samples, int startMs, int endMs)
    {
        var parts = new List<(LiteRtEngine.EmbedKind Kind, byte[] Data)>();
        var slices = AudioSlices(samples, startMs, endMs, frames.Count);
        if (slices.All(s => s is null))
        {
            parts.AddRange(frames.Select(f => (LiteRtEngine.EmbedKind.Image, f.Jpeg)));
            return parts;
        }
        for (var i = 0; i < frames.Count; i++)
        {
            var stamp = System.Text.Encoding.UTF8.GetBytes($"{frames[i].TimeMs / 60000}:{frames[i].TimeMs / 1000 % 60:00}");
            if (slices[i] is { } wav)
            {
                parts.Add((LiteRtEngine.EmbedKind.Text, stamp));
                parts.Add((LiteRtEngine.EmbedKind.Audio, wav));
            }
            parts.Add((LiteRtEngine.EmbedKind.Text, stamp));
            parts.Add((LiteRtEngine.EmbedKind.Image, frames[i].Jpeg));
        }
        return parts;
    }

    /// <summary>The window's sound split evenly between its frames, as 16 kHz WAVs; slices under a quarter second are left out.</summary>
    private static byte[]?[] AudioSlices(float[]? samples, int startMs, int endMs, int count)
    {
        var slices = new byte[]?[count];
        if (samples is null)
            return slices;
        var duration = Math.Max(1, endMs - startMs);
        for (var i = 0; i < count; i++)
        {
            var from = Math.Clamp((int)((startMs + (long)i * duration / count) * MediaDecoders.SampleRate / 1000), 0, samples.Length);
            var to = Math.Clamp((int)((startMs + (long)(i + 1) * duration / count) * MediaDecoders.SampleRate / 1000), from, samples.Length);
            slices[i] = to - from < MediaDecoders.SampleRate / 4 ? null : MediaDecoders.Wav(samples, from, to);
        }
        return slices;
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
            var vector = await embed(await EmbedderAsync(MaxInputTokens, CancellationToken.None), CancellationToken.None);
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
            else if (IsAudio)
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
            else
            {
                // The best moments across all the videos, as Android ranks them inside one.
                var present = new Dictionary<string, bool>();
                var matches = index.All.Select(v => new VideoMomentMatch(v.Id, v.StartMs, v.EndMs, MediaIndex.Dot(vector, v.Vector)))
                    .OrderByDescending(m => m.Score)
                    .Where(m => present.TryGetValue(m.Path, out var exists) ? exists : present[m.Path] = File.Exists(m.Path))
                    .Take(MaxVideoResults).ToList();
                Videos.Clear();
                foreach (var m in matches)
                    Videos.Add(m);
            }
            Status = Loc.F("SearchFound", (IsPhotos ? Photos.Count : IsAudio ? Audio.Count : Videos.Count).ToString(), clock.Elapsed.TotalSeconds.ToString("F1"));
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
