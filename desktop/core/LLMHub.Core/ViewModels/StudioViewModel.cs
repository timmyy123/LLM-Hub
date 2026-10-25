using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using LLMHub.Services;
using LLMHub.Services.Studio;
using LLMHub.Services.Studio.Npu;

namespace LLMHub.ViewModels;

/// <summary>A gallery entry: an image or video file in Pictures/LLM Hub.</summary>
public sealed record GalleryItem(string Path)
{
    public string Name => System.IO.Path.GetFileName(Path);
    public bool IsVideo => Path.EndsWith(".webp", StringComparison.OrdinalIgnoreCase) && Name.Contains("ltxv", StringComparison.OrdinalIgnoreCase)
                           || Path.EndsWith(".webm", StringComparison.OrdinalIgnoreCase) || Path.EndsWith(".mp4", StringComparison.OrdinalIgnoreCase);
    public Uri Uri => new(Path);
}

/// <summary>
/// Studio: Image and Video (ComfyUI, or Stable Diffusion and LTX-Video on the Hexagon NPU on Snapdragon PCs), Music (Stable
/// Audio 3 clips through ComfyUI; live Magenta RealTime on the Mac and on Linux with an NVIDIA GPU), Voice (Supertonic).
/// </summary>
public sealed partial class StudioViewModel : ObservableObject
{
    private readonly ComfyService _comfy = Ioc.Comfy;
    private readonly SdNpuService _npu = Ioc.SdNpu;
    private readonly LtxNpuService _ltx = Ioc.LtxNpu;
    private CancellationTokenSource? _run;

    /// <summary>Negative prompt for the Stable Diffusion 1.5 NPU models.</summary>
    private const string NpuNegative = "lowres, bad anatomy, bad hands, text, error, missing fingers, cropped, worst quality, low quality, jpeg artifacts, signature, watermark, blurry, deformed";

    public StudioViewModel()
    {
        // Workflow first: the Mode and Workflow change handlers read each other.
        Workflow = Workflows[0];
        Mode = "image";
        Prompt = "";
        Status = "";
        Seed = "";
        Steps = Workflow.DefaultSteps;
        Aspect = Aspects[0];
        Count = 1;
        VoiceText = "";
        Voice = AppSettings.Current.Voice;
        ReferenceImage = "";
        MusicWorkflow = ComfyWorkflow.StableAudioSmall;
        MusicPrompt = "";
        MusicSeconds = 30;
        MusicSeed = "";
        MusicStatus = "";
        LoadGallery();
        LoadMusic();
        _comfy.PropertyChanged += (_, e) => OnEngineChanged(e.PropertyName);
        _npu.PropertyChanged += (_, e) => OnEngineChanged(e.PropertyName);
        _ltx.PropertyChanged += (_, e) => OnEngineChanged(e.PropertyName);
    }

    /// <summary>Snapdragon PCs (Windows on Arm with a Hexagon NPU) generate on the NPU, since ComfyUI has no GPU backend there.</summary>
    public static bool UsesNpu => UsesNpuOnce.Value;
    private static readonly Lazy<bool> UsesNpuOnce = new(() => !ComfyService.IsSupported && SdNpuService.IsSupported);

    public IReadOnlyList<ComfyWorkflow> Workflows => UsesNpu ? ComfyWorkflow.NpuWorkflows : ComfyWorkflow.Visual;
    /// <summary>Snapdragon PCs make music clips with Stable Audio 3 Small on the Adreno GPU (<see cref="StableAudioDmlService"/>).</summary>
    public static bool UsesGpuMusic => !ComfyService.IsSupported && StableAudioDmlService.IsSupported;
    public IReadOnlyList<ComfyWorkflow> MusicWorkflows => UsesGpuMusic ? [ComfyWorkflow.StableAudioSmall] : ComfyWorkflow.Music;
    /// <summary>Music clips work here: through ComfyUI, or on the GPU of Snapdragon PCs.</summary>
    public bool MusicSupported => ComfyService.IsSupported || UsesGpuMusic;
    public ObservableCollection<GalleryItem> MusicClips { get; } = [];
    public static string MusicFolder => Directory.CreateDirectory(Path.Combine(ComfyService.Gallery, "Music")).FullName;
    /// <summary>Why there is no live jam (Windows, or Linux without an NVIDIA GPU), and that clips still work where ComfyUI or the Snapdragon GPU engine runs.</summary>
    public static string LiveMusicUnavailable =>
        Loc.S(OperatingSystem.IsWindows() ? "StudioMusicUnavailable" : "MusicNeedsNvidia") + (ComfyService.IsSupported || UsesGpuMusic ? " " + Loc.S("MusicClipsInstead") : "");
    /// <summary>The NPU models are compiled for one size: 512 x 512 images, and 512 x 320 (16:10) or 768 x 512 (3:2) video.</summary>
    public IReadOnlyList<string> Aspects => !UsesNpu ? ["3:2", "1:1", "16:9", "9:16"]
        : Workflow?.Kind == "video" ? [LtxNpuService.Build.Width * 2 == LtxNpuService.Build.Height * 3 ? "3:2" : "16:10"] : ["1:1"];
    public IReadOnlyList<int> Counts { get; } = [1, 2, 4];
    public IReadOnlyList<string> Voices => Services.Speech.SpeechService.Voices;
    public ObservableCollection<GalleryItem> Gallery { get; } = [];
    public ComfyService Comfy => _comfy;

    /// <summary>image, video, music or voice.</summary>
    [ObservableProperty] public partial string Mode { get; set; }
    [ObservableProperty] public partial ComfyWorkflow Workflow { get; set; }
    [ObservableProperty] public partial string Prompt { get; set; }
    [ObservableProperty] public partial string Aspect { get; set; }
    [ObservableProperty] public partial int Count { get; set; }
    [ObservableProperty] public partial int Steps { get; set; }
    /// <summary>A number, or empty for a random seed.</summary>
    [ObservableProperty] public partial string Seed { get; set; }
    [ObservableProperty] public partial string ReferenceImage { get; set; }
    [ObservableProperty] public partial bool IsGenerating { get; set; }
    [ObservableProperty] public partial double Progress { get; set; }
    [ObservableProperty] public partial string Status { get; set; }
    [ObservableProperty] public partial GalleryItem? SelectedItem { get; set; }
    [ObservableProperty] public partial string VoiceText { get; set; }
    [ObservableProperty] public partial string Voice { get; set; }
    [ObservableProperty] public partial string? LastSpeech { get; set; }
    [ObservableProperty] public partial ComfyWorkflow MusicWorkflow { get; set; }
    [ObservableProperty] public partial string MusicPrompt { get; set; }
    [ObservableProperty] public partial int MusicSeconds { get; set; }
    [ObservableProperty] public partial string MusicSeed { get; set; }
    [ObservableProperty] public partial bool IsGeneratingMusic { get; set; }
    [ObservableProperty] public partial double MusicProgress { get; set; }
    [ObservableProperty] public partial string MusicStatus { get; set; }

    /// <summary>"2.3 GB to download on first use" for the chosen music model, or "".</summary>
    public string MusicDownloadNote => UsesGpuMusic
        ? StableAudioDmlService.IsInstalled ? "" : Loc.F("StudioDownloadNote", Fmt.Bytes(StableAudioDmlService.DownloadSize))
        : ComfyService.HasModels(MusicWorkflow) ? "" : Loc.F("StudioDownloadNote", Fmt.Bytes(MusicWorkflow.DownloadSize));

    partial void OnMusicWorkflowChanged(ComfyWorkflow value)
    {
        MusicSeconds = Math.Min(MusicSeconds, value.MaxSeconds);
        OnPropertyChanged(nameof(MusicDownloadNote));
    }

    private CancellationTokenSource? _music;

    /// <summary>
    /// Stable Audio 3 through ComfyUI (or on the Adreno GPU of Snapdragon PCs): one clip of <see cref="MusicSeconds"/>, saved to
    /// Pictures/LLM Hub/Music.
    /// </summary>
    [RelayCommand]
    private async Task GenerateMusicAsync()
    {
        if (IsGeneratingMusic || MusicPrompt.Trim().Length == 0)
            return;
        IsGeneratingMusic = true;
        MusicProgress = 0;
        _music = new CancellationTokenSource();
        var ct = _music.Token;
        try
        {
            if (UsesGpuMusic)
            {
                await GenerateMusicOnSnapdragonAsync(ct);
                return;
            }
            MusicStatus = Loc.S("ComfyStarting");
            await _comfy.EnsureRunningAsync(ct);
            await _comfy.EnsureModelsAsync(MusicWorkflow, ct);
            OnPropertyChanged(nameof(MusicDownloadNote));
            var seed = long.TryParse(MusicSeed, out var s) ? s : Random.Shared.NextInt64(1, 1L << 48);
            var graph = MusicWorkflow.Build(MusicPrompt.Trim(), Math.Clamp(MusicSeconds, 1, MusicWorkflow.MaxSeconds), 0, 1, MusicWorkflow.DefaultSteps, seed, null);
            var clock = System.Diagnostics.Stopwatch.StartNew();
            var outputs = await _comfy.RunAsync(graph, (p, text) => UiThread.Post(() =>
            {
                MusicProgress = p;
                MusicStatus = text;
            }), ct);
            Log.Info($"Studio {MusicWorkflow.Id} {MusicSeconds}s seed {seed}: {outputs.Count} file(s) in {clock.Elapsed.TotalSeconds:F1}s");
            MusicStatus = Loc.F("StudioDone", outputs.Count.ToString(), clock.Elapsed.TotalSeconds.ToString("F0"));
            foreach (var file in outputs)
                MusicClips.Insert(0, new GalleryItem(file));
        }
        catch (OperationCanceledException)
        {
            MusicStatus = "";
        }
        catch (Exception e)
        {
            Log.Error("Music generation failed", e);
            MusicStatus = Loc.F("ErrorPrefix", e.Message);
        }
        finally
        {
            IsGeneratingMusic = false;
            MusicProgress = 0;
            _music.Dispose();
            _music = null;
        }
    }

    private async Task GenerateMusicOnSnapdragonAsync(CancellationToken ct)
    {
        var seed = long.TryParse(MusicSeed, out var s) ? s : Random.Shared.NextInt64(1, int.MaxValue);
        var seconds = Math.Clamp(MusicSeconds, 1, StableAudioDmlService.MaxSeconds);
        var clock = System.Diagnostics.Stopwatch.StartNew();
        var file = await Ioc.StableAudioDml.GenerateAsync(MusicPrompt.Trim(), seconds, seed, MusicFolder, (p, text) => UiThread.Post(() =>
        {
            MusicProgress = p;
            MusicStatus = text;
        }), ct);
        OnPropertyChanged(nameof(MusicDownloadNote));
        Log.Info($"Studio sa3-small-music (DirectML) {seconds}s seed {seed}: {file} in {clock.Elapsed.TotalSeconds:F1}s");
        MusicStatus = Loc.F("StudioDone", "1", clock.Elapsed.TotalSeconds.ToString("F0"));
        MusicClips.Insert(0, new GalleryItem(file));
    }

    [RelayCommand]
    private void CancelMusic() => _music?.Cancel();

    public void LoadMusic()
    {
        MusicClips.Clear();
        foreach (var file in Directory.EnumerateFiles(MusicFolder)
                     .Where(f => f.EndsWith(".flac", StringComparison.OrdinalIgnoreCase) || f.EndsWith(".wav", StringComparison.OrdinalIgnoreCase)
                                 || f.EndsWith(".mp3", StringComparison.OrdinalIgnoreCase) || f.EndsWith(".opus", StringComparison.OrdinalIgnoreCase))
                     .OrderByDescending(File.GetLastWriteTimeUtc).Take(100))
            MusicClips.Add(new GalleryItem(file));
    }

    /// <summary>ComfyUI video and upscaling need an image; the NPU workflows take one optionally (image to image or video).</summary>
    public bool NeedsReference => Workflow.NeedsImage || Workflow.IsNpu;
    public bool IsImageOrVideo => Mode is "image" or "video";
    public bool IsVoice => Mode == "voice";
    public bool IsMusic => Mode == "music";
    /// <summary>Photo and audio search with EmbeddingGemma 2 (<see cref="MediaSearchViewModel"/>).</summary>
    public bool IsSearch => Mode == "search";
    /// <summary>Image and video generation works here: ComfyUI on a GPU or Mac, or the Snapdragon NPU.</summary>
    public bool ComfySupported => ComfyService.IsSupported || UsesNpu;
    /// <summary>The current tab has a workflow (Snapdragon PCs have NPU image workflows, but no video one).</summary>
    public bool ModeSupported => ComfySupported && Workflows.Any(w => w.Kind == Mode);
    /// <summary>Shown instead of Generate when the current tab has no engine on this PC.</summary>
    public string UnsupportedNote => !ComfySupported ? Loc.S("ComfyUnsupported") : ModeSupported ? "" : Loc.S("NpuNoVideo");
    public bool IsNpu => Workflow.IsNpu;
    public bool UsesNpuEngine => UsesNpu;
    /// <summary>"Animate" in the gallery needs a video workflow.</summary>
    public bool CanAnimate => Workflows.Any(w => w.Kind == "video");
    /// <summary>"6.9 GB to download on first use" under the workflow picker, or "".</summary>
    public string DownloadNote => Workflow.NpuModel is { } model
        ? SdNpuService.IsInstalled(model) ? "" : Loc.F("StudioDownloadNote", Fmt.Bytes(model.DownloadSize))
        : Workflow.NpuVideo ? LtxNpuService.IsInstalled ? "" : Loc.F("StudioDownloadNote", Fmt.Bytes(LtxNpuService.DownloadSize))
        : ComfyService.HasModels(Workflow) ? "" : Loc.F("StudioDownloadNote", Fmt.Bytes(Workflow.DownloadSize));
    /// <summary>The engine line in the Studio header: ComfyUI's, or the NPU's on Snapdragon.</summary>
    public string EngineStatusLine => UsesNpu ? _npu.StatusLine : _comfy.StatusLine;
    public bool EngineRunning => UsesNpu ? _npu.IsLoaded : _comfy.IsRunning;
    /// <summary>Install, download or load progress of the engine.</summary>
    public string EngineStatus => UsesNpu ? _ltx.Status.Length > 0 ? _ltx.Status : _npu.Status : _comfy.Status;

    private void OnEngineChanged(string? property)
    {
        OnPropertyChanged(nameof(EngineStatusLine));
        OnPropertyChanged(nameof(EngineRunning));
        OnPropertyChanged(nameof(EngineStatus));
    }

    partial void OnModeChanged(string value)
    {
        if (value is "image" or "video" && Workflow.Kind != value && Workflows.FirstOrDefault(w => w.Kind == value) is { } first)
            Workflow = first;
        OnPropertyChanged(nameof(IsImageOrVideo));
        OnPropertyChanged(nameof(IsVoice));
        OnPropertyChanged(nameof(IsMusic));
        OnPropertyChanged(nameof(IsSearch));
        OnPropertyChanged(nameof(ModeSupported));
        OnPropertyChanged(nameof(UnsupportedNote));
    }

    partial void OnWorkflowChanged(ComfyWorkflow value)
    {
        Steps = value.DefaultSteps;
        OnPropertyChanged(nameof(NeedsReference));
        OnPropertyChanged(nameof(DownloadNote));
        OnPropertyChanged(nameof(IsNpu));
        OnPropertyChanged(nameof(Aspects));
        if (Aspect is not null && !Aspects.Contains(Aspect))
            Aspect = Aspects[0];
    }

    partial void OnVoiceChanged(string value)
    {
        AppSettings.Current.Voice = value;
        AppSettings.Save();
    }

    [RelayCommand]
    private async Task GenerateAsync()
    {
        if (IsGenerating || !ModeSupported)
            return;
        if (Workflow.NeedsImage && !File.Exists(ReferenceImage))
        {
            Status = Loc.S("StudioNeedsImage");
            return;
        }
        if (!Workflow.NeedsImage && Prompt.Trim().Length == 0)
            return;
        IsGenerating = true;
        Progress = 0;
        _run = new CancellationTokenSource();
        var ct = _run.Token;
        if (Workflow.IsNpu)
        {
            await GenerateOnNpuAsync(Workflow, ct);
            return;
        }
        try
        {
            Status = Loc.S("ComfyStarting");
            await _comfy.EnsureRunningAsync(ct);
            await _comfy.EnsureModelsAsync(Workflow, ct);
            OnPropertyChanged(nameof(DownloadNote));
            var image = Workflow.NeedsImage ? await _comfy.UploadAsync(ReferenceImage, ct) : null;
            var (width, height) = Size(Workflow, Aspect);
            var seed = long.TryParse(Seed, out var s) ? s : Random.Shared.NextInt64(1, 1L << 48);
            var graph = Workflow.Build(Prompt.Trim(), width, height, Workflow.Kind == "video" ? 1 : Count, Math.Max(1, Steps), seed, image);
            var clock = System.Diagnostics.Stopwatch.StartNew();
            var outputs = await _comfy.RunAsync(graph, (p, text) => UiThread.Post(() =>
            {
                Progress = p;
                Status = text;
            }), ct);
            Log.Info($"Studio {Workflow.Id} {width}x{height} seed {seed}: {outputs.Count} file(s) in {clock.Elapsed.TotalSeconds:F1}s");
            Status = Loc.F("StudioDone", outputs.Count.ToString(), clock.Elapsed.TotalSeconds.ToString("F0"));
            foreach (var file in outputs.AsEnumerable().Reverse())
                Gallery.Insert(0, new GalleryItem(file));
            SelectedItem = Gallery.FirstOrDefault();
        }
        catch (OperationCanceledException)
        {
            Status = "";
        }
        catch (Exception e)
        {
            Log.Error("Studio generation failed", e);
            Status = Loc.F("ErrorPrefix", e.Message);
        }
        finally
        {
            IsGenerating = false;
            Progress = 0;
            _run.Dispose();
            _run = null;
        }
    }

    /// <summary>
    /// The Hexagon NPU: Stable Diffusion 1.5 (with a reference image it is image to image) or LTX-Video (with a reference
    /// image it is image to video).
    /// </summary>
    private async Task GenerateOnNpuAsync(ComfyWorkflow workflow, CancellationToken ct)
    {
        try
        {
            Status = "";
            var seed = long.TryParse(Seed, out var s) ? s : Random.Shared.NextInt64(1, 1L << 31);
            var reference = File.Exists(ReferenceImage) ? ReferenceImage : null;
            var clock = System.Diagnostics.Stopwatch.StartNew();
            void Report(double p, string text) => UiThread.Post(() =>
            {
                Progress = p;
                Status = text;
            });
            var outputs = workflow.NpuModel is { } model
                ? await _npu.GenerateAsync(model, Prompt.Trim(), NpuNegative, Math.Clamp(Steps, 1, 50), 7f, seed, Count, reference, 0.6f, Report, ct)
                : [await _ltx.GenerateAsync(Prompt.Trim(), Math.Clamp(Steps, 1, LtxNpuService.MaxSteps), seed, reference, Report, ct)];
            var (width, height) = Size(workflow, Aspect);
            Log.Info($"Studio {workflow.Id} {width}x{height} seed {seed}{(reference is null ? "" : " with reference")}: {outputs.Count} file(s) in {clock.Elapsed.TotalSeconds:F1}s");
            Status = Loc.F("StudioDone", outputs.Count.ToString(), clock.Elapsed.TotalSeconds.ToString("F0"));
            OnPropertyChanged(nameof(DownloadNote));
            foreach (var file in outputs.AsEnumerable().Reverse())
                Gallery.Insert(0, new GalleryItem(file));
            SelectedItem = Gallery.FirstOrDefault();
        }
        catch (OperationCanceledException)
        {
            Status = "";
        }
        catch (Exception e)
        {
            Log.Error("Studio NPU generation failed", e);
            Status = Loc.F("ErrorPrefix", e.Message);
        }
        finally
        {
            IsGenerating = false;
            Progress = 0;
            _run?.Dispose();
            _run = null;
        }
    }

    [RelayCommand]
    private void Cancel() => _run?.Cancel();

    /// <summary>Back to text to image after a reference was picked for an NPU image workflow.</summary>
    [RelayCommand]
    private void ClearReference() => ReferenceImage = "";

    /// <summary>Asks the local chat model to expand a short idea into a detailed image prompt.</summary>
    [RelayCommand]
    private async Task ImprovePromptAsync()
    {
        if (Prompt.Trim().Length == 0 || Ioc.Chat.SelectedModel is not { } model || IsGenerating)
            return;
        Status = Loc.S("StudioImproving");
        try
        {
            if (!Ioc.Engines.IsReady(model))
                await Ioc.Engines.LoadAsync(model, CancellationToken.None);
            var instruction = Workflow.Kind == "video"
                ? "Rewrite this as one vivid prompt for an image-to-video model: describe the motion, camera movement and mood in under 60 words. Reply with the prompt only."
                : "Rewrite this as one detailed prompt for a text-to-image model: subject, setting, lighting, style and composition, under 70 words. Reply with the prompt only.";
            var text = "";
            await foreach (var delta in Ioc.Engines.StreamAsync(model, [new ChatTurn("system", instruction), new ChatTurn("user", Prompt.Trim())], CancellationToken.None))
                text += delta.Content;
            var (answer, _) = new ThinkSplitter().Push(text);
            Prompt = answer.Trim().Trim('"');
            Status = "";
        }
        catch (Exception e)
        {
            Log.Error("Improve prompt failed", e);
            Status = Loc.F("ErrorPrefix", e.Message);
        }
    }

    /// <summary>Uses a gallery image as the reference for video or upscaling.</summary>
    [RelayCommand]
    private void UseAsReference(GalleryItem? item)
    {
        if (item is null)
            return;
        ReferenceImage = item.Path;
    }

    [RelayCommand]
    private async Task SpeakAsync()
    {
        if (VoiceText.Trim().Length == 0 || IsGenerating)
            return;
        IsGenerating = true;
        Status = "";
        try
        {
            var (path, seconds) = await Ioc.Speech.SynthesizeAsync(VoiceText.Trim(), Voice, null, CancellationToken.None);
            var folder = Directory.CreateDirectory(Path.Combine(ComfyService.Gallery, "Voice")).FullName;
            var target = Path.Combine(folder, $"{DateTime.Now:yyyyMMdd-HHmmss}-{Voice}.wav");
            File.Copy(path, target, true);
            LastSpeech = target;
            Status = Loc.F("StudioSpeechDone", seconds.ToString("F1"));
            _ = Services.Speech.AudioIO.PlayAsync(target, seconds, CancellationToken.None);
        }
        catch (Exception e)
        {
            Log.Error("Voice generation failed", e);
            Status = Loc.F("ErrorPrefix", e.Message);
        }
        finally
        {
            IsGenerating = false;
        }
    }

    public void LoadGallery()
    {
        Gallery.Clear();
        foreach (var file in Directory.EnumerateFiles(ComfyService.Gallery)
                     .Where(f => (f.EndsWith(".png", StringComparison.OrdinalIgnoreCase) || f.EndsWith(".webp", StringComparison.OrdinalIgnoreCase)
                                  || f.EndsWith(".jpg", StringComparison.OrdinalIgnoreCase) || f.EndsWith(".webm", StringComparison.OrdinalIgnoreCase)
                                  || f.EndsWith(".mp4", StringComparison.OrdinalIgnoreCase))
                                 // NPU videos keep their first frame next to them as the thumbnail.
                                 && !f.EndsWith(".poster.png", StringComparison.OrdinalIgnoreCase))
                     .OrderByDescending(File.GetLastWriteTimeUtc).Take(200))
            Gallery.Add(new GalleryItem(file));
    }

    /// <summary>Pixel size for an aspect ratio; about one megapixel for images, LTX-Video's 768 x 512 class for video.</summary>
    public static (int Width, int Height) Size(ComfyWorkflow workflow, string aspect)
    {
        if (workflow.NpuVideo)
            return (LtxNpuService.Build.Width, LtxNpuService.Build.Height);
        if (workflow.IsNpu)
            return (SdQnnPipeline.Size, SdQnnPipeline.Size);
        if (workflow.Kind == "video")
            return aspect switch { "9:16" => (512, 768), "1:1" => (640, 640), _ => (768, 512) };
        return aspect switch { "1:1" => (1024, 1024), "16:9" => (1344, 768), "9:16" => (768, 1344), _ => (1216, 832) };
    }
}
