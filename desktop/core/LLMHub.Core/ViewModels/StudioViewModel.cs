using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using LLMHub.Services;
using LLMHub.Services.Studio;

namespace LLMHub.ViewModels;

/// <summary>A gallery entry: an image or video file in Pictures/LLM Hub.</summary>
public sealed record GalleryItem(string Path)
{
    public string Name => System.IO.Path.GetFileName(Path);
    public bool IsVideo => Path.EndsWith(".webp", StringComparison.OrdinalIgnoreCase) && Name.Contains("ltxv", StringComparison.OrdinalIgnoreCase)
                           || Path.EndsWith(".webm", StringComparison.OrdinalIgnoreCase) || Path.EndsWith(".mp4", StringComparison.OrdinalIgnoreCase);
    public Uri Uri => new(Path);
}

/// <summary>Studio: Image and Video (ComfyUI), Voice (Supertonic), Music (Magenta RealTime where available).</summary>
public sealed partial class StudioViewModel : ObservableObject
{
    private readonly ComfyService _comfy = Ioc.Comfy;
    private CancellationTokenSource? _run;

    public StudioViewModel()
    {
        // Workflow first: the Mode and Workflow change handlers read each other.
        Workflow = ComfyWorkflow.FluxSchnell;
        Mode = "image";
        Prompt = "";
        Status = "";
        Seed = "";
        Steps = ComfyWorkflow.FluxSchnell.DefaultSteps;
        Aspect = "3:2";
        Count = 1;
        VoiceText = "";
        Voice = AppSettings.Current.Voice;
        ReferenceImage = "";
        LoadGallery();
    }

    public IReadOnlyList<ComfyWorkflow> Workflows => ComfyWorkflow.All;
    public IReadOnlyList<string> Aspects { get; } = ["3:2", "1:1", "16:9", "9:16"];
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

    public bool NeedsReference => Workflow.NeedsImage;
    public bool IsImageOrVideo => Mode is "image" or "video";
    public bool IsVoice => Mode == "voice";
    public bool IsMusic => Mode == "music";
    public bool ComfySupported => ComfyService.IsSupported;
    /// <summary>"6.9 GB to download on first use" under the workflow picker, or "".</summary>
    public string DownloadNote => ComfyService.HasModels(Workflow) ? "" : Loc.F("StudioDownloadNote", Fmt.Bytes(Workflow.DownloadSize));

    partial void OnModeChanged(string value)
    {
        if (value == "video" && Workflow.Kind != "video")
            Workflow = ComfyWorkflow.LtxVideo;
        else if (value == "image" && Workflow.Kind != "image")
            Workflow = ComfyWorkflow.FluxSchnell;
        OnPropertyChanged(nameof(IsImageOrVideo));
        OnPropertyChanged(nameof(IsVoice));
        OnPropertyChanged(nameof(IsMusic));
    }

    partial void OnWorkflowChanged(ComfyWorkflow value)
    {
        Steps = value.DefaultSteps;
        OnPropertyChanged(nameof(NeedsReference));
        OnPropertyChanged(nameof(DownloadNote));
    }

    partial void OnVoiceChanged(string value)
    {
        AppSettings.Current.Voice = value;
        AppSettings.Save();
    }

    [RelayCommand]
    private async Task GenerateAsync()
    {
        if (IsGenerating)
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

    [RelayCommand]
    private void Cancel() => _run?.Cancel();

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
                     .Where(f => f.EndsWith(".png", StringComparison.OrdinalIgnoreCase) || f.EndsWith(".webp", StringComparison.OrdinalIgnoreCase)
                                 || f.EndsWith(".jpg", StringComparison.OrdinalIgnoreCase) || f.EndsWith(".webm", StringComparison.OrdinalIgnoreCase))
                     .OrderByDescending(File.GetLastWriteTimeUtc).Take(200))
            Gallery.Add(new GalleryItem(file));
    }

    /// <summary>Pixel size for an aspect ratio; about one megapixel for images, LTX-Video's 768 x 512 class for video.</summary>
    public static (int Width, int Height) Size(ComfyWorkflow workflow, string aspect)
    {
        if (workflow.Kind == "video")
            return aspect switch { "9:16" => (512, 768), "1:1" => (640, 640), _ => (768, 512) };
        return aspect switch { "1:1" => (1024, 1024), "16:9" => (1344, 768), "9:16" => (768, 1344), _ => (1216, 832) };
    }
}
