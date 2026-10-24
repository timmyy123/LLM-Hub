using System.Collections.ObjectModel;
using System.Diagnostics;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using LLMHub.Services;
using LLMHub.Services.Agent;
using LLMHub.Services.Speech;
using LLMHub.Services.Studio;

namespace LLMHub.ViewModels;

/// <summary>An engine row in Settings › Engines: what is on disk and what is running.</summary>
public sealed record EngineInfo(string Name, string Version, string Detail, bool Installed, string Folder);

/// <summary>A folder in Settings › Storage with its size on disk.</summary>
public sealed record StorageItem(string Key, string Name, string Folder, long Bytes)
{
    public string Size => Fmt.Bytes(Bytes);
    /// <summary>Only logs are offered for clearing; attachments back old chats.</summary>
    public bool CanClear => Key == "logs";
}

/// <summary>
/// The Settings sections beyond Models and Local API (General, Engines, Speech and voice, Studio backends, Agent,
/// Web search, Storage), shared by every platform's settings pages.
/// </summary>
public sealed partial class SettingsViewModel : ObservableObject
{
    public SettingsViewModel()
    {
        var s = AppSettings.Current;
        Appearance = s.Appearance;
        WebSearch = s.WebSearch;
        Voice = s.Voice;
        AgentContext = s.ApiContextLength;
        ClaudeVersion = "";
        SpeechStatus = "";
        StorageTotal = "";
    }

    public static IReadOnlyList<(string Value, string Label)> Appearances { get; } =
        [("dark", Loc.S("AppearanceDark")), ("light", Loc.S("AppearanceLight")), ("system", Loc.S("AppearanceSystem"))];
    public static IReadOnlyList<int> AgentContexts { get; } = [32768, 65536, 131072];
    public IReadOnlyList<string> Voices => SpeechService.Voices;

    public ObservableCollection<EngineInfo> Engines { get; } = [];
    public ObservableCollection<StorageItem> Storage { get; } = [];
    /// <summary>ComfyUI model files on disk, for Settings › Studio backends.</summary>
    public ObservableCollection<StorageItem> StudioModels { get; } = [];

    /// <summary>dark, light or system; the window applies it on the next start (and live where the toolkit allows).</summary>
    [ObservableProperty] public partial string Appearance { get; set; }
    [ObservableProperty] public partial bool WebSearch { get; set; }
    [ObservableProperty] public partial string Voice { get; set; }
    [ObservableProperty] public partial int AgentContext { get; set; }
    [ObservableProperty] public partial string ClaudeVersion { get; set; }
    [ObservableProperty] public partial string SpeechStatus { get; set; }
    [ObservableProperty] public partial bool IsSettingUpSpeech { get; set; }
    [ObservableProperty] public partial string StorageTotal { get; set; }

    public bool TtsInstalled => SpeechService.IsTtsInstalled;
    public bool AsrInstalled => SpeechService.IsAsrInstalled;
    public string TtsLine => $"{SpeechService.TtsName} · {Loc.S(TtsInstalled ? "Installed" : "SettingsNotInstalled")}";
    public string AsrLine => $"{SpeechService.AsrName} · {Loc.S(AsrInstalled ? "Installed" : "SettingsNotInstalled")}";
    public string ClaudePath => AgentService.FindClaude() ?? Loc.S("SettingsNotInstalled");
    public bool ClaudeInstalled => AgentService.FindClaude() is not null;
    /// <summary>Studio's image engine: ComfyUI, or on Snapdragon PCs Stable Diffusion on the Hexagon NPU.</summary>
    public string ComfyLine => ComfyService.IsSupported
        ? Ioc.Comfy.StatusLine
        : StudioViewModel.UsesNpu ? Ioc.SdNpu.StatusLine : Loc.S("ComfyUnsupported");
    public string ComfyFolder => StudioViewModel.UsesNpu ? Services.Studio.Npu.SdNpuService.Root : ComfyService.Root;
    public string MusicLine => Loc.S(OperatingSystem.IsMacOS() ? "SettingsMusicMac" : "StudioMusicUnavailable");

    partial void OnAppearanceChanged(string value) => Save(s => s.Appearance = value);
    partial void OnWebSearchChanged(bool value)
    {
        Save(s => s.WebSearch = value);
        Ioc.Chat.WebSearchEnabled = value;
    }
    partial void OnVoiceChanged(string value) => Save(s => s.Voice = value);
    partial void OnAgentContextChanged(int value) => Save(s => s.ApiContextLength = value);

    private static void Save(Action<AppSettings> change)
    {
        change(AppSettings.Current);
        AppSettings.Save();
    }

    /// <summary>Re-reads engines, Claude Code and folder sizes (sizes on a background thread).</summary>
    [RelayCommand]
    public async Task RefreshAsync()
    {
        OnPropertyChanged(nameof(TtsInstalled));
        OnPropertyChanged(nameof(AsrInstalled));
        OnPropertyChanged(nameof(TtsLine));
        OnPropertyChanged(nameof(AsrLine));
        OnPropertyChanged(nameof(ClaudePath));
        OnPropertyChanged(nameof(ClaudeInstalled));
        OnPropertyChanged(nameof(ComfyLine));

        // Engines first: folder sizes walk tens of thousands of files (ComfyUI's Python) and can take seconds.
        var engines = await Task.Run(ReadEngines);
        Engines.Clear();
        foreach (var e in engines)
            Engines.Add(e);
        ClaudeVersion = await Task.Run(ReadClaudeVersion);

        var (storage, studio) = await Task.Run(() => (ReadStorage(), ReadStudioModels()));
        Storage.Clear();
        foreach (var item in storage)
            Storage.Add(item);
        StudioModels.Clear();
        foreach (var item in studio)
            StudioModels.Add(item);
        StorageTotal = Fmt.Bytes(storage.Where(i => i.Key != "gallery").Sum(i => i.Bytes));
    }

    private static List<EngineInfo> ReadEngines()
    {
        var list = new List<EngineInfo>();
        var running = Ioc.Engines.Llama.Variant;
        var llamaRoot = Path.Combine(AppPaths.Engines, "llama.cpp");
        var builds = Directory.Exists(llamaRoot) ? Directory.GetDirectories(llamaRoot).Select(Path.GetFileName).OfType<string>().Order().ToList() : [];
        var tag = builds.Select(b => b.Split('-')[0]).Distinct().LastOrDefault() ?? "";
        var variants = string.Join(", ", builds.Select(b => b[(b.IndexOf('-') + 1)..]));
        list.Add(new EngineInfo("llama.cpp", tag, running is not null ? Loc.F("SettingsEngineRunning", running) : variants.Length > 0 ? variants : Loc.S("SettingsEngineOnDemand"),
            builds.Count > 0, llamaRoot));

        if (LiteRtEngine.IsSupported)
        {
            var root = Path.Combine(AppPaths.Engines, "litert-lm");
            var versions = Directory.Exists(root) ? Directory.GetDirectories(root).Select(Path.GetFileName).OfType<string>().Order().ToList() : [];
            list.Add(new EngineInfo("LiteRT-LM", versions.LastOrDefault() ?? "", versions.Count > 0 ? Loc.S("SettingsLiteRtDetail") : Loc.S("SettingsEngineOnDemand"),
                versions.Count > 0, root));
        }
        if (OperatingSystem.IsMacOS())
            list.Add(new EngineInfo("Apple Foundation Models", "", Loc.S("SettingsFoundationDetail"), true, ""));
        return list;
    }

    private static List<StorageItem> ReadStorage()
    {
        var engines = AppPaths.Engines;
        // Studio's NPU models live under models/ but count as Studio.
        var npu = Size(Services.Studio.Npu.SdNpuService.Root);
        return
        [
            new("models", Loc.S("SettingsStorageModels"), AppPaths.Models, Size(AppPaths.Models) - npu),
            new("engines", Loc.S("SettingsStorageEngines"), engines,
                Size(engines) - Size(ComfyService.Root) - Size(SpeechService.TtsRoot) - Size(SpeechService.WhisperRoot)),
            new("speech", Loc.S("SettingsSpeech"), engines, Size(SpeechService.TtsRoot) + Size(SpeechService.WhisperRoot)),
            new("studio", Loc.S("SettingsStudio"), StudioViewModel.UsesNpu ? Services.Studio.Npu.SdNpuService.Root : ComfyService.Root,
                Size(ComfyService.Root) + npu),
            new("chats", Loc.S("SettingsStorageChats"), AppPaths.Root, File.Exists(AppPaths.Database) ? new FileInfo(AppPaths.Database).Length : 0),
            new("attachments", Loc.S("SettingsStorageAttachments"), AppPaths.Attachments, Size(AppPaths.Attachments)),
            new("design", Loc.S("NavDesign"), Path.Combine(AppPaths.Root, "design"), Size(Path.Combine(AppPaths.Root, "design"))),
            new("agent", Loc.S("NavAgent"), Path.Combine(AppPaths.Root, "agent"), Size(Path.Combine(AppPaths.Root, "agent"))),
            new("logs", Loc.S("SettingsStorageLogs"), AppPaths.Logs, Size(AppPaths.Logs)),
            new("gallery", Loc.S("StudioGallery"), ComfyService.Gallery, Size(ComfyService.Gallery)),
        ];
    }

    private static List<StorageItem> ReadStudioModels()
    {
        if (StudioViewModel.UsesNpu)
        {
            // One entry per NPU model folder (QNN package plus text encoder).
            return Services.Studio.Npu.SdNpuModel.All.Select(m => Services.Studio.Npu.SdNpuService.ModelDir(m)).Where(Directory.Exists)
                .Select(d => new StorageItem(d, Path.GetFileName(d), d, Size(d))).OrderByDescending(i => i.Bytes).ToList();
        }
        if (!Directory.Exists(ComfyService.ModelsDir))
            return [];
        return Directory.EnumerateFiles(ComfyService.ModelsDir, "*.safetensors", SearchOption.AllDirectories)
            .Select(f => new StorageItem(f, Path.GetFileName(f), Path.GetDirectoryName(f)!, new FileInfo(f).Length))
            .OrderByDescending(i => i.Bytes).ToList();
    }

    private static string ReadClaudeVersion()
    {
        if (AgentService.FindClaude() is not { } claude)
            return "";
        try
        {
            using var p = Process.Start(new ProcessStartInfo(claude, "--version") { RedirectStandardOutput = true, UseShellExecute = false, CreateNoWindow = true })!;
            var output = p.StandardOutput.ReadToEnd().Trim();
            p.WaitForExit(5000);
            return output.Split('\n')[0];
        }
        catch (Exception e)
        {
            Log.Error("claude --version failed", e);
            return "";
        }
    }

    private static long Size(string folder)
    {
        if (!Directory.Exists(folder))
            return 0;
        try
        {
            return new DirectoryInfo(folder).EnumerateFiles("*", new EnumerationOptions { RecurseSubdirectories = true, IgnoreInaccessible = true, AttributesToSkip = FileAttributes.ReparsePoint })
                .Sum(f => f.Length);
        }
        catch (IOException)
        {
            return 0;
        }
    }

    [RelayCommand]
    private async Task SetUpSpeechAsync()
    {
        if (IsSettingUpSpeech)
            return;
        IsSettingUpSpeech = true;
        SpeechStatus = Loc.S("SettingsSettingUp");
        try
        {
            await Ioc.Speech.SetUpAsync(CancellationToken.None);
            SpeechStatus = "";
        }
        catch (Exception e)
        {
            Log.Error("Speech setup failed", e);
            SpeechStatus = Loc.F("ErrorPrefix", e.Message);
        }
        finally
        {
            IsSettingUpSpeech = false;
            await RefreshAsync();
        }
    }

    [RelayCommand]
    private async Task TestVoiceAsync() => await Ioc.Speech.ToggleSpeakAsync("settings-voice", Loc.S("SettingsVoiceSample"));

    [RelayCommand]
    private async Task RemoveSpeechAsync()
    {
        Ioc.Speech.Remove();
        await RefreshAsync();
    }

    [RelayCommand]
    private async Task InstallClaudeAsync()
    {
        await Ioc.Agent.InstallClaudeAsync();
        await RefreshAsync();
    }

    [RelayCommand]
    private async Task StartComfyAsync()
    {
        if (StudioViewModel.UsesNpu)
            return; // the NPU model loads when Studio generates
        try
        {
            await Ioc.Comfy.EnsureRunningAsync(CancellationToken.None);
        }
        catch (Exception e)
        {
            Log.Error("ComfyUI start failed", e);
        }
        await RefreshAsync();
    }

    [RelayCommand]
    private async Task StopComfyAsync()
    {
        if (StudioViewModel.UsesNpu)
            await Ioc.SdNpu.StopAsync();
        Ioc.Comfy.Stop();
        await RefreshAsync();
    }

    [RelayCommand]
    private async Task DeleteStudioModelAsync(StorageItem? item)
    {
        if (item is not null && Directory.Exists(item.Key))
        {
            // An NPU model folder.
            await Ioc.SdNpu.StopAsync();
            Directory.Delete(item.Key, true);
            await RefreshAsync();
            return;
        }
        if (item is null || !File.Exists(item.Key))
            return;
        File.Delete(item.Key);
        await RefreshAsync();
    }

    [RelayCommand]
    private async Task ClearAsync(StorageItem? item)
    {
        if (item is not { CanClear: true } || !Directory.Exists(item.Folder))
            return;
        foreach (var file in Directory.EnumerateFiles(item.Folder))
        {
            try
            {
                // Today's log is still open.
                File.Delete(file);
            }
            catch (IOException)
            {
            }
        }
        await RefreshAsync();
    }
}
