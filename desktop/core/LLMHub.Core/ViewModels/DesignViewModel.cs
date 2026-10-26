using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using LLMHub.Services;
using LLMHub.Services.Design;

namespace LLMHub.ViewModels;

public sealed record DesignKindOption(DesignKind Kind, string Label)
{
    public override string ToString() => Label;
}

/// <summary>The Design tab: projects, the conversation, the live preview with versions, the design-system panel, export.</summary>
public sealed partial class DesignViewModel : ObservableObject
{
    private readonly DesignService _design = Ioc.Design;
    private CancellationTokenSource? _run;

    public DesignViewModel()
    {
        Draft = "";
        Status = "";
        LiveText = "";
        SelectedElement = "";
        PreviewSize = "desktop";
        Kinds =
        [
            new(DesignKind.Prototype, Loc.S("DesignPrototype")), new(DesignKind.Slides, Loc.S("DesignSlides")),
            new(DesignKind.Image, Loc.S("DesignImage")), new(DesignKind.Motion, Loc.S("DesignMotion")),
        ];
        Kind = Kinds[0];
        System = DesignSystem.All[0];
        SelectedModel = Models.FirstOrDefault(m => m.Id == AppSettings.Current.DesignModelId)
                        ?? Models.FirstOrDefault(m => m.CatalogId == Ioc.Catalog.DefaultFor("agent"))
                        ?? Ioc.Chat.SelectedModel;
        Selected = _design.Projects.FirstOrDefault();
    }

    public ObservableCollection<DesignProject> Projects => _design.Projects;
    public ObservableCollection<InstalledModel> Models => Ioc.Models.ChatModels;
    public ObservableCollection<DesignMessage> Messages { get; } = [];
    public IReadOnlyList<DesignKindOption> Kinds { get; }
    public IReadOnlyList<DesignSystem> Systems => DesignSystem.All;

    [ObservableProperty] public partial DesignProject? Selected { get; set; }
    [ObservableProperty] public partial DesignKindOption Kind { get; set; }
    [ObservableProperty] public partial DesignSystem System { get; set; }
    [ObservableProperty] public partial InstalledModel? SelectedModel { get; set; }
    [ObservableProperty] public partial string Draft { get; set; }
    [ObservableProperty] public partial bool IsGenerating { get; set; }
    [ObservableProperty] public partial string Status { get; set; }
    /// <summary>The model's text as it streams, for the "Writing…" view.</summary>
    [ObservableProperty] public partial string LiveText { get; set; }
    /// <summary>Version shown in the preview (1-based), or 0 before the first.</summary>
    [ObservableProperty] public partial int Version { get; set; }
    /// <summary>desktop, tablet or phone.</summary>
    [ObservableProperty] public partial string PreviewSize { get; set; }
    /// <summary>Description of the element clicked in the preview ("section · hours"), sent with the next request.</summary>
    [ObservableProperty] public partial string SelectedElement { get; set; }

    /// <summary>"Fill: #b4532a", "Radius: 10 px"… of the element clicked in the preview, one per line.</summary>
    [ObservableProperty] public partial string SelectedProperties { get; set; } = "";
    /// <summary>Set after a new prototype version; the preview then measures it at phone width and calls <see cref="OnPreviewMessage"/>.</summary>
    public bool NeedsPhoneCheck { get; set; }

    public string? PreviewFile => Selected is { } p && Version > 0 ? p.VersionFile(Version) : null;
    public string VersionText => Selected is { Versions: > 0 } p ? Loc.F("DesignVersion", Version.ToString(), p.Versions.ToString()) : "";
    public bool HasPreview => PreviewFile is not null;
    public int PreviewWidth => PreviewSize switch { "phone" => 390, "tablet" => 820, _ => 1280 };

    partial void OnSelectedChanged(DesignProject? value)
    {
        Messages.Clear();
        if (value is not null)
        {
            foreach (var m in value.History)
                Messages.Add(m);
            System = value.System;
            Kind = Kinds.First(k => k.Kind == value.Kind);
        }
        Version = value?.Versions ?? 0;
        SelectedElement = "";
        SelectedProperties = "";
        NeedsPhoneCheck = false;
        Status = "";
    }

    partial void OnVersionChanged(int value) => NotifyPreview();
    partial void OnPreviewSizeChanged(string value) => OnPropertyChanged(nameof(PreviewWidth));

    partial void OnSystemChanged(DesignSystem value)
    {
        if (Selected is { } p && p.SystemId != value.Id)
        {
            p.SystemId = value.Id;
            _design.Save(p);
        }
    }

    partial void OnSelectedModelChanged(InstalledModel? value)
    {
        if (value is null)
            return;
        AppSettings.Current.DesignModelId = value.Id;
        AppSettings.Save();
    }

    private void NotifyPreview()
    {
        OnPropertyChanged(nameof(PreviewFile));
        OnPropertyChanged(nameof(VersionText));
        OnPropertyChanged(nameof(HasPreview));
    }

    [RelayCommand]
    private void NewProject()
    {
        Selected = _design.Create(Kind.Kind, System.Id);
    }

    [RelayCommand]
    private void DeleteProject(DesignProject? project)
    {
        if (project is null)
            return;
        _design.Delete(project);
        if (Selected == project)
            Selected = Projects.FirstOrDefault();
    }

    [RelayCommand]
    private async Task SendAsync()
    {
        var request = Draft.Trim();
        if (request.Length == 0 || IsGenerating)
            return;
        if (SelectedModel is not { } model)
        {
            Status = Loc.S("AgentNoModel");
            return;
        }
        Selected ??= _design.Create(Kind.Kind, System.Id);
        var project = Selected;
        Draft = "";
        Messages.Add(new DesignMessage("user", request));
        IsGenerating = true;
        LiveText = "";
        _run = new CancellationTokenSource();
        try
        {
            var element = SelectedElement;
            var clock = global::System.Diagnostics.Stopwatch.StartNew();
            var version = await _design.GenerateAsync(project, request, element, model, text => UiThread.Post(() => LiveText = text), _run.Token);
            Log.Info($"Design {project.Id} v{version} ({project.Kind}, {project.SystemId}) in {clock.Elapsed.TotalSeconds:F0}s");
            Messages.Add(project.History[^1]);
            NeedsPhoneCheck = project.Kind == DesignKind.Prototype;
            Version = version;
            NotifyPreview();
            SelectedElement = "";
            SelectedProperties = "";
            Status = "";
        }
        catch (OperationCanceledException)
        {
            Status = "";
        }
        catch (Exception e)
        {
            Log.Error("Design generation failed", e);
            Status = Loc.F("ErrorPrefix", e.Message);
        }
        finally
        {
            IsGenerating = false;
            LiveText = "";
            _run.Dispose();
            _run = null;
        }
    }

    [RelayCommand]
    private void Stop() => _run?.Cancel();

    /// <summary>A message from the preview's injected scripts: an element the user clicked, or the phone-width measurement.</summary>
    public void OnPreviewMessage(string json)
    {
        try
        {
            using var doc = global::System.Text.Json.JsonDocument.Parse(json);
            if (doc.RootElement.TryGetProperty("type", out var type) && type.GetString() == "phone")
            {
                ReportPhoneCheck(doc.RootElement.GetProperty("width").GetInt32());
                return;
            }
            SelectedElement = DesignService.DescribePicked(json);
            SelectedProperties = string.Join("\n", DesignService.PickedProperties(json));
        }
        catch (Exception e) when (e is global::System.Text.Json.JsonException or InvalidOperationException or KeyNotFoundException)
        {
            Log.Error("Preview message", e);
        }
    }

    /// <summary>Adds "Checked layout at phone width" (or how much too wide it is) to the latest reply's steps.</summary>
    private void ReportPhoneCheck(int contentWidth)
    {
        if (!NeedsPhoneCheck || Selected is not { History.Count: > 0 } project || project.History[^1] is not { Role: "assistant" } last)
            return;
        NeedsPhoneCheck = false;
        var over = contentWidth - 390;
        var step = over <= 2 ? Loc.S("DesignStepPhone") : Loc.F("DesignStepPhoneWide", over.ToString());
        var updated = last with { Steps = [.. last.Steps ?? [], step] };
        project.History[^1] = updated;
        _design.Save(project);
        var index = Messages.IndexOf(last);
        if (index >= 0)
            Messages[index] = updated;
        Log.Info($"Design {project.Id} v{Version} at phone width: content {contentWidth} px");
    }

    [RelayCommand]
    private void PreviousVersion()
    {
        if (Version > 1)
            Version--;
    }

    [RelayCommand]
    private void NextVersion()
    {
        if (Selected is { } p && Version < p.Versions)
            Version++;
    }

    /// <summary>Starts an Agent session in the project folder to turn the design into a real codebase.</summary>
    public async Task HandOffToAgentAsync()
    {
        if (Selected is not { CurrentFile: { } file } project)
            return;
        var folder = Directory.CreateDirectory(Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments), "LLM Hub", Safe(project.Name))).FullName;
        File.Copy(file, Path.Combine(folder, "index.html"), true);
        Ioc.AgentVm.NewSession(folder);
        Ioc.AgentVm.Draft = Loc.S("DesignHandOffPrompt");
        await Task.CompletedTask;
    }

    public static string Safe(string name)
    {
        var s = string.Concat(name.Split(Path.GetInvalidFileNameChars())).Trim().TrimEnd('…', '.');
        return s.Length > 0 ? s : "design";
    }
}
