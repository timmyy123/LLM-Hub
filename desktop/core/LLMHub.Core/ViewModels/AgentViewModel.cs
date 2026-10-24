using System.Collections.ObjectModel;
using System.Collections.Specialized;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using LLMHub.Services;
using LLMHub.Services.Agent;

namespace LLMHub.ViewModels;

/// <summary>Sessions of one project folder, for the sidebar.</summary>
public sealed class AgentGroup(string key, IEnumerable<AgentSession> items) : List<AgentSession>(items)
{
    public string Key { get; } = key;
}

public sealed record AgentModeOption(string Mode, string Label)
{
    public override string ToString() => Label;
}

/// <summary>The Agent tab: sessions by folder, the open session, its composer, and the side panel (Diff, Terminal, Files).</summary>
public sealed partial class AgentViewModel : ObservableObject
{
    private readonly AgentService _agent = Ioc.Agent;

    public AgentViewModel()
    {
        Draft = "";
        CommitMessage = "";
        Notice = "";
        SideTab = "diff";
        Modes = [new("default", Loc.S("AgentModeAsk")), new("acceptEdits", Loc.S("AgentModeAuto")), new("plan", Loc.S("AgentModePlan"))];
        Mode = Modes[0];
        EngineChoice = Engines.FirstOrDefault(e => e.Mode == AppSettings.Current.AgentEngine) ?? Engines[0];
        _agent.PropertyChanged += (_, _) => OnEngineStateChanged();
        Ioc.OpenClaw.PropertyChanged += (_, _) => OnEngineStateChanged();
        _agent.Sessions.CollectionChanged += (_, _) => RebuildGroups();
        RebuildGroups();
        SelectedModel = Models.FirstOrDefault(m => m.Id == AppSettings.Current.AgentModelId)
                        ?? Models.FirstOrDefault(m => m.CatalogId == Ioc.Catalog.DefaultFor("agent"))
                        ?? Models.FirstOrDefault(m => m.Format == "gguf");
        Selected = _agent.Sessions.FirstOrDefault();
    }

    public ObservableCollection<AgentGroup> Groups { get; } = [];
    public ObservableCollection<InstalledModel> Models => Ioc.Models.ChatModels;
    public IReadOnlyList<AgentModeOption> Modes { get; }
    /// <summary>The agent engines: Claude Code, or OpenClaw (installed by the app, Chinese-origin extensions removed).</summary>
    public IReadOnlyList<AgentModeOption> Engines { get; } = [new("claude", "Claude Code"), new("openclaw", "OpenClaw")];
    public ObservableCollection<string> Files { get; } = [];

    [ObservableProperty] public partial AgentSession? Selected { get; set; }
    [ObservableProperty] public partial string Draft { get; set; }
    [ObservableProperty] public partial AgentModeOption Mode { get; set; }
    /// <summary>The engine new sessions use; a session keeps the engine it started with.</summary>
    [ObservableProperty] public partial AgentModeOption EngineChoice { get; set; }
    [ObservableProperty] public partial InstalledModel? SelectedModel { get; set; }
    [ObservableProperty] public partial FileChange? SelectedChange { get; set; }
    /// <summary>diff, terminal, files or preview.</summary>
    [ObservableProperty] public partial string SideTab { get; set; }
    [ObservableProperty] public partial string CommitMessage { get; set; }
    /// <summary>Errors and results (commit output) shown above the composer.</summary>
    [ObservableProperty] public partial string Notice { get; set; }

    public bool HasSession => Selected is not null;
    public bool IsEngineInstalled => EngineChoice.Mode == "openclaw" ? OpenClawService.IsInstalled : _agent.IsClaudeInstalled;
    public bool IsEngineInstalling => EngineChoice.Mode == "openclaw" ? Ioc.OpenClaw.IsInstalling : _agent.IsInstalling;
    public string InstallStatus => EngineChoice.Mode == "openclaw" ? Ioc.OpenClaw.Status : _agent.InstallStatus;
    /// <summary>What setting up the chosen engine does, on the install card.</summary>
    public string InstallHelp => Loc.S(EngineChoice.Mode == "openclaw" ? "OpenClawInstallHelp" : "AgentInstallHelp");

    private void OnEngineStateChanged() => UiThread.Post(() =>
    {
        OnPropertyChanged(nameof(IsEngineInstalled));
        OnPropertyChanged(nameof(IsEngineInstalling));
        OnPropertyChanged(nameof(InstallStatus));
    });

    partial void OnEngineChoiceChanged(AgentModeOption value)
    {
        AppSettings.Current.AgentEngine = value.Mode;
        AppSettings.Save();
        OnPropertyChanged(nameof(InstallHelp));
        OnEngineStateChanged();
    }
    public AgentService Engine => _agent;
    /// <summary>"via llama.cpp · Gemma 4 26B" on the engine card.</summary>
    public string EngineLine => SelectedModel is { } m ? $"via {(m.Format == "litertlm" ? "LiteRT-LM" : "llama.cpp")} · {m.Name}" : Loc.S("AgentNoModel");

    partial void OnSelectedChanged(AgentSession? oldValue, AgentSession? newValue)
    {
        if (oldValue is not null)
            oldValue.Changes.CollectionChanged -= OnChangesChanged;
        OnPropertyChanged(nameof(HasSession));
        Notice = "";
        if (newValue is null)
            return;
        newValue.Changes.CollectionChanged += OnChangesChanged;
        Mode = Modes.FirstOrDefault(m => m.Mode == newValue.Info.Mode) ?? Modes[0];
        if (Models.FirstOrDefault(m => m.Id == newValue.Info.ModelId) is { } model)
            SelectedModel = model;
        _agent.RefreshChanges(newValue);
        SelectedChange = newValue.Changes.FirstOrDefault();
        LoadFiles();
    }

    private void OnChangesChanged(object? sender, NotifyCollectionChangedEventArgs e)
    {
        if (SelectedChange is null || Selected?.Changes.Contains(SelectedChange) != true)
            SelectedChange = Selected?.Changes.FirstOrDefault();
    }

    partial void OnModeChanged(AgentModeOption value)
    {
        if (Selected is { } s && s.Info.Mode != value.Mode)
            s.Info.Mode = value.Mode;
    }

    partial void OnSelectedModelChanged(InstalledModel? value)
    {
        OnPropertyChanged(nameof(EngineLine));
        if (value is null)
            return;
        AppSettings.Current.AgentModelId = value.Id;
        AppSettings.Save();
        if (Selected is { } s)
            s.Info.ModelId = value.Id;
    }

    /// <summary>Starts a session in <paramref name="folder"/> (from the folder picker).</summary>
    public void NewSession(string folder)
    {
        if (SelectedModel is null)
        {
            Notice = Loc.S("AgentNoModel");
            return;
        }
        Selected = _agent.CreateSession(folder, SelectedModel.Id, Mode.Mode, EngineChoice.Mode);
    }

    [RelayCommand]
    private async Task SendAsync()
    {
        var text = Draft.Trim();
        if (Selected is not { } session || text.Length == 0 || session.IsWorking)
            return;
        if (session.Info.Engine == "openclaw" ? !OpenClawService.IsInstalled : !_agent.IsClaudeInstalled)
        {
            Notice = Loc.S(session.Info.Engine == "openclaw" ? "OpenClawNotInstalled" : "AgentNotInstalled");
            return;
        }
        Draft = "";
        Notice = "";
        try
        {
            await _agent.SendAsync(session, text);
        }
        catch (Exception e)
        {
            Log.Error("Agent send failed", e);
            Notice = Loc.F("ErrorPrefix", e.Message);
        }
        SelectedChange ??= session.Changes.FirstOrDefault();
        LoadFiles();
        RebuildGroups();
    }

    [RelayCommand]
    private void Stop()
    {
        if (Selected is { } s)
            _agent.Stop(s);
    }

    [RelayCommand]
    private void Allow()
    {
        if (Selected is { } s)
            _agent.Answer(s, allow: true);
    }

    [RelayCommand]
    private void AllowForSession()
    {
        if (Selected is { } s)
            _agent.Answer(s, allow: true, forSession: true);
    }

    [RelayCommand]
    private void Deny()
    {
        if (Selected is { } s)
            _agent.Answer(s, allow: false);
    }

    [RelayCommand]
    private void Revert()
    {
        if (Selected is { } s)
            _agent.Revert(s);
        SelectedChange = Selected?.Changes.FirstOrDefault();
    }

    [RelayCommand]
    private void Commit()
    {
        if (Selected is not { } s)
            return;
        var message = CommitMessage.Trim().Length > 0 ? CommitMessage.Trim() : s.Title;
        var output = _agent.Commit(s, message);
        Notice = output.Split('\n').FirstOrDefault()?.Trim() ?? "";
        CommitMessage = "";
        SelectedChange = s.Changes.FirstOrDefault();
    }

    [RelayCommand]
    private void DeleteSession(AgentSession? session)
    {
        if (session is null)
            return;
        _agent.DeleteSession(session);
        if (Selected == session)
            Selected = _agent.Sessions.FirstOrDefault();
    }

    [RelayCommand]
    private async Task InstallEngineAsync()
    {
        if (EngineChoice.Mode != "openclaw")
        {
            await _agent.InstallClaudeAsync();
            return;
        }
        try
        {
            await Ioc.OpenClaw.InstallAsync(CancellationToken.None);
        }
        catch (Exception e) when (e is not OperationCanceledException)
        {
            Notice = Loc.F("ErrorPrefix", e.Message);
        }
        OnEngineStateChanged();
    }

    /// <summary>Project files for the Files tab (skipping dependency and build folders).</summary>
    private void LoadFiles()
    {
        Files.Clear();
        if (Selected is not { } s || !Directory.Exists(s.Folder))
            return;
        string[] skip = [".git", "node_modules", "bin", "obj", "build", "dist", ".venv", "__pycache__", ".next"];
        var queue = new Queue<string>([s.Folder]);
        while (queue.Count > 0 && Files.Count < 400)
        {
            var dir = queue.Dequeue();
            try
            {
                foreach (var sub in Directory.EnumerateDirectories(dir).Order())
                    if (!skip.Contains(Path.GetFileName(sub)))
                        queue.Enqueue(sub);
                foreach (var file in Directory.EnumerateFiles(dir).Order())
                    Files.Add(Path.GetRelativePath(s.Folder, file).Replace('\\', '/'));
            }
            catch (UnauthorizedAccessException)
            {
            }
        }
    }

    /// <summary>index.html in the project, for Preview, or null.</summary>
    public string? PreviewFile => Selected is { } s
        ? new[] { "index.html", "public/index.html", "dist/index.html" }.Select(f => Path.Combine(s.Folder, f)).FirstOrDefault(File.Exists)
        : null;

    private void RebuildGroups()
    {
        var selected = Selected;
        Groups.Clear();
        foreach (var group in _agent.Sessions.GroupBy(s => s.FolderName))
            Groups.Add(new AgentGroup(group.Key, group));
        if (selected is not null && Selected != selected)
            Selected = selected;
    }
}
