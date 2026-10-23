using System.Collections.ObjectModel;
using System.Text.Json.Serialization;
using CommunityToolkit.Mvvm.ComponentModel;

namespace LLMHub.Services.Agent;

public enum AgentStatus { Idle, Working, NeedsYou, Error }

public enum AgentItemKind { User, Assistant, Read, Edit, Write, Command, Search, Tasks, Tool, Error }

/// <summary>One line of a diff for display: '+' added, '-' removed, ' ' context, '@' hunk header.</summary>
public sealed record DiffLine(char Kind, string Text);

/// <summary>One step in the session transcript: a message, or a tool call with what it touched.</summary>
public sealed partial class AgentItem : ObservableObject
{
    public AgentItem(AgentItemKind kind, string title, string detail = "")
    {
        Kind = kind;
        Title = title;
        Detail = detail;
        Meta = "";
        Output = "";
    }

    public AgentItemKind Kind { get; }
    /// <summary>Tool use id, to attach the tool's result later.</summary>
    public string ToolUseId { get; init; } = "";
    /// <summary>Message text, or the file path / command / pattern of a tool call.</summary>
    [ObservableProperty] public partial string Title { get; set; }
    /// <summary>"Read", "Edit", "Run"…, shown as the step's label.</summary>
    [ObservableProperty] public partial string Detail { get; set; }
    /// <summary>"118 lines", "+24 −2".</summary>
    [ObservableProperty] public partial string Meta { get; set; }
    /// <summary>Command output or tool result text.</summary>
    [ObservableProperty] public partial string Output { get; set; }
    [ObservableProperty] public partial bool Failed { get; set; }
    public List<DiffLine> Diff { get; init; } = [];

    public bool IsMessage => Kind is AgentItemKind.User or AgentItemKind.Assistant;
    public bool IsUser => Kind == AgentItemKind.User;
    public bool IsAssistant => Kind == AgentItemKind.Assistant;
    public bool IsError => Kind == AgentItemKind.Error;
    public bool IsCommand => Kind == AgentItemKind.Command;
    /// <summary>A tool step shown as a card (Read, Edit, Write, Search, other tools).</summary>
    public bool IsStep => Kind is AgentItemKind.Read or AgentItemKind.Edit or AgentItemKind.Write or AgentItemKind.Search or AgentItemKind.Tool or AgentItemKind.Command;
    public bool HasDiff => Diff.Count > 0;
    public bool HasOutput => Output.Length > 0;

    partial void OnOutputChanged(string value) => OnPropertyChanged(nameof(HasOutput));
}

public sealed partial class AgentTask : ObservableObject
{
    public AgentTask(string text, string status)
    {
        Text = text;
        Status = status;
    }

    [ObservableProperty] public partial string Text { get; set; }
    /// <summary>pending, in_progress or completed.</summary>
    [ObservableProperty] public partial string Status { get; set; }
    public bool IsDone => Status == "completed";
    public bool IsActive => Status == "in_progress";
}

/// <summary>A file the agent changed in this session, for the Diff panel.</summary>
public sealed record FileChange(string Path, char Kind, int Added, int Removed, List<DiffLine> Diff)
{
    public string Stats => Kind == 'A' ? $"+{Added}" : $"+{Added} −{Removed}";
}

/// <summary>A tool call waiting for the user's Allow / Allow for this session / Deny.</summary>
public sealed class PermissionRequest
{
    public required string Tool { get; init; }
    /// <summary>"npm test -- forecast", or the file about to be edited.</summary>
    public required string Summary { get; init; }
    public required string Question { get; init; }
    internal TaskCompletionSource<bool> Decision { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously);
}

/// <summary>Saved per session in agent/sessions.json; the transcript itself is the raw event log next to it.</summary>
public sealed class AgentSessionInfo
{
    public string Id { get; set; } = Guid.NewGuid().ToString();
    public string Title { get; set; } = "";
    public string Folder { get; set; } = "";
    public string ModelId { get; set; } = "";
    /// <summary>default (ask before edits and commands), acceptEdits, or plan.</summary>
    public string Mode { get; set; } = "default";
    public bool Started { get; set; }
    public DateTimeOffset UpdatedAt { get; set; } = DateTimeOffset.Now;
    public List<string> AllowedTools { get; set; } = [];
}

public sealed partial class AgentSession : ObservableObject
{
    public AgentSession(AgentSessionInfo info)
    {
        Info = info;
        Title = info.Title;
        StatusText = "";
        Terminal = "";
        Branch = "";
    }

    [JsonIgnore] public AgentSessionInfo Info { get; }
    public string Id => Info.Id;
    public string Folder => Info.Folder;
    public string FolderName => Path.GetFileName(Info.Folder.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar));

    [ObservableProperty] public partial string Title { get; set; }
    [ObservableProperty] public partial AgentStatus Status { get; set; }
    /// <summary>"Working", "Needs you" in the session list.</summary>
    [ObservableProperty] public partial string StatusText { get; set; }
    [ObservableProperty] public partial PermissionRequest? Pending { get; set; }
    /// <summary>Commands the agent ran and their output, for the Terminal panel.</summary>
    [ObservableProperty] public partial string Terminal { get; set; }
    /// <summary>Current git branch of the folder, or "".</summary>
    [ObservableProperty] public partial string Branch { get; set; }

    public ObservableCollection<AgentItem> Items { get; } = [];
    public ObservableCollection<AgentTask> Tasks { get; } = [];
    public ObservableCollection<FileChange> Changes { get; } = [];
    public bool IsWorking => Status is AgentStatus.Working or AgentStatus.NeedsYou;

    partial void OnStatusChanged(AgentStatus value)
    {
        StatusText = value switch
        {
            AgentStatus.Working => Loc.S("AgentWorking"),
            AgentStatus.NeedsYou => Loc.S("AgentNeedsYou"),
            AgentStatus.Error => Loc.S("AgentError"),
            _ => "",
        };
        OnPropertyChanged(nameof(IsWorking));
    }
}
