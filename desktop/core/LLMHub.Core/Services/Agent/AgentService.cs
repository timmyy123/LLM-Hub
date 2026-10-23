using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using CommunityToolkit.Mvvm.ComponentModel;

namespace LLMHub.Services.Agent;

/// <summary>
/// The Agent tab's engine: Claude Code (installed by the user, or through Settings › Agent) driven headless, one process per
/// turn, talking to a local model through LLM Hub's private Agent bridge. Nothing leaves the computer.
/// Each turn: <c>claude -p --input-format stream-json --output-format stream-json --verbose</c> with
/// <c>--session-id</c> the first time and <c>--resume</c> after, in the session's project folder.
/// </summary>
public sealed partial class AgentService : ObservableObject
{
    private static readonly JsonSerializerOptions Json = new() { WriteIndented = true };
    private readonly Dictionary<string, Process> _running = [];

    public AgentService()
    {
        Permissions = new AgentPermissions(this);
        InstallStatus = "";
        foreach (var info in LoadInfos())
        {
            var session = new AgentSession(info);
            Replay(session);
            Sessions.Add(session);
        }
    }

    public AgentPermissions Permissions { get; }
    public ObservableCollection<AgentSession> Sessions { get; } = [];
    [ObservableProperty] public partial string InstallStatus { get; set; }
    [ObservableProperty] public partial bool IsInstalling { get; set; }

    private static string Root => Directory.CreateDirectory(Path.Combine(AppPaths.Root, "agent")).FullName;
    private static string SessionsFile => Path.Combine(Root, "sessions.json");
    private static string EventLog(string id) => Path.Combine(Root, id + ".jsonl");

    public AgentSession? FindSession(string id) => Sessions.FirstOrDefault(s => s.Id == id);

    // ---------------------------------------------------------------- Claude Code

    /// <summary>The claude executable: PATH, the native installer's ~/.local/bin, WinGet's links, Homebrew.</summary>
    public static string? FindClaude()
    {
        var exe = OperatingSystem.IsWindows() ? "claude.exe" : "claude";
        var home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        var candidates = (Environment.GetEnvironmentVariable("PATH") ?? "").Split(Path.PathSeparator)
            .Select(dir => Path.Combine(dir, exe))
            .Append(Path.Combine(home, ".local", "bin", exe))
            .Append(Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "Microsoft", "WinGet", "Links", exe))
            .Append("/opt/homebrew/bin/claude")
            .Append("/usr/local/bin/claude");
        return candidates.FirstOrDefault(File.Exists);
    }

    public bool IsClaudeInstalled => FindClaude() is not null;

    /// <summary>Runs Anthropic's official native installer (no Node needed).</summary>
    public async Task InstallClaudeAsync()
    {
        if (IsInstalling)
            return;
        IsInstalling = true;
        InstallStatus = Loc.S("AgentInstalling");
        try
        {
            var info = OperatingSystem.IsWindows()
                ? new ProcessStartInfo("powershell.exe", "-NoProfile -ExecutionPolicy Bypass -Command \"irm https://claude.ai/install.ps1 | iex\"")
                : new ProcessStartInfo("/bin/bash", "-c \"curl -fsSL https://claude.ai/install.sh | bash\"");
            info.UseShellExecute = false;
            info.CreateNoWindow = true;
            info.RedirectStandardOutput = true;
            info.RedirectStandardError = true;
            using var process = Process.Start(info)!;
            var output = await process.StandardOutput.ReadToEndAsync() + await process.StandardError.ReadToEndAsync();
            await process.WaitForExitAsync();
            Log.Info($"Claude Code installer exited {process.ExitCode}: {output[^Math.Min(output.Length, 400)..]}");
            InstallStatus = IsClaudeInstalled ? "" : Loc.F("ErrorPrefix", $"installer exit {process.ExitCode}");
        }
        catch (Exception e)
        {
            Log.Error("Claude Code install failed", e);
            InstallStatus = Loc.F("ErrorPrefix", e.Message);
        }
        finally
        {
            IsInstalling = false;
            OnPropertyChanged(nameof(IsClaudeInstalled));
        }
    }

    // ---------------------------------------------------------------- sessions

    public AgentSession CreateSession(string folder, string modelId, string mode)
    {
        var session = new AgentSession(new AgentSessionInfo { Folder = folder, ModelId = modelId, Mode = mode, Title = Loc.S("NewSession") });
        Sessions.Insert(0, session);
        UpdateBranch(session);
        SaveInfos();
        return session;
    }

    public void DeleteSession(AgentSession session)
    {
        Stop(session);
        Sessions.Remove(session);
        try
        {
            File.Delete(EventLog(session.Id));
        }
        catch (IOException)
        {
        }
        SaveInfos();
    }

    /// <summary>Sends one message and streams the agent's work into the session until it finishes or is stopped.</summary>
    public async Task SendAsync(AgentSession session, string text)
    {
        if (session.IsWorking || text.Trim().Length == 0)
            return;
        var claude = FindClaude() ?? throw new InvalidOperationException(Loc.S("AgentNotInstalled"));
        var model = Ioc.Models.Installed.FirstOrDefault(m => m.Id == session.Info.ModelId)
                    ?? throw new InvalidOperationException(Loc.S("AgentNoModel"));
        await Ioc.AgentBridge.EnsureBridgeAsync();

        if (!session.Info.Started)
            session.Title = session.Info.Title = text.Length <= 60 ? text : text[..57] + "…";
        Append(session, new JsonObject { ["type"] = "llmhub_user", ["text"] = text });
        session.Items.Add(new AgentItem(AgentItemKind.User, text));
        session.Status = AgentStatus.Working;

        var mcpConfig = Path.Combine(Root, session.Id + ".mcp.json");
        await File.WriteAllTextAsync(mcpConfig, new JsonObject
        {
            ["mcpServers"] = new JsonObject
            {
                ["llmhub"] = new JsonObject
                {
                    ["type"] = "http",
                    ["url"] = $"{Ioc.AgentBridge.BaseUrl}/mcp?session={session.Id}",
                    ["headers"] = new JsonObject { ["Authorization"] = "Bearer " + Ioc.AgentBridge.Key },
                },
            },
        }.ToJsonString());

        var apiModel = LocalApiServer.ApiName(model);
        var info = new ProcessStartInfo(claude)
        {
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardInput = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            StandardOutputEncoding = Encoding.UTF8,
            WorkingDirectory = session.Folder,
        };
        foreach (var arg in new[]
                 {
                     "-p", "--input-format", "stream-json", "--output-format", "stream-json", "--verbose",
                     "--model", apiModel, "--permission-mode", session.Info.Mode,
                     "--mcp-config", mcpConfig, "--permission-prompt-tool", $"mcp__llmhub__{AgentPermissions.ToolName}",
                     session.Info.Started ? "--resume" : "--session-id", session.Id,
                 })
            info.ArgumentList.Add(arg);
        info.Environment["ANTHROPIC_BASE_URL"] = Ioc.AgentBridge.BaseUrl;
        info.Environment["ANTHROPIC_AUTH_TOKEN"] = Ioc.AgentBridge.Key;
        info.Environment["ANTHROPIC_API_KEY"] = "";
        foreach (var tier in new[] { "OPUS", "SONNET", "HAIKU" })
            info.Environment[$"ANTHROPIC_DEFAULT_{tier}_MODEL"] = apiModel;
        info.Environment["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1";
        info.Environment["CLAUDE_CODE_ENABLE_TODO_TOOLS"] = "1";
        info.Environment["DISABLE_AUTOUPDATER"] = "1";
        info.Environment["DISABLE_TELEMETRY"] = "1";

        Process process;
        try
        {
            process = Process.Start(info)!;
        }
        catch (Exception e)
        {
            Fail(session, e.Message);
            return;
        }
        _running[session.Id] = process;
        Ioc.Engines.Llama.Track(process);
        session.Info.Started = true;
        session.Info.UpdatedAt = DateTimeOffset.Now;
        SaveInfos();
        _ = process.StandardError.ReadToEndAsync().ContinueWith(t =>
        {
            if (t.Result.Trim().Length > 0)
                Log.Info($"claude stderr: {t.Result[..Math.Min(t.Result.Length, 600)]}");
        });

        var message = new JsonObject
        {
            ["type"] = "user",
            ["message"] = new JsonObject { ["role"] = "user", ["content"] = text },
        };
        await process.StandardInput.WriteLineAsync(message.ToJsonString());
        process.StandardInput.Close();

        try
        {
            while (await process.StandardOutput.ReadLineAsync() is { } line)
            {
                if (line.Length == 0 || line[0] != '{')
                    continue;
                JsonObject? e;
                try
                {
                    e = JsonNode.Parse(line) as JsonObject;
                }
                catch (JsonException)
                {
                    continue;
                }
                if (e is null || e["type"]?.GetValue<string>() is "stream_event" || e["subtype"]?.GetValue<string>() == "thinking_tokens")
                    continue;
                AppendLine(session, line);
                UiThread.Run(() => Apply(session, e));
            }
            await process.WaitForExitAsync();
        }
        catch (Exception ex)
        {
            Log.Error("Agent stream failed", ex);
            Fail(session, ex.Message);
        }
        finally
        {
            _running.Remove(session.Id);
            UiThread.Run(() =>
            {
                if (session.Status != AgentStatus.Error)
                    session.Status = AgentStatus.Idle;
                session.Pending = null;
                RefreshChanges(session);
            });
            session.Info.UpdatedAt = DateTimeOffset.Now;
            SaveInfos();
        }
    }

    public void Stop(AgentSession session)
    {
        if (session.Pending is { } pending)
            pending.Decision.TrySetResult(false);
        if (_running.TryGetValue(session.Id, out var process) && !process.HasExited)
            process.Kill(entireProcessTree: true);
    }

    /// <summary>Called by the MCP permission tool; resolves when the user answers (or the turn is stopped).</summary>
    internal async Task<bool> RequestPermissionAsync(AgentSession session, string tool, JsonObject input, CancellationToken ct)
    {
        if (session.Info.AllowedTools.Contains(tool))
            return true;
        var (summary, question) = Describe(tool, input);
        var request = new PermissionRequest { Tool = tool, Summary = summary, Question = question };
        UiThread.Run(() =>
        {
            session.Pending = request;
            session.Status = AgentStatus.NeedsYou;
        });
        using var registration = ct.Register(() => request.Decision.TrySetResult(false));
        var allowed = await request.Decision.Task;
        UiThread.Run(() =>
        {
            if (session.Pending == request)
                session.Pending = null;
            if (session.IsWorking)
                session.Status = AgentStatus.Working;
        });
        return allowed;
    }

    /// <summary>The user's answer to <see cref="AgentSession.Pending"/>.</summary>
    public void Answer(AgentSession session, bool allow, bool forSession = false)
    {
        if (session.Pending is not { } pending)
            return;
        if (allow && forSession && !session.Info.AllowedTools.Contains(pending.Tool))
        {
            session.Info.AllowedTools.Add(pending.Tool);
            SaveInfos();
        }
        pending.Decision.TrySetResult(allow);
    }

    private static (string Summary, string Question) Describe(string tool, JsonObject input) => tool switch
    {
        "Bash" => (input["command"]?.GetValue<string>() ?? "", Loc.S("AgentRunCommand")),
        "Edit" or "MultiEdit" or "Write" or "NotebookEdit" => (input["file_path"]?.GetValue<string>() ?? input["notebook_path"]?.GetValue<string>() ?? "", Loc.S("AgentEditFile")),
        "WebFetch" => (input["url"]?.GetValue<string>() ?? "", Loc.S("AgentFetch")),
        _ => (input.ToJsonString().Length > 300 ? input.ToJsonString()[..300] : input.ToJsonString(), Loc.F("AgentUseTool", tool)),
    };

    // ---------------------------------------------------------------- events → transcript

    private readonly Dictionary<string, AgentItem> _toolItems = [];

    private void Apply(AgentSession session, JsonObject e)
    {
        switch (e["type"]?.GetValue<string>())
        {
            case "llmhub_user":
                session.Items.Add(new AgentItem(AgentItemKind.User, e["text"]?.GetValue<string>() ?? ""));
                break;
            case "assistant":
                foreach (var block in e["message"]?["content"]?.AsArray() ?? [])
                {
                    if (block?["type"]?.GetValue<string>() == "text" && block["text"]?.GetValue<string>() is { Length: > 0 } text)
                        session.Items.Add(new AgentItem(AgentItemKind.Assistant, text.Trim()));
                    else if (block?["type"]?.GetValue<string>() == "tool_use")
                        AddToolUse(session, block.AsObject());
                }
                break;
            case "user":
                foreach (var block in e["message"]?["content"] as JsonArray ?? [])
                {
                    if (block?["type"]?.GetValue<string>() != "tool_result")
                        continue;
                    var id = block["tool_use_id"]?.GetValue<string>() ?? "";
                    var content = block["content"] switch
                    {
                        JsonValue v => v.GetValue<string>(),
                        JsonArray a => string.Concat(a.Select(x => x?["text"]?.GetValue<string>())),
                        _ => "",
                    };
                    if (_toolItems.TryGetValue(id, out var item))
                    {
                        item.Failed = block["is_error"]?.GetValue<bool>() == true;
                        if (item.Kind == AgentItemKind.Command)
                        {
                            item.Output = content.Length > 4000 ? content[..4000] + "\n…" : content;
                            session.Terminal += item.Output + "\n";
                        }
                        else if (item.Kind == AgentItemKind.Read && !item.Failed)
                            item.Meta = Loc.F("AgentLines", content.Count(c => c == '\n').ToString());
                        else if (item.Failed)
                            item.Output = content.Length > 600 ? content[..600] : content;
                    }
                }
                break;
            case "result":
                if (e["is_error"]?.GetValue<bool>() == true && e["result"]?.GetValue<string>() is { } error)
                    Fail(session, error);
                break;
        }
    }

    private void AddToolUse(AgentSession session, JsonObject block)
    {
        var name = block["name"]?.GetValue<string>() ?? "";
        var input = block["input"] as JsonObject ?? new JsonObject();
        var id = block["id"]?.GetValue<string>() ?? "";
        string Str(string key) => input[key]?.GetValue<string>() ?? "";
        AgentItem item;
        switch (name)
        {
            case "Read":
                item = new AgentItem(AgentItemKind.Read, Relative(session, Str("file_path")), Loc.S("AgentRead")) { ToolUseId = id };
                break;
            case "Edit":
            case "MultiEdit":
            {
                var diff = new List<DiffLine>();
                var edits = name == "MultiEdit" ? input["edits"]?.AsArray().Select(x => (x?["old_string"]?.GetValue<string>() ?? "", x?["new_string"]?.GetValue<string>() ?? "")).ToList() ?? []
                    : [(Str("old_string"), Str("new_string"))];
                foreach (var (oldText, newText) in edits)
                    diff.AddRange(LineDiff(oldText, newText));
                item = new AgentItem(AgentItemKind.Edit, Relative(session, Str("file_path")), Loc.S("AgentEdit")) { ToolUseId = id, Diff = diff };
                item.Meta = $"+{diff.Count(d => d.Kind == '+')} −{diff.Count(d => d.Kind == '-')}";
                break;
            }
            case "Write":
            {
                var lines = Str("content").Split('\n').Select(l => new DiffLine('+', l)).ToList();
                item = new AgentItem(AgentItemKind.Write, Relative(session, Str("file_path")), Loc.S("AgentWrite")) { ToolUseId = id, Diff = lines };
                item.Meta = $"+{lines.Count}";
                break;
            }
            case "Bash":
                item = new AgentItem(AgentItemKind.Command, Str("command"), Loc.S("AgentRun")) { ToolUseId = id };
                session.Terminal += "$ " + Str("command") + "\n";
                break;
            case "Grep":
            case "Glob":
            case "LS":
                item = new AgentItem(AgentItemKind.Search, Str("pattern") is { Length: > 0 } p ? p : Relative(session, Str("path")), name) { ToolUseId = id };
                break;
            case "TodoWrite":
                session.Tasks.Clear();
                foreach (var todo in input["todos"]?.AsArray() ?? [])
                    session.Tasks.Add(new AgentTask(todo?["content"]?.GetValue<string>() ?? "", todo?["status"]?.GetValue<string>() ?? "pending"));
                return;
            default:
                item = new AgentItem(AgentItemKind.Tool, input.ToJsonString().Length > 160 ? input.ToJsonString()[..160] + "…" : input.ToJsonString(), name) { ToolUseId = id };
                break;
        }
        _toolItems[id] = item;
        session.Items.Add(item);
    }

    private static string Relative(AgentSession session, string path)
    {
        if (path.Length == 0)
            return path;
        try
        {
            var relative = Path.GetRelativePath(session.Folder, path);
            return relative.StartsWith("..", StringComparison.Ordinal) ? path : relative.Replace('\\', '/');
        }
        catch (ArgumentException)
        {
            return path;
        }
    }

    /// <summary>Removed lines, then added lines, keeping lines common to both as context.</summary>
    internal static List<DiffLine> LineDiff(string oldText, string newText)
    {
        var a = oldText.Split('\n');
        var b = newText.Split('\n');
        var prefix = 0;
        while (prefix < a.Length && prefix < b.Length && a[prefix] == b[prefix])
            prefix++;
        var suffix = 0;
        while (suffix < a.Length - prefix && suffix < b.Length - prefix && a[^(suffix + 1)] == b[^(suffix + 1)])
            suffix++;
        var lines = new List<DiffLine>();
        lines.AddRange(a.Take(prefix).Select(l => new DiffLine(' ', l)));
        lines.AddRange(a.Skip(prefix).Take(a.Length - prefix - suffix).Select(l => new DiffLine('-', l)));
        lines.AddRange(b.Skip(prefix).Take(b.Length - prefix - suffix).Select(l => new DiffLine('+', l)));
        lines.AddRange(a.Skip(a.Length - suffix).Select(l => new DiffLine(' ', l)));
        return lines;
    }

    private void Fail(AgentSession session, string message) => UiThread.Run(() =>
    {
        session.Items.Add(new AgentItem(AgentItemKind.Error, message));
        session.Status = AgentStatus.Error;
    });

    // ---------------------------------------------------------------- git: changes, revert, commit

    public void RefreshChanges(AgentSession session)
    {
        UpdateBranch(session);
        session.Changes.Clear();
        if (Git(session.Folder, "rev-parse", "--is-inside-work-tree").Trim() == "true")
        {
            foreach (var line in Git(session.Folder, "status", "--porcelain", "--untracked-files=all").Split('\n', StringSplitOptions.RemoveEmptyEntries))
            {
                var code = line[..2].Trim();
                var file = line[3..].Trim().Trim('"');
                if (code == "??")
                {
                    var full = Path.Combine(session.Folder, file);
                    var content = File.Exists(full) ? File.ReadAllLines(full) : [];
                    session.Changes.Add(new FileChange(file, 'A', content.Length, 0, content.Take(400).Select(l => new DiffLine('+', l)).ToList()));
                    continue;
                }
                var diff = ParseUnifiedDiff(Git(session.Folder, "diff", "HEAD", "--", file));
                session.Changes.Add(new FileChange(file, code.Contains('D') ? 'D' : 'M', diff.Count(d => d.Kind == '+'), diff.Count(d => d.Kind == '-'), diff));
            }
            return;
        }
        // Not a git repository: show what the agent's edits in this session did.
        foreach (var group in session.Items.Where(i => i.Kind is AgentItemKind.Edit or AgentItemKind.Write).GroupBy(i => i.Title))
        {
            var diff = group.SelectMany(i => i.Diff).ToList();
            session.Changes.Add(new FileChange(group.Key, group.Any(i => i.Kind == AgentItemKind.Write) ? 'A' : 'M',
                diff.Count(d => d.Kind == '+'), diff.Count(d => d.Kind == '-'), diff));
        }
    }

    public bool IsGitRepo(AgentSession session) => Git(session.Folder, "rev-parse", "--is-inside-work-tree").Trim() == "true";

    /// <summary>Puts every changed file back to the last commit and removes files the agent added.</summary>
    public void Revert(AgentSession session)
    {
        if (!IsGitRepo(session))
            return;
        foreach (var change in session.Changes)
        {
            if (change.Kind == 'A')
                File.Delete(Path.Combine(session.Folder, change.Path));
            else
                Git(session.Folder, "checkout", "HEAD", "--", change.Path);
        }
        RefreshChanges(session);
    }

    public string Commit(AgentSession session, string message)
    {
        if (!IsGitRepo(session) || session.Changes.Count == 0)
            return "";
        Git(session.Folder, ["add", "--", .. session.Changes.Select(c => c.Path)]);
        var output = Git(session.Folder, "commit", "-m", message);
        RefreshChanges(session);
        return output;
    }

    private static void UpdateBranch(AgentSession session) =>
        session.Branch = Directory.Exists(session.Folder) ? Git(session.Folder, "branch", "--show-current").Trim() : "";

    private static List<DiffLine> ParseUnifiedDiff(string diff)
    {
        var lines = new List<DiffLine>();
        foreach (var line in diff.Split('\n'))
        {
            if (line.StartsWith("@@", StringComparison.Ordinal))
                lines.Add(new DiffLine('@', line));
            else if (line.StartsWith("+++", StringComparison.Ordinal) || line.StartsWith("---", StringComparison.Ordinal) || line.StartsWith("diff ", StringComparison.Ordinal) || line.StartsWith("index ", StringComparison.Ordinal))
                continue;
            else if (line.Length > 0 && line[0] is '+' or '-' or ' ')
                lines.Add(new DiffLine(line[0], line[1..]));
        }
        return lines;
    }

    private static string Git(string folder, params string[] args)
    {
        try
        {
            var info = new ProcessStartInfo("git") { UseShellExecute = false, CreateNoWindow = true, RedirectStandardOutput = true, RedirectStandardError = true, WorkingDirectory = folder };
            foreach (var arg in args)
                info.ArgumentList.Add(arg);
            using var process = Process.Start(info)!;
            var output = process.StandardOutput.ReadToEnd();
            process.WaitForExit(15000);
            return output;
        }
        catch (Exception)
        {
            return "";
        }
    }

    // ---------------------------------------------------------------- storage

    private static List<AgentSessionInfo> LoadInfos()
    {
        try
        {
            return File.Exists(SessionsFile) ? JsonSerializer.Deserialize<List<AgentSessionInfo>>(File.ReadAllText(SessionsFile)) ?? [] : [];
        }
        catch (Exception e)
        {
            Log.Error("Agent sessions could not be read", e);
            return [];
        }
    }

    private void SaveInfos()
    {
        try
        {
            File.WriteAllText(SessionsFile, JsonSerializer.Serialize(Sessions.Select(s => s.Info).ToList(), Json));
        }
        catch (Exception e)
        {
            Log.Error("Agent sessions could not be saved", e);
        }
    }

    private static void Append(AgentSession session, JsonObject e) => AppendLine(session, e.ToJsonString());

    private static void AppendLine(AgentSession session, string line)
    {
        lock (session)
            File.AppendAllText(EventLog(session.Id), line + "\n");
    }

    /// <summary>Rebuilds a saved session's transcript from its event log.</summary>
    private void Replay(AgentSession session)
    {
        if (!File.Exists(EventLog(session.Id)))
            return;
        foreach (var line in File.ReadLines(EventLog(session.Id)))
        {
            try
            {
                if (JsonNode.Parse(line) is JsonObject e)
                    Apply(session, e);
            }
            catch (JsonException)
            {
            }
        }
        session.Status = AgentStatus.Idle;
    }
}
