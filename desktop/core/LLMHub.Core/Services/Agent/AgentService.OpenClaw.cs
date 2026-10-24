using System.Diagnostics;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace LLMHub.Services.Agent;

/// <summary>
/// OpenClaw sessions: one <c>openclaw acp</c> process per session, kept open between turns. Its Agent Client Protocol
/// updates are turned into the same transcript events Claude Code writes (assistant text, tool_use, tool_result), so the
/// tab, the event log and replay work the same for both engines.
/// </summary>
public sealed partial class AgentService
{
    private readonly Dictionary<string, AcpConnection> _acp = [];

    private async Task SendOpenClawAsync(AgentSession session, string text)
    {
        if (!OpenClawService.IsInstalled)
            throw new InvalidOperationException(Loc.S("OpenClawNotInstalled"));
        if (Ioc.Models.ChatModels.FirstOrDefault(m => m.Id == session.Info.ModelId) is null)
            throw new InvalidOperationException(Loc.S("AgentNoModel"));

        if (!session.Info.Started)
            session.Title = session.Info.Title = text.Length <= 60 ? text : text[..57] + "…";
        Append(session, new JsonObject { ["type"] = "llmhub_user", ["text"] = text });
        session.Items.Add(new AgentItem(AgentItemKind.User, text));
        session.Status = AgentStatus.Working;
        session.Info.Started = true;
        session.Info.UpdatedAt = DateTimeOffset.Now;
        SaveInfos();

        try
        {
            // This session's folder, plus the folders of sessions with an open connection.
            await Ioc.OpenClaw.EnsureGatewayAsync(Sessions.Where(s => _acp.TryGetValue(s.Id, out var open) && open.IsAlive).Append(session).Distinct(), CancellationToken.None);
            if (!_acp.TryGetValue(session.Id, out var acp) || !acp.IsAlive)
            {
                acp?.Dispose();
                acp = await AcpConnection.OpenAsync(this, session);
                _acp[session.Id] = acp;
            }
            var stop = await acp.PromptAsync(text);
            Log.Info($"OpenClaw turn in {session.Id} ended: {stop}");
            if (stop == "max_tokens")
                Fail(session, Loc.S("AgentMaxTokens"));
        }
        catch (Exception ex)
        {
            Log.Error("OpenClaw turn failed", ex);
            Fail(session, ex.Message);
        }
        finally
        {
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

    /// <summary>Records a translated event and shows it, exactly like a line from Claude Code.</summary>
    private void Emit(AgentSession session, JsonObject e)
    {
        AppendLine(session, e.ToJsonString());
        UiThread.Run(() => Apply(session, e));
    }

    /// <summary>OpenClaw's permission request ("exec" and friends) through the same Allow / Allow for this session / Deny card.</summary>
    private async Task<string> AskOpenClawAsync(AgentSession session, JsonObject toolCall, JsonArray options, CancellationToken ct)
    {
        var (name, input) = AcpConnection.Translate(toolCall);
        string? Option(params string[] kinds) =>
            options.Select(o => o as JsonObject).FirstOrDefault(o => kinds.Contains(o?["kind"]?.GetValue<string>()))?["optionId"]?.GetValue<string>();
        var allowed = await RequestPermissionAsync(session, name, input, ct);
        if (!allowed)
            return Option("reject_once", "reject_always") ?? "";
        return (session.Info.AllowedTools.Contains(name) ? Option("allow_always", "allow_once") : Option("allow_once", "allow_always")) ?? "";
    }

    /// <summary>One <c>openclaw acp</c> process and its ACP session, speaking newline-delimited JSON-RPC.</summary>
    private sealed class AcpConnection : IDisposable
    {
        private readonly AgentService _owner;
        private readonly AgentSession _session;
        private readonly Process _process;
        private readonly Dictionary<long, TaskCompletionSource<JsonNode?>> _calls = [];
        private readonly object _write = new();
        private long _next;
        private string _acpSession = "";
        private CancellationTokenSource _turn = new();
        // Text streams as chunks; it is written out as one message when a tool starts or the turn ends.
        private readonly System.Text.StringBuilder _text = new();
        // OpenClaw announces each tool twice: a dispatch (raw input {args, id: tool}), then the real call with id
        // "tool_call:<dispatch>:<tool>:<n>". A dispatch is shown only if no real call follows it.
        private readonly Dictionary<string, JsonObject> _dispatches = [];

        private AcpConnection(AgentService owner, AgentSession session, Process process)
        {
            _owner = owner;
            _session = session;
            _process = process;
        }

        public bool IsAlive => !_process.HasExited;

        public static async Task<AcpConnection> OpenAsync(AgentService owner, AgentSession session)
        {
            var process = Process.Start(Ioc.OpenClaw.Acp(session.Folder))!;
            Ioc.Engines.Llama.Track(process);
            var connection = new AcpConnection(owner, session, process);
            _ = Task.Run(connection.ReadLoopAsync);
            _ = process.StandardError.ReadToEndAsync().ContinueWith(t =>
            {
                if (t.Result.Trim().Length > 0)
                    Log.Info($"openclaw acp stderr: {t.Result[^Math.Min(t.Result.Length, 800)..]}");
            });
            await connection.CallAsync("initialize", new JsonObject
            {
                ["protocolVersion"] = 1,
                ["clientCapabilities"] = new JsonObject
                {
                    ["fs"] = new JsonObject { ["readTextFile"] = false, ["writeTextFile"] = false },
                    ["terminal"] = false,
                },
            });
            // The session key names the per-folder agent and this session, so a new process picks up the same history.
            var created = await connection.CallAsync("session/new", new JsonObject
            {
                ["cwd"] = session.Folder,
                ["mcpServers"] = new JsonArray(),
                ["_meta"] = new JsonObject { ["sessionKey"] = $"agent:{OpenClawService.AgentId(session)}:{session.Id}" },
            });
            connection._acpSession = created?["sessionId"]?.GetValue<string>() ?? throw new InvalidOperationException("OpenClaw did not open a session");
            return connection;
        }

        public async Task<string> PromptAsync(string text)
        {
            _turn = new CancellationTokenSource();
            var result = await CallAsync("session/prompt", new JsonObject
            {
                ["sessionId"] = _acpSession,
                ["prompt"] = new JsonArray(new JsonObject { ["type"] = "text", ["text"] = text }),
            });
            FlushText();
            // A dispatch nothing followed: show it as the tool it names, with its arguments.
            foreach (var (id, dispatch) in _dispatches)
            {
                var raw = (JsonObject)dispatch["rawInput"]!;
                dispatch["title"] = raw["id"]?.GetValue<string>() ?? "tool";
                dispatch["rawInput"] = raw["args"]?.DeepClone() ?? new JsonObject();
                EmitToolCall(dispatch, id);
            }
            _dispatches.Clear();
            return result?["stopReason"]?.GetValue<string>() ?? "end_turn";
        }

        public void Cancel()
        {
            _turn.Cancel();
            Notify("session/cancel", new JsonObject { ["sessionId"] = _acpSession });
        }

        private async Task<JsonNode?> CallAsync(string method, JsonObject parameters)
        {
            var id = Interlocked.Increment(ref _next);
            var call = new TaskCompletionSource<JsonNode?>(TaskCreationOptions.RunContinuationsAsynchronously);
            lock (_calls)
                _calls[id] = call;
            Send(new JsonObject { ["jsonrpc"] = "2.0", ["id"] = id, ["method"] = method, ["params"] = parameters });
            return await call.Task;
        }

        private void Notify(string method, JsonObject parameters) =>
            Send(new JsonObject { ["jsonrpc"] = "2.0", ["method"] = method, ["params"] = parameters });

        private void Send(JsonObject message)
        {
            if (_process.HasExited)
                return;
            lock (_write)
            {
                _process.StandardInput.WriteLine(message.ToJsonString());
                _process.StandardInput.Flush();
            }
        }

        private async Task ReadLoopAsync()
        {
            try
            {
                while (await _process.StandardOutput.ReadLineAsync() is { } line)
                {
                    if (line.Length == 0 || line[0] != '{')
                        continue;
                    JsonObject? message;
                    try
                    {
                        message = JsonNode.Parse(line) as JsonObject;
                    }
                    catch (JsonException)
                    {
                        continue;
                    }
                    if (message is null)
                        continue;
                    var method = message["method"]?.GetValue<string>();
                    if (method is null && message["id"] is { } id)
                    {
                        TaskCompletionSource<JsonNode?>? call;
                        lock (_calls)
                            _calls.Remove(id.GetValue<long>(), out call);
                        if (message["error"] is { } error)
                            call?.TrySetException(new InvalidOperationException(error["message"]?.GetValue<string>() ?? error.ToJsonString()));
                        else
                            call?.TrySetResult(message["result"]);
                    }
                    else if (method == "session/update")
                        OnUpdate(message["params"]?["update"] as JsonObject);
                    else if (method == "session/request_permission" && message["id"] is { } requestId)
                        _ = AnswerPermissionAsync(requestId.DeepClone(), message["params"] as JsonObject);
                    else if (message["id"] is { } unknown)
                        Send(new JsonObject { ["jsonrpc"] = "2.0", ["id"] = unknown.DeepClone(), ["error"] = new JsonObject { ["code"] = -32601, ["message"] = "Method not found" } });
                }
            }
            catch (Exception e) when (e is IOException or ObjectDisposedException)
            {
            }
            lock (_calls)
            {
                foreach (var call in _calls.Values)
                    call.TrySetException(new InvalidOperationException(Loc.S("OpenClawStopped")));
                _calls.Clear();
            }
        }

        private async Task AnswerPermissionAsync(JsonNode id, JsonObject? parameters)
        {
            var toolCall = parameters?["toolCall"] as JsonObject ?? new JsonObject();
            var options = parameters?["options"] as JsonArray ?? [];
            FlushText();
            string optionId;
            try
            {
                optionId = await _owner.AskOpenClawAsync(_session, toolCall, options, _turn.Token);
            }
            catch (OperationCanceledException)
            {
                optionId = "";
            }
            var outcome = optionId.Length > 0 && !_turn.IsCancellationRequested
                ? new JsonObject { ["outcome"] = "selected", ["optionId"] = optionId }
                : new JsonObject { ["outcome"] = "cancelled" };
            Send(new JsonObject { ["jsonrpc"] = "2.0", ["id"] = id, ["result"] = new JsonObject { ["outcome"] = outcome } });
        }

        private void OnUpdate(JsonObject? update)
        {
            switch (update?["sessionUpdate"]?.GetValue<string>())
            {
                case "agent_message_chunk":
                    if (update["content"]?["text"]?.GetValue<string>() is { } chunk)
                        _text.Append(chunk);
                    break;
                case "tool_call":
                {
                    FlushText();
                    var id = update["toolCallId"]?.GetValue<string>() ?? Guid.NewGuid().ToString("N");
                    if (update["rawInput"] is JsonObject dispatch && dispatch["id"] is JsonValue && dispatch.All(p => p.Key is "id" or "args"))
                    {
                        _dispatches[id] = (JsonObject)update.DeepClone();
                        break;
                    }
                    if (id.StartsWith("tool_call:", StringComparison.Ordinal) && id.Split(':') is { Length: > 1 } parts)
                        _dispatches.Remove(parts[1]);
                    EmitToolCall(update, id);
                    break;
                }
                case "tool_call_update":
                {
                    var status = update["status"]?.GetValue<string>();
                    if (status is not ("completed" or "failed"))
                        break;
                    var output = string.Concat((update["content"] as JsonArray ?? []).Select(c => c?["content"]?["text"]?.GetValue<string>() ?? ""));
                    if (output.Length == 0 && update["rawOutput"] is JsonObject raw)
                        output = raw["content"] is JsonValue v ? v.ToString() : raw["content"]?.ToJsonString() ?? "";
                    _owner.Emit(_session, new JsonObject
                    {
                        ["type"] = "user",
                        ["message"] = new JsonObject
                        {
                            ["content"] = new JsonArray(new JsonObject
                            {
                                ["type"] = "tool_result",
                                ["tool_use_id"] = update["toolCallId"]?.GetValue<string>() ?? "",
                                ["content"] = output,
                                ["is_error"] = status == "failed",
                            }),
                        },
                    });
                    break;
                }
            }
        }

        private void EmitToolCall(JsonObject call, string id)
        {
            var (name, input) = Translate(call);
            _owner.Emit(_session, new JsonObject
            {
                ["type"] = "assistant",
                ["message"] = new JsonObject
                {
                    ["content"] = new JsonArray(new JsonObject { ["type"] = "tool_use", ["id"] = id, ["name"] = name, ["input"] = input }),
                },
            });
        }

        private void FlushText()
        {
            var text = _text.ToString().Trim();
            _text.Clear();
            if (text.Length == 0)
                return;
            _owner.Emit(_session, new JsonObject
            {
                ["type"] = "assistant",
                ["message"] = new JsonObject { ["content"] = new JsonArray(new JsonObject { ["type"] = "text", ["text"] = text }) },
            });
        }

        /// <summary>
        /// An ACP tool call as the Claude Code tool it resembles, so it renders as Read, Edit, Write, Run or Search:
        /// OpenClaw's read / write / edit / apply_patch / exec / ls tools by their ACP kind and raw input.
        /// </summary>
        public static (string Name, JsonObject Input) Translate(JsonObject call)
        {
            var raw = call["rawInput"] as JsonObject ?? new JsonObject();
            string? Str(params string[] keys) => keys.Select(k => raw[k] is JsonValue v && v.TryGetValue<string>(out var s) ? s : null).FirstOrDefault(s => s is not null);
            var path = Str("path", "file_path", "filePath", "file") ?? (call["locations"] as JsonArray)?.FirstOrDefault()?["path"]?.GetValue<string>() ?? "";
            var title = call["title"]?.GetValue<string>() ?? "";
            var diff = (call["content"] as JsonArray ?? []).OfType<JsonObject>().FirstOrDefault(c => c["type"]?.GetValue<string>() == "diff");
            // OpenClaw's edit and apply_patch take a list of {oldText, newText} replacements: show them as one multi-edit.
            if (raw["edits"] is JsonArray edits)
            {
                var list = new JsonArray();
                foreach (var e in edits.OfType<JsonObject>())
                    list.Add(new JsonObject
                    {
                        ["old_string"] = e["oldText"]?.GetValue<string>() ?? e["old_string"]?.GetValue<string>() ?? "",
                        ["new_string"] = e["newText"]?.GetValue<string>() ?? e["new_string"]?.GetValue<string>() ?? "",
                    });
                return ("MultiEdit", new JsonObject { ["file_path"] = path, ["edits"] = list });
            }
            var tool = title.Split(':', ' ')[0].ToLowerInvariant();
            if (tool is "ls")
                return ("LS", new JsonObject { ["path"] = path.Length > 0 ? path : "." });
            if (tool is "exec" or "process" && Str("command", "cmd") is { } command)
                return ("Bash", new JsonObject { ["command"] = command });
            switch (call["kind"]?.GetValue<string>())
            {
                case "read":
                    return ("Read", new JsonObject { ["file_path"] = path });
                case "edit":
                    if (Str("content") is { } content && Str("oldText", "old_string", "oldString") is null)
                        return ("Write", new JsonObject { ["file_path"] = path, ["content"] = content });
                    var oldText = Str("oldText", "old_string", "oldString") ?? diff?["oldText"]?.GetValue<string>();
                    var newText = Str("newText", "new_string", "newString") ?? diff?["newText"]?.GetValue<string>();
                    if (oldText is not null || newText is not null)
                        return ("Edit", new JsonObject { ["file_path"] = path.Length > 0 ? path : diff?["path"]?.GetValue<string>() ?? "", ["old_string"] = oldText ?? "", ["new_string"] = newText ?? "" });
                    return ("apply_patch", (JsonObject)raw.DeepClone());
                case "execute":
                    return ("Bash", new JsonObject { ["command"] = Str("command", "cmd") ?? title });
                case "search":
                    return ("Grep", new JsonObject { ["pattern"] = Str("pattern", "query") ?? title, ["path"] = path });
                default:
                    var name = title.Split(':', ' ')[0];
                    return (name.Length > 0 ? name : "tool", (JsonObject)raw.DeepClone());
            }
        }

        public void Dispose()
        {
            if (!_process.HasExited)
                _process.Kill(entireProcessTree: true);
            _process.Dispose();
        }
    }
}
