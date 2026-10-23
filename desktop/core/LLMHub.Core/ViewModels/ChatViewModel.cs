using System.Collections.ObjectModel;
using System.Text;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using LLMHub.Services;

namespace LLMHub.ViewModels;

public sealed partial class ChatSummary : ObservableObject
{
    public ChatSummary(ChatRecord record)
    {
        Id = record.Id;
        Title = record.Title;
        UpdatedAt = record.UpdatedAt;
        Starred = record.Starred;
    }

    public string Id { get; }

    [ObservableProperty] public partial string Title { get; set; }
    [ObservableProperty] public partial DateTimeOffset UpdatedAt { get; set; }
    [ObservableProperty] public partial bool Starred { get; set; }
}

public sealed class ChatGroup(string key, IEnumerable<ChatSummary> items) : List<ChatSummary>(items)
{
    public string Key { get; } = key;
}

public sealed partial class MessageItem : ObservableObject
{
    public MessageItem(string role, string content, string? id = null)
    {
        Id = id ?? Guid.NewGuid().ToString("N");
        Role = role;
        Content = content;
        Reasoning = "";
        Status = "";
        Footer = "";
        Activity = "";
        Attachments.CollectionChanged += (_, _) => OnPropertyChanged(nameof(HasAttachments));
        Sources.CollectionChanged += (_, _) => OnPropertyChanged(nameof(HasSources));
    }

    public string Id { get; }
    public string Role { get; }
    public bool IsUser => Role == "user";

    [ObservableProperty] public partial string Content { get; set; }
    [ObservableProperty] public partial string Reasoning { get; set; }
    [ObservableProperty] public partial string Status { get; set; }
    /// <summary>Model and generation speed, shown under a finished reply.</summary>
    [ObservableProperty] public partial string Footer { get; set; }
    /// <summary>"Searched the web · 5 results" above a reply that used search.</summary>
    [ObservableProperty] public partial string Activity { get; set; }
    /// <summary>True once the reply is complete, which is when Copy / Read aloud / Regenerate show.</summary>
    [ObservableProperty] public partial bool IsDone { get; set; }
    /// <summary>Whether the reasoning under the "Thought for…" pill is expanded.</summary>
    [ObservableProperty] public partial bool ReasoningOpen { get; set; }

    public ObservableCollection<Attachment> Attachments { get; } = [];
    public ObservableCollection<WebSource> Sources { get; } = [];
    public bool HasAttachments => Attachments.Count > 0;
    public bool HasSources => Sources.Count > 0;
    /// <summary>Copy / Read aloud / Regenerate show under a finished reply that has text.</summary>
    public bool ShowActions => IsDone && Content.Length > 0;

    partial void OnIsDoneChanged(bool value) => OnPropertyChanged(nameof(ShowActions));
    partial void OnContentChanged(string value)
    {
        if (IsDone)
            OnPropertyChanged(nameof(ShowActions));
    }
}

public sealed partial class ChatViewModel : ObservableObject
{
    private readonly ChatStore _store = Ioc.Chats;
    private readonly List<ChatSummary> _all = [];
    private CancellationTokenSource? _generation;
    private bool _suppressLoad; // Set while selecting a chat programmatically, so its messages are not reloaded.
    private bool _rebuilding;   // Set while the grouped list is rebuilt; the list briefly reports "no selection" then.
    /// <summary>
    /// The open conversation. Kept apart from <see cref="SelectedChat"/>, which mirrors the sidebar list: a list can drop
    /// its selection when it is rebuilt or filtered (WinUI's grouped ListView does), and that must never detach the open
    /// chat, or the next message would start a new one.
    /// </summary>
    private ChatSummary? _current;

    public ChatViewModel()
    {
        Draft = "";
        SearchText = "";
        CurrentTitle = "";
        AttachError = "";
        ContextText = "";
        WebSearchEnabled = AppSettings.Current.WebSearch;
        ThinkingEnabled = AppSettings.Current.Thinking;
        foreach (var record in _store.ListChats())
            _all.Add(new ChatSummary(record));
        RebuildGroups();

        PendingAttachments.CollectionChanged += (_, _) => OnPropertyChanged(nameof(HasPendingAttachments));
        Models.CollectionChanged += (_, _) => UiThread.Post(EnsureModelSelected);
        EnsureModelSelected();
    }

    public ObservableCollection<ChatGroup> Groups { get; } = [];
    public ObservableCollection<MessageItem> Messages { get; } = [];
    public ObservableCollection<InstalledModel> Models => Ioc.Models.Installed;
    /// <summary>Files attached in the composer, sent with the next message.</summary>
    public ObservableCollection<Attachment> PendingAttachments { get; } = [];

    [ObservableProperty] public partial ChatSummary? SelectedChat { get; set; }
    [ObservableProperty] public partial string Draft { get; set; }
    [ObservableProperty] public partial string SearchText { get; set; }
    [ObservableProperty] public partial string CurrentTitle { get; set; }
    [ObservableProperty] public partial bool IsGenerating { get; set; }
    [ObservableProperty] public partial InstalledModel? SelectedModel { get; set; }
    [ObservableProperty] public partial bool WebSearchEnabled { get; set; }
    [ObservableProperty] public partial bool ThinkingEnabled { get; set; }
    [ObservableProperty] public partial string AttachError { get; set; }
    /// <summary>"11k of 32k context" for the open chat, or "" before the first reply.</summary>
    [ObservableProperty] public partial string ContextText { get; set; }
    [ObservableProperty] public partial bool IsCurrentStarred { get; set; }

    public bool HasMessages => Messages.Count > 0;
    public bool HasPendingAttachments => PendingAttachments.Count > 0;
    public bool HasOpenChat => _current is not null;

    /// <summary>The MTP toggle shows only for LiteRT-LM models with a drafter; changing it reloads the model on next send.</summary>
    public bool MtpVisible => SelectedModel?.SupportsMtp == true;

    public bool MtpEnabled
    {
        get => SelectedModel?.MtpEnabled == true;
        set
        {
            if (SelectedModel is not { SupportsMtp: true } model || value == model.MtpEnabled)
                return;
            if (value)
                AppSettings.Current.MtpOff.Remove(model.Id);
            else
                AppSettings.Current.MtpOff.Add(model.Id);
            AppSettings.Save();
            OnPropertyChanged();
        }
    }
    public bool HasModels => Models.Count > 0;

    partial void OnSearchTextChanged(string value) => RebuildGroups();

    partial void OnWebSearchEnabledChanged(bool value)
    {
        AppSettings.Current.WebSearch = value;
        AppSettings.Save();
    }

    partial void OnThinkingEnabledChanged(bool value)
    {
        AppSettings.Current.Thinking = value;
        AppSettings.Save();
    }

    partial void OnSelectedModelChanged(InstalledModel? value)
    {
        if (value is null)
            return;
        AppSettings.Current.LastChatModelId = value.Id;
        AppSettings.Save();
        OnPropertyChanged(nameof(MtpVisible));
        OnPropertyChanged(nameof(MtpEnabled));
    }

    partial void OnSelectedChatChanged(ChatSummary? value)
    {
        // Only picking a different chat opens it; losing the list selection leaves the open chat alone (New chat closes it).
        if (_suppressLoad || _rebuilding || value is null || value == _current)
            return;
        if (IsGenerating)
            Stop();
        SetCurrent(value);
        Messages.Clear();
        foreach (var m in _store.GetMessages(value.Id))
        {
            var item = new MessageItem(m.Role, m.Content, m.Id) { Reasoning = m.Reasoning ?? "", IsDone = true };
            foreach (var a in m.Attachments)
                item.Attachments.Add(a);
            foreach (var s in m.Sources)
                item.Sources.Add(s);
            if (m.Sources.Count > 0)
                item.Activity = Loc.F("SearchedWeb", m.Sources.Count.ToString());
            if (!item.IsUser && m.ModelId is { } id)
                item.Footer = Loc.F("OnThisComputer", Models.FirstOrDefault(x => x.Id == id)?.Name ?? id);
            Messages.Add(item);
        }
        ContextText = "";
        OnPropertyChanged(nameof(HasMessages));
    }

    private void SetCurrent(ChatSummary? chat)
    {
        _current = chat;
        CurrentTitle = chat?.Title ?? "";
        IsCurrentStarred = chat?.Starred == true;
        OnPropertyChanged(nameof(HasOpenChat));
    }

    [RelayCommand]
    private void NewChat()
    {
        if (IsGenerating)
            Stop();
        SetCurrent(null);
        SelectedChat = null;
        Messages.Clear();
        PendingAttachments.Clear();
        ContextText = "";
        AttachError = "";
        OnPropertyChanged(nameof(HasMessages));
    }

    [RelayCommand]
    private void DeleteChat(ChatSummary? chat)
    {
        if (chat is null)
            return;
        _store.DeleteChat(chat.Id);
        _all.Remove(chat);
        if (_current == chat)
            NewChat();
        RebuildGroups();
    }

    /// <summary>Stars or unstars a chat (the open one when none is given); starred chats get their own sidebar group.</summary>
    [RelayCommand]
    private void ToggleStar(ChatSummary? chat)
    {
        chat ??= _current;
        if (chat is null)
            return;
        chat.Starred = !chat.Starred;
        _store.SetStarred(chat.Id, chat.Starred);
        if (chat == _current)
            IsCurrentStarred = chat.Starred;
        RebuildGroups();
    }

    /// <summary>Adds files to the next message. Unsupported files are skipped with a message in <see cref="AttachError"/>.</summary>
    public void AttachFiles(IEnumerable<string> paths)
    {
        AttachError = "";
        foreach (var path in paths)
        {
            try
            {
                PendingAttachments.Add(Attachments.Import(path));
            }
            catch (Exception e)
            {
                Log.Error($"Attach {path} failed", e);
                AttachError = e.Message;
            }
        }
    }

    [RelayCommand]
    private void RemoveAttachment(Attachment? attachment)
    {
        if (attachment is not null)
            PendingAttachments.Remove(attachment);
    }

    [RelayCommand]
    private void Stop() => _generation?.Cancel();

    [RelayCommand]
    private async Task SendAsync()
    {
        var text = Draft.Trim();
        if ((text.Length == 0 && PendingAttachments.Count == 0) || IsGenerating)
            return;
        var model = SelectedModel;
        if (model is null)
            return;
        if (text.Length == 0)
            text = Loc.S("AttachDefaultPrompt");

        Draft = "";
        var chat = _current;
        if (chat is null)
        {
            var record = _store.CreateChat(MakeTitle(text), model.Id);
            chat = new ChatSummary(record);
            _all.Insert(0, chat);
            SetCurrent(chat);
            _suppressLoad = true;
            SelectedChat = chat;
            _suppressLoad = false;
        }

        var user = new MessageItem("user", text) { IsDone = true };
        foreach (var a in PendingAttachments)
            user.Attachments.Add(a);
        PendingAttachments.Clear();
        Messages.Add(user);
        _store.AddMessage(new MessageRecord(user.Id, chat.Id, "user", text, null, null, DateTimeOffset.Now) { Attachments = [.. user.Attachments] });

        await GenerateAsync(chat, model, text);
    }

    /// <summary>Replaces the last reply with a new one.</summary>
    [RelayCommand]
    private async Task RegenerateAsync(MessageItem? reply)
    {
        if (IsGenerating || _current is not { } chat || SelectedModel is not { } model || Messages.Count == 0)
            return;
        reply ??= Messages[^1];
        if (reply.IsUser || reply != Messages[^1])
            return;
        Messages.Remove(reply);
        _store.DeleteMessage(reply.Id);
        var question = Messages.LastOrDefault(m => m.IsUser)?.Content ?? "";
        await GenerateAsync(chat, model, question);
    }

    private async Task GenerateAsync(ChatSummary chat, InstalledModel model, string question)
    {
        var reply = new MessageItem("assistant", "");
        Messages.Add(reply);
        OnPropertyChanged(nameof(HasMessages));

        IsGenerating = true;
        _generation = new CancellationTokenSource();
        var ct = _generation.Token;
        try
        {
            var history = Messages.Where(m => m != reply && (m.Content.Length > 0 || m.HasAttachments))
                .Select(m => new ChatTurn(m.Role, m.Content, m.Attachments.Count > 0 ? [.. m.Attachments] : null)).ToList();

            if (WebSearchEnabled && question.Length > 0)
            {
                reply.Status = Loc.S("SearchingWeb");
                try
                {
                    var (context, sources) = await WebSearch.BuildContextAsync(question, ct);
                    foreach (var s in sources)
                        reply.Sources.Add(s);
                    reply.Activity = Loc.F("SearchedWeb", sources.Count.ToString());
                    if (context.Length > 0)
                        history.Insert(history.Count - 1, new ChatTurn("system", context));
                }
                catch (Exception e) when (e is not OperationCanceledException)
                {
                    Log.Error("Web search failed", e);
                    reply.Activity = Loc.S("WebSearchFailed");
                }
            }

            if (!Ioc.Engines.IsReady(model))
            {
                reply.Status = Loc.F("LoadingModel", model.Name);
                await Ioc.Engines.LoadAsync(model, ct);
                // The first load learns whether the file has an MTP drafter, which decides if the toggle shows.
                OnPropertyChanged(nameof(MtpVisible));
                OnPropertyChanged(nameof(MtpEnabled));
            }
            reply.Status = "";

            var splitter = new ThinkSplitter();
            await foreach (var delta in Ioc.Engines.StreamAsync(model, history, ct, new ChatOptions(ThinkingEnabled)))
            {
                if (delta.Reasoning is { } r)
                    reply.Reasoning += r;
                if (delta.Content is { } c)
                {
                    var (answer, thought) = splitter.Push(c);
                    reply.Reasoning += thought;
                    reply.Content += answer;
                }
                if (delta.TokensPerSecond is { } speed)
                {
                    reply.Footer = Loc.F("ReplyFooter", model.Name, speed.ToString("F1"));
                    Log.Info($"Chat speed {model.Id}: {speed:F1} tok/s on {Ioc.Engines.Backend}");
                }
                if (delta.ContextTokens is { } used)
                    ContextText = Loc.F("ContextMeter", Fmt.Tokens(used), Fmt.Tokens(ContextLimit(model)));
            }
        }
        catch (OperationCanceledException)
        {
            // Stopped by the user; keep what was generated.
        }
        catch (Exception e)
        {
            Log.Error("Chat failed", e);
            reply.Status = Loc.F("ErrorPrefix", e.Message);
        }
        finally
        {
            reply.Content = reply.Content.TrimStart(); // Text after a </think> block usually starts with blank lines.
            reply.IsDone = true;
            IsGenerating = false;
            _generation?.Dispose();
            _generation = null;
            if (reply.Content.Length > 0 || reply.Reasoning.Length > 0)
            {
                _store.AddMessage(new MessageRecord(reply.Id, chat.Id, "assistant", reply.Content,
                    reply.Reasoning.Length > 0 ? reply.Reasoning : null, model.Id, DateTimeOffset.Now) { Sources = [.. reply.Sources] });
            }
            chat.UpdatedAt = DateTimeOffset.Now;
            _all.Remove(chat);
            _all.Insert(0, chat);
            RebuildGroups();
        }
    }

    /// <summary>The context window the engine runs this model with (the smaller of the model's and the setting's).</summary>
    private static int ContextLimit(InstalledModel model) =>
        model.ContextLength > 0 ? Math.Min(model.ContextLength, AppSettings.Current.ContextLength) : AppSettings.Current.ContextLength;

    /// <summary>The open chat as Markdown, for Share.</summary>
    public string ExportMarkdown()
    {
        var sb = new StringBuilder();
        sb.Append("# ").AppendLine(CurrentTitle.Length > 0 ? CurrentTitle : "LLM Hub").AppendLine();
        foreach (var m in Messages)
        {
            sb.Append("**").Append(m.IsUser ? Loc.S("ExportYou") : SelectedModel?.Name ?? "LLM Hub").AppendLine("**").AppendLine();
            foreach (var a in m.Attachments)
                sb.Append("> ").AppendLine(a.Name);
            if (m.Attachments.Count > 0)
                sb.AppendLine();
            sb.AppendLine(m.Content).AppendLine();
            foreach (var s in m.Sources)
                sb.Append('[').Append(s.Number).Append("]: ").Append(s.Url).Append(" \"").Append(s.Title.Replace("\"", "'")).AppendLine("\"");
            if (m.Sources.Count > 0)
                sb.AppendLine();
        }
        return sb.ToString();
    }

    /// <summary>File name for Share: the chat title without characters file systems reject.</summary>
    public string ExportFileName()
    {
        var name = string.Concat((CurrentTitle.Length > 0 ? CurrentTitle : "chat").Split(Path.GetInvalidFileNameChars())).Trim().TrimEnd('…', '.');
        return (name.Length > 0 ? name : "chat") + ".md";
    }

    private void EnsureModelSelected()
    {
        OnPropertyChanged(nameof(HasModels));
        if (SelectedModel is not null && Models.Contains(SelectedModel))
            return;
        SelectedModel = Models.FirstOrDefault(m => m.Id == AppSettings.Current.LastChatModelId)
                        ?? Models.FirstOrDefault(m => m.CatalogId == Ioc.Catalog.DefaultFor("chat"))
                        ?? Models.FirstOrDefault();
    }

    private void RebuildGroups()
    {
        var query = SearchText.Trim();
        var visible = string.IsNullOrEmpty(query)
            ? _all
            : _all.Where(c => c.Title.Contains(query, StringComparison.CurrentCultureIgnoreCase)).ToList();

        var today = DateTimeOffset.Now.Date;
        string Bucket(ChatSummary c)
        {
            var day = c.UpdatedAt.LocalDateTime.Date;
            if (day == today) return Loc.S("GroupToday");
            if (day == today.AddDays(-1)) return Loc.S("GroupYesterday");
            if (day > today.AddDays(-7)) return Loc.S("GroupWeek");
            return Loc.S("GroupOlder");
        }

        _rebuilding = true;
        try
        {
            Groups.Clear();
            var starred = visible.Where(c => c.Starred).OrderByDescending(c => c.UpdatedAt).ToList();
            if (starred.Count > 0)
                Groups.Add(new ChatGroup(Loc.S("GroupStarred"), starred));
            foreach (var group in visible.Where(c => !c.Starred).OrderByDescending(c => c.UpdatedAt).GroupBy(Bucket))
                Groups.Add(new ChatGroup(group.Key, group));
            if (SelectedChat != _current)
                SelectedChat = _current;
        }
        finally
        {
            _rebuilding = false;
        }
    }

    private static string MakeTitle(string text)
    {
        var line = text.Split('\n', 2)[0].Trim();
        return line.Length <= 48 ? line : line[..45].TrimEnd() + "…";
    }
}
