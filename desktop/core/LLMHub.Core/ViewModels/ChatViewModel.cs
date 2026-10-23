using System.Collections.ObjectModel;
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
    }

    public string Id { get; }

    [ObservableProperty] public partial string Title { get; set; }
    [ObservableProperty] public partial DateTimeOffset UpdatedAt { get; set; }
}

public sealed class ChatGroup(string key, IEnumerable<ChatSummary> items) : List<ChatSummary>(items)
{
    public string Key { get; } = key;
}

public sealed partial class MessageItem : ObservableObject
{
    public MessageItem(string role, string content)
    {
        Role = role;
        Content = content;
        Reasoning = "";
        Status = "";
        Footer = "";
    }

    public string Role { get; }
    public bool IsUser => Role == "user";

    [ObservableProperty] public partial string Content { get; set; }
    [ObservableProperty] public partial string Reasoning { get; set; }
    [ObservableProperty] public partial string Status { get; set; }
    /// <summary>Model and generation speed, shown under a finished reply.</summary>
    [ObservableProperty] public partial string Footer { get; set; }
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
        foreach (var record in _store.ListChats())
            _all.Add(new ChatSummary(record));
        RebuildGroups();

        Models.CollectionChanged += (_, _) => UiThread.Post(EnsureModelSelected);
        EnsureModelSelected();
    }

    public ObservableCollection<ChatGroup> Groups { get; } = [];
    public ObservableCollection<MessageItem> Messages { get; } = [];
    public ObservableCollection<InstalledModel> Models => Ioc.Models.Installed;

    [ObservableProperty] public partial ChatSummary? SelectedChat { get; set; }
    [ObservableProperty] public partial string Draft { get; set; }
    [ObservableProperty] public partial string SearchText { get; set; }
    [ObservableProperty] public partial string CurrentTitle { get; set; }
    [ObservableProperty] public partial bool IsGenerating { get; set; }
    [ObservableProperty] public partial InstalledModel? SelectedModel { get; set; }

    public bool HasMessages => Messages.Count > 0;

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
        _current = value;
        if (IsGenerating)
            Stop();
        Messages.Clear();
        CurrentTitle = value.Title;
        foreach (var m in _store.GetMessages(value.Id))
            Messages.Add(new MessageItem(m.Role, m.Content) { Reasoning = m.Reasoning ?? "" });
        OnPropertyChanged(nameof(HasMessages));
    }

    [RelayCommand]
    private void NewChat()
    {
        if (IsGenerating)
            Stop();
        _current = null;
        SelectedChat = null;
        Messages.Clear();
        CurrentTitle = "";
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

    [RelayCommand]
    private void Stop() => _generation?.Cancel();

    [RelayCommand]
    private async Task SendAsync()
    {
        var text = Draft.Trim();
        if (text.Length == 0 || IsGenerating)
            return;
        var model = SelectedModel;
        if (model is null)
            return;

        Draft = "";
        var chat = _current;
        if (chat is null)
        {
            var record = _store.CreateChat(MakeTitle(text), model.Id);
            chat = new ChatSummary(record);
            _all.Insert(0, chat);
            _current = chat;
            _suppressLoad = true;
            SelectedChat = chat;
            _suppressLoad = false;
            CurrentTitle = chat.Title;
        }

        var user = new MessageItem("user", text);
        Messages.Add(user);
        _store.AddMessage(new MessageRecord(Guid.NewGuid().ToString("N"), chat.Id, "user", text, null, null, DateTimeOffset.Now));

        var reply = new MessageItem("assistant", "");
        Messages.Add(reply);
        OnPropertyChanged(nameof(HasMessages));

        IsGenerating = true;
        _generation = new CancellationTokenSource();
        var ct = _generation.Token;
        try
        {
            if (!Ioc.Engines.IsReady(model))
            {
                reply.Status = Loc.F("LoadingModel", model.Name);
                await Ioc.Engines.LoadAsync(model, ct);
                // The first load learns whether the file has an MTP drafter, which decides if the toggle shows.
                OnPropertyChanged(nameof(MtpVisible));
                OnPropertyChanged(nameof(MtpEnabled));
            }
            reply.Status = "";

            var history = Messages.Where(m => m != reply && m.Content.Length > 0)
                .Select(m => new ChatTurn(m.Role, m.Content)).ToList();
            var splitter = new ThinkSplitter();
            await foreach (var delta in Ioc.Engines.StreamAsync(model, history, ct))
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
            IsGenerating = false;
            _generation.Dispose();
            _generation = null;
            if (reply.Content.Length > 0 || reply.Reasoning.Length > 0)
            {
                _store.AddMessage(new MessageRecord(Guid.NewGuid().ToString("N"), chat.Id, "assistant", reply.Content,
                    reply.Reasoning.Length > 0 ? reply.Reasoning : null, model.Id, DateTimeOffset.Now));
            }
            chat.UpdatedAt = DateTimeOffset.Now;
            _all.Remove(chat);
            _all.Insert(0, chat);
            RebuildGroups();
        }
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
            foreach (var group in visible.OrderByDescending(c => c.UpdatedAt).GroupBy(Bucket))
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
