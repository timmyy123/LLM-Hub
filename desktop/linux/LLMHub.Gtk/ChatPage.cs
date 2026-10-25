using System.Collections.Specialized;
using System.ComponentModel;
using LLMHub.Services;
using LLMHub.ViewModels;

namespace LLMHub.Linux;

/// <summary>Messages, empty state and composer, bound to the shared ChatViewModel (layout from the Claude Design canvas).</summary>
internal sealed class ChatPage
{
    private readonly MainWindow _main;
    private readonly ChatViewModel _vm = Ioc.Chat;
    private readonly Gtk.Stack _body = Gtk.Stack.New();
    private readonly Gtk.Box _messages = Gtk.Box.New(Gtk.Orientation.Vertical, 28);
    private readonly Gtk.ScrolledWindow _scroll = Gtk.ScrolledWindow.New();
    private readonly Gtk.TextView _input = Gtk.TextView.New();
    private readonly Gtk.DropDown _models = Gtk.DropDown.NewFromStrings([]);
    private readonly Gtk.ToggleButton _mtp = Gtk.ToggleButton.NewWithLabel(Loc.S("Mtp"));
    private readonly Gtk.ToggleButton _webSearch = Gtk.ToggleButton.New();
    private readonly Gtk.ToggleButton _thinking = Gtk.ToggleButton.NewWithLabel(Loc.S("ExtendedThinking"));
    private readonly Gtk.Label _context = Ui.Label("", "caption", "dim-label");
    private readonly Gtk.Box _pending = Gtk.Box.New(Gtk.Orientation.Horizontal, 8);
    private readonly Gtk.Label _attachError = Ui.Label("", "caption", "error");
    private readonly Gtk.Button _send;
    private readonly Gtk.Button _stop;
    private readonly Gtk.Button _dictate;
    private readonly Adw.StatusPage _empty = Adw.StatusPage.New();
    private readonly Gtk.Button _getModel;
    private bool _syncingModels;

    public Gtk.Widget Root { get; }

    public ChatPage(MainWindow main)
    {
        _main = main;

        _empty.SetTitle(Loc.S("EmptyChatTitle"));
        _empty.Named("EmptyTitle");
        _getModel = Ui.Button(Loc.S("GetModel"), () => _main.ShowSettings(), "pill", "suggested-action").Named("GetModelButton");
        _getModel.SetHalign(Gtk.Align.Center);
        _empty.SetChild(_getModel);

        var clamp = Adw.Clamp.New();
        clamp.SetMaximumSize(760);
        clamp.SetChild(_messages.Margins(32, 24, 16, 24).Named("MessageList"));
        _scroll.SetChild(clamp);
        _scroll.SetPolicy(Gtk.PolicyType.Never, Gtk.PolicyType.Automatic);
        _body.AddNamed(_empty, "empty");
        _body.AddNamed(_scroll, "messages");
        _body.SetVexpand(true);

        _input.SetWrapMode(Gtk.WrapMode.WordChar);
        _input.SetAcceptsTab(false);
        _input.Named("Composer");
        _input.SetTooltipText(Loc.S("ComposerPlaceholder"));
        var keys = Gtk.EventControllerKey.New();
        keys.OnKeyPressed += (_, e) =>
        {
            // Enter sends, Shift+Enter adds a line.
            if (e.Keyval is not (0xff0d or 0xff8d) || (e.State & Gdk.ModifierType.ShiftMask) != 0)
                return false;
            Send();
            return true;
        };
        _input.AddController(keys);
        var inputScroll = Gtk.ScrolledWindow.New();
        inputScroll.SetChild(_input);
        inputScroll.SetPolicy(Gtk.PolicyType.Never, Gtk.PolicyType.Automatic);
        inputScroll.SetMinContentHeight(28);
        inputScroll.SetMaxContentHeight(220);
        inputScroll.SetPropagateNaturalHeight(true);

        _models.Named("ModelPicker");
        _models.SetHalign(Gtk.Align.End);
        _models.AddCssClass("flat");
        _models.OnNotify += (_, e) =>
        {
            if (!_syncingModels && e.Pspec.GetName() == "selected" && _models.GetSelected() is var i and < uint.MaxValue && i < _vm.Models.Count)
                _vm.SelectedModel = _vm.Models[(int)i];
        };

        var attach = Ui.IconButton("mail-attachment-symbolic", Loc.S("Attach"), PickFiles, "composer-icon").Named("AttachButton");
        _webSearch.SetChild(Ui.HBox(6, Gtk.Image.NewFromIconName("web-browser-symbolic"), Gtk.Label.New(Loc.S("WebSearch"))));
        _webSearch.AddCssClass("composer-toggle");
        _webSearch.Named("WebSearchToggle");
        _webSearch.SetActive(_vm.WebSearchEnabled);
        _webSearch.OnToggled += (_, _) => _vm.WebSearchEnabled = _webSearch.GetActive();
        _thinking.AddCssClass("composer-toggle");
        _thinking.Named("ThinkingToggle");
        _thinking.SetActive(_vm.ThinkingEnabled);
        _thinking.OnToggled += (_, _) => _vm.ThinkingEnabled = _thinking.GetActive();
        _mtp.AddCssClass("composer-toggle");
        _mtp.SetTooltipText(Loc.S("MtpHelp"));
        _mtp.Named("MtpToggle");
        _mtp.OnToggled += (_, _) => _vm.MtpEnabled = _mtp.GetActive();
        _context.Named("ContextMeter");
        _context.SetValign(Gtk.Align.Center);
        _context.SetWrap(false);
        _dictate = Ui.IconButton("audio-input-microphone-symbolic", Loc.S("Dictate"), Dictate, "composer-icon").Named("DictateButton");
        _send = Ui.IconButton("go-up-symbolic", Loc.S("Send"), Send, "suggested-action", "composer-send").Named("SendButton");
        _stop = Ui.IconButton("media-playback-stop-symbolic", Loc.S("Stop"), () => _vm.StopCommand.Execute(null), "composer-send").Named("StopButton");
        var spacer = Gtk.Box.New(Gtk.Orientation.Horizontal, 0).Expand(true, false);
        var actions = Ui.HBox(6, attach, _webSearch, _thinking, _mtp, spacer, _context, _models, _dictate, _send, _stop);
        _pending.Named("PendingAttachments");
        _attachError.Named("AttachError");
        var composer = Ui.VBox(10, _pending, _attachError, inputScroll, actions);
        composer.AddCssClass("composer");
        var note = Ui.Label(Loc.S("PrivacyNote"), "caption", "dim-label");
        note.SetXalign(0.5f);
        var composerBox = Ui.VBox(8, composer, note);
        var composerClamp = Adw.Clamp.New();
        composerClamp.SetMaximumSize(760);
        composerClamp.SetChild(composerBox);

        Root = Ui.VBox(0, _body, composerClamp.Margins(8, 24, 18, 24));

        // Files dropped anywhere on the chat are attached like the Attach button's. Capture phase, so a file dropped on
        // the composer is attached instead of the text view inserting its path.
        var drop = Gtk.DropTarget.New(Gdk.FileList.GetGType(), Gdk.DragAction.Copy);
        drop.SetPropagationPhase(Gtk.PropagationPhase.Capture);
        drop.OnDrop += (_, e) => AttachDropped(e.Value);
        Root.AddController(drop);

        _vm.Messages.CollectionChanged += OnMessagesChanged;
        _vm.PropertyChanged += OnVmChanged;
        _vm.Models.CollectionChanged += (_, _) => SyncModels(rebuild: true);
        _vm.PendingAttachments.CollectionChanged += (_, _) => RebuildPending();
        Ioc.Speech.PropertyChanged += (_, _) => GLib.Functions.IdleAdd(GLib.Constants.PRIORITY_DEFAULT, () =>
        {
            UpdateDictation();
            return false;
        });
        SyncModels(rebuild: true);
        RebuildMessages();
        RebuildPending();
        UpdateState();
        _mtp.SetVisible(_vm.MtpVisible);
        _mtp.SetActive(_vm.MtpEnabled);
    }

    public string InputText
    {
        get => _input.GetBuffer().Text ?? "";
        set => _input.GetBuffer().SetText(value, -1);
    }

    public void Send()
    {
        _vm.Draft = InputText;
        if (_vm.SendCommand.CanExecute(null))
        {
            InputText = "";
            _vm.SendCommand.Execute(null);
        }
    }

    private async void PickFiles()
    {
        try
        {
            var dialog = Gtk.FileDialog.New();
            dialog.SetTitle(Loc.S("Attach"));
            if (await dialog.OpenMultipleAsync(_main.Window) is { } list)
            {
                var paths = new List<string>();
                for (uint i = 0; i < list.GetNItems(); i++)
                    if (list.GetObject(i) is Gio.File file && file.GetPath() is { } path)
                        paths.Add(path);
                _vm.AttachFiles(paths);
            }
        }
        catch (GLib.GException)
        {
            // Dismissed.
        }
    }

    /// <summary>Attaches the files in a dropped GdkFileList; false when it held no local files.</summary>
    internal bool AttachDropped(GObject.Value value)
    {
        var paths = Native.FilePaths(value.GetBoxed());
        if (paths.Count == 0)
            return false;
        _vm.AttachFiles(paths);
        return true;
    }

    private async void Dictate()
    {
        var speech = Ioc.Speech;
        try
        {
            if (!speech.IsRecording)
            {
                speech.StartDictation();
                return;
            }
            var text = await speech.StopDictationAsync();
            if (text.Length > 0)
                InputText = InputText.Length == 0 ? text : InputText.TrimEnd() + " " + text;
        }
        catch (Exception e)
        {
            Log.Error("Dictation failed", e);
            _vm.AttachError = Loc.F("ErrorPrefix", e.Message);
        }
    }

    private void UpdateDictation()
    {
        var speech = Ioc.Speech;
        _dictate.SetIconName(speech.IsRecording ? "media-playback-stop-symbolic" : "audio-input-microphone-symbolic");
        _dictate.SetTooltipText(speech.IsRecording ? Loc.S("Listening") : speech.IsTranscribing ? Loc.S("Transcribing") : Loc.S("Dictate"));
        _dictate.SetSensitive(!speech.IsTranscribing);
        if (speech.Status.Length > 0)
            _vm.AttachError = speech.Status;
    }

    private void OnVmChanged(object? sender, PropertyChangedEventArgs e)
    {
        if (e.PropertyName is nameof(ChatViewModel.IsGenerating) or nameof(ChatViewModel.HasMessages) or nameof(ChatViewModel.HasModels))
            UpdateState();
        // The selection can change from inside the drop-down's own "selected" notification; touching the drop-down
        // there leaves GTK with a stale selection object (segfault on the next pick), so sync once it has returned.
        if (e.PropertyName is nameof(ChatViewModel.SelectedModel))
            GLib.Functions.IdleAdd(GLib.Constants.PRIORITY_DEFAULT, () =>
            {
                SyncModels(rebuild: false);
                return false;
            });
        if (e.PropertyName is nameof(ChatViewModel.MtpVisible) or nameof(ChatViewModel.MtpEnabled))
        {
            _mtp.SetVisible(_vm.MtpVisible);
            _mtp.SetActive(_vm.MtpEnabled);
        }
        if (e.PropertyName is nameof(ChatViewModel.ContextText))
            _context.SetText(_vm.ContextText);
        if (e.PropertyName is nameof(ChatViewModel.AttachError))
        {
            _attachError.SetText(_vm.AttachError);
            _attachError.SetVisible(_vm.AttachError.Length > 0);
        }
        if (e.PropertyName is nameof(ChatViewModel.CurrentTitle) or nameof(ChatViewModel.IsCurrentStarred) or nameof(ChatViewModel.HasOpenChat))
            _main.UpdateChatHeader();
    }

    private void UpdateState()
    {
        _body.SetVisibleChildName(_vm.HasMessages ? "messages" : "empty");
        _empty.SetDescription(_vm.HasModels ? "" : Loc.S("EmptyNoModel"));
        _getModel.SetVisible(!_vm.HasModels);
        _send.SetVisible(!_vm.IsGenerating);
        _stop.SetVisible(_vm.IsGenerating);
        _attachError.SetVisible(_vm.AttachError.Length > 0);
        _main.UpdateChatHeader();
    }

    private void SyncModels(bool rebuild)
    {
        _syncingModels = true;
        if (rebuild)
            _models.SetModel(Gtk.StringList.New(_vm.Models.Select(m => m.DisplayName).ToArray()));
        var index = _vm.SelectedModel is { } s ? _vm.Models.IndexOf(s) : -1;
        var selected = index >= 0 ? (uint)index : uint.MaxValue;
        if (_models.GetSelected() != selected)
            _models.SetSelected(selected);
        _models.SetVisible(_vm.HasModels);
        _syncingModels = false;
    }

    private void RebuildPending()
    {
        _pending.Clear();
        foreach (var a in _vm.PendingAttachments)
        {
            var remove = Ui.IconButton("window-close-symbolic", Loc.S("Delete"), () => _vm.RemoveAttachmentCommand.Execute(a), "flat", "circular").Named("RemoveAttachmentButton");
            remove.SetValign(Gtk.Align.Start);
            _pending.Append(Ui.HBox(0, Chip(a), remove));
        }
        _pending.SetVisible(_vm.PendingAttachments.Count > 0);
    }

    /// <summary>File chip: coloured badge (PDF, PNG…), name and detail.</summary>
    private static Gtk.Widget Chip(Attachment a)
    {
        var badge = Ui.Label(a.Badge, "badge", a.IsImage ? "badge-image" : "badge-doc");
        badge.SetXalign(0.5f);
        badge.SetValign(Gtk.Align.Center);
        var name = Ui.Label(a.Name, "caption-heading");
        name.SetWrap(false);
        name.SetEllipsize(Pango.EllipsizeMode.Middle);
        name.SetMaxWidthChars(24);
        var chip = Ui.HBox(10, badge, Ui.VBox(0, name, Ui.Label(a.Detail, "caption", "dim-label")));
        chip.AddCssClass("file-chip");
        chip.Named("AttachmentChip");
        chip.SetTooltipText(a.Name);
        return chip;
    }

    private void OnMessagesChanged(object? sender, NotifyCollectionChangedEventArgs e)
    {
        if (e.Action == NotifyCollectionChangedAction.Add && e.NewItems is not null)
        {
            foreach (MessageItem item in e.NewItems)
                _messages.Append(MessageRow(item));
            ScrollToEnd();
        }
        else
            RebuildMessages();
        UpdateState();
    }

    private void RebuildMessages()
    {
        _messages.Clear();
        foreach (var item in _vm.Messages)
            _messages.Append(MessageRow(item));
        ScrollToEnd();
    }

    private Gtk.Widget MessageRow(MessageItem item)
    {
        if (item.IsUser)
        {
            var text = Ui.Label(item.Content, "message", "user-bubble").Named("UserMessage");
            text.SetSelectable(true);
            text.SetHalign(Gtk.Align.End);
            var chips = Ui.HBox(8);
            chips.SetHalign(Gtk.Align.End);
            foreach (var a in item.Attachments)
                chips.Append(Chip(a));
            chips.SetVisible(item.HasAttachments);
            var column = Ui.VBox(8, chips, text);
            column.SetHalign(Gtk.Align.End);
            column.SetMarginStart(120);
            return column;
        }

        var activityText = Ui.Label("", "caption");
        activityText.SetWrap(false);
        var activity = Ui.HBox(8, Gtk.Image.NewFromIconName("web-browser-symbolic"), activityText);
        activity.AddCssClass("activity-pill");
        activity.Named("ActivityPill");
        var thinkingText = Ui.Label(item.Reasoning, "dim-label", "reasoning");
        thinkingText.SetSelectable(true);
        var reasoning = Gtk.Revealer.New();
        reasoning.SetChild(thinkingText);
        var thinkingPill = Gtk.ToggleButton.NewWithLabel(Loc.S("Thinking"));
        thinkingPill.AddCssClass("activity-pill");
        thinkingPill.Named("ThinkingExpander");
        thinkingPill.OnToggled += (_, _) => reasoning.SetRevealChild(thinkingPill.GetActive());
        var pills = Ui.HBox(8, activity, thinkingPill);

        var spinner = Gtk.Spinner.New();
        var status = Ui.Label(item.Status, "dim-label").Named("MessageStatus");
        var statusRow = Ui.HBox(10, spinner, status);
        var content = new MarkdownView("AssistantMessage");
        var sources = Ui.VBox(6);
        sources.Named("Sources");

        var copy = Ui.IconButton("edit-copy-symbolic", Loc.S("Copy"), () => Ui.Copy(_messages, item.Content), "flat").Named("CopyMessageButton");
        var read = Ui.IconButton("audio-speakers-symbolic", Loc.S("ReadAloud"), () => _ = Ioc.Speech.ToggleSpeakAsync(item.Id, item.Content), "flat").Named("ReadAloudButton");
        var regenerate = Ui.IconButton("view-refresh-symbolic", Loc.S("Regenerate"), () => _vm.RegenerateCommand.Execute(item), "flat").Named("RegenerateButton");
        var footer = Ui.Label(item.Footer, "caption", "dim-label").Named("MessageFooter");
        footer.SetValign(Gtk.Align.Center);
        footer.SetMarginStart(8);
        footer.SetWrap(false);
        var actionRow = Ui.HBox(4, copy, read, regenerate, footer);
        var box = Ui.VBox(14, pills, reasoning, statusRow, content.Widget, sources, actionRow);

        void Update()
        {
            content.SetText(item.Content);
            content.Widget.SetVisible(item.Content.Length > 0);
            activityText.SetText(item.Activity);
            activity.SetVisible(item.Activity.Length > 0);
            thinkingText.SetText(item.Reasoning);
            thinkingPill.SetLabel(Loc.S(item.IsDone ? "ThoughtLabel" : "Thinking"));
            thinkingPill.SetVisible(item.Reasoning.Length > 0);
            pills.SetVisible(item.Activity.Length > 0 || item.Reasoning.Length > 0);
            status.SetText(item.Status);
            statusRow.SetVisible(item.Status.Length > 0);
            spinner.SetSpinning(item.Status.Length > 0);
            footer.SetText(item.Footer);
            actionRow.SetVisible(item.ShowActions);
            if (sources.GetFirstChild() is null && item.HasSources)
            {
                foreach (var s in item.Sources)
                {
                    var link = Gtk.LinkButton.NewWithLabel(s.Url, s.Host);
                    link.AddCssClass("source-chip");
                    link.Named("SourceLink");
                    var title = Ui.Label($"{s.Number}.  {s.Title}", "caption");
                    title.SetWrap(false);
                    title.SetEllipsize(Pango.EllipsizeMode.End);
                    title.SetMaxWidthChars(60);
                    sources.Append(Ui.HBox(8, title, link));
                }
            }
            sources.SetVisible(item.HasSources);
        }
        Update();
        item.PropertyChanged += (_, _) =>
        {
            Update();
            ScrollToEnd();
        };
        item.Sources.CollectionChanged += (_, _) => Update();
        return box;
    }

    private void ScrollToEnd() => GLib.Functions.IdleAdd(GLib.Constants.PRIORITY_LOW, () =>
    {
        var adj = _scroll.GetVadjustment();
        // Follow the stream only while the reader is near the bottom.
        if (adj.GetUpper() - adj.GetPageSize() - adj.GetValue() < 200)
            adj.SetValue(adj.GetUpper());
        return false;
    });
}
