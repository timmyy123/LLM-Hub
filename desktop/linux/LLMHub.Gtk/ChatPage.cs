using System.Collections.Specialized;
using System.ComponentModel;
using LLMHub.Services;
using LLMHub.ViewModels;

namespace LLMHub.Linux;

/// <summary>Messages, empty state and composer, bound to the shared ChatViewModel.</summary>
internal sealed class ChatPage
{
    private readonly MainWindow _main;
    private readonly ChatViewModel _vm = Ioc.Chat;
    private readonly Gtk.Stack _body = Gtk.Stack.New();
    private readonly Gtk.Box _messages = Gtk.Box.New(Gtk.Orientation.Vertical, 24);
    private readonly Gtk.ScrolledWindow _scroll = Gtk.ScrolledWindow.New();
    private readonly Gtk.TextView _input = Gtk.TextView.New();
    private readonly Gtk.DropDown _models = Gtk.DropDown.NewFromStrings([]);
    private readonly Gtk.ToggleButton _mtp = Gtk.ToggleButton.NewWithLabel(Loc.S("Mtp"));
    private readonly Gtk.Button _send;
    private readonly Gtk.Button _stop;
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
        clamp.SetChild(_messages.Margins(28, 24, 12, 24).Named("MessageList"));
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
        _models.OnNotify += (_, e) =>
        {
            if (!_syncingModels && e.Pspec.GetName() == "selected" && _models.GetSelected() is var i and < uint.MaxValue && i < _vm.Models.Count)
                _vm.SelectedModel = _vm.Models[(int)i];
        };
        _send = Ui.IconButton("go-up-symbolic", Loc.S("Send"), Send, "circular", "suggested-action").Named("SendButton");
        _stop = Ui.IconButton("media-playback-stop-symbolic", Loc.S("Stop"), () => _vm.StopCommand.Execute(null), "circular").Named("StopButton");
        var spacer = Gtk.Box.New(Gtk.Orientation.Horizontal, 0).Expand(true, false);
        _mtp.SetTooltipText(Loc.S("MtpHelp"));
        _mtp.Named("MtpToggle");
        _mtp.OnToggled += (_, _) => _vm.MtpEnabled = _mtp.GetActive();
        var actions = Ui.HBox(8, spacer, _mtp, _models, _send, _stop);
        var composer = Ui.VBox(8, inputScroll, actions);
        composer.AddCssClass("composer");
        var composerClamp = Adw.Clamp.New();
        composerClamp.SetMaximumSize(760);
        composerClamp.SetChild(composer);

        Root = Ui.VBox(0, _body, composerClamp.Margins(8, 24, 20, 24));

        _vm.Messages.CollectionChanged += OnMessagesChanged;
        _vm.PropertyChanged += OnVmChanged;
        _vm.Models.CollectionChanged += (_, _) => SyncModels(rebuild: true);
        SyncModels(rebuild: true);
        RebuildMessages();
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
    }

    private void UpdateState()
    {
        _body.SetVisibleChildName(_vm.HasMessages ? "messages" : "empty");
        _empty.SetDescription(_vm.HasModels ? "" : Loc.S("EmptyNoModel"));
        _getModel.SetVisible(!_vm.HasModels);
        _send.SetVisible(!_vm.IsGenerating);
        _stop.SetVisible(_vm.IsGenerating);
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
            text.SetMarginStart(120);
            return text;
        }

        var content = Ui.Label(item.Content, "message").Named("AssistantMessage");
        content.SetSelectable(true);
        var thinkingText = Ui.Label(item.Reasoning, "dim-label");
        thinkingText.SetSelectable(true);
        var thinking = Gtk.Expander.New(Loc.S("Thinking"));
        thinking.SetChild(thinkingText.Margins(6, 0, 6, 12));
        thinking.Named("ThinkingExpander");
        var spinner = Gtk.Spinner.New();
        var status = Ui.Label(item.Status, "dim-label").Named("MessageStatus");
        var statusRow = Ui.HBox(10, spinner, status);
        var footer = Ui.Label(item.Footer, "caption", "dim-label").Named("MessageFooter");
        var box = Ui.VBox(10, thinking, statusRow, content, footer);

        void Update()
        {
            content.SetText(item.Content);
            content.SetVisible(item.Content.Length > 0);
            thinkingText.SetText(item.Reasoning);
            thinking.SetVisible(item.Reasoning.Length > 0);
            status.SetText(item.Status);
            statusRow.SetVisible(item.Status.Length > 0);
            spinner.SetSpinning(item.Status.Length > 0);
            footer.SetText(item.Footer);
            footer.SetVisible(item.Footer.Length > 0);
        }
        Update();
        item.PropertyChanged += (_, _) =>
        {
            Update();
            ScrollToEnd();
        };
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
