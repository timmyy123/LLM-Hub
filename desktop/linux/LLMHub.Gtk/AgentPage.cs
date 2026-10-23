using System.Collections.Specialized;
using LLMHub.Services;
using LLMHub.Services.Agent;
using LLMHub.ViewModels;

namespace LLMHub.Linux;

/// <summary>
/// The Agent tab from the design: the session transcript (messages, tool steps with diffs, tasks, permission prompts)
/// and composer on the left, Diff / Terminal / Files / Preview with Revert and Commit on the right.
/// </summary>
internal sealed class AgentPage
{
    private readonly MainWindow _main;
    private readonly AgentViewModel _vm = Ioc.AgentVm;
    private readonly Gtk.Stack _root = Gtk.Stack.New();
    private readonly Gtk.Label _title = Ui.Label("", "heading");
    private readonly Gtk.Label _folder = Ui.Label("", "mono");
    private readonly Gtk.Label _branch = Ui.Label("", "mono");
    private readonly Gtk.Box _branchChip;
    private readonly Gtk.Box _tasks = Gtk.Box.New(Gtk.Orientation.Vertical, 6);
    private readonly Gtk.Box _tasksCard;
    private readonly Gtk.Box _items = Gtk.Box.New(Gtk.Orientation.Vertical, 14);
    private readonly Gtk.Box _permission = Gtk.Box.New(Gtk.Orientation.Vertical, 10);
    private readonly Gtk.Label _permissionQuestion = Ui.Label("", "heading");
    private readonly Gtk.Label _permissionSummary = Ui.Label("", "mono");
    private readonly Gtk.ScrolledWindow _transcript;
    private readonly Gtk.TextView _input = Gtk.TextView.New();
    private readonly Gtk.Label _notice = Ui.Label("", "caption", "dim-label");
    private readonly Picker _modes;
    private readonly Picker _models;
    private readonly Gtk.Button _send;
    private readonly Gtk.Button _stop;
    private readonly Gtk.Stack _side = Gtk.Stack.New();
    private readonly Gtk.ListBox _changes = Gtk.ListBox.New();
    private readonly Gtk.Box _diff = Gtk.Box.New(Gtk.Orientation.Vertical, 0);
    private readonly Gtk.Label _noChanges = Ui.Label(Loc.S("AgentNoChanges"), "dim-label");
    private readonly Gtk.Label _terminal = Ui.Label("", "mono");
    private readonly Gtk.ListBox _files = Gtk.ListBox.New();
    private readonly Gtk.Label _previewText = Ui.Label("", "dim-label");
    private Gtk.Button _openPreview = null!;
    private readonly Gtk.Entry _commitMessage = Gtk.Entry.New();
    private readonly Gtk.Label _installStatus = Ui.Label("", "caption", "dim-label");
    private readonly Gtk.Widget _installCard;
    private AgentSession? _watched;
    private bool _syncingChanges;

    public Gtk.Widget Root => _root;

    public AgentPage(MainWindow main)
    {
        _main = main;

        // Empty state: pick a folder, and install the agent engine if it is missing.
        var emptyTitle = Ui.Label(Loc.S("AgentEmptyTitle"), "title-1").Named("AgentEmptyTitle");
        emptyTitle.SetXalign(0.5f);
        emptyTitle.SetJustify(Gtk.Justification.Center);
        var emptyBody = Ui.Label(Loc.S("AgentEmptyBody"), "dim-label");
        emptyBody.SetXalign(0.5f);
        emptyBody.SetJustify(Gtk.Justification.Center);
        var pick = Ui.Button(Loc.S("AgentPickFolder"), () => PickFolderAndStart(_main.Window), "pill", "suggested-action").Named("PickFolderButton");
        pick.SetHalign(Gtk.Align.Center);
        var install = Ui.Button(Loc.S("AgentInstall"), () => _vm.InstallClaudeCommand.Execute(null)).Named("InstallClaudeButton");
        var card = Ui.VBox(10, Ui.Label(Loc.S("AgentEngine"), "heading"), Ui.Label(Loc.S("AgentInstallHelp"), "dim-label"), Ui.HBox(10, install, _installStatus));
        card.AddCssClass("tasks-card");
        _installCard = card;
        var inner = Ui.VBox(16, emptyTitle, emptyBody, pick, card);
        inner.SetSizeRequest(520, -1);
        inner.SetValign(Gtk.Align.Center);
        inner.SetHalign(Gtk.Align.Center);

        // Session header: title, folder and branch chips.
        _title.SetWrap(false);
        _title.SetEllipsize(Pango.EllipsizeMode.End);
        _title.SetMaxWidthChars(40);
        _title.Named("AgentSessionTitle");
        _folder.SetWrap(false);
        _folder.SetEllipsize(Pango.EllipsizeMode.Middle);
        _folder.SetMaxWidthChars(36);
        _branch.SetWrap(false);
        var folderChip = Ui.HBox(6, Gtk.Image.NewFromIconName("folder-symbolic"), _folder);
        folderChip.AddCssClass("chip");
        _branchChip = Ui.HBox(6, Gtk.Image.NewFromIconName("media-playlist-shuffle-symbolic"), _branch);
        _branchChip.AddCssClass("chip");
        var header = Ui.HBox(10, _title, folderChip, _branchChip).Margins(10, 24, 10, 24);
        header.AddCssClass("divider-bottom");
        foreach (var chip in new[] { folderChip, _branchChip })
            chip.SetValign(Gtk.Align.Center);

        // Transcript: tasks, items, permission card.
        _tasksCard = Ui.VBox(8, Ui.Label(Loc.S("AgentTasks"), "field-label"), _tasks).Named("AgentTasks");
        _tasksCard.AddCssClass("tasks-card");
        var allow = Ui.Button(Loc.S("AgentAllow"), () => _vm.AllowCommand.Execute(null), "suggested-action").Named("AllowButton");
        var allowSession = Ui.Button(Loc.S("AgentAllowSession"), () => _vm.AllowForSessionCommand.Execute(null)).Named("AllowSessionButton");
        var deny = Ui.Button(Loc.S("AgentDeny"), () => _vm.DenyCommand.Execute(null)).Named("DenyButton");
        _permissionQuestion.Named("PermissionQuestion");
        var summaryBox = Ui.VBox(0, _permissionSummary);
        summaryBox.AddCssClass("code-block");
        _permission.Append(_permissionQuestion);
        _permission.Append(summaryBox);
        _permission.Append(Ui.HBox(8, allow, allowSession, deny));
        _permission.AddCssClass("permission-card");
        _permission.Named("PermissionCard");
        var column = Ui.VBox(16, _tasksCard, _items.Named("AgentItems"), _permission).Margins(24, 24, 12, 24);
        var clamp = Adw.Clamp.New();
        clamp.SetMaximumSize(760);
        clamp.SetChild(column);
        _transcript = Ui.Scroll(clamp);

        // Composer: prompt, permission mode, model, send / stop.
        _input.SetWrapMode(Gtk.WrapMode.WordChar);
        _input.SetAcceptsTab(false);
        _input.SetTooltipText(Loc.S("AgentPlaceholder"));
        _input.Named("AgentComposer");
        var keys = Gtk.EventControllerKey.New();
        keys.OnKeyPressed += (_, e) =>
        {
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
        inputScroll.SetMaxContentHeight(200);
        inputScroll.SetPropagateNaturalHeight(true);
        _modes = new Picker(_vm.Modes.Select(m => m.Label), i => _vm.Mode = _vm.Modes[i]);
        _modes.DropDown.Named("AgentMode");
        _models = new Picker(_vm.Models.Select(m => m.DisplayName), i =>
        {
            if (i < _vm.Models.Count)
                _vm.SelectedModel = _vm.Models[i];
        });
        _models.DropDown.AddCssClass("flat");
        _models.DropDown.Named("AgentModelPicker");
        _send = Ui.IconButton("go-up-symbolic", Loc.S("Send"), Send, "suggested-action", "composer-send").Named("AgentSend");
        _stop = Ui.IconButton("media-playback-stop-symbolic", Loc.S("Stop"), () => _vm.StopCommand.Execute(null), "composer-send").Named("AgentStop");
        var spacer = Gtk.Box.New(Gtk.Orientation.Horizontal, 0).Expand(true, false);
        var composer = Ui.VBox(10, inputScroll, Ui.HBox(8, _modes.DropDown, spacer, _models.DropDown, _send, _stop));
        composer.AddCssClass("composer");
        _notice.Named("AgentNotice");
        var footer = Ui.Label(Loc.S("AgentFooter"), "caption", "dim-label");
        footer.SetXalign(0.5f);
        footer.SetJustify(Gtk.Justification.Center);
        var composerClamp = Adw.Clamp.New();
        composerClamp.SetMaximumSize(760);
        composerClamp.SetChild(Ui.VBox(8, _notice, composer, footer));

        var conversation = Ui.VBox(0, header, _transcript, composerClamp.Margins(8, 24, 16, 24));
        conversation.SetHexpand(true);

        var session = Ui.HBox(0, conversation, BuildSidePanel());
        _root.AddNamed(inner, "empty");
        _root.AddNamed(session, "session");

        Ui.Watch(_vm, OnVmChanged);
        Ui.Watch(_vm.Engine, p =>
        {
            if (p == nameof(AgentService.InstallStatus))
                _installStatus.SetText(_vm.Engine.InstallStatus);
            _installCard.SetVisible(!_vm.IsClaudeInstalled);
        });
        _vm.Models.CollectionChanged += (_, _) => UiThread.Run(SyncModels);
        _vm.Files.CollectionChanged += (_, _) => UiThread.Run(RebuildFiles);
        SyncModels();
        _modes.Select(_vm.Modes.ToList().IndexOf(_vm.Mode));
        Watch(_vm.Selected);
        UpdateState();
        _installCard.SetVisible(!_vm.IsClaudeInstalled);
    }

    private Gtk.Widget BuildSidePanel()
    {
        // Diff: changed files on top, the selected file's diff below.
        _changes.AddCssClass("navigation-sidebar");
        _changes.Named("ChangeList");
        _changes.OnRowSelected += (_, e) =>
        {
            if (_syncingChanges || e.Row is null || _vm.Selected is not { } s || e.Row.GetIndex() >= s.Changes.Count)
                return;
            _vm.SelectedChange = s.Changes[e.Row.GetIndex()];
        };
        var changesScroll = Gtk.ScrolledWindow.New();
        changesScroll.SetChild(_changes);
        changesScroll.SetPolicy(Gtk.PolicyType.Never, Gtk.PolicyType.Automatic);
        changesScroll.SetMaxContentHeight(200);
        changesScroll.SetPropagateNaturalHeight(true);
        var diffPanel = Ui.VBox(8, _noChanges.Margins(8, 4, 0, 4), changesScroll, Ui.Scroll(_diff, horizontal: true)).Margins(4, 12, 0, 12);

        _terminal.SetSelectable(true);
        _terminal.Named("AgentTerminalText");
        var terminalPanel = Ui.Scroll(_terminal.Margins(8, 14, 8, 14));

        _files.AddCssClass("navigation-sidebar");
        _files.OnRowActivated += (_, e) =>
        {
            if (_vm.Selected is { } s && e.Row.GetIndex() < _vm.Files.Count)
                Ui.Open(Path.Combine(s.Folder, _vm.Files[e.Row.GetIndex()]));
        };
        var filesPanel = Ui.Scroll(_files);

        _openPreview = Ui.Button(Loc.S("AgentPreview"), () =>
        {
            if (_vm.PreviewFile is { } file)
                Ui.Open(file);
        });
        _openPreview.SetHalign(Gtk.Align.Start);
        var previewPanel = Ui.VBox(12, _previewText, _openPreview).Margins(16, 16, 16, 16);

        _side.AddTitled(diffPanel, "diff", Loc.S("AgentDiff"));
        _side.AddTitled(terminalPanel, "terminal", Loc.S("AgentTerminal"));
        _side.AddTitled(filesPanel, "files", Loc.S("AgentFiles"));
        _side.AddTitled(previewPanel, "preview", Loc.S("AgentPreview"));
        _side.SetVexpand(true);
        _side.OnNotify += (_, e) =>
        {
            if (e.Pspec.GetName() != "visible-child-name")
                return;
            _vm.SideTab = _side.GetVisibleChildName() ?? "diff";
            _previewText.SetText(_vm.PreviewFile ?? Loc.S("AgentNoPreview"));
            _openPreview.SetSensitive(_vm.PreviewFile is not null);
        };
        var switcher = Gtk.StackSwitcher.New();
        switcher.SetStack(_side);
        switcher.Margins(10, 10, 10, 10);
        switcher.Named("AgentSideTabs");

        _commitMessage.SetPlaceholderText(Loc.S("AgentCommitMessage"));
        _commitMessage.Named("CommitMessageBox");
        _commitMessage.OnChanged += (_, _) => _vm.CommitMessage = _commitMessage.GetText();
        var revert = Ui.Button(Loc.S("AgentRevert"), () => _vm.RevertCommand.Execute(null)).Named("RevertButton").Expand(true, false);
        var commit = Ui.Button(Loc.S("AgentCommit"), () => _vm.CommitCommand.Execute(null), "commit-button").Named("CommitButton").Expand(true, false);
        var actions = Ui.VBox(8, _commitMessage, Ui.HBox(8, revert, commit)).Margins(12, 12, 12, 12);
        actions.AddCssClass("divider-top");

        var panel = Ui.VBox(0, switcher, _side, actions);
        panel.SetSizeRequest(360, -1);
        panel.AddCssClass("side-panel");
        panel.Named("AgentSidePanel");
        return panel;
    }

    public static async void PickFolderAndStart(Gtk.Window window)
    {
        try
        {
            var dialog = Gtk.FileDialog.New();
            dialog.SetTitle(Loc.S("AgentPickFolder"));
            if (await dialog.SelectFolderAsync(window) is { } folder && folder.GetPath() is { } path)
                Ioc.AgentVm.NewSession(path);
        }
        catch (GLib.GException)
        {
            // Dismissed.
        }
    }

    private void Send()
    {
        _vm.Draft = _input.GetBuffer().Text ?? "";
        if (_vm.Draft.Trim().Length == 0)
            return;
        _input.GetBuffer().SetText("", 0);
        _vm.SendCommand.Execute(null);
    }

    private void OnVmChanged(string? property)
    {
        switch (property)
        {
            case nameof(AgentViewModel.Selected):
                Watch(_vm.Selected);
                UpdateState();
                break;
            case nameof(AgentViewModel.HasSession):
                UpdateState();
                break;
            case nameof(AgentViewModel.Notice):
                _notice.SetText(_vm.Notice);
                _notice.SetVisible(_vm.Notice.Length > 0);
                break;
            case nameof(AgentViewModel.Mode):
                _modes.Select(_vm.Modes.ToList().IndexOf(_vm.Mode));
                break;
            case nameof(AgentViewModel.SelectedModel):
                SyncModels();
                break;
            case nameof(AgentViewModel.SelectedChange):
                RenderDiff();
                SyncChangeSelection();
                break;
            case nameof(AgentViewModel.Draft) when (_input.GetBuffer().Text ?? "") != _vm.Draft:
                _input.GetBuffer().SetText(_vm.Draft, -1);
                break;
            case nameof(AgentViewModel.CommitMessage) when _commitMessage.GetText() != _vm.CommitMessage:
                _commitMessage.SetText(_vm.CommitMessage);
                break;
        }
    }

    private void SyncModels()
    {
        _models.SetItems(_vm.Models.Select(m => m.DisplayName), _vm.SelectedModel is { } m ? _vm.Models.IndexOf(m) : -1);
    }

    // ---------------------------------------------------------------- session

    private void Watch(AgentSession? session)
    {
        if (_watched is not null)
        {
            _watched.Items.CollectionChanged -= OnItemsChanged;
            _watched.Tasks.CollectionChanged -= OnTasksChanged;
            _watched.Changes.CollectionChanged -= OnChangesChanged;
            _watched.PropertyChanged -= OnSessionChanged;
        }
        _watched = session;
        if (session is not null)
        {
            session.Items.CollectionChanged += OnItemsChanged;
            session.Tasks.CollectionChanged += OnTasksChanged;
            session.Changes.CollectionChanged += OnChangesChanged;
            session.PropertyChanged += OnSessionChanged;
        }
        RebuildItems();
        RebuildTasks();
        RebuildChanges();
        UpdateSession();
    }

    private void OnItemsChanged(object? sender, NotifyCollectionChangedEventArgs e) => UiThread.Run(() =>
    {
        if (e.Action == NotifyCollectionChangedAction.Add && e.NewItems is { } added && sender == _watched?.Items)
        {
            foreach (AgentItem item in added)
                _items.Append(ItemView(item));
            ScrollToEnd();
        }
        else
            RebuildItems();
    });

    private void OnTasksChanged(object? sender, NotifyCollectionChangedEventArgs e) => UiThread.Run(RebuildTasks);
    private void OnChangesChanged(object? sender, NotifyCollectionChangedEventArgs e) => UiThread.Run(RebuildChanges);
    private void OnSessionChanged(object? sender, System.ComponentModel.PropertyChangedEventArgs e) => UiThread.Run(UpdateSession);

    private void UpdateState()
    {
        _root.SetVisibleChildName(_vm.HasSession ? "session" : "empty");
        _notice.SetVisible(_vm.Notice.Length > 0);
    }

    private void UpdateSession()
    {
        var s = _vm.Selected;
        _title.SetText(s?.Title ?? "");
        _folder.SetText(s?.Folder ?? "");
        _branch.SetText(s?.Branch ?? "");
        _branchChip.SetVisible(s is { Branch.Length: > 0 });
        _terminal.SetText(s?.Terminal ?? "");
        _send.SetVisible(s?.IsWorking != true);
        _stop.SetVisible(s?.IsWorking == true);
        _permission.SetVisible(s?.Pending is not null);
        if (s?.Pending is { } p)
        {
            _permissionQuestion.SetText(p.Question);
            _permissionSummary.SetText(p.Summary);
            ScrollToEnd();
        }
    }

    private void RebuildItems()
    {
        _items.Clear();
        if (_vm.Selected is { } s)
            foreach (var item in s.Items)
                _items.Append(ItemView(item));
        ScrollToEnd();
    }

    private Gtk.Widget ItemView(AgentItem item)
    {
        switch (item.Kind)
        {
            case AgentItemKind.User:
                var text = Ui.Label(item.Title, "message").Named("AgentUserMessage");
                text.SetSelectable(true);
                var bubble = Ui.VBox(0, text);
                bubble.AddCssClass("user-bubble");
                bubble.SetHalign(Gtk.Align.End);
                text.SetMaxWidthChars(70);
                return bubble;
            case AgentItemKind.Assistant:
                var markdown = new MarkdownView("AgentMessage");
                markdown.SetText(item.Title);
                Ui.Watch(item, p =>
                {
                    if (p == nameof(AgentItem.Title))
                        markdown.SetText(item.Title);
                });
                return markdown.Widget;
            case AgentItemKind.Error:
                return Ui.Label(item.Title, "error");
            case AgentItemKind.Tasks:
                return Gtk.Box.New(Gtk.Orientation.Vertical, 0);
        }

        // Tool step: label, target, meta; diff or output below.
        var label = Ui.Label(item.Detail, "dim-label");
        label.SetWrap(false);
        var target = Ui.Label(item.Title, "mono");
        target.SetWrap(false);
        target.SetEllipsize(Pango.EllipsizeMode.Middle);
        target.SetHexpand(true);
        var meta = Ui.Label(item.Meta, "caption", "dim-label");
        meta.SetWrap(false);
        var card = Ui.VBox(0, Ui.HBox(10, label, target, meta).Margins(9, 12, 9, 12));
        card.AddCssClass("step-card");
        card.Named("AgentStep");
        card.SetTooltipText(item.Title);
        if (item.HasDiff)
        {
            var diff = Ui.VBox(0);
            foreach (var line in item.Diff)
                diff.Append(DiffLine(line));
            card.Append(diff.Margins(0, 0, 8, 0));
        }
        var output = Ui.Label(item.Output, "mono", "dim-label").Margins(0, 12, 10, 12);
        output.SetSelectable(true);
        output.SetLines(12);
        output.SetEllipsize(Pango.EllipsizeMode.End);
        output.SetVisible(item.HasOutput);
        card.Append(output);
        Ui.Watch(item, p =>
        {
            meta.SetText(item.Meta);
            output.SetText(item.Output);
            output.SetVisible(item.HasOutput);
            if (item.Failed)
                label.AddCssClass("error");
        });
        return card;
    }

    private static Gtk.Widget DiffLine(DiffLine line)
    {
        var label = Gtk.Label.New(line.Kind switch { '+' => "+ ", '-' => "- ", '@' => "", _ => "  " } + line.Text);
        label.SetXalign(0);
        label.SetSelectable(true);
        label.AddCssClass("diff-line");
        label.AddCssClass(line.Kind switch { '+' => "diff-add", '-' => "diff-del", '@' => "diff-hunk", _ => "diff-ctx" });
        return label;
    }

    private void RebuildTasks()
    {
        _tasks.Clear();
        var tasks = _vm.Selected?.Tasks;
        _tasksCard.SetVisible(tasks is { Count: > 0 });
        if (tasks is null)
            return;
        foreach (var task in tasks)
        {
            var icon = Gtk.Image.NewFromIconName(task.IsDone ? "object-select-symbolic" : task.IsActive ? "media-record-symbolic" : "radio-symbolic");
            var text = Ui.Label(task.Text);
            if (task.IsDone)
            {
                text.SetMarkup($"<s>{GLib.Functions.MarkupEscapeText(task.Text, -1)}</s>");
                text.AddCssClass("dim-label");
                icon.AddCssClass("success");
            }
            else if (task.IsActive)
                icon.AddCssClass("accent");
            else
                icon.AddCssClass("dim-label");
            _tasks.Append(Ui.HBox(10, icon, text));
            task.PropertyChanged += (_, _) => UiThread.Run(RebuildTasks);
        }
    }

    private void RebuildChanges()
    {
        _syncingChanges = true;
        _changes.RemoveAll();
        var changes = _vm.Selected?.Changes;
        _noChanges.SetVisible(changes is not { Count: > 0 });
        if (changes is not null)
            foreach (var change in changes)
            {
                var badge = Ui.Label(change.Kind.ToString(), "mono", change.Kind switch { 'A' => "success", 'D' => "error", _ => "warning" });
                badge.SetWrap(false);
                var path = Ui.Label(change.Path, "mono").Named("ChangedFile");
                path.SetWrap(false);
                path.SetEllipsize(Pango.EllipsizeMode.Start);
                path.SetHexpand(true);
                var stats = Ui.Label(change.Stats, "caption", "dim-label");
                stats.SetWrap(false);
                var row = Gtk.ListBoxRow.New();
                row.SetChild(Ui.HBox(8, badge, path, stats).Margins(4, 6, 4, 6));
                _changes.Append(row);
            }
        _syncingChanges = false;
        SyncChangeSelection();
        RenderDiff();
    }

    private void SyncChangeSelection()
    {
        _syncingChanges = true;
        var index = _vm.Selected is { } s && _vm.SelectedChange is { } c ? s.Changes.IndexOf(c) : -1;
        if (index >= 0 && _changes.GetRowAtIndex(index) is { } row)
            _changes.SelectRow(row);
        else
            _changes.UnselectAll();
        _syncingChanges = false;
    }

    private void RenderDiff()
    {
        _diff.Clear();
        if (_vm.SelectedChange is { } change)
            foreach (var line in change.Diff.Take(2000))
                _diff.Append(DiffLine(line));
    }

    private void RebuildFiles()
    {
        _files.RemoveAll();
        foreach (var file in _vm.Files)
        {
            var label = Ui.Label(file, "mono");
            label.SetWrap(false);
            label.SetEllipsize(Pango.EllipsizeMode.Start);
            _files.Append(label.Margins(3, 6, 3, 6));
        }
    }

    private void ScrollToEnd() => GLib.Functions.TimeoutAdd(GLib.Constants.PRIORITY_DEFAULT, 60, () =>
    {
        var adj = _transcript.GetVadjustment()!;
        adj.SetValue(adj.GetUpper() - adj.GetPageSize());
        return false;
    });
}
