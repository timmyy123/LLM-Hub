using LLMHub.Services;
using LLMHub.Services.Design;
using LLMHub.ViewModels;

namespace LLMHub.Linux;

/// <summary>
/// Design from the design canvas: the conversation with kind tabs on the left, the live WebKit preview with sizes,
/// versions, Comment / Hand off / Export in the middle, and the design-system panel on the right.
/// </summary>
internal sealed class DesignPage
{
    private readonly MainWindow _main;
    private readonly DesignViewModel _vm = Ioc.DesignVm;
    private readonly Action<string> _selectKind;
    private readonly Action<string> _selectSize;
    private readonly Gtk.Box _messages = Gtk.Box.New(Gtk.Orientation.Vertical, 14);
    private readonly Gtk.Box _empty;
    private readonly Gtk.Label _progress = Ui.Label("", "caption", "dim-label");
    private readonly Gtk.Box _progressRow;
    private readonly Gtk.Label _status = Ui.Label("", "caption", "dim-label");
    private readonly Gtk.ScrolledWindow _conversation;
    private readonly Gtk.TextView _input = Gtk.TextView.New();
    private readonly Picker _models;
    private readonly Gtk.Button _send;
    private readonly Gtk.Button _stop;
    private readonly Gtk.Box _versions;
    private readonly Gtk.Label _version = Ui.Label("", "caption", "dim-label");
    private readonly Gtk.Box _frame = Gtk.Box.New(Gtk.Orientation.Vertical, 0);
    private readonly Gtk.Label _nothing = Ui.Label(Loc.S("DesignNothingYet"));
    private readonly WebKit.WebView? _web;
    private readonly Picker _systems;
    private readonly Gtk.Box _swatches = Gtk.Box.New(Gtk.Orientation.Horizontal, 8);
    private readonly Gtk.Label _headingFont = Ui.Label("", "title-3");
    private readonly Gtk.Label _bodyFont = Ui.Label("", "dim-label");
    private readonly Gtk.Label _selected = Ui.Label("", "dim-label");
    private string? _shown;

    public Gtk.Widget Root { get; }

    public DesignPage(MainWindow main)
    {
        _main = main;

        // Conversation column.
        var (kinds, selectKind) = Ui.Tabs(_vm.Kinds.Select(k => (k.Kind.ToString(), k.Label)), OnKind);
        _selectKind = selectKind;
        var kindBar = kinds.Margins(10, 12, 10, 12);
        kindBar.AddCssClass("divider-bottom");
        _empty = Ui.VBox(8, Ui.Label(Loc.S("DesignEmptyTitle"), "title-4").Named("DesignEmptyTitle"), Ui.Label(Loc.S("DesignEmptyBody"), "dim-label"));
        var dot = Gtk.Box.New(Gtk.Orientation.Horizontal, 0);
        dot.AddCssClass("dot");
        dot.AddCssClass("ready");
        dot.SetValign(Gtk.Align.Center);
        _progress.Named("DesignProgress");
        _progressRow = Ui.HBox(8, dot, _progress);
        _status.Named("DesignStatus");
        _conversation = Ui.Scroll(Ui.VBox(14, _empty, _messages, _progressRow, _status).Margins(16, 16, 16, 16));

        _input.SetWrapMode(Gtk.WrapMode.WordChar);
        _input.SetAcceptsTab(false);
        _input.Named("DesignComposer");
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
        inputScroll.SetMinContentHeight(44);
        inputScroll.SetMaxContentHeight(160);
        inputScroll.SetPropagateNaturalHeight(true);
        _models = new Picker(_vm.Models.Select(m => m.DisplayName), i =>
        {
            if (i < _vm.Models.Count)
                _vm.SelectedModel = _vm.Models[i];
        });
        _models.DropDown.AddCssClass("flat");
        _models.DropDown.Named("DesignModelPicker");
        _models.DropDown.SetHexpand(true);
        _models.DropDown.SetHalign(Gtk.Align.Start);
        _send = Ui.IconButton("go-up-symbolic", Loc.S("Send"), Send, "suggested-action", "composer-send").Named("DesignSend");
        _stop = Ui.IconButton("media-playback-stop-symbolic", Loc.S("Stop"), () => _vm.StopCommand.Execute(null), "composer-send").Named("DesignStop");
        var composer = Ui.VBox(8, inputScroll, Ui.HBox(6, _models.DropDown, _send, _stop)).Margins(12, 12, 12, 12);
        composer.AddCssClass("composer");
        var left = Ui.VBox(0, kindBar, _conversation, composer);
        left.SetSizeRequest(340, -1);
        left.AddCssClass("divider-right");

        // Preview column.
        var (sizes, selectSize) = Ui.Tabs([("desktop", Loc.S("DesignDesktop")), ("tablet", Loc.S("DesignTablet")), ("phone", Loc.S("DesignPhone"))],
            size => _vm.PreviewSize = size);
        _selectSize = selectSize;
        sizes.AddCssClass("chip");
        _version.Named("DesignVersion");
        _version.SetWrap(false);
        _versions = Ui.HBox(2,
            Ui.IconButton("go-previous-symbolic", "Previous version", () => _vm.PreviousVersionCommand.Execute(null), "flat"),
            _version,
            Ui.IconButton("go-next-symbolic", "Next version", () => _vm.NextVersionCommand.Execute(null), "flat"));
        foreach (var w in new Gtk.Widget[] { sizes, _versions, _version })
            w.SetValign(Gtk.Align.Center);
        var spacer = Gtk.Box.New(Gtk.Orientation.Horizontal, 0).Expand(true, false);
        var comment = Ui.Button(Loc.S("DesignComment"), () => _input.GrabFocus()).Named("CommentButton");
        var handOff = Ui.Button(Loc.S("DesignHandOff"), HandOff).Named("HandOffButton");
        var export = Gtk.MenuButton.New();
        export.SetLabel(Loc.S("DesignExport"));
        export.AddCssClass("commit-button");
        export.Named("ExportButton");
        var menu = Ui.VBox(2);
        foreach (var (label, format) in new[] { ("HTML", "html"), ("PDF", "pdf"), ("ZIP", "zip"), ("Markdown", "md") })
            menu.Append(Ui.Button(label, () =>
            {
                export.Popdown();
                Export(format);
            }, "flat"));
        var popover = Gtk.Popover.New();
        popover.SetChild(menu);
        export.SetPopover(popover);
        var toolbar = Ui.HBox(8, sizes, _versions, spacer, comment, handOff, export).Margins(8, 16, 8, 16);
        toolbar.AddCssClass("divider-bottom");

        try
        {
            _web = WebKit.WebView.New();
            var content = _web.GetUserContentManager()!;
            content.AddScript(WebKit.UserScript.New(DesignService.PickerScript("window.webkit.messageHandlers.llmhub.postMessage"),
                WebKit.UserContentInjectedFrames.TopFrame, WebKit.UserScriptInjectionTime.End, null, null));
            content.OnScriptMessageReceived += (_, e) =>
            {
                if (e.Value.ToJson(0) is { } json)
                    _vm.SelectedElement = DesignService.DescribePicked(json);
            };
            content.RegisterScriptMessageHandler("llmhub", null);
            _web.OnLoadChanged += (_, e) => Log.Info($"Design preview {e.LoadEvent} {_web.GetUri()}");
            _web.OnLoadFailed += (_, e) =>
            {
                Log.Info($"Design preview failed {e.FailingUri}: {e.Error.Message}");
                return false;
            };
            _web.OnWebProcessTerminated += (_, e) => Log.Info($"Design preview web process ended: {e.Reason}");
            _web.Expand();
            _web.Named("DesignPreview");
            _frame.Append(_web);
        }
        catch (Exception e)
        {
            // WebKitGTK 6.0 missing: say what to install instead of a blank pane.
            Log.Error("WebKitGTK unavailable", e);
            _frame.Append(Ui.Label(Loc.S("DesignNeedsWebKit"), "dim-label").Margins(24, 24, 24, 24));
        }
        _nothing.SetHalign(Gtk.Align.Center);
        _nothing.SetValign(Gtk.Align.Center);
        _nothing.AddCssClass("preview-empty");
        var overlay = Gtk.Overlay.New();
        overlay.SetChild(_frame);
        overlay.AddOverlay(_nothing);
        _frame.AddCssClass("preview-frame");
        _frame.SetOverflow(Gtk.Overflow.Hidden);
        overlay.SetHalign(Gtk.Align.Center);
        overlay.SetVexpand(true);
        overlay.Margins(24, 24, 24, 24);
        var middle = Ui.VBox(0, toolbar, overlay);
        middle.AddCssClass("preview-area");
        middle.SetHexpand(true);

        // Design system panel.
        _systems = new Picker(_vm.Systems.Select(s => s.Name), i => _vm.System = _vm.Systems[i]);
        _systems.DropDown.Named("SystemPicker");
        _swatches.SetHomogeneous(true);
        var type = Ui.VBox(6, _headingFont, _bodyFont);
        type.AddCssClass("tasks-card");
        _selected.Named("DesignSelectedElement");
        var selectedBox = Ui.VBox(0, _selected);
        selectedBox.AddCssClass("tasks-card");
        var right = Ui.VBox(18,
            Ui.Field(Loc.S("DesignSystemLabel"), _systems.DropDown),
            Ui.Field(Loc.S("DesignColors"), _swatches),
            Ui.Field(Loc.S("DesignType"), type),
            Ui.Field(Loc.S("DesignSelected"), selectedBox),
            Ui.Label(Loc.S("DesignExports"), "caption", "dim-label")).Margins(16, 16, 16, 16);
        var rightScroll = Ui.Scroll(right);
        rightScroll.SetSizeRequest(260, -1);
        rightScroll.AddCssClass("side-panel");

        Root = Ui.HBox(0, left, middle, rightScroll);

        Ui.Watch(_vm, OnVmChanged);
        _vm.Messages.CollectionChanged += (_, _) => UiThread.Run(RebuildMessages);
        _vm.Models.CollectionChanged += (_, _) => UiThread.Run(SyncModels);
        SyncModels();
        RebuildMessages();
        SyncAll();
    }

    private void OnKind(string tag)
    {
        _vm.Kind = _vm.Kinds.First(k => k.Kind.ToString() == tag);
        // A different kind starts a new project, so a deck never overwrites a prototype.
        if (_vm.Selected is { } p && p.Kind != _vm.Kind.Kind && p.Versions > 0)
            _vm.NewProjectCommand.Execute(null);
        else if (_vm.Selected is { Versions: 0 } empty)
            empty.Kind = _vm.Kind.Kind;
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
        if (property is nameof(DesignViewModel.SelectedModel))
            SyncModels();
        else
            SyncAll();
        if (property is nameof(DesignViewModel.LiveText) or nameof(DesignViewModel.Selected))
            ScrollToEnd();
    }

    private void SyncAll()
    {
        _selectKind(_vm.Kind.Kind.ToString());
        _selectSize(_vm.PreviewSize);
        _empty.SetVisible(_vm.Messages.Count == 0);
        _progressRow.SetVisible(_vm.IsGenerating);
        _progress.SetText(Loc.S("DesignWorking") + (_vm.LiveText.Length > 0 ? $" ({_vm.LiveText.Length / 1000.0:F1}k)" : "")
                          + (_vm.SelectedElement.Length > 0 ? " · " + _vm.SelectedElement : ""));
        _status.SetText(_vm.Status);
        _status.SetVisible(_vm.Status.Length > 0);
        _input.SetTooltipText(Loc.S(_vm.HasPreview ? "DesignPlaceholder" : "DesignStartPlaceholder"));
        _send.SetVisible(!_vm.IsGenerating);
        _stop.SetVisible(_vm.IsGenerating);
        _versions.SetVisible(_vm.HasPreview);
        _version.SetText(_vm.VersionText);
        _nothing.SetVisible(!_vm.HasPreview);
        _frame.SetSizeRequest(Math.Min(_vm.PreviewWidth, 1280), -1);
        _systems.Select(_vm.Systems.ToList().IndexOf(_vm.System));
        _swatches.Clear();
        foreach (var color in _vm.System.Colors)
        {
            var swatch = Gtk.Box.New(Gtk.Orientation.Horizontal, 0);
            swatch.AddCssClass("swatch");
            swatch.SetTooltipText(color);
            var css = Gtk.CssProvider.New();
            css.LoadFromString($"box {{ background: {color}; }}");
            swatch.GetStyleContext().AddProvider(css, 700);
            _swatches.Append(swatch);
        }
        _headingFont.SetText(_vm.System.HeadingFont);
        _bodyFont.SetText(_vm.System.BodyFont);
        _selected.SetText(_vm.SelectedElement.Length > 0 ? _vm.SelectedElement : Loc.S("DesignSelectHint"));
        ShowPreview();
    }

    private void ShowPreview()
    {
        var file = _vm.PreviewFile;
        if (_web is null || file == _shown)
            return;
        _shown = file;
        if (file is not null && File.Exists(file))
            _web.LoadUri(new Uri(file).AbsoluteUri);
        else
            _web.LoadHtml("<html><body style='background:#fff'></body></html>", null);
    }

    private void SyncModels() =>
        _models.SetItems(_vm.Models.Select(m => m.DisplayName), _vm.SelectedModel is { } m ? _vm.Models.IndexOf(m) : -1);

    private void RebuildMessages()
    {
        _messages.Clear();
        foreach (var m in _vm.Messages)
        {
            var text = Ui.Label(m.Text).Named(m.Role == "user" ? "DesignUserMessage" : "DesignAssistantMessage");
            text.SetSelectable(true);
            if (m.Role == "user")
            {
                var bubble = Ui.VBox(0, text);
                bubble.AddCssClass("user-bubble");
                bubble.SetHalign(Gtk.Align.End);
                text.SetMaxWidthChars(36);
                _messages.Append(bubble);
            }
            else
                _messages.Append(text);
        }
        _empty.SetVisible(_vm.Messages.Count == 0);
        ScrollToEnd();
    }

    private async void HandOff()
    {
        await _vm.HandOffToAgentAsync();
        _main.ShowSection("agent");
    }

    private async void Export(string format)
    {
        if (_vm.Selected is not { CurrentFile: not null } project)
            return;
        try
        {
            var dialog = Gtk.FileDialog.New();
            dialog.SetInitialName(DesignViewModel.Safe(project.Name) + format switch { "md" => ".md", "zip" => ".zip", "pdf" => ".pdf", _ => ".html" });
            if (await dialog.SaveAsync(_main.Window) is not { } file || file.GetPath() is not { } path)
                return;
            if (format == "pdf" && _web is not null)
            {
                var settings = Gtk.PrintSettings.New();
                settings.SetPrinter("Print to File");
                settings.Set("output-file-format", "pdf");
                settings.Set("output-uri", new Uri(path).AbsoluteUri);
                var print = WebKit.PrintOperation.New(_web);
                print.SetPrintSettings(settings);
                print.Print();
            }
            else
                await DesignService.ExportAsync(project, format, path);
            _vm.Status = Loc.F("StudioSavedTo", path);
        }
        catch (GLib.GException)
        {
            // Dismissed.
        }
        catch (Exception e)
        {
            Log.Error("Design export failed", e);
            _vm.Status = Loc.F("ErrorPrefix", e.Message);
        }
    }

#if DEBUG
    /// <summary>
    /// Checks what WebKit loaded (title, size, the picker script) and clicks the first heading through the page, so the
    /// picker → message handler → selected element path runs. GTK's window capture leaves web content blank on Xvfb.
    /// </summary>
    public async Task ProbePreviewAsync()
    {
        if (_web is null)
            return;
        try
        {
            const string probe = "JSON.stringify({html: document.documentElement.outerHTML.slice(0, 160), title: document.title, chars: document.body.innerText.length, elements: document.querySelectorAll('*').length, " +
                                 "height: document.documentElement.scrollHeight, picker: !!window.__llmhubPicker, " +
                                 "clicked: (() => { const h = document.querySelector('h1'); if (h) h.click(); return h ? h.innerText.slice(0, 40) : null; })()})";
            var value = await _web.EvaluateJavascriptAsync(probe);
            await Task.Delay(500);
            Log.Info($"PREVIEW uri={_web.GetUri()} {value.ToString()} selected='{_vm.SelectedElement}'");
        }
        catch (Exception e)
        {
            Log.Error("Preview probe failed", e);
        }
    }
#endif

    private void ScrollToEnd() => GLib.Functions.TimeoutAdd(GLib.Constants.PRIORITY_DEFAULT, 60, () =>
    {
        var adj = _conversation.GetVadjustment()!;
        adj.SetValue(adj.GetUpper() - adj.GetPageSize());
        return false;
    });
}
