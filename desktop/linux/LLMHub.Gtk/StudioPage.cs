using LLMHub.Services;
using LLMHub.Services.Studio;
using LLMHub.ViewModels;

namespace LLMHub.Linux;

/// <summary>
/// Studio from the design: Image / Video / Music / Voice tabs. Image and video run on ComfyUI (form on the left,
/// gallery on the right), Voice on Supertonic; Music explains that Magenta RealTime needs a Mac.
/// </summary>
internal sealed class StudioPage
{
    private readonly MainWindow _main;
    private readonly StudioViewModel _vm = Ioc.StudioVm;
    private readonly Gtk.Stack _body = Gtk.Stack.New();
    private readonly Action<string> _selectTab;
    private readonly Gtk.Box _engine;
    private readonly Gtk.Box _engineDot = Gtk.Box.New(Gtk.Orientation.Horizontal, 0);
    private readonly Gtk.Label _engineText = Ui.Label("", "caption", "dim-label");
    private readonly Picker _workflows;
    private readonly Gtk.Label _downloadNote = Ui.Label("", "caption", "dim-label");
    private readonly Gtk.TextView _prompt = Gtk.TextView.New();
    private readonly Picker _aspects;
    private readonly Picker _counts;
    private readonly Gtk.SpinButton _steps = Gtk.SpinButton.NewWithRange(1, 100, 1);
    private readonly Gtk.Entry _seed = Gtk.Entry.New();
    private readonly Gtk.Button _reference;
    private readonly Gtk.Label _referenceLabel = Ui.Label("");
    private readonly Gtk.Label _status = Ui.Label("", "caption", "dim-label");
    private readonly Gtk.Label _comfyStatus = Ui.Label("", "caption", "dim-label");
    private readonly Gtk.Button _generate;
    private readonly Gtk.Button _cancel;
    private readonly Gtk.ProgressBar _progress = Gtk.ProgressBar.New();
    private readonly Gtk.FlowBox _gallery = Gtk.FlowBox.New();
    private readonly Gtk.TextView _voiceText = Gtk.TextView.New();
    private readonly Picker _voices;
    private readonly Gtk.Label _voiceStatus = Ui.Label("", "caption", "dim-label");
    private readonly Gtk.Label _lastSpeech = Ui.Label("", "mono", "dim-label");
    private bool _syncingText;

    public Gtk.Widget Root { get; }

    public StudioPage(MainWindow main)
    {
        _main = main;

        // Mode tabs and engine status.
        var (tabs, select) = Ui.Tabs([("image", Loc.S("StudioImage")), ("video", Loc.S("StudioVideo")), ("music", Loc.S("StudioMusic")), ("voice", Loc.S("StudioVoice"))],
            mode => _vm.Mode = mode);
        _selectTab = select;
        _engineDot.AddCssClass("dot");
        _engineDot.SetValign(Gtk.Align.Center);
        _engineText.Named("ComfyStatus");
        _engineText.SetWrap(false);
        _engine = Ui.HBox(8, _engineDot, _engineText);
        _engine.SetHexpand(true);
        _engine.SetHalign(Gtk.Align.End);
        var header = Ui.HBox(8, tabs, _engine).Margins(10, 20, 10, 20);
        header.AddCssClass("divider-bottom");

        // Image and video form.
        _workflows = new Picker(_vm.Workflows.Select(w => w.Name), i => _vm.Workflow = _vm.Workflows[i]);
        _workflows.DropDown.Named("WorkflowPicker");
        _prompt.SetWrapMode(Gtk.WrapMode.WordChar);
        _prompt.Named("PromptBox");
        _prompt.GetBuffer().OnChanged += (_, _) =>
        {
            if (!_syncingText)
                _vm.Prompt = _prompt.GetBuffer().Text ?? "";
        };
        var promptFrame = Gtk.Frame.New(null);
        promptFrame.SetChild(_prompt.Margins(8, 8, 8, 8));
        promptFrame.SetSizeRequest(-1, 110);
        var improve = Ui.Button(Loc.S("StudioImprove"), () => _vm.ImprovePromptCommand.Execute(null)).Named("ImproveButton");
        improve.SetHalign(Gtk.Align.Start);
        _aspects = new Picker(_vm.Aspects, i => _vm.Aspect = _vm.Aspects[i]);
        _aspects.DropDown.Named("AspectPicker");
        _counts = new Picker(_vm.Counts.Select(c => c.ToString()), i => _vm.Count = _vm.Counts[i]);
        _counts.DropDown.Named("CountPicker");
        _steps.Named("StepsBox");
        _steps.OnValueChanged += (_, _) => _vm.Steps = _steps.GetValueAsInt();
        _seed.SetPlaceholderText(Loc.S("StudioRandom"));
        _seed.Named("SeedBox");
        _seed.OnChanged += (_, _) => _vm.Seed = _seed.GetText();
        var grid = Gtk.Grid.New();
        grid.SetColumnSpacing(10);
        grid.SetRowSpacing(10);
        grid.SetColumnHomogeneous(true);
        grid.Attach(Ui.Field(Loc.S("StudioAspect"), _aspects.DropDown), 0, 0, 1, 1);
        grid.Attach(Ui.Field(Loc.S("StudioImages"), _counts.DropDown), 1, 0, 1, 1);
        grid.Attach(Ui.Field(Loc.S("StudioSteps"), _steps), 0, 1, 1, 1);
        grid.Attach(Ui.Field(Loc.S("StudioSeed"), _seed), 1, 1, 1, 1);
        _referenceLabel.SetWrap(false);
        _referenceLabel.SetEllipsize(Pango.EllipsizeMode.Middle);
        _reference = Gtk.Button.New();
        _reference.SetChild(Ui.HBox(8, Gtk.Image.NewFromIconName("list-add-symbolic"), _referenceLabel));
        _reference.Named("ReferenceButton");
        _reference.OnClicked += (_, _) => PickReference();
        _status.Named("StudioStatus");
        var unsupported = Ui.Label(Loc.S("ComfyUnsupported"), "caption", "warning");
        unsupported.SetVisible(!_vm.ComfySupported);
        _generate = Ui.Button(Loc.S("StudioGenerate"), () => _vm.GenerateCommand.Execute(null), "suggested-action", "pill").Named("GenerateButton");
        _generate.SetSensitive(_vm.ComfySupported);
        _cancel = Ui.Button(Loc.S("Cancel"), () => _vm.CancelCommand.Execute(null), "pill").Named("CancelGenerateButton");
        var form = Ui.VBox(16,
            Ui.VBox(6, Ui.Label(Loc.S("StudioWorkflow"), "field-label"), _workflows.DropDown, _downloadNote),
            Ui.VBox(6, Ui.Label(Loc.S("StudioPrompt"), "field-label"), promptFrame, improve),
            grid, _reference, _status, _comfyStatus, unsupported, _generate, _cancel).Margins(20, 20, 20, 20);
        var formScroll = Ui.Scroll(form);
        formScroll.SetSizeRequest(340, -1);
        formScroll.AddCssClass("divider-right");

        // Gallery.
        var savedTo = Ui.Label(Loc.F("StudioSavedTo", ComfyService.Gallery), "caption", "dim-label");
        savedTo.SetHexpand(true);
        savedTo.SetWrap(false);
        savedTo.SetEllipsize(Pango.EllipsizeMode.Middle);
        _progress.SetSizeRequest(160, -1);
        _progress.SetValign(Gtk.Align.Center);
        var galleryHeader = Ui.HBox(10, Ui.Label(Loc.S("StudioGallery"), "heading"), savedTo, _progress);
        _gallery.SetSelectionMode(Gtk.SelectionMode.Single);
        _gallery.SetHomogeneous(true);
        _gallery.SetMaxChildrenPerLine(6);
        _gallery.SetColumnSpacing(12);
        _gallery.SetRowSpacing(12);
        _gallery.SetValign(Gtk.Align.Start);
        _gallery.Named("GalleryGrid");
        _gallery.OnChildActivated += (_, e) =>
        {
            if (e.Child.GetIndex() < _vm.Gallery.Count)
                Ui.Open(_vm.Gallery[e.Child.GetIndex()].Path);
        };
        var actions = Ui.HBox(8,
            Ui.Button(Loc.S("StudioUseInChat"), UseInChat),
            Ui.Button(Loc.S("StudioAnimate"), () => UseAs("video", ComfyWorkflow.LtxVideo)),
            Ui.Button(Loc.S("StudioUpscale"), () => UseAs("image", ComfyWorkflow.Upscale)),
            Ui.Button(Loc.S("StudioOpenComfy"), () =>
            {
                if (_vm.Comfy.Url is { } url)
                    Ui.Open(url);
            }));
        var galleryPane = Ui.VBox(14, galleryHeader, Ui.Scroll(_gallery), actions).Margins(20, 20, 20, 20);
        galleryPane.SetHexpand(true);
        _body.AddNamed(Ui.HBox(0, formScroll, galleryPane), "media");

        // Voice.
        _voiceText.SetWrapMode(Gtk.WrapMode.WordChar);
        _voiceText.Named("VoiceTextBox");
        _voiceText.GetBuffer().OnChanged += (_, _) => _vm.VoiceText = _voiceText.GetBuffer().Text ?? "";
        var voiceFrame = Gtk.Frame.New(null);
        voiceFrame.SetChild(_voiceText.Margins(8, 8, 8, 8));
        voiceFrame.SetSizeRequest(-1, 150);
        _voices = new Picker(_vm.Voices, i => _vm.Voice = _vm.Voices[i]);
        _voices.DropDown.Named("VoicePicker");
        _voices.Select(_vm.Voices.ToList().IndexOf(_vm.Voice));
        var speak = Ui.Button(Loc.S("StudioSpeak"), () => _vm.SpeakCommand.Execute(null), "suggested-action").Named("SpeakButton");
        _lastSpeech.Named("LastSpeechPath");
        _lastSpeech.SetSelectable(true);
        var voice = Ui.VBox(16, Ui.Label(Loc.S("StudioVoiceHelp"), "dim-label"),
            Ui.VBox(6, Ui.Label(Loc.S("StudioVoiceText"), "field-label"), voiceFrame),
            Ui.HBox(10, _voices.DropDown, speak), _voiceStatus, _lastSpeech).Margins(28, 28, 28, 28);
        var voiceClamp = Adw.Clamp.New();
        voiceClamp.SetMaximumSize(760);
        voiceClamp.SetChild(voice);
        voiceClamp.SetHalign(Gtk.Align.Start);
        _body.AddNamed(Ui.Scroll(voiceClamp), "voice");

        // Music.
        var music = Adw.StatusPage.New();
        music.SetIconName("audio-x-generic-symbolic");
        music.SetDescription(Loc.S("StudioMusicUnavailable"));
        music.Named("MusicUnavailable");
        _body.AddNamed(music, "music");
        _body.SetVexpand(true);

        Root = Ui.VBox(0, header, _body);

        Ui.Watch(_vm, OnVmChanged);
        Ui.Watch(_vm.Comfy, _ => UpdateEngine());
        _vm.Gallery.CollectionChanged += (_, _) => UiThread.Run(RebuildGallery);
        RebuildGallery();
        SyncAll();
    }

    private void SyncAll()
    {
        _selectTab(_vm.Mode);
        _body.SetVisibleChildName(_vm.IsImageOrVideo ? "media" : _vm.Mode);
        _engine.SetVisible(_vm.IsImageOrVideo);
        _workflows.Select(_vm.Workflows.ToList().IndexOf(_vm.Workflow));
        _aspects.Select(_vm.Aspects.ToList().IndexOf(_vm.Aspect));
        _counts.Select(_vm.Counts.ToList().IndexOf(_vm.Count));
        if (_steps.GetValueAsInt() != _vm.Steps)
            _steps.SetValue(_vm.Steps);
        _downloadNote.SetText(_vm.DownloadNote);
        _downloadNote.SetVisible(_vm.DownloadNote.Length > 0);
        _reference.SetVisible(_vm.NeedsReference);
        _referenceLabel.SetText(_vm.ReferenceImage.Length > 0 ? Path.GetFileName(_vm.ReferenceImage) : Loc.S("StudioReference"));
        _status.SetText(_vm.Status);
        _status.SetVisible(_vm.Status.Length > 0);
        _voiceStatus.SetText(_vm.Status);
        _lastSpeech.SetText(_vm.LastSpeech ?? "");
        _generate.SetVisible(!_vm.IsGenerating);
        _cancel.SetVisible(_vm.IsGenerating);
        _progress.SetVisible(_vm.IsGenerating);
        _progress.SetFraction(_vm.Progress);
        if ((_prompt.GetBuffer().Text ?? "") != _vm.Prompt)
        {
            _syncingText = true;
            _prompt.GetBuffer().SetText(_vm.Prompt, -1);
            _syncingText = false;
        }
        UpdateEngine();
    }

    private void OnVmChanged(string? property)
    {
        if (property is nameof(StudioViewModel.SelectedItem))
            return;
        SyncAll();
    }

    private void UpdateEngine()
    {
        _engineText.SetText(_vm.Comfy.StatusLine);
        _engineDot.RemoveCssClass("ready");
        if (_vm.Comfy.IsRunning)
            _engineDot.AddCssClass("ready");
        _comfyStatus.SetText(_vm.Comfy.Status);
        _comfyStatus.SetVisible(_vm.Comfy.Status.Length > 0);
    }

    private void RebuildGallery()
    {
        _gallery.RemoveAll();
        foreach (var item in _vm.Gallery)
        {
            Gtk.Widget content;
            if (item.IsVideo)
                content = Ui.VBox(6, Gtk.Image.NewFromIconName("video-x-generic-symbolic"), Ui.Label(item.Name, "caption"));
            else
            {
                var picture = Gtk.Picture.NewForFilename(item.Path);
                picture.SetContentFit(Gtk.ContentFit.Cover);
                content = picture;
            }
            content.SetSizeRequest(240, 160);
            var tile = Ui.VBox(0, content);
            tile.AddCssClass("gallery-tile");
            tile.SetOverflow(Gtk.Overflow.Hidden);
            tile.SetTooltipText(item.Name);
            tile.Named("GalleryItem");
            var click = Gtk.GestureClick.New();
            click.OnPressed += (_, _) => _vm.SelectedItem = item;
            tile.AddController(click);
            _gallery.Append(tile);
        }
    }

    private async void PickReference()
    {
        try
        {
            var dialog = Gtk.FileDialog.New();
            var filter = Gtk.FileFilter.New();
            filter.AddMimeType("image/*");
            var filters = Gio.ListStore.New(Gtk.FileFilter.GetGType());
            filters.Append(filter);
            dialog.SetFilters(filters);
            if (await dialog.OpenAsync(_main.Window) is { } file && file.GetPath() is { } path)
                _vm.ReferenceImage = path;
        }
        catch (GLib.GException)
        {
            // Dismissed.
        }
    }

    private void UseInChat()
    {
        if (_vm.SelectedItem is not { } item)
            return;
        Ioc.Chat.AttachFiles([item.Path]);
        _main.ShowChat();
    }

    private void UseAs(string mode, ComfyWorkflow workflow)
    {
        if (_vm.SelectedItem is not { } item)
            return;
        _vm.Mode = mode;
        _vm.Workflow = workflow;
        _vm.ReferenceImage = item.Path;
    }
}
