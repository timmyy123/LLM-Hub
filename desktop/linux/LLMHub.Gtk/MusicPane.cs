using LLMHub.Services;
using LLMHub.Services.Speech;
using LLMHub.Services.Studio;
using LLMHub.ViewModels;

namespace LLMHub.Linux;

/// <summary>
/// Studio › Music (Music.dc.html): the live Magenta RealTime jam on NVIDIA GPUs (transport, level bars, style sliders the
/// music follows while it plays, sampling settings), Stable Audio 3 clips through ComfyUI, and the saved music.
/// </summary>
internal sealed class MusicPane
{
    private readonly StudioViewModel _vm = Ioc.StudioVm;
    private readonly MagentaService _live = Ioc.Magenta;
    private readonly Gtk.Button _play = Gtk.Button.New();
    private readonly Gtk.Image _playIcon = Gtk.Image.NewFromIconName("media-playback-start-symbolic");
    private readonly Gtk.Spinner _playSpinner = Gtk.Spinner.New();
    private readonly Gtk.Label _subtitle = Ui.Label("", "dim-label");
    private readonly Gtk.Label _liveStatus = Ui.Label("", "caption", "dim-label");
    private readonly Gtk.ToggleButton _record = Gtk.ToggleButton.New();
    private readonly Gtk.DrawingArea _bars = Gtk.DrawingArea.New();
    private readonly Gtk.Box _styles = Ui.VBox(10);
    private readonly Gtk.Label _runsOn = Ui.Label("", "dim-label");
    private readonly Picker _clipModels;
    private readonly Gtk.Label _clipNote = Ui.Label("", "caption", "dim-label");
    private readonly Gtk.TextView _clipPrompt = Gtk.TextView.New();
    private readonly Gtk.SpinButton _clipSeconds = Gtk.SpinButton.NewWithRange(1, 380, 1);
    private readonly Gtk.Entry _clipSeed = Gtk.Entry.New();
    private readonly Gtk.Label _clipStatus = Ui.Label("", "caption", "dim-label");
    private readonly Gtk.ProgressBar _clipProgress = Gtk.ProgressBar.New();
    private readonly Gtk.Button _clipGenerate;
    private readonly Gtk.Button _clipCancel;
    private readonly Gtk.Box _clips = Ui.VBox(6);
    private string? _playingClip;
    private bool _syncing;

    public Gtk.Widget Root { get; }

    public MusicPane()
    {
        var sections = new List<Gtk.Widget>();
        if (MagentaService.IsSupported)
        {
            sections.Add(NowPlaying());
            sections.Add(StylesCard());
            sections.Add(SettingsCard());
        }
        else
        {
            var note = Ui.Label(Loc.S("MusicNeedsNvidia"), "dim-label");
            note.Named("MusicUnavailable");
            sections.Add(Card(Ui.HBox(12, Gtk.Image.NewFromIconName("audio-x-generic-symbolic"), note)));
        }

        // Stable Audio 3 clips.
        _clipModels = new Picker(_vm.MusicWorkflows.Select(w => w.Name), i => _vm.MusicWorkflow = _vm.MusicWorkflows[i]);
        _clipModels.DropDown.Named("MusicModelPicker");
        _clipPrompt.SetWrapMode(Gtk.WrapMode.WordChar);
        _clipPrompt.Named("MusicPromptBox");
        _clipPrompt.GetBuffer().OnChanged += (_, _) =>
        {
            if (!_syncing)
                _vm.MusicPrompt = _clipPrompt.GetBuffer().Text ?? "";
        };
        var promptFrame = Gtk.Frame.New(null);
        promptFrame.SetChild(_clipPrompt.Margins(8, 8, 8, 8));
        promptFrame.SetSizeRequest(-1, 90);
        _clipSeconds.Named("MusicSecondsBox");
        _clipSeconds.OnValueChanged += (_, _) =>
        {
            if (!_syncing)
                _vm.MusicSeconds = Math.Clamp(_clipSeconds.GetValueAsInt(), 1, _vm.MusicWorkflow.MaxSeconds);
        };
        _clipSeed.SetPlaceholderText(Loc.S("StudioRandom"));
        _clipSeed.OnChanged += (_, _) => _vm.MusicSeed = _clipSeed.GetText();
        _clipStatus.Named("MusicStatus");
        _clipGenerate = Ui.Button(Loc.S("StudioGenerate"), () => _vm.GenerateMusicCommand.Execute(null), "suggested-action", "pill").Named("GenerateMusicButton");
        _clipGenerate.SetSensitive(_vm.ComfySupported);
        _clipCancel = Ui.Button(Loc.S("Cancel"), () => _vm.CancelMusicCommand.Execute(null), "pill");
        var grid = Gtk.Grid.New();
        grid.SetColumnSpacing(10);
        grid.SetRowSpacing(10);
        grid.SetColumnHomogeneous(true);
        grid.Attach(Ui.Field(Loc.S("ClipModel"), _clipModels.DropDown), 0, 0, 1, 1);
        grid.Attach(Ui.Field(Loc.S("MusicLength"), _clipSeconds), 1, 0, 1, 1);
        grid.Attach(Ui.Field(Loc.S("StudioSeed"), _clipSeed), 2, 0, 1, 1);
        var powered = Ui.Label(Loc.S("PoweredByStability"), "caption", "dim-label").Named("PoweredByStability");
        powered.SetHexpand(true);
        powered.SetHalign(Gtk.Align.End);
        var unsupported = Ui.Label(Loc.S("ComfyUnsupported"), "caption", "warning");
        unsupported.SetVisible(!_vm.ComfySupported);
        sections.Add(Card(Ui.VBox(14,
            Ui.HBox(10, Ui.VBox(2, Ui.Label(Loc.S("ClipTitle"), "heading"), Ui.Label(Loc.S("MusicPromptPlaceholder"), "caption", "dim-label")), powered),
            promptFrame, grid, _clipNote, _clipStatus, _clipProgress, unsupported, Ui.HBox(10, _clipGenerate, _clipCancel))));

        // Saved music: clips and live recordings.
        var savedTo = Ui.Label(Loc.F("StudioSavedTo", StudioViewModel.MusicFolder), "caption", "dim-label");
        savedTo.SetWrap(false);
        savedTo.SetEllipsize(Pango.EllipsizeMode.Middle);
        _clips.Named("MusicClips");
        sections.Add(Card(Ui.VBox(10, Ui.Label(Loc.S("MusicRecordings"), "heading"), savedTo, _clips)));

        var column = Ui.VBox(20, sections.ToArray()).Margins(24, 28, 24, 28);
        var clamp = Adw.Clamp.New();
        clamp.SetMaximumSize(1100);
        clamp.SetChild(column);
        Root = Ui.Scroll(clamp);

        Ui.Watch(_vm, _ => SyncClips());
        Ui.Watch(_live, OnLiveChanged);
        _vm.MusicClips.CollectionChanged += (_, _) => UiThread.Run(RebuildClips);
        _live.Styles.CollectionChanged += (_, _) => UiThread.Run(RebuildStyles);
        _live.Saved += _ => UiThread.Post(_vm.LoadMusic);
        RebuildClips();
        RebuildStyles();
        SyncClips();
        SyncLive();
    }

    // ---------------------------------------------------------------- live jam

    private Gtk.Widget NowPlaying()
    {
        _play.SetChild(Ui.HBox(0, _playIcon, _playSpinner));
        _play.AddCssClass("circular");
        _play.AddCssClass("suggested-action");
        _play.SetSizeRequest(48, 48);
        _play.SetValign(Gtk.Align.Center);
        _play.Named("MusicPlayButton");
        _play.OnClicked += async (_, _) => await _live.ToggleAsync();
        _subtitle.Named("MusicSubtitle");
        _liveStatus.Named("MusicStatusText");
        _liveStatus.SetSelectable(true);
        var title = Ui.VBox(2, Ui.Label(Loc.S("MusicLiveJam"), "title-3"), _subtitle, _liveStatus);
        title.SetHexpand(true);
        title.SetValign(Gtk.Align.Center);
        _record.SetLabel(Loc.S("MusicRecord"));
        _record.Named("MusicRecordButton");
        _record.SetValign(Gtk.Align.Center);
        _record.OnToggled += (_, _) =>
        {
            if (!_syncing)
                _live.SetRecording(_record.GetActive());
        };
        var export = Ui.Button(Loc.S("MusicExport"), _live.Export).Named("MusicExportButton");
        export.SetValign(Gtk.Align.Center);
        _bars.SetContentHeight(96);
        _bars.SetHexpand(true);
        _bars.AddCssClass("accent");
        _bars.SetDrawFunc(DrawBars);
        return Card(Ui.VBox(18, Ui.HBox(16, _play, title, _record, export), _bars));
    }

    private void DrawBars(Gtk.DrawingArea area, Cairo.Context cr, int width, int height)
    {
        var levels = _live.Levels;
        var gap = 3.0;
        var bar = (width - gap * (levels.Length - 1)) / levels.Length;
        area.GetColor(out var color);
        for (var i = 0; i < levels.Length; i++)
        {
            var idle = 0.12 + 0.1 * Math.Abs(Math.Sin(i * 0.37) * Math.Cos(i * 0.11));
            var value = _live.IsPlaying ? Math.Max(0.06, levels[i]) : idle;
            var h = Math.Max(3, height * value);
            if (_live.IsPlaying)
                cr.SetSourceRgba(color.Red, color.Green, color.Blue, 1);
            else
                cr.SetSourceRgba(color.Red, color.Green, color.Blue, 0.25);
            RoundedBar(cr, i * (bar + gap), (height - h) / 2, bar, h);
            cr.Fill();
        }
    }

    private static void RoundedBar(Cairo.Context cr, double x, double y, double w, double h)
    {
        var r = Math.Min(w, h) / 2;
        cr.NewSubPath();
        cr.Arc(x + w - r, y + r, r, -Math.PI / 2, 0);
        cr.Arc(x + w - r, y + h - r, r, 0, Math.PI / 2);
        cr.Arc(x + r, y + h - r, r, Math.PI / 2, Math.PI);
        cr.Arc(x + r, y + r, r, Math.PI, 3 * Math.PI / 2);
        cr.ClosePath();
    }

    private Gtk.Widget StylesCard()
    {
        var add = Ui.Button(Loc.S("MusicAddStyle"), _live.AddStyle).Named("MusicAddStyle");
        add.SetHalign(Gtk.Align.Start);
        return Card(Ui.VBox(12, Ui.Label(Loc.S("MusicStyles"), "heading"), Ui.Label(Loc.S("MusicStylesHelp"), "caption", "dim-label"), _styles, add));
    }

    private void RebuildStyles()
    {
        _styles.Clear();
        foreach (var style in _live.Styles)
        {
            var text = Gtk.Entry.New();
            text.SetText(style.Text);
            text.SetPlaceholderText(Loc.S("MusicStylePlaceholder"));
            text.SetHexpand(true);
            text.Named("MusicStyleText");
            text.OnChanged += (_, _) => style.Text = text.GetText();
            var weight = Gtk.Scale.NewWithRange(Gtk.Orientation.Horizontal, 0, 1, 0.01);
            weight.SetValue(style.Weight);
            weight.SetDrawValue(false);
            weight.SetSizeRequest(260, -1);
            weight.Named("MusicStyleWeight");
            weight.OnValueChanged += (_, _) => style.Weight = weight.GetValue();
            var remove = Ui.IconButton("window-close-symbolic", Loc.S("MusicRemoveStyle"), () => _live.RemoveStyle(style), "flat");
            _styles.Append(Ui.HBox(12, text, weight, remove));
        }
    }

    private Gtk.Widget SettingsCard()
    {
        var grid = Gtk.Grid.New();
        grid.SetColumnSpacing(24);
        grid.SetRowSpacing(10);
        grid.SetColumnHomogeneous(true);
        grid.Attach(Slider(Loc.S("MusicTemperature"), _live.Temperature, v => _live.Temperature = v, "MusicTemperature"), 0, 0, 1, 1);
        grid.Attach(Slider(Loc.S("MusicGuidance"), _live.Guidance, v => _live.Guidance = v, "MusicGuidance"), 1, 0, 1, 1);
        grid.Attach(Slider(Loc.S("MusicVariety"), _live.Variety, v => _live.Variety = v, "MusicVariety"), 2, 0, 1, 1);
        grid.Attach(Ui.Field(Loc.S("MusicRunsOn"), _runsOn), 3, 0, 1, 1);
        return Card(grid);
    }

    private static Gtk.Widget Slider(string label, double value, Action<double> set, string name)
    {
        var scale = Gtk.Scale.NewWithRange(Gtk.Orientation.Horizontal, 0, 1, 0.01);
        scale.SetValue(value);
        scale.SetDrawValue(false);
        scale.Named(name);
        scale.OnValueChanged += (_, _) => set(scale.GetValue());
        return Ui.Field(label, scale);
    }

    private void OnLiveChanged(string? property)
    {
        if (property is nameof(MagentaService.Levels))
        {
            _bars.QueueDraw();
            UpdateSubtitle();
            return;
        }
        SyncLive();
    }

    private void SyncLive()
    {
        if (!MagentaService.IsSupported)
            return;
        _playIcon.SetFromIconName(_live.IsPlaying ? "media-playback-stop-symbolic" : "media-playback-start-symbolic");
        _playIcon.SetVisible(!_live.IsBusy);
        _playSpinner.SetVisible(_live.IsBusy);
        _playSpinner.SetSpinning(_live.IsBusy);
        _play.SetSensitive(!_live.IsBusy);
        _play.SetTooltipText(Loc.S(_live.IsPlaying ? "MusicStop" : "MusicPlay"));
        _syncing = true;
        _record.SetActive(_live.IsRecording);
        _syncing = false;
        _record.SetLabel(Loc.S(_live.IsRecording ? "MusicRecording" : "MusicRecord"));
        if (_live.IsRecording)
            _record.AddCssClass("destructive-action");
        else
            _record.RemoveCssClass("destructive-action");
        _runsOn.SetText(Loc.F("MusicCuda", _live.Device.Length > 0 ? _live.Device : Ioc.Hardware.GpuName));
        var status = _live.Status.Length > 0 ? _live.Status
            : _live.IsPlaying && _live.Elapsed > 4 && _live.Speed is > 0 and < 0.95 ? Loc.S("MusicSlowPc")
            : !MagentaService.IsInstalled ? Loc.F("MusicStatusDownload", Fmt.Bytes(MagentaService.DownloadSize))
            : "";
        _liveStatus.SetText(status);
        _liveStatus.SetVisible(status.Length > 0);
        UpdateSubtitle();
        _bars.QueueDraw();
    }

    private void UpdateSubtitle()
    {
        var seconds = (int)_live.Elapsed;
        var time = $"{seconds / 60:00}:{seconds % 60:00}";
        _subtitle.SetText(_live.ActiveStyleCount == 1 ? Loc.F("MusicOneStyle", time) : Loc.F("MusicBlending", _live.ActiveStyleCount.ToString(), time));
    }

    // ---------------------------------------------------------------- clips

    private void SyncClips()
    {
        _syncing = true;
        _clipModels.Select(_vm.MusicWorkflows.ToList().IndexOf(_vm.MusicWorkflow));
        _clipSeconds.SetRange(1, _vm.MusicWorkflow.MaxSeconds);
        if (_clipSeconds.GetValueAsInt() != _vm.MusicSeconds)
            _clipSeconds.SetValue(_vm.MusicSeconds);
        if ((_clipPrompt.GetBuffer().Text ?? "") != _vm.MusicPrompt)
            _clipPrompt.GetBuffer().SetText(_vm.MusicPrompt, -1);
        _syncing = false;
        _clipNote.SetText(_vm.MusicDownloadNote);
        _clipNote.SetVisible(_vm.MusicDownloadNote.Length > 0);
        _clipStatus.SetText(_vm.MusicStatus);
        _clipStatus.SetVisible(_vm.MusicStatus.Length > 0);
        _clipProgress.SetVisible(_vm.IsGeneratingMusic);
        _clipProgress.SetFraction(_vm.MusicProgress);
        _clipGenerate.SetVisible(!_vm.IsGeneratingMusic);
        _clipCancel.SetVisible(_vm.IsGeneratingMusic);
    }

    private void RebuildClips()
    {
        _clips.Clear();
        if (_vm.MusicClips.Count == 0)
            _clips.Append(Ui.Label(Loc.F("MusicNoRecordings", StudioViewModel.MusicFolder), "caption", "dim-label"));
        foreach (var clip in _vm.MusicClips)
        {
            var name = Ui.Label(clip.Name);
            name.SetHexpand(true);
            name.SetWrap(false);
            name.SetEllipsize(Pango.EllipsizeMode.Middle);
            var play = Ui.IconButton("media-playback-start-symbolic", Loc.S("MusicPlay"), () => PlayClip(clip.Path), "flat");
            var folder = Ui.IconButton("folder-open-symbolic", Loc.S("SettingsOpenFolder"), () => Ui.Open(Path.GetDirectoryName(clip.Path)!), "flat");
            _clips.Append(Ui.HBox(8, play, name, folder).Named("MusicClip"));
        }
    }

    /// <summary>Plays a clip (pw-play reads FLAC and WAV); clicking the playing clip again stops it.</summary>
    private async void PlayClip(string path)
    {
        if (_playingClip == path)
        {
            AudioIO.Stop();
            _playingClip = null;
            return;
        }
        _playingClip = path;
        try
        {
            await AudioIO.PlayAsync(path, 0, CancellationToken.None);
        }
        catch (InvalidOperationException e)
        {
            _vm.MusicStatus = Loc.F("ErrorPrefix", e.Message);
        }
        if (_playingClip == path)
            _playingClip = null;
    }

    private static Gtk.Widget Card(Gtk.Widget child)
    {
        var card = Ui.VBox(0, child.Margins(20, 20, 20, 20));
        card.AddCssClass("card");
        return card;
    }
}
