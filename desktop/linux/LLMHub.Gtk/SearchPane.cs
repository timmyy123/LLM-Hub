using System.Diagnostics;
using LLMHub.Services;
using LLMHub.Services.MediaSearch;
using LLMHub.ViewModels;

namespace LLMHub.Linux;

/// <summary>
/// Studio › Search: photos and audio found by description or by example with EmbeddingGemma 2, as in the Android app's
/// Photo Search and Audio Search. Also plugs in Linux decoding: GdkPixbuf for photos, GStreamer (or ffmpeg) for audio.
/// </summary>
internal sealed class SearchPane
{
    private readonly MainWindow _main;
    private readonly MediaSearchViewModel _vm = Ioc.MediaSearchVm;
    private readonly Gtk.ToggleButton _photosTab = Gtk.ToggleButton.NewWithLabel(Loc.S("SearchPhotos"));
    private readonly Gtk.ToggleButton _audioTab = Gtk.ToggleButton.NewWithLabel(Loc.S("SearchAudio"));
    private readonly Gtk.Label _getModel = Ui.Label(Loc.S("SearchGetModel"), "dim-label");
    private readonly Picker _models;
    private readonly Gtk.Box _modelField;
    private readonly Gtk.Label _folder = Ui.Label("", "caption");
    private readonly Gtk.Button _index;
    private readonly Gtk.Button _stop;
    private readonly Gtk.ProgressBar _progress = Gtk.ProgressBar.New();
    private readonly Gtk.Label _indexed = Ui.Label("", "caption", "dim-label");
    private readonly Gtk.Label _status = Ui.Label("", "caption", "dim-label");
    private readonly Gtk.SearchEntry _query = Gtk.SearchEntry.New();
    private readonly Gtk.Button _byPhoto;
    private readonly Gtk.FlowBox _photos = Gtk.FlowBox.New();
    private readonly Gtk.ListBox _audio = Gtk.ListBox.New();
    private readonly Gtk.Stack _results = Gtk.Stack.New();
    private bool _syncing;

    public Gtk.Widget Root { get; }

    public SearchPane(MainWindow main)
    {
        _main = main;
        _photosTab.SetGroup(null);
        _audioTab.SetGroup(_photosTab);
        _photosTab.AddCssClass("pill");
        _audioTab.AddCssClass("pill");
        _photosTab.Named("SearchPhotosTab");
        _audioTab.Named("SearchAudioTab");
        _photosTab.OnToggled += (_, _) =>
        {
            if (!_syncing && _photosTab.GetActive())
                _vm.Mode = "photos";
        };
        _audioTab.OnToggled += (_, _) =>
        {
            if (!_syncing && _audioTab.GetActive())
                _vm.Mode = "audio";
        };
        _getModel.Named("SearchGetModel");
        _models = new Picker([], i => _vm.SelectedModel = _vm.Models[i]);
        _models.DropDown.Named("SearchModelPicker");
        _modelField = Ui.Field(Loc.S("SearchModel"), _models.DropDown);
        _folder.SetWrap(false);
        _folder.SetEllipsize(Pango.EllipsizeMode.Middle);
        _folder.SetHexpand(true);
        _folder.Named("SearchFolder");
        var change = Ui.Button(Loc.S("SearchChange"), PickFolder);
        _index = Ui.Button(Loc.S("SearchIndex"), () => _vm.IndexCommand.Execute(null), "suggested-action").Named("SearchIndexButton");
        _index.SetHexpand(true);
        _stop = Ui.Button(Loc.S("Stop"), () => _vm.StopIndexingCommand.Execute(null));
        _indexed.Named("SearchIndexed");
        _status.Named("SearchStatus");
        var left = Ui.VBox(16,
            Ui.HBox(6, _photosTab, _audioTab), _getModel, _modelField,
            Ui.Field(Loc.S("SearchFolder"), Ui.HBox(8, _folder, change)),
            Ui.HBox(8, _index, _stop), _progress, _indexed, _status,
            Ui.Label(Loc.S("SearchHelp"), "caption", "dim-label")).Margins(20, 20, 20, 20);
        var leftScroll = Ui.Scroll(left);
        leftScroll.SetSizeRequest(320, -1);
        leftScroll.AddCssClass("divider-right");

        _query.SetHexpand(true);
        _query.Named("SearchQueryBox");
        _query.OnActivate += (_, _) =>
        {
            _vm.Query = _query.GetText();
            _vm.SearchCommand.Execute(null);
        };
        _byPhoto = Ui.Button(Loc.S("SearchByPhoto"), PickPhoto).Named("SearchByPhotoButton");
        _photos.SetSelectionMode(Gtk.SelectionMode.None);
        _photos.SetHomogeneous(true);
        _photos.SetMaxChildrenPerLine(8);
        _photos.SetColumnSpacing(10);
        _photos.SetRowSpacing(10);
        _photos.SetValign(Gtk.Align.Start);
        _photos.Named("SearchPhotoResults");
        _audio.AddCssClass("boxed-list");
        _audio.SetSelectionMode(Gtk.SelectionMode.None);
        _audio.SetValign(Gtk.Align.Start);
        _audio.Named("SearchAudioResults");
        _results.AddNamed(Ui.Scroll(_photos), "photos");
        _results.AddNamed(Ui.Scroll(_audio), "audio");
        _results.SetVexpand(true);
        var right = Ui.VBox(14, Ui.HBox(8, _query, Ui.IconButton("system-search-symbolic", Loc.S("StudioSearch"), () =>
        {
            _vm.Query = _query.GetText();
            _vm.SearchCommand.Execute(null);
        }), _byPhoto), _results).Margins(20, 20, 20, 20);
        right.SetHexpand(true);
        Root = Ui.HBox(0, leftScroll, right);

        Ui.Watch(_vm, _ => Sync());
        _vm.Models.CollectionChanged += (_, _) => UiThread.Run(Sync);
        _vm.Photos.CollectionChanged += (_, _) => UiThread.Run(RebuildPhotos);
        _vm.Audio.CollectionChanged += (_, _) => UiThread.Run(RebuildAudio);
        Sync();
    }

    private void Sync()
    {
        _syncing = true;
        _photosTab.SetActive(_vm.IsPhotos);
        _audioTab.SetActive(_vm.IsAudio);
        _syncing = false;
        _models.SetItems(_vm.Models.Select(m => m.Name), _vm.SelectedModel is { } m ? _vm.Models.IndexOf(m) : -1);
        _getModel.SetVisible(!_vm.HasModel);
        _modelField.SetVisible(_vm.HasModel);
        _folder.SetText(_vm.Folder);
        _folder.SetTooltipText(_vm.Folder);
        _index.SetSensitive(!_vm.IsIndexing && _vm.HasModel);
        _stop.SetVisible(_vm.IsIndexing);
        _progress.SetVisible(_vm.IsIndexing);
        _progress.SetFraction(_vm.Progress);
        _indexed.SetText(_vm.IndexedText);
        _status.SetText(_vm.Status);
        _status.SetVisible(_vm.Status.Length > 0);
        _query.SetPlaceholderText(Loc.S(_vm.IsPhotos ? "SearchPlaceholderPhotos" : "SearchPlaceholderAudio"));
        _byPhoto.SetVisible(_vm.IsPhotos);
        _results.SetVisibleChildName(_vm.IsPhotos ? "photos" : "audio");
    }

    private void RebuildPhotos()
    {
        _photos.RemoveAll();
        foreach (var match in _vm.Photos)
        {
            Gtk.Widget image;
            try
            {
                var pixbuf = GdkPixbuf.Pixbuf.NewFromFileAtScale(match.Path, 200, 150, true)!;
                var picture = Gtk.Picture.NewForPaintable(Gdk.Texture.NewForPixbuf(pixbuf));
                picture.SetContentFit(Gtk.ContentFit.Cover);
                image = picture;
            }
            catch (GLib.GException)
            {
                image = Gtk.Image.NewFromIconName("image-x-generic-symbolic");
            }
            image.SetSizeRequest(200, 150);
            var score = Ui.Label(match.ScoreText, "caption", "dim-label").Named("SearchPhotoScore");
            var tile = Ui.VBox(4, image, score);
            tile.AddCssClass("gallery-tile");
            tile.SetOverflow(Gtk.Overflow.Hidden);
            tile.SetTooltipText(match.Name);
            var click = Gtk.GestureClick.New();
            click.OnReleased += (_, _) => Ui.Open(match.Path);
            tile.AddController(click);
            _photos.Append(tile);
        }
    }

    private void RebuildAudio()
    {
        _audio.RemoveAll();
        foreach (var match in _vm.Audio)
        {
            var row = Adw.ActionRow.New();
            row.SetTitle(match.Name);
            row.SetSubtitle(match.ScoreText);
            row.Named("SearchAudioRow");
            var moments = Ui.HBox(6);
            foreach (var moment in match.Moments)
            {
                var button = Gtk.Button.New();
                button.SetChild(Ui.HBox(6, Gtk.Image.NewFromIconName("media-playback-start-symbolic"), Ui.Label(moment.Label, "caption")));
                button.AddCssClass("pill");
                button.SetValign(Gtk.Align.Center);
                button.SetTooltipText(Loc.S("SearchPlay"));
                button.Named("SearchMoment");
                button.OnClicked += async (_, _) => await _vm.PlayMomentAsync(match, moment);
                moments.Append(button);
            }
            row.AddSuffix(moments);
            _audio.Append(row);
        }
    }

    private async void PickFolder()
    {
        try
        {
            if (await Gtk.FileDialog.New().SelectFolderAsync(_main.Window) is { } folder && folder.GetPath() is { } path)
                _vm.SetFolder(path);
        }
        catch (GLib.GException)
        {
            // Dismissed.
        }
    }

    private async void PickPhoto()
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
                await _vm.SearchByPhotoAsync(path);
        }
        catch (GLib.GException)
        {
            // Dismissed.
        }
    }

    // ---------------------------------------------------------------- Linux decoding

    public static void RegisterDecoders()
    {
        MediaDecoders.Image = ImageAsync;
        MediaDecoders.Audio = AudioAsync;
    }

    /// <summary>GdkPixbuf reads the photo already scaled to maxEdge (it knows every format the desktop's loaders do).</summary>
    private static Task<byte[]?> ImageAsync(string path, int maxEdge, CancellationToken ct) => Task.Run(() =>
    {
        var temp = Path.Combine(Path.GetTempPath(), $"llmhub-photo-{Guid.NewGuid():N}.jpg");
        try
        {
            var pixbuf = GdkPixbuf.Pixbuf.NewFromFileAtScale(path, maxEdge, maxEdge, true);
            if (pixbuf is null)
                return null;
            pixbuf = pixbuf.ApplyEmbeddedOrientation() ?? pixbuf;
            pixbuf.Savev(temp, "jpeg", ["quality"], ["90"]);
            return File.ReadAllBytes(temp);
        }
        catch (Exception e) when (e is GLib.GException or IOException)
        {
            Log.Info($"Media search: could not read image {path}: {e.Message}");
            return null;
        }
        finally
        {
            try
            {
                File.Delete(temp);
            }
            catch (IOException)
            {
            }
        }
    }, ct);

    /// <summary>WAV directly; other formats through GStreamer (gst-launch-1.0) or ffmpeg, whichever is installed.</summary>
    private static async Task<float[]?> AudioAsync(string path, int maxSeconds, CancellationToken ct)
    {
        if (MediaDecoders.ReadWav(path, maxSeconds) is { } wav)
            return wav;
        var temp = Path.Combine(Path.GetTempPath(), $"llmhub-audio-{Guid.NewGuid():N}.wav");
        try
        {
            ProcessStartInfo? info = null;
            if (Which("gst-launch-1.0") is { } gst)
            {
                info = new ProcessStartInfo(gst);
                foreach (var arg in new[] { "-q", "filesrc", $"location={path}", "!", "decodebin", "!", "audioconvert", "!", "audioresample", "!",
                             $"audio/x-raw,format=S16LE,channels=1,rate={MediaDecoders.SampleRate}", "!", "wavenc", "!", "filesink", $"location={temp}" })
                    info.ArgumentList.Add(arg);
            }
            else if (Which("ffmpeg") is { } ffmpeg)
            {
                info = new ProcessStartInfo(ffmpeg);
                foreach (var arg in new[] { "-v", "error", "-y", "-i", path, "-t", maxSeconds.ToString(), "-ac", "1", "-ar", MediaDecoders.SampleRate.ToString(), temp })
                    info.ArgumentList.Add(arg);
            }
            if (info is null)
            {
                Log.Info($"Media search: no audio decoder for {path} (install gstreamer1.0-plugins-good or ffmpeg)");
                return null;
            }
            info.UseShellExecute = false;
            info.RedirectStandardError = true;
            using var process = Process.Start(info)!;
            await process.WaitForExitAsync(ct);
            return process.ExitCode == 0 ? MediaDecoders.ReadWav(temp, maxSeconds) : null;
        }
        catch (Exception e) when (e is not OperationCanceledException)
        {
            Log.Info($"Media search: could not decode audio {path}: {e.Message}");
            return null;
        }
        finally
        {
            try
            {
                File.Delete(temp);
            }
            catch (IOException)
            {
            }
        }
    }

    private static string? Which(string tool) =>
        (Environment.GetEnvironmentVariable("PATH") ?? "").Split(Path.PathSeparator).Select(d => Path.Combine(d, tool)).FirstOrDefault(File.Exists);
}
