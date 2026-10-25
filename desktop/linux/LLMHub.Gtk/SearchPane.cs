using System.Diagnostics;
using LLMHub.Services;
using LLMHub.Services.MediaSearch;
using LLMHub.ViewModels;

namespace LLMHub.Linux;

/// <summary>
/// Studio › Search: photos, audio and video moments found by description or by example with EmbeddingGemma 2, as in the
/// Android app's Photo Search, Audio Search and Video Moment Finder. Also plugs in Linux decoding: GdkPixbuf for photos,
/// GStreamer or ffmpeg for audio and video frames.
/// </summary>
internal sealed class SearchPane
{
    private readonly MainWindow _main;
    private readonly MediaSearchViewModel _vm = Ioc.MediaSearchVm;
    private readonly Gtk.ToggleButton _photosTab = Gtk.ToggleButton.NewWithLabel(Loc.S("SearchPhotos"));
    private readonly Gtk.ToggleButton _audioTab = Gtk.ToggleButton.NewWithLabel(Loc.S("SearchAudio"));
    private readonly Gtk.ToggleButton _videosTab = Gtk.ToggleButton.NewWithLabel(Loc.S("SearchVideos"));
    private readonly Gtk.Label _getModel = Ui.Label(Loc.S("SearchGetModel"), "dim-label");
    private readonly Picker _models;
    private readonly Gtk.Box _modelField;
    private readonly Gtk.Label _folder = Ui.Label("", "caption");
    private readonly Gtk.Button _index;
    private readonly Gtk.Button _stop;
    private readonly Gtk.ProgressBar _progress = Gtk.ProgressBar.New();
    private readonly Gtk.Label _indexed = Ui.Label("", "caption", "dim-label");
    private readonly Gtk.Label _status = Ui.Label("", "caption", "dim-label");
    private readonly Gtk.Label _framesOnly = Ui.Label(Loc.S("SearchVideoFramesOnly"), "caption", "warning");
    private readonly Gtk.Label _videoHelp = Ui.Label(Loc.S("SearchVideoHelp"), "caption", "dim-label");
    private readonly Gtk.SearchEntry _query = Gtk.SearchEntry.New();
    private readonly Gtk.Button _byPhoto;
    private readonly Gtk.FlowBox _photos = Gtk.FlowBox.New();
    private readonly Gtk.ListBox _audio = Gtk.ListBox.New();
    private readonly Gtk.ListBox _videos = Gtk.ListBox.New();
    private readonly Gtk.Video _player = Gtk.Video.New();
    private readonly Gtk.Stack _results = Gtk.Stack.New();
    private bool _syncing;

    public Gtk.Widget Root { get; }

    /// <summary>The pane in the window, for dev runs (--playmoment).</summary>
    public static SearchPane? Instance { get; private set; }

    public SearchPane(MainWindow main)
    {
        _main = main;
        Instance = this;
        _photosTab.SetGroup(null);
        _audioTab.SetGroup(_photosTab);
        _videosTab.SetGroup(_photosTab);
        _photosTab.AddCssClass("pill");
        _audioTab.AddCssClass("pill");
        _videosTab.AddCssClass("pill");
        _photosTab.Named("SearchPhotosTab");
        _audioTab.Named("SearchAudioTab");
        _videosTab.Named("SearchVideosTab");
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
        _videosTab.OnToggled += (_, _) =>
        {
            if (!_syncing && _videosTab.GetActive())
                _vm.Mode = "videos";
        };
        _framesOnly.Named("SearchVideoFramesOnly");
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
            Ui.HBox(6, _photosTab, _audioTab, _videosTab), _getModel, _modelField,
            Ui.Field(Loc.S("SearchFolder"), Ui.HBox(8, _folder, change)),
            Ui.HBox(8, _index, _stop), _progress, _indexed, _status, _framesOnly, _videoHelp,
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
        _videos.AddCssClass("boxed-list");
        _videos.SetSelectionMode(Gtk.SelectionMode.None);
        _videos.SetValign(Gtk.Align.Start);
        _videos.Named("SearchVideoResults");
        _player.SetVisible(false);
        _player.AddCssClass("card");
        _player.SetOverflow(Gtk.Overflow.Hidden);
        _player.SetMarginBottom(12);
        _player.Named("SearchVideoPlayer");
        // The player above the moments, 320 px high to start with; the divider resizes it.
        var videos = Gtk.Paned.New(Gtk.Orientation.Vertical);
        videos.SetStartChild(_player);
        videos.SetEndChild(Ui.Scroll(_videos));
        videos.SetResizeStartChild(false);
        videos.SetShrinkStartChild(true);
        videos.SetPosition(320);
        _results.AddNamed(Ui.Scroll(_photos), "photos");
        _results.AddNamed(Ui.Scroll(_audio), "audio");
        _results.AddNamed(videos, "videos");
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
        _vm.Videos.CollectionChanged += (_, _) => UiThread.Run(RebuildVideos);
        Sync();
    }

    private void Sync()
    {
        _syncing = true;
        _photosTab.SetActive(_vm.IsPhotos);
        _audioTab.SetActive(_vm.IsAudio);
        _videosTab.SetActive(_vm.IsVideos);
        _syncing = false;
        _framesOnly.SetVisible(_vm.VideoFramesOnly);
        _videoHelp.SetVisible(_vm.IsVideos);
        if (!_vm.IsVideos)
            _player.GetMediaStream()?.Pause();
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
        _query.SetPlaceholderText(_vm.Placeholder);
        _byPhoto.SetVisible(_vm.IsPhotos);
        _results.SetVisibleChildName(_vm.Mode);
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

    /// <summary>The best video moments: the frame where each starts, its video and timestamp; choosing one plays it from there.</summary>
    private void RebuildVideos()
    {
        _videos.RemoveAll();
        foreach (var moment in _vm.Videos)
        {
            var row = Adw.ActionRow.New();
            row.SetTitle(moment.Name);
            row.SetSubtitle($"{Loc.F("SearchPlayFrom", moment.Label)} · {moment.ScoreText}");
            row.SetActivatable(true);
            row.Named("SearchVideoMoment");
            var thumb = Gtk.Picture.New();
            thumb.SetSizeRequest(112, 63);
            thumb.SetContentFit(Gtk.ContentFit.Cover);
            thumb.SetValign(Gtk.Align.Center);
            thumb.AddCssClass("gallery-tile");
            thumb.SetOverflow(Gtk.Overflow.Hidden);
            row.AddPrefix(thumb);
            _ = LoadThumbnailAsync(thumb, moment);
            var play = Gtk.Button.New();
            var time = Gtk.Label.New(moment.Label);
            time.AddCssClass("caption");
            play.SetChild(Ui.HBox(6, Gtk.Image.NewFromIconName("media-playback-start-symbolic"), time));
            play.AddCssClass("pill");
            play.SetValign(Gtk.Align.Center);
            play.SetTooltipText(Loc.F("SearchPlayFrom", moment.Label));
            play.Named("SearchVideoTime");
            play.OnClicked += (_, _) => PlayMoment(moment);
            row.OnActivated += (_, _) => PlayMoment(moment);
            row.AddSuffix(play);
            _videos.Append(row);
        }
    }

    private static async Task LoadThumbnailAsync(Gtk.Picture picture, VideoMomentMatch moment)
    {
        var jpeg = await MediaDecoders.VideoThumbnail(moment.Path, moment.StartMs, 224, CancellationToken.None);
        if (jpeg is null)
            return;
        var temp = Path.Combine(Path.GetTempPath(), $"llmhub-thumb-{Guid.NewGuid():N}.jpg");
        try
        {
            await File.WriteAllBytesAsync(temp, jpeg);
            // A picture asks for its image's size, so the frame is scaled to the thumbnail's.
            var pixbuf = GdkPixbuf.Pixbuf.NewFromFileAtScale(temp, 112, 63, true);
            if (pixbuf is not null)
                UiThread.Run(() => picture.SetPaintable(Gdk.Texture.NewForPixbuf(pixbuf)));
        }
        catch (Exception e) when (e is GLib.GException or IOException)
        {
            Log.Info($"Media search: thumbnail of {moment.Path} unreadable: {e.Message}");
        }
        finally
        {
            File.Delete(temp);
        }
    }

    /// <summary>What the moment player is doing, for dev runs.</summary>
    public string PlayerState => _player.GetVisible() && _player.GetMediaStream() is { } stream
        ? $"GTK player at {stream.GetTimestamp() / 1e6:F1}s of {stream.GetDuration() / 1e6:F1}s, playing {stream.GetPlaying()}, muted {stream.GetMuted()}, " +
          $"ended {stream.GetEnded()}, seekable {stream.IsSeekable()}, error {stream.GetError()?.Message ?? "none"}"
        : $"not in the GTK player (last: {_lastPlayer})";

    private string _lastPlayer = "none";

    /// <summary>
    /// Plays the video from the moment's start in GTK's own player; without a GTK media backend (the
    /// gtk4-media-gstreamer module), in mpv, VLC or ffplay at that time, or else from the start in the default app.
    /// Dev runs play muted, and never in another app.
    /// </summary>
    public void PlayMoment(VideoMomentMatch moment, bool muted = false)
    {
        var stream = Gtk.MediaFile.NewForFilename(moment.Path);
        if (stream.GetError() is null)
        {
            _lastPlayer = "GTK";
            stream.SetMuted(muted);
            void Start()
            {
                if (_player.GetMediaStream() != stream)
                    return;
                Log.Info($"Media search: playing {moment.Name} from {moment.Label} in GTK's player{(muted ? ", muted" : "")}");
                // Set again now the player exists (GTK only passes a change on).
                stream.SetMuted(!muted);
                stream.SetMuted(muted);
                stream.Seek(moment.StartMs * 1000L);
                stream.Play();
            }
            _player.GetMediaStream()?.Pause();
            _player.SetMediaStream(stream);
            _player.SetVisible(true);
            stream.OnNotify += (_, e) =>
            {
                if (e.Pspec.GetName() == "prepared" && stream.IsPrepared())
                    Start();
                else if (e.Pspec.GetName() == "error" && stream.GetError() is not null && _player.GetMediaStream() == stream)
                {
                    Log.Info($"Media search: GTK cannot play {moment.Path}: {stream.GetError()?.Message}");
                    _player.SetMediaStream(null);
                    _player.SetVisible(false);
                    _lastPlayer = muted ? "none, muted run" : PlayExternally(moment);
                }
            };
            if (stream.IsPrepared())
                Start();
            return;
        }
        Log.Info($"Media search: no GTK media backend for {moment.Path}: {stream.GetError()?.Message}");
        _lastPlayer = muted ? "none, muted run" : PlayExternally(moment);
    }

    /// <summary>Starts an installed player at the moment and says which one.</summary>
    private static string PlayExternally(VideoMomentMatch moment)
    {
        var seconds = (moment.StartMs / 1000.0).ToString("0.###", System.Globalization.CultureInfo.InvariantCulture);
        foreach (var (tool, args) in new[]
                 {
                     ("mpv", new[] { $"--start={seconds}", moment.Path }),
                     ("vlc", new[] { $"--start-time={seconds}", moment.Path }),
                     ("ffplay", new[] { "-autoexit", "-ss", seconds, moment.Path }),
                 })
        {
            if (Which(tool) is not { } path)
                continue;
            var info = new ProcessStartInfo(path) { UseShellExecute = false };
            foreach (var arg in args)
                info.ArgumentList.Add(arg);
            try
            {
                Process.Start(info);
                return tool;
            }
            catch (System.ComponentModel.Win32Exception e)
            {
                Log.Info($"Media search: {tool} did not start: {e.Message}");
            }
        }
        Ui.Open(moment.Path);
        return "default app";
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
        MediaDecoders.Video = VideoAsync;
        MediaDecoders.VideoThumbnail = VideoThumbnailAsync;
        MediaDecoders.VideoUnavailable = () => Which("ffmpeg") is null && Which("gst-launch-1.0") is null ? Loc.S("SearchNoVideoDecoder") : null;
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

    /// <summary>
    /// WAV directly; other formats (and the sound of videos) through GStreamer (gst-launch-1.0) or ffmpeg, whichever is
    /// installed, trying the other when the first cannot decode the file (GStreamer without libav has no H.264 or AAC).
    /// </summary>
    private static async Task<float[]?> AudioAsync(string path, int maxSeconds, CancellationToken ct)
    {
        if (MediaDecoders.ReadWav(path, maxSeconds) is { } wav)
            return wav;
        var temp = Path.Combine(Path.GetTempPath(), $"llmhub-audio-{Guid.NewGuid():N}.wav");
        try
        {
            var tried = false;
            if (Which("gst-launch-1.0") is { } gst)
            {
                tried = true;
                if (await RunAsync(gst, ["-q", "filesrc", $"location={path}", "!", "decodebin", "!", "audioconvert", "!", "audioresample", "!",
                        $"audio/x-raw,format=S16LE,channels=1,rate={MediaDecoders.SampleRate}", "!", "wavenc", "!", "filesink", $"location={temp}"], ct) == 0
                    && MediaDecoders.ReadWav(temp, maxSeconds) is { } decoded)
                    return decoded;
            }
            if (Which("ffmpeg") is { } ffmpeg)
            {
                tried = true;
                if (await RunAsync(ffmpeg, ["-v", "error", "-y", "-i", path, "-t", maxSeconds.ToString(), "-vn", "-ac", "1", "-ar", MediaDecoders.SampleRate.ToString(), temp], ct) == 0)
                    return MediaDecoders.ReadWav(temp, maxSeconds);
            }
            if (!tried)
                Log.Info($"Media search: no audio decoder for {path} (install gstreamer1.0-plugins-good or ffmpeg)");
            return null;
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

    /// <summary>Runs a decoder to the end and returns its exit code (its last error line is logged on failure).</summary>
    private static async Task<int> RunAsync(string tool, IEnumerable<string> args, CancellationToken ct) => (await RunWithLogAsync(tool, args, ct)).Code;

    // ------------------------------------------------ video

    /// <summary>
    /// A video's frames, one per second, decoded in one pass into a temporary folder: ffmpeg scales them to maxEdge itself;
    /// GStreamer writes them full size and GdkPixbuf scales each one as it is read. A frame at t is the one nearest t.
    /// </summary>
    private static async Task<VideoFrames?> VideoAsync(string path, int maxEdge, int maxSeconds, CancellationToken ct)
    {
        var folder = Directory.CreateTempSubdirectory("llmhub-frames-").FullName;
        try
        {
            if (Which("ffmpeg") is { } ffmpeg)
            {
                var (code, log) = await RunWithLogAsync(ffmpeg, ["-hide_banner", "-nostats", "-y", "-i", path, "-t", maxSeconds.ToString(), "-an",
                    "-vf", $"fps=1,scale='min({maxEdge},iw)':'min({maxEdge},ih)':force_original_aspect_ratio=decrease", "-q:v", "4",
                    Path.Combine(folder, "%05d.jpg")], ct);
                var frames = Directory.GetFiles(folder, "*.jpg").Length;
                if (code == 0 && frames > 0)
                    return new LinuxVideo(folder, Math.Min(ParseDuration(log) ?? frames * 1000, maxSeconds * 1000), frames, null);
            }
            if (Which("gst-launch-1.0") is { } gst)
            {
                foreach (var file in Directory.GetFiles(folder))
                    File.Delete(file);
                // drop-only: no frame is repeated at the end, so the count says how long the video is.
                var code = await RunAsync(gst, ["-q", "filesrc", $"location={path}", "!", "decodebin", "!", "videoconvert", "!", "videorate", "drop-only=true", "!",
                    "video/x-raw,framerate=1/1", "!", "identity", $"eos-after={maxSeconds + 1}", "!", "jpegenc", "quality=90", "!",
                    "multifilesink", $"location={Path.Combine(folder, "%05d.jpg")}", "index=1"], ct);
                var frames = Directory.GetFiles(folder, "*.jpg").Length;
                if (code == 0 && frames > 0)
                    // The frame at n s exists for n up to the length, so the length is at least (frames - 1) s.
                    return new LinuxVideo(folder, Math.Min((frames - 1) * 1000, maxSeconds * 1000), frames, maxEdge);
            }
            Log.Info($"Media search: could not decode video {path} (install ffmpeg, or GStreamer with the codec)");
        }
        catch (Exception e) when (e is not OperationCanceledException)
        {
            Log.Info($"Media search: could not decode video {path}: {e.Message}");
        }
        Directory.Delete(folder, true);
        return null;
    }

    /// <summary>One frame for a result thumbnail: a quick seek with ffmpeg, else the frames up to it with GStreamer.</summary>
    private static async Task<byte[]?> VideoThumbnailAsync(string path, int timeMs, int maxEdge, CancellationToken ct)
    {
        if (Which("ffmpeg") is not { } ffmpeg)
        {
            using var video = await VideoAsync(path, maxEdge, timeMs / 1000 + 1, ct);
            return video is null ? null : (await video.FramesAsync([timeMs], ct))[0];
        }
        var temp = Path.Combine(Path.GetTempPath(), $"llmhub-frame-{Guid.NewGuid():N}.jpg");
        try
        {
            var seconds = (timeMs / 1000.0).ToString("0.###", System.Globalization.CultureInfo.InvariantCulture);
            return await RunAsync(ffmpeg, ["-v", "error", "-y", "-ss", seconds, "-i", path, "-frames:v", "1",
                       "-vf", $"scale='min({maxEdge},iw)':'min({maxEdge},ih)':force_original_aspect_ratio=decrease", "-q:v", "4", temp], ct) == 0 && File.Exists(temp)
                ? await File.ReadAllBytesAsync(temp, ct)
                : null;
        }
        finally
        {
            File.Delete(temp);
        }
    }

    private static async Task<(int Code, string Log)> RunWithLogAsync(string tool, IEnumerable<string> args, CancellationToken ct)
    {
        var info = new ProcessStartInfo(tool) { UseShellExecute = false, RedirectStandardError = true, RedirectStandardOutput = true };
        foreach (var arg in args)
            info.ArgumentList.Add(arg);
        using var process = Process.Start(info)!;
        var errors = process.StandardError.ReadToEndAsync(CancellationToken.None);
        _ = process.StandardOutput.ReadToEndAsync(CancellationToken.None);
        try
        {
            await process.WaitForExitAsync(ct);
        }
        catch (OperationCanceledException)
        {
            process.Kill(true);
            throw;
        }
        var log = await errors;
        if (process.ExitCode != 0)
            Log.Info($"Media search: {Path.GetFileName(tool)} exited with {process.ExitCode}: {log.Trim().Split('\n').LastOrDefault()}");
        return (process.ExitCode, log);
    }

    /// <summary>"Duration: 00:01:02.50," from ffmpeg's description of its input, in milliseconds.</summary>
    private static int? ParseDuration(string log)
    {
        var match = System.Text.RegularExpressions.Regex.Match(log, @"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)");
        if (!match.Success)
            return null;
        return (int)((int.Parse(match.Groups[1].Value) * 3600 + int.Parse(match.Groups[2].Value) * 60) * 1000
                     + double.Parse(match.Groups[3].Value, System.Globalization.CultureInfo.InvariantCulture) * 1000);
    }

    /// <summary>Frames decoded at one per second (file n is the frame at n-1 seconds); deleted with the video.</summary>
    private sealed class LinuxVideo(string folder, int durationMs, int count, int? rescaleTo) : VideoFrames
    {
        public override int DurationMs => durationMs;

        public override Task<byte[]?[]> FramesAsync(IReadOnlyList<int> timesMs, CancellationToken ct) => Task.Run(() =>
            timesMs.Select(t => Frame(Math.Clamp((int)Math.Round(t / 1000.0), 0, count - 1))).ToArray(), ct);

        private byte[]? Frame(int index)
        {
            var file = Path.Combine(folder, $"{index + 1:00000}.jpg");
            try
            {
                if (rescaleTo is not { } edge)
                    return File.ReadAllBytes(file);
                // GStreamer's frames are full size: scale them like ffmpeg's, JPEG quality 80 as on Android.
                var pixbuf = GdkPixbuf.Pixbuf.NewFromFileAtScale(file, edge, edge, true);
                if (pixbuf is null)
                    return null;
                var temp = file + ".small.jpg";
                pixbuf.Savev(temp, "jpeg", ["quality"], ["80"]);
                var bytes = File.ReadAllBytes(temp);
                File.Delete(temp);
                return bytes;
            }
            catch (Exception e) when (e is GLib.GException or IOException)
            {
                Log.Info($"Media search: frame {file} unreadable: {e.Message}");
                return null;
            }
        }

        public override void Dispose()
        {
            try
            {
                Directory.Delete(folder, true);
            }
            catch (IOException)
            {
            }
        }
    }

    private static string? Which(string tool) =>
        (Environment.GetEnvironmentVariable("PATH") ?? "").Split(Path.PathSeparator).Select(d => Path.Combine(d, tool)).FirstOrDefault(File.Exists);
}
