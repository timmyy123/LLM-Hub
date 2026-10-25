using LLMHub.Services;
using LLMHub.Services.MediaSearch;
using LLMHub.Services.Studio;
using LLMHub.ViewModels;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Controls.Primitives;
using Windows.Storage.Pickers;
using Windows.System;

namespace LLMHub.Views;

public sealed partial class StudioPage : Page
{
    public StudioPage()
    {
        InitializeComponent();
        UpdateTabs();
        Loaded += (_, _) => UpdateSearchTabs();
        CountPicker.SelectedItem = ViewModel.Count;
        StepsBox.Value = ViewModel.Steps;
        MusicSecondsBox.Value = ViewModel.MusicSeconds;
        Unloaded += (_, _) => _player?.Dispose();
        // On a narrow window "Open in ComfyUI" keeps only its icon so the action row fits.
        SizeChanged += (_, e) => OpenComfyLabel.Visibility = e.NewSize.Width >= 1080 ? Visibility.Visible : Visibility.Collapsed;
        ViewModel.PropertyChanged += (_, e) =>
        {
            if (e.PropertyName == nameof(StudioViewModel.Steps) && (int)StepsBox.Value != ViewModel.Steps)
                StepsBox.Value = ViewModel.Steps;
        MusicSecondsBox.Value = ViewModel.MusicSeconds;
        Unloaded += (_, _) => _player?.Dispose();
        };
    }

    private void OnCountChanged(object sender, SelectionChangedEventArgs e)
    {
        if (CountPicker.SelectedItem is int count)
            ViewModel.Count = count;
    }

    private void OnStepsChanged(NumberBox sender, NumberBoxValueChangedEventArgs args)
    {
        if (!double.IsNaN(args.NewValue))
            ViewModel.Steps = (int)args.NewValue;
    }

    public StudioViewModel ViewModel => Ioc.StudioVm;
    public MediaSearchViewModel Search => Ioc.MediaSearchVm;

    private void OnSearchModeClick(object sender, RoutedEventArgs e)
    {
        if (sender is FrameworkElement { Tag: string mode })
            Search.Mode = mode;
        UpdateSearchTabs();
        if (!Search.IsVideos)
            _momentPlayer?.Pause();
        Bindings.Update();
    }

    /// <summary>The Photos, Audio and Videos switch follows the view model (dev runs change the mode before the page opens).</summary>
    private void UpdateSearchTabs()
    {
        SearchPhotosTab.IsChecked = Search.IsPhotos;
        SearchAudioTab.IsChecked = Search.IsAudio;
        SearchVideosTab.IsChecked = Search.IsVideos;
    }

    private async void OnSearchFolder(object sender, RoutedEventArgs e)
    {
        var picker = new FolderPicker
        {
            SuggestedStartLocation = Search.Mode switch { "photos" => PickerLocationId.PicturesLibrary, "audio" => PickerLocationId.MusicLibrary, _ => PickerLocationId.VideosLibrary },
        };
        picker.FileTypeFilter.Add("*");
        WinRT.Interop.InitializeWithWindow.Initialize(picker, WinRT.Interop.WindowNative.GetWindowHandle(App.Window));
        if (await picker.PickSingleFolderAsync() is { } folder)
            Search.SetFolder(folder.Path);
    }

    private void OnSearchKeyDown(object sender, Microsoft.UI.Xaml.Input.KeyRoutedEventArgs e)
    {
        if (e.Key != Windows.System.VirtualKey.Enter)
            return;
        e.Handled = true;
        Search.SearchCommand.Execute(null);
    }

    private async void OnSearchByPhoto(object sender, RoutedEventArgs e)
    {
        var picker = new FileOpenPicker { ViewMode = PickerViewMode.Thumbnail, SuggestedStartLocation = PickerLocationId.PicturesLibrary };
        foreach (var ext in MediaDecoders.ImageExtensions)
            picker.FileTypeFilter.Add(ext);
        WinRT.Interop.InitializeWithWindow.Initialize(picker, WinRT.Interop.WindowNative.GetWindowHandle(App.Window));
        if (await picker.PickSingleFileAsync() is { } file)
            await Search.SearchByPhotoAsync(file.Path);
    }

    private async void OnPhotoMatchClick(object sender, ItemClickEventArgs e)
    {
        if (e.ClickedItem is PhotoMatch match)
            await Windows.System.Launcher.LaunchFileAsync(await Windows.Storage.StorageFile.GetFileFromPathAsync(match.Path));
    }

    private async void OnPlayMoment(object sender, RoutedEventArgs e)
    {
        if (sender is not FrameworkElement { Tag: AudioMoment moment })
            return;
        var match = Search.Audio.FirstOrDefault(a => a.Moments.Contains(moment));
        if (match is not null)
            await Search.PlayMomentAsync(match, moment);
    }

    // ---------------------------------------------------------------- Search › Videos

    private Windows.Media.Playback.MediaPlayer? _momentPlayer;
    private string? _momentPath;
    private int _momentStartMs;

    private async void OnVideoMomentClick(object sender, ItemClickEventArgs e)
    {
        if (e.ClickedItem is not VideoMomentMatch moment)
            return;
        try
        {
            await PlayMomentAsync(moment);
        }
        catch (Exception ex)
        {
            Log.Error($"Could not play {moment.Path}", ex);
            Search.Status = Loc.F("ErrorPrefix", ex.Message);
        }
    }

    /// <summary>Where the moment player is and whether it is muted (dev runs check it started at the moment).</summary>
    internal (TimeSpan? Position, bool Muted) MomentState => (_momentPlayer?.PlaybackSession.Position, _momentPlayer?.IsMuted ?? false);

    /// <summary>Plays the moment's video in the player above the results, from the moment's start (silently for dev runs).</summary>
    internal async Task PlayMomentAsync(VideoMomentMatch moment, bool muted = false)
    {
        if (!File.Exists(moment.Path))
            return;
        if (_momentPlayer is null)
        {
            _momentPlayer = new Windows.Media.Playback.MediaPlayer { AutoPlay = false };
            // The position can only be set once the video is open.
            _momentPlayer.MediaOpened += (player, _) =>
            {
                player.PlaybackSession.Position = TimeSpan.FromMilliseconds(_momentStartMs);
                player.Play();
            };
            SearchVideoPlayer.SetMediaPlayer(_momentPlayer);
            Unloaded += (_, _) => _momentPlayer?.Pause();
        }
        SearchVideoStage.Visibility = Visibility.Visible;
        _momentStartMs = moment.StartMs;
        _momentPlayer.IsMuted = muted;
        if (_momentPath == moment.Path)
        {
            _momentPlayer.PlaybackSession.Position = TimeSpan.FromMilliseconds(moment.StartMs);
            _momentPlayer.Play();
            return;
        }
        _momentPath = moment.Path;
        _momentPlayer.Source = Windows.Media.Core.MediaSource.CreateFromStorageFile(await Windows.Storage.StorageFile.GetFileFromPathAsync(moment.Path));
    }

    /// <summary>Each result shows the frame where its moment starts.</summary>
    private async void OnVideoMomentContainer(ListViewBase sender, ContainerContentChangingEventArgs args)
    {
        if (args.InRecycleQueue || args.Item is not VideoMomentMatch moment || args.ItemContainer.ContentTemplateRoot is not Grid root
            || root.Children.FirstOrDefault() is not Grid thumb || thumb.Children.FirstOrDefault() is not Microsoft.UI.Xaml.Controls.Image image)
            return;
        image.Source = null;
        image.Tag = moment;
        try
        {
            var jpeg = await MediaDecoders.VideoThumbnail(moment.Path, moment.StartMs, 320, CancellationToken.None);
            // The container may show another moment by the time the frame is ready.
            if (jpeg is null || !ReferenceEquals(image.Tag, moment))
                return;
            var bitmap = new Microsoft.UI.Xaml.Media.Imaging.BitmapImage();
            using var stream = new Windows.Storage.Streams.InMemoryRandomAccessStream();
            await stream.WriteAsync(System.Runtime.InteropServices.WindowsRuntime.WindowsRuntimeBufferExtensions.AsBuffer(jpeg));
            stream.Seek(0);
            await bitmap.SetSourceAsync(stream);
            if (ReferenceEquals(image.Tag, moment))
                image.Source = bitmap;
        }
        catch (Exception ex)
        {
            Log.Info($"Media search: no thumbnail for {moment.Path} at {moment.Label}: {ex.Message}");
        }
    }

    private void OnModeClick(object sender, RoutedEventArgs e)
    {
        if (sender is FrameworkElement { Tag: string mode })
            ViewModel.Mode = mode;
        UpdateTabs();
    }

    private void UpdateTabs()
    {
        foreach (var tab in new ToggleButton[] { TabImage, TabVideo, TabMusic, TabVoice, TabSearch })
            tab.IsChecked = (string)tab.Tag == ViewModel.Mode;
    }

    private async void OnPickReference(object sender, RoutedEventArgs e)
    {
        var picker = new FileOpenPicker { ViewMode = PickerViewMode.Thumbnail, SuggestedStartLocation = PickerLocationId.PicturesLibrary };
        foreach (var ext in new[] { ".png", ".jpg", ".jpeg", ".webp" })
            picker.FileTypeFilter.Add(ext);
        WinRT.Interop.InitializeWithWindow.Initialize(picker, WinRT.Interop.WindowNative.GetWindowHandle(App.Window));
        if (await picker.PickSingleFileAsync() is { } file)
            ViewModel.ReferenceImage = file.Path;
    }

    /// <summary>Opens the selected picture or video in its default Windows app.</summary>
    private async void OnOpenItem(object sender, RoutedEventArgs e)
    {
        if (ViewModel.SelectedItem is not { } item || !File.Exists(item.Path))
            return;
        await Launcher.LaunchFileAsync(await Windows.Storage.StorageFile.GetFileFromPathAsync(item.Path));
    }

    private void OnUseInChat(object sender, RoutedEventArgs e)
    {
        if (ViewModel.SelectedItem is not { } item)
            return;
        Ioc.Chat.AttachFiles([item.Path]);
        App.Window?.ShowChat();
    }

    private void OnAnimate(object sender, RoutedEventArgs e)
    {
        if (ViewModel.SelectedItem is not { } item)
            return;
        ViewModel.Mode = "video";
        ViewModel.Workflow = ViewModel.Workflows.FirstOrDefault(w => w.Kind == "video") ?? ComfyWorkflow.LtxVideo;
        ViewModel.ReferenceImage = item.Path;
        UpdateTabs();
    }

    /// <summary>Snapdragon NPU: the selected picture becomes the reference of the current image model (image to image).</summary>
    private void OnImageToImage(object sender, RoutedEventArgs e)
    {
        if (ViewModel.SelectedItem is not { } item)
            return;
        ViewModel.Mode = "image";
        ViewModel.ReferenceImage = item.Path;
        UpdateTabs();
    }

    private void OnUpscale(object sender, RoutedEventArgs e)
    {
        if (ViewModel.SelectedItem is not { } item)
            return;
        ViewModel.Mode = "image";
        ViewModel.Workflow = ComfyWorkflow.Upscale;
        ViewModel.ReferenceImage = item.Path;
        UpdateTabs();
    }

    private Windows.Media.Playback.MediaPlayer? _player;

    private void OnMusicSecondsChanged(NumberBox sender, NumberBoxValueChangedEventArgs args)
    {
        if (!double.IsNaN(args.NewValue))
            ViewModel.MusicSeconds = Math.Clamp((int)args.NewValue, 1, ViewModel.MusicWorkflow.MaxSeconds);
    }

    /// <summary>Plays a clip (ComfyUI saves FLAC, which Windows' media player handles); clicking again stops it.</summary>
    private void OnPlayClip(object sender, RoutedEventArgs e)
    {
        if (sender is not FrameworkElement { Tag: string path })
            return;
        var playing = _player?.Source is Windows.Media.Core.MediaSource { Uri: { } current } && current.LocalPath == path
                      && _player.PlaybackSession.PlaybackState == Windows.Media.Playback.MediaPlaybackState.Playing;
        _player?.Pause();
        if (playing)
            return;
        _player ??= new Windows.Media.Playback.MediaPlayer();
        _player.Source = Windows.Media.Core.MediaSource.CreateFromUri(new Uri(path));
        _player.Play();
    }

    private async void OnShowClip(object sender, RoutedEventArgs e)
    {
        if (sender is FrameworkElement { Tag: string path })
            await Launcher.LaunchFolderPathAsync(Path.GetDirectoryName(path)!);
    }

    private async void OnOpenComfy(object sender, RoutedEventArgs e)
    {
        if (ViewModel.Comfy.Url is { } url)
            await Launcher.LaunchUriAsync(new Uri(url));
    }
}
