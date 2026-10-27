using System.Collections.Specialized;
using System.ComponentModel;
using LLMHub.Services;
using LLMHub.ViewModels;
using Microsoft.UI.Input;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Automation;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Input;
using Windows.ApplicationModel.DataTransfer;
using Windows.Storage;
using Windows.Storage.Pickers;
using Windows.System;
using Windows.UI.Core;

namespace LLMHub.Views;

public sealed partial class ChatPage : Page
{
    /// <summary>Keeps a streaming reply in view: on until the reader scrolls up, back on once they return to the bottom.</summary>
    private bool _follow = true;
    private double _lastOffset;

    public ChatPage()
    {
        InitializeComponent();
        Scroller.ViewChanged += OnScrollerViewChanged;
        Scroller.SizeChanged += OnMessagesResized;
        MessageList.SizeChanged += OnMessagesResized;
        ViewModel.Messages.CollectionChanged += OnMessagesChanged;
        Ioc.Speech.PropertyChanged += OnSpeechChanged;
        Unloaded += (_, _) =>
        {
            ViewModel.Messages.CollectionChanged -= OnMessagesChanged;
            Ioc.Speech.PropertyChanged -= OnSpeechChanged;
        };
    }

    public ChatViewModel ViewModel => Ioc.Chat;

    private void OnComposerKeyDown(object sender, KeyRoutedEventArgs e)
    {
        if (e.Key != VirtualKey.Enter)
            return;
        var shift = InputKeyboardSource.GetKeyStateForCurrentThread(VirtualKey.Shift).HasFlag(CoreVirtualKeyStates.Down);
        if (shift)
            return;
        e.Handled = true;
        if (ViewModel.SendCommand.CanExecute(null))
            ViewModel.SendCommand.Execute(null);
    }

    private void OnGetModel(object sender, RoutedEventArgs e) => App.Window?.ShowSettings("models");

    // ---------------------------------------------------------------- attachments

    private async void OnAttach(object sender, RoutedEventArgs e)
    {
        var picker = new FileOpenPicker { ViewMode = PickerViewMode.List };
        foreach (var ext in Attachments.Extensions)
            picker.FileTypeFilter.Add(ext);
        WinRT.Interop.InitializeWithWindow.Initialize(picker, WinRT.Interop.WindowNative.GetWindowHandle(App.Window));
        var files = await picker.PickMultipleFilesAsync();
        if (files.Count > 0)
            ViewModel.AttachFiles(files.Select(f => f.Path));
    }

    private void OnRemoveAttachment(object sender, RoutedEventArgs e)
    {
        if (sender is FrameworkElement { Tag: Attachment a })
            ViewModel.RemoveAttachmentCommand.Execute(a);
    }

    private void OnDragOver(object sender, DragEventArgs e)
    {
        if (e.DataView.Contains(StandardDataFormats.StorageItems))
            e.AcceptedOperation = DataPackageOperation.Copy;
    }

    private async void OnDrop(object sender, DragEventArgs e)
    {
        if (!e.DataView.Contains(StandardDataFormats.StorageItems))
            return;
        var items = await e.DataView.GetStorageItemsAsync();
        ViewModel.AttachFiles(items.OfType<StorageFile>().Select(f => f.Path));
    }

    // ---------------------------------------------------------------- message actions

    private void OnCopyMessage(object sender, RoutedEventArgs e)
    {
        if (sender is not FrameworkElement { Tag: MessageItem m })
            return;
        var data = new DataPackage();
        data.SetText(m.Content);
        Clipboard.SetContent(data);
        if (sender is Button b)
            ToolTipService.SetToolTip(b, Loc.S("Copied"));
    }

    private async void OnReadAloud(object sender, RoutedEventArgs e)
    {
        if (sender is FrameworkElement { Tag: MessageItem m })
            await Ioc.Speech.ToggleSpeakAsync(m.Id, m.Content);
    }

    private async void OnRegenerate(object sender, RoutedEventArgs e)
    {
        if (sender is FrameworkElement { Tag: MessageItem m })
            await ViewModel.RegenerateCommand.ExecuteAsync(m);
    }

    private async void OnShare(object sender, RoutedEventArgs e)
    {
        var picker = new FileSavePicker { SuggestedFileName = Path.GetFileNameWithoutExtension(ViewModel.ExportFileName()) };
        picker.FileTypeChoices.Add("Markdown", [".md"]);
        WinRT.Interop.InitializeWithWindow.Initialize(picker, WinRT.Interop.WindowNative.GetWindowHandle(App.Window));
        if (await picker.PickSaveFileAsync() is { } file)
            await FileIO.WriteTextAsync(file, ViewModel.ExportMarkdown());
    }

    // ---------------------------------------------------------------- dictation

    private async void OnDictate(object sender, RoutedEventArgs e)
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
                ViewModel.Draft = ViewModel.Draft.Length == 0 ? text : ViewModel.Draft.TrimEnd() + " " + text;
            Composer.Focus(FocusState.Programmatic);
            Composer.SelectionStart = ViewModel.Draft.Length;
        }
        catch (Exception ex)
        {
            Log.Error("Dictation failed", ex);
            ViewModel.AttachError = Loc.F("ErrorPrefix", ex.Message);
        }
    }

    private void OnSpeechChanged(object? sender, PropertyChangedEventArgs e) => DispatcherQueue.TryEnqueue(() =>
    {
        var speech = Ioc.Speech;
        DictateIcon.Glyph = speech.IsRecording ? "" : "";
        var tip = speech.IsRecording ? Loc.S("Listening") : speech.IsTranscribing ? Loc.S("Transcribing") : Loc.S("Dictate");
        ToolTipService.SetToolTip(DictateButton, tip);
        AutomationProperties.SetName(DictateButton, tip);
        DictateButton.IsEnabled = !speech.IsTranscribing;
        if (speech.Status.Length > 0 && (speech.IsRecording || speech.IsTranscribing || speech.SpeakingId is not null))
            ViewModel.AttachError = speech.Status;
        else if (speech.Status.Length == 0 && ViewModel.AttachError.StartsWith(Loc.F("DownloadingWhisper", "").Split('(')[0], StringComparison.Ordinal))
            ViewModel.AttachError = "";
    });

    // ---------------------------------------------------------------- scrolling

    private void OnMessagesChanged(object? sender, NotifyCollectionChangedEventArgs e)
    {
        _follow = true;
        ScrollToEnd();
    }

    // New text changes the height but not the offset, so an offset that moves up means the reader scrolled. Deciding
    // here, rather than per token, keeps a reply that grows by several lines at once (Markdown re-renders in batches)
    // from looking like the reader left the bottom.
    private void OnScrollerViewChanged(object? sender, ScrollViewerViewChangedEventArgs e)
    {
        var offset = Scroller.VerticalOffset;
        if (offset >= Scroller.ScrollableHeight - 40)
            _follow = true;
        else if (offset < _lastOffset - 1)
            _follow = false;
        _lastOffset = offset;
    }

    // After layout, so following scrolls to where the new text actually ends.
    private void OnMessagesResized(object sender, SizeChangedEventArgs e)
    {
        if (_follow)
            ScrollToEnd();
    }

    private void ScrollToEnd() =>
        DispatcherQueue.TryEnqueue(() => Scroller.ChangeView(null, Scroller.ScrollableHeight, null, disableAnimation: true));
}
