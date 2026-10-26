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
    public ChatPage()
    {
        InitializeComponent();
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
        if (e.NewItems is not null)
            foreach (MessageItem item in e.NewItems)
                item.PropertyChanged += OnMessagePropertyChanged;
        ScrollToEnd();
    }

    private void OnMessagePropertyChanged(object? sender, PropertyChangedEventArgs e)
    {
        // Follow the stream only while the reader is already at the bottom.
        if (Scroller.VerticalOffset >= Scroller.ScrollableHeight - 80)
            ScrollToEnd();
    }

    private void ScrollToEnd() =>
        DispatcherQueue.TryEnqueue(() => Scroller.ChangeView(null, Scroller.ScrollableHeight, null, disableAnimation: true));
}
