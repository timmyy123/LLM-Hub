using System.Collections.Specialized;
using System.ComponentModel;
using LLMHub.Services;
using LLMHub.Services.Agent;
using LLMHub.ViewModels;
using Microsoft.UI.Input;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Input;
using Windows.Storage.Pickers;
using Windows.System;
using Windows.UI.Core;

namespace LLMHub.Views;

public sealed partial class AgentPage : Page
{
    private AgentSession? _watched;

    public AgentPage()
    {
        InitializeComponent();
        ViewModel.PropertyChanged += OnViewModelChanged;
        Watch(ViewModel.Selected);
        Unloaded += (_, _) =>
        {
            ViewModel.PropertyChanged -= OnViewModelChanged;
            Watch(null);
        };
    }

    public AgentViewModel ViewModel => Ioc.AgentVm;

    /// <summary>Opens the folder picker and starts a session there (also used by the sidebar's New session button).</summary>
    public static async Task PickFolderAndStartAsync()
    {
        var picker = new FolderPicker();
        picker.FileTypeFilter.Add("*");
        WinRT.Interop.InitializeWithWindow.Initialize(picker, WinRT.Interop.WindowNative.GetWindowHandle(App.Window));
        if (await picker.PickSingleFolderAsync() is { } folder)
            Ioc.AgentVm.NewSession(folder.Path);
    }

    private async void OnPickFolder(object sender, RoutedEventArgs e) => await PickFolderAndStartAsync();

    private void OnComposerKeyDown(object sender, KeyRoutedEventArgs e)
    {
        if (e.Key != VirtualKey.Enter || InputKeyboardSource.GetKeyStateForCurrentThread(VirtualKey.Shift).HasFlag(CoreVirtualKeyStates.Down))
            return;
        e.Handled = true;
        if (ViewModel.SendCommand.CanExecute(null))
            ViewModel.SendCommand.Execute(null);
    }

    private void OnSideTabChanged(SelectorBar sender, SelectorBarSelectionChangedEventArgs args)
    {
        var tab = sender.SelectedItem?.Tag as string ?? "diff";
        ViewModel.SideTab = tab;
        DiffPanel.Visibility = tab == "diff" ? Visibility.Visible : Visibility.Collapsed;
        TerminalPanel.Visibility = tab == "terminal" ? Visibility.Visible : Visibility.Collapsed;
        FilesPanel.Visibility = tab == "files" ? Visibility.Visible : Visibility.Collapsed;
        PreviewPanel.Visibility = tab == "preview" ? Visibility.Visible : Visibility.Collapsed;
        PreviewText.Text = ViewModel.PreviewFile ?? Loc.S("AgentNoPreview");
        OpenPreviewButton.IsEnabled = ViewModel.PreviewFile is not null;
    }

    private async void OnFileClick(object sender, ItemClickEventArgs e)
    {
        if (e.ClickedItem is string file && ViewModel.Selected is { } s)
            await Launcher.LaunchUriAsync(new Uri(Path.Combine(s.Folder, file)));
    }

    private async void OnOpenPreview(object sender, RoutedEventArgs e)
    {
        if (ViewModel.PreviewFile is { } file)
            await Launcher.LaunchUriAsync(new Uri(file));
    }

    private void OnViewModelChanged(object? sender, PropertyChangedEventArgs e)
    {
        if (e.PropertyName == nameof(AgentViewModel.Selected))
            Watch(ViewModel.Selected);
    }

    private void Watch(AgentSession? session)
    {
        if (_watched is not null)
            _watched.Items.CollectionChanged -= OnItemsChanged;
        _watched = session;
        if (session is not null)
            session.Items.CollectionChanged += OnItemsChanged;
        ScrollToEnd();
    }

    private void OnItemsChanged(object? sender, NotifyCollectionChangedEventArgs e) => ScrollToEnd();

    private void ScrollToEnd() =>
        DispatcherQueue.TryEnqueue(() => Transcript.ChangeView(null, Transcript.ScrollableHeight, null, disableAnimation: true));
}
