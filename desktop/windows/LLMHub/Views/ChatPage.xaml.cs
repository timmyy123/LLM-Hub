using System.Collections.Specialized;
using System.ComponentModel;
using LLMHub.Services;
using LLMHub.ViewModels;
using Microsoft.UI.Input;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Input;
using Windows.System;
using Windows.UI.Core;

namespace LLMHub.Views;

public sealed partial class ChatPage : Page
{
    public ChatPage()
    {
        InitializeComponent();
        ViewModel.Messages.CollectionChanged += OnMessagesChanged;
        Unloaded += (_, _) => ViewModel.Messages.CollectionChanged -= OnMessagesChanged;
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
