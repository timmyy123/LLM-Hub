using System.ComponentModel;
using System.Runtime.InteropServices;
using LLMHub.Services;
using LLMHub.ViewModels;
using LLMHub.Views;
using Microsoft.UI.Windowing;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Data;
using Microsoft.UI.Xaml.Media;
using Windows.Graphics;

namespace LLMHub;

public sealed partial class MainWindow : Window
{
    public MainWindow()
    {
        UiThread.Initialize(action => DispatcherQueue.TryEnqueue(() => action()), () => DispatcherQueue.HasThreadAccess);
        ChatGroupsSource = new CollectionViewSource { IsSourceGrouped = true, Source = Chat.Groups };
        InitializeComponent();

        ExtendsContentIntoTitleBar = true;
        SetTitleBar(AppTitleBar);
        SizeToScreen();
        AppWindow.SetIcon(Path.Combine(AppContext.BaseDirectory, "Assets", "llmhub.ico"));
        ApplyAppearance();

        UpdateEngineCard();
        Engine.PropertyChanged += OnEnginePropertyChanged;
        Root.ActualThemeChanged += (_, _) => UpdateEngineDot();
        UpdateEngineDot();

        NavList.SelectedIndex = 0;
#if DEBUG
        if (Environment.GetCommandLineArgs().SkipWhile(a => a != "--theme").Skip(1).FirstOrDefault() is { } theme)
            Root.RequestedTheme = theme == "dark" ? ElementTheme.Dark : ElementTheme.Light;
#endif
    }

    public void ApplyAppearance() => Root.RequestedTheme = AppSettings.Current.Appearance switch
    {
        "light" => ElementTheme.Light,
        "system" => ElementTheme.Default,
        _ => ElementTheme.Dark,
    };

    /// <summary>AppWindow sizes are physical pixels; scale the design size (1440 x 920) by the monitor DPI.</summary>
    private void SizeToScreen()
    {
        var scale = GetDpiForWindow(WinRT.Interop.WindowNative.GetWindowHandle(this)) / 96.0;
        var area = DisplayArea.GetFromWindowId(AppWindow.Id, DisplayAreaFallback.Primary).WorkArea;
        var width = Math.Min((int)(1440 * scale), (int)(area.Width * 0.94));
        var height = Math.Min((int)(920 * scale), (int)(area.Height * 0.94));
        AppWindow.MoveAndResize(new RectInt32(area.X + (area.Width - width) / 2, area.Y + (area.Height - height) / 2, width, height));
        if (AppWindow.Presenter is OverlappedPresenter presenter)
        {
            presenter.PreferredMinimumWidth = (int)(1000 * scale);
            presenter.PreferredMinimumHeight = (int)(600 * scale);
        }
    }

    [LibraryImport("user32.dll")]
    private static partial uint GetDpiForWindow(IntPtr hwnd);

    public ChatViewModel Chat => Ioc.Chat;
    public AgentViewModel AgentVm => Ioc.AgentVm;
    public CollectionViewSource AgentGroupsSource { get; } = new() { IsSourceGrouped = true, Source = Ioc.AgentVm.Groups };
    public EngineHub Engine => Ioc.Engines;
    public CollectionViewSource ChatGroupsSource { get; }

    public void ShowSettings(string? section = null)
    {
        section ??= "models";
        SetSettingsMode(true);
        SectionTitle.Text = Loc.S("NavSettings");
        if (ContentFrame.Content is not SettingsPage)
            ContentFrame.Navigate(typeof(SettingsPage), section);
        SettingsSectionList.SelectedItem = SettingsSectionList.Items.Cast<ListViewItem>().First(i => (string)i.Tag == section);
    }

    private void SetSettingsMode(bool on)
    {
        var app = on ? Visibility.Collapsed : Visibility.Visible;
        NewChatButton.Visibility = app;
        NavList.Visibility = app;
        SettingsNav.Visibility = on ? Visibility.Visible : Visibility.Collapsed;
        if (on)
        {
            NavList.SelectedItem = null;
            SetSidebarChatsVisible(false);
        }
    }

    private void OnSettingsSectionChanged(object sender, SelectionChangedEventArgs e)
    {
        if (SettingsSectionList.SelectedItem is ListViewItem { Tag: string tag } && ContentFrame.Content is SettingsPage page)
            page.ShowSection(tag);
    }

    private void OnBackFromSettings(object sender, RoutedEventArgs e)
    {
        SetSettingsMode(false);
        ShowChat();
    }

    public void ShowChat()
    {
        NavList.SelectedIndex = 0;
    }

    /// <summary>Shows an app section by tag: chat, agent, design or studio.</summary>
    public void ShowSection(string tag)
    {
        SetSettingsMode(false);
        NavList.SelectedItem = NavList.Items.Cast<ListViewItem>().First(i => (string)i.Tag == tag);
    }

    private void OnNavChanged(object sender, SelectionChangedEventArgs e)
    {
        if (NavList.SelectedItem is not ListViewItem { Tag: string tag })
            return;
        SetSidebarChatsVisible(tag == "chat");
        switch (tag)
        {
            case "chat":
                SectionTitle.Text = Loc.S("NavChat");
                if (ContentFrame.Content is not ChatPage)
                    ContentFrame.Navigate(typeof(ChatPage));
                break;
            case "agent":
                SectionTitle.Text = Loc.S("NavAgent");
                if (ContentFrame.Content is not AgentPage)
                    ContentFrame.Navigate(typeof(AgentPage));
                break;
            case "design":
                SectionTitle.Text = Loc.S("NavDesign");
                ContentFrame.Navigate(typeof(PlaceholderPage), "Design");
                break;
            case "studio":
                SectionTitle.Text = Loc.S("NavStudio");
                ContentFrame.Navigate(typeof(PlaceholderPage), "Studio");
                break;
        }
    }

    private void SetSidebarChatsVisible(bool visible)
    {
        var v = visible ? Visibility.Visible : Visibility.Collapsed;
        SearchField.Visibility = v;
        ChatList.Visibility = v;
        var agent = NavList.SelectedItem is ListViewItem { Tag: "agent" };
        AgentSessionList.Visibility = agent ? Visibility.Visible : Visibility.Collapsed;
        NewButtonLabel.Text = Loc.S(agent ? "NewSession" : "NewChat");
        UpdateEngineCard();
    }

    private async void OnNewChat(object sender, RoutedEventArgs e)
    {
        if (NavList.SelectedItem is ListViewItem { Tag: "agent" })
        {
            await AgentPage.PickFolderAndStartAsync();
            return;
        }
        Chat.NewChatCommand.Execute(null);
        ShowChat();
    }

    private void OnDeleteSession(object sender, RoutedEventArgs e)
    {
        if (sender is FrameworkElement { Tag: Services.Agent.AgentSession session })
            AgentVm.DeleteSessionCommand.Execute(session);
    }

    /// <summary>In Agent mode the card names the engine (as in the design); elsewhere it shows the loaded model.</summary>
    private void UpdateEngineCard()
    {
        if (NavList.SelectedItem is ListViewItem { Tag: "agent" })
        {
            EngineStatus.Text = Loc.S("AgentEngine");
            HardwareLine.Text = AgentVm.EngineLine;
        }
        else
        {
            EngineStatus.Text = Engine.StatusText;
            HardwareLine.Text = Ioc.Hardware.Summary(withRam: false);
        }
    }

    private void OnStarChat(object sender, RoutedEventArgs e)
    {
        if (sender is FrameworkElement { Tag: ChatSummary chat })
            Chat.ToggleStarCommand.Execute(chat);
    }

    private void OnDeleteChat(object sender, RoutedEventArgs e)
    {
        if (sender is FrameworkElement { Tag: ChatSummary chat })
            Chat.DeleteChatCommand.Execute(chat);
    }

    private void OnOpenSettings(object sender, RoutedEventArgs e) => ShowSettings(ContentFrame.Content is SettingsPage ? "models" : null);

    private void OnEnginePropertyChanged(object? sender, PropertyChangedEventArgs e)
    {
        if (e.PropertyName == nameof(EngineHub.State))
            UpdateEngineDot();
        if (e.PropertyName == nameof(EngineHub.StatusText))
            UpdateEngineCard();
    }

    private void UpdateEngineDot()
    {
        var key = Engine.State switch
        {
            EngineState.Ready => "SuccessBrush",
            EngineState.Installing or EngineState.Loading => "WarningBrush",
            EngineState.Error => "DangerBrush",
            _ => "TextFaintBrush",
        };
        if (Ui.ThemeBrush(Root, key) is { } brush)
            EngineDot.Fill = brush;
    }
}
