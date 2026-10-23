using System.Collections.Specialized;
using System.ComponentModel;
using LLMHub.Services;
using LLMHub.ViewModels;

namespace LLMHub.Linux;

/// <summary>
/// One sidebar for everything (as on Windows and macOS): app sections and chats, or in Settings a Back button
/// and the settings sections. Content on the right is a stack of pages.
/// </summary>
internal sealed class MainWindow
{
    private readonly Adw.ApplicationWindow _window;
    private readonly Adw.WindowTitle _title = Adw.WindowTitle.New("LLM Hub", "");
    private readonly Gtk.Stack _content = Gtk.Stack.New();
    private readonly Gtk.Stack _sidebarModes = Gtk.Stack.New();
    private readonly Gtk.ListBox _nav = Gtk.ListBox.New();
    private readonly Gtk.ListBox _settingsNav = Gtk.ListBox.New();
    private readonly Gtk.ListBox _chats = Gtk.ListBox.New();
    private readonly Gtk.SearchEntry _search = Gtk.SearchEntry.New();
    private readonly Gtk.Box _chatsArea;
    private readonly Gtk.Label _engineText = Ui.Label("", "heading");
    private readonly Gtk.Box _dot = Gtk.Box.New(Gtk.Orientation.Horizontal, 0);
    private readonly SettingsPages _settings;
    private bool _syncingChats;

    public ChatPage Chat { get; }
    public Gtk.Window Window => _window;

    public MainWindow(Adw.Application app)
    {
        _window = Adw.ApplicationWindow.New(app);
        _window.SetTitle("LLM Hub");
        _window.SetDefaultSize(1440, 920);
        _window.SetSizeRequest(900, 600);

        Chat = new ChatPage(this);
        _settings = new SettingsPages(this);
        _content.AddNamed(Chat.Root, "chat");
        foreach (var section in new[] { "Agent", "Design", "Studio" })
            _content.AddNamed(Placeholder(section), section.ToLowerInvariant());
        _content.AddNamed(_settings.General, "general");
        _content.AddNamed(_settings.Models, "models");
        _content.AddNamed(_settings.Api, "api");

        _chatsArea = BuildChatsArea();
        _sidebarModes.AddNamed(BuildAppSidebar(), "app");
        _sidebarModes.AddNamed(BuildSettingsSidebar(), "settings");
        _sidebarModes.SetVexpand(true);

        var sidebar = Ui.VBox(0, _sidebarModes, BuildEngineCard());
        var sidebarView = Adw.ToolbarView.New();
        sidebarView.AddTopBar(Adw.HeaderBar.New());
        sidebarView.SetContent(sidebar);

        var contentHeader = Adw.HeaderBar.New();
        contentHeader.SetTitleWidget(_title);
        var contentView = Adw.ToolbarView.New();
        contentView.AddTopBar(contentHeader);
        contentView.SetContent(_content);

        var split = Adw.NavigationSplitView.New();
        split.SetSidebar(Adw.NavigationPage.New(sidebarView, "LLM Hub"));
        split.SetContent(Adw.NavigationPage.New(contentView, "LLM Hub"));
        split.SetMinSidebarWidth(240);
        split.SetMaxSidebarWidth(300);
        _window.SetContent(split);

        Ioc.Chat.Groups.CollectionChanged += (_, _) => RebuildChats();
        Ioc.Chat.PropertyChanged += OnChatChanged;
        Ioc.Engines.PropertyChanged += (_, _) => UpdateEngine();
        RebuildChats();
        UpdateEngine();
        ShowChat();
    }

    public void Present() => _window.Present();

    public void ShowChat()
    {
        _sidebarModes.SetVisibleChildName("app");
        _nav.SelectRow(_nav.GetRowAtIndex(0));
        Show("chat", Loc.S("NavChat"));
    }

    public void ShowSettings(string section = "models")
    {
        _sidebarModes.SetVisibleChildName("settings");
        var index = section switch { "general" => 0, "api" => 2, _ => 1 };
        _settingsNav.SelectRow(_settingsNav.GetRowAtIndex(index));
        Show(section, Loc.S("NavSettings"));
        if (section == "models")
            _ = Ioc.ModelsVm.LoadCatalogAsync();
    }

    private void Show(string page, string title)
    {
        _content.SetVisibleChildName(page);
        _chatsArea.SetVisible(page == "chat");
        _title.SetTitle(page == "chat" && Ioc.Chat.CurrentTitle.Length > 0 ? Ioc.Chat.CurrentTitle : title);
    }

    // ---------------------------------------------------------------- sidebar

    private Gtk.Widget BuildAppSidebar()
    {
        var newChat = Gtk.Button.New();
        var content = Adw.ButtonContent.New();
        content.SetIconName("list-add-symbolic");
        content.SetLabel(Loc.S("NewChat"));
        newChat.SetChild(content);
        newChat.AddCssClass("pill");
        newChat.Named("NewChatButton").Margins(6, 10, 6, 10);
        newChat.OnClicked += (_, _) =>
        {
            Ioc.Chat.NewChatCommand.Execute(null);
            ShowChat();
        };

        _nav.AddCssClass("navigation-sidebar");
        foreach (var (key, icon) in new[] { ("NavChat", "chat-message-new-symbolic"), ("NavAgent", "utilities-terminal-symbolic"),
                                             ("NavDesign", "applications-graphics-symbolic"), ("NavStudio", "image-x-generic-symbolic") })
            _nav.Append(NavRow(key, icon).Named(key));
        _nav.OnRowActivated += (_, e) =>
        {
            switch (e.Row.GetIndex())
            {
                case 0: Show("chat", Loc.S("NavChat")); break;
                case 1: Show("agent", Loc.S("NavAgent")); break;
                case 2: Show("design", Loc.S("NavDesign")); break;
                default: Show("studio", Loc.S("NavStudio")); break;
            }
        };

        return Ui.VBox(4, newChat, _nav, _chatsArea);
    }

    private Gtk.Box BuildChatsArea()
    {
        _search.SetPlaceholderText(Loc.S("SearchChats"));
        _search.Named("ChatSearch").Margins(8, 10, 4, 10);
        _search.OnSearchChanged += (_, _) => Ioc.Chat.SearchText = _search.GetText();

        _chats.AddCssClass("navigation-sidebar");
        _chats.Named("ChatList");
        _chats.OnRowActivated += (_, e) =>
        {
            if (_syncingChats || e.Row.GetName() is not { Length: > 0 } id || !id.StartsWith("chat:"))
                return;
            Ioc.Chat.SelectedChat = Ioc.Chat.Groups.SelectMany(g => g).FirstOrDefault(c => "chat:" + c.Id == id);
            Show("chat", Loc.S("NavChat"));
        };
        var scroll = Gtk.ScrolledWindow.New();
        scroll.SetChild(_chats);
        scroll.SetVexpand(true);
        scroll.SetPolicy(Gtk.PolicyType.Never, Gtk.PolicyType.Automatic);
        var area = Ui.VBox(0, _search, scroll);
        area.SetVexpand(true);
        return area;
    }

    private Gtk.Widget BuildSettingsSidebar()
    {
        var back = Gtk.Button.New();
        var content = Adw.ButtonContent.New();
        content.SetIconName("go-previous-symbolic");
        content.SetLabel(Loc.S("Back"));
        back.SetChild(content);
        back.AddCssClass("flat");
        back.SetHalign(Gtk.Align.Start);
        back.Named("BackButton").Margins(6, 10, 6, 10);
        back.OnClicked += (_, _) => ShowChat();

        _settingsNav.AddCssClass("navigation-sidebar");
        foreach (var (key, icon, id) in new[] { ("SettingsGeneral", "emblem-system-symbolic", "SectionGeneral"),
                                                 ("SettingsModels", "folder-download-symbolic", "SectionModels"),
                                                 ("SettingsLocalApi", "network-server-symbolic", "SectionApi") })
            _settingsNav.Append(NavRow(key, icon).Named(id));
        _settingsNav.OnRowActivated += (_, e) => ShowSettings(e.Row.GetIndex() switch { 0 => "general", 2 => "api", _ => "models" });

        var header = Ui.Label(Loc.S("NavSettings"), "heading", "dim-label").Margins(10, 12, 2, 16);
        return Ui.VBox(2, back, header, _settingsNav);
    }

    private static Gtk.ListBoxRow NavRow(string key, string icon)
    {
        var row = Gtk.ListBoxRow.New();
        row.SetChild(Ui.HBox(12, Gtk.Image.NewFromIconName(icon), Ui.Label(Loc.S(key))).Margins(6, 6, 6, 6));
        return row;
    }

    private Gtk.Widget BuildEngineCard()
    {
        _dot.AddCssClass("dot");
        _dot.SetValign(Gtk.Align.Center);
        var hardware = Ui.Label(Ioc.Hardware.Summary(withRam: false), "caption", "dim-label");
        hardware.SetEllipsize(Pango.EllipsizeMode.End);
        _engineText.SetEllipsize(Pango.EllipsizeMode.End);
        var texts = Ui.VBox(0, _engineText, hardware);
        texts.SetHexpand(true);
        var card = Gtk.Button.New();
        card.SetChild(Ui.HBox(10, _dot, texts, Gtk.Image.NewFromIconName("emblem-system-symbolic")));
        card.AddCssClass("engine-card");
        card.AddCssClass("flat");
        card.Named("SettingsButton").Margins(6, 8, 8, 8);
        card.OnClicked += (_, _) => ShowSettings();
        return card;
    }

    private void UpdateEngine()
    {
        _engineText.SetText(Ioc.Engines.StatusText);
        foreach (var c in new[] { "ready", "busy", "error" })
            _dot.RemoveCssClass(c);
        _dot.AddCssClass(Ioc.Engines.State switch
        {
            EngineState.Ready => "ready",
            EngineState.Installing or EngineState.Loading => "busy",
            EngineState.Error => "error",
            _ => "idle",
        });
    }

    private void OnChatChanged(object? sender, PropertyChangedEventArgs e)
    {
        if (e.PropertyName is nameof(ChatViewModel.CurrentTitle) && _content.GetVisibleChildName() == "chat")
            _title.SetTitle(Ioc.Chat.CurrentTitle.Length > 0 ? Ioc.Chat.CurrentTitle : Loc.S("NavChat"));
        if (e.PropertyName is nameof(ChatViewModel.SelectedChat))
            SyncChatSelection();
    }

    private void RebuildChats()
    {
        _syncingChats = true;
        _chats.RemoveAll();
        foreach (var group in Ioc.Chat.Groups)
        {
            var header = Gtk.ListBoxRow.New();
            header.SetChild(Ui.Label(group.Key, "caption-heading", "dim-label").Margins(10, 8, 2, 8));
            header.SetSelectable(false);
            header.SetActivatable(false);
            _chats.Append(header);
            foreach (var chat in group)
            {
                var row = Gtk.ListBoxRow.New();
                var label = Ui.Label(chat.Title).Named("ChatTitle");
                label.SetWrap(false);
                label.SetEllipsize(Pango.EllipsizeMode.End);
                row.SetChild(label.Margins(6, 6, 6, 6));
                row.SetName("chat:" + chat.Id);
                row.SetTooltipText(chat.Title);
                AddDeleteMenu(row, chat);
                _chats.Append(row);
            }
        }
        _syncingChats = false;
        SyncChatSelection();
    }

    private void AddDeleteMenu(Gtk.ListBoxRow row, ChatSummary chat)
    {
        var popover = Gtk.Popover.New();
        popover.SetParent(row);
        popover.SetChild(Ui.Button(Loc.S("DeleteChat"), () =>
        {
            popover.Popdown();
            Ioc.Chat.DeleteChatCommand.Execute(chat);
        }, "flat", "destructive-action").Named("DeleteChatMenuItem"));
        var click = Gtk.GestureClick.New();
        click.SetButton(3);
        click.OnPressed += (_, _) => popover.Popup();
        row.AddController(click);
    }

    private void SyncChatSelection()
    {
        _syncingChats = true;
        var id = Ioc.Chat.SelectedChat is { } c ? "chat:" + c.Id : null;
        var index = 0;
        while (_chats.GetRowAtIndex(index++) is { } row)
        {
            if (row.GetName() == id)
            {
                _chats.SelectRow(row);
                _syncingChats = false;
                return;
            }
        }
        _chats.UnselectAll();
        _syncingChats = false;
    }

    private static Gtk.Widget Placeholder(string section)
    {
        var page = Adw.StatusPage.New();
        page.SetIconName(section switch { "Agent" => "utilities-terminal-symbolic", "Design" => "applications-graphics-symbolic", _ => "image-x-generic-symbolic" });
        page.SetTitle(Loc.F("ComingSoonTitle", Loc.S($"Nav{section}")));
        page.SetDescription(Loc.S($"{section}Description"));
        page.Named("PlaceholderHeading");
        return page;
    }
}
