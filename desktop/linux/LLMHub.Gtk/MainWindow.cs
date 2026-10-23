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
    private readonly Gtk.ListBox _agentList = Gtk.ListBox.New();
    private readonly Gtk.ListBox _designList = Gtk.ListBox.New();
    private readonly Gtk.Widget _agentArea;
    private readonly Gtk.Widget _designArea;
    private string _section = "chat";
    private bool _syncingLists;
    private readonly List<(Services.Agent.AgentSession Session, System.ComponentModel.PropertyChangedEventHandler Handler)> _agentRowHandlers = [];
    private readonly Gtk.Label _engineText = Ui.Label("", "heading");
    private readonly Gtk.Box _dot = Gtk.Box.New(Gtk.Orientation.Horizontal, 0);
    private readonly SettingsPages _settings;
    private readonly SettingsSections _sections = new();
    private static readonly string[] SettingsTags = ["general", "models", "engines", "api", "speech", "studio", "agent", "websearch", "storage"];
    private Adw.NavigationSplitView _split = null!;
    private Gtk.Widget _newButton = null!;
    private Gtk.Widget _engineTexts = null!;
    private Gtk.Widget _engineGear = null!;
    private readonly List<Gtk.Label> _navLabels = [];
    private readonly Adw.WindowTitle _sidebarTitle = Adw.WindowTitle.New("LLM Hub", "");
    private readonly Gtk.Button _star;
    private readonly Gtk.Label _newLabel = Ui.Label(Loc.S("NewChat"), "heading");
    private readonly Gtk.Button _share;
    private bool _syncingChats;

    public ChatPage Chat { get; }
    public AgentPage Agent { get; }
    public DesignPage Design { get; }
    public StudioPage Studio { get; }
    public Gtk.Window Window => _window;

    public MainWindow(Adw.Application app)
    {
        _window = Adw.ApplicationWindow.New(app);
        _window.SetTitle("LLM Hub");
        _window.SetDefaultSize(1440, 920);
        _window.SetSizeRequest(900, 600);

        Chat = new ChatPage(this);
        _settings = new SettingsPages(this);
        Agent = new AgentPage(this);
        Design = new DesignPage(this);
        Studio = new StudioPage(this);
        _content.AddNamed(Chat.Root, "chat");
        _content.AddNamed(Agent.Root, "agent");
        _content.AddNamed(Design.Root, "design");
        _content.AddNamed(Studio.Root, "studio");
        _content.AddNamed(_settings.General, "general");
        _content.AddNamed(_settings.Models, "models");
        _content.AddNamed(_settings.Api, "api");
        _content.AddNamed(_sections.Engines, "engines");
        _content.AddNamed(_sections.Speech, "speech");
        _content.AddNamed(_sections.Studio, "studio-settings");
        _content.AddNamed(_sections.Agent, "agent-settings");
        _content.AddNamed(_sections.WebSearch, "websearch");
        _content.AddNamed(_sections.Storage, "storage");

        _chatsArea = BuildChatsArea();
        _agentArea = BuildAgentArea();
        _designArea = BuildDesignArea();
        _sidebarModes.AddNamed(BuildAppSidebar(), "app");
        _sidebarModes.AddNamed(BuildSettingsSidebar(), "settings");
        _sidebarModes.SetVexpand(true);

        var sidebar = Ui.VBox(0, _sidebarModes, BuildEngineCard());
        var sidebarView = Adw.ToolbarView.New();
        var sidebarHeader = Adw.HeaderBar.New();
        sidebarHeader.SetTitleWidget(_sidebarTitle);
        sidebarView.AddTopBar(sidebarHeader);
        sidebarView.SetContent(sidebar);

        var contentHeader = Adw.HeaderBar.New();
        contentHeader.SetTitleWidget(_title);
        // Chat header actions from the design: star the chat, share it as Markdown.
        _share = Ui.Button(Loc.S("Share"), ShareChat).Named("ShareButton");
        _star = Ui.IconButton("non-starred-symbolic", Loc.S("Star"), () => Ioc.Chat.ToggleStarCommand.Execute(null), "flat").Named("StarButton");
        contentHeader.PackEnd(_share);
        contentHeader.PackEnd(_star);
        var contentView = Adw.ToolbarView.New();
        contentView.AddTopBar(contentHeader);
        contentView.SetContent(_content);

        var split = _split = Adw.NavigationSplitView.New();
        split.SetSidebar(Adw.NavigationPage.New(sidebarView, "LLM Hub"));
        split.SetContent(Adw.NavigationPage.New(contentView, "LLM Hub"));
        split.SetMinSidebarWidth(240);
        split.SetMaxSidebarWidth(300);
        _window.SetContent(split);

        Ioc.Chat.Groups.CollectionChanged += (_, _) => RebuildChats();
        Ioc.Chat.PropertyChanged += OnChatChanged;
        Ioc.Engines.PropertyChanged += (_, _) => UpdateEngine();
        Ioc.AgentVm.Groups.CollectionChanged += (_, _) => UiThread.Run(RebuildAgentList);
        Ui.Watch(Ioc.AgentVm, p =>
        {
            if (p == nameof(AgentViewModel.Selected))
                SyncAgentSelection();
        });
        Ioc.DesignVm.Projects.CollectionChanged += (_, _) => UiThread.Run(RebuildDesignList);
        Ui.Watch(Ioc.DesignVm, p =>
        {
            if (p == nameof(DesignViewModel.Selected))
                SyncDesignSelection();
            if (p is nameof(DesignViewModel.Selected) or nameof(DesignViewModel.IsGenerating))
                UpdateDesignTitle();
        });
        RebuildAgentList();
        RebuildDesignList();
        RebuildChats();
        UpdateEngine();
        ShowChat();
    }

    public void Present() => _window.Present();

    /// <summary>Star and Share show for an open chat on the Chat page.</summary>
    public void UpdateChatHeader()
    {
        if (_star is null)
            return;
        var onChat = _content.GetVisibleChildName() == "chat";
        _star.SetVisible(onChat && Ioc.Chat.HasOpenChat);
        _share.SetVisible(onChat && Ioc.Chat.HasMessages);
        _star.SetIconName(Ioc.Chat.IsCurrentStarred ? "starred-symbolic" : "non-starred-symbolic");
        _star.SetTooltipText(Loc.S(Ioc.Chat.IsCurrentStarred ? "Unstar" : "Star"));
        if (onChat)
            _title.SetTitle(Ioc.Chat.CurrentTitle.Length > 0 ? Ioc.Chat.CurrentTitle : Loc.S("NavChat"));
    }

    private async void ShareChat()
    {
        try
        {
            var dialog = Gtk.FileDialog.New();
            dialog.SetInitialName(Ioc.Chat.ExportFileName());
            if (await dialog.SaveAsync(_window) is { } file && file.GetPath() is { } path)
                await File.WriteAllTextAsync(path, Ioc.Chat.ExportMarkdown());
        }
        catch (GLib.GException)
        {
            // Dismissed.
        }
    }

    public void ShowChat()
    {
        SetCompact(false);
        _sidebarModes.SetVisibleChildName("app");
        _nav.SelectRow(_nav.GetRowAtIndex(0));
        Show("chat", Loc.S("NavChat"));
    }

    public void ShowSettings(string section = "models")
    {
        SetCompact(false);
        _sidebarModes.SetVisibleChildName("settings");
        var index = Math.Max(0, Array.IndexOf(SettingsTags, section));
        _settingsNav.SelectRow(_settingsNav.GetRowAtIndex(index));
        // Studio and Agent settings share their tag with the app sections, so their pages carry a suffix.
        Show(section is "studio" or "agent" ? section + "-settings" : section, Loc.S("NavSettings"));
        if (section == "models")
            _ = Ioc.ModelsVm.LoadCatalogAsync();
        else if (section is not ("general" or "api" or "websearch"))
            _sections.Refresh();
    }

    /// <summary>Shows an app section (chat, agent, design, studio) with its sidebar list.</summary>
    public void ShowSection(string section)
    {
        _sidebarModes.SetVisibleChildName("app");
        var index = section switch { "agent" => 1, "design" => 2, "studio" => 3, _ => 0 };
        _nav.SelectRow(_nav.GetRowAtIndex(index));
        SetCompact(section is "design" or "studio");
        Show(section, Loc.S(section switch { "agent" => "NavAgent", "design" => "NavDesign", "studio" => "NavStudio", _ => "NavChat" }));
    }

    private void Show(string page, string title)
    {
        _content.SetVisibleChildName(page);
        if (page is "chat" or "agent" or "design" or "studio")
            _section = page;
        UpdateChatHeader();
        _chatsArea.SetVisible(page == "chat");
        _agentArea.SetVisible(page == "agent");
        _designArea.SetVisible(false);
        _newLabel.SetText(Loc.S(page switch { "agent" => "NewSession", "design" => "DesignNewProject", _ => "NewChat" }));
        _title.SetTitle(page == "chat" && Ioc.Chat.CurrentTitle.Length > 0 ? Ioc.Chat.CurrentTitle : title);
        UpdateDesignTitle();
    }

    /// <summary>"Design · Crumb &amp; Co" while Design is open, as in the design's title bar.</summary>
    private void UpdateDesignTitle()
    {
        if (_content.GetVisibleChildName() == "design")
            _title.SetTitle(Ioc.DesignVm.Selected is { Name.Length: > 0 } p ? $"{Loc.S("NavDesign")} · {p.Name}" : Loc.S("NavDesign"));
    }

    // ---------------------------------------------------------------- sidebar

    private Gtk.Widget BuildAppSidebar()
    {
        // Left-aligned with the shortcut on the right, as in the design.
        var newChat = Gtk.Button.New();
        var shortcut = Ui.Label("Ctrl+N", "caption", "dim-label");
        shortcut.SetHexpand(true);
        shortcut.SetXalign(1);
        _newLabel.SetXalign(0);
        _newLabel.SetWrap(false);
        newChat.SetChild(Ui.HBox(10, Gtk.Image.NewFromIconName("list-add-symbolic"), _newLabel, shortcut));
        newChat.AddCssClass("new-button");
        newChat.Named("NewChatButton").Margins(6, 10, 6, 10);
        newChat.OnClicked += (_, _) =>
        {
            switch (_section)
            {
                case "agent":
                    AgentPage.PickFolderAndStart(_window);
                    break;
                case "design":
                    Ioc.DesignVm.NewProjectCommand.Execute(null);
                    break;
                default:
                    Ioc.Chat.NewChatCommand.Execute(null);
                    ShowChat();
                    break;
            }
        };
        var keys = Gtk.EventControllerKey.New();
        keys.OnKeyPressed += (_, e) =>
        {
            if ((e.Keyval is 0x6e or 0x4e) && (e.State & Gdk.ModifierType.ControlMask) != 0)
            {
                newChat.Activate();
                return true;
            }
            return false;
        };
        _window.AddController(keys);

        _nav.AddCssClass("navigation-sidebar");
        foreach (var (key, icon) in new[] { ("NavChat", "chat-message-new-symbolic"), ("NavAgent", "utilities-terminal-symbolic"),
                                             ("NavDesign", "applications-graphics-symbolic"), ("NavStudio", "image-x-generic-symbolic") })
        {
            var row = NavRow(key, icon).Named(key);
            if (row.GetChild() is Gtk.Box box && box.GetLastChild() is Gtk.Label label)
                _navLabels.Add(label);
            row.SetTooltipText(Loc.S(key));
            _nav.Append(row);
        }
        _nav.OnRowActivated += (_, e) =>
        {
            ShowSection(e.Row.GetIndex() switch { 1 => "agent", 2 => "design", 3 => "studio", _ => "chat" });
        };

        _newButton = newChat;
        return Ui.VBox(4, newChat, _nav, _chatsArea, _agentArea, _designArea);
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
                                                 ("SettingsEngines", "system-run-symbolic", "SectionEngines"),
                                                 ("SettingsLocalApi", "network-server-symbolic", "SectionApi"),
                                                 ("SettingsSpeech", "audio-input-microphone-symbolic", "SectionSpeech"),
                                                 ("SettingsStudio", "image-x-generic-symbolic", "SectionStudio"),
                                                 ("SettingsAgent", "utilities-terminal-symbolic", "SectionAgent"),
                                                 ("SettingsWebSearch", "web-browser-symbolic", "SectionWebSearch"),
                                                 ("SettingsStorage", "drive-harddisk-symbolic", "SectionStorage") })
            _settingsNav.Append(NavRow(key, icon).Named(id));
        _settingsNav.OnRowActivated += (_, e) => ShowSettings(SettingsTags[e.Row.GetIndex()]);

        var header = Ui.Label(Loc.S("NavSettings"), "heading", "dim-label").Margins(10, 12, 2, 16);
        return Ui.VBox(2, back, header, _settingsNav);
    }

    /// <summary>
    /// The design's 64px icon rail on Design and Studio: the same sidebar with only nav icons and the engine dot,
    /// so the canvas gets the room. Chat, Agent and Settings keep the full sidebar.
    /// </summary>
    private void SetCompact(bool compact)
    {
        if (_split is null)
            return;
        _split.SetMinSidebarWidth(compact ? 64 : 240);
        _split.SetMaxSidebarWidth(compact ? 64 : 300);
        foreach (var label in _navLabels)
            label.SetVisible(!compact);
        _newButton.SetVisible(!compact);
        _engineTexts.SetVisible(!compact);
        _engineGear.SetVisible(!compact);
        _sidebarTitle.SetTitle(compact ? "" : "LLM Hub");
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
        var gear = Gtk.Image.NewFromIconName("emblem-system-symbolic");
        _engineTexts = texts;
        _engineGear = gear;
        card.SetChild(Ui.HBox(10, _dot, texts, gear));
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

    // ---------------------------------------------------------------- agent sessions and design projects

    private Gtk.Widget BuildAgentArea()
    {
        _agentList.AddCssClass("navigation-sidebar");
        _agentList.Named("AgentSessionList");
        _agentList.OnRowActivated += (_, e) =>
        {
            if (!_syncingLists && e.Row.GetName() is { } id && id.StartsWith("agent:"))
                Ioc.AgentVm.Selected = Ioc.Agent.FindSession(id[6..]);
        };
        var scroll = Ui.Scroll(_agentList);
        scroll.SetVisible(false);
        return scroll;
    }

    private Gtk.Widget BuildDesignArea()
    {
        _designList.AddCssClass("navigation-sidebar");
        _designList.Named("DesignProjectList");
        _designList.OnRowActivated += (_, e) =>
        {
            if (!_syncingLists && e.Row.GetIndex() < Ioc.DesignVm.Projects.Count)
                Ioc.DesignVm.Selected = Ioc.DesignVm.Projects[e.Row.GetIndex()];
        };
        var scroll = Ui.Scroll(_designList);
        scroll.SetVisible(false);
        return scroll;
    }

    private void RebuildAgentList()
    {
        _syncingLists = true;
        _agentList.RemoveAll();
        foreach (var (s, h) in _agentRowHandlers)
            s.PropertyChanged -= h;
        _agentRowHandlers.Clear();
        foreach (var group in Ioc.AgentVm.Groups)
        {
            var header = Gtk.ListBoxRow.New();
            header.SetChild(Ui.HBox(8, Gtk.Image.NewFromIconName("folder-symbolic"), Ui.Label(group.Key, "caption-heading", "dim-label")).Margins(12, 8, 2, 8));
            header.SetSelectable(false);
            header.SetActivatable(false);
            _agentList.Append(header);
            foreach (var session in group)
            {
                var dot = Gtk.Box.New(Gtk.Orientation.Horizontal, 0);
                dot.AddCssClass("dot");
                dot.SetValign(Gtk.Align.Center);
                var title = Ui.Label("");
                title.SetWrap(false);
                title.SetEllipsize(Pango.EllipsizeMode.End);
                title.SetHexpand(true);
                var status = Ui.Label("", "caption", "dim-label");
                status.SetWrap(false);
                void Update()
                {
                    title.SetText(session.Title.Length > 0 ? session.Title : session.FolderName);
                    status.SetText(session.StatusText);
                    foreach (var c in new[] { "ready", "busy", "error" })
                        dot.RemoveCssClass(c);
                    dot.AddCssClass(session.Status switch
                    {
                        Services.Agent.AgentStatus.Working or Services.Agent.AgentStatus.NeedsYou => "busy",
                        Services.Agent.AgentStatus.Error => "error",
                        _ => "ready",
                    });
                }
                Update();
                System.ComponentModel.PropertyChangedEventHandler handler = (_, _) => UiThread.Run(Update);
                session.PropertyChanged += handler;
                _agentRowHandlers.Add((session, handler));
                var row = Gtk.ListBoxRow.New();
                row.SetChild(Ui.HBox(8, dot, title, status).Margins(6, 6, 6, 6));
                row.SetName("agent:" + session.Id);
                AddContextMenu(row, Loc.S("Delete"), () => Ioc.AgentVm.DeleteSessionCommand.Execute(session));
                _agentList.Append(row);
            }
        }
        _syncingLists = false;
        SyncAgentSelection();
    }

    private void SyncAgentSelection()
    {
        _syncingLists = true;
        var id = Ioc.AgentVm.Selected is { } s ? "agent:" + s.Id : null;
        var index = 0;
        _agentList.UnselectAll();
        while (_agentList.GetRowAtIndex(index++) is { } row)
            if (row.GetName() == id)
                _agentList.SelectRow(row);
        _syncingLists = false;
    }

    private void RebuildDesignList()
    {
        _syncingLists = true;
        _designList.RemoveAll();
        foreach (var project in Ioc.DesignVm.Projects)
        {
            var title = Ui.Label(project.Name.Length > 0 ? project.Name : Loc.S("DesignUntitled"));
            title.SetWrap(false);
            title.SetEllipsize(Pango.EllipsizeMode.End);
            var row = Gtk.ListBoxRow.New();
            row.SetChild(Ui.HBox(10, Gtk.Image.NewFromIconName("applications-graphics-symbolic"), title).Margins(6, 6, 6, 6));
            AddContextMenu(row, Loc.S("Delete"), () => Ioc.DesignVm.DeleteProjectCommand.Execute(project));
            project.PropertyChanged += (_, _) => UiThread.Run(() => title.SetText(project.Name));
            _designList.Append(row);
        }
        _syncingLists = false;
        SyncDesignSelection();
    }

    private void SyncDesignSelection()
    {
        _syncingLists = true;
        var index = Ioc.DesignVm.Selected is { } p ? Ioc.DesignVm.Projects.IndexOf(p) : -1;
        if (index >= 0 && _designList.GetRowAtIndex(index) is { } row)
            _designList.SelectRow(row);
        else
            _designList.UnselectAll();
        _syncingLists = false;
    }

    private static void AddContextMenu(Gtk.ListBoxRow row, string label, Action action)
    {
        var popover = Gtk.Popover.New();
        popover.SetParent(row);
        popover.SetChild(Ui.Button(label, () =>
        {
            popover.Popdown();
            action();
        }, "flat", "destructive-action"));
        var click = Gtk.GestureClick.New();
        click.SetButton(3);
        click.OnPressed += (_, _) => popover.Popup();
        row.AddController(click);
    }
}
