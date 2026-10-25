#if DEBUG
using LLMHub.Services;

namespace LLMHub.Linux;

/// <summary>
/// Debug builds only: <c>llmhub --uitest [name,name]</c> drives the real window the way a user would, finding widgets by the
/// names set with <see cref="Ui.Named"/> and clicking, typing and toggling them. Linux counterpart of the FlaUI suite on
/// Windows and the XCUITest suite on macOS. Prints PASS/FAIL lines, then a summary, and exits with the failure count.
/// Needs Gemma 4 12B (GGUF) and Gemma 4 E2B (LiteRT, standard file) installed; downloads LFM2.5 1.2B Thinking itself.
/// </summary>
internal sealed class UiTests(MainWindow main)
{
    private readonly string _tag = $"kiwi{DateTime.Now:HHmmss}";
    private int _passed, _failed;

    public static async void Run(MainWindow main, string? filter)
    {
        var tests = new UiTests(main);
        var failed = await tests.RunAll(filter);
        Environment.Exit(failed);
    }

    private async Task<int> RunAll(string? filter)
    {
        var all = new (string Name, Func<Task> Body)[]
        {
            ("SidebarSections", SidebarSections),
            ("SettingsSingleSidebar", SettingsSingleSidebar),
            ("SettingsSectionsContent", SettingsSectionsContent),
            ("CatalogContents", CatalogContents),
            ("LanguageAndContext", LanguageAndContext),
            ("HuggingFaceSearchAndGated", HuggingFaceSearchAndGated),
            ("ChatReplyWithFooter", ChatReplyWithFooter),
            ("FollowUpsStayInOneChat", FollowUpsStayInOneChat),
            ("ChatListSearchAndSwitch", ChatListSearchAndSwitch),
            ("StopGeneration", StopGeneration),
            ("WebSearchSources", WebSearchSources),
            ("MtpToggle", MtpToggle),
            ("LiteRtChat", LiteRtChat),
            ("DownloadThinkingModel", DownloadThinkingModel),
            ("ThinkingShown", ThinkingShown),
            ("DeleteInstalledModel", DeleteInstalledModel),
            ("ApiToggle", ApiToggle),
            ("DeleteChat", DeleteChat),
        };
        var only = filter?.Split(',', StringSplitOptions.RemoveEmptyEntries);
        foreach (var (name, body) in all.Where(t => only is null || only.Any(f => t.Name.Contains(f, StringComparison.OrdinalIgnoreCase))))
        {
            var started = DateTime.UtcNow;
            try
            {
                await body();
                _passed++;
                Log.Info($"UITEST PASS {name} ({(DateTime.UtcNow - started).TotalSeconds:F1}s)");
            }
            catch (Exception e)
            {
                _failed++;
                Log.Info($"UITEST FAIL {name}: {e.Message}");
                DevTools.Capture(main.Window, Path.Combine(AppPaths.Logs, $"uitest-{name}.png"));
            }
        }
        Log.Info($"UITEST DONE {_passed} passed, {_failed} failed");
        return _failed;
    }

    // ---------------------------------------------------------------- tests

    private async Task SidebarSections()
    {
        await GoToChat();
        await Activate(One("NavAgent"));
        await Until(() => Visible("AgentEmptyTitle") || Visible("AgentComposer"), "Agent page");
        Expect(!Visible("ChatList"), "chat list hidden outside Chat");
        await Activate(One("NavDesign"));
        await Until(() => Visible("DesignComposer") && Visible("SystemPicker"), "Design page");
        // Design and Studio collapse the sidebar to the icon rail: no New button, no labels.
        Expect(!Visible("NewChatButton"), "New button hidden on the rail");
        await Until(() => One("NavDesign").GetWidth() < 90, $"rail is narrow, got {One("NavDesign").GetWidth()}");
        await Activate(One("NavStudio"));
        await Until(() => Visible("WorkflowPicker") && Visible("GenerateButton"), "Studio page");
        Ioc.StudioVm.Mode = "search";
        await Until(() => Visible("SearchQueryBox") && Visible("SearchIndexButton") && Visible("SearchFolder"), "Studio search tab");
        // Videos: the third kind, with its own folder and the moment list.
        await Activate(One("SearchVideosTab"));
        await Until(() => Ioc.MediaSearchVm.IsVideos && Visible("SearchVideoResults") && Visible("SearchIndexButton")
                          && ((Gtk.Label)One("SearchFolder")).GetText() == Ioc.MediaSearchVm.VideoFolder, "Search › Videos");
        await Activate(One("SearchPhotosTab"));
        await Until(() => Ioc.MediaSearchVm.IsPhotos, "back to Search › Photos");
        Ioc.StudioVm.Mode = "image";
        await Activate(One("NavChat"));
        await Until(() => Visible("Composer") && Visible("ChatList") && Visible("NewChatButton"), "back on Chat with the full sidebar");
    }

    private async Task SettingsSingleSidebar()
    {
        await GoToChat();
        await Activate(One("SettingsButton"));
        await Until(() => Visible("BackButton") && Visible("CatalogList"), "settings open on Models");
        Expect(!Visible("NavAgent") && !Visible("ChatList"), "app sections replaced, not shown beside settings");
        await Activate(One("SectionGeneral"));
        await Until(() => Visible("LanguageBox") && Visible("TokenBox"), "General section");
        await Activate(One("SectionApi"));
        await Until(() => Visible("ApiToggle") && Visible("PortBox"), "Local API section");
        await Activate(One("SectionModels"));
        await Until(() => Visible("ModelTabs"), "Models section");
        foreach (var id in new[] { "SectionEngines", "SectionSpeech", "SectionStudio", "SectionAgent", "SectionWebSearch", "SectionStorage" })
            Expect(Visible(id), $"{id} listed");
        await Activate(One("BackButton"));
        await Until(() => Visible("NavAgent") && Visible("Composer"), "Back returns to the app sidebar");
    }

    private async Task SettingsSectionsContent()
    {
        foreach (var (section, content) in new[] { ("engines", "EngineCard"), ("speech", "TtsLine"), ("studio", "ComfyLine"),
                                                    ("agent", "ClaudeVersion"), ("websearch", "WebSearchToggle"), ("storage", "StorageRow") })
        {
            main.ShowSettings(section);
            await Until(() => Visible(content), $"{section} shows {content}", 60);
        }
        await Until(() => Shown("StorageRow").Count() >= 8, "storage lists every folder", 60);
    }

    private async Task WebSearchSources()
    {
        await NewChatWith(m => m.CatalogId == "gemma-4-12b" && m.Format == "gguf");
        var toggle = (Gtk.ToggleButton)One("WebSearchToggle");
        toggle.SetActive(true);
        try
        {
            var reply = await SendAndWait($"{_tag} Search the web: what is the latest stable version of the Rust programming language? One sentence, cite the source.");
            Expect(reply.Length > 10, $"reply: '{reply}'");
            await Until(() => Visible("SourceLink"), "source links under the reply", 15);
            Expect(Visible("ActivityPill"), "activity pill for the search");
            Log.Info($"UITEST web search reply: {reply[..Math.Min(reply.Length, 160)]}");
        }
        finally
        {
            toggle.SetActive(false);
        }
    }

    private async Task CatalogContents()
    {
        main.ShowSettings("models");
        await Until(() => Named("CatalogList").FirstOrDefault() is Gtk.ListBox l && l.GetRowAtIndex(10) is not null, "catalog rows");
        var titles = Descendants(One("CatalogList")).OfType<Adw.ActionRow>().Select(r => r.GetTitle() ?? "").ToList();
        Expect(titles.Contains("Gemma 4 12B") && titles.Contains("Granite 4.2 8B"), $"expected models listed: {string.Join(", ", titles)}");
        string[] banned = ["qwen", "deepseek", "glm", "kimi", "minimax", "yi", "baichuan", "internlm", "gemma 3", "granite 4.0", "granite 4.1"];
        var bad = titles.Where(t => banned.Any(b => System.Text.RegularExpressions.Regex.IsMatch(t, $@"(^|\W){b}(\W|$)", System.Text.RegularExpressions.RegexOptions.IgnoreCase))).ToList();
        Expect(bad.Count == 0, $"catalog must not list: {string.Join(", ", bad)}");
        // The Embedding chip leaves only embedding models; All brings the rest back.
        ((Gtk.ToggleButton)One("Filter_embedding")).SetActive(true);
        await Until(() => Descendants(One("CatalogList")).OfType<Adw.ActionRow>().Select(r => r.GetTitle() ?? "").All(n => n.StartsWith("EmbeddingGemma 2")), "Embedding chip shows EmbeddingGemma only");
        ((Gtk.ToggleButton)One("Filter_all")).SetActive(true);
        await Until(() => Descendants(One("CatalogList")).OfType<Adw.ActionRow>().Count() > 10, "All chip shows every model again");
        var row = One("CatalogRow-gemma-4-12b");
        await Until(() => Descendants(row).Any(w => w.GetName() == "VariantPicker" && w.GetMapped()) || Descendants(row).Any(w => w.GetName() == "InstalledLabel" && w.GetMapped()),
            "quant picker or Installed on Gemma 4 12B");
    }

    private async Task LanguageAndContext()
    {
        main.ShowSettings("general");
        var language = (Adw.ComboRow)One("LanguageBox");
        var context = (Adw.ComboRow)One("ContextBox");
        var (oldLanguage, oldContext, savedLanguage) = (language.GetSelected(), context.GetSelected(), AppSettings.Current.Language);
        try
        {
            language.SetSelected((uint)(Array.FindIndex(Loc.Languages, l => l.Tag == "de-DE") + 1));
            await Until(() => Visible("RestartHint"), "restart hint after changing language");
            Expect(AppSettings.Current.Language == "de-DE", $"language saved, got '{AppSettings.Current.Language}'");
            context.SetSelected(1);
            Expect(AppSettings.Current.ContextLength == 8192, $"context saved, got {AppSettings.Current.ContextLength}");
        }
        finally
        {
            language.SetSelected(oldLanguage);
            context.SetSelected(oldContext);
        }
        Expect(AppSettings.Current.Language == savedLanguage, "language restored");
    }

    private async Task HuggingFaceSearchAndGated()
    {
        var oldToken = AppSettings.Current.HuggingFaceToken;
        // Gated repository and no token: listing works, the download must explain how to get access.
        AppSettings.Current.HuggingFaceToken = "";
        try
        {
            main.ShowSettings("models");
            ((Adw.ViewStack)One("ModelTabs")).SetVisibleChildName("hf");
            ((Gtk.SearchEntry)One("HfQuery")).SetText("gemma-3-1b-it-qat-q4_0-gguf");
            await Activate(One("HfSearchButton"));
            var results = (Gtk.ListBox)One("HfResults");
            await Until(() => results.GetRowAtIndex(0) is not null, "Hugging Face results", 30);
            var index = 0;
            while (results.GetRowAtIndex(index) is Adw.ActionRow r && r.GetTitle() != "google/gemma-3-1b-it-qat-q4_0-gguf")
                index++;
            Expect(results.GetRowAtIndex(index) is not null, "google/gemma-3-1b-it-qat-q4_0-gguf in results");
            results.SelectRow(results.GetRowAtIndex(index));
            await Until(() => Shown("HfFileTable").Any(), "repository files");
            var detail = One("HfDetail");
            Expect(Shown("HfGatedBar").Any(), "gated notice with sign-in");
            await Until(() => Descendants(detail).Any(w => w.GetName() == "HfFileDownload" && w.GetMapped() && w.GetSensitive()), "Download in the file table", 30);
            await Activate(Descendants(detail).First(w => w.GetName() == "HfFileDownload" && w.GetSensitive()));
            await Until(() => Descendants(One("HfDetail")).OfType<Gtk.Label>().Any(l => l.GetName() == "RepoStatus" && l.GetMapped()
                    && (l.GetText() ?? "").Contains(Loc.S("NeedsToken")[..24])),
                "gated model explains the token", 60);
        }
        finally
        {
            AppSettings.Current.HuggingFaceToken = oldToken;
            ((Adw.ViewStack)One("ModelTabs")).SetVisibleChildName("catalog");
        }
    }

    private async Task ChatReplyWithFooter()
    {
        await NewChatWith(m => m.CatalogId == "gemma-4-12b" && m.Format == "gguf");
        await SendAndWait($"{_tag} Reply with exactly one word: apple");
        var reply = Shown("AssistantMessage").Select(TextOf).LastOrDefault() ?? "";
        Expect(reply.Contains("apple", StringComparison.OrdinalIgnoreCase), $"reply mentions apple: '{reply}'");
        var footer = Shown("MessageFooter").OfType<Gtk.Label>().LastOrDefault()?.GetText() ?? "";
        Expect(footer.Contains("tok/s"), $"footer shows speed: '{footer}'");
        Expect(Shown("UserMessage").OfType<Gtk.Label>().Any(l => l.GetText()!.Contains(_tag)), "user bubble shown");
        await Until(() => ChatTitles().Any(t => t.Contains(_tag)), "new chat appears in the sidebar");
        Expect(!Visible("MtpToggle"), "MTP toggle hidden for a GGUF model");
    }

    private async Task FollowUpsStayInOneChat()
    {
        // Continues the chat from ChatReplyWithFooter. The sidebar is rebuilt after every reply; that must not detach the
        // open chat (it did on Windows, saving each follow-up as a new chat).
        var chats = ChatRows().Count;
        await SendAndWait("Remember this: my favourite colour is teal. Just answer OK.");
        var reply = await SendAndWait("What is my favourite colour? Answer with one word.");
        Expect(reply.Contains("teal", StringComparison.OrdinalIgnoreCase), $"context kept: '{reply}'");
        Expect(ChatRows().Count == chats, $"follow-ups started new chats ({chats} -> {ChatRows().Count})");
        await Activate(One("NewChatButton"));
        await Until(() => !Visible("UserMessage"), "new chat is empty");
        await Activate(ChatRows().First(r => (r.GetTooltipText() ?? "").Contains(_tag)));
        await Until(() => Shown("UserMessage").Count() == 3, "all three messages saved in the one chat");
    }

    private async Task ChatListSearchAndSwitch()
    {
        await GoToChat();
        var search = (Gtk.SearchEntry)One("ChatSearch");
        search.SetText(_tag);
        await Until(() => ChatTitles().Count > 0 && ChatTitles().All(t => t.Contains(_tag)), "search narrows the chat list");
        search.SetText("");
        await Until(() => ChatTitles().Any(t => !t.Contains(_tag)) || ChatTitles().Count >= 1, "search cleared");

        await Activate(One("NewChatButton"));
        await Until(() => !Visible("UserMessage"), "new chat is empty");
        var row = ChatRows().First(r => (r.GetTooltipText() ?? "").Contains(_tag));
        await Activate(row);
        await Until(() => Shown("UserMessage").OfType<Gtk.Label>().Any(l => l.GetText()!.Contains(_tag)), "reopened chat restores its messages");
        Expect(Shown("AssistantMessage").Any(), "assistant reply restored");
    }

    private async Task StopGeneration()
    {
        await NewChatWith(m => m.CatalogId == "gemma-4-12b" && m.Format == "gguf");
        main.Chat.InputText = $"{_tag} Write a 600-word essay about the history of lighthouses.";
        await Activate(One("SendButton"));
        await Until(() => Shown("AssistantMessage").Any(w => TextOf(w).Length > 40), "reply streaming", 120);
        Expect(Visible("StopButton") && !Visible("SendButton"), "Stop replaces Send while generating");
        await Activate(One("StopButton"));
        await Until(() => !Ioc.Chat.IsGenerating && Visible("SendButton"), "generation stops", 15);
        var length = TextOf(Shown("AssistantMessage").Last()).Length;
        await Task.Delay(1500);
        Expect(TextOf(Shown("AssistantMessage").Last()).Length == length, "no more text after Stop");
    }

    private async Task MtpToggle()
    {
        var model = await NewChatWith(m => m.CatalogId == "gemma-4-e2b" && m.Format == "litertlm" && !m.Quant.Contains("GPU"));
        await Until(() => Visible("MtpToggle"), "MTP toggle shown for a LiteRT model with a drafter");
        var toggle = (Gtk.ToggleButton)One("MtpToggle");
        var wasOn = toggle.GetActive();
        try
        {
            toggle.SetActive(false);
            Expect(AppSettings.Current.MtpOff.Contains(model.Id) && !model.MtpEnabled, "turning MTP off is saved for this model");
            toggle.SetActive(true);
            Expect(!AppSettings.Current.MtpOff.Contains(model.Id) && model.MtpEnabled, "turning MTP on is saved for this model");
        }
        finally
        {
            toggle.SetActive(wasOn);
        }
        await NewChatWith(m => m.CatalogId == "gemma-4-12b" && m.Format == "gguf");
        Expect(!Visible("MtpToggle"), "MTP toggle hidden again for a GGUF model");
    }

    private async Task LiteRtChat()
    {
        await NewChatWith(m => m.CatalogId == "gemma-4-e2b" && m.Format == "litertlm" && !m.Quant.Contains("GPU"));
        await SendAndWait($"{_tag} Name the capital of France in one word.");
        var reply = Shown("AssistantMessage").Select(TextOf).LastOrDefault() ?? "";
        Expect(reply.Contains("Paris", StringComparison.OrdinalIgnoreCase), $"LiteRT-LM reply: '{reply}'");
        Expect((Shown("MessageFooter").OfType<Gtk.Label>().LastOrDefault()?.GetText() ?? "").Contains("tok/s"), "LiteRT-LM footer shows speed");
    }

    private async Task DownloadThinkingModel()
    {
        const string id = "lfm2-5-1-2b-thinking";
        if (Ioc.Models.Installed.FirstOrDefault(m => m.CatalogId == id) is { } existing)
            Ioc.ModelsVm.DeleteInstalledCommand.Execute(existing);
        main.ShowSettings("models");
        await Until(() => Named($"CatalogRow-{id}").Any(), "LFM2.5 1.2B Thinking in the catalog");
        var row = One($"CatalogRow-{id}");
        await Until(() => Descendants(row).Any(w => w.GetName() == "DownloadButton" && w.GetMapped()), "Download button", 30);
        await Activate(Descendants(row).First(w => w.GetName() == "DownloadButton"));
        var status = Descendants(row).OfType<Gtk.Label>().First(l => l.GetName() == "RepoStatus");
        await Until(() => Descendants(row).Any(w => w.GetName() is "DownloadProgress" or "InstalledLabel" && w.GetMapped()),
            $"download starts (status '{status.GetText()}')");
        // The catalog list is rebuilt when the install lands, so look the row up again each time.
        await Until(() => Shown($"CatalogRow-{id}").SelectMany(Descendants).Any(w => w.GetName() == "InstalledLabel" && w.GetMapped()), "Installed after the download", 600);
        Expect(Ioc.Models.Installed.Any(m => m.CatalogId == id), "model added to Installed");
    }

    private async Task ThinkingShown()
    {
        await NewChatWith(m => m.CatalogId == "lfm2-5-1-2b-thinking");
        await SendAndWait($"{_tag} What is 17 times 3? Answer with the number.");
        Expect(Shown("ThinkingExpander").Any(), "reasoning shown in the Thinking section");
        var reply = Shown("AssistantMessage").Select(TextOf).LastOrDefault() ?? "";
        Expect(reply.Contains("51") && !reply.Contains("<think>"), $"answer without raw think tags: '{reply}'");
    }

    private async Task DeleteInstalledModel()
    {
        main.ShowSettings("models");
        ((Adw.ViewStack)One("ModelTabs")).SetVisibleChildName("installed");
        var before = Ioc.Models.Installed.Count;
        var row = Shown("InstalledRow").OfType<Adw.ActionRow>().First(r => r.GetTitle()!.StartsWith("LFM2.5 1.2B Thinking"));
        await Activate(Descendants(row).First(w => w.GetName() == "DeleteInstalledButton"));
        await Until(() => Ioc.Models.Installed.Count == before - 1 && Shown("InstalledRow").Count() == before - 1, "model removed from the list");
        ((Adw.ViewStack)One("ModelTabs")).SetVisibleChildName("catalog");
    }

    private async Task ApiToggle()
    {
        main.ShowSettings("api");
        var toggle = (Adw.SwitchRow)One("ApiToggle");
        var wasOn = toggle.GetActive();
        using var http = new HttpClient { Timeout = TimeSpan.FromSeconds(3) };
        async Task<bool> Reachable()
        {
            try { await http.GetAsync(Ioc.Api.BaseUrl + "/"); return true; }
            catch { return false; }
        }
        try
        {
            toggle.SetActive(true);
            await UntilAsync(Reachable, "API answers when Serving is on");
            toggle.SetActive(false);
            await UntilAsync(async () => !await Reachable(), "API stops when Serving is off");
        }
        finally
        {
            toggle.SetActive(wasOn);
        }
    }

    private async Task DeleteChat()
    {
        await GoToChat();
        ((Gtk.SearchEntry)One("ChatSearch")).SetText("");
        await Task.Delay(400);
        var before = ChatRows().Count(r => (r.GetTooltipText() ?? "").Contains(_tag));
        Expect(before > 0, "test chats exist");
        var row = ChatRows().First(r => (r.GetTooltipText() ?? "").Contains(_tag));
        // The menu a right click opens.
        Descendants(row).OfType<Gtk.Popover>().First().Popup();
        await Until(() => Visible("DeleteChatMenuItem"), "Delete chat menu");
        await Activate(One("DeleteChatMenuItem"));
        await Until(() => ChatRows().Count(r => (r.GetTooltipText() ?? "").Contains(_tag)) == before - 1, "chat removed from the sidebar");
    }

    // ---------------------------------------------------------------- helpers

    private async Task GoToChat()
    {
        main.ShowChat();
        await Until(() => Visible("Composer"), "chat page");
    }

    private async Task<InstalledModel> NewChatWith(Func<InstalledModel, bool> pick)
    {
        await GoToChat();
        await Activate(One("NewChatButton"));
        await Until(() => Ioc.Chat.SelectedChat is null && !Visible("UserMessage") && Visible("EmptyTitle"), "new chat is empty");
        var model = Ioc.Chat.Models.FirstOrDefault(pick) ?? throw new InvalidOperationException("required model is not installed");
        ((Gtk.DropDown)One("ModelPicker")).SetSelected((uint)Ioc.Chat.Models.IndexOf(model));
        Expect(Ioc.Chat.SelectedModel == model, "model picker selects the model");
        return model;
    }

    private async Task<string> SendAndWait(string text)
    {
        var before = Ioc.Chat.Messages.Count;
        main.Chat.InputText = text;
        await Activate(One("SendButton"));
        await Until(() => Ioc.Chat.Messages.Count >= before + 2 && Shown("UserMessage").OfType<Gtk.Label>().Any(l => l.GetText() == text), "message sent", 30);
        await Until(() => !Ioc.Chat.IsGenerating, "reply finished", 300);
        var status = Shown("MessageStatus").OfType<Gtk.Label>().Select(l => l.GetText()).FirstOrDefault(t => t?.Length > 0);
        Expect(status is null, $"no error status: '{status}'");
        return Shown("AssistantMessage").Select(TextOf).LastOrDefault() ?? "";
    }

    private List<Gtk.ListBoxRow> ChatRows()
    {
        var list = (Gtk.ListBox)One("ChatList");
        var rows = new List<Gtk.ListBoxRow>();
        for (var i = 0; list.GetRowAtIndex(i) is { } row; i++)
            if ((row.GetName() ?? "").StartsWith("chat:"))
                rows.Add(row);
        return rows;
    }

    private List<string> ChatTitles() => ChatRows().Select(r => r.GetTooltipText() ?? "").ToList();

    private static IEnumerable<Gtk.Widget> Descendants(Gtk.Widget root)
    {
        for (var child = root.GetFirstChild(); child is not null; child = child.GetNextSibling())
        {
            yield return child;
            foreach (var d in Descendants(child))
                yield return d;
        }
    }

    /// <summary>A label's text, or all label text inside a widget (replies are Markdown boxes of labels).</summary>
    private static string TextOf(Gtk.Widget widget) =>
        widget is Gtk.Label label ? label.GetText() ?? "" : string.Join("\n", Descendants(widget).OfType<Gtk.Label>().Select(l => l.GetText()));

    private IEnumerable<Gtk.Widget> Named(string name) => Descendants(main.Window).Where(w => w.GetName() == name);
    private IEnumerable<Gtk.Widget> Shown(string name) => Named(name).Where(w => w.GetMapped());
    private bool Visible(string name) => Shown(name).Any();

    private Gtk.Widget One(string name) =>
        Shown(name).FirstOrDefault() ?? Named(name).FirstOrDefault() ?? throw new InvalidOperationException($"no widget named {name}");

    /// <summary>Clicks a button or opens a row once it is on screen (a button that is not realized ignores activation).</summary>
    private static async Task Activate(Gtk.Widget widget)
    {
        await Until(() => widget.GetMapped() && widget.IsSensitive(), $"{widget.GetName()} on screen and enabled");
        if (!widget.Activate())
            throw new InvalidOperationException($"{widget.GetName()} could not be activated");
    }

    private static void Expect(bool condition, string what)
    {
        if (!condition)
            throw new InvalidOperationException(what);
    }

    private static Task Until(Func<bool> condition, string what, int seconds = 10) => UntilAsync(() => Task.FromResult(condition()), what, seconds);

    private static async Task UntilAsync(Func<Task<bool>> condition, string what, int seconds = 10)
    {
        var deadline = DateTime.UtcNow.AddSeconds(seconds);
        while (!await condition())
        {
            if (DateTime.UtcNow > deadline)
                throw new TimeoutException($"timed out waiting for: {what}");
            await Task.Delay(200);
        }
    }
}
#endif
