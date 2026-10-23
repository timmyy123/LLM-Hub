using System.Collections.Specialized;
using System.ComponentModel;
using LLMHub.Services;
using LLMHub.ViewModels;

namespace LLMHub.Linux;

/// <summary>General, Models (catalog, Hugging Face, installed) and Local API pages, as libadwaita preference pages.</summary>
internal sealed class SettingsPages
{
    private static readonly int[] ContextOptions = [4096, 8192, 16384, 32768, 65536, 131072];
    private readonly MainWindow _main;
    private readonly ModelsViewModel _models = Ioc.ModelsVm;

    public Gtk.Widget General { get; }
    public Gtk.Widget Models { get; }
    public Gtk.Widget Api { get; }

    public SettingsPages(MainWindow main)
    {
        _main = main;
        General = BuildGeneral();
        Models = BuildModels();
        Api = BuildApi();
    }

    // ---------------------------------------------------------------- General

    private Gtk.Widget BuildGeneral()
    {
        var page = Adw.PreferencesPage.New();
        var group = Adw.PreferencesGroup.New();
        group.SetTitle(Loc.S("SettingsGeneral"));

        var language = Adw.ComboRow.New();
        language.SetTitle(Loc.S("Language"));
        var tags = new[] { "" }.Concat(Loc.Languages.Select(l => l.Tag)).ToArray();
        language.SetModel(Gtk.StringList.New(new[] { Loc.S("LanguageSystem") }.Concat(Loc.Languages.Select(l => l.Name)).ToArray()));
        language.SetSelected((uint)Math.Max(0, Array.IndexOf(tags, AppSettings.Current.Language)));
        var restart = Adw.ActionRow.New();
        restart.SetTitle(Loc.S("LanguageRestart"));
        restart.SetVisible(false);
        restart.Named("RestartHint");
        language.OnNotify += (_, e) =>
        {
            if (e.Pspec.GetName() != "selected")
                return;
            AppSettings.Current.Language = tags[language.GetSelected()];
            AppSettings.Save();
            restart.SetVisible(true);
        };
        language.Named("LanguageBox");

        var context = Adw.ComboRow.New();
        context.SetTitle(Loc.S("ContextLength"));
        context.SetModel(Gtk.StringList.New(ContextOptions.Select(c => $"{c / 1024}K").ToArray()));
        context.SetSelected((uint)Math.Max(0, Array.IndexOf(ContextOptions, AppSettings.Current.ContextLength)));
        context.OnNotify += (_, e) =>
        {
            if (e.Pspec.GetName() != "selected")
                return;
            AppSettings.Current.ContextLength = ContextOptions[context.GetSelected()];
            AppSettings.Save();
        };
        context.Named("ContextBox");

        var token = Adw.PasswordEntryRow.New();
        token.SetTitle(Loc.S("HfToken"));
        token.SetText(AppSettings.Current.HuggingFaceToken);
        token.OnChanged += (_, _) =>
        {
            AppSettings.Current.HuggingFaceToken = token.GetText();
            AppSettings.Save();
        };
        token.Named("TokenBox");

        group.Add(language);
        group.Add(restart);
        group.Add(context);
        group.Add(token);
        var help = Adw.PreferencesGroup.New();
        help.SetDescription(Loc.S("HfTokenHelp"));
        page.Add(group);
        page.Add(help);
        return page;
    }

    // ---------------------------------------------------------------- Models

    private Gtk.Widget BuildModels()
    {
        var stack = Adw.ViewStack.New();
        stack.AddTitledWithIcon(BuildCatalog(), "catalog", Loc.S("TabCatalog"), "view-list-symbolic");
        stack.AddTitledWithIcon(BuildHuggingFace(), "hf", Loc.S("TabHuggingFace"), "system-search-symbolic");
        stack.AddTitledWithIcon(BuildInstalled(), "installed", Loc.S("TabInstalled"), "drive-harddisk-symbolic");
        stack.Named("ModelTabs");
        var switcher = Adw.ViewSwitcher.New();
        switcher.SetStack(stack);
        switcher.SetPolicy(Adw.ViewSwitcherPolicy.Wide);
        switcher.SetHalign(Gtk.Align.Start);

        var header = Ui.VBox(4, Ui.Label(Loc.S("SettingsModels"), "title-2"), Ui.Label(_models.HardwareText, "dim-label"));
        var top = Ui.VBox(14, header, switcher).Margins(24, 28, 8, 28);
        stack.SetVexpand(true);
        return Ui.VBox(0, top, stack);
    }

    private Gtk.Widget BuildCatalog()
    {
        var list = Gtk.ListBox.New();
        list.AddCssClass("boxed-list");
        list.SetSelectionMode(Gtk.SelectionMode.None);
        list.Named("CatalogList");
        void Rebuild()
        {
            list.RemoveAll();
            foreach (var item in _models.Catalog)
                list.Append(ModelRow(item, item.Name, item.Subtitle, item.EngineLabel, $"CatalogRow-{item.Model.Id}"));
        }
        _models.Catalog.CollectionChanged += (_, _) => Rebuild();
        Rebuild();
        return Scroll(list);
    }

    private Gtk.Widget BuildHuggingFace()
    {
        var query = Gtk.SearchEntry.New();
        query.SetPlaceholderText(Loc.S("SearchHf"));
        query.Named("HfQuery");
        query.SetHexpand(true);
        query.OnActivate += (_, _) =>
        {
            _models.Query = query.GetText();
            _models.SearchCommand.Execute(null);
        };
        var search = Ui.IconButton("system-search-symbolic", Loc.S("SearchHf"), () =>
        {
            _models.Query = query.GetText();
            _models.SearchCommand.Execute(null);
        }).Named("HfSearchButton");

        var results = Gtk.ListBox.New();
        results.AddCssClass("boxed-list");
        results.Named("HfResults");
        var detail = Gtk.Box.New(Gtk.Orientation.Vertical, 12).Named("HfDetail");
        detail.SetHexpand(true);
        results.OnRowSelected += (_, e) =>
        {
            if (e.Row is { } row && row.GetIndex() < _models.SearchResults.Count)
                _models.SelectedRepo = _models.SearchResults[row.GetIndex()];
        };
        _models.SearchResults.CollectionChanged += (_, _) =>
        {
            results.RemoveAll();
            foreach (var repo in _models.SearchResults)
            {
                var row = Adw.ActionRow.New();
                row.SetTitle(repo.Id);
                row.SetSubtitle(repo.Subtitle);
                row.AddCssClass("monospace");
                results.Append(row);
            }
        };
        _models.PropertyChanged += (_, e) =>
        {
            if (e.PropertyName != nameof(ModelsViewModel.SelectedRepo))
                return;
            detail.Clear();
            if (_models.SelectedRepo is { } repo)
            {
                var list = Gtk.ListBox.New();
                list.AddCssClass("boxed-list");
                list.SetSelectionMode(Gtk.SelectionMode.None);
                list.Append(ModelRow(repo, repo.Id, Loc.S("HfHint"), "llama.cpp", "HfRepoRow"));
                detail.Append(list);
            }
        };

        var left = Ui.VBox(10, Ui.HBox(8, query, search), Scroll(results, margins: false));
        left.SetSizeRequest(380, -1);
        return Ui.HBox(16, left, detail).Margins(8, 28, 24, 28);
    }

    private Gtk.Widget BuildInstalled()
    {
        var list = Gtk.ListBox.New();
        list.AddCssClass("boxed-list");
        list.SetSelectionMode(Gtk.SelectionMode.None);
        list.Named("InstalledList");
        var empty = Ui.Label(Loc.S("NoInstalled"), "dim-label");
        void Rebuild()
        {
            list.RemoveAll();
            foreach (var model in Ioc.Models.Installed)
            {
                var row = Adw.ActionRow.New();
                row.SetTitle(model.DisplayName);
                row.SetSubtitle($"{model.Repo} · {model.SizeText}");
                row.Named("InstalledRow");
                row.AddSuffix(Ui.IconButton("user-trash-symbolic", Loc.S("Delete"), () => Ioc.ModelsVm.DeleteInstalledCommand.Execute(model), "flat").Named("DeleteInstalledButton"));
                list.Append(row);
            }
            empty.SetVisible(Ioc.Models.Installed.Count == 0);
            list.SetVisible(Ioc.Models.Installed.Count > 0);
        }
        Ioc.Models.Installed.CollectionChanged += (_, _) => Rebuild();
        Rebuild();
        return Scroll(Ui.VBox(12, empty, list));
    }

    /// <summary>A model row: name and tags, a quant picker, and Download / progress + cancel / Installed + delete.</summary>
    private static Gtk.Widget ModelRow(DownloadableViewModel item, string title, string subtitle, string engine, string name)
    {
        var row = Adw.ActionRow.New();
        row.SetTitle(title);
        row.SetSubtitle(subtitle);
        row.Named(name);
        var tag = Ui.Label(engine, "tag");
        tag.SetWrap(false);
        tag.SetValign(Gtk.Align.Center);
        row.AddPrefix(tag);

        var variants = Gtk.DropDown.NewFromStrings([]).Named("VariantPicker");
        variants.SetValign(Gtk.Align.Center);
        var download = Ui.Button(Loc.S("Download"), () => item.DownloadCommand.Execute(null), "suggested-action").Named("DownloadButton");
        var progress = Gtk.ProgressBar.New().Named("DownloadProgress");
        progress.SetShowText(true);
        progress.SetSizeRequest(150, -1);
        var cancel = Ui.IconButton("process-stop-symbolic", Loc.S("Cancel"), () => item.CancelCommand.Execute(null), "flat").Named("CancelDownloadButton");
        var installed = Ui.Label("✓ " + Loc.S("Installed"), "success").Named("InstalledLabel");
        installed.SetWrap(false);
        var delete = Ui.IconButton("user-trash-symbolic", Loc.S("Delete"), () => item.DeleteCommand.Execute(null), "flat").Named("DeleteModelButton");
        var status = Ui.Label("", "caption", "dim-label").Named("RepoStatus");
        foreach (var w in new Gtk.Widget[] { variants, download, progress, cancel, installed, delete })
        {
            w.SetValign(Gtk.Align.Center);
            row.AddSuffix(w);
        }
        status.SetMaxWidthChars(48);
        row.AddSuffix(status);

        var syncing = false;
        variants.OnNotify += (_, e) =>
        {
            if (!syncing && e.Pspec.GetName() == "selected" && variants.GetSelected() is var i && i < item.Variants.Count)
                item.SelectedVariant = item.Variants[(int)i];
        };
        PropertyChangedEventHandler? jobHandler = null;
        DownloadJob? watched = null;
        void SyncVariants(bool rebuild)
        {
            syncing = true;
            if (rebuild)
                variants.SetModel(Gtk.StringList.New(item.Variants.Select(v => v.ToString()).ToArray()));
            var index = item.SelectedVariant is { } s ? item.Variants.IndexOf(s) : -1;
            var selected = index >= 0 ? (uint)index : uint.MaxValue;
            if (variants.GetSelected() != selected)
                variants.SetSelected(selected);
            syncing = false;
        }
        void Update()
        {
            variants.SetVisible(item.CanPick);
            download.SetVisible(item.CanDownload);
            progress.SetVisible(item.IsDownloading);
            cancel.SetVisible(item.IsDownloading);
            installed.SetVisible(item.IsInstalled);
            delete.SetVisible(item.IsInstalled);
            status.SetText(item.Status);
            status.SetVisible(item.Status.Length > 0);
            if (!ReferenceEquals(watched, item.Job))
            {
                if (watched is not null && jobHandler is not null)
                    watched.PropertyChanged -= jobHandler;
                watched = item.Job;
                if (watched is not null)
                {
                    jobHandler = (_, _) =>
                    {
                        progress.SetFraction(watched.Percent / 100);
                        progress.SetText(watched.Status);
                    };
                    watched.PropertyChanged += jobHandler;
                }
            }
        }
        item.PropertyChanged += (_, e) =>
        {
            Update();
            // May run inside the drop-down's own "selected" notification (see ChatPage.SyncModels), so sync afterwards.
            if (e.PropertyName == nameof(DownloadableViewModel.SelectedVariant))
                GLib.Functions.IdleAdd(GLib.Constants.PRIORITY_DEFAULT, () =>
                {
                    SyncVariants(rebuild: false);
                    return false;
                });
        };
        item.Variants.CollectionChanged += (_, _) =>
        {
            SyncVariants(rebuild: true);
            Update();
        };
        SyncVariants(rebuild: true);
        Update();
        return row;
    }

    // ---------------------------------------------------------------- Local API

    private Gtk.Widget BuildApi()
    {
        var s = AppSettings.Current;
        if (string.IsNullOrEmpty(s.ApiKey))
        {
            s.ApiKey = LocalApiServer.NewKey();
            AppSettings.Save();
        }
        var page = Adw.PreferencesPage.New();

        var main = Adw.PreferencesGroup.New();
        main.SetTitle(Loc.S("SettingsLocalApi"));
        main.SetDescription(Loc.S("ApiIntro"));
        var serving = Adw.SwitchRow.New();
        serving.SetTitle(Loc.S("ApiServing"));
        serving.SetActive(s.ApiEnabled);
        serving.Named("ApiToggle");
        serving.OnNotify += (_, e) =>
        {
            if (e.Pspec.GetName() != "active")
                return;
            s.ApiEnabled = serving.GetActive();
            AppSettings.Save();
            RestartApi();
        };
        var error = Adw.ActionRow.New();
        error.AddCssClass("error");
        error.Named("ApiError");
        main.Add(serving);
        main.Add(error);

        var access = Adw.PreferencesGroup.New();
        access.SetTitle(Loc.S("ApiAccess"));
        var baseUrl = Adw.ActionRow.New();
        baseUrl.SetTitle(Loc.S("ApiBaseUrl"));
        baseUrl.AddCssClass("property");
        baseUrl.AddSuffix(Ui.IconButton("edit-copy-symbolic", Loc.S("Copy"), () => Ui.Copy(baseUrl, Ioc.Api.BaseUrl), "flat"));
        var port = Adw.SpinRow.NewWithRange(1024, 65535, 1);
        port.SetTitle(Loc.S("ApiPort"));
        port.SetValue(s.ApiPort);
        port.Named("PortBox");
        var lan = Adw.SwitchRow.New();
        lan.SetTitle(Loc.S("ApiAllowLan"));
        lan.SetSubtitle(Loc.S("ApiAllowLanHelp"));
        lan.SetActive(s.ApiAllowLan);
        var requireKey = Adw.SwitchRow.New();
        requireKey.SetTitle(Loc.S("ApiRequireKey"));
        requireKey.SetActive(s.ApiRequireKey);
        var key = Adw.ActionRow.New();
        key.SetTitle(Loc.S("ApiKey"));
        key.AddSuffix(Ui.IconButton("edit-copy-symbolic", Loc.S("Copy"), () => Ui.Copy(key, AppSettings.Current.ApiKey), "flat"));
        var snippet = Ui.Label("", "monospace", "caption");
        snippet.SetSelectable(true);
        key.AddSuffix(Ui.Button(Loc.S("ApiNewKey"), () =>
        {
            s.ApiKey = LocalApiServer.NewKey();
            AppSettings.Save();
            UpdateTexts();
        }, "flat"));
        access.Add(baseUrl);
        access.Add(port);
        access.Add(lan);
        access.Add(requireKey);
        access.Add(key);

        var endpoints = Adw.PreferencesGroup.New();
        endpoints.SetTitle(Loc.S("ApiEndpoints"));
        var list = Ui.Label("POST /v1/chat/completions   (OpenAI)\nPOST /v1/messages           (Anthropic)\nPOST /v1/messages/count_tokens\nGET  /v1/models", "monospace");
        list.SetSelectable(true);
        endpoints.Add(list);
        endpoints.Add(Ui.Label(Loc.S("ApiClaudeCode"), "dim-label").Margins(12, 0, 4, 0));
        endpoints.Add(snippet);

        var recent = Adw.PreferencesGroup.New();
        recent.SetTitle(Loc.S("ApiRecent"));
        var requests = Gtk.ListBox.New();
        requests.AddCssClass("boxed-list");
        requests.SetSelectionMode(Gtk.SelectionMode.None);
        var none = Ui.Label(Loc.S("ApiNoRequests"), "dim-label");
        recent.Add(none);
        recent.Add(requests);

        port.OnNotify += (_, e) =>
        {
            if (e.Pspec.GetName() != "value")
                return;
            s.ApiPort = (int)port.GetValue();
            AppSettings.Save();
            UpdateTexts();
            RestartApi();
        };
        lan.OnNotify += (_, e) =>
        {
            if (e.Pspec.GetName() != "active")
                return;
            s.ApiAllowLan = lan.GetActive();
            AppSettings.Save();
            RestartApi();
        };
        requireKey.OnNotify += (_, e) =>
        {
            if (e.Pspec.GetName() != "active")
                return;
            s.ApiRequireKey = requireKey.GetActive();
            AppSettings.Save();
            UpdateTexts();
        };
        Ioc.Api.PropertyChanged += (_, _) => UpdateTexts();
        Ioc.Api.Requests.CollectionChanged += (_, _) =>
        {
            requests.RemoveAll();
            foreach (var r in Ioc.Api.Requests)
            {
                var row = Adw.ActionRow.New();
                row.SetTitle(string.IsNullOrEmpty(r.Model) ? r.Path : $"{r.Path}  ·  {r.Model}");
                row.SetSubtitle(string.IsNullOrEmpty(r.Client) ? r.Time : $"{r.Time}  ·  {r.Client}");
                var status = Ui.Label(r.Status).Named("ApiRequestStatus");
                status.SetWrap(false);
                row.AddSuffix(status);
                requests.Append(row);
            }
            none.SetVisible(Ioc.Api.Requests.Count == 0);
        };

        void UpdateTexts()
        {
            baseUrl.SetSubtitle(Ioc.Api.BaseUrl);
            key.SetSubtitle(s.ApiKey[..Math.Min(14, s.ApiKey.Length)] + "…");
            error.SetTitle(Ioc.Api.Error);
            error.SetVisible(Ioc.Api.Error.Length > 0);
            var model = Ioc.Models.Installed.FirstOrDefault(m => m.Format == "gguf");
            snippet.SetText($"export ANTHROPIC_BASE_URL={Ioc.Api.BaseUrl}\nexport ANTHROPIC_AUTH_TOKEN={(s.ApiRequireKey ? s.ApiKey : "none")}\nclaude --model {(model is null ? "gemma-4-12b" : LocalApiServer.ApiName(model))}");
        }
        UpdateTexts();

        page.Add(main);
        page.Add(access);
        page.Add(endpoints);
        page.Add(recent);
        return page;
    }

    private static void RestartApi() => Ioc.Api.RestartAsync().ContinueWith(t => Log.Error("Local API restart failed", t.Exception), TaskContinuationOptions.OnlyOnFaulted);

    private static Gtk.Widget Scroll(Gtk.Widget child, bool margins = true)
    {
        var clamp = Adw.Clamp.New();
        clamp.SetMaximumSize(1100);
        clamp.SetChild(margins ? child.Margins(8, 28, 24, 28) : child);
        var scroll = Gtk.ScrolledWindow.New();
        scroll.SetChild(clamp);
        scroll.SetVexpand(true);
        scroll.SetPolicy(Gtk.PolicyType.Never, Gtk.PolicyType.Automatic);
        return scroll;
    }
}
