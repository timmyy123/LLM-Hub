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

        var appearance = Adw.ComboRow.New();
        appearance.SetTitle(Loc.S("Appearance"));
        var looks = SettingsViewModel.Appearances;
        appearance.SetModel(Gtk.StringList.New(looks.Select(a => a.Label).ToArray()));
        appearance.SetSelected((uint)Math.Max(0, looks.ToList().FindIndex(a => a.Value == AppSettings.Current.Appearance)));
        appearance.OnNotify += (_, e) =>
        {
            if (e.Pspec.GetName() != "selected" || appearance.GetSelected() >= looks.Count)
                return;
            Ioc.SettingsVm.Appearance = looks[(int)appearance.GetSelected()].Value;
            Styles.ApplyAppearance();
        };
        appearance.Named("AppearanceBox");

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
        group.Add(appearance);
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
        var installedPage = stack.AddTitledWithIcon(BuildInstalled(), "installed", Loc.S("TabInstalled"), "drive-harddisk-symbolic");
        void CountInstalled() => installedPage.SetTitle($"{Loc.S("TabInstalled")} · {Ioc.Models.Installed.Count}");
        Ioc.Models.Installed.CollectionChanged += (_, _) => UiThread.Run(CountInstalled);
        CountInstalled();
        stack.Named("ModelTabs");
        // The Hugging Face tab opens on what is trending, like the design.
        stack.OnNotify += (_, e) =>
        {
            if (e.Pspec.GetName() == "visible-child-name" && stack.GetVisibleChildName() == "hf" && _models.SearchResults.Count == 0 && !_models.IsSearching)
                _models.SearchCommand.Execute(null);
        };
        var switcher = Adw.ViewSwitcher.New();
        switcher.SetStack(stack);
        switcher.SetPolicy(Adw.ViewSwitcherPolicy.Wide);
        switcher.SetHalign(Gtk.Align.Start);

        var addStatus = Ui.Label("", "caption", "dim-label").Named("AddLocalStatus");
        addStatus.SetVisible(false);
        var addLocal = Gtk.Button.New();
        addLocal.SetChild(Ui.HBox(8, Gtk.Image.NewFromIconName("list-add-symbolic"), Ui.Label(Loc.S("ModelsAddLocalFile"))));
        addLocal.SetValign(Gtk.Align.Start);
        addLocal.Named("AddLocalButton");
        addLocal.OnClicked += async (_, _) =>
        {
            try
            {
                var dialog = Gtk.FileDialog.New();
                var filter = Gtk.FileFilter.New();
                filter.AddSuffix("gguf");
                filter.AddSuffix("litertlm");
                var filters = Gio.ListStore.New(Gtk.FileFilter.GetGType());
                filters.Append(filter);
                dialog.SetFilters(filters);
                if (await dialog.OpenAsync(_main.Window) is { } file && file.GetPath() is { } path)
                {
                    addStatus.SetText(_models.AddLocalFile(path));
                    addStatus.SetVisible(true);
                }
            }
            catch (GLib.GException)
            {
                // Dismissed.
            }
        };
        var titles = Ui.VBox(4, Ui.Label(Loc.S("SettingsModels"), "title-2"), Ui.Label(_models.HardwareText, "dim-label"), addStatus);
        titles.SetHexpand(true);
        var top = Ui.VBox(14, Ui.HBox(16, titles, addLocal), switcher).Margins(24, 28, 8, 28);
        stack.SetVexpand(true);
        return Ui.VBox(0, top, stack);
    }

    private Gtk.Widget BuildCatalog()
    {
        // Engine cards: "GGUF · CUDA · running" and ".litertlm · GPU · ready".
        Gtk.Widget EngineCard(string title, Func<string> line, Func<bool> running, string name)
        {
            var dot = Gtk.Box.New(Gtk.Orientation.Horizontal, 0);
            dot.AddCssClass("dot");
            dot.SetValign(Gtk.Align.Center);
            var text = Ui.Label("", "caption", "dim-label").Named(name);
            void Sync()
            {
                text.SetText(line());
                if (running())
                    dot.AddCssClass("ready");
                else
                    dot.RemoveCssClass("ready");
            }
            Ui.Watch(_models, _ => Sync());
            Sync();
            var card = Ui.VBox(0, Ui.HBox(10, dot, Ui.VBox(2, Ui.Label(title, "heading"), text)).Margins(10, 14, 10, 14));
            card.AddCssClass("card");
            card.SetHexpand(true);
            return card;
        }
        var engines = Ui.HBox(12,
            EngineCard("llama.cpp", () => _models.LlamaEngineLine, () => _models.LlamaRunning, "LlamaEngineLine"),
            EngineCard("LiteRT-LM", () => _models.LiteRtEngineLine, () => _models.LiteRtRunning, "LiteRtEngineLine"));

        // Filter chips.
        var chips = Ui.HBox(8);
        Gtk.ToggleButton? first = null;
        foreach (var (key, label) in ModelsViewModel.Filters)
        {
            var chip = Gtk.ToggleButton.NewWithLabel(label);
            chip.AddCssClass("pill");
            chip.Named("Filter_" + key);
            if (first is null)
                first = chip;
            else
                chip.SetGroup(first);
            chip.SetActive(key == _models.Filter);
            chip.OnToggled += (_, _) =>
            {
                if (chip.GetActive())
                    _models.Filter = key;
            };
            chips.Append(chip);
        }

        var list = Gtk.ListBox.New();
        list.AddCssClass("boxed-list");
        list.SetSelectionMode(Gtk.SelectionMode.None);
        list.Named("CatalogList");
        void Rebuild()
        {
            list.RemoveAll();
            foreach (var item in _models.VisibleCatalog)
                list.Append(ModelRow(item, item.Name, item.Subtitle, item.EngineLabel, $"CatalogRow-{item.Model.Id}", showFit: true));
        }
        _models.VisibleCatalog.CollectionChanged += (_, _) => UiThread.Run(Rebuild);
        Rebuild();
        return Scroll(Ui.VBox(14, engines, chips, list));
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
        var formats = new Picker(ModelsViewModel.HfFormats.Select(f => f.Label), i => _models.HfFormat = ModelsViewModel.HfFormats[i].Key);
        formats.DropDown.Named("HfFormatBox");
        formats.DropDown.SetHexpand(true);
        var sorts = new Picker(ModelsViewModel.HfSorts.Select(f => f.Label), i => _models.HfSort = ModelsViewModel.HfSorts[i].Key);
        sorts.DropDown.Named("HfSortBox");
        sorts.DropDown.SetHexpand(true);
        sorts.DropDown.SetTooltipText(Loc.S("HfSort"));

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
        _models.SearchResults.CollectionChanged += (_, _) => UiThread.Run(() =>
        {
            results.RemoveAll();
            foreach (var repo in _models.SearchResults)
            {
                var row = Adw.ActionRow.New();
                row.SetTitle(repo.Id);
                row.SetSubtitle(repo.Meta);
                row.AddCssClass("monospace");
                var format = Ui.Label(repo.FormatLabel, "tag");
                format.SetWrap(false);
                format.SetValign(Gtk.Align.Center);
                row.AddSuffix(format);
                results.Append(row);
            }
        });
        var error = Ui.Label("", "warning").Named("HfSearchError");
        error.SetVisible(false);
        _models.PropertyChanged += (_, e) =>
        {
            if (e.PropertyName == nameof(ModelsViewModel.HfSearchError))
                UiThread.Run(() =>
                {
                    error.SetText(_models.HfSearchError ?? "");
                    error.SetVisible(_models.HfSearchError is not null);
                });
            if (e.PropertyName != nameof(ModelsViewModel.SelectedRepo))
                return;
            detail.Clear();
            if (_models.SelectedRepo is { } repo)
                detail.Append(RepoDetail(repo));
        };

        var left = Ui.VBox(10, Ui.HBox(8, query, search), Ui.HBox(8, formats.DropDown, sorts.DropDown), error, Scroll(results, margins: false));
        left.SetSizeRequest(340, -1);
        return Ui.HBox(16, left, Scroll(detail, margins: false)).Margins(8, 28, 24, 28);
    }

    /// <summary>The selected repo: name, what it is, a gated notice with sign-in, and its files with how each fits.</summary>
    private Gtk.Widget RepoDetail(HfRepoViewModel repo)
    {
        var id = Ui.Label(repo.Id, "title-4", "monospace");
        id.SetSelectable(true);
        var info = Ui.VBox(4, id, Ui.Label(repo.Detail, "dim-label").Named("RepoDetail"));
        info.SetHexpand(true);
        var open = Gtk.LinkButton.NewWithLabel(repo.Url, Loc.S("HfOpenOnHub"));
        open.SetValign(Gtk.Align.Start);

        var token = Adw.PasswordEntryRow.New();
        token.SetTitle(Loc.S("HfToken"));
        token.SetText(AppSettings.Current.HuggingFaceToken);
        token.OnChanged += (_, _) =>
        {
            AppSettings.Current.HuggingFaceToken = token.GetText();
            AppSettings.Save();
        };
        token.OnApply += (_, _) => _ = repo.LoadVariantsAsync(CancellationToken.None);
        token.SetShowApplyButton(true);
        var tokenList = Gtk.ListBox.New();
        tokenList.AddCssClass("boxed-list");
        tokenList.Append(token);
        var tokenBox = Ui.VBox(6, tokenList, Ui.Label(Loc.S("HfSignInHelp"), "caption", "dim-label"));
        tokenBox.SetVisible(false);
        var signIn = Ui.Button(Loc.S("HfSignIn"), () =>
        {
            tokenBox.SetVisible(true);
            Ui.Open("https://huggingface.co/settings/tokens");
        });
        signIn.SetVisible(repo.NeedsSignIn);
        signIn.SetValign(Gtk.Align.Center);
        var gatedText = Ui.Label(Loc.S("HfGated"), "warning");
        gatedText.SetHexpand(true);
        var gated = Ui.HBox(12, gatedText, signIn).Named("HfGatedBar");
        gated.SetVisible(repo.IsGated);

        var status = Ui.Label(repo.Status, "caption", "dim-label").Named("RepoStatus");
        var files = Gtk.ListBox.New();
        files.AddCssClass("boxed-list");
        files.SetSelectionMode(Gtk.SelectionMode.None);
        files.Named("HfFileTable");
        void Rows()
        {
            files.RemoveAll();
            foreach (var r in repo.Rows)
            {
                var row = Adw.ActionRow.New();
                row.SetTitle(r.QuantAndSize);
                row.SetSubtitle(r.File);
                row.AddCssClass("monospace");
                if (r.IsBest)
                    row.AddCssClass("accent");
                var fit = Ui.Label(r.FitText, r.FitLevel switch { "ok" => "success", "warn" => "warning", _ => "error" }).Named("HfFileFit");
                fit.SetWrap(false);
                row.AddSuffix(fit);
                if (r.IsInstalled)
                    row.AddSuffix(Ui.Label("✓ " + Loc.S("Installed"), "success"));
                else
                {
                    var download = Ui.Button(Loc.S("Download"), () => r.DownloadCommand.Execute(null), r.IsBest ? "suggested-action" : "flat").Named("HfFileDownload");
                    download.SetSensitive(r.CanDownload);
                    download.SetValign(Gtk.Align.Center);
                    row.AddSuffix(download);
                }
                files.Append(row);
            }
            files.SetVisible(repo.Rows.Count > 0);
        }
        repo.Rows.CollectionChanged += (_, _) => UiThread.Run(Rows);
        Rows();
        // Download progress for the file being fetched (the shared row shows the bar and Cancel).
        var progressList = Gtk.ListBox.New();
        progressList.AddCssClass("boxed-list");
        progressList.SetSelectionMode(Gtk.SelectionMode.None);
        progressList.Append(ModelRow(repo, repo.Id, "", repo.FormatLabel, "HfRepoRow"));
        void Sync()
        {
            status.SetText(repo.Status);
            status.SetVisible(repo.Status.Length > 0);
            progressList.SetVisible(repo.IsDownloading);
        }
        Ui.Watch(repo, _ => Sync());
        Sync();

        var columns = Ui.Label($"{Loc.S("HfColFile")} · {Loc.S("HfColFit")} · {Loc.S("HfColAction")}", "caption", "dim-label");
        return Ui.VBox(12, Ui.HBox(12, info, open), gated, tokenBox, status, columns, files, progressList,
            Ui.Label(Loc.S("HfMmprojNote"), "caption", "dim-label"));
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

    /// <summary>A model row: name and tags, a quant picker (with its fit, for the catalog), and Download / progress + cancel / Installed + delete.</summary>
    private static Gtk.Widget ModelRow(DownloadableViewModel item, string title, string subtitle, string engine, string name, bool showFit = false)
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
        var fit = Ui.Label("", "caption").Named("CatalogFit");
        fit.SetWrap(false);
        if (showFit)
            row.AddSuffix(fit);
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
            fit.SetText(item.FitText);
            foreach (var css in new[] { "success", "warning", "error" })
                fit.RemoveCssClass(css);
            fit.AddCssClass(item.FitLevel switch { "ok" => "success", "warn" => "warning", _ => "error" });
            fit.SetVisible(showFit && item.FitText.Length > 0);
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

        // Endpoint tables from the design: OpenAI-compatible, then Anthropic-compatible with the Claude Code setup.
        Adw.PreferencesGroup EndpointGroup(string title, bool anthropic)
        {
            var group = Adw.PreferencesGroup.New();
            group.SetTitle(title);
            var rows = Gtk.ListBox.New();
            rows.AddCssClass("boxed-list");
            rows.SetSelectionMode(Gtk.SelectionMode.None);
            rows.Named(anthropic ? "AnthropicEndpoints" : "OpenAiEndpoints");
            foreach (var e in LocalApiServer.Endpoints.Where(e => e.Anthropic == anthropic))
            {
                var row = Adw.ActionRow.New();
                row.SetTitle($"{e.Method,-4} {e.Path}");
                row.AddCssClass("monospace");
                row.SetTitleSelectable(true);
                if (e.ServedBy.Length > 0)
                {
                    var servedBy = Ui.Label(e.ServedBy, "dim-label");
                    servedBy.SetWrap(false);
                    row.AddSuffix(servedBy);
                }
                rows.Append(row);
            }
            group.Add(rows);
            return group;
        }
        var openAi = EndpointGroup(Loc.S("ApiOpenAiCompatible"), anthropic: false);
        var endpoints = EndpointGroup(Loc.S("ApiAnthropicCompatible"), anthropic: true);
        endpoints.Add(Ui.Label(Loc.S("ApiClaudeCode"), "dim-label").Margins(16, 0, 4, 0));
        endpoints.Add(snippet);

        // Models offered, each with the engine that serves it, and when idle models are unloaded.
        var offered = Adw.PreferencesGroup.New();
        offered.SetTitle(Loc.S("ApiModelsOffered"));
        offered.SetDescription(Loc.S("ApiModelsOfferedHelp"));
        var unload = Adw.ComboRow.New();
        unload.SetTitle(Loc.S("ApiUnloadAfter"));
        unload.SetModel(Gtk.StringList.New(LocalApiServer.UnloadChoices.Select(c => c.Label).ToArray()));
        unload.SetSelected((uint)Math.Max(0, LocalApiServer.UnloadChoices.ToList().FindIndex(c => c.Minutes == s.ApiUnloadAfterMinutes)));
        unload.Named("UnloadPicker");
        unload.OnNotify += (_, e) =>
        {
            if (e.Pspec.GetName() != "selected" || unload.GetSelected() >= LocalApiServer.UnloadChoices.Count)
                return;
            s.ApiUnloadAfterMinutes = LocalApiServer.UnloadChoices[(int)unload.GetSelected()].Minutes;
            AppSettings.Save();
        };
        offered.Add(unload);
        var models = Gtk.ListBox.New();
        models.AddCssClass("boxed-list");
        models.SetSelectionMode(Gtk.SelectionMode.None);
        models.Named("ApiModelsOffered");
        models.SetMarginTop(12);
        offered.Add(models);
        void ListModels()
        {
            models.RemoveAll();
            foreach (var m in Ioc.Models.Installed)
            {
                var row = Adw.ActionRow.New();
                row.SetTitle(LocalApiServer.ApiName(m));
                row.SetTitleSelectable(true);
                row.AddCssClass("monospace");
                var engine = Ui.Label(LocalApiServer.EngineLabel(m), "dim-label");
                engine.SetWrap(false);
                row.AddSuffix(engine);
                models.Append(row);
            }
            models.SetVisible(Ioc.Models.Installed.Count > 0);
        }
        Ioc.Models.Installed.CollectionChanged += (_, _) => UiThread.Run(ListModels);
        ListModels();

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
        page.Add(openAi);
        page.Add(endpoints);
        page.Add(offered);
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
