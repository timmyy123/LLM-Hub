using LLMHub.Services;
using LLMHub.ViewModels;

namespace LLMHub.Linux;

/// <summary>
/// The Settings sections beyond General, Models and Local API: Engines, Speech and voice, Studio backends, Agent,
/// Web search and Storage, as libadwaita preference pages over the shared SettingsViewModel.
/// </summary>
internal sealed class SettingsSections
{
    private readonly SettingsViewModel _vm = Ioc.SettingsVm;
    private readonly Adw.PreferencesGroup _engines = Adw.PreferencesGroup.New();
    private readonly Adw.ActionRow _tts = Adw.ActionRow.New();
    private readonly Adw.ActionRow _asr = Adw.ActionRow.New();
    private readonly Gtk.Label _speechStatus = Ui.Label("", "caption", "dim-label");
    private readonly Adw.ActionRow _comfy = Adw.ActionRow.New();
    private readonly Adw.PreferencesGroup _studioModels = Adw.PreferencesGroup.New();
    private readonly Adw.ActionRow _claude = Adw.ActionRow.New();
    private readonly Gtk.Button _installClaude;
    private readonly Adw.PreferencesGroup _storage = Adw.PreferencesGroup.New();
    private readonly List<Gtk.Widget> _engineRows = [];
    private readonly List<Gtk.Widget> _studioRows = [];
    private readonly List<Gtk.Widget> _storageRows = [];

    public Gtk.Widget Engines { get; }
    public Gtk.Widget Speech { get; }
    public Gtk.Widget Studio { get; }
    public Gtk.Widget Agent { get; }
    public Gtk.Widget WebSearch { get; }
    public Gtk.Widget Storage { get; }

    public SettingsSections()
    {
        _installClaude = Ui.Button(Loc.S("AgentInstall"), () => _vm.InstallClaudeCommand.Execute(null), "suggested-action");
        Engines = Page(Group(Loc.S("SettingsEngines"), Loc.S("SettingsEnginesHelp")), _engines);
        Speech = BuildSpeech();
        Studio = BuildStudio();
        Agent = BuildAgent();
        WebSearch = BuildWebSearch();
        Storage = BuildStorage();

        _vm.Engines.CollectionChanged += (_, _) => UiThread.Run(RebuildEngines);
        _vm.StudioModels.CollectionChanged += (_, _) => UiThread.Run(RebuildStudioModels);
        _vm.Storage.CollectionChanged += (_, _) => UiThread.Run(RebuildStorage);
        Ui.Watch(_vm, _ => Sync());
        Ui.Watch(Ioc.Comfy, _ => Sync());
        Sync();
    }

    /// <summary>Called when a section opens: re-reads engines and folder sizes.</summary>
    public void Refresh() => _ = _vm.RefreshAsync();

    private static Adw.PreferencesPage Page(params Adw.PreferencesGroup[] groups)
    {
        var page = Adw.PreferencesPage.New();
        foreach (var g in groups)
            page.Add(g);
        return page;
    }

    private static Adw.PreferencesGroup Group(string title, string description)
    {
        var group = Adw.PreferencesGroup.New();
        group.SetTitle(title);
        group.SetDescription(description);
        return group;
    }

    private void Sync()
    {
        _tts.SetSubtitle(_vm.TtsLine);
        _asr.SetSubtitle(_vm.AsrLine);
        _speechStatus.SetText(_vm.SpeechStatus);
        _comfy.SetTitle(_vm.ComfyLine);
        _claude.SetSubtitle(_vm.ClaudePath);
        _claude.SetTitle(_vm.ClaudeVersion.Length > 0 ? $"{Loc.S("SettingsClaudeCode")} · {_vm.ClaudeVersion}" : Loc.S("SettingsClaudeCode"));
        _installClaude.SetVisible(!_vm.ClaudeInstalled);
        _storage.SetDescription(Loc.S("SettingsStorageHelp") + (_vm.StorageTotal.Length > 0 ? "  " + Loc.F("SettingsTotal", _vm.StorageTotal) : ""));
    }

    private static void Replace(Adw.PreferencesGroup group, List<Gtk.Widget> rows, IEnumerable<Gtk.Widget> fresh)
    {
        foreach (var row in rows)
            group.Remove(row);
        rows.Clear();
        foreach (var row in fresh)
        {
            group.Add(row);
            rows.Add(row);
        }
    }

    // ---------------------------------------------------------------- Engines

    private void RebuildEngines() => Replace(_engines, _engineRows, _vm.Engines.Select(e =>
    {
        var row = Adw.ActionRow.New();
        row.SetTitle(e.Version.Length > 0 ? $"{e.Name} · {e.Version}" : e.Name);
        row.SetSubtitle(e.Detail);
        row.Named("EngineCard");
        var dot = Gtk.Box.New(Gtk.Orientation.Horizontal, 0);
        dot.AddCssClass("dot");
        if (e.Installed)
            dot.AddCssClass("ready");
        dot.SetValign(Gtk.Align.Center);
        row.AddPrefix(dot);
        if (e.Folder.Length > 0)
            row.AddSuffix(OpenFolder(e.Folder));
        return (Gtk.Widget)row;
    }));

    private static Gtk.Button OpenFolder(string folder)
    {
        var button = Ui.IconButton("folder-open-symbolic", Loc.S("SettingsOpenFolder"), () =>
        {
            if (Directory.Exists(folder))
                Ui.Open(folder);
        }, "flat");
        button.SetValign(Gtk.Align.Center);
        return button;
    }

    // ---------------------------------------------------------------- Speech and voice

    private Gtk.Widget BuildSpeech()
    {
        var group = Group(Loc.S("SettingsSpeech"), Loc.S("SettingsSpeechHelp"));
        _tts.SetTitle(Loc.S("SettingsTts"));
        _tts.Named("TtsLine");
        _asr.SetTitle(Loc.S("SettingsAsr"));
        _asr.Named("AsrLine");
        var voice = Adw.ComboRow.New();
        voice.SetTitle(Loc.S("StudioVoice"));
        voice.SetModel(Gtk.StringList.New(_vm.Voices.ToArray()));
        voice.SetSelected((uint)Math.Max(0, _vm.Voices.ToList().IndexOf(_vm.Voice)));
        voice.OnNotify += (_, e) =>
        {
            if (e.Pspec.GetName() == "selected" && voice.GetSelected() < _vm.Voices.Count)
                _vm.Voice = _vm.Voices[(int)voice.GetSelected()];
        };
        var test = Ui.Button(Loc.S("SettingsTestVoice"), () => _vm.TestVoiceCommand.Execute(null), "flat");
        test.SetValign(Gtk.Align.Center);
        voice.AddSuffix(test);
        group.Add(_tts);
        group.Add(_asr);
        group.Add(voice);
        var actions = Adw.PreferencesGroup.New();
        var buttons = Ui.HBox(10,
            Ui.Button(Loc.S("SettingsSetUpNow"), () => _vm.SetUpSpeechCommand.Execute(null), "suggested-action").Named("SetUpSpeechButton"),
            Ui.Button(Loc.S("SettingsRemove"), () => _vm.RemoveSpeechCommand.Execute(null)),
            _speechStatus);
        actions.Add(buttons);
        return Page(group, actions);
    }

    // ---------------------------------------------------------------- Studio backends

    private Gtk.Widget BuildStudio()
    {
        var group = Group(Loc.S("SettingsStudio"), Loc.S("SettingsStudioHelp"));
        _comfy.SetSubtitle(_vm.ComfyFolder);
        _comfy.Named("ComfyLine");
        foreach (var (label, action) in new (string, Action)[]
                 {
                     (Loc.S("SettingsStart"), () => _vm.StartComfyCommand.Execute(null)),
                     (Loc.S("SettingsStop"), () => _vm.StopComfyCommand.Execute(null)),
                 })
        {
            var button = Ui.Button(label, action, "flat");
            button.SetValign(Gtk.Align.Center);
            _comfy.AddSuffix(button);
        }
        _comfy.AddSuffix(OpenFolder(_vm.ComfyFolder));
        group.Add(_comfy);
        _studioModels.SetTitle(Loc.S("SettingsStudioModels"));
        var music = Group(Loc.S("SettingsMusic"), _vm.MusicLine);
        return Page(group, _studioModels, music);
    }

    private void RebuildStudioModels() => Replace(_studioModels, _studioRows, _vm.StudioModels.Select(item =>
    {
        var row = Adw.ActionRow.New();
        row.SetTitle(item.Name);
        row.SetSubtitle(item.Size);
        var delete = Ui.IconButton("user-trash-symbolic", Loc.S("Delete"), () => _vm.DeleteStudioModelCommand.Execute(item), "flat");
        delete.SetValign(Gtk.Align.Center);
        row.AddSuffix(delete);
        return (Gtk.Widget)row;
    }));

    // ---------------------------------------------------------------- Agent

    private Gtk.Widget BuildAgent()
    {
        var group = Group(Loc.S("SettingsAgent"), Loc.S("SettingsAgentHelp"));
        _claude.Named("ClaudeVersion");
        _installClaude.SetValign(Gtk.Align.Center);
        _claude.AddSuffix(_installClaude);
        group.Add(_claude);
        var context = Adw.ComboRow.New();
        context.SetTitle(Loc.S("SettingsAgentContext"));
        context.SetSubtitle(Loc.S("SettingsAgentContextHelp"));
        var sizes = SettingsViewModel.AgentContexts;
        context.SetModel(Gtk.StringList.New(sizes.Select(c => $"{c / 1024}K").ToArray()));
        context.SetSelected((uint)Math.Max(0, sizes.ToList().IndexOf(_vm.AgentContext)));
        context.OnNotify += (_, e) =>
        {
            if (e.Pspec.GetName() == "selected" && context.GetSelected() < sizes.Count)
                _vm.AgentContext = sizes[(int)context.GetSelected()];
        };
        group.Add(context);
        var note = Adw.PreferencesGroup.New();
        note.SetDescription(Loc.S("SettingsAgentNote"));
        return Page(group, note);
    }

    // ---------------------------------------------------------------- Web search

    private Gtk.Widget BuildWebSearch()
    {
        var group = Group(Loc.S("SettingsWebSearch"), Loc.S("SettingsWebSearchHelp"));
        var toggle = Adw.SwitchRow.New();
        toggle.SetTitle(Loc.S("SettingsWebSearchDefault"));
        toggle.SetActive(_vm.WebSearch);
        toggle.OnNotify += (_, e) =>
        {
            if (e.Pspec.GetName() == "active")
                _vm.WebSearch = toggle.GetActive();
        };
        toggle.Named("WebSearchToggle");
        group.Add(toggle);
        return Page(group);
    }

    // ---------------------------------------------------------------- Storage

    private Gtk.Widget BuildStorage()
    {
        _storage.SetTitle(Loc.S("SettingsStorage"));
        _storage.Named("StorageGroup");
        return Page(_storage);
    }

    private void RebuildStorage() => Replace(_storage, _storageRows, _vm.Storage.Select(item =>
    {
        var row = Adw.ActionRow.New();
        row.SetTitle(item.Name);
        row.SetSubtitle(item.Folder);
        row.Named("StorageRow");
        var size = Ui.Label(item.Size, "dim-label");
        size.SetWrap(false);
        size.SetValign(Gtk.Align.Center);
        row.AddSuffix(size);
        if (item.CanClear)
        {
            var clear = Ui.Button(Loc.S("SettingsClear"), () => _vm.ClearCommand.Execute(item), "flat");
            clear.SetValign(Gtk.Align.Center);
            row.AddSuffix(clear);
        }
        row.AddSuffix(OpenFolder(item.Folder));
        return (Gtk.Widget)row;
    }));
}
