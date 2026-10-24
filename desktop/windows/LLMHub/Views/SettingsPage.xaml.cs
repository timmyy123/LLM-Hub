using LLMHub.Services;
using LLMHub.ViewModels;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Controls.Primitives;
using Microsoft.UI.Xaml.Input;
using Microsoft.UI.Xaml.Navigation;
using Windows.System;

namespace LLMHub.Views;

public sealed partial class SettingsPage : Page
{
    private static readonly int[] ContextOptions = [4096, 8192, 16384, 32768, 65536, 131072];
    private bool _initializing = true;

    public SettingsPage()
    {
        InitializeComponent();

        LanguageBox.Items.Add(new ComboBoxItem { Content = Loc.S("LanguageSystem"), Tag = "" });
        foreach (var (tag, name) in Loc.Languages)
            LanguageBox.Items.Add(new ComboBoxItem { Content = name, Tag = tag });
        LanguageBox.SelectedItem = LanguageBox.Items.Cast<ComboBoxItem>()
            .FirstOrDefault(i => (string)i.Tag == AppSettings.Current.Language) ?? LanguageBox.Items[0];

        foreach (var size in ContextOptions)
            ContextBox.Items.Add(new ComboBoxItem { Content = $"{size / 1024}K", Tag = size });
        ContextBox.SelectedItem = ContextBox.Items.Cast<ComboBoxItem>()
            .FirstOrDefault(i => (int)i.Tag == AppSettings.Current.ContextLength) ?? ContextBox.Items[2];

        TokenBox.Password = AppSettings.Current.HuggingFaceToken;

        foreach (var (value, label) in SettingsViewModel.Appearances)
            AppearanceBox.Items.Add(new ComboBoxItem { Content = label, Tag = value });
        AppearanceBox.SelectedItem = AppearanceBox.Items.Cast<ComboBoxItem>().FirstOrDefault(i => (string)i.Tag == Settings.Appearance) ?? AppearanceBox.Items[0];
        VoiceBox.SelectedItem = Settings.Voice;
        foreach (var (value, label) in SettingsViewModel.SpeechPrecisions)
            SpeechModelBox.Items.Add(new ComboBoxItem { Content = label, Tag = value });
        SpeechModelBox.SelectedItem = SpeechModelBox.Items.Cast<ComboBoxItem>().FirstOrDefault(i => (string)i.Tag == Settings.SpeechPrecision) ?? SpeechModelBox.Items[0];
        foreach (var (code, label) in SettingsViewModel.SpeechLanguages)
            SpeechLanguageBox.Items.Add(new ComboBoxItem { Content = label, Tag = code });
        SpeechLanguageBox.SelectedItem = SpeechLanguageBox.Items.Cast<ComboBoxItem>().FirstOrDefault(i => (string)i.Tag == Settings.SpeechLanguage) ?? SpeechLanguageBox.Items[0];
        SpeechSpeedSlider.Value = Settings.SpeechSpeed;
        DeleteVoiceButton.IsEnabled = !Services.Speech.SpeechService.BuiltInVoices.Contains(Settings.Voice);
        foreach (var size in SettingsViewModel.AgentContexts)
            AgentContextBox.Items.Add(new ComboBoxItem { Content = $"{size / 1024}K", Tag = size });
        AgentContextBox.SelectedItem = AgentContextBox.Items.Cast<ComboBoxItem>().FirstOrDefault(i => (int)i.Tag == Settings.AgentContext) ?? AgentContextBox.Items[1];
        WebSearchToggle.IsOn = Settings.WebSearch;
        foreach (var (minutes, label) in LocalApiServer.UnloadChoices)
            UnloadBox.Items.Add(new ComboBoxItem { Content = label, Tag = minutes });
        UnloadBox.SelectedItem = UnloadBox.Items.Cast<ComboBoxItem>().FirstOrDefault(i => (int)i.Tag == AppSettings.Current.ApiUnloadAfterMinutes) ?? UnloadBox.Items[0];

        var s = AppSettings.Current;
        ApiToggle.IsOn = s.ApiEnabled;
        PortBox.Value = s.ApiPort;
        LanToggle.IsOn = s.ApiAllowLan;
        KeyToggle.IsOn = s.ApiRequireKey;
        if (string.IsNullOrEmpty(s.ApiKey))
        {
            s.ApiKey = LocalApiServer.NewKey();
            AppSettings.Save();
        }
        UpdateApiTexts();
        Api.Requests.CollectionChanged += (_, _) => Bindings.Update();
        BuildModelFilters();
        _initializing = false;
    }

    public ModelsViewModel Models => Ioc.ModelsVm;
    public SettingsViewModel Settings => Ioc.SettingsVm;
    public IReadOnlyList<ApiEndpoint> OpenAiEndpoints { get; } = LocalApiServer.Endpoints.Where(e => !e.Anthropic).ToList();
    public IReadOnlyList<ApiEndpoint> AnthropicEndpoints { get; } = LocalApiServer.Endpoints.Where(e => e.Anthropic).ToList();
    public LocalApiServer Api => Ioc.Api;
    public bool HasApiRequests => Api.Requests.Count > 0;

    private void UpdateApiTexts()
    {
        var s = AppSettings.Current;
        BaseUrlBox.Text = Api.BaseUrl;
        KeyBox.Password = s.ApiKey;
        var model = Ioc.Models.Installed.FirstOrDefault(m => m.Format == "gguf");
        var name = model is null ? "gemma-4-12b" : LocalApiServer.ApiName(model);
        ClaudeSnippet.Text = $"$env:ANTHROPIC_BASE_URL=\"{Api.BaseUrl}\"\n$env:ANTHROPIC_AUTH_TOKEN=\"{(s.ApiRequireKey ? s.ApiKey : "none")}\"\nclaude --model {name}";
    }

    private void OnApiToggled(object sender, RoutedEventArgs e)
    {
        if (_initializing)
            return;
        AppSettings.Current.ApiEnabled = ApiToggle.IsOn;
        AppSettings.Save();
        _ = Api.RestartAsync();
    }

    private void OnPortChanged(NumberBox sender, NumberBoxValueChangedEventArgs args)
    {
        if (_initializing || double.IsNaN(args.NewValue))
            return;
        AppSettings.Current.ApiPort = (int)args.NewValue;
        AppSettings.Save();
        UpdateApiTexts();
        _ = Api.RestartAsync();
    }

    private void OnAccessChanged(object sender, RoutedEventArgs e)
    {
        if (_initializing)
            return;
        var lanChanged = AppSettings.Current.ApiAllowLan != LanToggle.IsOn;
        AppSettings.Current.ApiAllowLan = LanToggle.IsOn;
        AppSettings.Current.ApiRequireKey = KeyToggle.IsOn;
        AppSettings.Save();
        UpdateApiTexts();
        if (lanChanged)
            _ = Api.RestartAsync();
    }

    private void OnNewKey(object sender, RoutedEventArgs e)
    {
        AppSettings.Current.ApiKey = LocalApiServer.NewKey();
        AppSettings.Save();
        UpdateApiTexts();
    }

    private static void Copy(string text)
    {
        var package = new Windows.ApplicationModel.DataTransfer.DataPackage();
        package.SetText(text);
        Windows.ApplicationModel.DataTransfer.Clipboard.SetContent(package);
    }

    private void OnCopyBaseUrl(object sender, RoutedEventArgs e) => Copy(Api.BaseUrl);
    private void OnCopyKey(object sender, RoutedEventArgs e) => Copy(AppSettings.Current.ApiKey);

    protected override void OnNavigatedTo(NavigationEventArgs e) => ShowSection(e.Parameter as string ?? "models");

    /// <summary>Called by the main sidebar's settings list.</summary>
    public void ShowSection(string tag)
    {
        GeneralPanel.Visibility = Ui.Show(tag == "general");
        ModelsPanel.Visibility = Ui.Show(tag == "models");
        ApiPanel.Visibility = Ui.Show(tag == "api");
        EnginesPanel.Visibility = Ui.Show(tag == "engines");
        SpeechPanel.Visibility = Ui.Show(tag == "speech");
        StudioPanel.Visibility = Ui.Show(tag == "studio");
        AgentPanel.Visibility = Ui.Show(tag == "agent");
        WebSearchPanel.Visibility = Ui.Show(tag == "websearch");
        StoragePanel.Visibility = Ui.Show(tag == "storage");
        if (tag == "models")
            _ = Models.LoadCatalogAsync();
        else if (tag is "engines" or "speech" or "studio" or "agent" or "storage")
            _ = Settings.RefreshAsync();
    }

    private void OnModelTabChanged(SelectorBar sender, SelectorBarSelectionChangedEventArgs args)
    {
        var tag = sender.SelectedItem?.Tag as string;
        CatalogPanel.Visibility = Ui.Show(tag == "catalog");
        HfPanel.Visibility = Ui.Show(tag == "hf");
        InstalledPanel.Visibility = Ui.Show(tag == "installed");
        // The Hugging Face tab opens on what is trending, like the design.
        if (tag == "hf" && Models.SearchResults.Count == 0 && !Models.IsSearching)
            Models.SearchCommand.Execute(null);
    }

    /// <summary>Opens the catalog, hf or installed tab (dev captures).</summary>
    public void SelectModelTab(string tag) =>
        ModelTabs.SelectedItem = ModelTabs.Items.FirstOrDefault(i => (string)i.Tag == tag) ?? ModelTabs.SelectedItem;

    private void BuildModelFilters()
    {
        foreach (var (key, label) in ModelsViewModel.Filters)
        {
            var chip = new ToggleButton { Content = label, Tag = key, CornerRadius = new CornerRadius(14), IsChecked = key == Models.Filter, Padding = new Thickness(14, 4, 14, 5) };
            Microsoft.UI.Xaml.Automation.AutomationProperties.SetAutomationId(chip, "Filter_" + key);
            chip.Click += (_, _) =>
            {
                Models.Filter = key;
                foreach (var other in FilterChips.Children.OfType<ToggleButton>())
                    other.IsChecked = (string)other.Tag == key;
            };
            FilterChips.Children.Add(chip);
        }
        foreach (var (key, label) in ModelsViewModel.HfFormats)
            HfFormatBox.Items.Add(new ComboBoxItem { Content = label, Tag = key });
        HfFormatBox.SelectedIndex = 0;
        foreach (var (key, label) in ModelsViewModel.HfSorts)
            HfSortBox.Items.Add(new ComboBoxItem { Content = label, Tag = key });
        HfSortBox.SelectedIndex = 0;
        InstalledTab.Text = $"{Loc.S("TabInstalled")} · {Models.Installed.Count}";
        Models.Installed.CollectionChanged += (_, _) => InstalledTab.Text = $"{Loc.S("TabInstalled")} · {Models.Installed.Count}";
        Models.PropertyChanged += (_, e) =>
        {
            if (e.PropertyName == nameof(ModelsViewModel.SelectedRepo))
                HfTokenPanel.Visibility = Visibility.Collapsed;
        };
    }

    private void OnHfFormatChanged(object sender, SelectionChangedEventArgs e)
    {
        if (!_initializing && HfFormatBox.SelectedItem is ComboBoxItem { Tag: string key })
            Models.HfFormat = key;
    }

    private void OnHfSortChanged(object sender, SelectionChangedEventArgs e)
    {
        if (!_initializing && HfSortBox.SelectedItem is ComboBoxItem { Tag: string key })
            Models.HfSort = key;
    }

    private async void OnAddLocalFile(object sender, RoutedEventArgs e)
    {
        var picker = new Windows.Storage.Pickers.FileOpenPicker();
        picker.FileTypeFilter.Add(".gguf");
        picker.FileTypeFilter.Add(".litertlm");
        WinRT.Interop.InitializeWithWindow.Initialize(picker, WinRT.Interop.WindowNative.GetWindowHandle(App.Window));
        if (await picker.PickSingleFileAsync() is not { } file)
            return;
        AddLocalStatus.Text = Models.AddLocalFile(file.Path);
        AddLocalStatus.Visibility = Visibility.Visible;
    }

    private async void OnOpenOnHub(object sender, RoutedEventArgs e)
    {
        if (Models.SelectedRepo is { } repo)
            await Launcher.LaunchUriAsync(new Uri(repo.Url));
    }

    /// <summary>Opens Hugging Face's token page and shows a token field right here, for gated repos.</summary>
    private async void OnHfSignIn(object sender, RoutedEventArgs e)
    {
        HfTokenBox.Password = AppSettings.Current.HuggingFaceToken;
        HfTokenPanel.Visibility = Visibility.Visible;
        await Launcher.LaunchUriAsync(new Uri("https://huggingface.co/settings/tokens"));
    }

    private void OnHfTokenChanged(object sender, RoutedEventArgs e)
    {
        AppSettings.Current.HuggingFaceToken = HfTokenBox.Password;
        AppSettings.Save();
        TokenBox.Password = HfTokenBox.Password;
        if (HfTokenBox.Password.Length > 0 && Models.SelectedRepo is { } repo)
            _ = repo.LoadVariantsAsync(CancellationToken.None);
    }

    private void OnHfSearchKeyDown(object sender, KeyRoutedEventArgs e)
    {
        if (e.Key != VirtualKey.Enter)
            return;
        e.Handled = true;
        Models.SearchCommand.Execute(null);
    }

    private void OnDeleteInstalled(object sender, RoutedEventArgs e)
    {
        if (sender is FrameworkElement { Tag: InstalledModel model })
            Models.DeleteInstalledCommand.Execute(model);
    }

    private void OnLanguageChanged(object sender, SelectionChangedEventArgs e)
    {
        if (_initializing || LanguageBox.SelectedItem is not ComboBoxItem { Tag: string tag })
            return;
        AppSettings.Current.Language = tag;
        AppSettings.Save();
        Loc.ApplyLanguage(tag);
        RestartHint.Visibility = Visibility.Visible;
    }

    private void OnContextChanged(object sender, SelectionChangedEventArgs e)
    {
        if (_initializing || ContextBox.SelectedItem is not ComboBoxItem { Tag: int size })
            return;
        AppSettings.Current.ContextLength = size;
        AppSettings.Save();
    }

    private void OnTokenChanged(object sender, RoutedEventArgs e)
    {
        if (_initializing)
            return;
        AppSettings.Current.HuggingFaceToken = TokenBox.Password;
        AppSettings.Save();
    }

    private void OnAppearanceChanged(object sender, SelectionChangedEventArgs e)
    {
        if (_initializing || AppearanceBox.SelectedItem is not ComboBoxItem { Tag: string tag })
            return;
        Settings.Appearance = tag;
        App.Window?.ApplyAppearance();
    }

    private void OnUnloadChanged(object sender, SelectionChangedEventArgs e)
    {
        if (_initializing || UnloadBox.SelectedItem is not ComboBoxItem { Tag: int minutes })
            return;
        AppSettings.Current.ApiUnloadAfterMinutes = minutes;
        AppSettings.Save();
    }

    private void OnVoiceChanged(object sender, SelectionChangedEventArgs e)
    {
        if (!_initializing && VoiceBox.SelectedItem is string voice)
            Settings.Voice = voice;
        if (VoiceBox.SelectedItem is string selected)
            DeleteVoiceButton.IsEnabled = !Services.Speech.SpeechService.BuiltInVoices.Contains(selected);
    }

    private async void OnImportVoice(object sender, RoutedEventArgs e)
    {
        var picker = new Windows.Storage.Pickers.FileOpenPicker();
        picker.FileTypeFilter.Add(".json");
        WinRT.Interop.InitializeWithWindow.Initialize(picker, WinRT.Interop.WindowNative.GetWindowHandle(App.Window));
        if (await picker.PickSingleFileAsync() is not { } file)
            return;
        Settings.ImportVoice(file.Path);
        VoiceBox.SelectedItem = Settings.Voice;
    }

    private void OnDeleteVoice(object sender, RoutedEventArgs e)
    {
        if (VoiceBox.SelectedItem is not string voice)
            return;
        Settings.DeleteVoiceCommand.Execute(voice);
        VoiceBox.SelectedItem = Settings.Voice;
    }

    private void OnSpeechModelChanged(object sender, SelectionChangedEventArgs e)
    {
        if (!_initializing && SpeechModelBox.SelectedItem is ComboBoxItem { Tag: string value })
            Settings.SpeechPrecision = value;
    }

    private void OnSpeechLanguageChanged(object sender, SelectionChangedEventArgs e)
    {
        if (!_initializing && SpeechLanguageBox.SelectedItem is ComboBoxItem { Tag: string code })
            Settings.SpeechLanguage = code;
    }

    private void OnSpeechSpeedChanged(object sender, Microsoft.UI.Xaml.Controls.Primitives.RangeBaseValueChangedEventArgs e)
    {
        if (!_initializing)
            Settings.SpeechSpeed = e.NewValue;
    }

    private void OnAgentContextChanged(object sender, SelectionChangedEventArgs e)
    {
        if (!_initializing && AgentContextBox.SelectedItem is ComboBoxItem { Tag: int size })
            Settings.AgentContext = size;
    }

    private void OnWebSearchToggled(object sender, RoutedEventArgs e)
    {
        if (!_initializing)
            Settings.WebSearch = WebSearchToggle.IsOn;
    }

    private async void OnOpenFolder(object sender, RoutedEventArgs e)
    {
        if (sender is FrameworkElement { Tag: string folder } && Directory.Exists(folder))
            await Launcher.LaunchFolderPathAsync(folder);
    }

    private void OnDeleteStudioModel(object sender, RoutedEventArgs e)
    {
        if (sender is FrameworkElement { Tag: StorageItem item })
            Settings.DeleteStudioModelCommand.Execute(item);
    }

    private void OnClearStorage(object sender, RoutedEventArgs e)
    {
        if (sender is FrameworkElement { Tag: StorageItem item })
            Settings.ClearCommand.Execute(item);
    }
}
