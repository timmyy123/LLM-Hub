using LLMHub.Services;
using LLMHub.ViewModels;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
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
        _initializing = false;
    }

    public ModelsViewModel Models => Ioc.ModelsVm;
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
        if (tag == "models")
            _ = Models.LoadCatalogAsync();
    }

    private void OnModelTabChanged(SelectorBar sender, SelectorBarSelectionChangedEventArgs args)
    {
        var tag = sender.SelectedItem?.Tag as string;
        CatalogList.Visibility = Ui.Show(tag == "catalog");
        HfPanel.Visibility = Ui.Show(tag == "hf");
        InstalledPanel.Visibility = Ui.Show(tag == "installed");
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
}
