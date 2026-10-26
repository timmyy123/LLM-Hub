using System.ComponentModel;
using LLMHub.Services;
using LLMHub.Services.Design;
using LLMHub.ViewModels;
using Microsoft.UI.Input;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Controls.Primitives;
using Microsoft.UI.Xaml.Input;
using Microsoft.Web.WebView2.Core;
using Windows.Storage.Pickers;
using Windows.System;
using Windows.UI.Core;

namespace LLMHub.Views;

public sealed partial class DesignPage : Page
{
    public DesignPage()
    {
        InitializeComponent();
        ViewModel.PropertyChanged += OnViewModelChanged;
        Unloaded += (_, _) => ViewModel.PropertyChanged -= OnViewModelChanged;
        Loaded += async (_, _) =>
        {
            await Preview.EnsureCoreWebView2Async();
            Preview.CoreWebView2.Settings.AreDevToolsEnabled = false;
            Preview.CoreWebView2.WebMessageReceived += OnPreviewMessage;
            Preview.CoreWebView2.NavigationCompleted += async (_, _) =>
            {
                await Preview.CoreWebView2.ExecuteScriptAsync(DesignService.PickerScript("window.chrome.webview.postMessage"));
                if (ViewModel.NeedsPhoneCheck)
                    await Preview.CoreWebView2.ExecuteScriptAsync(DesignService.PhoneCheckScript("window.chrome.webview.postMessage"));
            };
            ShowPreview();
        };
        UpdateTabs();
        SizeChanged += OnPageSizeChanged;
    }

    public DesignViewModel ViewModel => Ioc.DesignVm;

    private void OnViewModelChanged(object? sender, PropertyChangedEventArgs e)
    {
        if (e.PropertyName is nameof(DesignViewModel.PreviewFile))
            ShowPreview();
        if (e.PropertyName is nameof(DesignViewModel.Kind) or nameof(DesignViewModel.PreviewSize))
            UpdateTabs();
        if (e.PropertyName is nameof(DesignViewModel.LiveText) or nameof(DesignViewModel.Selected))
            DispatcherQueue.TryEnqueue(() => ConversationScroll.ChangeView(null, ConversationScroll.ScrollableHeight, null, true));
    }

    private void ShowPreview()
    {
        if (Preview.CoreWebView2 is null)
            return;
        if (ViewModel.PreviewFile is { } file && File.Exists(file))
            Preview.CoreWebView2.Navigate(new Uri(file).AbsoluteUri);
        else
            Preview.CoreWebView2.NavigateToString("<html><body style='background:#fff'></body></html>");
    }

    private void OnPreviewMessage(CoreWebView2 sender, CoreWebView2WebMessageReceivedEventArgs args) =>
        ViewModel.OnPreviewMessage(args.WebMessageAsJson);

#if DEBUG
    /// <summary>What WebView2 loaded (RenderTargetBitmap cannot capture it), after clicking the first heading through the picker.</summary>
    public async Task ProbePreviewAsync()
    {
        if (Preview.CoreWebView2 is null)
            return;
        var result = await Preview.CoreWebView2.ExecuteScriptAsync(
            "JSON.stringify({title: document.title, chars: document.body.innerText.length, elements: document.querySelectorAll('*').length, picker: !!window.__llmhubPicker, " +
            "clicked: (() => { const h = document.querySelector('h1'); if (h) h.click(); return h ? h.innerText.slice(0, 40) : null; })()})");
        await Task.Delay(500);
        Log.Info($"PREVIEW {Preview.Source} {result} selected='{ViewModel.SelectedElement}'");
    }
#endif

    /// <summary>Fills the Projects menu: every design (current one checked), then New and Delete.</summary>
    private void OnProjectsOpening(object sender, object e)
    {
        ProjectsMenu.Items.Clear();
        foreach (var project in ViewModel.Projects)
        {
            var item = new ToggleMenuFlyoutItem
            {
                Text = project.Name.Length > 0 ? project.Name : Loc.S("DesignUntitled"),
                IsChecked = project == ViewModel.Selected,
            };
            item.Click += (_, _) => ViewModel.Selected = project;
            ProjectsMenu.Items.Add(item);
        }
        if (ViewModel.Projects.Count > 0)
            ProjectsMenu.Items.Add(new MenuFlyoutSeparator());
        var create = new MenuFlyoutItem { Text = Loc.S("DesignNewProject"), Icon = new FontIcon { Glyph = "\uE710" } };
        create.Click += (_, _) => ViewModel.NewProjectCommand.Execute(null);
        ProjectsMenu.Items.Add(create);
        if (ViewModel.Selected is { } current)
        {
            var delete = new MenuFlyoutItem { Text = Loc.S("Delete"), Icon = new FontIcon { Glyph = "\uE74D" } };
            delete.Click += (_, _) => ViewModel.DeleteProjectCommand.Execute(current);
            ProjectsMenu.Items.Add(delete);
        }
    }

    /// <summary>
    /// On a narrow window Comment and Hand off drop their labels so Export stays visible. Measured from the page: the
    /// toolbar itself reports its desired width, not the visible part of the preview column.
    /// </summary>
    private void OnPageSizeChanged(object sender, SizeChangedEventArgs e)
    {
        var width = e.NewSize.Width - 340 - 260;
        var labels = width >= 720 ? Visibility.Visible : Visibility.Collapsed;
        CommentLabel.Visibility = labels;
        HandOffLabel.Visibility = labels;
        VersionLabel.Visibility = width >= 600 ? Visibility.Visible : Visibility.Collapsed;
        // Below that the size tabs become icons (desktop, tablet, phone).
        var icons = width < 640;
        foreach (var (tab, glyph, key) in new[] { (SizeDesktop, "\uE7F4", "DesignDesktop"), (SizeTablet, "\uE70A", "DesignTablet"), (SizePhone, "\uE8EA", "DesignPhone") })
        {
            tab.Content = icons ? new FontIcon { Glyph = glyph, FontSize = 13 } : Loc.S(key);
            ToolTipService.SetToolTip(tab, icons ? Loc.S(key) : null);
        }
    }

    private void OnKindClick(object sender, RoutedEventArgs e)
    {
        if (sender is FrameworkElement { Tag: string tag })
        {
            ViewModel.Kind = ViewModel.Kinds.First(k => k.Kind.ToString() == tag);
            // A different kind starts a new project, so a deck never overwrites a prototype.
            if (ViewModel.Selected is { } p && p.Kind != ViewModel.Kind.Kind && p.Versions > 0)
                ViewModel.NewProjectCommand.Execute(null);
            else if (ViewModel.Selected is { Versions: 0 } empty)
                empty.Kind = ViewModel.Kind.Kind;
        }
        UpdateTabs();
    }

    private void OnSizeClick(object sender, RoutedEventArgs e)
    {
        if (sender is FrameworkElement { Tag: string size })
            ViewModel.PreviewSize = size;
        UpdateTabs();
    }

    private void UpdateTabs()
    {
        foreach (var tab in new ToggleButton[] { KindPrototype, KindSlides, KindImage, KindMotion })
            tab.IsChecked = (string)tab.Tag == ViewModel.Kind.Kind.ToString();
        foreach (var tab in new ToggleButton[] { SizeDesktop, SizeTablet, SizePhone })
            tab.IsChecked = (string)tab.Tag == ViewModel.PreviewSize;
    }

    private void OnComposerKeyDown(object sender, KeyRoutedEventArgs e)
    {
        if (e.Key != VirtualKey.Enter || InputKeyboardSource.GetKeyStateForCurrentThread(VirtualKey.Shift).HasFlag(CoreVirtualKeyStates.Down))
            return;
        e.Handled = true;
        if (ViewModel.SendCommand.CanExecute(null))
            ViewModel.SendCommand.Execute(null);
    }

    /// <summary>Comment: focuses the composer with the selected element named, so the note becomes the next change.</summary>
    private void OnComment(object sender, RoutedEventArgs e)
    {
        if (ViewModel.SelectedElement.Length > 0 && ViewModel.Draft.Length == 0)
            ViewModel.Draft = "";
        DesignComposer.Focus(FocusState.Programmatic);
    }

    private async void OnHandOff(object sender, RoutedEventArgs e)
    {
        await ViewModel.HandOffToAgentAsync();
        App.Window?.ShowSection("agent");
    }

    private async void OnExport(object sender, RoutedEventArgs e)
    {
        if (sender is not FrameworkElement { Tag: string format } || ViewModel.Selected is not { } project || project.CurrentFile is null)
            return;
        var picker = new FileSavePicker { SuggestedFileName = DesignViewModel.Safe(project.Name) };
        picker.FileTypeChoices.Add(format.ToUpperInvariant(), [format switch { "md" => ".md", "zip" => ".zip", "pdf" => ".pdf", "pptx" => ".pptx", _ => ".html" }]);
        WinRT.Interop.InitializeWithWindow.Initialize(picker, WinRT.Interop.WindowNative.GetWindowHandle(App.Window));
        if (await picker.PickSaveFileAsync() is not { } file)
            return;
        try
        {
            if (format == "pdf")
                await Preview.CoreWebView2.PrintToPdfAsync(file.Path, null);
            else
                await DesignService.ExportAsync(project, format, file.Path);
            ViewModel.Status = Loc.F("StudioSavedTo", file.Path);
        }
        catch (Exception ex)
        {
            Log.Error("Design export failed", ex);
            ViewModel.Status = Loc.F("ErrorPrefix", ex.Message);
        }
    }
}
