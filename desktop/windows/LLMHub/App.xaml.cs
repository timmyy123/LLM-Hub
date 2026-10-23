using LLMHub.Services;
using Microsoft.UI.Xaml;

namespace LLMHub;

public partial class App : Application
{
    public static MainWindow? Window { get; private set; }

    public App()
    {
        AppSettings.Load();
        Loc.ApplyLanguage(AppSettings.Current.Language);
        InitializeComponent();
        UnhandledException += (_, e) => Log.Error("Unhandled", e.Exception);
    }

    protected override void OnLaunched(LaunchActivatedEventArgs args)
    {
        Window = new MainWindow();
        Window.Closed += (_, _) => Ioc.Engines.Stop();
        Window.Activate();
        _ = Ioc.Api.RestartAsync();
#if DEBUG
        DevCapture.RunIfRequested(Window);
#endif
    }
}
