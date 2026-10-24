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
        // Popups and flyouts follow the application theme, not the window's, so set it here (it cannot change later).
        if (AppSettings.Current.Appearance == "light")
            RequestedTheme = ApplicationTheme.Light;
        else if (AppSettings.Current.Appearance != "system")
            RequestedTheme = ApplicationTheme.Dark;
        InitializeComponent();
        UnhandledException += (_, e) => Log.Error("Unhandled", e.Exception);
        Services.Studio.Npu.SdNpuService.ImageLoader = NpuImages.LoadRgbAsync;
        Services.Studio.Npu.LtxNpuService.VideoEncoder = NpuImages.EncodeMp4Async;
    }

    protected override void OnLaunched(LaunchActivatedEventArgs args)
    {
        Window = new MainWindow();
        Window.Closed += (_, _) =>
        {
            Ioc.Engines.Stop();
            Ioc.Embeddings.Stop();
            Ioc.Speech.Stop();
            Ioc.Comfy.Stop();
            Ioc.SdNpu.Unload();
        };
        Window.Activate();
        _ = Ioc.Api.RestartAsync();
#if DEBUG
        DevCapture.RunIfRequested(Window);
#endif
    }
}
