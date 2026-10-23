using LLMHub.Linux;
using LLMHub.Services;

// Child process that tries LiteRT-LM's GPU backend, which can crash outright on a bad driver.
if (args is ["--litert-probe", .. var probe])
    return LiteRtEngine.RunProbe(probe);

AppSettings.Load();
Loc.ApplyLanguage(AppSettings.Current.Language);
Adw.Module.Initialize();

var mainThread = Environment.CurrentManagedThreadId;
UiThread.Initialize(
    action => GLib.Functions.IdleAdd(GLib.Constants.PRIORITY_DEFAULT, () =>
    {
        action();
        return false;
    }),
    () => Environment.CurrentManagedThreadId == mainThread);

var app = Adw.Application.New("app.llmhub.Desktop", Gio.ApplicationFlags.DefaultFlags);
MainWindow? window = null;
app.OnActivate += (_, _) =>
{
    if (window is null)
    {
        Styles.Load();
        window = new MainWindow(app);
        _ = Ioc.Api.RestartAsync();
#if DEBUG
        DevTools.RunIfRequested(window, args);
#endif
    }
    window.Present();
};
app.OnShutdown += (_, _) =>
{
    Ioc.Engines.Stop();
    Ioc.Speech.Stop();
};
return app.RunWithSynchronizationContext(null);
