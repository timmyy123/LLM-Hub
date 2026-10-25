using LLMHub.Linux;
using LLMHub.Services;

// Child process that tries LiteRT-LM's GPU backend, which can crash outright on a bad driver.
if (args is ["--litert-probe", .. var probe])
    return LiteRtEngine.RunProbe(probe);
if (args is ["--litert-embed-probe", .. var embedProbe])
    return LiteRtEngine.RunEmbedProbe(embedProbe);

AppSettings.Load();
Loc.ApplyLanguage(AppSettings.Current.Language);
Adw.Module.Initialize();
// WebKitGTK's bubblewrap sandbox starts under WSL but its web process never receives page content, so the Design
// preview stays blank; real Linux desktops keep the sandbox.
if (Environment.GetEnvironmentVariable("WSL_DISTRO_NAME") is { Length: > 0 })
    Native.setenv("WEBKIT_DISABLE_SANDBOX_THIS_IS_DANGEROUS", "1", 1);
try
{
    WebKit.Module.Initialize();
}
catch (Exception e)
{
    // The Design preview says what to install; everything else works without WebKitGTK.
    Log.Error("WebKitGTK 6.0 not available", e);
}

var mainThread = Environment.CurrentManagedThreadId;
UiThread.Initialize(
    action => GLib.Functions.IdleAdd(GLib.Constants.PRIORITY_DEFAULT, () =>
    {
        action();
        return false;
    }),
    () => Environment.CurrentManagedThreadId == mainThread);
LLMHub.Linux.SearchPane.RegisterDecoders();

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
    Ioc.Embeddings.Stop();
    Ioc.Speech.Stop();
    Ioc.Comfy.Stop();
    Ioc.OpenClaw.StopGateway();
    Ioc.Magenta.Shutdown();
    _ = Ioc.MediaEmbedder.UnloadAsync();
};
return app.RunWithSynchronizationContext(null);

internal static partial class Native
{
    // Environment.SetEnvironmentVariable only changes .NET's copy; WebKit reads the process environment.
    [System.Runtime.InteropServices.LibraryImport("libc", StringMarshalling = System.Runtime.InteropServices.StringMarshalling.Utf8)]
    internal static partial int setenv(string name, string value, int overwrite);
}
