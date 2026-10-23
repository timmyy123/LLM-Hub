namespace LLMHub.Services;

/// <summary>Runs work on the UI thread. Each app plugs in its toolkit's dispatcher (WinUI DispatcherQueue, GLib main loop).</summary>
public static class UiThread
{
    private static Action<Action>? _post;
    private static Func<bool>? _hasAccess;

    public static void Initialize(Action<Action> post, Func<bool> hasAccess)
    {
        _post = post;
        _hasAccess = hasAccess;
    }

    /// <summary>Always queues, so collection and selection changes never re-enter a binding that is mid-update.</summary>
    public static void Post(Action action)
    {
        if (_post is null)
            action();
        else
            _post(action);
    }

    public static void Run(Action action)
    {
        if (_post is null || _hasAccess!())
            action();
        else
            _post(action);
    }
}
