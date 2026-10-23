namespace LLMHub.Linux;

/// <summary>Small GTK helpers so pages read like a layout, plus the few app-specific CSS rules.</summary>
internal static class Ui
{
    public static Gtk.Box VBox(int spacing = 0, params Gtk.Widget[] children) => Box(Gtk.Orientation.Vertical, spacing, children);
    public static Gtk.Box HBox(int spacing = 0, params Gtk.Widget[] children) => Box(Gtk.Orientation.Horizontal, spacing, children);

    private static Gtk.Box Box(Gtk.Orientation orientation, int spacing, Gtk.Widget[] children)
    {
        var box = Gtk.Box.New(orientation, spacing);
        foreach (var child in children)
            box.Append(child);
        return box;
    }

    public static Gtk.Label Label(string text, params string[] css)
    {
        var label = Gtk.Label.New(text);
        label.SetXalign(0);
        label.SetWrap(true);
        label.SetWrapMode(Pango.WrapMode.WordChar);
        foreach (var c in css)
            label.AddCssClass(c);
        return label;
    }

    public static Gtk.Button Button(string label, Action onClick, params string[] css)
    {
        var button = Gtk.Button.NewWithLabel(label);
        foreach (var c in css)
            button.AddCssClass(c);
        button.OnClicked += (_, _) => onClick();
        return button;
    }

    public static Gtk.Button IconButton(string icon, string tooltip, Action onClick, params string[] css)
    {
        var button = Gtk.Button.NewFromIconName(icon);
        button.SetTooltipText(tooltip);
        foreach (var c in css)
            button.AddCssClass(c);
        button.OnClicked += (_, _) => onClick();
        return button;
    }

    /// <summary>Gives a widget a name the built-in UI test runner (and the GTK inspector) can find.</summary>
    public static T Named<T>(this T widget, string name) where T : Gtk.Widget
    {
        widget.SetName(name);
        return widget;
    }

    public static T Margins<T>(this T widget, int top, int right, int bottom, int left) where T : Gtk.Widget
    {
        widget.SetMarginTop(top);
        widget.SetMarginEnd(right);
        widget.SetMarginBottom(bottom);
        widget.SetMarginStart(left);
        return widget;
    }

    public static T Expand<T>(this T widget, bool h = true, bool v = true) where T : Gtk.Widget
    {
        widget.SetHexpand(h);
        widget.SetVexpand(v);
        return widget;
    }

    public static void Clear(this Gtk.Box box)
    {
        while (box.GetFirstChild() is { } child)
            box.Remove(child);
    }

    public static void Copy(Gtk.Widget widget, string text) => widget.GetClipboard().SetText(text);
}

internal static class Styles
{
    // libadwaita supplies the look; these only cover what has no built-in style class.
    private const string Css = """
        .user-bubble { background: alpha(@accent_bg_color, 0.16); border-radius: 18px; padding: 10px 16px; }
        .composer { border-radius: 22px; padding: 10px 10px 8px 14px; background: @card_bg_color; border: 1px solid alpha(@borders, 0.6); }
        .composer textview, .composer textview text { background: transparent; font-size: 15px; }
        .message { font-size: 15px; }
        .engine-card { border-radius: 12px; padding: 8px 10px; }
        .dot { min-width: 8px; min-height: 8px; border-radius: 4px; background: @dim_label_color; }
        .dot.ready { background: @success_color; }
        .dot.busy { background: @warning_color; }
        .dot.error { background: @error_color; }
        .tag { font-size: 11px; padding: 1px 7px; border-radius: 6px; background: alpha(@accent_bg_color, 0.15); color: @accent_color; }
        """;

    public static void Load()
    {
        var provider = Gtk.CssProvider.New();
        provider.LoadFromString(Css);
        Gtk.StyleContext.AddProviderForDisplay(Gdk.Display.GetDefault()!, provider, 600);
    }
}
