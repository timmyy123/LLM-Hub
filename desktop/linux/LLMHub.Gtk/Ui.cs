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

    /// <summary>Runs <paramref name="onChange"/> on the GTK main loop whenever <paramref name="source"/> raises PropertyChanged.</summary>
    public static void Watch(System.ComponentModel.INotifyPropertyChanged source, Action<string?> onChange) =>
        source.PropertyChanged += (_, e) => Services.UiThread.Run(() => onChange(e.PropertyName));

    public static Gtk.ScrolledWindow Scroll(Gtk.Widget child, bool horizontal = false)
    {
        var scroll = Gtk.ScrolledWindow.New();
        scroll.SetChild(child);
        scroll.SetPolicy(horizontal ? Gtk.PolicyType.Automatic : Gtk.PolicyType.Never, Gtk.PolicyType.Automatic);
        scroll.SetVexpand(true);
        return scroll;
    }

    /// <summary>A labelled field: small caps-style label above the control, as in the design's forms.</summary>
    public static Gtk.Box Field(string label, Gtk.Widget control) => VBox(6, Label(label, "field-label"), control);

    /// <summary>A row of toggle tabs ("Image  Video  Music  Voice") where exactly one is on.</summary>
    public static (Gtk.Box Box, Action<string> Select) Tabs(IEnumerable<(string Tag, string Label)> tabs, Action<string> onPick, string css = "studio-tab")
    {
        var box = HBox(4);
        var buttons = new List<(string Tag, Gtk.ToggleButton Button)>();
        var syncing = false;
        void Select(string tag)
        {
            syncing = true;
            foreach (var (t, b) in buttons)
                b.SetActive(t == tag);
            syncing = false;
        }
        foreach (var (tag, label) in tabs)
        {
            var button = Gtk.ToggleButton.NewWithLabel(label);
            button.AddCssClass(css);
            button.Named("Tab" + tag);
            button.OnToggled += (_, _) =>
            {
                if (syncing)
                    return;
                Select(tag);
                onPick(tag);
            };
            buttons.Add((tag, button));
            box.Append(button);
        }
        return (box, Select);
    }

    /// <summary>Opens a file or URL with the desktop's default handler.</summary>
    public static void Open(string pathOrUrl)
    {
        try
        {
            var uri = pathOrUrl.Contains("://") ? pathOrUrl : new Uri(pathOrUrl).AbsoluteUri;
            Gio.AppInfoHelper.LaunchDefaultForUri(uri, null);
        }
        catch (Exception e)
        {
            Services.Log.Error($"Could not open {pathOrUrl}", e);
        }
    }
}

internal static class Styles
{
    /// <summary>
    /// The Claude Design palette, applied by redefining libadwaita's named colours (dark appearance, the designed look).
    /// </summary>
    private const string DesignPalette = """
        @define-color window_bg_color #1b1b1a;
        @define-color window_fg_color #ecebe7;
        @define-color view_bg_color #1b1b1a;
        @define-color view_fg_color #ecebe7;
        @define-color headerbar_bg_color #141413;
        @define-color headerbar_fg_color #ecebe7;
        @define-color headerbar_border_color #2c2c29;
        @define-color sidebar_bg_color #161615;
        @define-color sidebar_fg_color #ecebe7;
        @define-color sidebar_border_color #2c2c29;
        @define-color card_bg_color #1f1f1d;
        @define-color card_fg_color #ecebe7;
        @define-color popover_bg_color #252523;
        @define-color dialog_bg_color #1f1f1d;
        @define-color accent_bg_color #8ab4ff;
        @define-color accent_fg_color #10131a;
        @define-color accent_color #8ab4ff;
        @define-color borders #33332f;
        @define-color success_color #5fd38d;
        @define-color warning_color #e8b04a;
        @define-color error_color #ff9a9a;
        """;

    // libadwaita supplies the look; these only cover what has no built-in style class.
    private const string Css = """
        .user-bubble { background: #2b2b29; border-radius: 16px; padding: 12px 16px; }
        .composer { border-radius: 18px; padding: 14px 14px 10px 14px; background: #252523; border: 1px solid #3a3a36; }
        .composer textview, .composer textview text { background: transparent; font-size: 15px; }
        .composer-toggle { border-radius: 10px; min-height: 34px; padding: 0 12px; border: 1px solid #3a3a36; background: transparent; color: #c9c8c2; font-size: 13px; }
        .composer-toggle:checked { background: #232a38; border-color: #3d4a63; color: #8ab4ff; }
        .composer-icon { border-radius: 10px; min-width: 34px; min-height: 34px; border: 1px solid #3a3a36; background: transparent; color: #c9c8c2; }
        .composer-send { border-radius: 10px; min-width: 36px; min-height: 36px; }
        .activity-pill { border-radius: 15px; padding: 4px 12px; border: 1px solid #33332f; background: transparent; color: #c9c8c2; font-size: 13px; }
        .reasoning { border-left: 2px solid #3a3a36; padding-left: 14px; font-size: 13px; }
        .file-chip { border-radius: 10px; padding: 8px 12px 8px 8px; border: 1px solid #33332f; background: #222220; }
        .badge { min-width: 32px; min-height: 36px; border-radius: 6px; font-size: 10px; font-weight: 600; }
        .badge-doc { background: #3a2f2a; color: #f0a98a; }
        .badge-image { background: #263040; color: #9cc2ff; }
        .source-chip { font-size: 12px; padding: 1px 6px; border-radius: 6px; background: #252b38; min-height: 0; }
        .code-block { background: #1f1f1d; border: 1px solid #2c2c29; border-radius: 10px; padding: 10px 14px; font-size: 13px; }
        .quote { border-left: 3px solid #3a3a36; padding-left: 12px; }
        .message { font-size: 15px; }
        .new-button { border-radius: 10px; min-height: 40px; padding: 0 12px; background: #252523; border: 1px solid #33332f; }
        .engine-card { border-radius: 12px; padding: 8px 10px; }
        .dot { min-width: 8px; min-height: 8px; border-radius: 4px; background: @dim_label_color; }
        .dot.ready { background: @success_color; }
        .dot.busy { background: @warning_color; }
        .dot.error { background: @error_color; }
        .field-label { font-size: 12px; font-weight: 500; color: #8f8e88; }
        .studio-tab { border-radius: 9px; min-height: 32px; padding: 0 14px; background: transparent; color: #a5a49e; border: none; font-size: 13px; }
        .studio-tab:checked { background: #2a2a27; color: #ecebe7; }
        .step-card { border-radius: 10px; border: 1px solid #33332f; background: #1f1f1d; }
        .step-head { padding: 9px 12px; }
        .diff-line { font-family: monospace; font-size: 12px; padding: 1px 10px; }
        .diff-add { background: #1e3326; color: #c5f0d3; }
        .diff-del { background: #3a2222; color: #ffb4b4; }
        .diff-hunk { color: #8f8e88; }
        .tasks-card { border-radius: 12px; border: 1px solid #33332f; background: #1f1f1d; padding: 12px 14px; }
        .permission-card { border-radius: 12px; border: 1px solid #e8b04a; background: #1f1f1d; padding: 14px; }
        .mono { font-family: monospace; font-size: 12px; }
        .chip { border-radius: 8px; background: #1f1f1d; padding: 3px 10px; color: #c9c8c2; }
        .side-panel { background: #161615; border-left: 1px solid #2c2c29; }
        .divider-bottom { border-bottom: 1px solid #2c2c29; }
        .divider-right { border-right: 1px solid #2c2c29; }
        .divider-top { border-top: 1px solid #2c2c29; }
        .preview-area { background: #141413; }
        .preview-frame { border-radius: 10px; border: 1px solid #2c2c29; background: #ffffff; }
        .preview-empty { color: #8f8e88; }
        .swatch { min-height: 32px; border-radius: 8px; border: 1px solid #3a3a36; }
        .gallery-tile { border-radius: 12px; background: #1f1f1d; }
        .commit-button { background: #ecebe7; color: #1b1b1a; }
        .tag { font-size: 11px; padding: 1px 7px; border-radius: 6px; background: alpha(@accent_bg_color, 0.15); color: @accent_color; }
        """;

    public static void Load()
    {
        var appearance = Services.AppSettings.Current.Appearance;
        Adw.StyleManager.GetDefault().SetColorScheme(appearance switch
        {
            "light" => Adw.ColorScheme.ForceLight,
            "system" => Adw.ColorScheme.Default,
            _ => Adw.ColorScheme.ForceDark,
        });
        var provider = Gtk.CssProvider.New();
        provider.LoadFromString((appearance == "dark" ? DesignPalette : "") + Css);
        Gtk.StyleContext.AddProviderForDisplay(Gdk.Display.GetDefault()!, provider, 600);
    }
}

/// <summary>A drop-down that reports the user's picks by index, but not selections made from code.</summary>
internal sealed class Picker
{
    private bool _syncing;

    public Picker(IEnumerable<string> items, Action<int> onPick)
    {
        DropDown = Gtk.DropDown.NewFromStrings(items.ToArray());
        DropDown.OnNotify += (_, e) =>
        {
            if (!_syncing && e.Pspec.GetName() == "selected" && DropDown.GetSelected() is var i and < uint.MaxValue)
                onPick((int)i);
        };
    }

    public Gtk.DropDown DropDown { get; }

    public void SetItems(IEnumerable<string> items, int selected)
    {
        _syncing = true;
        DropDown.SetModel(Gtk.StringList.New(items.ToArray()));
        _syncing = false;
        Select(selected);
    }

    /// <summary>Selects <paramref name="index"/> (-1 for none) after the current notification has returned, as GTK needs.</summary>
    public void Select(int index)
    {
        var target = index >= 0 ? (uint)index : uint.MaxValue;
        GLib.Functions.IdleAdd(GLib.Constants.PRIORITY_DEFAULT, () =>
        {
            if (DropDown.GetSelected() != target)
            {
                _syncing = true;
                DropDown.SetSelected(target);
                _syncing = false;
            }
            return false;
        });
    }
}
