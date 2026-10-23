using System.Text;
using LLMHub.Services;

namespace LLMHub.Linux;

/// <summary>
/// A reply as formatted text (headings, lists, code blocks, bold/italic, links, [n] citations) from the shared Markdown
/// parser, built from GTK labels with Pango markup. Streaming updates are coalesced to a few renders a second.
/// </summary>
internal sealed class MarkdownView
{
    private readonly Gtk.Box _box = Gtk.Box.New(Gtk.Orientation.Vertical, 12);
    private string _text = "";
    private string _rendered = "";
    private bool _scheduled;

    public MarkdownView(string name)
    {
        _box.SetName(name);
    }

    public Gtk.Widget Widget => _box;
    public string Text => _text;

    public void SetText(string text)
    {
        _text = text;
        if (_rendered.Length == 0 || text.Length < 400)
        {
            Render();
            return;
        }
        if (_scheduled)
            return;
        _scheduled = true;
        GLib.Functions.TimeoutAdd(GLib.Constants.PRIORITY_DEFAULT, 120, () =>
        {
            _scheduled = false;
            Render();
            return false;
        });
    }

    private void Render()
    {
        if (_text == _rendered)
            return;
        _rendered = _text;
        _box.Clear();
        foreach (var block in Markdown.Parse(_text))
        {
            switch (block.Kind)
            {
                case MdBlockKind.Heading:
                    _box.Append(Label($"<span size=\"{(block.Level <= 2 ? "large" : "medium")}\" weight=\"bold\">{Inline(block.Text)}</span>"));
                    break;
                case MdBlockKind.Code:
                    var code = Gtk.Label.New(block.Text);
                    code.SetXalign(0);
                    code.SetSelectable(true);
                    code.SetWrap(true);
                    code.AddCssClass("monospace");
                    code.AddCssClass("code-block");
                    _box.Append(code);
                    break;
                case MdBlockKind.Bullet:
                    _box.Append(ListRow("•", block.Text, block.Level * 18));
                    break;
                case MdBlockKind.Numbered:
                    _box.Append(ListRow($"{block.Level}.", block.Text, 0));
                    break;
                case MdBlockKind.Quote:
                    var quote = Label($"<i>{Inline(block.Text)}</i>");
                    quote.AddCssClass("quote");
                    _box.Append(quote);
                    break;
                case MdBlockKind.Rule:
                    _box.Append(Gtk.Separator.New(Gtk.Orientation.Horizontal));
                    break;
                default:
                    _box.Append(Label(Inline(block.Text)));
                    break;
            }
        }
    }

    private static Gtk.Widget ListRow(string marker, string text, int indent)
    {
        var row = Gtk.Box.New(Gtk.Orientation.Horizontal, 10);
        row.SetMarginStart(indent);
        var bullet = Gtk.Label.New(marker);
        bullet.AddCssClass("dim-label");
        bullet.SetValign(Gtk.Align.Start);
        row.Append(bullet);
        var label = Label(Inline(text));
        label.SetHexpand(true);
        row.Append(label);
        return row;
    }

    private static Gtk.Label Label(string markup)
    {
        var label = Gtk.Label.New(null);
        label.SetMarkup(markup);
        label.SetXalign(0);
        label.SetWrap(true);
        label.SetWrapMode(Pango.WrapMode.WordChar);
        label.SetSelectable(true);
        label.AddCssClass("message");
        return label;
    }

    /// <summary>Pango markup for one line: escaped text with bold, italic, code, links and citations.</summary>
    private static string Inline(string text)
    {
        var sb = new StringBuilder();
        foreach (var span in Markdown.Inline(text))
        {
            var t = GLib.Functions.MarkupEscapeText(span.Text, -1);
            sb.Append(span.Kind switch
            {
                MdSpanKind.Bold => $"<b>{t}</b>",
                MdSpanKind.Italic => $"<i>{t}</i>",
                MdSpanKind.BoldItalic => $"<b><i>{t}</i></b>",
                MdSpanKind.Code => $"<tt>{t}</tt>",
                MdSpanKind.Link => $"<a href=\"{GLib.Functions.MarkupEscapeText(span.Url, -1)}\">{t}</a>",
                MdSpanKind.Citation => $"<sup><span foreground=\"#8ab4ff\">[{t}]</span></sup>",
                _ => t,
            });
        }
        return sb.ToString();
    }
}
