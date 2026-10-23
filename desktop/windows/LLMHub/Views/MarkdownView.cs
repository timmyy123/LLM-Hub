using LLMHub.Services;
using Microsoft.UI.Text;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Documents;
using Microsoft.UI.Xaml.Media;
using Windows.UI.Text;

namespace LLMHub.Views;

/// <summary>
/// Shows a reply as formatted text (headings, lists, code blocks, bold/italic, links, [n] citations) from the shared
/// Markdown parser. While a reply streams in, re-rendering is coalesced to a few times a second.
/// </summary>
public sealed partial class MarkdownView : UserControl
{
    public static readonly DependencyProperty TextProperty =
        DependencyProperty.Register(nameof(Text), typeof(string), typeof(MarkdownView), new PropertyMetadata("", OnTextChanged));

    private readonly StackPanel _panel = new() { Spacing = 12 };
    private readonly DispatcherTimer _timer = new() { Interval = TimeSpan.FromMilliseconds(120) };
    private string _rendered = "";

    public MarkdownView()
    {
        Content = _panel;
        ActualThemeChanged += (_, _) =>
        {
            _rendered = "";
            Render();
        };
        _timer.Tick += (_, _) =>
        {
            _timer.Stop();
            Render();
        };
    }

    public string Text
    {
        get => (string)GetValue(TextProperty);
        set => SetValue(TextProperty, value);
    }

    /// <summary>Called when a [n] citation is clicked.</summary>
    public event Action<int>? CitationClicked;

    private static void OnTextChanged(DependencyObject d, DependencyPropertyChangedEventArgs e)
    {
        var view = (MarkdownView)d;
        // Short or first text renders at once (also keeps UI tests deterministic); streaming updates are batched.
        if (view._rendered.Length == 0 || ((string)e.NewValue).Length < 400)
            view.Render();
        else if (!view._timer.IsEnabled)
            view._timer.Start();
    }

    private void Render()
    {
        var text = Text ?? "";
        if (text == _rendered)
            return;
        _rendered = text;
        _panel.Children.Clear();
        var number = 0;
        foreach (var block in Markdown.Parse(text))
        {
            switch (block.Kind)
            {
                case MdBlockKind.Heading:
                    _panel.Children.Add(Rich(block.Text, block.Level <= 2 ? 17 : 15, FontWeights.SemiBold, "TextPrimaryBrush"));
                    break;
                case MdBlockKind.Code:
                    _panel.Children.Add(new Border
                    {
                        Background = Brush("SurfaceAltBrush"),
                        BorderBrush = Brush("DividerBrush"),
                        BorderThickness = new Thickness(1),
                        CornerRadius = new CornerRadius(10),
                        Padding = new Thickness(14, 10, 14, 10),
                        Child = new ScrollViewer
                        {
                            HorizontalScrollBarVisibility = ScrollBarVisibility.Auto,
                            VerticalScrollBarVisibility = ScrollBarVisibility.Disabled,
                            Content = new TextBlock
                            {
                                Text = block.Text,
                                FontFamily = new FontFamily("Cascadia Mono, Consolas"),
                                FontSize = 13,
                                IsTextSelectionEnabled = true,
                                Foreground = Brush("TextSecondaryBrush"),
                            },
                        },
                    });
                    break;
                case MdBlockKind.Bullet:
                case MdBlockKind.Numbered:
                    number = block.Kind == MdBlockKind.Numbered ? block.Level : 0;
                    var row = new Grid { ColumnSpacing = 10, Margin = new Thickness(block.Kind == MdBlockKind.Bullet ? block.Level * 18 : 0, -6, 0, 0) };
                    row.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
                    row.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
                    row.Children.Add(new TextBlock { Text = number > 0 ? $"{number}." : "•", Foreground = Brush("TextMutedBrush"), FontSize = 15 });
                    var item = Rich(block.Text, 15, FontWeights.Normal, "BodyTextBrush");
                    Grid.SetColumn(item, 1);
                    row.Children.Add(item);
                    _panel.Children.Add(row);
                    break;
                case MdBlockKind.Quote:
                    _panel.Children.Add(new Border
                    {
                        BorderBrush = Brush("ControlBorderBrush"),
                        BorderThickness = new Thickness(3, 0, 0, 0),
                        Padding = new Thickness(12, 2, 0, 2),
                        Child = Rich(block.Text, 15, FontWeights.Normal, "TextMutedBrush"),
                    });
                    break;
                case MdBlockKind.Rule:
                    _panel.Children.Add(new Border { Height = 1, Background = Brush("DividerBrush") });
                    break;
                default:
                    _panel.Children.Add(Rich(block.Text, 15, FontWeights.Normal, "BodyTextBrush"));
                    break;
            }
        }
    }

    private RichTextBlock Rich(string text, double size, FontWeight weight, string brush)
    {
        var block = new RichTextBlock
        {
            FontSize = size,
            FontWeight = weight,
            LineHeight = size * 1.6,
            TextWrapping = TextWrapping.Wrap,
            IsTextSelectionEnabled = true,
            Foreground = Brush(brush),
        };
        var paragraph = new Paragraph();
        foreach (var span in Markdown.Inline(text))
        {
            switch (span.Kind)
            {
                case MdSpanKind.Bold:
                    paragraph.Inlines.Add(new Run { Text = span.Text, FontWeight = FontWeights.SemiBold });
                    break;
                case MdSpanKind.Italic:
                    paragraph.Inlines.Add(new Run { Text = span.Text, FontStyle = FontStyle.Italic });
                    break;
                case MdSpanKind.BoldItalic:
                    paragraph.Inlines.Add(new Run { Text = span.Text, FontWeight = FontWeights.SemiBold, FontStyle = FontStyle.Italic });
                    break;
                case MdSpanKind.Code:
                    paragraph.Inlines.Add(new Run { Text = span.Text, FontFamily = new FontFamily("Cascadia Mono, Consolas"), FontSize = size - 1, Foreground = Brush("CodeTextBrush") });
                    break;
                case MdSpanKind.Link when Uri.TryCreate(span.Url, UriKind.Absolute, out var uri):
                    var link = new Hyperlink { NavigateUri = uri, Foreground = Brush("LinkBrush") };
                    link.Inlines.Add(new Run { Text = span.Text });
                    paragraph.Inlines.Add(link);
                    break;
                case MdSpanKind.Citation:
                    var cite = new Hyperlink { Foreground = Brush("LinkBrush"), UnderlineStyle = UnderlineStyle.None };
                    cite.Inlines.Add(new Run { Text = $"[{span.Text}]", FontSize = size - 3 });
                    var n = int.Parse(span.Url);
                    cite.Click += (_, _) => CitationClicked?.Invoke(n);
                    paragraph.Inlines.Add(cite);
                    break;
                default:
                    paragraph.Inlines.Add(new Run { Text = span.Text });
                    break;
            }
        }
        block.Blocks.Add(paragraph);
        return block;
    }

    private Brush Brush(string key) => Ui.ThemeBrush(this, key) ?? new SolidColorBrush(Microsoft.UI.Colors.Gray);
}
