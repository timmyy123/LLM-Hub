using System.Text.RegularExpressions;

namespace LLMHub.Services;

public enum MdBlockKind { Paragraph, Heading, Bullet, Numbered, Code, Quote, Rule }

/// <summary>One block of a reply. <see cref="Level"/> is the heading level, list nesting, or the number of a numbered item.</summary>
public sealed record MdBlock(MdBlockKind Kind, string Text, int Level = 0, string Language = "");

public enum MdSpanKind { Text, Bold, Italic, BoldItalic, Code, Link, Citation }

/// <summary>Inline run. For links <see cref="Url"/> is the target; for citations ("[2]") it is the source number.</summary>
public sealed record MdSpan(MdSpanKind Kind, string Text, string Url = "");

/// <summary>
/// The Markdown subset local models actually write (headings, lists, quotes, fenced code, bold/italic, inline code,
/// links and [n] citations), parsed once so the Windows and Linux renderers stay in step. Tolerant of half-streamed text:
/// an unclosed code fence is treated as code to the end.
/// </summary>
public static partial class Markdown
{
    public static List<MdBlock> Parse(string text)
    {
        var blocks = new List<MdBlock>();
        var lines = text.Replace("\r\n", "\n").Split('\n');
        var paragraph = new List<string>();

        void Flush()
        {
            if (paragraph.Count > 0)
                blocks.Add(new MdBlock(MdBlockKind.Paragraph, string.Join(" ", paragraph).Trim()));
            paragraph.Clear();
        }

        for (var i = 0; i < lines.Length; i++)
        {
            var line = lines[i];
            var trimmed = line.TrimStart();
            if (trimmed.StartsWith("```", StringComparison.Ordinal))
            {
                Flush();
                var language = trimmed[3..].Trim();
                var code = new List<string>();
                for (i++; i < lines.Length && !lines[i].TrimStart().StartsWith("```", StringComparison.Ordinal); i++)
                    code.Add(lines[i]);
                blocks.Add(new MdBlock(MdBlockKind.Code, string.Join("\n", code), 0, language));
                continue;
            }
            if (trimmed.Length == 0)
            {
                Flush();
                continue;
            }
            if (HeadingRegex().Match(trimmed) is { Success: true } h)
            {
                Flush();
                blocks.Add(new MdBlock(MdBlockKind.Heading, h.Groups[2].Value.Trim().TrimEnd('#').Trim(), h.Groups[1].Length));
                continue;
            }
            if (RuleRegex().IsMatch(trimmed))
            {
                Flush();
                blocks.Add(new MdBlock(MdBlockKind.Rule, ""));
                continue;
            }
            var indent = (line.Length - trimmed.Length) / 2;
            if (BulletRegex().Match(trimmed) is { Success: true } b)
            {
                Flush();
                blocks.Add(new MdBlock(MdBlockKind.Bullet, b.Groups[1].Value, indent));
                continue;
            }
            if (NumberedRegex().Match(trimmed) is { Success: true } n)
            {
                Flush();
                blocks.Add(new MdBlock(MdBlockKind.Numbered, n.Groups[2].Value, int.Parse(n.Groups[1].Value)));
                continue;
            }
            if (trimmed.StartsWith('>'))
            {
                Flush();
                blocks.Add(new MdBlock(MdBlockKind.Quote, trimmed.TrimStart('>').Trim()));
                continue;
            }
            paragraph.Add(trimmed);
        }
        Flush();
        return blocks;
    }

    public static List<MdSpan> Inline(string text)
    {
        var spans = new List<MdSpan>();
        var last = 0;
        foreach (Match m in InlineRegex().Matches(text))
        {
            if (m.Index > last)
                spans.Add(new MdSpan(MdSpanKind.Text, text[last..m.Index]));
            if (m.Groups["code"].Success)
                spans.Add(new MdSpan(MdSpanKind.Code, m.Groups["code"].Value));
            else if (m.Groups["bi"].Success)
                spans.Add(new MdSpan(MdSpanKind.BoldItalic, m.Groups["bi"].Value));
            else if (m.Groups["b"].Success)
                spans.Add(new MdSpan(MdSpanKind.Bold, m.Groups["b"].Value));
            else if (m.Groups["i"].Success)
                spans.Add(new MdSpan(MdSpanKind.Italic, m.Groups["i"].Value));
            else if (m.Groups["label"].Success)
                spans.Add(new MdSpan(MdSpanKind.Link, m.Groups["label"].Value, m.Groups["url"].Value));
            else if (m.Groups["cite"].Success)
                spans.Add(new MdSpan(MdSpanKind.Citation, m.Groups["cite"].Value, m.Groups["cite"].Value));
            last = m.Index + m.Length;
        }
        if (last < text.Length)
            spans.Add(new MdSpan(MdSpanKind.Text, text[last..]));
        return spans;
    }

    /// <summary>Plain text for reading aloud and copying without Markdown symbols.</summary>
    public static string ToPlainText(string text) =>
        string.Join("\n", Parse(text).Select(b => b.Kind == MdBlockKind.Code ? b.Text : string.Concat(Inline(b.Text).Select(s => s.Kind == MdSpanKind.Citation ? "" : s.Text))));

    [GeneratedRegex(@"^(#{1,6})\s+(.*)$")]
    private static partial Regex HeadingRegex();

    [GeneratedRegex(@"^([-*_])(\s*\1){2,}\s*$")]
    private static partial Regex RuleRegex();

    [GeneratedRegex(@"^[-*+]\s+(.*)$")]
    private static partial Regex BulletRegex();

    [GeneratedRegex(@"^(\d{1,3})[.)]\s+(.*)$")]
    private static partial Regex NumberedRegex();

    [GeneratedRegex(@"`(?<code>[^`]+)`|\*\*\*(?<bi>[^*]+)\*\*\*|\*\*(?<b>[^*]+)\*\*|__(?<b>[^_]+)__|(?<![\w*])\*(?<i>[^*\s][^*]*)\*(?!\w)|\[(?<label>[^\]]+)\]\((?<url>https?://[^)\s]+)\)|\[(?<cite>\d{1,2})\]")]
    private static partial Regex InlineRegex();
}
