namespace LLMHub.Services;

/// <summary>
/// Some models (LFM2.5 Thinking, many fine-tunes) write their reasoning inline as &lt;think&gt;…&lt;/think&gt;
/// instead of a separate reasoning field. Feed streamed text in; reasoning and answer come out separately,
/// even when a tag is split across chunks.
/// </summary>
public sealed class ThinkSplitter
{
    private const string Open = "<think>";
    private const string Close = "</think>";
    private string _pending = "";
    private bool _inThink;

    public (string Answer, string Reasoning) Push(string chunk)
    {
        var text = _pending + chunk;
        _pending = "";
        var answer = new System.Text.StringBuilder();
        var reasoning = new System.Text.StringBuilder();
        while (text.Length > 0)
        {
            var tag = _inThink ? Close : Open;
            var at = text.IndexOf(tag, StringComparison.Ordinal);
            if (at >= 0)
            {
                (_inThink ? reasoning : answer).Append(text[..at]);
                text = text[(at + tag.Length)..];
                _inThink = !_inThink;
                continue;
            }
            // Hold back a possible partial tag at the end ("<thi") until the next chunk.
            var keep = PartialTagLength(text, tag);
            (_inThink ? reasoning : answer).Append(text[..^keep]);
            _pending = text[^keep..];
            break;
        }
        return (answer.ToString(), reasoning.ToString());
    }

    public (string Answer, string Reasoning) Flush()
    {
        var rest = _pending;
        _pending = "";
        return _inThink ? ("", rest) : (rest, "");
    }

    private static int PartialTagLength(string text, string tag)
    {
        for (var n = Math.Min(tag.Length - 1, text.Length); n > 0; n--)
            if (text.EndsWith(tag[..n], StringComparison.Ordinal))
                return n;
        return 0;
    }
}
