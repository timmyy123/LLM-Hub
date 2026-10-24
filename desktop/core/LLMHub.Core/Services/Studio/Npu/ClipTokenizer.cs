using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace LLMHub.Services.Studio.Npu;

/// <summary>
/// CLIP's byte-level BPE tokenizer (as in the Android app's ClipBpeTokenizer): BOS, BPE tokens, EOS, truncated and
/// padded with EOS to 77 tokens. Reads the vocabulary and merges from a Hugging Face tokenizer.json.
/// </summary>
public sealed partial class ClipTokenizer
{
    public const int Length = 77;
    private readonly Dictionary<string, int> _vocab;
    private readonly Dictionary<(string, string), int> _ranks = [];
    private readonly Dictionary<string, string[]> _cache = [];
    private readonly string[] _byteChars = BytesToUnicode();
    private readonly int _bos;
    private readonly int _eos;

    public ClipTokenizer(string tokenizerJson)
    {
        using var doc = JsonDocument.Parse(File.ReadAllBytes(tokenizerJson));
        var model = doc.RootElement.GetProperty("model");
        _vocab = model.GetProperty("vocab").EnumerateObject().ToDictionary(p => p.Name, p => p.Value.GetInt32());
        var rank = 0;
        foreach (var merge in model.GetProperty("merges").EnumerateArray())
        {
            var (a, b) = merge.ValueKind == JsonValueKind.Array
                ? (merge[0].GetString()!, merge[1].GetString()!)
                : merge.GetString()!.Split(' ') is [var x, var y] ? (x, y) : throw new FormatException("merge");
            _ranks.TryAdd((a, b), rank++);
        }
        _bos = _vocab["<|startoftext|>"];
        _eos = _vocab["<|endoftext|>"];
    }

    public int[] Encode(string text)
    {
        var normalized = Whitespace().Replace(text.Normalize(NormalizationForm.FormC), " ").Trim().ToLowerInvariant();
        var tokens = new List<int> { _bos };
        foreach (Match chunk in Chunk().Matches(normalized))
        {
            var mapped = string.Concat(Encoding.UTF8.GetBytes(chunk.Value).Select(b => _byteChars[b]));
            foreach (var word in Bpe(mapped))
            {
                if (_vocab.TryGetValue(word, out var id))
                    tokens.Add(id);
            }
        }
        tokens.Add(_eos);
        var ids = new int[Length];
        for (var i = 0; i < Length; i++)
            ids[i] = i < tokens.Count ? tokens[i] : _eos;
        ids[Length - 1] = tokens.Count >= Length ? _eos : ids[Length - 1];
        return ids;
    }

    private string[] Bpe(string chunk)
    {
        if (_cache.TryGetValue(chunk, out var cached))
            return cached;
        // Every mapped character is a single UTF-16 unit (the byte table stays below U+0144).
        var words = chunk.Select(c => c.ToString()).ToList();
        words[^1] += "</w>";
        while (words.Count > 1)
        {
            var best = -1;
            var bestRank = int.MaxValue;
            for (var i = 0; i < words.Count - 1; i++)
            {
                if (_ranks.TryGetValue((words[i], words[i + 1]), out var r) && r < bestRank)
                {
                    bestRank = r;
                    best = i;
                }
            }
            if (best < 0)
                break;
            var (first, second) = (words[best], words[best + 1]);
            var merged = new List<string>(words.Count);
            for (var i = 0; i < words.Count; i++)
            {
                if (i < words.Count - 1 && words[i] == first && words[i + 1] == second)
                {
                    merged.Add(first + second);
                    i++;
                }
                else
                {
                    merged.Add(words[i]);
                }
            }
            words = merged;
        }
        return _cache[chunk] = [.. words];
    }

    /// <summary>GPT-2's reversible byte to printable character map.</summary>
    private static string[] BytesToUnicode()
    {
        var table = new string[256];
        var extra = 0;
        for (var b = 0; b < 256; b++)
        {
            var printable = b is >= '!' and <= '~' or >= 0xA1 and <= 0xAC or >= 0xAE and <= 0xFF;
            table[b] = char.ConvertFromUtf32(printable ? b : 256 + extra++);
        }
        return table;
    }

    [GeneratedRegex(@"\s+")]
    private static partial Regex Whitespace();

    [GeneratedRegex(@"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+", RegexOptions.IgnoreCase)]
    private static partial Regex Chunk();
}
