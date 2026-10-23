using System.Text;
using UglyToad.PdfPig;

namespace LLMHub.Services;

/// <summary>A file attached to a chat message. Images go to the model as pictures; everything else as extracted text.</summary>
public sealed record Attachment(string Name, string Path, string Kind, string MediaType, string Detail)
{
    public bool IsImage => Kind == "image";
    /// <summary>Short label for the file chip: "PDF", "PNG", "TXT".</summary>
    public string Badge => System.IO.Path.GetExtension(Name).TrimStart('.').ToUpperInvariant() is { Length: > 0 and <= 4 } ext ? ext : "FILE";
}

public static class Attachments
{
    private static readonly Dictionary<string, string> ImageTypes = new(StringComparer.OrdinalIgnoreCase)
    {
        [".png"] = "image/png", [".jpg"] = "image/jpeg", [".jpeg"] = "image/jpeg", [".webp"] = "image/webp",
        [".gif"] = "image/gif", [".bmp"] = "image/bmp",
    };

    /// <summary>Text-like files read as UTF-8. Anything not listed (and not an image or PDF) is refused.</summary>
    private static readonly HashSet<string> TextTypes = new(StringComparer.OrdinalIgnoreCase)
    {
        ".txt", ".md", ".markdown", ".csv", ".tsv", ".json", ".xml", ".yaml", ".yml", ".html", ".htm", ".css", ".js", ".ts",
        ".tsx", ".jsx", ".py", ".cs", ".swift", ".kt", ".java", ".c", ".h", ".cpp", ".hpp", ".go", ".rs", ".rb", ".php",
        ".sh", ".ps1", ".sql", ".toml", ".ini", ".log", ".tex", ".rtf",
    };

    /// <summary>Most text a single document may add to the prompt, so one big file cannot crowd out the conversation.</summary>
    public const int MaxDocumentChars = 24_000;

    public static bool IsSupported(string path)
    {
        var ext = System.IO.Path.GetExtension(path);
        return ImageTypes.ContainsKey(ext) || TextTypes.Contains(ext) || ext.Equals(".pdf", StringComparison.OrdinalIgnoreCase);
    }

    /// <summary>File-picker filter list, e.g. for WinUI's FileOpenPicker.</summary>
    public static IEnumerable<string> Extensions => ImageTypes.Keys.Concat(TextTypes).Append(".pdf");

    /// <summary>Copies the file into the app's data folder (so chats keep working if the original moves) and describes it.</summary>
    public static Attachment Import(string sourcePath)
    {
        if (!IsSupported(sourcePath))
            throw new NotSupportedException(Loc.F("AttachUnsupported", System.IO.Path.GetFileName(sourcePath)));
        var name = System.IO.Path.GetFileName(sourcePath);
        var folder = Directory.CreateDirectory(System.IO.Path.Combine(AppPaths.Attachments, Guid.NewGuid().ToString("N")[..12])).FullName;
        var path = System.IO.Path.Combine(folder, name);
        File.Copy(sourcePath, path);
        var ext = System.IO.Path.GetExtension(name);
        if (ImageTypes.TryGetValue(ext, out var mime))
            return new Attachment(name, path, "image", mime, Loc.S("AttachImage"));
        if (ext.Equals(".pdf", StringComparison.OrdinalIgnoreCase))
        {
            using var pdf = PdfDocument.Open(path);
            return new Attachment(name, path, "document", "application/pdf", Loc.F("AttachPages", pdf.NumberOfPages.ToString()));
        }
        return new Attachment(name, path, "document", "text/plain", Fmt.Bytes(new FileInfo(path).Length));
    }

    /// <summary>The document's text for the prompt, cut to <see cref="MaxDocumentChars"/>.</summary>
    public static string ReadText(Attachment a)
    {
        string text;
        if (a.MediaType == "application/pdf")
        {
            var sb = new StringBuilder();
            using var pdf = PdfDocument.Open(a.Path);
            foreach (var page in pdf.GetPages())
            {
                sb.Append("[Page ").Append(page.Number).AppendLine("]");
                sb.AppendLine(string.Join(' ', page.GetWords().Select(w => w.Text)));
                if (sb.Length > MaxDocumentChars)
                    break;
            }
            text = sb.ToString();
        }
        else
            text = File.ReadAllText(a.Path);
        return text.Length <= MaxDocumentChars ? text : text[..MaxDocumentChars] + "\n[…truncated]";
    }

    /// <summary>The user's message with attached documents folded in as quoted text.</summary>
    public static string WithDocuments(string text, IEnumerable<Attachment>? attachments)
    {
        var docs = attachments?.Where(a => !a.IsImage).ToList();
        if (docs is null || docs.Count == 0)
            return text;
        var sb = new StringBuilder();
        foreach (var doc in docs)
        {
            sb.Append("<file name=\"").Append(doc.Name).AppendLine("\">");
            try
            {
                sb.AppendLine(ReadText(doc));
            }
            catch (Exception e)
            {
                Log.Error($"Could not read {doc.Path}", e);
                sb.AppendLine("[could not read this file]");
            }
            sb.AppendLine("</file>");
        }
        return sb.Append(text).ToString();
    }

    public static string DataUrl(Attachment a) => $"data:{a.MediaType};base64,{Convert.ToBase64String(File.ReadAllBytes(a.Path))}";
}
