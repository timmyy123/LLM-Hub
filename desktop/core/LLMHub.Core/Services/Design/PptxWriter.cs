using System.IO.Compression;
using System.Security;
using System.Text;
using System.Text.RegularExpressions;

namespace LLMHub.Services.Design;

/// <summary>One slide for <see cref="PptxWriter"/>: a title, body lines and an optional picture (PNG or JPEG path).</summary>
public sealed record PptxSlide(string Title, IReadOnlyList<string> Lines, string? Picture);

/// <summary>
/// A small PowerPoint (.pptx) writer with no dependencies: 16:9 slides with editable text in the design system's colors and
/// fonts, and the generated photos. Each slide has a title, body text and an optional picture.
/// </summary>
public static partial class PptxWriter
{
    private const long SlideWidth = 12192000, SlideHeight = 6858000;
    private const string P = "http://schemas.openxmlformats.org/presentationml/2006/main";
    private const string A = "http://schemas.openxmlformats.org/drawingml/2006/main";
    private const string R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships";
    private const string Rel = "http://schemas.openxmlformats.org/package/2006/relationships";
    private const string RelType = "http://schemas.openxmlformats.org/officeDocument/2006/relationships";

    public static void Write(string path, string title, IReadOnlyList<PptxSlide> slides, DesignSystem system)
    {
        if (slides.Count == 0)
            throw new InvalidOperationException(Loc.S("DesignNothingYet"));
        if (File.Exists(path))
            File.Delete(path);
        var background = Hex(system.Colors[0]);
        var ink = Hex(system.Colors[^1]);
        var accent = Hex(system.Colors.Length > 3 ? system.Colors[3] : system.Colors[^1]);
        using var zip = ZipFile.Open(path, ZipArchiveMode.Create);
        void Add(string name, string xml)
        {
            using var writer = new StreamWriter(zip.CreateEntry(name, CompressionLevel.Optimal).Open(), new UTF8Encoding(false));
            writer.Write("<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>\n");
            writer.Write(xml);
        }

        var types = new StringBuilder();
        types.Append($"""<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Default Extension="png" ContentType="image/png"/><Default Extension="jpeg" ContentType="image/jpeg"/>""");
        types.Append("""<Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>""");
        types.Append("""<Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>""");
        types.Append("""<Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>""");
        types.Append("""<Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>""");
        types.Append("""<Override PartName="/ppt/presProps.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presProps+xml"/>""");
        types.Append("""<Override PartName="/ppt/viewProps.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.viewProps+xml"/>""");
        types.Append("""<Override PartName="/ppt/tableStyles.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.tableStyles+xml"/>""");
        types.Append("""<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>""");
        types.Append("""<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>""");
        for (var i = 1; i <= slides.Count; i++)
            types.Append($"""<Override PartName="/ppt/slides/slide{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>""");
        types.Append("</Types>");
        Add("[Content_Types].xml", types.ToString());

        Add("_rels/.rels", $"""<Relationships xmlns="{Rel}"><Relationship Id="rId1" Type="{RelType}/officeDocument" Target="ppt/presentation.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/><Relationship Id="rId3" Type="{RelType}/extended-properties" Target="docProps/app.xml"/></Relationships>""");
        Add("docProps/core.xml", $"""<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"><dc:title>{X(title)}</dc:title><dc:creator>LLM Hub</dc:creator><dcterms:created xsi:type="dcterms:W3CDTF">{DateTime.UtcNow:yyyy-MM-ddTHH:mm:ssZ}</dcterms:created></cp:coreProperties>""");
        Add("docProps/app.xml", $"""<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"><Application>LLM Hub</Application><Slides>{slides.Count}</Slides></Properties>""");

        var ids = new StringBuilder();
        var presRels = new StringBuilder($"""<Relationships xmlns="{Rel}"><Relationship Id="rId1" Type="{RelType}/slideMaster" Target="slideMasters/slideMaster1.xml"/><Relationship Id="rId2" Type="{RelType}/theme" Target="theme/theme1.xml"/><Relationship Id="rId3" Type="{RelType}/presProps" Target="presProps.xml"/><Relationship Id="rId4" Type="{RelType}/viewProps" Target="viewProps.xml"/><Relationship Id="rId5" Type="{RelType}/tableStyles" Target="tableStyles.xml"/>""");
        for (var i = 1; i <= slides.Count; i++)
        {
            ids.Append($"""<p:sldId id="{255 + i}" r:id="rId{10 + i}"/>""");
            presRels.Append($"""<Relationship Id="rId{10 + i}" Type="{RelType}/slide" Target="slides/slide{i}.xml"/>""");
        }
        presRels.Append("</Relationships>");
        Add("ppt/presentation.xml", $"""<p:presentation xmlns:a="{A}" xmlns:r="{R}" xmlns:p="{P}"><p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst><p:sldIdLst>{ids}</p:sldIdLst><p:sldSz cx="{SlideWidth}" cy="{SlideHeight}"/><p:notesSz cx="6858000" cy="9144000"/></p:presentation>""");
        Add("ppt/_rels/presentation.xml.rels", presRels.ToString());
        Add("ppt/presProps.xml", $"""<p:presentationPr xmlns:a="{A}" xmlns:r="{R}" xmlns:p="{P}"/>""");
        Add("ppt/viewProps.xml", $"""<p:viewPr xmlns:a="{A}" xmlns:r="{R}" xmlns:p="{P}"/>""");
        Add("ppt/tableStyles.xml", "<a:tblStyleLst xmlns:a=\"" + A + "\" def=\"{5C22544A-7EE6-4342-B048-85BDC9FD1C3A}\"/>");

        const string emptyTree = """<p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr/></p:spTree>""";
        Add("ppt/slideMasters/slideMaster1.xml", $"""<p:sldMaster xmlns:a="{A}" xmlns:r="{R}" xmlns:p="{P}"><p:cSld><p:bg><p:bgPr><a:solidFill><a:srgbClr val="{background}"/></a:solidFill><a:effectLst/></p:bgPr></p:bg>{emptyTree}</p:cSld><p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/><p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst></p:sldMaster>""");
        Add("ppt/slideMasters/_rels/slideMaster1.xml.rels", $"""<Relationships xmlns="{Rel}"><Relationship Id="rId1" Type="{RelType}/slideLayout" Target="../slideLayouts/slideLayout1.xml"/><Relationship Id="rId2" Type="{RelType}/theme" Target="../theme/theme1.xml"/></Relationships>""");
        Add("ppt/slideLayouts/slideLayout1.xml", $"""<p:sldLayout xmlns:a="{A}" xmlns:r="{R}" xmlns:p="{P}" type="blank" preserve="1"><p:cSld name="Blank">{emptyTree}</p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sldLayout>""");
        Add("ppt/slideLayouts/_rels/slideLayout1.xml.rels", $"""<Relationships xmlns="{Rel}"><Relationship Id="rId1" Type="{RelType}/slideMaster" Target="../slideMasters/slideMaster1.xml"/></Relationships>""");
        Add("ppt/theme/theme1.xml", Theme(system, background, ink, accent));

        for (var i = 0; i < slides.Count; i++)
        {
            var slide = slides[i];
            var shapes = new StringBuilder();
            var hasPicture = slide.Picture is { } pic && File.Exists(pic) && ImageSize(pic) is not null;
            long textWidth = hasPicture ? 6100000 : 10668000;
            shapes.Append(TextBox(2, "Title", 762000, 640080, textWidth, 1300000, [slide.Title], 4000, bold: true, ink, system.HeadingFont));
            if (slide.Lines.Count > 0)
                shapes.Append(TextBox(3, "Body", 762000, 2057400, textWidth, 4114800, slide.Lines, slide.Lines.Count > 6 ? 1600 : 2000, bold: false, ink, system.BodyFont));
            var rels = new StringBuilder($"""<Relationships xmlns="{Rel}"><Relationship Id="rId1" Type="{RelType}/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>""");
            if (hasPicture)
            {
                var ext = Path.GetExtension(slide.Picture!).ToLowerInvariant() is ".jpg" or ".jpeg" ? "jpeg" : "png";
                var media = $"ppt/media/image{i + 1}.{ext}";
                zip.CreateEntryFromFile(slide.Picture!, media);
                rels.Append($"""<Relationship Id="rId2" Type="{RelType}/image" Target="../media/image{i + 1}.{ext}"/>""");
                // Fit the picture into the right column, keeping its aspect ratio.
                var (w, h) = ImageSize(slide.Picture!)!.Value;
                long boxX = 7239000, boxY = 640080, boxW = 4191000, boxH = 5577840;
                var scale = Math.Min((double)boxW / w, (double)boxH / h);
                long cx = (long)(w * scale), cy = (long)(h * scale);
                shapes.Append($"""<p:pic><p:nvPicPr><p:cNvPr id="4" name="Picture"/><p:cNvPicPr><a:picLocks noChangeAspect="1"/></p:cNvPicPr><p:nvPr/></p:nvPicPr><p:blipFill><a:blip r:embed="rId2"/><a:stretch><a:fillRect/></a:stretch></p:blipFill><p:spPr><a:xfrm><a:off x="{boxX + (boxW - cx) / 2}" y="{boxY + (boxH - cy) / 2}"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr></p:pic>""");
            }
            rels.Append("</Relationships>");
            // A short accent rule under the title, like the design system's one accent color.
            shapes.Append($"""<p:sp><p:nvSpPr><p:cNvPr id="5" name="Accent"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr><p:spPr><a:xfrm><a:off x="762000" y="1905000"/><a:ext cx="914400" cy="54864"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:solidFill><a:srgbClr val="{accent}"/></a:solidFill><a:ln><a:noFill/></a:ln></p:spPr></p:sp>""");
            Add($"ppt/slides/slide{i + 1}.xml", $"""<p:sld xmlns:a="{A}" xmlns:r="{R}" xmlns:p="{P}"><p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr/>{shapes}</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>""");
            Add($"ppt/slides/_rels/slide{i + 1}.xml.rels", rels.ToString());
        }
    }

    private static string TextBox(int id, string name, long x, long y, long cx, long cy, IReadOnlyList<string> lines, int size, bool bold, string color, string font)
    {
        var paragraphs = new StringBuilder();
        foreach (var line in lines)
            paragraphs.Append($"""<a:p><a:pPr><a:spcAft><a:spcPts val="{(bold ? 0 : 900)}"/></a:spcAft></a:pPr><a:r><a:rPr lang="en-US" sz="{size}" b="{(bold ? 1 : 0)}" dirty="0"><a:solidFill><a:srgbClr val="{color}"/></a:solidFill><a:latin typeface="{X(font)}"/></a:rPr><a:t>{X(line)}</a:t></a:r></a:p>""");
        return $"""<p:sp><p:nvSpPr><p:cNvPr id="{id}" name="{name}"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr><p:spPr><a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/></p:spPr><p:txBody><a:bodyPr wrap="square" lIns="0" tIns="0" rIns="0" bIns="0" anchor="t"><a:normAutofit/></a:bodyPr><a:lstStyle/>{paragraphs}</p:txBody></p:sp>""";
    }

    private static string Theme(DesignSystem system, string background, string ink, string accent) =>
        $"""<a:theme xmlns:a="{A}" name="{X(system.Name)}"><a:themeElements><a:clrScheme name="{X(system.Name)}"><a:dk1><a:srgbClr val="{ink}"/></a:dk1><a:lt1><a:srgbClr val="{background}"/></a:lt1><a:dk2><a:srgbClr val="{ink}"/></a:dk2><a:lt2><a:srgbClr val="{Hex(system.Colors.Length > 1 ? system.Colors[1] : system.Colors[0])}"/></a:lt2><a:accent1><a:srgbClr val="{accent}"/></a:accent1><a:accent2><a:srgbClr val="{Hex(system.Colors.Length > 2 ? system.Colors[2] : system.Colors[0])}"/></a:accent2><a:accent3><a:srgbClr val="{accent}"/></a:accent3><a:accent4><a:srgbClr val="{accent}"/></a:accent4><a:accent5><a:srgbClr val="{accent}"/></a:accent5><a:accent6><a:srgbClr val="{accent}"/></a:accent6><a:hlink><a:srgbClr val="{accent}"/></a:hlink><a:folHlink><a:srgbClr val="{accent}"/></a:folHlink></a:clrScheme><a:fontScheme name="{X(system.Name)}"><a:majorFont><a:latin typeface="{X(system.HeadingFont)}"/><a:ea typeface=""/><a:cs typeface=""/></a:majorFont><a:minorFont><a:latin typeface="{X(system.BodyFont)}"/><a:ea typeface=""/><a:cs typeface=""/></a:minorFont></a:fontScheme><a:fmtScheme name="LLM Hub"><a:fillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:fillStyleLst><a:lnStyleLst><a:ln w="6350"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln><a:ln w="12700"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln><a:ln w="19050"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln></a:lnStyleLst><a:effectStyleLst><a:effectStyle><a:effectLst/></a:effectStyle><a:effectStyle><a:effectLst/></a:effectStyle><a:effectStyle><a:effectLst/></a:effectStyle></a:effectStyleLst><a:bgFillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:bgFillStyleLst></a:fmtScheme></a:themeElements></a:theme>""";

    /// <summary>Width and height from a PNG or JPEG header, or null for anything else.</summary>
    private static (int W, int H)? ImageSize(string path)
    {
        try
        {
            var head = new byte[64 * 1024];
            int n;
            using (var f = File.OpenRead(path))
                n = f.Read(head, 0, head.Length);
            if (n > 24 && head[0] == 0x89 && head[1] == 'P')
                return ((head[16] << 24) | (head[17] << 16) | (head[18] << 8) | head[19], (head[20] << 24) | (head[21] << 16) | (head[22] << 8) | head[23]);
            for (var i = 2; i + 9 < n; i++)
            {
                if (head[i] == 0xFF && head[i + 1] is 0xC0 or 0xC2)
                    return ((head[i + 7] << 8) | head[i + 8], (head[i + 5] << 8) | head[i + 6]);
            }
        }
        catch (IOException)
        {
        }
        return null;
    }

    private static string Hex(string color) => color.TrimStart('#').ToUpperInvariant() is { Length: 6 } hex ? hex : "FFFFFF";
    private static string X(string text) => SecurityElement.Escape(InvalidXmlRegex().Replace(text, "")) ?? "";

    [GeneratedRegex(@"[\x00-\x08\x0B\x0C\x0E-\x1F]")]
    private static partial Regex InvalidXmlRegex();
}
