import Foundation

/// One slide for `PptxWriter`: a title, body lines and an optional picture (PNG or JPEG).
struct PptxSlide {
    let title: String
    let lines: [String]
    let picture: URL?
}

/// A small PowerPoint (.pptx) writer with no dependencies: 16:9 slides with editable text in the design system's colors and
/// fonts, and the generated photos. Each slide has a title, body text and an optional picture. A port of
/// core/LLMHub.Core/Services/Design/PptxWriter.cs; the parts are written to a folder and zipped with /usr/bin/zip.
enum PptxWriter {
    private static let slideWidth: Int64 = 12_192_000, slideHeight: Int64 = 6_858_000
    private static let p = "http://schemas.openxmlformats.org/presentationml/2006/main"
    private static let a = "http://schemas.openxmlformats.org/drawingml/2006/main"
    private static let r = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    private static let rel = "http://schemas.openxmlformats.org/package/2006/relationships"
    private static let relType = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"

    static func write(to target: URL, title: String, slides: [PptxSlide], system: DesignSystem) async throws {
        guard !slides.isEmpty else {
            throw NSError(domain: "LLMHub", code: 43, userInfo: [NSLocalizedDescriptionKey: L.s("DesignNothingYet")])
        }
        let fm = FileManager.default
        let staging = fm.temporaryDirectory.appendingPathComponent("llmhub-pptx-\(UUID().uuidString)", isDirectory: true)
        defer { try? fm.removeItem(at: staging) }
        func add(_ name: String, _ xml: String) throws {
            let url = staging.appendingPathComponent(name)
            try fm.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Data(("<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>\n" + xml).utf8).write(to: url)
        }
        let colors = system.colors
        let background = hex(colors[0])
        let ink = hex(colors[colors.count - 1])
        let accent = hex(colors.count > 3 ? colors[3] : colors[colors.count - 1])

        var types = #"<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Default Extension="png" ContentType="image/png"/><Default Extension="jpeg" ContentType="image/jpeg"/>"#
        types += #"<Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>"#
        types += #"<Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>"#
        types += #"<Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>"#
        types += #"<Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>"#
        types += #"<Override PartName="/ppt/presProps.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presProps+xml"/>"#
        types += #"<Override PartName="/ppt/viewProps.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.viewProps+xml"/>"#
        types += #"<Override PartName="/ppt/tableStyles.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.tableStyles+xml"/>"#
        types += #"<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>"#
        types += #"<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>"#
        for i in 1...slides.count {
            types += #"<Override PartName="/ppt/slides/slide\#(i).xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>"#
        }
        types += "</Types>"
        try add("[Content_Types].xml", types)

        try add("_rels/.rels", #"<Relationships xmlns="\#(rel)"><Relationship Id="rId1" Type="\#(relType)/officeDocument" Target="ppt/presentation.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/><Relationship Id="rId3" Type="\#(relType)/extended-properties" Target="docProps/app.xml"/></Relationships>"#)
        let created = ISO8601DateFormatter().string(from: Date())
        try add("docProps/core.xml", #"<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"><dc:title>\#(x(title))</dc:title><dc:creator>LLM Hub</dc:creator><dcterms:created xsi:type="dcterms:W3CDTF">\#(created)</dcterms:created></cp:coreProperties>"#)
        try add("docProps/app.xml", #"<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"><Application>LLM Hub</Application><Slides>\#(slides.count)</Slides></Properties>"#)

        var ids = ""
        var presRels = #"<Relationships xmlns="\#(rel)"><Relationship Id="rId1" Type="\#(relType)/slideMaster" Target="slideMasters/slideMaster1.xml"/><Relationship Id="rId2" Type="\#(relType)/theme" Target="theme/theme1.xml"/><Relationship Id="rId3" Type="\#(relType)/presProps" Target="presProps.xml"/><Relationship Id="rId4" Type="\#(relType)/viewProps" Target="viewProps.xml"/><Relationship Id="rId5" Type="\#(relType)/tableStyles" Target="tableStyles.xml"/>"#
        for i in 1...slides.count {
            ids += #"<p:sldId id="\#(255 + i)" r:id="rId\#(10 + i)"/>"#
            presRels += #"<Relationship Id="rId\#(10 + i)" Type="\#(relType)/slide" Target="slides/slide\#(i).xml"/>"#
        }
        presRels += "</Relationships>"
        try add("ppt/presentation.xml", #"<p:presentation xmlns:a="\#(a)" xmlns:r="\#(r)" xmlns:p="\#(p)"><p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst><p:sldIdLst>\#(ids)</p:sldIdLst><p:sldSz cx="\#(slideWidth)" cy="\#(slideHeight)"/><p:notesSz cx="6858000" cy="9144000"/></p:presentation>"#)
        try add("ppt/_rels/presentation.xml.rels", presRels)
        try add("ppt/presProps.xml", #"<p:presentationPr xmlns:a="\#(a)" xmlns:r="\#(r)" xmlns:p="\#(p)"/>"#)
        try add("ppt/viewProps.xml", #"<p:viewPr xmlns:a="\#(a)" xmlns:r="\#(r)" xmlns:p="\#(p)"/>"#)
        try add("ppt/tableStyles.xml", #"<a:tblStyleLst xmlns:a="\#(a)" def="{5C22544A-7EE6-4342-B048-85BDC9FD1C3A}"/>"#)

        let emptyTree = #"<p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr/></p:spTree>"#
        try add("ppt/slideMasters/slideMaster1.xml", #"<p:sldMaster xmlns:a="\#(a)" xmlns:r="\#(r)" xmlns:p="\#(p)"><p:cSld><p:bg><p:bgPr><a:solidFill><a:srgbClr val="\#(background)"/></a:solidFill><a:effectLst/></p:bgPr></p:bg>\#(emptyTree)</p:cSld><p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/><p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst></p:sldMaster>"#)
        try add("ppt/slideMasters/_rels/slideMaster1.xml.rels", #"<Relationships xmlns="\#(rel)"><Relationship Id="rId1" Type="\#(relType)/slideLayout" Target="../slideLayouts/slideLayout1.xml"/><Relationship Id="rId2" Type="\#(relType)/theme" Target="../theme/theme1.xml"/></Relationships>"#)
        try add("ppt/slideLayouts/slideLayout1.xml", #"<p:sldLayout xmlns:a="\#(a)" xmlns:r="\#(r)" xmlns:p="\#(p)" type="blank" preserve="1"><p:cSld name="Blank">\#(emptyTree)</p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sldLayout>"#)
        try add("ppt/slideLayouts/_rels/slideLayout1.xml.rels", #"<Relationships xmlns="\#(rel)"><Relationship Id="rId1" Type="\#(relType)/slideMaster" Target="../slideMasters/slideMaster1.xml"/></Relationships>"#)
        try add("ppt/theme/theme1.xml", theme(system, background: background, ink: ink, accent: accent))

        for (i, slide) in slides.enumerated() {
            var shapes = ""
            let size = slide.picture.flatMap(imageSize)
            let textWidth: Int64 = size != nil ? 6_100_000 : 10_668_000
            shapes += textBox(id: 2, name: "Title", x: 762_000, y: 640_080, cx: textWidth, cy: 1_300_000, lines: [slide.title], size: 4000, bold: true,
                              color: ink, font: system.headingFont)
            if !slide.lines.isEmpty {
                shapes += textBox(id: 3, name: "Body", x: 762_000, y: 2_057_400, cx: textWidth, cy: 4_114_800, lines: slide.lines,
                                  size: slide.lines.count > 6 ? 1600 : 2000, bold: false, color: ink, font: system.bodyFont)
            }
            var rels = #"<Relationships xmlns="\#(rel)"><Relationship Id="rId1" Type="\#(relType)/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>"#
            if let picture = slide.picture, let size {
                let (w, h) = size
                let ext = ["jpg", "jpeg"].contains(picture.pathExtension.lowercased()) ? "jpeg" : "png"
                let media = staging.appendingPathComponent("ppt/media/image\(i + 1).\(ext)")
                try fm.createDirectory(at: media.deletingLastPathComponent(), withIntermediateDirectories: true)
                try fm.copyItem(at: picture, to: media)
                rels += #"<Relationship Id="rId2" Type="\#(relType)/image" Target="../media/image\#(i + 1).\#(ext)"/>"#
                // Fit the picture into the right column, keeping its aspect ratio.
                let boxX: Int64 = 7_239_000, boxY: Int64 = 640_080, boxW: Int64 = 4_191_000, boxH: Int64 = 5_577_840
                let scale = min(Double(boxW) / Double(w), Double(boxH) / Double(h))
                let cx = Int64(Double(w) * scale), cy = Int64(Double(h) * scale)
                shapes += #"<p:pic><p:nvPicPr><p:cNvPr id="4" name="Picture"/><p:cNvPicPr><a:picLocks noChangeAspect="1"/></p:cNvPicPr><p:nvPr/></p:nvPicPr><p:blipFill><a:blip r:embed="rId2"/><a:stretch><a:fillRect/></a:stretch></p:blipFill><p:spPr><a:xfrm><a:off x="\#(boxX + (boxW - cx) / 2)" y="\#(boxY + (boxH - cy) / 2)"/><a:ext cx="\#(cx)" cy="\#(cy)"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr></p:pic>"#
            }
            rels += "</Relationships>"
            // A short accent rule under the title, like the design system's one accent color.
            shapes += #"<p:sp><p:nvSpPr><p:cNvPr id="5" name="Accent"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr><p:spPr><a:xfrm><a:off x="762000" y="1905000"/><a:ext cx="914400" cy="54864"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:solidFill><a:srgbClr val="\#(accent)"/></a:solidFill><a:ln><a:noFill/></a:ln></p:spPr></p:sp>"#
            try add("ppt/slides/slide\(i + 1).xml", #"<p:sld xmlns:a="\#(a)" xmlns:r="\#(r)" xmlns:p="\#(p)"><p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr/>\#(shapes)</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>"#)
            try add("ppt/slides/_rels/slide\(i + 1).xml.rels", rels)
        }
        try? fm.removeItem(at: target)
        try await DesignService.zip(staging, to: target)
    }

    private static func textBox(id: Int, name: String, x: Int64, y: Int64, cx: Int64, cy: Int64, lines: [String], size: Int, bold: Bool,
                                color: String, font: String) -> String {
        var paragraphs = ""
        for line in lines {
            paragraphs += #"<a:p><a:pPr><a:spcAft><a:spcPts val="\#(bold ? 0 : 900)"/></a:spcAft></a:pPr><a:r><a:rPr lang="en-US" sz="\#(size)" b="\#(bold ? 1 : 0)" dirty="0"><a:solidFill><a:srgbClr val="\#(color)"/></a:solidFill><a:latin typeface="\#(Self.x(font))"/></a:rPr><a:t>\#(Self.x(line))</a:t></a:r></a:p>"#
        }
        return #"<p:sp><p:nvSpPr><p:cNvPr id="\#(id)" name="\#(name)"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr><p:spPr><a:xfrm><a:off x="\#(x)" y="\#(y)"/><a:ext cx="\#(cx)" cy="\#(cy)"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/></p:spPr><p:txBody><a:bodyPr wrap="square" lIns="0" tIns="0" rIns="0" bIns="0" anchor="t"><a:normAutofit/></a:bodyPr><a:lstStyle/>\#(paragraphs)</p:txBody></p:sp>"#
    }

    private static func theme(_ system: DesignSystem, background: String, ink: String, accent: String) -> String {
        let colors = system.colors
        let lt2 = hex(colors.count > 1 ? colors[1] : colors[0])
        let accent2 = hex(colors.count > 2 ? colors[2] : colors[0])
        let name = x(system.name)
        return #"<a:theme xmlns:a="\#(a)" name="\#(name)"><a:themeElements><a:clrScheme name="\#(name)"><a:dk1><a:srgbClr val="\#(ink)"/></a:dk1><a:lt1><a:srgbClr val="\#(background)"/></a:lt1><a:dk2><a:srgbClr val="\#(ink)"/></a:dk2><a:lt2><a:srgbClr val="\#(lt2)"/></a:lt2><a:accent1><a:srgbClr val="\#(accent)"/></a:accent1><a:accent2><a:srgbClr val="\#(accent2)"/></a:accent2><a:accent3><a:srgbClr val="\#(accent)"/></a:accent3><a:accent4><a:srgbClr val="\#(accent)"/></a:accent4><a:accent5><a:srgbClr val="\#(accent)"/></a:accent5><a:accent6><a:srgbClr val="\#(accent)"/></a:accent6><a:hlink><a:srgbClr val="\#(accent)"/></a:hlink><a:folHlink><a:srgbClr val="\#(accent)"/></a:folHlink></a:clrScheme><a:fontScheme name="\#(name)"><a:majorFont><a:latin typeface="\#(x(system.headingFont))"/><a:ea typeface=""/><a:cs typeface=""/></a:majorFont><a:minorFont><a:latin typeface="\#(x(system.bodyFont))"/><a:ea typeface=""/><a:cs typeface=""/></a:minorFont></a:fontScheme><a:fmtScheme name="LLM Hub"><a:fillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:fillStyleLst><a:lnStyleLst><a:ln w="6350"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln><a:ln w="12700"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln><a:ln w="19050"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln></a:lnStyleLst><a:effectStyleLst><a:effectStyle><a:effectLst/></a:effectStyle><a:effectStyle><a:effectLst/></a:effectStyle><a:effectStyle><a:effectLst/></a:effectStyle></a:effectStyleLst><a:bgFillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:bgFillStyleLst></a:fmtScheme></a:themeElements></a:theme>"#
    }

    /// Width and height from a PNG or JPEG header, or nil for anything else.
    static func imageSize(_ url: URL) -> (Int, Int)? {
        guard let handle = try? FileHandle(forReadingFrom: url) else { return nil }
        defer { try? handle.close() }
        guard let data = try? handle.read(upToCount: 64 * 1024) else { return nil }
        let head = [UInt8](data)
        let n = head.count
        func be(_ i: Int, _ count: Int) -> Int { (i..<i + count).reduce(0) { $0 << 8 | Int(head[$1]) } }
        if n > 24 && head[0] == 0x89 && head[1] == UInt8(ascii: "P") { return (be(16, 4), be(20, 4)) }
        var i = 2
        while i + 9 < n {
            if head[i] == 0xFF && (head[i + 1] == 0xC0 || head[i + 1] == 0xC2) { return (be(i + 7, 2), be(i + 5, 2)) }
            i += 1
        }
        return nil
    }

    private static func hex(_ color: String) -> String {
        let value = color.trimmingCharacters(in: CharacterSet(charactersIn: "#")).uppercased()
        return value.count == 6 ? value : "FFFFFF"
    }

    /// Text for an XML element or attribute: control characters XML forbids are dropped, markup characters escaped.
    private static func x(_ text: String) -> String {
        var out = ""
        for scalar in text.unicodeScalars {
            switch scalar {
            case "&": out += "&amp;"
            case "<": out += "&lt;"
            case ">": out += "&gt;"
            case "\"": out += "&quot;"
            case "'": out += "&apos;"
            case "\t", "\n", "\r": out.unicodeScalars.append(scalar)
            default: if scalar.value >= 0x20 { out.unicodeScalars.append(scalar) }
            }
        }
        return out
    }
}
