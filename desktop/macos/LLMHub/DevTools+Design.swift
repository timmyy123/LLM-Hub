#if DEBUG
import AppKit
import WebKit

extension DevTools {
    /// `--designtest "<request>" [--kind Prototype|Slides|Image|Motion] [--model <id or catalog id>] [--followup "<change>"]
    /// [--designexport <folder>]`:
    /// creates a design, writes version 1 (and version 2 from the follow-up) with the real local model through the same
    /// DesignService the tab uses, logs each result, then probes the WebKit preview and clicks the first heading through the
    /// element picker (same checks as Windows/Linux ProbePreviewAsync).
    static func designTest(_ args: [String]) async {
        guard let request = value("--designtest", args) else { return }
        let app = AppModel.shared
        let design = app.design
        if let id = value("--model", args) {
            if let model = design.models.first(where: { $0.id == id || $0.catalogId == id }) {
                design.selectedModelId = model.id
            } else {
                Log.info("DESIGNTEST model \(id) is not installed; using \(design.selectedModelId ?? "none")")
            }
        }
        let kindName = (value("--kind", args) ?? "Prototype").lowercased()
        design.kind = DesignKind.allCases.first { $0.rawValue.lowercased() == kindName } ?? .prototype
        app.section = .design
        design.newProject()
        Log.info("DESIGNTEST start: kind=\(design.kind.rawValue) system=\(design.system.id) model=\(design.selectedModelId ?? "none")")

        for text in [request, value("--followup", args)].compactMap({ $0 }) {
            let start = Date()
            design.draft = text
            await design.send()?.value
            let seconds = Int(Date().timeIntervalSince(start).rounded())
            let file = design.previewFile
            let bytes = file.flatMap { (try? FileManager.default.attributesOfItem(atPath: $0.path))?[.size] as? Int } ?? 0
            Log.info("DESIGNTEST v\(design.version) in \(seconds)s: \(file.map { "\(bytes) bytes \($0.path)" } ?? "no file") status='\(design.status)' name='\(design.selected?.name ?? "")' steps='\(design.messages.last?.steps.joined(separator: ";") ?? "")'")
        }
        await probePreview(snapshot: value("--capture", args).map { ($0 as NSString).deletingPathExtension + "-preview.png" })
        if let folder = value("--designexport", args) { await exportAll(to: URL(fileURLWithPath: folder, isDirectory: true)) }
    }

    /// `--exportdesign html|pdf|pptx|zip|md [--out <path>] [--kind Prototype|Slides|Image|Motion]`, as on Windows: exports the
    /// newest design that has a version, without generating, and logs the slides a PPTX is made from.
    static func exportDesignTest(_ args: [String]) async {
        guard let format = value("--exportdesign", args) else { return }
        let design = AppModel.shared.design
        let kind = value("--kind", args)?.lowercased()
        guard let project = design.projects.filter({ $0.versions > 0 && (kind == nil || $0.kind.rawValue.lowercased() == kind) })
            .max(by: { $0.updatedAt < $1.updatedAt }), let file = project.currentFile else {
            Log.info("EXPORTTEST no design project")
            return
        }
        design.select(project.id)
        let target = URL(fileURLWithPath: value("--out", args) ?? NSTemporaryDirectory() + "design.\(format)")
        do {
            try await design.export(format, to: target)
            let bytes = (try? FileManager.default.attributesOfItem(atPath: target.path))?[.size] as? Int ?? 0
            let slides = DesignDocument.slides((try? String(contentsOf: file, encoding: .utf8)) ?? "", folder: project.folder, name: project.name)
            Log.info("EXPORTTEST \(project.name) (\(project.kind.rawValue)) -> \(target.path): \(bytes) bytes, \(slides.count) slides: "
                + slides.map { "\($0.title) [\($0.lines.count)\($0.picture == nil ? "" : " + photo")]" }.joined(separator: " | "))
        } catch {
            Log.error("EXPORTTEST \(format) failed", error)
        }
    }

    /// `--designexport <folder>`: writes the open design as HTML, PDF, PPTX, ZIP and Markdown there (the menu's formats, no
    /// save panel), and logs the slides the PPTX was made from.
    private static func exportAll(to folder: URL) async {
        let design = AppModel.shared.design
        try? FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        for format in ["html", "pdf", "pptx", "zip", "md"] {
            let target = folder.appendingPathComponent("design.\(format)")
            do {
                try await design.export(format, to: target)
                let bytes = (try? FileManager.default.attributesOfItem(atPath: target.path))?[.size] as? Int ?? 0
                Log.info("DESIGNTEST export \(format): \(bytes) bytes \(target.path)")
            } catch {
                Log.error("DESIGNTEST export \(format) failed", error)
            }
        }
        if let project = design.selected, let file = project.currentFile, let html = try? String(contentsOf: file, encoding: .utf8) {
            let slides = DesignDocument.slides(html, folder: project.folder, name: project.name)
            Log.info("DESIGNTEST slides \(slides.count): \(slides.map { "\($0.title) [\($0.lines.count)\($0.picture == nil ? "" : " + photo")]" }.joined(separator: " | "))")
        }
    }

    /// What WebKit loaded (title, size, the picker script), then a click on the first heading through the page, so the
    /// picker → message handler → selected element path runs.
    /// `snapshot`: also saves what WebKit drew (the window capture cannot see web content).
    private static func probePreview(snapshot: String?) async {
        let design = AppModel.shared.design
        // Let SwiftUI put the preview on screen, then make sure the newest version is the one loaded.
        try? await Task.sleep(for: .milliseconds(800))
        design.preview.show(design.previewFile)
        await design.preview.waitForLoad()
        try? await Task.sleep(for: .milliseconds(300))
        let probe = "JSON.stringify({title: document.title, chars: document.body.innerText.length, elements: document.querySelectorAll('*').length, "
            + "height: document.documentElement.scrollHeight, picker: !!window.__llmhubPicker, "
            + "clicked: (() => { const h = document.querySelector('h1'); if (h) h.click(); return h ? h.innerText.slice(0, 40) : null; })()})"
        // The phone-width check runs when the new version loads and posts back after 400 ms.
        try? await Task.sleep(for: .milliseconds(1000))
        let result = await design.preview.evaluate(probe)
        try? await Task.sleep(for: .milliseconds(500))
        Log.info("PREVIEW \(design.preview.webView.url?.path ?? "-") \(result) selected='\(design.selectedElement)' "
            + "properties='\(design.selectedProperties.replacingOccurrences(of: "\n", with: "; "))' steps='\(design.messages.last?.steps.joined(separator: ";") ?? "")'")
        if let snapshot, let image = try? await design.preview.webView.takeSnapshot(configuration: nil),
           let tiff = image.tiffRepresentation, let png = NSBitmapImageRep(data: tiff)?.representation(using: .png, properties: [:]) {
            try? png.write(to: URL(fileURLWithPath: snapshot))
            Log.info("PREVIEW snapshot \(snapshot)")
        }
    }
}
#endif
