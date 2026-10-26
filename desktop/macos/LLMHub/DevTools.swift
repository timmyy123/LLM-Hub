#if DEBUG
import AppKit
import ScreenCaptureKit

/// Debug builds only, same flags as Windows:
/// `LLMHub [--selftest catalogId|apple [--format litertlm] [--quant label] [--prompt text]] [--capture out.png]
/// [--page chat|agent|design|studio|settings|models|<settings section>] [--wait ms] [--theme dark|light] [--exit]`
/// Feature end-to-end runs (--agenttest, --designtest, --studiotest, --speechtest, --dictatetest, --modelstest) live in
/// DevTools+<Feature>.swift.
@MainActor
enum DevTools {
    static func runIfRequested() {
        let args = ProcessInfo.processInfo.arguments
        if let theme = value("--theme", args) {
            NSApp.appearance = NSAppearance(named: theme == "dark" ? .darkAqua : .aqua)
        }
        guard args.contains(where: { $0 == "--capture" || $0 == "--exportdesign" || ($0.hasPrefix("--") && $0.hasSuffix("test")) }) else { return }
        Task {
            if let id = value("--selftest", args) {
                await selfTest(id, format: value("--format", args) ?? "gguf", quant: value("--quant", args), prompt: value("--prompt", args))
            }
            await agentTest(args)
            await designTest(args)
            await exportDesignTest(args)
            await studioTest(args)
            await speechTest(args)
            await dictateTest(args)
            await chatTest(args)
            await modelsTest(args)
            switch value("--page", args) {
            case "settings": AppModel.shared.section = .settings(.general)
            case "models": AppModel.shared.section = .settings(.models)
            case "agent": AppModel.shared.section = .agent
            case "design": AppModel.shared.section = .design
            case "studio": AppModel.shared.section = .studio
            case let page? where SettingsSection(rawValue: page.replacingOccurrences(of: "settings", with: "")) != nil && page != "settings":
                AppModel.shared.section = .settings(SettingsSection(rawValue: page.replacingOccurrences(of: "settings", with: ""))!)
            case nil, "chat": AppModel.shared.section = .chat
            default: break
            }
            try? await Task.sleep(for: .milliseconds(Int(value("--wait", args) ?? "") ?? 2500))
            if let path = value("--capture", args) { await capture(to: path) }
            if args.contains("--exit") { NSApp.terminate(nil) }
        }
    }

    static func value(_ name: String, _ args: [String]) -> String? {
        guard let i = args.firstIndex(of: name), i + 1 < args.count else { return nil }
        return args[i + 1]
    }

    /// The window as the window server composites it (Liquid Glass included), through ScreenCaptureKit's own-process
    /// content, which needs no screen recording permission. Falls back to rendering the views, which cannot draw glass.
    private static func capture(to path: String) async {
        guard let window = NSApp.windows.first(where: { $0.isVisible && $0.frame.width > 600 }) ?? NSApp.windows.first(where: { $0.isVisible }),
              let view = window.contentView?.superview else {
            Log.info("Capture skipped: no visible window (\(NSApp.windows.count) windows)")
            return
        }
        do {
            // As the key window, the way people see it (an inactive window greys its controls).
            NSApp.activate()
            window.makeKeyAndOrderFront(nil)
            try? await Task.sleep(for: .milliseconds(400))
            let content = try await SCShareableContent.currentProcess
            guard let shared = content.windows.first(where: { $0.windowID == CGWindowID(window.windowNumber) }),
                  let display = content.displays.first(where: { $0.frame.intersects(shared.frame) }) ?? content.displays.first else {
                throw NSError(domain: "capture", code: 1, userInfo: [NSLocalizedDescriptionKey: "window \(window.windowNumber) not shareable"])
            }
            // The window alone, cut from its display: a single-window filter would also take the input-source bubble macOS
            // shows at a focused text field (a separate window) and scale both into the picture.
            let filter = SCContentFilter(display: display, including: [shared])
            let config = SCStreamConfiguration()
            config.sourceRect = shared.frame.offsetBy(dx: -display.frame.minX, dy: -display.frame.minY)
            config.width = Int(shared.frame.width * CGFloat(filter.pointPixelScale))
            config.height = Int(shared.frame.height * CGFloat(filter.pointPixelScale))
            config.showsCursor = false
            config.backgroundColor = .clear
            let image = try await SCScreenshotManager.captureImage(contentFilter: filter, configuration: config)
            try NSBitmapImageRep(cgImage: image).representation(using: .png, properties: [:])?.write(to: URL(fileURLWithPath: path))
            Log.info("Captured \(path) (window server, \(image.width)x\(image.height))")
            return
        } catch {
            Log.info("Window server capture unavailable (\(error.localizedDescription)); rendering the views instead")
        }
        guard let rep = view.bitmapImageRepForCachingDisplay(in: view.bounds) else { return }
        view.cacheDisplay(in: view.bounds, to: rep)
        try? rep.representation(using: .png, properties: [:])?.write(to: URL(fileURLWithPath: path))
        Log.info("Captured \(path)")
    }

    private static func selfTest(_ catalogId: String, format: String, quant: String?, prompt: String?) async {
        let app = AppModel.shared
        if catalogId == "apple" {
            Log.info("SELFTEST apple availability: \(AppleModel.unavailableReason ?? "available")")
            await chat(AppleModel.installed, prompt: prompt)
            return
        }
        do {
            guard let model = app.catalog.models.first(where: { $0.id == catalogId }),
                  let source = model.sources.first(where: { $0.format == format }) else { throw NSError(domain: "selftest", code: 1) }
            let variants = format == "litertlm" ? try await app.hub.liteRtVariants(source.repo) : try await app.hub.ggufVariants(source.repo)
            Log.info("SELFTEST variants: \(variants.map { "\($0.quant) \(Fmt.bytes($0.totalBytes)) \($0.files.map(\.fileName).joined(separator: "+"))" }.joined(separator: ", "))")
            let variant = variants.first { $0.quant == quant } ?? variants.first { $0.quant == source.preferredQuant } ?? variants[0]
            Log.info("SELFTEST \(variant.modelId): \(variant.files.count) file(s), mmproj=\(variant.mmproj?.fileName ?? "-"), \(Fmt.bytes(variant.totalBytes))")
            if app.modelStore.find(variant.modelId) == nil {
                app.downloads.start(variant, name: model.name, catalogId: model.id, tags: model.tags, contextLength: model.contextLength)
                var last = Date()
                while let job = app.downloads.find(variant.modelId) {
                    try await Task.sleep(for: .seconds(1))
                    if Date().timeIntervalSince(last) > 15 {
                        Log.info("SELFTEST download \(String(format: "%.1f", job.percent))% \(job.status)")
                        last = Date()
                    }
                }
                if let error = app.downloads.errors[variant.modelId] { Log.info("SELFTEST download ended: \(error)") }
            }
            guard let installed = app.modelStore.find(variant.modelId) else { throw NSError(domain: "selftest", code: 2) }
            await chat(installed, prompt: prompt)
        } catch {
            Log.error("SELFTEST failed", error)
        }
    }

    private static func chat(_ installed: InstalledModel, prompt: String?) async {
        let app = AppModel.shared
        do {
            app.chat.newChat()
            app.chat.selectedModelId = installed.id
            app.chat.draft = prompt ?? "In two sentences, what makes Lisbon worth visiting?"
            let start = Date()
            app.chat.send()
            while app.chat.isGenerating { try await Task.sleep(for: .milliseconds(250)) }
            let reply = app.chat.messages.last
            Log.info("SELFTEST reply in \(String(format: "%.1f", Date().timeIntervalSince(start)))s (incl. load) status='\(reply?.status ?? "")': \(reply?.content ?? "")")
        } catch {
            Log.error("SELFTEST failed", error)
        }
    }
}
#endif
