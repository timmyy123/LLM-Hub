import AppKit
import Foundation
import Observation

/// The Design tab, after open-design's approach (ported from LLMHub.Core's DesignService and DesignViewModel): the local model
/// writes one complete, self-contained HTML document inside an <artifact> block, shown in a live WebKit preview. Each result is
/// saved as a new version (design/<id>/v1.html, v2.html, … plus project.json); edits send the current document back with the
/// request, and the element the user clicked when there is one.
@MainActor @Observable
final class DesignService {
    private weak var app: AppModel?

    /// Every design, newest first.
    private(set) var projects: [DesignProject] = []
    /// The open design's id; nil before the first one.
    private(set) var selectedId: String?
    /// The open design's conversation (the user's request shows before the reply arrives).
    private(set) var messages: [DesignMessage] = []

    var kind = DesignKind.prototype
    var system = DesignSystem.all[0] {
        didSet {
            guard let index = selectedIndex, projects[index].systemId != system.id else { return }
            projects[index].systemId = system.id
            save(projects[index])
        }
    }
    var selectedModelId: String? {
        didSet {
            guard let id = selectedModelId, id != oldValue else { return }
            AppSettings.current.designModelId = id
            AppSettings.save()
        }
    }
    var draft = ""
    private(set) var isGenerating = false
    var status = ""
    /// Characters of the reply streamed so far, for "Designing… (2.1k)".
    private(set) var liveCount = 0
    /// Version shown in the preview (1-based), or 0 before the first.
    var version = 0
    /// desktop, tablet or phone.
    var previewSize = "desktop"
    /// The element clicked in the preview ("section · hours · "Open today""), sent with the next request.
    var selectedElement = ""

    /// The WebKit preview; lives as long as the app so the page survives switching tabs.
    @ObservationIgnored private(set) lazy var preview: DesignPreview = {
        let preview = DesignPreview()
        preview.onPick = { [weak self] picked in self?.selectedElement = picked }
        return preview
    }()
    @ObservationIgnored private var generation: Task<Void, Never>?

    func attach(_ app: AppModel) {
        self.app = app
        loadProjects()
        if let first = projects.first { select(first.id) }
        selectedModelId = defaultModel()?.id
    }

    // MARK: - State for the view

    var selected: DesignProject? { selectedIndex.map { projects[$0] } }
    private var selectedIndex: Int? { selectedId.flatMap { id in projects.firstIndex { $0.id == id } } }

    /// Name of the open design for the window title ("Design · Crumb & Co"), or "".
    var currentName: String { selected.map { $0.versions > 0 || !$0.name.isEmpty ? $0.displayName : "" } ?? "" }

    /// Installed models that can write a page (GGUF and LiteRT-LM; Apple's on-device model has too small a window).
    var models: [InstalledModel] { app?.modelStore.installed ?? [] }
    var selectedModel: InstalledModel? { models.first { $0.id == selectedModelId } }

    var previewFile: URL? {
        guard let project = selected, version > 0, version <= project.versions else { return nil }
        return project.versionFile(version)
    }
    var hasPreview: Bool { previewFile != nil }
    var versionText: String {
        guard let project = selected, project.versions > 0 else { return "" }
        return L.f("DesignVersion", "\(version)", "\(project.versions)")
    }
    /// Width the preview page is laid out at; nil fills the column.
    var previewWidth: CGFloat? {
        switch previewSize {
        case "phone": return 390
        case "tablet": return 820
        default: return nil
        }
    }

    /// "Designing… (2.1k) · h1 · hero-title" while a version is written.
    var progressText: String {
        L.s("DesignWorking") + (liveCount > 0 ? String(format: " (%.1fk)", Double(liveCount) / 1000) : "")
            + (selectedElement.isEmpty ? "" : " · " + selectedElement)
    }

    // MARK: - Projects

    private func loadProjects() {
        let dirs = (try? FileManager.default.contentsOfDirectory(at: AppPaths.design, includingPropertiesForKeys: nil)) ?? []
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        projects = dirs.compactMap { dir in
            guard let data = try? Data(contentsOf: dir.appendingPathComponent("project.json")) else { return nil }
            return try? decoder.decode(DesignProject.self, from: data)
        }.sorted { $0.updatedAt > $1.updatedAt }
    }

    private func save(_ project: DesignProject) {
        var project = project
        project.updatedAt = Date()
        if let index = projects.firstIndex(where: { $0.id == project.id }) { projects[index] = project }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        do {
            try encoder.encode(project).write(to: project.folder.appendingPathComponent("project.json"), options: .atomic)
        } catch {
            Log.error("Could not save design \(project.id)", error)
        }
    }

    func select(_ id: String?) {
        guard !isGenerating || id == selectedId else { return }
        selectedId = id
        messages = selected?.history ?? []
        if let project = selected {
            system = project.system
            kind = project.kind
        }
        version = selected?.versions ?? 0
        selectedElement = ""
        status = ""
    }

    @discardableResult
    func newProject() -> DesignProject {
        let project = DesignProject(kind: kind, systemId: system.id)
        projects.insert(project, at: 0)
        save(project)
        select(project.id)
        return project
    }

    func delete(_ id: String) {
        guard let index = projects.firstIndex(where: { $0.id == id }) else { return }
        if id == selectedId { stop() }
        let project = projects.remove(at: index)
        try? FileManager.default.removeItem(at: project.folder)
        if selectedId == id { select(projects.first?.id) }
    }

    /// A different kind starts a new project, so a deck never overwrites a prototype; an empty project just changes kind.
    func chooseKind(_ kind: DesignKind) {
        guard !isGenerating else { return }
        self.kind = kind
        guard let index = selectedIndex else { return }
        if projects[index].kind != kind && projects[index].versions > 0 {
            newProject()
        } else if projects[index].versions == 0, projects[index].kind != kind {
            projects[index].kind = kind
            save(projects[index])
        }
    }

    // MARK: - Versions

    func previousVersion() { if version > 1 { version -= 1 } }
    func nextVersion() { if let p = selected, version < p.versions { version += 1 } }

    // MARK: - Generation

    /// The design model: the last one picked here, else the catalog's agent default, else the chat model, else any GGUF.
    private func defaultModel() -> InstalledModel? {
        if let id = AppSettings.current.designModelId, let model = models.first(where: { $0.id == id }) { return model }
        if let catalogId = app?.catalog.defaults["agent"], let model = models.first(where: { $0.catalogId == catalogId }) { return model }
        if let id = app?.chat.selectedModelId, let model = models.first(where: { $0.id == id }) { return model }
        return models.first { $0.format == "gguf" } ?? models.first
    }

    /// Sends the composer text; the returned task finishes when the new version is saved (or the run failed or stopped).
    @discardableResult
    func send() -> Task<Void, Never>? {
        let request = draft.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !request.isEmpty, !isGenerating, let app else { return nil }
        if selectedModel == nil { selectedModelId = defaultModel()?.id }
        guard let model = selectedModel else {
            status = L.s("AgentNoModel")
            return nil
        }
        if selected == nil { newProject() }
        guard let project = selected else { return nil }
        draft = ""
        messages.append(DesignMessage(role: "user", text: request))
        isGenerating = true
        liveCount = 0
        status = ""
        let element = selectedElement
        let task = Task { [weak self] in
            guard let self else { return }
            let start = Date()
            do {
                let saved = try await self.generate(project, request: request, element: element, model: model, app: app)
                Log.info("Design \(saved.id) v\(saved.versions) (\(saved.kind.rawValue), \(saved.systemId)) in \(Int(Date().timeIntervalSince(start)))s")
                if self.selectedId == saved.id {
                    self.messages = saved.history
                    self.version = saved.versions
                    self.selectedElement = ""
                }
                self.status = ""
            } catch is CancellationError {
                self.status = ""
            } catch let error as URLError where error.code == .cancelled {
                self.status = ""
            } catch {
                Log.error("Design generation failed", error)
                self.status = L.f("ErrorPrefix", error.localizedDescription)
            }
            self.isGenerating = false
            self.liveCount = 0
            self.generation = nil
        }
        generation = task
        return task
    }

    func stop() { generation?.cancel() }

    /// Writes the next version of `project` and returns the saved project.
    private func generate(_ project: DesignProject, request: String, element: String, model: InstalledModel, app: AppModel) async throws -> DesignProject {
        // Design documents are long (the current page goes back with every edit): at least a 32K window.
        let context = max(AppSettings.current.contextLength, 32768)
        if !app.engines.isReady(model, contextLength: context) {
            status = L.f("LoadingModel", model.name)
            try await app.engines.load(model, contextLength: context)
        }
        try Task.checkCancellation()
        status = L.s("DesignWorking")

        var turns = [ChatTurn(role: "system", content: DesignDocument.systemPrompt(kind: project.kind, system: project.system))]
        turns += project.history.suffix(6).map { ChatTurn(role: $0.role, content: $0.text) }
        let current = project.currentFile.flatMap { try? String(contentsOf: $0, encoding: .utf8) }
        turns.append(ChatTurn(role: "user", content: DesignDocument.userTurn(request: request, current: current, selectedElement: element)))

        var text = ""
        var splitter = ThinkSplitter()
        for try await delta in app.engines.stream(model, turns: turns) {
            try Task.checkCancellation()
            if let c = delta.content {
                let answer = splitter.push(c).answer
                text += answer
                liveCount += answer.count
            }
        }
        try Task.checkCancellation()
        guard let html = DesignDocument.extractHtml(text) else {
            Log.info("Design reply without a page (\(text.count) chars): \(text.prefix(300))")
            throw NSError(domain: "LLMHub", code: 40, userInfo: [NSLocalizedDescriptionKey: L.s("DesignNoHtml")])
        }

        // The project may have been renamed or re-styled meanwhile; build on the stored copy.
        var saved = projects.first { $0.id == project.id } ?? project
        saved.versions += 1
        try html.write(to: saved.versionFile(saved.versions), atomically: true, encoding: .utf8)
        let note = DesignDocument.summary(text)
        saved.history.append(DesignMessage(role: "user", text: request))
        saved.history.append(DesignMessage(role: "assistant", text: note.isEmpty ? "Updated the design." : note))
        if saved.versions == 1 {
            saved.name = DesignDocument.title(html) ?? (request.count <= 40 ? request : String(request.prefix(37)) + "…")
        }
        if !projects.contains(where: { $0.id == saved.id }) { projects.insert(saved, at: 0) }
        save(saved)
        // Newest first.
        if let index = projects.firstIndex(where: { $0.id == saved.id }), index > 0 {
            projects.insert(projects.remove(at: index), at: 0)
        }
        return projects.first { $0.id == saved.id } ?? saved
    }

    // MARK: - Hand off and export

    /// Copies the current version to ~/Documents/LLM Hub/<name>/index.html and opens the Agent tab there.
    func handOffToAgent() {
        guard let project = selected, let file = project.currentFile else { return }
        let folder = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("LLM Hub", isDirectory: true)
            .appendingPathComponent(Self.safe(project.name), isDirectory: true)
        do {
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            let target = folder.appendingPathComponent("index.html")
            try? FileManager.default.removeItem(at: target)
            try FileManager.default.copyItem(at: file, to: target)
            Log.info("Design \(project.id) handed off to \(folder.path)")
        } catch {
            Log.error("Design hand-off failed", error)
            status = L.f("ErrorPrefix", error.localizedDescription)
            return
        }
        // As on Windows and Linux: a session in that folder with the hand-off request ready in the composer.
        if let agent = app?.agent.ui, agent.newSession(folder.path) != nil {
            agent.draft = L.s("DesignHandOffPrompt")
        }
        app?.section = .agent
    }

    /// Exports the current version: "html" (the file), "zip" (every version, newest as index.html), "md" (page text), "pdf".
    func export(_ format: String, to target: URL) async throws {
        guard let project = selected, let current = project.currentFile else {
            throw NSError(domain: "LLMHub", code: 41, userInfo: [NSLocalizedDescriptionKey: L.s("DesignNothingYet")])
        }
        let fm = FileManager.default
        switch format {
        case "html":
            try? fm.removeItem(at: target)
            try fm.copyItem(at: current, to: target)
        case "md":
            let html = try String(contentsOf: current, encoding: .utf8)
            try DesignDocument.htmlToMarkdown(html).write(to: target, atomically: true, encoding: .utf8)
        case "pdf":
            // The preview shows the version being exported (export always means the one on screen).
            preview.show(previewFile ?? current)
            await preview.waitForLoad()
            try await preview.pdf().write(to: target, options: .atomic)
        case "zip":
            let staging = fm.temporaryDirectory.appendingPathComponent("llmhub-design-\(UUID().uuidString)", isDirectory: true)
            try fm.createDirectory(at: staging.appendingPathComponent("versions", isDirectory: true), withIntermediateDirectories: true)
            defer { try? fm.removeItem(at: staging) }
            for n in 1...project.versions {
                let name = n == project.versions ? "index.html" : "versions/v\(n).html"
                try fm.copyItem(at: project.versionFile(n), to: staging.appendingPathComponent(name))
            }
            if project.versions == 1 { try? fm.removeItem(at: staging.appendingPathComponent("versions")) }
            try? fm.removeItem(at: target)
            try await Self.run("/usr/bin/zip", ["-q", "-r", target.path, "."], in: staging)
        default:
            throw NSError(domain: "LLMHub", code: 42, userInfo: [NSLocalizedDescriptionKey: "Unknown format \(format)"])
        }
        Log.info("Design \(project.id) exported \(format) to \(target.path)")
    }

    private static func run(_ tool: String, _ arguments: [String], in directory: URL) async throws {
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            let process = Process()
            process.executableURL = URL(fileURLWithPath: tool)
            process.arguments = arguments
            process.currentDirectoryURL = directory
            process.terminationHandler = { p in
                if p.terminationStatus == 0 {
                    continuation.resume()
                } else {
                    continuation.resume(throwing: NSError(domain: "LLMHub", code: Int(p.terminationStatus),
                                                          userInfo: [NSLocalizedDescriptionKey: "\(tool) exited with \(p.terminationStatus)"]))
                }
            }
            do { try process.run() } catch { continuation.resume(throwing: error) }
        }
    }

    /// A file and folder name from a design name.
    static func safe(_ name: String) -> String {
        let invalid = CharacterSet(charactersIn: "/\\:*?\"<>|").union(.controlCharacters)
        var s = String(name.unicodeScalars.filter { !invalid.contains($0) }).trimmingCharacters(in: .whitespaces)
        while let last = s.last, last == "…" || last == "." { s.removeLast() }
        s = s.trimmingCharacters(in: .whitespaces)
        return s.isEmpty ? "design" : s
    }

    /// The element picker, posting to `postMessage` (the web view's message function).
    nonisolated static func pickerScript(_ postMessage: String) -> String { DesignDocument.pickerScript(postMessage) }
}
