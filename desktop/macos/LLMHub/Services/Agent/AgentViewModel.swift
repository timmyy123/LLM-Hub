import AppKit
import Foundation
import Observation

/// The Agent tab: the open session, its composer, and the side panel (Diff, Terminal, Files, Preview).
/// Same behavior as core ViewModels/AgentViewModel.cs.
@MainActor @Observable
final class AgentViewModel {
    struct Mode: Hashable {
        let mode: String
        let label: String
    }

    struct Group: Identifiable {
        var id: String { folder }
        let folder: String
        let name: String
        let sessions: [AgentSession]
    }

    enum SideTab: String, CaseIterable { case diff, terminal, files, preview }

    @ObservationIgnored private unowned let service: AgentService
    @ObservationIgnored private unowned let app: AppModel

    let modes = [Mode(mode: "default", label: L.s("AgentModeAsk")), Mode(mode: "acceptEdits", label: L.s("AgentModeAuto")),
                 Mode(mode: "plan", label: L.s("AgentModePlan"))]
    /// The agent engines: Claude Code, or OpenClaw (installed by the app, Chinese-origin extensions removed).
    let engines = [Mode(mode: "claude", label: "Claude Code"), Mode(mode: "openclaw", label: "OpenClaw")]
    /// The engine new sessions use; a session keeps the engine it started with.
    var engineChoice = AppSettings.current.agentEngine {
        didSet {
            guard engineChoice != AppSettings.current.agentEngine else { return }
            AppSettings.current.agentEngine = engineChoice
            AppSettings.save()
        }
    }
    var selectedId: String? { didSet { if selectedId != oldValue { onSelected() } } }
    var draft = ""
    var mode = "default" { didSet { selected?.info.mode = mode } }
    var selectedModelId: String? { didSet { onModelChanged() } }
    var selectedChangePath: String?
    var sideTab = SideTab.diff
    var commitMessage = ""
    /// Errors and results (commit output) shown above the composer.
    var notice = ""
    private(set) var files: [String] = []

    init(service: AgentService, app: AppModel) {
        self.service = service
        self.app = app
        let models = app.modelStore.chatModels
        selectedModelId = (models.first { $0.id == AppSettings.current.agentModelId }
            ?? models.first { $0.catalogId != nil && $0.catalogId == app.catalog.defaults["agent"] }
            ?? models.first { $0.format == "gguf" })?.id
        selectedId = service.sessions.first?.id
        onSelected()
    }

    var selected: AgentSession? { selectedId.flatMap { service.find($0) } }
    var models: [InstalledModel] { app.modelStore.chatModels }
    var selectedChange: FileChange? {
        guard let selected else { return nil }
        return selected.changes.first { $0.path == selectedChangePath } ?? selected.changes.first
    }

    /// Sessions by project folder, for the sidebar.
    var groups: [Group] {
        var order: [String] = []
        var byFolder: [String: [AgentSession]] = [:]
        for session in service.sessions {
            if byFolder[session.folder] == nil { order.append(session.folder) }
            byFolder[session.folder, default: []].append(session)
        }
        return order.map { Group(folder: $0, name: URL(fileURLWithPath: $0).lastPathComponent, sessions: byFolder[$0]!) }
    }

    /// "via llama.cpp · Gemma 4 12B" for the engine line.
    var engineLine: String {
        guard let model = models.first(where: { $0.id == selectedModelId }) else { return L.s("AgentNoModel") }
        return "via \(model.format == "litertlm" ? "LiteRT-LM" : "llama.cpp") · \(model.name)"
    }

    var isEngineInstalled: Bool { engineChoice == "openclaw" ? service.openClaw.isInstalled : service.isClaudeInstalled }
    var isEngineInstalling: Bool { engineChoice == "openclaw" ? service.openClaw.isInstalling : service.isInstalling }
    var installStatus: String { engineChoice == "openclaw" ? service.openClaw.status : service.installStatus }
    /// What setting up the chosen engine does, on the install card.
    var installHelp: String { L.s(engineChoice == "openclaw" ? "OpenClawInstallHelp" : "AgentInstallHelp") }

    private func onSelected() {
        notice = ""
        guard let session = selected else { return }
        mode = session.info.mode
        if models.contains(where: { $0.id == session.info.modelId }) { selectedModelId = session.info.modelId }
        service.refreshChanges(session)
        selectedChangePath = session.changes.first?.path
        loadFiles()
    }

    private func onModelChanged() {
        guard let id = selectedModelId else { return }
        if AppSettings.current.agentModelId != id {
            AppSettings.current.agentModelId = id
            AppSettings.save()
        }
        selected?.info.modelId = id
    }

    /// Opens the folder picker and starts a session there (also used by the sidebar's New session button).
    func pickFolderAndStart() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.canCreateDirectories = true
        panel.allowsMultipleSelection = false
        panel.prompt = L.s("AgentPickFolder")
        guard panel.runModal() == .OK, let url = panel.url else { return }
        newSession(url.path)
    }

    /// Starts a session in `folder`.
    @discardableResult
    func newSession(_ folder: String) -> AgentSession? {
        guard let modelId = selectedModelId else {
            notice = L.s("AgentNoModel")
            return nil
        }
        let session = service.createSession(folder: folder, modelId: modelId, mode: mode, engine: engineChoice)
        selectedId = session.id
        return session
    }

    func send() {
        let text = draft.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let session = selected, !text.isEmpty, !session.isWorking else { return }
        let openClaw = session.info.engine == "openclaw"
        guard openClaw ? service.openClaw.isInstalled : service.isClaudeInstalled else {
            notice = L.s(openClaw ? "OpenClawNotInstalled" : "AgentNotInstalled")
            return
        }
        draft = ""
        notice = ""
        Task { await run(session, text) }
    }

    /// One turn, start to finish (the composer's Send and `--agenttest`).
    func run(_ session: AgentSession, _ text: String) async {
        do {
            try await service.send(session, text)
        } catch {
            Log.error("Agent send failed", error)
            notice = L.f("ErrorPrefix", error.localizedDescription)
        }
        if selectedChangePath == nil || !session.changes.contains(where: { $0.path == selectedChangePath }) {
            selectedChangePath = session.changes.first?.path
        }
        if session === selected { loadFiles() }
    }

    func stop() { selected.map(service.stop) }
    func allow() { selected.map { service.answer($0, allow: true) } }
    func allowForSession() { selected.map { service.answer($0, allow: true, forSession: true) } }
    func deny() { selected.map { service.answer($0, allow: false) } }

    func revert() {
        guard let session = selected else { return }
        service.revert(session)
        selectedChangePath = session.changes.first?.path
        loadFiles()
    }

    func commit() {
        guard let session = selected else { return }
        let message = commitMessage.trimmingCharacters(in: .whitespacesAndNewlines)
        let output = service.commit(session, message: message.isEmpty ? session.title : message)
        notice = output.split(separator: "\n").first.map { $0.trimmingCharacters(in: .whitespaces) } ?? ""
        commitMessage = ""
        selectedChangePath = session.changes.first?.path
    }

    func delete(_ session: AgentSession) {
        service.delete(session)
        if selectedId == session.id { selectedId = service.sessions.first?.id }
    }

    /// Installs the engine chosen on the empty-state card.
    func installEngine() async {
        guard engineChoice == "openclaw" else {
            await service.installClaude()
            return
        }
        do {
            try await service.openClaw.install()
        } catch {
            notice = L.f("ErrorPrefix", error.localizedDescription)
        }
    }

    /// Project files for the Files tab (skipping dependency and build folders).
    func loadFiles() {
        guard let session = selected, FileManager.default.fileExists(atPath: session.folder) else {
            files = []
            return
        }
        let skip: Set<String> = [".git", "node_modules", "bin", "obj", "build", "dist", ".venv", "__pycache__", ".next"]
        let root = URL(fileURLWithPath: session.folder, isDirectory: true)
        var result: [String] = []
        var queue = [root]
        while !queue.isEmpty && result.count < 400 {
            let dir = queue.removeFirst()
            let entries = ((try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: [.isDirectoryKey])) ?? [])
                .sorted { $0.lastPathComponent < $1.lastPathComponent }
            for entry in entries where (try? entry.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true {
                if !skip.contains(entry.lastPathComponent) { queue.append(entry) }
            }
            for entry in entries where (try? entry.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) != true {
                result.append(String(entry.path.dropFirst(root.path.count + 1)))
            }
        }
        files = result
    }

    /// index.html in the project, for Preview, or nil.
    var previewFile: String? {
        guard let session = selected else { return nil }
        return ["index.html", "public/index.html", "dist/index.html"].map { (session.folder as NSString).appendingPathComponent($0) }
            .first { FileManager.default.fileExists(atPath: $0) }
    }

    func open(file: String) {
        guard let session = selected else { return }
        NSWorkspace.shared.open(URL(fileURLWithPath: (session.folder as NSString).appendingPathComponent(file)))
    }
}
