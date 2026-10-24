import AppKit
import Foundation
import Observation

/// The Agent tab's engine: Claude Code (installed by the user, or from the Agent tab) driven headless, one process per
/// turn, talking to a local model through LLM Hub's private Agent bridge. Nothing leaves the computer.
/// Each turn: `claude -p --input-format stream-json --output-format stream-json --verbose` with `--session-id` the first
/// time and `--resume` after, in the session's project folder. Same behavior as core Services/Agent/AgentService.cs.
@MainActor @Observable
final class AgentService {
    static let permissionTool = "approve"

    private(set) var sessions: [AgentSession] = []
    private(set) var installStatus = ""
    private(set) var isInstalling = false
    /// Bumped after an install so views re-check `isClaudeInstalled`.
    private(set) var installGeneration = 0

    @ObservationIgnored private weak var app: AppModel?
    @ObservationIgnored private var running: [String: Process] = [:]
    @ObservationIgnored private var toolItems: [String: AgentItem] = [:]
    /// The private Local API instance Claude Code talks to (own port and key, loopback only, always on).
    @ObservationIgnored private var bridge: LocalApiServer?
    /// The Agent tab's selection, draft, mode and side panel.
    @ObservationIgnored private(set) var ui: AgentViewModel!
    /// Debug runs (`--agenttest`) answer permission requests through this.
    @ObservationIgnored var onPermissionRequest: ((AgentSession, PermissionRequest) -> Void)?

    func attach(_ app: AppModel) {
        self.app = app
        for info in Self.loadInfos() {
            let session = AgentSession(info)
            replay(session)
            sessions.append(session)
        }
        ui = AgentViewModel(service: self, app: app)
        NotificationCenter.default.addObserver(forName: NSApplication.willTerminateNotification, object: nil, queue: .main) { [weak self] _ in
            MainActor.assumeIsolated {
                for process in Array(self?.running.values ?? [String: Process]().values) where process.isRunning { Self.kill(process) }
            }
        }
    }

    private static var root: URL { AppPaths.agent }
    private static var sessionsFile: URL { root.appendingPathComponent("sessions.json") }
    private static func eventLog(_ id: String) -> URL { root.appendingPathComponent(id + ".jsonl") }

    func find(_ id: String) -> AgentSession? { sessions.first { $0.id == id } }

    // MARK: Claude Code

    /// The claude executable: PATH, the native installer's ~/.local/bin, Homebrew.
    static func findClaude() -> String? {
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        let path = (ProcessInfo.processInfo.environment["PATH"] ?? "").split(separator: ":").map { "\($0)/claude" }
        let candidates = path + ["\(home)/.local/bin/claude", "/opt/homebrew/bin/claude", "/usr/local/bin/claude"]
        return candidates.first { FileManager.default.isExecutableFile(atPath: $0) }
    }

    var isClaudeInstalled: Bool {
        _ = installGeneration
        return Self.findClaude() != nil
    }

    /// Runs Anthropic's official native installer (no Node needed).
    func installClaude() async {
        guard !isInstalling else { return }
        isInstalling = true
        installStatus = L.s("AgentInstalling")
        defer {
            isInstalling = false
            installGeneration += 1
        }
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/bash")
        process.arguments = ["-c", "curl -fsSL https://claude.ai/install.sh | bash"]
        process.environment = Self.childEnvironment()
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        do {
            let (status, output) = try await Self.runToEnd(process, pipe)
            Log.info("Claude Code installer exited \(status): \(output.suffix(400))")
            installStatus = Self.findClaude() != nil ? "" : L.f("ErrorPrefix", "installer exit \(status)")
        } catch {
            Log.error("Claude Code install failed", error)
            installStatus = L.f("ErrorPrefix", error.localizedDescription)
        }
    }

    private static func runToEnd(_ process: Process, _ pipe: Pipe) async throws -> (Int32, String) {
        try process.run()
        let data = await Task.detached { pipe.fileHandleForReading.readDataToEndOfFile() }.value
        while process.isRunning { try? await Task.sleep(for: .milliseconds(50)) }
        return (process.terminationStatus, String(decoding: data, as: UTF8.self))
    }

    /// The app's environment without anything inherited from another Claude Code or an Anthropic account, with the
    /// usual tool folders on PATH (apps started from Finder get only /usr/bin:/bin).
    private static func childEnvironment() -> [String: String] {
        var env = ProcessInfo.processInfo.environment.filter { key, _ in
            !key.hasPrefix("CLAUDE") && !key.hasPrefix("ANTHROPIC") && key != "AI_AGENT"
        }
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        var path = (env["PATH"] ?? "/usr/bin:/bin:/usr/sbin:/sbin").split(separator: ":").map(String.init)
        for extra in ["\(home)/.local/bin", "/opt/homebrew/bin", "/usr/local/bin"] where !path.contains(extra) { path.append(extra) }
        env["PATH"] = path.joined(separator: ":")
        return env
    }

    private func ensureBridge() async throws -> LocalApiServer {
        if bridge == nil, let app { bridge = LocalApiServer(app: app, agentBridge: true) }
        guard let bridge else { throw Self.error("No app") }
        try await bridge.ensureBridge()
        return bridge
    }

    // MARK: Sessions

    func createSession(folder: String, modelId: String, mode: String) -> AgentSession {
        let session = AgentSession(AgentSessionInfo(title: L.s("NewSession"), folder: folder, modelId: modelId, mode: mode))
        sessions.insert(session, at: 0)
        updateBranch(session)
        saveInfos()
        return session
    }

    func delete(_ session: AgentSession) {
        stop(session)
        sessions.removeAll { $0 === session }
        try? FileManager.default.removeItem(at: Self.eventLog(session.id))
        try? FileManager.default.removeItem(at: Self.root.appendingPathComponent(session.id + ".mcp.json"))
        saveInfos()
    }

    /// Sends one message and streams the agent's work into the session until it finishes or is stopped.
    func send(_ session: AgentSession, _ text: String) async throws {
        let text = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !session.isWorking, !text.isEmpty else { return }
        guard let claude = Self.findClaude() else { throw Self.error(L.s("AgentNotInstalled")) }
        guard let app, let model = app.modelStore.installed.first(where: { $0.id == session.info.modelId }) else {
            throw Self.error(L.s("AgentNoModel"))
        }
        let bridge = try await ensureBridge()

        if !session.info.started {
            let title = text.count <= 60 ? text : String(text.prefix(57)) + "…"
            session.title = title
            session.info.title = title
        }
        append(session, ["type": "llmhub_user", "text": text])
        session.items.append(AgentItem(.user, text))
        session.status = .working

        let mcpConfig = Self.root.appendingPathComponent(session.id + ".mcp.json")
        let config: [String: Any] = ["mcpServers": ["llmhub": [
            "type": "http",
            "url": "\(bridge.baseURL)/mcp?session=\(session.id)",
            "headers": ["Authorization": "Bearer \(bridge.key)"],
        ]]]
        try JSONSerialization.data(withJSONObject: config).write(to: mcpConfig)

        let apiModel = bridge.apiName(model)
        let process = Process()
        process.executableURL = URL(fileURLWithPath: claude)
        process.currentDirectoryURL = URL(fileURLWithPath: session.folder, isDirectory: true)
        process.arguments = [
            "-p", "--input-format", "stream-json", "--output-format", "stream-json", "--verbose",
            "--model", apiModel, "--permission-mode", session.info.mode,
            "--mcp-config", mcpConfig.path, "--permission-prompt-tool", "mcp__llmhub__\(Self.permissionTool)",
            session.info.started ? "--resume" : "--session-id", session.id,
        ]
        var env = Self.childEnvironment()
        env["ANTHROPIC_BASE_URL"] = bridge.baseURL
        env["ANTHROPIC_AUTH_TOKEN"] = bridge.key
        env["ANTHROPIC_API_KEY"] = ""
        for tier in ["OPUS", "SONNET", "HAIKU"] { env["ANTHROPIC_DEFAULT_\(tier)_MODEL"] = apiModel }
        env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
        env["CLAUDE_CODE_ENABLE_TODO_TOOLS"] = "1"
        env["DISABLE_AUTOUPDATER"] = "1"
        env["DISABLE_TELEMETRY"] = "1"
        process.environment = env
        let stdin = Pipe(), stdout = Pipe(), stderr = Pipe()
        process.standardInput = stdin
        process.standardOutput = stdout
        process.standardError = stderr

        do {
            try process.run()
        } catch {
            fail(session, error.localizedDescription)
            return
        }
        running[session.id] = process
        session.info.started = true
        session.info.updatedAt = Date()
        saveInfos()
        Log.info("Agent turn started: claude \(apiModel) in \(session.folder) (\(session.info.mode))")
        Task.detached {
            let data = stderr.fileHandleForReading.readDataToEndOfFile()
            let text = String(decoding: data, as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines)
            if !text.isEmpty { Log.info("claude stderr: \(text.prefix(600))") }
        }

        let message: [String: Any] = ["type": "user", "message": ["role": "user", "content": text]]
        if let data = try? JSONSerialization.data(withJSONObject: message) {
            try? stdin.fileHandleForWriting.write(contentsOf: data + Data("\n".utf8))
        }
        try? stdin.fileHandleForWriting.close()

        do {
            for try await line in stdout.fileHandleForReading.bytes.lines {
                guard line.first == "{", let event = Self.parse(line) else { continue }
                if event["type"] as? String == "stream_event" || event["subtype"] as? String == "thinking_tokens" { continue }
                appendLine(session, line)
                apply(session, event)
            }
            while process.isRunning { try? await Task.sleep(for: .milliseconds(50)) }
        } catch {
            Log.error("Agent stream failed", error)
            fail(session, error.localizedDescription)
        }
        running[session.id] = nil
        if session.status != .error { session.status = .idle }
        session.pending?.resolve(false)
        session.pending = nil
        refreshChanges(session)
        session.info.updatedAt = Date()
        saveInfos()
        Log.info("Agent turn ended: exit \(process.terminationStatus)")
    }

    func stop(_ session: AgentSession) {
        session.pending?.resolve(false)
        if let process = running[session.id], process.isRunning { Self.kill(process) }
    }

    /// Stops claude and the commands it started.
    private static func kill(_ process: Process) {
        let children = Process()
        children.executableURL = URL(fileURLWithPath: "/usr/bin/pkill")
        children.arguments = ["-TERM", "-P", "\(process.processIdentifier)"]
        try? children.run()
        children.waitUntilExit()
        process.terminate()
    }

    /// Called by the MCP permission tool; resolves when the user answers (or the turn is stopped).
    func requestPermission(sessionId: String, tool: String, input: [String: Any]) async -> Bool {
        guard let session = find(sessionId) else { return false }
        if session.info.allowedTools.contains(tool) { return true }
        let (summary, question) = Self.describe(tool, input)
        let request = PermissionRequest(tool: tool, summary: summary, question: question)
        session.pending = request
        session.status = .needsYou
        onPermissionRequest?(session, request)
        let allowed = await request.decision
        if session.pending === request { session.pending = nil }
        if session.isWorking { session.status = .working }
        return allowed
    }

    /// The user's answer to `session.pending`.
    func answer(_ session: AgentSession, allow: Bool, forSession: Bool = false) {
        guard let pending = session.pending else { return }
        if allow && forSession && !session.info.allowedTools.contains(pending.tool) {
            session.info.allowedTools.append(pending.tool)
            saveInfos()
        }
        pending.resolve(allow)
    }

    private static func describe(_ tool: String, _ input: [String: Any]) -> (String, String) {
        switch tool {
        case "Bash": return (input["command"] as? String ?? "", L.s("AgentRunCommand"))
        case "Edit", "MultiEdit", "Write", "NotebookEdit":
            return (input["file_path"] as? String ?? input["notebook_path"] as? String ?? "", L.s("AgentEditFile"))
        case "WebFetch": return (input["url"] as? String ?? "", L.s("AgentFetch"))
        default:
            let json = jsonString(input)
            return (json.count > 300 ? String(json.prefix(300)) : json, L.f("AgentUseTool", tool))
        }
    }

    // MARK: Events → transcript

    private func apply(_ session: AgentSession, _ e: [String: Any]) {
        switch e["type"] as? String {
        case "llmhub_user":
            session.items.append(AgentItem(.user, e["text"] as? String ?? ""))
        case "assistant":
            let content = (e["message"] as? [String: Any])?["content"] as? [[String: Any]] ?? []
            for block in content {
                if block["type"] as? String == "text", let text = block["text"] as? String,
                   !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    session.items.append(AgentItem(.assistant, text.trimmingCharacters(in: .whitespacesAndNewlines)))
                } else if block["type"] as? String == "tool_use" {
                    addToolUse(session, block)
                }
            }
        case "user":
            let content = (e["message"] as? [String: Any])?["content"] as? [[String: Any]] ?? []
            for block in content where block["type"] as? String == "tool_result" {
                let id = block["tool_use_id"] as? String ?? ""
                let text: String
                if let s = block["content"] as? String {
                    text = s
                } else {
                    text = (block["content"] as? [[String: Any]] ?? []).compactMap { $0["text"] as? String }.joined()
                }
                guard let item = toolItems[id] else { continue }
                item.failed = block["is_error"] as? Bool == true
                if item.kind == .command {
                    item.output = text.count > 4000 ? String(text.prefix(4000)) + "\n…" : text
                    session.terminal += item.output + "\n"
                } else if item.kind == .read && !item.failed {
                    item.meta = L.f("AgentLines", "\(text.filter { $0 == "\n" }.count)")
                } else if item.failed {
                    item.output = text.count > 600 ? String(text.prefix(600)) : text
                }
            }
        case "result":
            if e["is_error"] as? Bool == true, let error = e["result"] as? String { fail(session, error) }
        default:
            break
        }
    }

    private func addToolUse(_ session: AgentSession, _ block: [String: Any]) {
        let name = block["name"] as? String ?? ""
        let input = block["input"] as? [String: Any] ?? [:]
        let id = block["id"] as? String ?? ""
        func str(_ key: String) -> String { input[key] as? String ?? "" }
        let item: AgentItem
        switch name {
        case "Read":
            item = AgentItem(.read, relative(session, str("file_path")), detail: L.s("AgentRead"), toolUseId: id)
        case "Edit", "MultiEdit":
            let edits: [(String, String)] = name == "MultiEdit"
                ? (input["edits"] as? [[String: Any]] ?? []).map { ($0["old_string"] as? String ?? "", $0["new_string"] as? String ?? "") }
                : [(str("old_string"), str("new_string"))]
            let diff = edits.flatMap { Self.lineDiff($0.0, $0.1) }
            item = AgentItem(.edit, relative(session, str("file_path")), detail: L.s("AgentEdit"), toolUseId: id, diff: diff)
            item.meta = "+\(diff.filter { $0.kind == "+" }.count) −\(diff.filter { $0.kind == "-" }.count)"
        case "Write":
            let lines = str("content").components(separatedBy: "\n").map { DiffLine(kind: "+", text: $0) }
            item = AgentItem(.write, relative(session, str("file_path")), detail: L.s("AgentWrite"), toolUseId: id, diff: lines)
            item.meta = "+\(lines.count)"
        case "Bash":
            item = AgentItem(.command, str("command"), detail: L.s("AgentRun"), toolUseId: id)
            session.terminal += "$ " + str("command") + "\n"
        case "Grep", "Glob", "LS":
            item = AgentItem(.search, str("pattern").isEmpty ? relative(session, str("path")) : str("pattern"), detail: name, toolUseId: id)
        case "TodoWrite":
            session.tasks = (input["todos"] as? [[String: Any]] ?? []).map {
                AgentTask(text: $0["content"] as? String ?? "", status: $0["status"] as? String ?? "pending")
            }
            // The Tasks card sits where the agent first planned, as in the design.
            if !session.items.contains(where: { $0.kind == .tasks }) {
                session.items.append(AgentItem(.tasks, L.s("AgentTasks")))
            }
            return
        default:
            let json = Self.jsonString(input)
            item = AgentItem(.tool, json.count > 160 ? String(json.prefix(160)) + "…" : json, detail: name, toolUseId: id)
        }
        toolItems[id] = item
        session.items.append(item)
    }

    private func relative(_ session: AgentSession, _ path: String) -> String {
        guard !path.isEmpty else { return path }
        let folder = session.folder.hasSuffix("/") ? session.folder : session.folder + "/"
        let resolved = (folder as NSString).resolvingSymlinksInPath + "/"
        let candidates = [folder, resolved, "/private" + folder, resolved.hasPrefix("/private/") ? String(resolved.dropFirst(8)) : resolved]
        for prefix in candidates where path.hasPrefix(prefix) { return String(path.dropFirst(prefix.count)) }
        return path
    }

    /// Removed lines, then added lines, keeping lines common to both as context.
    static func lineDiff(_ oldText: String, _ newText: String) -> [DiffLine] {
        let a = oldText.components(separatedBy: "\n")
        let b = newText.components(separatedBy: "\n")
        var prefix = 0
        while prefix < a.count && prefix < b.count && a[prefix] == b[prefix] { prefix += 1 }
        var suffix = 0
        while suffix < a.count - prefix && suffix < b.count - prefix && a[a.count - 1 - suffix] == b[b.count - 1 - suffix] { suffix += 1 }
        return a[..<prefix].map { DiffLine(kind: " ", text: $0) }
            + a[prefix..<(a.count - suffix)].map { DiffLine(kind: "-", text: $0) }
            + b[prefix..<(b.count - suffix)].map { DiffLine(kind: "+", text: $0) }
            + a[(a.count - suffix)...].map { DiffLine(kind: " ", text: $0) }
    }

    private func fail(_ session: AgentSession, _ message: String) {
        session.items.append(AgentItem(.error, message))
        session.status = .error
    }

    // MARK: Git: changes, revert, commit

    func refreshChanges(_ session: AgentSession) {
        updateBranch(session)
        guard isGitRepo(session) else {
            // Not a git repository: show what the agent's edits in this session did.
            var order: [String] = []
            var groups: [String: [AgentItem]] = [:]
            for item in session.items where item.kind == .edit || item.kind == .write {
                if groups[item.title] == nil { order.append(item.title) }
                groups[item.title, default: []].append(item)
            }
            session.changes = order.map { path in
                let items = groups[path]!
                let diff = items.flatMap(\.diff)
                return FileChange(path: path, kind: items.contains { $0.kind == .write } ? "A" : "M",
                                  added: diff.filter { $0.kind == "+" }.count, removed: diff.filter { $0.kind == "-" }.count, diff: diff)
            }
            return
        }
        var changes: [FileChange] = []
        for line in Self.git(session.folder, "status", "--porcelain", "--untracked-files=all").split(separator: "\n") where line.count > 3 {
            let code = line.prefix(2).trimmingCharacters(in: .whitespaces)
            let file = line.dropFirst(3).trimmingCharacters(in: .whitespaces).trimmingCharacters(in: CharacterSet(charactersIn: "\""))
            if code == "??" {
                let full = URL(fileURLWithPath: session.folder).appendingPathComponent(file)
                let content = (try? String(contentsOf: full, encoding: .utf8))?.components(separatedBy: "\n") ?? []
                let lines = content.last == "" ? Array(content.dropLast()) : content
                changes.append(FileChange(path: file, kind: "A", added: lines.count, removed: 0,
                                          diff: lines.prefix(400).map { DiffLine(kind: "+", text: $0) }))
                continue
            }
            let diff = Self.parseUnifiedDiff(Self.git(session.folder, "diff", "HEAD", "--", file))
            changes.append(FileChange(path: file, kind: code.contains("D") ? "D" : "M",
                                      added: diff.filter { $0.kind == "+" }.count, removed: diff.filter { $0.kind == "-" }.count, diff: diff))
        }
        session.changes = changes
    }

    func isGitRepo(_ session: AgentSession) -> Bool {
        FileManager.default.fileExists(atPath: session.folder)
            && Self.git(session.folder, "rev-parse", "--is-inside-work-tree").trimmingCharacters(in: .whitespacesAndNewlines) == "true"
    }

    /// Puts every changed file back to the last commit and removes files the agent added.
    func revert(_ session: AgentSession) {
        guard isGitRepo(session) else { return }
        for change in session.changes {
            if change.kind == "A" {
                try? FileManager.default.removeItem(at: URL(fileURLWithPath: session.folder).appendingPathComponent(change.path))
            } else {
                _ = Self.git(session.folder, "checkout", "HEAD", "--", change.path)
            }
        }
        refreshChanges(session)
    }

    func commit(_ session: AgentSession, message: String) -> String {
        guard isGitRepo(session), !session.changes.isEmpty else { return "" }
        _ = Self.git(session.folder, ["add", "--"] + session.changes.map(\.path))
        let output = Self.git(session.folder, "commit", "-m", message)
        refreshChanges(session)
        return output
    }

    private func updateBranch(_ session: AgentSession) {
        session.branch = FileManager.default.fileExists(atPath: session.folder)
            ? Self.git(session.folder, "branch", "--show-current").trimmingCharacters(in: .whitespacesAndNewlines) : ""
    }

    private static func parseUnifiedDiff(_ diff: String) -> [DiffLine] {
        var lines: [DiffLine] = []
        for line in diff.components(separatedBy: "\n") {
            if line.hasPrefix("@@") {
                lines.append(DiffLine(kind: "@", text: line))
            } else if line.hasPrefix("+++") || line.hasPrefix("---") || line.hasPrefix("diff ") || line.hasPrefix("index ") {
                continue
            } else if let first = line.first, "+- ".contains(first) {
                lines.append(DiffLine(kind: first, text: String(line.dropFirst())))
            }
        }
        return lines
    }

    private static func git(_ folder: String, _ args: String...) -> String { git(folder, args) }

    private static func git(_ folder: String, _ args: [String]) -> String {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/git")
        process.arguments = args
        process.currentDirectoryURL = URL(fileURLWithPath: folder, isDirectory: true)
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = FileHandle.nullDevice
        do { try process.run() } catch { return "" }
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        return String(decoding: data, as: UTF8.self)
    }

    // MARK: Storage

    private static func loadInfos() -> [AgentSessionInfo] {
        guard let data = try? Data(contentsOf: sessionsFile) else { return [] }
        do {
            let decoder = JSONDecoder()
            decoder.dateDecodingStrategy = .iso8601
            return try decoder.decode([AgentSessionInfo].self, from: data)
        } catch {
            Log.error("Agent sessions could not be read", error)
            return []
        }
    }

    private func saveInfos() {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        do {
            try encoder.encode(sessions.map(\.info)).write(to: Self.sessionsFile, options: .atomic)
        } catch {
            Log.error("Agent sessions could not be saved", error)
        }
    }

    private func append(_ session: AgentSession, _ event: [String: Any]) { appendLine(session, Self.jsonString(event)) }

    private func appendLine(_ session: AgentSession, _ line: String) {
        let url = Self.eventLog(session.id)
        let data = Data((line + "\n").utf8)
        if let handle = try? FileHandle(forWritingTo: url) {
            handle.seekToEndOfFile()
            handle.write(data)
            try? handle.close()
        } else {
            try? data.write(to: url)
        }
    }

    /// Rebuilds a saved session's transcript from its event log.
    private func replay(_ session: AgentSession) {
        guard let text = try? String(contentsOf: Self.eventLog(session.id), encoding: .utf8) else { return }
        for line in text.split(separator: "\n") {
            if let event = Self.parse(String(line)) { apply(session, event) }
        }
        session.status = .idle
    }

    private static func parse(_ line: String) -> [String: Any]? {
        (try? JSONSerialization.jsonObject(with: Data(line.utf8))) as? [String: Any]
    }

    static func jsonString(_ object: Any) -> String {
        guard let data = try? JSONSerialization.data(withJSONObject: object, options: [.withoutEscapingSlashes]) else { return "" }
        return String(decoding: data, as: UTF8.self)
    }

    private static func error(_ message: String) -> NSError {
        NSError(domain: "LLMHub.Agent", code: 1, userInfo: [NSLocalizedDescriptionKey: message])
    }
}
