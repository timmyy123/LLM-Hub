import Foundation
import Observation

enum AgentStatus { case idle, working, needsYou, error }

enum AgentItemKind: String { case user, assistant, read, edit, write, command, search, tasks, tool, error }

/// One line of a diff for display: "+" added, "-" removed, " " context, "@" hunk header.
struct DiffLine: Hashable {
    let kind: Character
    let text: String
}

/// One step in the session transcript: a message, or a tool call with what it touched.
@MainActor @Observable
final class AgentItem: Identifiable {
    let id = UUID()
    let kind: AgentItemKind
    /// Tool use id, to attach the tool's result later.
    let toolUseId: String
    /// Message text, or the file path / command / pattern of a tool call.
    var title: String
    /// "Read", "Edit", "Run"…, shown as the step's label.
    var detail: String
    /// "118 lines", "+24 −2".
    var meta = ""
    /// Command output or tool result text.
    var output = ""
    var failed = false
    let diff: [DiffLine]

    init(_ kind: AgentItemKind, _ title: String, detail: String = "", toolUseId: String = "", diff: [DiffLine] = []) {
        self.kind = kind
        self.title = title
        self.detail = detail
        self.toolUseId = toolUseId
        self.diff = diff
    }

    /// A tool step shown as a card (Read, Edit, Write, Search, Run, other tools).
    var isStep: Bool { [.read, .edit, .write, .search, .tool, .command].contains(kind) }
}

struct AgentTask: Identifiable, Hashable {
    let id = UUID()
    let text: String
    /// pending, in_progress or completed.
    let status: String
}

/// A file the agent changed in this session, for the Diff panel.
struct FileChange: Identifiable, Hashable {
    var id: String { path }
    let path: String
    /// "M", "A" or "D".
    let kind: Character
    let added: Int
    let removed: Int
    let diff: [DiffLine]

    var stats: String { kind == "A" ? "+\(added)" : "+\(added) −\(removed)" }
}

/// A tool call waiting for the user's Allow / Allow for this session / Deny.
@MainActor
final class PermissionRequest: Identifiable {
    let id = UUID()
    let tool: String
    /// "npm test -- forecast", or the file about to be edited.
    let summary: String
    let question: String
    private var result: Bool?
    private var waiters: [CheckedContinuation<Bool, Never>] = []

    init(tool: String, summary: String, question: String) {
        self.tool = tool
        self.summary = summary
        self.question = question
    }

    /// Resolves the request once; later answers are ignored.
    func resolve(_ allow: Bool) {
        guard result == nil else { return }
        result = allow
        waiters.forEach { $0.resume(returning: allow) }
        waiters = []
    }

    var decision: Bool {
        get async {
            if let result { return result }
            return await withCheckedContinuation { waiters.append($0) }
        }
    }
}

/// Saved per session in agent/sessions.json; the transcript itself is the raw event log next to it.
struct AgentSessionInfo: Codable {
    var id = UUID().uuidString.lowercased()
    var title = ""
    var folder = ""
    var modelId = ""
    /// default (ask before edits and commands), acceptEdits, or plan.
    var mode = "default"
    /// "claude" (Claude Code) or "openclaw"; sessions keep the engine they started with.
    var engine = "claude"
    var started = false
    var updatedAt = Date()
    var allowedTools: [String] = []
}

extension AgentSessionInfo {
    /// Missing keys keep their defaults, so sessions saved before a field existed still load.
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decodeIfPresent(String.self, forKey: .id) ?? UUID().uuidString.lowercased()
        title = try c.decodeIfPresent(String.self, forKey: .title) ?? ""
        folder = try c.decodeIfPresent(String.self, forKey: .folder) ?? ""
        modelId = try c.decodeIfPresent(String.self, forKey: .modelId) ?? ""
        mode = try c.decodeIfPresent(String.self, forKey: .mode) ?? "default"
        engine = try c.decodeIfPresent(String.self, forKey: .engine) ?? "claude"
        started = try c.decodeIfPresent(Bool.self, forKey: .started) ?? false
        updatedAt = try c.decodeIfPresent(Date.self, forKey: .updatedAt) ?? Date()
        allowedTools = try c.decodeIfPresent([String].self, forKey: .allowedTools) ?? []
    }
}

@MainActor @Observable
final class AgentSession: Identifiable {
    var info: AgentSessionInfo
    var title: String
    var status = AgentStatus.idle
    var pending: PermissionRequest?
    /// Commands the agent ran and their output, for the Terminal panel.
    var terminal = ""
    /// Current git branch of the folder, or "".
    var branch = ""
    var items: [AgentItem] = []
    var tasks: [AgentTask] = []
    var changes: [FileChange] = []

    let id: String

    init(_ info: AgentSessionInfo) {
        self.info = info
        id = info.id
        title = info.title
    }

    var folder: String { info.folder }
    var folderName: String { URL(fileURLWithPath: info.folder).lastPathComponent }
    var isWorking: Bool { status == .working || status == .needsYou }

    /// "Working", "Needs you" in the session list.
    var statusText: String {
        switch status {
        case .working: return L.s("AgentWorking")
        case .needsYou: return L.s("AgentNeedsYou")
        case .error: return L.s("AgentError")
        case .idle: return ""
        }
    }
}
