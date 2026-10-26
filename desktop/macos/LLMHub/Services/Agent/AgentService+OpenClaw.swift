import Foundation

/// OpenClaw sessions: one `openclaw acp` process per session, kept open between turns. Its Agent Client Protocol updates
/// are turned into the same transcript events Claude Code writes (assistant text, tool_use, tool_result), so the tab, the
/// event log and replay work the same for both engines. Same behavior as core Services/Agent/AgentService.OpenClaw.cs.
extension AgentService {
    func sendOpenClaw(_ session: AgentSession, _ text: String) async throws {
        guard openClaw.isInstalled else { throw Self.error(L.s("OpenClawNotInstalled")) }
        guard let app, app.modelStore.chatModels.contains(where: { $0.id == session.info.modelId }) else {
            throw Self.error(L.s("AgentNoModel"))
        }

        if !session.info.started {
            let title = text.count <= 60 ? text : String(text.prefix(57)) + "…"
            session.title = title
            session.info.title = title
        }
        append(session, ["type": "llmhub_user", "text": text])
        session.items.append(AgentItem(.user, text))
        session.status = .working
        session.info.started = true
        session.info.updatedAt = Date()
        saveInfos()

        do {
            let bridge = try await ensureBridge()
            // This session's folder, plus the folders of sessions with an open connection.
            var engaged = sessions.filter { $0 !== session && acp[$0.id]?.isAlive == true }
            engaged.append(session)
            try await openClaw.ensureGateway(engaged, bridge: bridge, models: app.modelStore.chatModels)
            let connection: AcpConnection
            if let open = acp[session.id], open.isAlive {
                connection = open
            } else {
                acp[session.id]?.close()
                connection = try await AcpConnection.open(self, session)
                acp[session.id] = connection
            }
            let stop = try await connection.prompt(text)
            Log.info("OpenClaw turn in \(session.id) ended: \(stop)")
            if stop == "max_tokens" { fail(session, L.s("AgentMaxTokens")) }
        } catch {
            Log.error("OpenClaw turn failed", error)
            fail(session, error.localizedDescription)
        }
        if session.status != .error { session.status = .idle }
        session.pending?.resolve(false)
        session.pending = nil
        refreshChanges(session)
        session.info.updatedAt = Date()
        saveInfos()
    }

    /// Records a translated event and shows it, exactly like a line from Claude Code.
    func emit(_ session: AgentSession, _ event: [String: Any]) {
        appendLine(session, Self.jsonString(event))
        apply(session, event)
    }

    /// OpenClaw's permission request ("exec" and friends) through the same Allow / Allow for this session / Deny card.
    /// Returns the ACP option to select, or nil to cancel.
    func askOpenClaw(_ session: AgentSession, toolCall: [String: Any], options: [[String: Any]]) async -> String? {
        let (name, input) = AcpConnection.translate(toolCall)
        func option(_ kinds: String...) -> String? {
            options.first { kinds.contains($0["kind"] as? String ?? "") }?["optionId"] as? String
        }
        let allowed = await requestPermission(sessionId: session.id, tool: name, input: input)
        guard allowed else { return option("reject_once", "reject_always") }
        return session.info.allowedTools.contains(name) ? option("allow_always", "allow_once") : option("allow_once", "allow_always")
    }
}

/// One `openclaw acp` process and its ACP session, speaking newline-delimited JSON-RPC.
@MainActor
final class AcpConnection {
    private unowned let owner: AgentService
    private let session: AgentSession
    private let process: Process
    private let input: FileHandle
    private var calls: [Int: CheckedContinuation<Any?, Error>] = [:]
    private var next = 0
    private var acpSession = ""
    private var cancelled = false
    /// Set when the process's output ends; later calls fail at once instead of waiting forever.
    private var finished = false
    /// OpenClaw's tool dispatches with no real call yet (and their results), and the dispatch ids whose real call came.
    private var dispatches: [(id: String, call: [String: Any], result: [String: Any]?)] = []
    private var dispatched: Set<String> = []
    /// Text streams as chunks; it is written out as one message when a tool starts or the turn ends.
    private var text = ""

    private init(owner: AgentService, session: AgentSession, process: Process, input: FileHandle) {
        self.owner = owner
        self.session = session
        self.process = process
        self.input = input
    }

    var isAlive: Bool { process.isRunning }

    static func open(_ owner: AgentService, _ session: AgentSession) async throws -> AcpConnection {
        let process = owner.openClaw.acp(folder: session.folder)
        let stdin = Pipe(), stdout = Pipe(), stderr = Pipe()
        process.standardInput = stdin
        process.standardOutput = stdout
        process.standardError = stderr
        // A write after the process died returns EPIPE instead of a SIGPIPE that would end the app.
        _ = fcntl(stdin.fileHandleForWriting.fileDescriptor, F_SETNOSIGPIPE, 1)
        try process.run()
        let connection = AcpConnection(owner: owner, session: session, process: process, input: stdin.fileHandleForWriting)
        Task { await connection.readLoop(stdout.fileHandleForReading) }
        Task.detached {
            let text = String(decoding: stderr.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines)
            if !text.isEmpty { Log.info("openclaw acp stderr: \(text.suffix(800))") }
        }
        _ = try await connection.call("initialize", [
            "protocolVersion": 1,
            "clientCapabilities": ["fs": ["readTextFile": false, "writeTextFile": false], "terminal": false],
        ])
        // The session key names the per-folder agent and this session, so a new process picks up the same history.
        let created = try await connection.call("session/new", [
            "cwd": session.folder,
            "mcpServers": [Any](),
            "_meta": ["sessionKey": "agent:\(OpenClawService.agentId(session)):\(session.id)"],
        ]) as? [String: Any]
        guard let id = created?["sessionId"] as? String else {
            connection.close()
            throw NSError(domain: "LLMHub.OpenClaw", code: 2, userInfo: [NSLocalizedDescriptionKey: "OpenClaw did not open a session"])
        }
        connection.acpSession = id
        return connection
    }

    func prompt(_ text: String) async throws -> String {
        cancelled = false
        let result = try await call("session/prompt", ["sessionId": acpSession, "prompt": [["type": "text", "text": text]]]) as? [String: Any]
        flushText()
        // A dispatch nothing followed: show it as the tool it names, with its arguments (and its result).
        for dispatch in dispatches {
            emitToolUse(dispatch.id, dispatch.call)
            if let output = dispatch.result { owner.emit(session, output) }
        }
        dispatches = []
        dispatched = []
        return result?["stopReason"] as? String ?? "end_turn"
    }

    func cancel() {
        cancelled = true
        send(["jsonrpc": "2.0", "method": "session/cancel", "params": ["sessionId": acpSession]])
    }

    func close() {
        if process.isRunning { AgentService.kill(process) }
    }

    private static var stopped: NSError {
        NSError(domain: "LLMHub.OpenClaw", code: 4, userInfo: [NSLocalizedDescriptionKey: L.s("OpenClawStopped")])
    }

    private func call(_ method: String, _ params: [String: Any]) async throws -> Any? {
        guard !finished else { throw Self.stopped }
        next += 1
        let id = next
        return try await withCheckedThrowingContinuation { continuation in
            calls[id] = continuation
            send(["jsonrpc": "2.0", "id": id, "method": method, "params": params])
        }
    }

    private func send(_ message: [String: Any]) {
        guard process.isRunning, let data = try? JSONSerialization.data(withJSONObject: message, options: [.withoutEscapingSlashes]) else { return }
        try? input.write(contentsOf: data + Data("\n".utf8))
    }

    private func readLoop(_ output: FileHandle) async {
        do {
            for try await line in output.bytes.lines {
                guard line.first == "{", let message = (try? JSONSerialization.jsonObject(with: Data(line.utf8))) as? [String: Any] else { continue }
                let method = message["method"] as? String
                if method == nil, let id = (message["id"] as? NSNumber)?.intValue {
                    guard let call = calls.removeValue(forKey: id) else { continue }
                    if let error = message["error"] as? [String: Any] {
                        let text = error["message"] as? String ?? AgentService.jsonString(error)
                        call.resume(throwing: NSError(domain: "LLMHub.OpenClaw", code: 3, userInfo: [NSLocalizedDescriptionKey: text]))
                    } else {
                        call.resume(returning: message["result"])
                    }
                } else if method == "session/update" {
                    onUpdate((message["params"] as? [String: Any])?["update"] as? [String: Any])
                } else if method == "session/request_permission", let id = message["id"] {
                    let params = message["params"] as? [String: Any]
                    Task { await answerPermission(id, params) }
                } else if let id = message["id"] {
                    send(["jsonrpc": "2.0", "id": id, "error": ["code": -32601, "message": "Method not found"]])
                }
            }
        } catch {
            Log.error("openclaw acp output failed", error)
        }
        finished = true
        for call in calls.values { call.resume(throwing: Self.stopped) }
        calls = [:]
    }

    private func answerPermission(_ id: Any, _ params: [String: Any]?) async {
        flushText()
        let optionId = await owner.askOpenClaw(session, toolCall: params?["toolCall"] as? [String: Any] ?? [:],
                                                options: params?["options"] as? [[String: Any]] ?? [])
        let outcome: [String: Any] = if let optionId, !optionId.isEmpty, !cancelled {
            ["outcome": "selected", "optionId": optionId]
        } else {
            ["outcome": "cancelled"]
        }
        send(["jsonrpc": "2.0", "id": id, "result": ["outcome": outcome]])
    }

    private func onUpdate(_ update: [String: Any]?) {
        guard let update else { return }
        switch update["sessionUpdate"] as? String {
        case "agent_message_chunk":
            if let chunk = (update["content"] as? [String: Any])?["text"] as? String { text += chunk }
        case "tool_call":
            flushText()
            let id = update["toolCallId"] as? String ?? UUID().uuidString
            // OpenClaw announces each tool twice: a dispatch (raw input {args, id: tool}), then the real call with id
            // "tool_call:<dispatch>:<tool>:<n>". A dispatch is shown, at the end of the turn, only if no real call follows.
            if Self.isDispatch(update["rawInput"]) {
                dispatches.append((id, update, nil))
                return
            }
            if id.hasPrefix("tool_call:") {
                let dispatch = String(id.split(separator: ":")[1])
                dispatches.removeAll { $0.id == dispatch }
                dispatched.insert(dispatch)
            }
            emitToolUse(id, update)
        case "tool_call_update":
            let status = update["status"] as? String
            guard status == "completed" || status == "failed" else { return }
            let id = update["toolCallId"] as? String ?? ""
            guard !dispatched.contains(id) else { return }
            var output = (update["content"] as? [[String: Any]] ?? []).map { (($0["content"] as? [String: Any])?["text"] as? String) ?? "" }.joined()
            if output.isEmpty, let raw = update["rawOutput"] as? [String: Any], let content = raw["content"] {
                output = content as? String ?? AgentService.jsonString(content)
            }
            let result: [String: Any] = ["type": "user", "message": ["content": [[
                "type": "tool_result", "tool_use_id": id, "content": output, "is_error": status == "failed",
            ]]]]
            // A pending dispatch's result waits for the dispatch to be shown.
            if let index = dispatches.firstIndex(where: { $0.id == id }) {
                dispatches[index].result = result
            } else {
                owner.emit(session, result)
            }
        default:
            break
        }
    }

    private func emitToolUse(_ id: String, _ call: [String: Any]) {
        let (name, input) = Self.translate(call)
        owner.emit(session, ["type": "assistant", "message": ["content": [["type": "tool_use", "id": id, "name": name, "input": input]]]])
    }

    private func flushText() {
        let message = text.trimmingCharacters(in: .whitespacesAndNewlines)
        text = ""
        guard !message.isEmpty else { return }
        owner.emit(session, ["type": "assistant", "message": ["content": [["type": "text", "text": message]]]])
    }

    /// An ACP tool call as the Claude Code tool it resembles, so it renders as Read, Edit, Write, Run or Search:
    /// OpenClaw's read / write / edit / apply_patch / exec / ls tools by their ACP kind and raw input.
    /// OpenClaw's tool dispatch: raw input with a string `id` (the tool) and nothing but `args` besides.
    static func isDispatch(_ rawInput: Any?) -> Bool {
        guard let raw = rawInput as? [String: Any], raw["id"] is String else { return false }
        return raw.keys.allSatisfy { $0 == "id" || $0 == "args" }
    }

    static func translate(_ call: [String: Any]) -> (String, [String: Any]) {
        let raw = call["rawInput"] as? [String: Any] ?? [:]
        // A dispatch ({id: "exec", args: {…}}) is shown as the tool it dispatches.
        if isDispatch(raw), let tool = raw["id"] as? String {
            var inner = call
            inner["title"] = tool
            inner["rawInput"] = raw["args"] as? [String: Any] ?? [:]
            return translate(inner)
        }
        func str(_ keys: String...) -> String? { keys.lazy.compactMap { raw[$0] as? String }.first }
        let path = str("path", "file_path", "filePath", "file") ?? ((call["locations"] as? [[String: Any]])?.first?["path"] as? String) ?? ""
        let title = call["title"] as? String ?? ""
        let diff = (call["content"] as? [[String: Any]] ?? []).first { $0["type"] as? String == "diff" }
        // OpenClaw's edit and apply_patch take a list of {oldText, newText} replacements: show them as one multi-edit.
        if let edits = raw["edits"] as? [[String: Any]] {
            let list = edits.map { e -> [String: Any] in
                ["old_string": e["oldText"] as? String ?? e["old_string"] as? String ?? "",
                 "new_string": e["newText"] as? String ?? e["new_string"] as? String ?? ""]
            }
            return ("MultiEdit", ["file_path": path, "edits": list])
        }
        let first = String(title.split(whereSeparator: { $0 == ":" || $0 == " " }).first ?? "")
        let tool = first.lowercased()
        if tool == "ls" { return ("LS", ["path": path.isEmpty ? "." : path]) }
        if tool == "read" { return ("Read", ["file_path": path]) }
        if tool == "exec" || tool == "process", let command = str("command", "cmd") { return ("Bash", ["command": command]) }
        switch call["kind"] as? String {
        case "read":
            return ("Read", ["file_path": path])
        case "edit":
            let oldText = str("oldText", "old_string", "oldString") ?? diff?["oldText"] as? String
            if let content = str("content"), str("oldText", "old_string", "oldString") == nil {
                return ("Write", ["file_path": path, "content": content])
            }
            let newText = str("newText", "new_string", "newString") ?? diff?["newText"] as? String
            if oldText != nil || newText != nil {
                return ("Edit", ["file_path": path.isEmpty ? diff?["path"] as? String ?? "" : path, "old_string": oldText ?? "", "new_string": newText ?? ""])
            }
            return ("apply_patch", raw)
        case "execute":
            return ("Bash", ["command": str("command", "cmd") ?? title])
        case "search":
            return ("Grep", ["pattern": str("pattern", "query") ?? title, "path": path])
        default:
            return (first.isEmpty ? "tool" : first, raw)
        }
    }
}
