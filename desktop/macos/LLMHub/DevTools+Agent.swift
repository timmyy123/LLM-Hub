#if DEBUG
import Foundation

extension DevTools {
    /// `--agenttest <folder> [--model <id>] [--prompt <text>] [--engine claude|openclaw] [--followup <text>] [--session <id>]`:
    /// an end-to-end Agent run on this Mac, results in the log. Installs the engine if missing, allows every permission
    /// request (logging each) and opens the session so a `--capture` shows it. `--session` continues a saved session
    /// (after a relaunch, to check resume) instead of starting one in the folder.
    static func agentTest(_ args: [String]) async {
        guard let folder = value("--agenttest", args) else { return }
        let app = AppModel.shared
        let agent = app.agent
        let engine = value("--engine", args) ?? "claude"
        if engine == "openclaw" && !agent.openClaw.isInstalled {
            let install = Date()
            // Logs each install step and download percentage as it changes.
            let watcher = Task { @MainActor in
                var last = ""
                while !Task.isCancelled {
                    if agent.openClaw.status != last, !agent.openClaw.status.isEmpty {
                        last = agent.openClaw.status
                        Log.info("AGENTTEST \(last)")
                    }
                    try? await Task.sleep(for: .milliseconds(500))
                }
            }
            do {
                try await agent.openClaw.install()
                Log.info("AGENTTEST OpenClaw installed in \(String(format: "%.0f", Date().timeIntervalSince(install)))s")
            } catch {
                Log.error("AGENTTEST OpenClaw install failed", error)
            }
            watcher.cancel()
        } else if engine != "openclaw" && !agent.isClaudeInstalled {
            await agent.installClaude()
            Log.info("AGENTTEST install: \(agent.installStatus) installed=\(agent.isClaudeInstalled)")
        }
        let wanted = value("--model", args)
        guard let model = app.modelStore.installed.first(where: { $0.id == wanted || $0.catalogId == wanted })
                ?? app.modelStore.installed.first(where: { $0.format == "gguf" }) else {
            Log.error("AGENTTEST failed: no model installed")
            return
        }
        let prompt = value("--prompt", args) ?? "average() in stats.py returns the wrong value. Fix it and run the script to check."
        let session: AgentSession
        if let id = value("--session", args), let saved = agent.find(id) {
            session = saved
            Log.info("AGENTTEST resuming \(id) (\(saved.info.engine)) with \(saved.items.count) replayed items")
        } else {
            session = agent.createSession(folder: folder, modelId: model.id, mode: "default", engine: engine)
        }
        Log.info("AGENTTEST session \(session.id) engine=\(session.info.engine) model=\(model.id)")
        agent.ui.selectedId = session.id
        app.section = .agent
        agent.onPermissionRequest = { session, request in
            Log.info("AGENTTEST permission: \(request.question) \(request.summary)")
            Task { @MainActor in agent.answer(session, allow: true) }
        }
        let start = Date()
        await agent.ui.run(session, prompt)
        if let followUp = value("--followup", args) {
            Log.info("AGENTTEST first turn in \(String(format: "%.1f", Date().timeIntervalSince(start)))s status=\(session.status)")
            let second = Date()
            await agent.ui.run(session, followUp)
            Log.info("AGENTTEST follow-up in \(String(format: "%.1f", Date().timeIntervalSince(second)))s status=\(session.status)")
        }
        agent.onPermissionRequest = nil
        for item in session.items {
            let title = item.title.prefix(160).replacingOccurrences(of: "\n", with: " ")
            Log.info("AGENTTEST \(item.kind.rawValue) \(item.detail) \(title) \(item.meta)")
        }
        for task in session.tasks { Log.info("AGENTTEST task [\(task.status)] \(task.text)") }
        for change in session.changes { Log.info("AGENTTEST change \(change.kind) \(change.path) \(change.stats)") }
        if !agent.ui.notice.isEmpty { Log.info("AGENTTEST notice: \(agent.ui.notice)") }
        Log.info("AGENTTEST done in \(String(format: "%.1f", Date().timeIntervalSince(start)))s status=\(session.status)")
    }
}
#endif
