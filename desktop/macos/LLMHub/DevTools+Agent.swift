#if DEBUG
import Foundation

extension DevTools {
    /// `--agenttest <folder> [--model <id>] [--prompt <text>]`: an end-to-end Agent run on this Mac, results in the log.
    /// Installs Claude Code if missing, allows every permission request (logging each) and opens the session so a
    /// `--capture` shows it.
    static func agentTest(_ args: [String]) async {
        guard let folder = value("--agenttest", args) else { return }
        let app = AppModel.shared
        let agent = app.agent
        if !agent.isClaudeInstalled {
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
        let session = agent.createSession(folder: folder, modelId: model.id, mode: "default")
        agent.ui.selectedId = session.id
        app.section = .agent
        agent.onPermissionRequest = { session, request in
            Log.info("AGENTTEST permission: \(request.question) \(request.summary)")
            Task { @MainActor in agent.answer(session, allow: true) }
        }
        let start = Date()
        await agent.ui.run(session, prompt)
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
