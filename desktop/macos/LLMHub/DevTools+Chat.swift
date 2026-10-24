#if DEBUG
import AppKit

extension DevTools {
    /// `--chattest "<prompt>" [--model <catalog or model id>] [--websearch on|off] [--thinking on|off] [--attach file[,file]]
    /// [--regenerate yes] [--star yes] [--export out.md]`: one chat turn through the real view model, results in the log
    /// (reply, reasoning time, web sources and the [n] citations used, context meter). The chat stays open for --capture.
    static func chatTest(_ args: [String]) async {
        guard value("--chatopen", args) != nil || value("--chattest", args) != nil else { return }
        let app = AppModel.shared
        let chat = app.chat
        // Changing the chat while SwiftUI is still creating the window at launch can leave the app without one;
        // wait for the window first.
        for i in 0..<40 where !NSApp.windows.contains(where: { $0.isVisible && $0.frame.width > 600 }) {
            if i == 8 { NSApp.activate() }
            try? await Task.sleep(for: .milliseconds(250))
        }
        Log.info("CHATTEST window \(NSApp.windows.first { $0.isVisible && $0.frame.width > 600 }.map { "\($0.frame)" } ?? "missing")")
        // `--capture out.png --chatopen <title words> [--reasoning yes] [--attach files]`: opens a saved chat for a screenshot,
        // with the reasoning expanded and files waiting in the composer.
        if let title = value("--chatopen", args), let match = chat.groups.flatMap(\.chats).first(where: { $0.title.localizedCaseInsensitiveContains(title) }) {
            chat.selectedChatId = match.id
            if let reply = chat.messages.last(where: { !$0.isUser && !$0.reasoning.isEmpty }), value("--reasoning", args) == "yes" { reply.reasoningOpen = true }
            Log.info("CHATTEST opened \(match.title): \(chat.messages.count) messages")
            if value("--chattest", args) == nil, let files = value("--attach", args) {
                chat.attachFiles(files.split(separator: ",").map { URL(fileURLWithPath: String($0)) })
            }
        }
        guard let prompt = value("--chattest", args) else { return }
        chat.newChat()
        if let id = value("--model", args) {
            guard let model = chat.models.first(where: { $0.id == id || $0.catalogId == id }) else {
                Log.info("CHATTEST no installed model \(id); have \(chat.models.map(\.id))")
                return
            }
            chat.selectedModelId = model.id
        }
        let webSearch = chat.webSearchEnabled, thinking = chat.thinkingEnabled
        if let search = value("--websearch", args) { chat.webSearchEnabled = search == "on" }
        if let think = value("--thinking", args) { chat.thinkingEnabled = think == "on" }
        if let files = value("--attach", args) {
            chat.attachFiles(files.split(separator: ",").map { URL(fileURLWithPath: String($0)) })
            Log.info("CHATTEST attached \(chat.pendingAttachments.map { "\($0.name) [\($0.badge) \($0.kind) \($0.detail)]" }) error='\(chat.attachError)'")
        }
        Log.info("CHATTEST model=\(chat.selectedModelId ?? "-") web=\(chat.webSearchEnabled) thinking=\(chat.thinkingEnabled)")

        func run(_ start: () -> Void) async {
            let clock = Date()
            start()
            while chat.isGenerating { try? await Task.sleep(for: .milliseconds(250)) }
            guard let reply = chat.messages.last, !reply.isUser else { return }
            let cited = Set(reply.content.matches(of: /\[(\d{1,2})\]/).compactMap { Int($0.1) }).sorted()
            Log.info("CHATTEST reply in \(String(format: "%.1f", Date().timeIntervalSince(clock)))s status='\(reply.status)' activity='\(reply.activity)' "
                + "thinking='\(reply.reasoning.isEmpty ? "-" : reply.thinkingLabel)' (\(reply.reasoning.count) chars) context='\(chat.contextText)' footer='\(reply.footer)'")
            for s in reply.sources { Log.info("CHATTEST source [\(s.number)] \(s.host) \(s.title)") }
            Log.info("CHATTEST citations \(cited) of \(reply.sources.count) sources; all valid=\(cited.allSatisfy { $0 >= 1 && $0 <= reply.sources.count })")
            Log.info("CHATTEST content: \(reply.content.replacingOccurrences(of: "\n", with: " ⏎ "))")
        }

        chat.draft = prompt
        await run { chat.send() }
        if value("--regenerate", args) == "yes" {
            await run { chat.regenerate() }
            Log.info("CHATTEST after regenerate: \(chat.messages.count) messages, stored \(app.chatStore.messages(chat.selectedChatId ?? "").count)")
        }
        if value("--star", args) == "yes" {
            chat.toggleStar()
            Log.info("CHATTEST starred=\(chat.isCurrentStarred) groups=\(chat.groups.map { "\($0.title):\($0.chats.count)" })")
        }
        if let out = value("--export", args) {
            try? chat.exportMarkdown().write(toFile: out, atomically: true, encoding: .utf8)
            Log.info("CHATTEST exported \(chat.exportFileName()) to \(out)")
        }
        // Leave the user's composer switches as they were.
        chat.webSearchEnabled = webSearch
        chat.thinkingEnabled = thinking
    }
}
#endif
