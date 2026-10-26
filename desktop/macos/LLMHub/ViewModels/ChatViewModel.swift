import Foundation
import Observation

@MainActor @Observable
final class MessageItem: Identifiable {
    let id: String
    let role: String
    var content: String
    var reasoning = ""
    var status = ""
    /// Model and generation speed, shown under a finished reply.
    var footer = ""
    /// "Searched the web · 5 results" above a reply that used search.
    var activity = ""
    /// True once the reply is complete, which is when Copy / Read aloud / Regenerate show.
    var isDone = false
    /// Whether the reasoning under the "Thought for…" pill is expanded.
    var reasoningOpen = false
    var attachments: [Attachment] = []
    var sources: [WebSource] = []
    /// How long the model thought before answering, when it was measured (live replies).
    var thinkingSeconds: Double?

    init(role: String, content: String, id: String = UUID().uuidString.lowercased()) {
        self.id = id
        self.role = role
        self.content = content
    }

    var isUser: Bool { role == "user" }
    var showActions: Bool { isDone && !content.isEmpty }

    /// "Thinking" while the model reasons, then "Thought for 6s" (or "a few seconds" for older chats).
    var thinkingLabel: String {
        if !isDone && content.isEmpty { return L.s("Thinking") }
        if let seconds = thinkingSeconds, seconds >= 1 { return L.f("ThoughtForSeconds", "\(Int(seconds.rounded()))") }
        return L.s("ThoughtLabel")
    }
}

struct ChatSummary: Identifiable, Hashable {
    let id: String
    var title: String
    var updatedAt: Date
    var starred = false
}

@MainActor @Observable
final class ChatViewModel {
    private weak var app: AppModel?
    private var all: [ChatSummary] = []
    private var generation: Task<Void, Never>?

    var messages: [MessageItem] = []
    var selectedChatId: String? {
        didSet { if !suppressLoad, oldValue != selectedChatId { loadSelected() } }
    }
    var draft = ""
    var search = ""
    var isGenerating = false
    var selectedModelId: String? {
        didSet {
            AppSettings.current.lastChatModelId = selectedModelId
            AppSettings.save()
        }
    }
    /// Files attached in the composer, sent with the next message.
    var pendingAttachments: [Attachment] = []
    /// Why a file could not be attached, or a speech error, shown above the composer field.
    var attachError = ""
    /// "11k of 32k context" for the open chat, or "" before the first reply.
    var contextText = ""
    var webSearchEnabled = AppSettings.current.webSearch {
        didSet {
            AppSettings.current.webSearch = webSearchEnabled
            AppSettings.save()
        }
    }
    var thinkingEnabled = AppSettings.current.thinking {
        didSet {
            AppSettings.current.thinking = thinkingEnabled
            AppSettings.save()
        }
    }
    private var suppressLoad = false

    func attach(_ app: AppModel) {
        self.app = app
        all = app.chatStore.listChats().map { ChatSummary(id: $0.id, title: $0.title, updatedAt: $0.updatedAt, starred: $0.starred) }
        ensureModelSelected()
    }

    /// Downloaded models, plus Apple Intelligence when this Mac has it.
    var models: [InstalledModel] {
        (AppleModel.unavailableReason == nil ? [AppleModel.installed] : []) + (app?.modelStore.chatModels ?? [])
    }
    var selectedModel: InstalledModel? { models.first { $0.id == selectedModelId } }
    var currentChat: ChatSummary? { all.first { $0.id == selectedChatId } }
    var currentTitle: String { currentChat?.title ?? "" }
    var isCurrentStarred: Bool { currentChat?.starred == true }
    var hasOpenChat: Bool { currentChat != nil }

    /// The MTP toggle shows only for LiteRT-LM models with a drafter; changing it reloads the model on next send.
    var mtpVisible: Bool { selectedModel?.supportsMtp == true }
    var mtpEnabled: Bool {
        get { _ = mtpRevision; return selectedModel?.mtpEnabled == true }
        set {
            guard let model = selectedModel, model.supportsMtp else { return }
            if newValue { AppSettings.current.mtpOff.removeAll { $0 == model.id } } else { AppSettings.current.mtpOff.append(model.id) }
            AppSettings.save()
            mtpRevision += 1
        }
    }
    private var mtpRevision = 0

    var canSend: Bool {
        !isGenerating && selectedModel != nil
            && (!draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || !pendingAttachments.isEmpty)
    }

    /// Starred chats first, then Today / Yesterday / Previous 7 days / Older, filtered by the search box.
    var groups: [(title: String, chats: [ChatSummary])] {
        let query = search.trimmingCharacters(in: .whitespaces)
        let visible = (query.isEmpty ? all : all.filter { $0.title.localizedCaseInsensitiveContains(query) })
            .sorted { $0.updatedAt > $1.updatedAt }
        let calendar = Calendar.current
        func bucket(_ c: ChatSummary) -> String {
            if calendar.isDateInToday(c.updatedAt) { return L.s("GroupToday") }
            if calendar.isDateInYesterday(c.updatedAt) { return L.s("GroupYesterday") }
            if c.updatedAt > calendar.date(byAdding: .day, value: -7, to: Date())! { return L.s("GroupWeek") }
            return L.s("GroupOlder")
        }
        var result: [(title: String, chats: [ChatSummary])] = []
        let starred = visible.filter(\.starred)
        if !starred.isEmpty { result.append((L.s("GroupStarred"), starred)) }
        var order: [String] = []
        var map: [String: [ChatSummary]] = [:]
        for chat in visible where !chat.starred {
            let key = bucket(chat)
            if map[key] == nil { order.append(key) }
            map[key, default: []].append(chat)
        }
        return result + order.map { ($0, map[$0]!) }
    }

    func ensureModelSelected() {
        if let id = selectedModelId, models.contains(where: { $0.id == id }) { return }
        let defaultId = app?.catalog.defaults["chat"]
        selectedModelId = (models.first { $0.id == AppSettings.current.lastChatModelId }
            ?? models.first { $0.catalogId == defaultId } ?? models.first)?.id
    }

    func newChat() {
        stop()
        suppressLoad = true
        selectedChatId = nil
        suppressLoad = false
        messages = []
        pendingAttachments = []
        contextText = ""
        attachError = ""
    }

    func delete(_ chat: ChatSummary) {
        app?.chatStore.deleteChat(chat.id)
        all.removeAll { $0.id == chat.id }
        if selectedChatId == chat.id { newChat() }
    }

    /// Stars or unstars a chat (the open one when none is given); starred chats get their own sidebar group.
    func toggleStar(_ chat: ChatSummary? = nil) {
        guard let id = chat?.id ?? selectedChatId, let index = all.firstIndex(where: { $0.id == id }) else { return }
        all[index].starred.toggle()
        app?.chatStore.setStarred(id, all[index].starred)
    }

    func stop() { generation?.cancel() }

    private func loadSelected() {
        stop()
        pendingAttachments = []
        contextText = ""
        attachError = ""
        messages = (selectedChatId.map { app?.chatStore.messages($0) ?? [] } ?? []).map { m in
            let item = MessageItem(role: m.role, content: m.content, id: m.id)
            item.reasoning = m.reasoning ?? ""
            item.isDone = true
            item.attachments = m.attachments
            item.sources = m.sources
            if !m.sources.isEmpty { item.activity = L.f("SearchedWeb", "\(m.sources.count)") }
            if !item.isUser, let id = m.modelId {
                item.footer = L.f("OnThisComputer", models.first { $0.id == id }?.name ?? id)
            }
            return item
        }
    }

    // MARK: Attachments

    /// Adds files to the next message. Unsupported files are skipped with a message in `attachError`.
    func attachFiles(_ urls: [URL]) {
        attachError = ""
        for url in urls {
            do {
                pendingAttachments.append(try Attachments.importFile(url))
            } catch {
                Log.error("Attach \(url.path) failed", error)
                attachError = error.localizedDescription
            }
        }
    }

    func removeAttachment(_ attachment: Attachment) {
        pendingAttachments.removeAll { $0 == attachment }
    }

    // MARK: Sending

    /// The composer's mic button: the first call starts recording; the second stops, transcribes with Whisper and adds the
    /// text to the draft. Errors go to the line above the field.
    func toggleDictation() async {
        guard let speech = app?.speech else { return }
        if !speech.isRecording {
            do {
                attachError = ""
                try await speech.startDictation()
            } catch {
                Log.error("Dictation failed", error)
                attachError = L.f("ErrorPrefix", error.localizedDescription)
            }
            return
        }
        let text = await speech.stopDictation()
        if !text.isEmpty {
            draft = draft.isEmpty ? text : draft.trimmingCharacters(in: .whitespaces) + " " + text
        } else if !speech.status.isEmpty {
            attachError = speech.status
        }
    }

    func send() {
        var text = draft.trimmingCharacters(in: .whitespacesAndNewlines)
        guard canSend, let app, let model = selectedModel else { return }
        if text.isEmpty { text = L.s("AttachDefaultPrompt") }
        draft = ""

        let chatId: String
        if let id = selectedChatId {
            chatId = id
        } else {
            let record = app.chatStore.createChat(title: Self.makeTitle(text), modelId: model.id)
            all.insert(ChatSummary(id: record.id, title: record.title, updatedAt: record.updatedAt), at: 0)
            suppressLoad = true
            selectedChatId = record.id
            suppressLoad = false
            chatId = record.id
        }

        let user = MessageItem(role: "user", content: text)
        user.isDone = true
        user.attachments = pendingAttachments
        pendingAttachments = []
        attachError = ""
        messages.append(user)
        app.chatStore.add(MessageRecord(id: user.id, chatId: chatId, role: "user", content: text, reasoning: nil, modelId: nil,
                                        createdAt: Date(), attachments: user.attachments))
        generate(chatId: chatId, model: model, question: text)
    }

    /// Replaces the last reply with a new one.
    func regenerate(_ reply: MessageItem? = nil) {
        guard !isGenerating, let chatId = selectedChatId, let model = selectedModel, let last = messages.last else { return }
        let reply = reply ?? last
        guard !reply.isUser, reply === last else { return }
        messages.removeLast()
        app?.chatStore.deleteMessage(reply.id)
        let question = messages.last { $0.isUser }?.content ?? ""
        generate(chatId: chatId, model: model, question: question)
    }

    private func generate(chatId: String, model: InstalledModel, question: String) {
        guard let app else { return }
        let reply = MessageItem(role: "assistant", content: "")
        messages.append(reply)
        var history = messages.filter { $0 !== reply && (!$0.content.isEmpty || !$0.attachments.isEmpty) }.map {
            ChatTurn(role: $0.role, content: Attachments.withDocuments($0.content, $0.attachments),
                     images: $0.attachments.filter(\.isImage).map(\.url))
        }
        let search = webSearchEnabled && !question.isEmpty
        let options = ChatOptions(thinking: thinkingEnabled)

        isGenerating = true
        generation = Task {
            do {
                if search {
                    reply.status = L.s("SearchingWeb")
                    do {
                        let (context, sources) = try await WebSearch.buildContext(question)
                        reply.sources = sources
                        reply.activity = L.f("SearchedWeb", "\(sources.count)")
                        // First in the conversation: some chat templates (Gemma's) reject a system turn anywhere else.
                        if !context.isEmpty { history.insert(ChatTurn(role: "system", content: context), at: 0) }
                    } catch is CancellationError {
                        throw CancellationError()
                    } catch let error as URLError where error.code == .cancelled {
                        throw CancellationError()
                    } catch {
                        Log.error("Web search failed", error)
                        reply.activity = L.s("WebSearchFailed")
                    }
                }
                if !app.engines.isReady(model) {
                    reply.status = L.f("LoadingModel", model.name)
                    try await app.engines.load(model)
                    mtpRevision += 1 // The first load learns whether the file has an MTP drafter.
                }
                reply.status = ""
                var splitter = ThinkSplitter()
                var thinkStart: Date?
                for try await delta in app.engines.stream(model, turns: history, options: options) {
                    var thought = delta.reasoning ?? ""
                    if let c = delta.content {
                        let (answer, inline) = splitter.push(c)
                        thought += inline
                        if !answer.isEmpty {
                            if let start = thinkStart, reply.thinkingSeconds == nil { reply.thinkingSeconds = Date().timeIntervalSince(start) }
                            reply.content += answer
                        }
                    }
                    if !thought.isEmpty {
                        thinkStart = thinkStart ?? Date()
                        reply.reasoning += thought
                    }
                    if let speed = delta.tokensPerSecond {
                        reply.footer = L.f("ReplyFooter", model.name, String(format: "%.1f", speed))
                        Log.info("Chat speed \(model.id): \(String(format: "%.1f", speed)) tok/s on \(app.engines.backend)")
                    }
                    if let used = delta.contextTokens {
                        contextText = L.f("ContextMeter", Self.tokens(used), Self.tokens(Self.contextLimit(model)))
                    }
                }
                if let start = thinkStart, reply.thinkingSeconds == nil { reply.thinkingSeconds = Date().timeIntervalSince(start) }
            } catch is CancellationError {
                // Stopped by the user; keep what was generated.
            } catch let error as URLError where error.code == .cancelled {
            } catch {
                Log.error("Chat failed", error)
                reply.status = L.f("ErrorPrefix", error.localizedDescription)
            }
            // Text after a </think> block usually starts with blank lines.
            reply.content = String(reply.content.drop { $0.isWhitespace })
            if reply.status == L.s("SearchingWeb") || reply.status == L.f("LoadingModel", model.name) { reply.status = "" }
            reply.isDone = true
            isGenerating = false
            if !reply.content.isEmpty || !reply.reasoning.isEmpty {
                app.chatStore.add(MessageRecord(id: reply.id, chatId: chatId, role: "assistant", content: reply.content,
                                                reasoning: reply.reasoning.isEmpty ? nil : reply.reasoning, modelId: model.id, createdAt: Date(),
                                                sources: reply.sources))
            }
            if let index = all.firstIndex(where: { $0.id == chatId }) {
                var chat = all.remove(at: index)
                chat.updatedAt = Date()
                all.insert(chat, at: 0)
            }
        }
    }

    /// The context window the engine runs this model with (the smaller of the model's and the setting's).
    private static func contextLimit(_ model: InstalledModel) -> Int {
        model.format == "gguf" ? LlamaEngine.context(for: model, wanted: AppSettings.current.contextLength)
            : model.contextLength > 0 ? min(model.contextLength, AppSettings.current.contextLength) : AppSettings.current.contextLength
    }

    static func tokens(_ n: Int) -> String { n < 1000 ? "\(n)" : "\(Int((Double(n) / 1000).rounded()))k" }

    // MARK: Share

    /// The open chat as Markdown, for Share.
    func exportMarkdown() -> String {
        var md = "# \(currentTitle.isEmpty ? "LLM Hub" : currentTitle)\n\n"
        for m in messages {
            md += "**\(m.isUser ? L.s("ExportYou") : selectedModel?.name ?? "LLM Hub")**\n\n"
            for a in m.attachments { md += "> \(a.name)\n" }
            if !m.attachments.isEmpty { md += "\n" }
            md += m.content + "\n\n"
            for s in m.sources { md += "[\(s.number)]: \(s.url) \"\(s.title.replacingOccurrences(of: "\"", with: "'"))\"\n" }
            if !m.sources.isEmpty { md += "\n" }
        }
        return md
    }

    /// File name for Share: the chat title without characters file systems reject.
    func exportFileName() -> String {
        let bad = CharacterSet(charactersIn: "/\\:?%*|\"<>")
        let name = (currentTitle.isEmpty ? "chat" : currentTitle).components(separatedBy: bad).joined()
            .trimmingCharacters(in: CharacterSet(charactersIn: "…. ").union(.whitespaces))
        return (name.isEmpty ? "chat" : name) + ".md"
    }

    private static func makeTitle(_ text: String) -> String {
        let line = text.split(separator: "\n", maxSplits: 1).first.map(String.init)?.trimmingCharacters(in: .whitespaces) ?? text
        return line.count <= 48 ? line : String(line.prefix(45)).trimmingCharacters(in: .whitespaces) + "…"
    }
}
