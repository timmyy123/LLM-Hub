import Foundation
import Observation

@MainActor @Observable
final class MessageItem: Identifiable {
    let id = UUID()
    let role: String
    var content: String
    var reasoning = ""
    var status = ""
    /// Model and generation speed, shown under a finished reply.
    var footer = ""

    init(role: String, content: String) {
        self.role = role
        self.content = content
    }

    var isUser: Bool { role == "user" }
}

struct ChatSummary: Identifiable, Hashable {
    let id: String
    var title: String
    var updatedAt: Date
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
    private var suppressLoad = false

    func attach(_ app: AppModel) {
        self.app = app
        all = app.chatStore.listChats().map { ChatSummary(id: $0.id, title: $0.title, updatedAt: $0.updatedAt) }
        ensureModelSelected()
    }

    /// Downloaded models, plus Apple Intelligence when this Mac has it.
    var models: [InstalledModel] {
        (AppleModel.unavailableReason == nil ? [AppleModel.installed] : []) + (app?.modelStore.installed ?? [])
    }
    var selectedModel: InstalledModel? { models.first { $0.id == selectedModelId } }
    var currentTitle: String { all.first { $0.id == selectedChatId }?.title ?? "" }

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

    /// Chats grouped Today / Yesterday / Previous 7 days / Older, filtered by the search box.
    var groups: [(title: String, chats: [ChatSummary])] {
        let query = search.trimmingCharacters(in: .whitespaces)
        let visible = query.isEmpty ? all : all.filter { $0.title.localizedCaseInsensitiveContains(query) }
        let calendar = Calendar.current
        func bucket(_ c: ChatSummary) -> String {
            if calendar.isDateInToday(c.updatedAt) { return L.s("GroupToday") }
            if calendar.isDateInYesterday(c.updatedAt) { return L.s("GroupYesterday") }
            if c.updatedAt > calendar.date(byAdding: .day, value: -7, to: Date())! { return L.s("GroupWeek") }
            return L.s("GroupOlder")
        }
        var order: [String] = []
        var map: [String: [ChatSummary]] = [:]
        for chat in visible.sorted(by: { $0.updatedAt > $1.updatedAt }) {
            let key = bucket(chat)
            if map[key] == nil { order.append(key) }
            map[key, default: []].append(chat)
        }
        return order.map { ($0, map[$0]!) }
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
    }

    func delete(_ chat: ChatSummary) {
        app?.chatStore.deleteChat(chat.id)
        all.removeAll { $0.id == chat.id }
        if selectedChatId == chat.id { newChat() }
    }

    func stop() { generation?.cancel() }

    private func loadSelected() {
        stop()
        messages = (selectedChatId.map { app?.chatStore.messages($0) ?? [] } ?? []).map {
            let item = MessageItem(role: $0.role, content: $0.content)
            item.reasoning = $0.reasoning ?? ""
            return item
        }
    }

    func send() {
        let text = draft.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty, !isGenerating, let app, let model = selectedModel else { return }
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

        messages.append(MessageItem(role: "user", content: text))
        app.chatStore.add(MessageRecord(id: UUID().uuidString, chatId: chatId, role: "user", content: text, reasoning: nil, modelId: nil, createdAt: Date()))
        let reply = MessageItem(role: "assistant", content: "")
        messages.append(reply)
        let history = messages.filter { $0 !== reply && !$0.content.isEmpty }.map { ChatTurn(role: $0.role, content: $0.content) }

        isGenerating = true
        generation = Task {
            do {
                if !app.engines.isReady(model) {
                    reply.status = L.f("LoadingModel", model.name)
                    try await app.engines.load(model)
                }
                reply.status = ""
                var splitter = ThinkSplitter()
                for try await delta in app.engines.stream(model, turns: history) {
                    if let r = delta.reasoning { reply.reasoning += r }
                    if let c = delta.content {
                        let (answer, thought) = splitter.push(c)
                        reply.reasoning += thought
                        reply.content += answer
                    }
                    if let speed = delta.tokensPerSecond {
                        reply.footer = L.f("ReplyFooter", model.name, String(format: "%.1f", speed))
                        Log.info("Chat speed \(model.id): \(String(format: "%.1f", speed)) tok/s on \(app.engines.backend)")
                    }
                }
            } catch is CancellationError {
                // Stopped by the user; keep what was generated.
            } catch let error as URLError where error.code == .cancelled {
            } catch {
                Log.error("Chat failed", error)
                reply.status = L.f("ErrorPrefix", error.localizedDescription)
            }
            isGenerating = false
            if !reply.content.isEmpty || !reply.reasoning.isEmpty {
                app.chatStore.add(MessageRecord(id: UUID().uuidString, chatId: chatId, role: "assistant", content: reply.content,
                                                reasoning: reply.reasoning.isEmpty ? nil : reply.reasoning, modelId: model.id, createdAt: Date()))
            }
            if let index = all.firstIndex(where: { $0.id == chatId }) {
                var chat = all.remove(at: index)
                chat.updatedAt = Date()
                all.insert(chat, at: 0)
            }
        }
    }

    private static func makeTitle(_ text: String) -> String {
        let line = text.split(separator: "\n", maxSplits: 1).first.map(String.init)?.trimmingCharacters(in: .whitespaces) ?? text
        return line.count <= 48 ? line : String(line.prefix(45)).trimmingCharacters(in: .whitespaces) + "…"
    }
}
