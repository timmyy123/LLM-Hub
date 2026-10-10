import Foundation

@MainActor
class ChatStore: ObservableObject {
    static let shared = ChatStore()
    
    @Published var chatSessions: [ChatSession] = []
    
    private let storageURL: URL = {
        let fileManager = FileManager.default
        let documentsDir = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first!
        return documentsDir.appendingPathComponent("chat_sessions.json")
    }()
    
    private init() {
        loadSessions()
    }
    
    func saveSessions() {
        do {
            let data = try JSONEncoder().encode(chatSessions)
            try data.write(to: storageURL)
        } catch {
            print("Failed to save chat sessions: \(error)")
        }
    }
    
    func loadSessions() {
        if !FileManager.default.fileExists(atPath: storageURL.path) {
            let session = ChatSession(title: AppSettings.shared.localized("drawer_new_chat"))
            chatSessions = [session]
            return
        }
        
        do {
            let data = try Data(contentsOf: storageURL)
            chatSessions = try JSONDecoder().decode([ChatSession].self, from: data)
        } catch {
            print("Failed to load chat sessions: \(error)")
            let session = ChatSession(title: AppSettings.shared.localized("drawer_new_chat"))
            chatSessions = [session]
        }
    }
    
    func addSession(_ session: ChatSession) {
        chatSessions.insert(session, at: 0)
        saveSessions()
    }
    
    func deleteSession(id: UUID) {
        chatSessions.removeAll { $0.id == id }
        saveSessions()
    }
    
    func clearAll() {
        chatSessions.removeAll()
        saveSessions()
    }

    @Published var currentConversationId: UUID?

    var conversations: [ChatSession] {
        chatSessions
    }

    var currentMessages: [ChatMessage] {
        get {
            if let currentId = currentConversationId,
               let session = chatSessions.first(where: { $0.id == currentId }) {
                return session.messages
            }
            return chatSessions.first?.messages ?? []
        }
        set {
            let targetId = currentConversationId ?? chatSessions.first?.id
            if let targetId = targetId,
               let idx = chatSessions.firstIndex(where: { $0.id == targetId }) {
                chatSessions[idx].messages = newValue
                saveSessions()
            }
        }
    }

    func selectConversation(_ id: UUID) {
        currentConversationId = id
    }

    func deleteConversation(_ id: UUID) {
        deleteSession(id: id)
        if currentConversationId == id {
            currentConversationId = chatSessions.first?.id
        }
    }

    func clearCurrentConversation() {
        let targetId = currentConversationId ?? chatSessions.first?.id
        if let targetId = targetId,
           let idx = chatSessions.firstIndex(where: { $0.id == targetId }) {
            chatSessions[idx].messages.removeAll()
            saveSessions()
        }
    }

    func appendMessage(_ message: ChatMessage) {
        if chatSessions.isEmpty {
            _ = createNewConversation()
        }
        let targetId = currentConversationId ?? chatSessions.first!.id
        if let idx = chatSessions.firstIndex(where: { $0.id == targetId }) {
            chatSessions[idx].messages.append(message)
            saveSessions()
        }
    }

    func appendChunkToMessage(id: UUID, chunk: String) {
        let targetId = currentConversationId ?? chatSessions.first?.id
        guard let targetId = targetId,
              let sessionIdx = chatSessions.firstIndex(where: { $0.id == targetId }),
              let msgIdx = chatSessions[sessionIdx].messages.firstIndex(where: { $0.id == id }) else {
            return
        }
        chatSessions[sessionIdx].messages[msgIdx].content += chunk
    }

    @discardableResult
    func createNewConversation() -> ChatSession {
        let session = ChatSession(title: AppSettings.shared.localized("drawer_new_chat"))
        addSession(session)
        currentConversationId = session.id
        return session
    }

    func clearAllConversations() {
        clearAll()
        currentConversationId = nil
    }
}
