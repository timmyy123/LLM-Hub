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
}
