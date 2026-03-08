import Foundation

// MARK: - Data Models
public struct ChatMessage: Identifiable, Equatable, Sendable, Codable {
    public let id: UUID
    public var content: String
    public let isFromUser: Bool
    public let timestamp: Date
    public var isGenerating: Bool

    public init(id: UUID = UUID(), content: String, isFromUser: Bool, timestamp: Date = Date(), isGenerating: Bool = false) {
        self.id = id
        self.content = content
        self.isFromUser = isFromUser
        self.timestamp = timestamp
        self.isGenerating = isGenerating
    }
}

public struct ChatSession: Identifiable, Sendable, Codable {
    public let id: UUID
    public var title: String
    public var messages: [ChatMessage]
    public let createdAt: Date

    public init(id: UUID = UUID(), title: String = "", messages: [ChatMessage] = [], createdAt: Date = Date()) {
        self.id = id
        self.title = title
        self.messages = messages
        self.createdAt = createdAt
    }
}
