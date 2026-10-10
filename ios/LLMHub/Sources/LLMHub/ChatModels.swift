import Foundation

// MARK: - Data Models

public struct ChatMessage: Identifiable, Equatable, Sendable, Codable {
    public let id: UUID
    public var content: String
    public let isFromUser: Bool
    public let timestamp: Date
    public var isGenerating: Bool
    public var tokenCount: Int?
    public var tokensPerSecond: Double?
    public var attachmentImagePath: String?
    public var attachmentAudioPath: String?
    public var attachmentDocumentName: String?   // display name of attached text document

    public enum Role: String, Codable, Sendable {
        case user
        case assistant
        case system
    }

    public var role: Role {
        get { isFromUser ? .user : .assistant }
    }

    public var imageURL: URL? {
        guard let attachmentImagePath = attachmentImagePath else { return nil }
        return URL(fileURLWithPath: attachmentImagePath)
    }

    public var thinkingContent: String? {
        let parsed = parseThinkingAndAnswer(content)
        return parsed.thinking.isEmpty ? nil : parsed.thinking
    }

    public init(
        id: UUID = UUID(),
        content: String,
        isFromUser: Bool,
        timestamp: Date = Date(),
        isGenerating: Bool = false,
        tokenCount: Int? = nil,
        tokensPerSecond: Double? = nil,
        attachmentImagePath: String? = nil,
        attachmentAudioPath: String? = nil,
        attachmentDocumentName: String? = nil
    ) {
        self.id = id
        self.content = content
        self.isFromUser = isFromUser
        self.timestamp = timestamp
        self.isGenerating = isGenerating
        self.tokenCount = tokenCount
        self.tokensPerSecond = tokensPerSecond
        self.attachmentImagePath = attachmentImagePath
        self.attachmentAudioPath = attachmentAudioPath
        self.attachmentDocumentName = attachmentDocumentName
    }

    public init(
        id: UUID = UUID(),
        role: Role,
        content: String,
        imageURL: URL? = nil
    ) {
        self.init(
            id: id,
            content: content,
            isFromUser: role == .user,
            attachmentImagePath: imageURL?.path
        )
    }
}

public struct ChatSession: Identifiable, Sendable, Codable {
    public let id: UUID
    public var title: String
    public var messages: [ChatMessage]
    public let createdAt: Date

    public var lastUpdated: Date {
        messages.last?.timestamp ?? createdAt
    }

    public init(id: UUID = UUID(), title: String = "", messages: [ChatMessage] = [], createdAt: Date = Date()) {
        self.id = id
        self.title = title
        self.messages = messages
        self.createdAt = createdAt
    }
}
