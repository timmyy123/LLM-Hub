import Foundation

// MARK: - Data Models

/// An image/audio global memory that was retrieved for an assistant reply; shown under the reply.
public struct ReferencedMemoryMedia: Identifiable, Equatable, Hashable, Sendable, Codable {
    public let docId: String
    public let type: String       // MemoryMedia.typeImage | MemoryMedia.typeAudio
    public let fileName: String

    public var id: String { docId }

    public init(docId: String, type: String, fileName: String) {
        self.docId = docId
        self.type = type
        self.fileName = fileName
    }
}

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
    public var referencedMedia: [ReferencedMemoryMedia]?

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
