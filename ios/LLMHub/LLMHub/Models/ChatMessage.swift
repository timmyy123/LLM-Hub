//
//  ChatMessage.swift
//  LLMHub
//
//  Data model for chat messages
//

import Foundation

struct ChatMessage: Identifiable, Codable {
    let id: String
    let text: String
    let isUser: Bool
    let timestamp: Date
    
    init(id: String = UUID().uuidString,
         text: String,
         isUser: Bool,
         timestamp: Date = Date()) {
        self.id = id
        self.text = text
        self.isUser = isUser
        self.timestamp = timestamp
    }
}
