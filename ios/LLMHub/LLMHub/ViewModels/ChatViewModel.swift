//
//  ChatViewModel.swift
//  LLMHub
//
//  View model for chat functionality
//

import Foundation
import Combine

@MainActor
class ChatViewModel: ObservableObject {
    @Published var messages: [ChatMessage] = []
    @Published var currentInput: String = ""
    @Published var isGenerating: Bool = false
    @Published var errorMessage: String?
    
    private let inferenceService: InferenceService
    private var cancellables = Set<AnyCancellable>()
    
    init(inferenceService: InferenceService) {
        self.inferenceService = inferenceService
        
        // Observe inference service state
        inferenceService.$isGenerating
            .assign(to: &$isGenerating)
        
        inferenceService.$errorMessage
            .assign(to: &$errorMessage)
    }
    
    func sendMessage() async {
        guard !currentInput.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return
        }
        
        let userMessage = ChatMessage(text: currentInput, isUser: true)
        messages.append(userMessage)
        
        let prompt = currentInput
        currentInput = ""
        
        do {
            // Use streaming response
            var responseText = ""
            let placeholderMessage = ChatMessage(text: "", isUser: false)
            messages.append(placeholderMessage)
            let messageIndex = messages.count - 1
            
            for try await chunk in inferenceService.generateResponseStream(prompt: prompt) {
                responseText += chunk
                // Update the message in place
                messages[messageIndex] = ChatMessage(
                    id: placeholderMessage.id,
                    text: responseText,
                    isUser: false,
                    timestamp: placeholderMessage.timestamp
                )
            }
        } catch {
            errorMessage = error.localizedDescription
            // Remove the placeholder message if error occurred
            if let lastMessage = messages.last, !lastMessage.isUser, lastMessage.text.isEmpty {
                messages.removeLast()
            }
        }
    }
    
    func clearChat() {
        messages.removeAll()
    }
}
