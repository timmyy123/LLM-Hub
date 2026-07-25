import Foundation
import SwiftUI
import Combine

public enum AgentVoiceMode: String, CaseIterable, Identifiable {
    case system = "System"
    case whisper = "Whisper"
    case gemma = "Gemma"

    public var id: String { rawValue }
}

public enum AgentMessageItem: Identifiable {
    case text(id: String, sender: Sender, content: String, timestamp: Date)
    case toolCall(id: String, name: String, args: String, status: ToolStatus, result: String?)
    case map(id: String, label: String, latitude: Double, longitude: Double)

    public enum Sender {
        case user
        case agent
        case system
    }

    public enum ToolStatus {
        case running
        case success
        case failed
    }

    public var id: String {
        switch self {
        case .text(let id, _, _, _): return id
        case .toolCall(let id, _, _, _, _): return id
        case .map(let id, _, _, _): return id
        }
    }
}

@MainActor
public class AgentViewModel: ObservableObject {

    @Published public var messages: [AgentMessageItem] = []
    @Published public var isGenerating: Bool = false
    @Published public var voiceMode: AgentVoiceMode = .system
    @Published public var selectedAsrModelName: String? = nil
    @Published public var inputText: String = ""
    @Published public var isWebSearchEnabled: Bool = false

    public init() {}

    func setupWelcomeMessage(settings: AppSettings, isDownloaded: Bool) {
        if isDownloaded {
            let welcome = AgentMessageItem.text(
                id: "welcome",
                sender: .agent,
                content: settings.localized("agent_welcome_message"),
                timestamp: Date()
            )
            messages = [welcome]
        } else {
            messages = []
        }
    }

    public func clearMessages() {
        messages = []
    }

    public func sendMessage(_ text: String) {
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty, !isGenerating else { return }

        let userMsg = AgentMessageItem.text(
            id: UUID().uuidString,
            sender: .user,
            content: text,
            timestamp: Date()
        )
        messages.append(userMsg)
        isGenerating = true

        Task {
            // Lazy load model if not loaded yet
            if !LLMBackend.shared.isLoaded {
                let savedName = UserDefaults.standard.string(forKey: "agent_model_name") ?? ""
                if let modelToLoad = ModelData.allModels().first(where: { $0.name == savedName && ModelData.isModelFullyAvailableLocally($0) })
                    ?? ModelData.allModels().first(where: { ModelData.isModelFullyAvailableLocally($0) && !$0.isDependencyOnly && $0.category != .embedding && $0.category != .asr }) {
                    do {
                        try await LLMBackend.shared.loadModel(modelToLoad)
                    } catch {
                        print("Agent lazy load failed: \(error.localizedDescription)")
                    }
                }
            }

            await processPromptWithTools(prompt: text)
            isGenerating = false
        }
    }

    private func processPromptWithTools(prompt: String) async {
        let lower = prompt.lowercased()

        // Direct tool handling for flashlight queries
        if lower.contains("flashlight") || lower.contains("torch") {
            let turnOn = lower.contains("on") || !lower.contains("off")
            let toolId = UUID().uuidString
            messages.append(.toolCall(id: toolId, name: "toggle_flashlight", args: turnOn ? "true" : "false", status: .running, result: nil))

            let res = AgentTools.shared.toggleFlashlight(enabled: turnOn)
            updateToolCall(id: toolId, status: .success, result: res)
            messages.append(.text(id: UUID().uuidString, sender: .agent, content: res, timestamp: Date()))
            return
        }

        if isWebSearchEnabled {
            let toolId = UUID().uuidString
            messages.append(.toolCall(id: toolId, name: "web_search", args: prompt, status: .running, result: nil))

            let searchResult = await AgentTools.shared.webSearch(query: prompt)
            updateToolCall(id: toolId, status: .success, result: searchResult)

            if LLMBackend.shared.isLoaded {
                let fullPrompt = "Web search results:\n\(searchResult)\n\nUser Question: \(prompt)"
                var response = ""
                do {
                    try await LLMBackend.shared.generate(prompt: fullPrompt, onUpdate: { text, _, _ in
                        response = text
                    })
                } catch {
                    response = "Error: \(error.localizedDescription)"
                }
                messages.append(.text(id: UUID().uuidString, sender: .agent, content: response, timestamp: Date()))
            } else {
                messages.append(.text(id: UUID().uuidString, sender: .agent, content: "Search Results:\n\n\(searchResult)", timestamp: Date()))
            }
        } else if LLMBackend.shared.isLoaded {
            var response = ""
            do {
                try await LLMBackend.shared.generate(prompt: prompt, onUpdate: { text, _, _ in
                    response = text
                })
            } catch {
                response = "Error: \(error.localizedDescription)"
            }
            messages.append(.text(id: UUID().uuidString, sender: .agent, content: response, timestamp: Date()))
        } else {
            messages.append(.text(
                id: UUID().uuidString,
                sender: .system,
                content: AppSettings.shared.localized("agent_no_model_ios"),
                timestamp: Date()
            ))
        }
    }

    private func updateToolCall(id: String, status: AgentMessageItem.ToolStatus, result: String) {
        if let idx = messages.firstIndex(where: { $0.id == id }) {
            if case .toolCall(_, let name, let args, _, _) = messages[idx] {
                messages[idx] = .toolCall(id: id, name: name, args: args, status: status, result: result)
            }
        }
    }
}
