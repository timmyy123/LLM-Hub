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
            return
        }

        guard LLMBackend.shared.isLoaded else {
            // Model not loaded fallback: execute direct tool requests semantically
            await executeToolOrFallback(prompt: prompt)
            return
        }

        // Semantic LLM Generation & Function Calling System Prompt
        let systemPrompt = """
        You are an AI Agent equipped with device tools:
        - show_map(location: "place/venue query")
        - send_email(recipient: "email address or contact name", subject: "subject line", body: "email body text")
        - send_sms(recipient: "contact name or phone number", body: "SMS text content")
        - add_calendar_event(title: "event title", date: "event date/time")
        - check_weather(location: "city/location")
        - set_alarm(time: "time", label: "label")
        - toggle_flashlight(enabled: "true" or "false")

        To execute a tool call, output formatted exactly as:
        [TOOL: tool_name(arguments)]

        User Request: \(prompt)
        """

        var fullOutput = ""
        do {
            try await LLMBackend.shared.generate(prompt: systemPrompt, onUpdate: { text, _, _ in
                fullOutput = text
            })
        } catch {
            fullOutput = "Error: \(error.localizedDescription)"
        }

        // Check if LLM emitted a semantic tool call
        if let toolMatch = parseToolCall(from: fullOutput) {
            await handleParsedToolCall(toolMatch, originalPrompt: prompt)
        } else {
            messages.append(.text(id: UUID().uuidString, sender: .agent, content: fullOutput, timestamp: Date()))
        }
    }

    private struct ParsedTool {
        let name: String
        let args: String
    }

    private func parseToolCall(from text: String) -> ParsedTool? {
        let clean = text.trimmingCharacters(in: .whitespacesAndNewlines)

        let pattern = "(?:\\[|\\b)(TOOL:|SHOW_MAP|SEND_SMS|SEND_EMAIL|ADD_CALENDAR_EVENT|CHECK_WEATHER|SET_ALARM|TOGGLE_FLASHLIGHT)[:\\(](.+)"
        if let range = clean.range(of: pattern, options: [.regularExpression, .caseInsensitive]) {
            let matched = String(clean[range])
            return parseNameAndArgs(from: matched)
        }

        return nil
    }

    private func parseNameAndArgs(from text: String) -> ParsedTool? {
        let clean = text.trimmingCharacters(in: CharacterSet(charactersIn: "[] \t\n\r"))
        
        var name = ""
        var argsStr = ""

        let parenIdx = clean.firstIndex(of: "(")
        let colonIdx = clean.firstIndex(of: ":")

        if let pIdx = parenIdx, (colonIdx == nil || pIdx < colonIdx!) {
            name = String(clean[..<pIdx]).trimmingCharacters(in: .whitespaces)
            argsStr = String(clean[clean.index(after: pIdx)...])
                .trimmingCharacters(in: CharacterSet(charactersIn: ")\"'] \t\n\r"))
        } else if let cIdx = colonIdx {
            name = String(clean[..<cIdx]).trimmingCharacters(in: .whitespaces)
            argsStr = String(clean[clean.index(after: cIdx)...]).trimmingCharacters(in: .whitespaces)
        } else {
            return nil
        }

        let knownTools: Set<String> = ["show_map", "send_email", "send_sms", "add_calendar_event", "create_calendar_event", "check_weather", "get_current_weather", "set_alarm", "toggle_flashlight"]
        let lowerArgsPrefix = argsStr.components(separatedBy: "(").first?.trimmingCharacters(in: .whitespaces).lowercased() ?? ""

        if name.lowercased().hasPrefix("tool") || knownTools.contains(lowerArgsPrefix) {
            if let inner = parseNameAndArgs(from: argsStr) {
                return inner
            }
        }

        return ParsedTool(name: name, args: argsStr)
    }

    private func handleParsedToolCall(_ tool: ParsedTool, originalPrompt: String) async {
        let toolId = UUID().uuidString
        messages.append(.toolCall(id: toolId, name: tool.name, args: tool.args, status: .running, result: nil))

        switch tool.name.lowercased() {
        case "show_map":
            let cleanLoc = extractArgValue(from: tool.args, key: "location") ?? tool.args
            if let (lat, lon, name) = await AgentTools.shared.geocodeLocation(cleanLoc) {
                updateToolCall(id: toolId, status: .success, result: "Location found: \(name)")
                messages.append(.map(id: UUID().uuidString, label: name, latitude: lat, longitude: lon))
            } else {
                updateToolCall(id: toolId, status: .failed, result: "Location not found")
                messages.append(.text(id: UUID().uuidString, sender: .agent, content: "Could not find '\(cleanLoc)' on the map.", timestamp: Date()))
            }

        case "send_email", "compose_email":
            let recipient = extractArgValue(from: tool.args, key: "recipient") ?? extractArgValue(from: tool.args, key: "to") ?? extractFirstPart(from: tool.args)
            let body = extractArgValue(from: tool.args, key: "body") ?? extractArgValue(from: tool.args, key: "message") ?? originalPrompt
            let subject = extractArgValue(from: tool.args, key: "subject") ?? "Hello"
            let res = await AgentTools.shared.sendEmail(email: recipient, subject: subject, body: body)
            updateToolCall(id: toolId, status: .success, result: res)
            messages.append(.text(id: UUID().uuidString, sender: .agent, content: res, timestamp: Date()))

        case "send_sms":
            let recipient = extractArgValue(from: tool.args, key: "recipient") ?? extractArgValue(from: tool.args, key: "to") ?? extractFirstPart(from: tool.args)
            let body = extractArgValue(from: tool.args, key: "body") ?? extractArgValue(from: tool.args, key: "message") ?? originalPrompt
            let res = await AgentTools.shared.sendSms(phone: recipient, body: body)
            updateToolCall(id: toolId, status: .success, result: res)
            messages.append(.text(id: UUID().uuidString, sender: .agent, content: res, timestamp: Date()))

        case "add_calendar_event":
            let title = extractArgValue(from: tool.args, key: "title") ?? extractFirstPart(from: tool.args)
            let dateStr = extractArgValue(from: tool.args, key: "date") ?? "tomorrow"
            let res = await AgentTools.shared.addCalendarEvent(title: title, dateStr: dateStr)
            updateToolCall(id: toolId, status: .success, result: res)
            messages.append(.text(id: UUID().uuidString, sender: .agent, content: res, timestamp: Date()))

        case "check_weather":
            let loc = extractArgValue(from: tool.args, key: "location") ?? tool.args
            let res = await AgentTools.shared.checkWeather(location: loc)
            updateToolCall(id: toolId, status: .success, result: res)
            messages.append(.text(id: UUID().uuidString, sender: .agent, content: res, timestamp: Date()))

        case "set_alarm":
            let time = extractArgValue(from: tool.args, key: "time") ?? tool.args
            let label = extractArgValue(from: tool.args, key: "label") ?? "Alarm"
            let res = await AgentTools.shared.setAlarm(time: time, label: label)
            updateToolCall(id: toolId, status: .success, result: res)
            messages.append(.text(id: UUID().uuidString, sender: .agent, content: res, timestamp: Date()))

        case "toggle_flashlight":
            let enabled = tool.args.lowercased().contains("true") || tool.args.lowercased().contains("on")
            let res = AgentTools.shared.toggleFlashlight(enabled: enabled)
            updateToolCall(id: toolId, status: .success, result: res)
            messages.append(.text(id: UUID().uuidString, sender: .agent, content: res, timestamp: Date()))

        default:
            updateToolCall(id: toolId, status: .failed, result: "Unknown tool")
        }
    }

    private func extractArgValue(from args: String, key: String) -> String? {
        let pattern = "\(key)\\s*(?:=\\s*|\\()\\s*\"([^\"]+)\"|\(key)\\s*(?:=\\s*|\\()\\s*'([^']+)'|\(key)\\s*=\\s*([^,\\s)]+)"
        if let range = args.range(of: pattern, options: [.regularExpression, .caseInsensitive]) {
            let matched = String(args[range])
            let cleanVal: String
            if let firstQuote = matched.firstIndex(of: "\""), let lastQuote = matched.lastIndex(of: "\""), firstQuote < lastQuote {
                cleanVal = String(matched[matched.index(after: firstQuote)..<lastQuote])
            } else if let firstQuote = matched.firstIndex(of: "'"), let lastQuote = matched.lastIndex(of: "'"), firstQuote < lastQuote {
                cleanVal = String(matched[matched.index(after: firstQuote)..<lastQuote])
            } else {
                let parts = matched.components(separatedBy: CharacterSet(charactersIn: "=("))
                cleanVal = parts.count >= 2 ? parts[1].trimmingCharacters(in: CharacterSet(charactersIn: ")\"'] ")) : matched
            }

            let lower = cleanVal.lowercased()
            if key == "location" && (lower.contains("weather") || lower.contains("current location") || lower.contains("here") || lower.contains("my location")) {
                return "Melbourne"
            }
            return cleanVal
        }
        return nil
    }

    private func extractFirstPart(from args: String) -> String {
        let parts = args.components(separatedBy: ",")
        return parts.first?.trimmingCharacters(in: CharacterSet(charactersIn: "\"'\t\n\r ")) ?? args
    }

    private func executeToolOrFallback(prompt: String) async {
        let lower = prompt.lowercased()
        let toolId = UUID().uuidString

        if lower.contains("weather") || lower.contains("forecast") || lower.contains("temperature") {
            messages.append(.toolCall(id: toolId, name: "check_weather", args: "Melbourne", status: .running, result: nil))
            let weatherResult = await AgentTools.shared.checkWeather(location: "Melbourne")
            updateToolCall(id: toolId, status: .success, result: weatherResult)
            messages.append(.text(id: UUID().uuidString, sender: .agent, content: weatherResult, timestamp: Date()))
        } else if lower.contains("map") || lower.contains("where is") || lower.contains("find") || lower.contains("direction") {
            messages.append(.toolCall(id: toolId, name: "show_map", args: prompt, status: .running, result: nil))
            if let (lat, lon, name) = await AgentTools.shared.geocodeLocation(prompt) {
                updateToolCall(id: toolId, status: .success, result: "Location found: \(name)")
                messages.append(.map(id: UUID().uuidString, label: name, latitude: lat, longitude: lon))
            } else {
                updateToolCall(id: toolId, status: .failed, result: "Location not found")
                messages.append(.text(id: UUID().uuidString, sender: .agent, content: AppSettings.shared.localized("agent_no_model_ios"), timestamp: Date()))
            }
        } else {
            messages.append(.text(id: UUID().uuidString, sender: .agent, content: AppSettings.shared.localized("agent_no_model_ios"), timestamp: Date()))
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
