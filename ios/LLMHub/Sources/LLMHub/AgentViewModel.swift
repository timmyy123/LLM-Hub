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
                        let savedMaxTokens = UserDefaults.standard.double(forKey: "agent_max_tokens")
                        let maxTok = savedMaxTokens > 0 ? savedMaxTokens : 4096
                        let modelContextCap = modelToLoad.contextWindowSize > 0 ? modelToLoad.contextWindowSize : 4096
                        let effectiveContext = min(max(1, Int(maxTok)), modelContextCap)
                        LLMBackend.shared.maxTokens = min(Int(maxTok), effectiveContext)
                        LLMBackend.shared.contextWindow = effectiveContext
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

    private func updateTextMessage(id: String, newContent: String) {
        if let idx = messages.firstIndex(where: { $0.id == id }) {
            if case .text(_, let sender, _, let timestamp) = messages[idx] {
                messages[idx] = .text(id: id, sender: sender, content: newContent, timestamp: timestamp)
            }
        }
    }

    private func processPromptWithTools(prompt: String) async {
        let todayStr: String = {
            let fmt = DateFormatter()
            fmt.dateFormat = "yyyy-MM-dd"
            return fmt.string(from: Date())
        }()

        if isWebSearchEnabled {
            let searchResults = await WebSearchService.shared.search(query: prompt, maxResults: 5)
            
            let resultsText = searchResults.enumerated().map { i, r in
                "SOURCE: \(r.source)\nTITLE: \(r.title)\nURL: \(r.url)\nCONTENT: \(r.snippet)\n---"
            }.joined(separator: "\n\n")

            let systemPrompt = """
            CURRENT WEB SEARCH RESULTS:
            \(resultsText)

            Based on the above current web search results, please answer the user's question: "\(prompt)"

            IMPORTANT INSTRUCTIONS:
            - Use ONLY the information from the web search results above
            - If the search results contain the answer, provide a clear and specific response
            - If the search results don't contain enough information, say so clearly
            - For dates and events, be specific based on what you find in the results
            - Do not make up information not found in the search results
            - Cite factual claims with the provided source URLs when possible
            """

            let aiMsgId = UUID().uuidString
            messages.append(.text(id: aiMsgId, sender: .agent, content: "", timestamp: Date()))

            if LLMBackend.shared.isLoaded {
                let savedMaxTokens = UserDefaults.standard.double(forKey: "agent_max_tokens")
                let maxTok = savedMaxTokens > 0 ? savedMaxTokens : 4096
                if let loadedName = LLMBackend.shared.currentlyLoadedModel,
                   let model = ModelData.allModels().first(where: { $0.name == loadedName }) {
                    let modelContextCap = model.contextWindowSize > 0 ? model.contextWindowSize : 4096
                    let effectiveContext = min(max(1, Int(maxTok)), modelContextCap)
                    LLMBackend.shared.contextWindow = effectiveContext
                    LLMBackend.shared.maxTokens = min(Int(maxTok), effectiveContext)
                }
                do {
                    try await LLMBackend.shared.generate(prompt: prompt, systemPrompt: systemPrompt) { [weak self] text, _, _ in
                        Task { @MainActor [weak self] in
                            self?.updateTextMessage(id: aiMsgId, newContent: text)
                        }
                    }
                } catch {
                    updateTextMessage(id: aiMsgId, newContent: "Error: \(error.localizedDescription)")
                }
            } else {
                updateTextMessage(id: aiMsgId, newContent: "Search Results:\n\n\(resultsText)")
            }

            // Yield to MainActor so all enqueued stream update tasks finish before appending sources
            await Task.yield()

            // Append Sources markdown list exactly like AI Chat
            let uniqueSources = Dictionary(grouping: searchResults.filter { !$0.url.isEmpty }, by: \.url).compactMap { $0.value.first }
            if !uniqueSources.isEmpty {
                let sourcesText = uniqueSources.map { "- [\($0.title.isEmpty ? $0.source : $0.title)](\($0.url))" }.joined(separator: "\n")
                if let idx = messages.firstIndex(where: { $0.id == aiMsgId }),
                   case .text(let id, let sender, let content, let ts) = messages[idx] {
                    let baseContent = content.components(separatedBy: "\n\n### Sources\n").first ?? content
                    messages[idx] = .text(id: id, sender: sender, content: baseContent + "\n\n### Sources\n\(sourcesText)", timestamp: ts)
                }
            }
            return
        }

        guard LLMBackend.shared.isLoaded else {
            await executeToolOrFallback(prompt: prompt)
            return
        }

        // Semantic LLM Generation & Function Calling System Prompt
        let systemPrompt = """
        You are an AI Agent equipped with device tools. Today's date is \(todayStr).
        - show_map(location: "place/venue query")
        - send_email(recipient: "email address or contact name", subject: "subject line", body: "email body text")
        - send_sms(recipient: "contact name or phone number", body: "SMS text content")
        - add_calendar_event(title: "event title", date: "event date/time")
        - check_weather(location: "city/location")
        - set_alarm(time: "time e.g. 9 PM", label: "label")
        - toggle_flashlight(enabled: "true" or "false")

        IMPORTANT: For alarm requests, ALWAYS use set_alarm — never add_calendar_event.

        To execute a tool call, output formatted exactly as:
        [TOOL: tool_name(arguments)]

        User Request: \(prompt)
        """

        let savedMaxTokens = UserDefaults.standard.double(forKey: "agent_max_tokens")
        let maxTok = savedMaxTokens > 0 ? savedMaxTokens : 4096
        if let loadedName = LLMBackend.shared.currentlyLoadedModel,
           let model = ModelData.allModels().first(where: { $0.name == loadedName }) {
            let modelContextCap = model.contextWindowSize > 0 ? model.contextWindowSize : 4096
            let effectiveContext = min(max(1, Int(maxTok)), modelContextCap)
            LLMBackend.shared.contextWindow = effectiveContext
            LLMBackend.shared.maxTokens = min(Int(maxTok), effectiveContext)
        }

        let aiMsgId = UUID().uuidString
        messages.append(.text(id: aiMsgId, sender: .agent, content: "", timestamp: Date()))

        do {
            try await LLMBackend.shared.generate(prompt: systemPrompt) { [weak self] text, _, _ in
                Task { @MainActor [weak self] in
                    guard let self = self else { return }
                    if let toolMatch = self.parseToolCall(from: text) {
                        self.messages.removeAll(where: { $0.id == aiMsgId })
                        await self.handleParsedToolCall(toolMatch, originalPrompt: prompt)
                    } else {
                        self.updateTextMessage(id: aiMsgId, newContent: text)
                    }
                }
            }
        } catch {
            updateTextMessage(id: aiMsgId, newContent: "Error: \(error.localizedDescription)")
        }

        if let idx = messages.firstIndex(where: { $0.id == aiMsgId }),
           case .text(_, _, let content, _) = messages[idx], content.isEmpty {
            messages.remove(at: idx)
        }
    }

    private struct ParsedTool {
        let name: String
        let args: String
    }

    private func parseToolCall(from text: String) -> ParsedTool? {
        let clean = text.trimmingCharacters(in: .whitespacesAndNewlines)

        // Separators between words are optional to handle CamelCase LLM outputs
        // e.g. ADDCalendarEvent, SetAlarm, ShowMap
        let pattern = "(?:\\[|\\b)(TOOL:|SHOW[_./ ]?MAP|SEND[_./ ]?SMS|SEND[_./ ]?EMAIL|ADD[_./ ]?CALENDAR[_./ ]?EVENT|CREATE[_./ ]?CALENDAR[_./ ]?EVENT|CHECK[_./ ]?WEATHER|SET[_./ ]?ALARM|TOGGLE[_./ ]?FLASHLIGHT)[:\\(](.+)"
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

        let normalizedName = name.replacingOccurrences(of: ".", with: "_")
            .replacingOccurrences(of: "/", with: "_")
            .lowercased()
        return ParsedTool(name: normalizedName, args: argsStr)
    }

    private func handleParsedToolCall(_ tool: ParsedTool, originalPrompt: String) async {
        let toolId = UUID().uuidString
        let isAlarmIntent = originalPrompt.lowercased().contains("alarm") || tool.args.lowercased().contains("alarm")
        let effectiveToolName = (isAlarmIntent && (tool.name.contains("calendar") || tool.name.contains("event"))) ? "set_alarm" : tool.name.lowercased()

        messages.append(.toolCall(id: toolId, name: effectiveToolName, args: tool.args, status: .running, result: nil))

        switch effectiveToolName {
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

        case "add_calendar_event", "create_calendar_event":
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
            var rawTime = extractArgValue(from: tool.args, key: "time") ?? extractArgValue(from: tool.args, key: "date") ?? tool.args
            // If the LLM returned a full datetime string like "2023-09-01 21:00:00",
            // strip the date portion and keep only the time part.
            if let spaceIdx = rawTime.lastIndex(of: " ") {
                let possibleTime = String(rawTime[rawTime.index(after: spaceIdx)...])
                // Only use the suffix if it looks like HH:MM or HH:MM:SS
                if possibleTime.contains(":") {
                    rawTime = possibleTime
                }
            }
            let label = extractArgValue(from: tool.args, key: "label") ?? extractArgValue(from: tool.args, key: "title") ?? "Alarm"
            let res = await AgentTools.shared.setAlarm(time: rawTime, label: label)
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
        let pattern = "\(key)\\s*(?:=\\s*|:\\s*|\\()\\s*\"([^\"]+)\"|\(key)\\s*(?:=\\s*|:\\s*|\\()\\s*'([^']+)'|\(key)\\s*(?:=\\s*|:\\s*)\\s*([^,\\s)]+)"
        if let range = args.range(of: pattern, options: [.regularExpression, .caseInsensitive]) {
            let matched = String(args[range])
            var cleanVal: String
            if let firstQuote = matched.firstIndex(of: "\""), let lastQuote = matched.lastIndex(of: "\""), firstQuote < lastQuote {
                cleanVal = String(matched[matched.index(after: firstQuote)..<lastQuote])
            } else if let firstQuote = matched.firstIndex(of: "'"), let lastQuote = matched.lastIndex(of: "'"), firstQuote < lastQuote {
                cleanVal = String(matched[matched.index(after: firstQuote)..<lastQuote])
            } else {
                let parts = matched.components(separatedBy: CharacterSet(charactersIn: "=:("))
                cleanVal = parts.count >= 2 ? parts[1].trimmingCharacters(in: CharacterSet(charactersIn: ")\"'] ")) : matched
            }

            let lowerKey = key.lowercased()
            if cleanVal.lowercased().hasPrefix("\(lowerKey):") {
                cleanVal = String(cleanVal.dropFirst(lowerKey.count + 1)).trimmingCharacters(in: CharacterSet(charactersIn: " \"'"))
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
