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
        case pendingApproval
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
    @Published var mcpSettings: MCPSettings = MCPClient.shared.loadSettings()
    @Published var mcpTools: [MCPTool] = []
    @Published var mcpStatus: String = ""
    private var pendingMCPCalls: [String: (MCPTool, [String: Any])] = [:]

    public init() {
        if mcpSettings.enabled && !mcpSettings.url.isEmpty { connectMCP(mcpSettings) }
    }

    func connectMCP(_ settings: MCPSettings) {
        MCPClient.shared.saveSettings(settings)
        mcpSettings = settings
        mcpStatus = "connecting"
        Task {
            do {
                mcpTools = try await MCPClient.shared.connectAndDiscover(settings: settings)
                mcpStatus = "connected"
            } catch {
                mcpTools = []
                mcpStatus = error.localizedDescription
            }
        }
    }

    func setMCPEnabled(_ enabled: Bool) {
        if !enabled {
            disconnectMCP()
            return
        }
        var updated = mcpSettings
        updated.enabled = true
        MCPClient.shared.saveSettings(updated)
        mcpSettings = updated
    }

    func disconnectMCP() {
        var disabled = mcpSettings
        disabled.enabled = false
        MCPClient.shared.saveSettings(disabled)
        mcpSettings = disabled
        mcpTools = []
        mcpStatus = ""
        MCPClient.shared.disconnect()
    }

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
        inputText = ""
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
                let enableThinking = UserDefaults.standard.object(forKey: "agent_enable_thinking") as? Bool ?? true
                LLMBackend.shared.enableThinking = enableThinking
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
        let mcpToolDefinitions = mcpTools.map(\.promptLine).joined(separator: "\n")
        let mcpPrompt = mcpToolDefinitions.isEmpty ? "" : """

        Remote MCP Tools (arguments MUST be one valid JSON object):
        \(mcpToolDefinitions)
        """
        let systemPrompt = """
        You are an AI Agent equipped with device tools. Today's date is \(todayStr).
        Available Tools:
        - show_map(location: "query"): Search for and display any place, venue, business category, address, or points of interest near the user (e.g. "bar near me", "coffee", "gas station", "Eiffel Tower").
        - send_email(recipient: "email address or contact name", subject: "subject line", body: "email body text"): Open email app to compose an email.
        - send_sms(recipient: "contact name or phone number", body: "SMS text content"): Open SMS app to send a text message.
        - add_calendar_event(title: "event title", date: "event date/time"): Add an event to device calendar.
        - check_weather(location: "city or 'my location'"): Check weather forecast for a specified city or the user's current location (e.g. "my location", "Tokyo", "London").
        - set_alarm(time: "time e.g. 7:00 AM", label: "alarm label"): Set an alarm on device.
        - toggle_flashlight(enabled: "true" or "false"): Turn flashlight ON or OFF.
        - calculate_hash(text: "text string", algorithm: "SHA-256 or MD5 or SHA-512"): Calculate cryptographic hash of input text.
        - calculate_math(expression: "math expression string"): Calculate mathematical expression (e.g. "1+1", "15 * 8", "100 / 4").
        \(mcpPrompt)

        Instructions:
        - ALWAYS call a tool when the user asks to perform an action supported by the tools.
        - Use show_map ONLY for physical places, venues, businesses, addresses, points of interest, or directions.
        - Use MCP tools for requests that match their tool names, descriptions, and schemas.
        - For any request about weather (e.g. "How's the weather", "Is it cold outside?"), YOU MUST call check_weather(location: "my location").
        - For any request to calculate a hash or hash a text (e.g. "Calculate SHA-256 hash for 'Hello World'"), YOU MUST call calculate_hash(text: "Hello World", algorithm: "SHA-256").
        - For any request to evaluate or solve a math expression or calculation (e.g. "1+1", "What's 15*8"), YOU MUST call calculate_math(expression: "expression string").
        - Output tool calls in this format ONLY:
        [TOOL: tool_name(arguments)]
        - For MCP tools, pass exactly one valid JSON object using the shown argument keys.
        - MCP arguments marked REQUIRED must be present. If a required MCP argument is unknown, ask a short question instead of calling the tool.

        User Request: \(prompt)
        """

        let enableThinking = UserDefaults.standard.object(forKey: "agent_enable_thinking") as? Bool ?? true
        LLMBackend.shared.enableThinking = enableThinking
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

        var executedTool = false
        do {
            try await LLMBackend.shared.generate(prompt: systemPrompt, enableAgentToolsOverride: false) { [weak self] text, _, _ in
                Task { @MainActor [weak self] in
                    guard let self = self, !executedTool else { return }
                    if let toolMatch = self.parseToolCall(from: text) {
                        executedTool = true
                        self.messages.removeAll(where: { $0.id == aiMsgId })
                        await self.handleParsedToolCall(toolMatch, originalPrompt: prompt)
                    } else {
                        let cleanText = text
                            .replacingOccurrences(of: "\\[\\s*(?:TOOL|calc|calculator|math)\\s*:[^\\]]*(?:\\]|$)", with: "", options: [.regularExpression, .caseInsensitive])
                        let formattedContent = cleanText.contains("\u{200B}") ? cleanText : cleanText.trimmingCharacters(in: .whitespacesAndNewlines)
                        self.updateTextMessage(id: aiMsgId, newContent: formattedContent)
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
        if let explicitTool = parseBracketedToolCall(from: text, requireToolPrefix: true) {
            return explicitTool
        }
        return parseBracketedToolCall(from: text, requireToolPrefix: false)
    }

    private func parseBracketedToolCall(from text: String, requireToolPrefix: Bool) -> ParsedTool? {
        let pattern = requireToolPrefix
            ? "\\[\\s*TOOL:\\s*([a-zA-Z0-9._]+)\\s*\\("
            : "\\[\\s*([a-zA-Z0-9._]+)\\s*\\("
        guard let regex = try? NSRegularExpression(pattern: pattern, options: .caseInsensitive) else {
            return nil
        }
        let nsText = text as NSString
        guard let match = regex.firstMatch(in: text, options: [], range: NSRange(location: 0, length: nsText.length)) else {
            return nil
        }

        var rawName = nsText.substring(with: match.range(at: 1)).lowercased().replacingOccurrences(of: ".", with: "_")
        let argsStart = NSMaxRange(match.range)
        var parenthesisDepth = 1
        var inString = false
        var quote: unichar = 0
        var escaped = false
        var cursor = argsStart

        while cursor < nsText.length {
            let character = nsText.character(at: cursor)
            if inString {
                if escaped {
                    escaped = false
                } else if character == 92 {
                    escaped = true
                } else if character == quote {
                    inString = false
                }
            } else if character == 34 || character == 39 {
                inString = true
                quote = character
            } else if character == 40 {
                parenthesisDepth += 1
            } else if character == 41 {
                parenthesisDepth -= 1
                if parenthesisDepth == 0 { break }
            }
            cursor += 1
        }

        guard parenthesisDepth == 0 else { return nil }
        var closingBracket = cursor + 1
        while closingBracket < nsText.length,
              let scalar = UnicodeScalar(nsText.character(at: closingBracket)),
              CharacterSet.whitespacesAndNewlines.contains(scalar) {
            closingBracket += 1
        }
        guard closingBracket < nsText.length, nsText.character(at: closingBracket) == 93 else { return nil }

        let rawArgs = nsText.substring(with: NSRange(location: argsStart, length: cursor - argsStart))
            .trimmingCharacters(in: .whitespacesAndNewlines)

        if rawName == "c" || rawName == "calc" || rawName == "math" || rawName == "calculate" {
            rawName = "calculate_math"
        }

        if !requireToolPrefix, !isKnownToolName(rawName) {
            return nil
        }

        return ParsedTool(name: rawName, args: rawArgs)
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

        let knownTools: Set<String> = ["show_map", "send_email", "send_sms", "add_calendar_event", "create_calendar_event", "check_weather", "get_current_weather", "set_alarm", "toggle_flashlight", "calculate_math"]
        let lowerArgsPrefix = argsStr.components(separatedBy: "(").first?.trimmingCharacters(in: .whitespaces).lowercased() ?? ""

        var normalizedName = name.lowercased().replacingOccurrences(of: ".", with: "_")
        if knownTools.contains(lowerArgsPrefix) {
            normalizedName = lowerArgsPrefix
            if let pIdx = argsStr.firstIndex(of: "(") {
                argsStr = String(argsStr[argsStr.index(after: pIdx)...]).trimmingCharacters(in: CharacterSet(charactersIn: ")\"'] \t\n\r"))
            }
        }

        return ParsedTool(name: normalizedName, args: argsStr)
    }

    private func handleParsedToolCall(_ tool: ParsedTool, originalPrompt: String) async {
        let toolId = UUID().uuidString
        let canonicalRequestedName = canonicalToolName(tool.name)
        if let mcpTool = mcpTools.first(where: {
            $0.exposedName.caseInsensitiveCompare(tool.name) == .orderedSame ||
                canonicalToolName($0.exposedName) == canonicalRequestedName
        }) {
            let arguments = parseMCPArguments(tool.args)
            queueMCPToolCall(mcpTool, arguments: arguments, fallbackArgs: tool.args, toolId: toolId)
            return
        }
        let isAlarmIntent = originalPrompt.lowercased().contains("alarm") || tool.args.lowercased().contains("alarm")
        var effectiveToolName = (isAlarmIntent && (tool.name.contains("calendar") || tool.name.contains("event"))) ? "set_alarm" : tool.name.lowercased()

        // Normalize tool name aliases — LiteRT-LM Gemma-4 may output open_map, find_map, display_map etc.
        if effectiveToolName.contains("map") { effectiveToolName = "show_map" }
        if effectiveToolName.contains("weather") { effectiveToolName = "check_weather" }
        if effectiveToolName.contains("email") || effectiveToolName.contains("mail") { effectiveToolName = "send_email" }
        if effectiveToolName.contains("sms") || effectiveToolName == "send_message" { effectiveToolName = "send_sms" }
        if effectiveToolName.contains("flashlight") || effectiveToolName.contains("torch") { effectiveToolName = "toggle_flashlight" }
        if effectiveToolName.contains("alarm") && effectiveToolName != "set_alarm" { effectiveToolName = "set_alarm" }
        if effectiveToolName.contains("calendar") || effectiveToolName.contains("event") { effectiveToolName = "add_calendar_event" }
        if effectiveToolName.contains("math") || effectiveToolName.contains("calc") { effectiveToolName = "calculate_math" }

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
            // If rawTime starts with a YYYY-MM-DD ISO date prefix, extract the time component
            if let regex = try? NSRegularExpression(pattern: "^\\d{4}-\\d{2}-\\d{2}\\s+(.+)"),
               let match = regex.firstMatch(in: rawTime, options: [], range: NSRange(location: 0, length: rawTime.utf16.count)),
               let r = Range(match.range(at: 1), in: rawTime) {
                rawTime = String(rawTime[r])
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

        case "calculate_hash":
            let text = extractArgValue(from: tool.args, key: "text") ?? extractFirstPart(from: tool.args)
            let algo = extractArgValue(from: tool.args, key: "algorithm") ?? "SHA-256"
            let res = AgentTools.shared.calculateHash(text: text, algorithm: algo)
            let resStr = "Hash (\(algo)) for '\(text)':\n\(res)"
            updateToolCall(id: toolId, status: .success, result: resStr)
            messages.append(.text(id: UUID().uuidString, sender: .agent, content: resStr, timestamp: Date()))

        case "calculate_math", "math", "calc", "calculate":
            let expr = extractArgValue(from: tool.args, key: "expression") ?? extractArgValue(from: tool.args, key: "calc") ?? (tool.args.isEmpty ? originalPrompt : tool.args)
            let res = AgentTools.shared.calculateMath(expression: expr)
            updateToolCall(id: toolId, status: .success, result: res)
            messages.append(.text(id: UUID().uuidString, sender: .agent, content: res, timestamp: Date()))

        default:
            updateToolCall(id: toolId, status: .failed, result: "Unknown tool")
        }
    }

    private func canonicalToolName(_ name: String) -> String {
        String(name.lowercased().filter { $0.isLetter || $0.isNumber })
    }

    private func isKnownToolName(_ name: String) -> Bool {
        let canonicalName = canonicalToolName(name)
        let knownDeviceTools = [
            "showmap", "sendemail", "sendsms", "addcalendarevent", "createcalendarevent",
            "checkweather", "getcurrentweather", "setalarm", "toggleflashlight",
            "calculatehash", "calculatemath"
        ]
        if knownDeviceTools.contains(canonicalName) { return true }
        return mcpTools.contains {
            canonicalToolName($0.exposedName) == canonicalName || canonicalToolName($0.name) == canonicalName
        }
    }

    private func queueMCPToolCall(_ mcpTool: MCPTool, arguments: [String: Any], fallbackArgs: String, toolId: String = UUID().uuidString) {
        let repairedArguments = repairMCPArguments(for: mcpTool, arguments: arguments)
        let missing = missingRequiredMCPArguments(for: mcpTool, arguments: repairedArguments)
        guard missing.isEmpty else {
            let data = try? JSONSerialization.data(withJSONObject: repairedArguments, options: [.sortedKeys])
            let displayArgs = data.flatMap { String(data: $0, encoding: .utf8) } ?? fallbackArgs
            messages.append(.toolCall(
                id: toolId,
                name: mcpTool.exposedName,
                args: displayArgs,
                status: .failed,
                result: String(format: NSLocalizedString("agent_mcp_missing_required_arguments", comment: ""), missing.joined(separator: ", "))
            ))
            return
        }
        pendingMCPCalls[toolId] = (mcpTool, repairedArguments)
        let data = try? JSONSerialization.data(withJSONObject: repairedArguments, options: [.sortedKeys])
        let displayArgs = data.flatMap { String(data: $0, encoding: .utf8) } ?? fallbackArgs
        messages.append(.toolCall(id: toolId, name: mcpTool.exposedName, args: displayArgs, status: .pendingApproval, result: nil))
    }

    private func repairMCPArguments(for mcpTool: MCPTool, arguments: [String: Any]) -> [String: Any] {
        var repaired = arguments
        let required = mcpTool.inputSchema["required"] as? [String] ?? []
        let properties = mcpTool.inputSchema["properties"] as? [String: Any] ?? [:]
        for (key, schemaValue) in properties {
            let schema = schemaValue as? [String: Any]
            if key.lowercased() == "path",
               schema?["type"] as? String == "string",
               let path = repaired[key] as? String,
               let normalized = normalizedMCPPathPlaceholder(path) {
                repaired[key] = normalized
            }
        }
        for key in required where repaired[key] == nil || repaired[key] is NSNull {
            let schema = properties[key] as? [String: Any]
            let type = schema?["type"] as? String
            if key.lowercased() == "path", type == "string" {
                repaired[key] = "."
            }
        }
        return repaired
    }

    private func normalizedMCPPathPlaceholder(_ value: String) -> String? {
        let lower = value.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let placeholders: Set<String> = [
            "", ".", "./", "here", "this folder", "this directory",
            "current folder", "current directory", "current location",
            "working folder", "working directory", "cwd", "my location"
        ]
        return placeholders.contains(lower) ? "." : nil
    }

    private func missingRequiredMCPArguments(for mcpTool: MCPTool, arguments: [String: Any]) -> [String] {
        let required = mcpTool.inputSchema["required"] as? [String] ?? []
        return required.filter { key in
            guard let value = arguments[key], !(value is NSNull) else { return true }
            if let string = value as? String { return string.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
            return false
        }
    }

    private func parseMCPArguments(_ raw: String) -> [String: Any] {
        let clean = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        if let data = clean.data(using: .utf8), let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            return object
        }
        var result: [String: Any] = [:]
        for item in parseCommaSeparatedArguments(clean) {
            let parts = item.split(separator: "=", maxSplits: 1, omittingEmptySubsequences: false)
            guard parts.count == 2 else { continue }
            let key = String(parts[0]).trimmingCharacters(in: .whitespacesAndNewlines)
            let value = parseLooseMCPValue(String(parts[1]).trimmingCharacters(in: .whitespacesAndNewlines))
            if !key.isEmpty { result[key] = value }
        }
        if !result.isEmpty { return result }

        let pattern = "([A-Za-z0-9_.-]+)\\s*[:=]\\s*(?:\\\"([^\\\"]*)\\\"|'([^']*)'|([^,}]+))"
        if let regex = try? NSRegularExpression(pattern: pattern) {
            let ns = clean as NSString
            for match in regex.matches(in: clean, range: NSRange(location: 0, length: ns.length)) {
                let key = ns.substring(with: match.range(at: 1))
                let ranges = [2, 3, 4].map { match.range(at: $0) }
                if let range = ranges.first(where: { $0.location != NSNotFound }) {
                    result[key] = ns.substring(with: range).trimmingCharacters(in: .whitespacesAndNewlines)
                }
            }
        }
        return result
    }

    private func parseCommaSeparatedArguments(_ raw: String) -> [String] {
        var items: [String] = []
        var current = ""
        var squareDepth = 0
        var braceDepth = 0
        var parenDepth = 0
        var inString = false
        var quote: Character = "\0"
        var escaped = false

        for character in raw {
            if inString {
                current.append(character)
                if escaped {
                    escaped = false
                } else if character == "\\" {
                    escaped = true
                } else if character == quote {
                    inString = false
                }
                continue
            }
            switch character {
            case "\"", "'":
                inString = true
                quote = character
                current.append(character)
            case "[":
                squareDepth += 1
                current.append(character)
            case "]":
                squareDepth = max(0, squareDepth - 1)
                current.append(character)
            case "{":
                braceDepth += 1
                current.append(character)
            case "}":
                braceDepth = max(0, braceDepth - 1)
                current.append(character)
            case "(":
                parenDepth += 1
                current.append(character)
            case ")":
                parenDepth = max(0, parenDepth - 1)
                current.append(character)
            case "," where squareDepth == 0 && braceDepth == 0 && parenDepth == 0:
                items.append(current)
                current = ""
            default:
                current.append(character)
            }
        }
        if !current.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            items.append(current)
        }
        return items
    }

    private func parseLooseMCPValue(_ raw: String) -> Any {
        let clean = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        if let data = clean.data(using: .utf8),
           let value = try? JSONSerialization.jsonObject(with: data) {
            return value
        }
        if (clean.hasPrefix("'") && clean.hasSuffix("'")) || (clean.hasPrefix("\"") && clean.hasSuffix("\"")) {
            return String(clean.dropFirst().dropLast())
        }
        if clean.caseInsensitiveCompare("true") == .orderedSame { return true }
        if clean.caseInsensitiveCompare("false") == .orderedSame { return false }
        return clean
    }

    func approveMCPTool(id: String) {
        guard let pending = pendingMCPCalls.removeValue(forKey: id) else { return }
        updateToolCall(id: id, status: .running, result: nil)
        Task {
            do {
                let result = try await MCPClient.shared.callTool(name: pending.0.name, arguments: pending.1, settings: mcpSettings)
                updateToolCall(id: id, status: .success, result: result)
                messages.append(.text(id: UUID().uuidString, sender: .agent, content: result, timestamp: Date()))
            } catch {
                updateToolCall(id: id, status: .failed, result: error.localizedDescription)
            }
        }
    }

    func denyMCPTool(id: String) {
        pendingMCPCalls.removeValue(forKey: id)
        updateToolCall(id: id, status: .failed, result: NSLocalizedString("agent_mcp_denied", comment: ""))
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
            if key == "location" && (lower == "query" || lower.contains("weather") || lower.contains("current location") || lower.contains("here") || lower.contains("my location")) {
                return "my location"
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
            messages.append(.toolCall(id: toolId, name: "check_weather", args: "my location", status: .running, result: nil))
            let weatherResult = await AgentTools.shared.checkWeather(location: "my location")
            updateToolCall(id: toolId, status: .success, result: weatherResult)
            messages.append(.text(id: UUID().uuidString, sender: .agent, content: weatherResult, timestamp: Date()))
        } else if lower.contains("alarm") || lower.contains("remind") || lower.contains("wake") {
            messages.append(.toolCall(id: toolId, name: "set_alarm", args: prompt, status: .running, result: nil))
            let alarmResult = await AgentTools.shared.setAlarm(time: prompt, label: "Alarm")
            updateToolCall(id: toolId, status: .success, result: alarmResult)
            messages.append(.text(id: UUID().uuidString, sender: .agent, content: alarmResult, timestamp: Date()))
        } else if lower.contains("hash") || lower.contains("sha") || lower.contains("md5") {
            var algo = "SHA-256"
            if lower.contains("md5") { algo = "MD5" }
            else if lower.contains("sha-1") || lower.contains("sha1") { algo = "SHA-1" }
            else if lower.contains("sha-512") || lower.contains("sha512") { algo = "SHA-512" }
            
            var textToHash = prompt
            if let firstQuote = prompt.firstIndex(of: "'"), let lastQuote = prompt.lastIndex(of: "'"), firstQuote < lastQuote {
                textToHash = String(prompt[prompt.index(after: firstQuote)..<lastQuote])
            } else if let firstQuote = prompt.firstIndex(of: "\""), let lastQuote = prompt.lastIndex(of: "\""), firstQuote < lastQuote {
                textToHash = String(prompt[prompt.index(after: firstQuote)..<lastQuote])
            }
            
            messages.append(.toolCall(id: toolId, name: "calculate_hash", args: "text: \"\(textToHash)\", algorithm: \"\(algo)\"", status: .running, result: nil))
            let hashResult = AgentTools.shared.calculateHash(text: textToHash, algorithm: algo)
            let resStr = "Hash (\(algo)) for '\(textToHash)':\n\(hashResult)"
            updateToolCall(id: toolId, status: .success, result: resStr)
            messages.append(.text(id: UUID().uuidString, sender: .agent, content: resStr, timestamp: Date()))
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

    private func updateToolCall(id: String, status: AgentMessageItem.ToolStatus, result: String?) {
        if let idx = messages.firstIndex(where: { $0.id == id }) {
            if case .toolCall(_, let name, let args, _, _) = messages[idx] {
                messages[idx] = .toolCall(id: id, name: name, args: args, status: status, result: result)
            }
        }
    }
}
