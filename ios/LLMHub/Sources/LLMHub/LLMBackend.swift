import Foundation
import LlamaCppRuntime
#if canImport(UIKit)
import UIKit
import ImageIO
#endif

@MainActor
class LLMBackend: ObservableObject {
    static let shared = LLMBackend()
    private static let thinkingSentinelOpen = "\u{200B}\u{200B}THINK\u{200B}\u{200B}"
    private static let thinkingSentinelClose = "\u{200B}\u{200B}ENDTHINK\u{200B}\u{200B}"
    private static let harmonyAnalysisHeader = "<|channel|>analysis<|message|>"
    private static let harmonyEndTag = "<|end|>"
    private static let harmonyFinalHeader = "<|start|>assistant<|channel|>final<|message|>"
    private static let harmonyAssistantHeader = "<|start|>assistant"
    private static let appleFoundationAliasId = "apple.foundation.system"

    @Published var isLoaded: Bool = false
    @Published var currentlyLoadedModel: String? = nil
    @Published var isBackendLoading: Bool = false
    @Published var loadedContextWindow: Int? = nil

    // Generation parameters
    var maxTokens: Int = 2048
    var contextWindow: Int = 2048
    var topK: Int = 64
    var topP: Float = 0.95
    var temperature: Float = 1.0
    var selectedBackend: String = "GPU"
    var enableVision: Bool = true
    var enableAudio: Bool = true
    var enableThinking: Bool = true
    var enableAgentTools: Bool = true


    private init() {}

    private struct HarmonyPromptMessage {
        let role: String
        let content: String
    }

    private static func isHarmonyModelName(_ modelName: String?) -> Bool {
        guard let normalized = modelName?.lowercased() else { return false }
        return normalized.contains("gpt-oss") || normalized.contains("gpt_oss")
    }

    private static func isMuseGlimmerModelName(_ modelName: String?) -> Bool {
        guard let normalized = modelName?.lowercased() else { return false }
        return normalized.contains("muse glimmer") || normalized.contains("muse-glimmer")
    }

    private static func buildMuseGlimmerPrompt(prompt: String, systemPrompt: String?, thinkingEnabled: Bool, includeRawPrefix: Bool = true) -> String {
        let cleanPrompt = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        let systemContent = systemPrompt?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        let effectiveSystem = systemContent.isEmpty ? "You are a helpful AI assistant." : systemContent

        let dateFormatter = DateFormatter()
        dateFormatter.calendar = Calendar(identifier: .gregorian)
        dateFormatter.locale = Locale(identifier: "en_US_POSIX")
        dateFormatter.dateFormat = "yyyy-MM-dd"

        var parts = includeRawPrefix ? ["__RAW_PROMPT__"] : []
        let validRecipients = thinkingEnabled ? "\"self\", \"user\"" : "\"user\""
        parts.append(contentsOf: [
            "<|begin_of_text|>",
            "<|start|>system<|message|>\(effectiveSystem)\nKnowledge cutoff: 2026-01-04.\nCurrent date: \(dateFormatter.string(from: Date())).\n\nReasoning strength: high.\n\n# Valid recipients: \(validRecipients).<|eot|>",
            "<|start|>user<|message|>\(cleanPrompt)<|eot|>"
        ])
        parts.append(thinkingEnabled
            ? "<|start|>assistant"
            : "<|start|>assistant to=user<|message|>")
        return parts.joined()
    }

    private static func isGranite42ModelName(_ name: String) -> Bool {
        let lower = name.lowercased()
        return lower.contains("granite-4.2") || lower.contains("granite 4.2")
    }

    private static func buildGranite42Prompt(prompt: String, systemPrompt: String?, thinkingEnabled: Bool, includeRawPrefix: Bool = true) -> String {
        let cleanPrompt = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        let systemContent = systemPrompt?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        var parts = includeRawPrefix ? ["__RAW_PROMPT__"] : []
        if !systemContent.isEmpty {
            parts.append("<|im_start|>system\n\(systemContent)<|im_end|>\n")
        }
        parts.append("<|im_start|>user\n\(cleanPrompt)<|im_end|>\n")
        if thinkingEnabled {
            parts.append("<|im_start|>assistant\n<think>\n")
        } else {
            parts.append("<|im_start|>assistant\n<think></think>")
        }
        return parts.joined()
    }

    /// Muse Glimmer may emit an ATEM recipient header before its text and may
    /// produce a private `self` reasoning turn before the answer for `user`.
    /// Convert those protocol fields to the same thinking sentinels used by the UI.
    private static func normalizeMuseGlimmerOutput(_ raw: String, thinkingEnabled: Bool) -> String {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        let selfHeaders = [
            "<|start|>assistant to=self<|message|>",
            "to=self<|message|>",
            "<|start|>assistant to=self",
            "assistant to=self",
            "to=self",
        ]
        let userHeaders = [
            "<|start|>assistant to=user<|message|>",
            "to=user<|message|>",
            "<|start|>assistant to=user",
            "assistant to=user",
            "to=user",
        ]

        let protocolHeaders = selfHeaders + userHeaders
        if !trimmed.isEmpty,
           protocolHeaders.contains(where: { $0.hasPrefix(trimmed) && $0 != trimmed }) {
            return ""
        }

        func cleanMuseAnswer(_ text: String) -> String {
            var answer = text.trimmingCharacters(in: .whitespacesAndNewlines)
            for header in userHeaders where answer.hasPrefix(header) {
                answer = String(answer.dropFirst(header.count))
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                break
            }
            for token in ["<|eot|>", "<|end_of_text|>"] where answer.hasSuffix(token) {
                answer = String(answer.dropLast(token.count))
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                break
            }
            return answer
        }

        func cleanMuseThinking(_ text: String) -> String {
            var thinking = text.trimmingCharacters(in: .whitespacesAndNewlines)
            for marker in ["Final output.", "Final output:", "Final answer.", "Final answer:", "Proceed.", "Proceed:"] where thinking.hasSuffix(marker) {
                thinking = String(thinking.dropLast(marker.count))
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                break
            }
            return thinking
        }

        if let selfHeader = selfHeaders.first(where: { trimmed.hasPrefix($0) }) {
            guard thinkingEnabled else { return raw }
            let reasoningAndAnswer = String(trimmed.dropFirst(selfHeader.count))
            if let reasoningEnd = reasoningAndAnswer.range(of: "<|eom|>") {
                let reasoning = cleanMuseThinking(String(reasoningAndAnswer[..<reasoningEnd.lowerBound]))
                let answer = cleanMuseAnswer(String(reasoningAndAnswer[reasoningEnd.upperBound...]))
                return thinkingSentinelOpen + reasoning + thinkingSentinelClose + answer
            }
            for header in userHeaders {
                if let answerRange = reasoningAndAnswer.range(of: header) {
                    let reasoning = cleanMuseThinking(String(reasoningAndAnswer[..<answerRange.lowerBound]))
                    let answer = cleanMuseAnswer(String(reasoningAndAnswer[answerRange.lowerBound...]))
                    return thinkingSentinelOpen + reasoning + thinkingSentinelClose + answer
                }
            }
            if let finalRange = reasoningAndAnswer.range(
                of: #"(?i)final (?:output|answer)[\.:]\s*"#,
                options: .regularExpression
            ) {
                let reasoning = cleanMuseThinking(String(reasoningAndAnswer[..<finalRange.lowerBound]))
                let answer = cleanMuseAnswer(String(reasoningAndAnswer[finalRange.upperBound...]))
                return thinkingSentinelOpen + reasoning + thinkingSentinelClose + answer
            }
            return thinkingSentinelOpen + reasoningAndAnswer
        }

        for header in userHeaders where trimmed.hasPrefix(header) {
            return cleanMuseAnswer(String(trimmed.dropFirst(header.count)))
        }
        for header in userHeaders {
            if let answerRange = trimmed.range(of: header) {
                guard thinkingEnabled else { return raw }
                let prefix = cleanMuseThinking(String(trimmed[..<answerRange.lowerBound]))
                let answer = cleanMuseAnswer(String(trimmed[answerRange.lowerBound...]))
                if prefix.isEmpty {
                    return answer
                }
                return thinkingSentinelOpen + prefix + thinkingSentinelClose + answer
            }
        }
        return raw
    }

    private static func buildHarmonyPrompt(prompt: String, systemPrompt: String?, thinkingEnabled: Bool) -> String {
        let cleanPrompt = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        let effectiveSystemPrompt = systemPrompt?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        var messages: [HarmonyPromptMessage] = []

        if cleanPrompt.contains("user: ") || cleanPrompt.contains("assistant: ") {
            let pattern = "(?m)(?=^(?:user|assistant):\\s)"
            let regex = try? NSRegularExpression(pattern: pattern)
            let fullRange = NSRange(cleanPrompt.startIndex..<cleanPrompt.endIndex, in: cleanPrompt)
            let matches = regex?.matches(in: cleanPrompt, range: fullRange) ?? []

            if !matches.isEmpty {
                for index in matches.indices {
                    let start = matches[index].range.location
                    let end = index + 1 < matches.count ? matches[index + 1].range.location : fullRange.length
                    let range = NSRange(location: start, length: end - start)
                    guard let swiftRange = Range(range, in: cleanPrompt) else { continue }
                    let segment = cleanPrompt[swiftRange].trimmingCharacters(in: .whitespacesAndNewlines)
                    if segment.hasPrefix("user: ") {
                        messages.append(HarmonyPromptMessage(role: "user", content: String(segment.dropFirst(6)).trimmingCharacters(in: .whitespacesAndNewlines)))
                    } else if segment.hasPrefix("assistant: ") {
                        messages.append(HarmonyPromptMessage(role: "assistant", content: String(segment.dropFirst(11)).trimmingCharacters(in: .whitespacesAndNewlines)))
                    }
                }
            }
        }

        if messages.isEmpty {
            let featureSeparators = [
                "Text to rewrite:\n",
                "Content to analyze:\n",
                "Text to translate:\n",
                "Text to transcribe:\n",
                "Text to process:\n",
            ]

            if let separator = featureSeparators.first(where: { cleanPrompt.contains($0) }),
               let separatorRange = cleanPrompt.range(of: separator) {
                let instructions = String(cleanPrompt[..<separatorRange.lowerBound]).trimmingCharacters(in: .whitespacesAndNewlines)
                let userContentBody = String(cleanPrompt[separatorRange.upperBound...]).trimmingCharacters(in: .whitespacesAndNewlines)
                let userContent = "\(separator.trimmingCharacters(in: .whitespacesAndNewlines))\n\(userContentBody)".trimmingCharacters(in: .whitespacesAndNewlines)
                if !instructions.isEmpty {
                    messages.append(HarmonyPromptMessage(role: "system", content: instructions))
                }
                if !userContent.isEmpty {
                    messages.append(HarmonyPromptMessage(role: "user", content: userContent))
                }
            } else if cleanPrompt.lowercased().hasPrefix("you are ") && cleanPrompt.contains("\n\n") {
                let chunks = cleanPrompt.components(separatedBy: "\n\n").filter { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
                if chunks.count >= 2 {
                    let systemContent = chunks.dropLast().joined(separator: "\n\n").trimmingCharacters(in: .whitespacesAndNewlines)
                    let userContent = chunks.last?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
                    if !systemContent.isEmpty {
                        messages.append(HarmonyPromptMessage(role: "system", content: systemContent))
                    }
                    if !userContent.isEmpty {
                        messages.append(HarmonyPromptMessage(role: "user", content: userContent))
                    }
                }
            }
        }

        if messages.isEmpty {
            if !effectiveSystemPrompt.isEmpty {
                messages.append(HarmonyPromptMessage(role: "system", content: effectiveSystemPrompt))
            }
            messages.append(HarmonyPromptMessage(role: "user", content: cleanPrompt))
        } else if !effectiveSystemPrompt.isEmpty && !messages.contains(where: { $0.role == "system" }) {
            messages.insert(HarmonyPromptMessage(role: "system", content: effectiveSystemPrompt), at: 0)
        }

        if !messages.contains(where: { $0.role == "system" }) {
            messages.insert(HarmonyPromptMessage(role: "system", content: "You are a helpful assistant."), at: 0)
        }

        var parts = ["__RAW_PROMPT__"]
        for message in messages {
            let content = message.content.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !content.isEmpty else { continue }
            parts.append("<|start|>\(message.role)<|message|>\(content)<|end|>")
        }
        if thinkingEnabled {
            parts.append(Self.harmonyAssistantHeader)
        } else {
            parts.append(Self.harmonyAssistantHeader + Self.harmonyAnalysisHeader + Self.harmonyEndTag + Self.harmonyFinalHeader)
        }
        return parts.joined()
    }

    // Strips the rendered analysis<|message|> prefix (and optional <|channel|> variant)
    // to return pure thinking text. <|channel|> is a non-rendering special token so the
    // llama.cpp stream shows "analysis<|message|>..." not "<|channel|>analysis<|message|>...".
    // While the short prefix is still assembling token-by-token (e.g. "analy"), we hide
    // it entirely rather than showing partial scaffold text.
    private static let harmonyAnalysisPrefixShort = "analysis<|message|>"
    private static func extractHarmonyThinking(_ raw: String) -> String {
        let shortPrefix = Self.harmonyAnalysisPrefixShort
        // Full short prefix present — strip and return content after it.
        if raw.hasPrefix(shortPrefix) {
            return String(raw.dropFirst(shortPrefix.count))
        }
        // Full long prefix present.
        if let range = raw.range(of: Self.harmonyAnalysisHeader) {
            return String(raw[range.upperBound...])
        }
        // Prefix is still assembling (e.g. "a", "anal", "analysis<|"); hide until complete.
        if shortPrefix.hasPrefix(raw) || Self.harmonyAnalysisHeader.hasPrefix(raw) {
            return ""
        }
        return raw
    }

    private static func normalizeHarmonyOutput(_ raw: String) -> (text: String, hasHarmonyMarkers: Bool) {
        let analysisHeader = Self.harmonyAnalysisHeader
        let analysisHeaderShort = "analysis<|message|>"  // <|channel|> is non-rendering
        let endTag = Self.harmonyEndTag
        let finalHeader = Self.harmonyFinalHeader

        // Suppress partial Harmony prefixes that are still assembling token-by-token.
        // Without this, early chunks like "<|sta" or "<|start|>assi" would pass through
        // unrecognized and leak into the display / TTS.
        if !raw.isEmpty {
            let knownPrefixes = [
                Self.harmonyFinalHeader,              // <|start|>assistant<|channel|>final<|message|>
                Self.harmonyAssistantHeader + "<|channel|>analysis<|message|>",
                Self.harmonyAssistantHeader + "analysis<|message|>",
                Self.harmonyAssistantHeader + "final<|message|>",
                Self.harmonyAssistantHeader,          // <|start|>assistant
                analysisHeader,                       // <|channel|>analysis<|message|>
                analysisHeaderShort,                  // analysis<|message|>
            ]
            for pfx in knownPrefixes {
                if pfx.hasPrefix(raw) && raw.count < pfx.count {
                    return ("", true)
                }
            }
        }

        // Handle rendered short form: stream starts with analysis<|message|>THINKING
        if raw.hasPrefix(analysisHeaderShort) && !raw.hasPrefix(analysisHeader) {
            let analysisBody = raw.dropFirst(analysisHeaderShort.count)
            if let endRange = analysisBody.range(of: endTag) {
                let thinking = String(analysisBody[..<endRange.lowerBound])
                let afterEnd = String(analysisBody[endRange.upperBound...])
                if let finalRange = afterEnd.range(of: finalHeader) {
                    let answer = String(afterEnd[finalRange.upperBound...])
                    return (Self.thinkingSentinelOpen + thinking + Self.thinkingSentinelClose + answer, true)
                }
                return (Self.thinkingSentinelOpen + thinking + Self.thinkingSentinelClose, true)
            }
            return (Self.thinkingSentinelOpen + String(analysisBody), true)
        }

        if let analysisRange = raw.range(of: analysisHeader) {
            let analysisBody = raw[analysisRange.upperBound...]
            if let endRange = analysisBody.range(of: endTag) {
                let thinking = String(analysisBody[..<endRange.lowerBound])
                let afterEnd = String(analysisBody[endRange.upperBound...])
                if let finalRange = afterEnd.range(of: finalHeader) {
                    let answer = String(afterEnd[finalRange.upperBound...])
                    return (Self.thinkingSentinelOpen + thinking + Self.thinkingSentinelClose + answer, true)
                }
                return (Self.thinkingSentinelOpen + thinking + Self.thinkingSentinelClose, true)
            }
            return (Self.thinkingSentinelOpen + String(analysisBody), true)
        }

        if let finalRange = raw.range(of: finalHeader) {
            return (String(raw[finalRange.upperBound...]), true)
        }
        // <|channel|> may be non-rendering: "<|start|>assistantfinal<|message|>ANSWER"
        let shortFinalHeader = Self.harmonyAssistantHeader + "final<|message|>"
        if let shortFinalRange = raw.range(of: shortFinalHeader) {
            return (String(raw[shortFinalRange.upperBound...]), true)
        }

        if let assistantRange = raw.range(of: Self.harmonyAssistantHeader) {
            let remainder = String(raw[assistantRange.upperBound...])
            if remainder.isEmpty {
                return ("", true)
            }
            // <|channel|> may be non-rendering, so remainder starts with "analysis<|message|>..."
            let shortAnalysis = Self.harmonyAnalysisPrefixShort
            if remainder.hasPrefix(shortAnalysis) {
                let body = String(remainder.dropFirst(shortAnalysis.count))
                let endTag = Self.harmonyEndTag
                if let endRange = body.range(of: endTag) {
                    let thinking = String(body[..<endRange.lowerBound])
                    let afterEnd = String(body[endRange.upperBound...])
                    if let fRange = afterEnd.range(of: Self.harmonyFinalHeader) {
                        return (Self.thinkingSentinelOpen + thinking + Self.thinkingSentinelClose + String(afterEnd[fRange.upperBound...]), true)
                    }
                    // Also handle final header without <|channel|> (non-rendering)
                    let shortFinal = "<|start|>assistant" + "final<|message|>"
                    if let fRange = afterEnd.range(of: shortFinal) {
                        return (Self.thinkingSentinelOpen + thinking + Self.thinkingSentinelClose + String(afterEnd[fRange.upperBound...]), true)
                    }
                    return (Self.thinkingSentinelOpen + thinking + Self.thinkingSentinelClose, true)
                }
                return (Self.thinkingSentinelOpen + body, true)
            }
            // Check for "final<|message|>" (non-rendering <|channel|> before it)
            let shortFinal = "final<|message|>"
            if remainder.hasPrefix(shortFinal) {
                return (String(remainder.dropFirst(shortFinal.count)), true)
            }
            return (remainder, true)
        }

        return (raw, false)
    }

    private static func cleanGemma4Output(_ raw: String) -> String {
        let startTokens = ["<|channel|>thought", "<|channel>thought"]

        // 1. If the raw string is a prefix of any start token, hide it (prevent flashing).
        // Also suppress a bare leading "thought" — the <|channel|> special token can decode as
        // empty string in llama.cpp, making the stream start with just "thought" instead of
        // "<|channel|>thought". This check is safe because it only fires at position 0.
        if !raw.isEmpty {
            for pfx in startTokens {
                if pfx.hasPrefix(raw) && raw.count < pfx.count {
                    return ""
                }
            }
            // Bare "thought" at position 0 = <|channel|> decoded as empty
            if raw == "thought" {
                return ""
            }
        }

        // 2. Look for the start of the thought channel.
        var hasStartTag = false
        var startTagEndIndex: String.Index? = nil
        for tag in startTokens {
            if let range = raw.range(of: tag) {
                hasStartTag = true
                startTagEndIndex = range.upperBound
                break
            }
        }
        // Fallback: <|channel|> decoded as empty, so raw starts with bare "thought"
        if !hasStartTag && raw.hasPrefix("thought") {
            hasStartTag = true
            startTagEndIndex = raw.index(raw.startIndex, offsetBy: "thought".count)
        }
        
        var remainder: String
        
        if hasStartTag, let startIndex = startTagEndIndex {
            // 3. Look for any closing token after the start tag
            let closingTokens = [
                "<channel|>",
                "<|channel|>",
                "<|channel|>text",
                "<|channel>text",
                "<|turn|>model",
                "<|turn>model"
            ]
            
            let searchArea = raw[startIndex...]
            var firstCloseRange: Range<String.Index>? = nil
            
            for tag in closingTokens {
                if let range = searchArea.range(of: tag) {
                    if firstCloseRange == nil || range.lowerBound < firstCloseRange!.lowerBound {
                        firstCloseRange = range
                    }
                }
            }
            
            if let closeRange = firstCloseRange {
                // Thought channel is closed; get everything after the closing token
                remainder = String(raw[closeRange.upperBound...])
            } else {
                // Thought channel is open and still streaming; hide all output
                return ""
            }
        } else {
            remainder = raw
        }
        
        // 4. Strip any intermediate or stray channel/header tokens from the remainder
        let tokensToRemove = [
            "<|channel|>text",
            "<|channel>text",
            "<channel|>",
            "<|channel|>",
            "<|turn|>model",
            "<|turn>model",
            "<turn|>"
        ]
        for tok in tokensToRemove {
            remainder = remainder.replacingOccurrences(of: tok, with: "")
        }
        
        // 5. If the remainder ends with a prefix of any special token, strip it from the end (prevent flashing)
        let allSpecialTokens = [
            "<|channel|>thought",
            "<|channel>thought",
            "<channel|>",
            "<|channel|>",
            "<|channel|>text",
            "<|channel>text",
            "<|turn|>model",
            "<|turn>model",
            "<turn|>",
            "<end_of_turn>",
            "</s>",
            "<eos>"
        ]
        for token in allSpecialTokens {
            for len in (1...token.count).reversed() {
                let prefixOfToken = String(token.prefix(len))
                if remainder.hasSuffix(prefixOfToken) {
                    remainder = String(remainder.dropLast(len))
                    break
                }
            }
        }
        
        return remainder.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func legacyModelDirectory(for model: AIModel) -> URL? {
        guard let documentsDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else { return nil }
        return documentsDir.appendingPathComponent("models").appendingPathComponent(model.id)
    }

    private func hasAllRequiredFiles(in directory: URL, for model: AIModel) -> Bool {
        guard !model.requiredFileNames.isEmpty else { return false }
        return model.requiredFileNames.allSatisfy { fileName in
            FileManager.default.fileExists(atPath: directory.appendingPathComponent(fileName).path)
        }
    }

    private func installedModelDirectory(for model: AIModel) -> URL? {
        try? SimplifiedFileManager.shared.getModelFolderURL(modelId: model.id, framework: model.inferenceFramework)
    }

    private func isModelAvailableLocally(_ model: AIModel) -> Bool {
        // Custom imported models store their file at model.url (an absolute file path).
        if model.source == "Custom", FileManager.default.fileExists(atPath: model.url) {
            return true
        }

        if let runAnywhereDir = installedModelDirectory(for: model),
           FileManager.default.fileExists(atPath: runAnywhereDir.path),
           hasAllRequiredFiles(in: runAnywhereDir, for: model) {
            return true
        }

        if let legacyDir = legacyModelDirectory(for: model),
           FileManager.default.fileExists(atPath: legacyDir.path),
           hasAllRequiredFiles(in: legacyDir, for: model) {
            return true
        }

        return false
    }

    private func filename(from url: URL) -> String {
        URLComponents(url: url, resolvingAgainstBaseURL: false)?.path.split(separator: "/").last.map(String.init) ?? url.lastPathComponent
    }

    private func loadedAIModel() -> AIModel? {
        guard let modelName = currentlyLoadedModel else { return nil }
        if let model = ModelData.allModels().first(where: { $0.name == modelName }) {
            return model
        }
        if modelName == "Apple Foundation Model" {
            #if canImport(FoundationModels)
            if #available(iOS 26.0, *) {
                return AIModel(
                    id: "apple.foundation.system",
                    name: "Apple Foundation Model",
                    description: "On-device Apple Intelligence foundation model.",
                    url: "apple://foundation-model",
                    category: .text,
                    sizeBytes: 0,
                    source: "Apple",
                    supportsVision: false,
                    supportsAudio: false,
                    supportsThinking: false,
                    supportsGpu: true,
                    requirements: ModelRequirements(minRamGB: 8, recommendedRamGB: 8),
                    contextWindowSize: 4096,
                    modelFormat: .platform,
                    additionalFiles: []
                )
            }
            #endif
        }
        return nil
    }

    func listGGUFFiles(in directory: URL) -> [URL] {
        guard let contents = try? FileManager.default.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        ) else {
            return []
        }

        return contents
            .filter { $0.pathExtension.lowercased() == "gguf" }
            .sorted { $0.lastPathComponent.lowercased() < $1.lastPathComponent.lowercased() }
    }

    func ggufFileURL(for model: AIModel) -> URL? {
        guard model.modelFormat == .gguf,
              let path = try? resolveModelGGUFPath(for: model) else { return nil }
        return URL(fileURLWithPath: path)
    }

    private func resolveModelGGUFPath(for model: AIModel) throws -> String {
        // Custom imported models store the GGUF path directly in model.url.
        if model.source == "Custom" {
            // Safety: if model.url somehow points to an mmproj file, find the real main model
            // in the same directory instead (mmproj/CLIP files can't be loaded as main models).
            if model.url.lowercased().contains("mmproj") {
                let directory = URL(fileURLWithPath: model.url).deletingLastPathComponent()
                if let mainModel = listGGUFFiles(in: directory).first(where: {
                    let name = $0.lastPathComponent.lowercased()
                    return !name.contains("mmproj") && !name.contains("projector")
                }) {
                    return mainModel.path
                }
            }
            guard FileManager.default.fileExists(atPath: model.url) else {
                throw NSError(domain: "LLMBackend", code: -101, userInfo: [NSLocalizedDescriptionKey: "Custom model file missing: \(model.url)"])
            }
            return model.url
        }

        let folderURL = try SimplifiedFileManager.shared.getModelFolderURL(modelId: model.id, framework: model.inferenceFramework)
        let files = listGGUFFiles(in: folderURL)

        if let modelURL = URL(string: model.url) {
            let preferredFilename = filename(from: modelURL).lowercased()
            if let exact = files.first(where: { $0.lastPathComponent.lowercased() == preferredFilename }) {
                return exact.path
            }
        }

        if let quantTag = quantizationTag(from: model.name),
           let quantMatched = files.first(where: {
               let lower = $0.lastPathComponent.lowercased()
               return !lower.contains("mmproj") && lower.contains(quantTag)
           }) {
            return quantMatched.path
        }

        if let preferred = files.first(where: { !$0.lastPathComponent.lowercased().contains("mmproj") }) {
            return preferred.path
        }

        if let first = files.first {
            return first.path
        }

        throw NSError(domain: "LLMBackend", code: -101, userInfo: [NSLocalizedDescriptionKey: "Main GGUF file not found for model \(model.name)"])
    }

    private func quantizationTag(from modelName: String) -> String? {
        guard let leftParen = modelName.lastIndex(of: "("),
              let rightParen = modelName.lastIndex(of: ")"),
              leftParen < rightParen else {
            return nil
        }
        let tag = modelName[modelName.index(after: leftParen)..<rightParen]
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        return tag.isEmpty ? nil : tag
    }

    private func familyStem(from modelName: String) -> String {
        modelName
            .replacingOccurrences(of: "\\s*\\([^)]*\\)\\s*$", with: "", options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
    }

    private func resolveVisionProjectorPath(for model: AIModel) -> String? {
        // Custom imported models store the vision projector path in additionalFiles.
        if model.source == "Custom" {
            let mmprojPath = model.additionalFiles.first {
                let lower = $0.lowercased()
                return lower.contains("mmproj") || lower.hasSuffix(".gguf")
            }
            if let path = mmprojPath, FileManager.default.fileExists(atPath: path) {
                return path
            }
            return nil
        }

        let stem = familyStem(from: model.name)
        let quantTag = quantizationTag(from: model.name)

        let allDependencyModels = ModelData.allModels().filter { $0.isDependencyOnly && $0.inferenceFramework == model.inferenceFramework }
        let candidates = allDependencyModels.filter { isModelAvailableLocally($0) }

        print("🔍 [LLMBackend] resolveVisionProjectorPath model=\(model.name) stem='\(stem)' quantTag=\(quantTag ?? "nil") totalDeps=\(allDependencyModels.count) downloadedDeps=\(candidates.count) downloadedNames=\(candidates.map(\.name))")

        let scored = candidates.compactMap { candidate -> (score: Int, path: String)? in
            let candidateName = candidate.name.lowercased()
            var score = 0

            if candidateName.contains(stem) {
                score += 3
            }
            if let quantTag,
               candidateName.contains(quantTag) || candidate.url.lowercased().contains(quantTag) {
                score += 3
            }
            if candidateName.contains("vision projector") || candidateName.contains("mmproj") {
                score += 1
            }

            guard let folderURL = try? SimplifiedFileManager.shared.getModelFolderURL(modelId: candidate.id, framework: candidate.inferenceFramework) else {
                return nil
            }

            let files = listGGUFFiles(in: folderURL)
            guard let mmprojFile = files.first(where: { $0.lastPathComponent.lowercased().contains("mmproj") }) ?? files.first else {
                return nil
            }

            return (score, mmprojFile.path)
        }
        .sorted { lhs, rhs in
            if lhs.score == rhs.score {
                return lhs.path < rhs.path
            }
            return lhs.score > rhs.score
        }

        print("🔍 [LLMBackend] resolveVisionProjectorPath scored=\(scored.map { "score=\($0.score) path=\($0.path)" })")
        // Require a minimum score of 3 (stem match) to avoid cross-family projector matches.
        // A score of 1 (only mmproj keyword) is not enough — the projector must belong to the same model family.
        return scored.first(where: { $0.score >= 3 })?.path
    }

    func isVisionProjectorAvailable(for model: AIModel) -> Bool {
        if model.modelFormat == .litertlm {
            return true
        }
        let path = resolveVisionProjectorPath(for: model)
        print("🔍 [LLMBackend] isVisionProjectorAvailable model=\(model.name) result=\(path != nil) path=\(path ?? "nil")")
        return path != nil
    }

#if canImport(UIKit)
    private func downsampledUIImage(from imageURL: URL, maxDimension: CGFloat = 448) -> UIImage? {
        let sourceOptions = [kCGImageSourceShouldCache: false] as CFDictionary
        guard let source = CGImageSourceCreateWithURL(imageURL as CFURL, sourceOptions) else {
            return nil
        }

        let downsampleOptions = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceShouldCacheImmediately: false,
            kCGImageSourceThumbnailMaxPixelSize: maxDimension,
        ] as CFDictionary

        guard let cgImage = CGImageSourceCreateThumbnailAtIndex(source, 0, downsampleOptions) else {
            return nil
        }

        return UIImage(cgImage: cgImage)
    }
#endif

    func modelMaxContextWindow(for model: AIModel) -> Int {
        if model.modelFormat == .gguf,
           let url = ggufFileURL(for: model),
           let fileContext = GGUFLayerLimits.readContextLength(from: url) {
            return fileContext
        }
        let advertised = model.contextWindowSize > 0 ? model.contextWindowSize : 2048
        return max(1, advertised)
    }

    func clampedContextWindow(_ requested: Int, for model: AIModel) -> Int {
        min(max(1, requested), modelMaxContextWindow(for: model))
    }

    func loadModel(_ model: AIModel) async throws {
        isBackendLoading = true
        defer { isBackendLoading = false }

        print("ℹ️ [LLMBackend] loadModel name=\(model.name) visionEnabled=\(enableVision) audioEnabled=\(enableAudio)")

        #if canImport(LiteRTLM)
        await LiteRTLMBackend.shared.unload()
        #endif
        if model.modelFormat != .gguf {
            await DirectLlamaCppBackend.shared.unload()
        }
        self.isLoaded = false
        self.currentlyLoadedModel = nil
        self.loadedContextWindow = nil

        if model.id == Self.appleFoundationAliasId {
            guard let nativeModel = appleFoundationModelIfAvailable() else {
                throw NSError(domain: "LLMBackend", code: -101, userInfo: [
                    NSLocalizedDescriptionKey: "Apple Intelligence is unavailable. Enable it in Settings and wait for its model to finish downloading."
                ])
            }
            isLoaded = true
            currentlyLoadedModel = model.name
            loadedContextWindow = nativeModel.contextWindowSize
            return
        }

        // ── LiteRT-LM path ──────────────────────────────────────────────────
        #if canImport(LiteRTLM)
        if model.modelFormat == .litertlm {
            guard isModelAvailableLocally(model) else {
                throw NSError(domain: "LLMBackend", code: -100,
                    userInfo: [NSLocalizedDescriptionKey: "Model is not downloaded locally"])
            }
            let filePath = try resolveLiteRTModelPath(for: model)
            let effectiveContext = clampedContextWindow(contextWindow, for: model)
            try await LiteRTLMBackend.shared.loadModel(
                at: filePath,
                modelName: model.name,
                supportsVision: model.supportsVision && enableVision,
                supportsAudio: model.supportsAudio && enableAudio,
                supportsGpu: model.supportsGpu,
                supportsMtp: model.supportsMtp,
                maxTokens: effectiveContext
            )
            isLoaded = true
            currentlyLoadedModel = model.name
            loadedContextWindow = effectiveContext
            return
        }
        #endif
        // ────────────────────────────────────────────────────────────────────

        if model.modelFormat == .gguf {
            guard isModelAvailableLocally(model) else {
                throw NSError(domain: "LLMBackend", code: -100, userInfo: [
                    NSLocalizedDescriptionKey: "Model is not downloaded locally"
                ])
            }
            let modelPath = try resolveModelGGUFPath(for: model)
            let projector = model.supportsVision && enableVision
                ? resolveVisionProjectorPath(for: model) : nil
            let contextSize = clampedContextWindow(contextWindow, for: model)
            let isCPU = selectedBackend.caseInsensitiveCompare("CPU") == .orderedSame
            let storedLayers = UserDefaults.standard.object(forKey: "gpu_layers_\(model.id)") != nil
                ? UserDefaults.standard.integer(forKey: "gpu_layers_\(model.id)") : 999
            let gpuLayers = isCPU ? 0 : max(0, storedLayers)
            try await DirectLlamaCppBackend.shared.load(
                path: modelPath, projector: projector,
                contextSize: contextSize, gpuLayers: gpuLayers
            )
            isLoaded = true
            currentlyLoadedModel = model.name
            loadedContextWindow = contextSize
            return
        }

        throw NSError(domain: "LLMBackend", code: -122, userInfo: [
            NSLocalizedDescriptionKey: "Unsupported language model format"
        ])
    }

    // MARK: - LiteRT-LM load path

    private func resolveLiteRTModelPath(for model: AIModel) throws -> String {
        let folderURL = try SimplifiedFileManager.shared.getModelFolderURL(
            modelId: model.id, framework: model.inferenceFramework
        )
        // The required file name is derived from the download URL (e.g. "gemma-4-E2B-it.litertlm")
        guard let fileName = model.requiredFileNames.first else {
            throw NSError(domain: "LLMBackend", code: -120,
                userInfo: [NSLocalizedDescriptionKey: "No required file name for LiteRT model \(model.name)"])
        }
        let filePath = folderURL.appendingPathComponent(fileName).path
        guard FileManager.default.fileExists(atPath: filePath) else {
            throw NSError(domain: "LLMBackend", code: -121,
                userInfo: [NSLocalizedDescriptionKey: "LiteRT model file not found: \(filePath)"])
        }
        return filePath
    }

    func unloadModel() {
        Task {
            await DirectLlamaCppBackend.shared.unload()
            #if canImport(LiteRTLM)
            await LiteRTLMBackend.shared.unload()
            #endif
            await MainActor.run {
                self.isLoaded = false
                self.currentlyLoadedModel = nil
                self.loadedContextWindow = nil
            }
        }
    }

    func generate(
        prompt: String,
        imageURL: URL? = nil,
        audioURL: URL? = nil,
        systemPrompt: String? = nil,
        maxTokensOverride: Int? = nil,
        stopSequences: [String] = [],
        enableAgentToolsOverride: Bool? = nil,
        onUpdate: @escaping (String, Int, Double) -> Void
    ) async throws {
        if currentlyLoadedModel == "Apple Foundation Model" {
            guard isLoaded else { throw CancellationError() }
            try await generateAppleFoundationResponse(
                prompt: prompt,
                systemPrompt: systemPrompt,
                temperature: Double(temperature),
                maxTokens: max(1, maxTokensOverride ?? maxTokens),
                onUpdate: onUpdate
            )
            return
        }

        // ── LiteRT-LM path ──────────────────────────────────────────────────
        #if canImport(LiteRTLM)
        if let model = loadedAIModel(), model.modelFormat == .litertlm {
            let effectiveMaxTokens: Int = {
                if let override = maxTokensOverride { return max(1, override) }
                return max(1, maxTokens)
            }()
            _ = effectiveMaxTokens // LiteRT-LM respects SamplerConfig; maxTokens passed for parity
            try await LiteRTLMBackend.shared.generateStream(
                prompt: prompt,
                imageURL: enableVision ? imageURL : nil,
                audioURL: enableAudio ? audioURL : nil,
                systemPrompt: systemPrompt,
                temperature: temperature,
                topK: topK,
                topP: topP,
                maxTokens: effectiveMaxTokens,
                useThinking: model.supportsThinking && enableThinking,
                enableAgentTools: (enableAgentToolsOverride ?? enableAgentTools) && model.name.contains("Gemma 4") && !model.name.contains("Translate") && model.modelFormat == .litertlm,
                onUpdate: onUpdate
            )
            return
        }
        #endif
        // ────────────────────────────────────────────────────────────────────

        _ = audioURL

        let effectiveMaxTokens: Int = {
            if let override = maxTokensOverride { return max(1, override) }
            if let model = loadedAIModel() {
                let effectiveContext = clampedContextWindow(contextWindow, for: model)
                return min(max(1, maxTokens), effectiveContext)
            }
            return max(1, maxTokens)
        }()

        let loadedModel = loadedAIModel()
        let loadedModelName = currentlyLoadedModel ?? loadedModel?.name ?? "<nil>"
        let modelSupportsThinking = loadedModel?.supportsThinking == true
        let isHarmonyModel = Self.isHarmonyModelName(loadedModelName)
        let isMuseGlimmerModel = Self.isMuseGlimmerModelName(loadedModelName)
        let isGranite42Model = Self.isGranite42ModelName(loadedModelName)
        let useMuseGlimmerThinking = modelSupportsThinking && enableThinking
        let rawPrompt: String
        if isHarmonyModel && !prompt.hasPrefix("__RAW_PROMPT__") {
            rawPrompt = Self.buildHarmonyPrompt(prompt: prompt, systemPrompt: systemPrompt, thinkingEnabled: enableThinking)
        } else if isMuseGlimmerModel && !prompt.hasPrefix("__RAW_PROMPT__") {
            rawPrompt = Self.buildMuseGlimmerPrompt(prompt: prompt, systemPrompt: systemPrompt, thinkingEnabled: useMuseGlimmerThinking)
        } else if isGranite42Model && !prompt.hasPrefix("__RAW_PROMPT__") {
            rawPrompt = Self.buildGranite42Prompt(prompt: prompt, systemPrompt: systemPrompt, thinkingEnabled: enableThinking)
        } else {
            rawPrompt = prompt
        }
        let effectiveSystemPrompt: String?
        if prompt.hasPrefix("__RAW_PROMPT__") {
            effectiveSystemPrompt = nil
        } else if isHarmonyModel || isMuseGlimmerModel || isGranite42Model {
            effectiveSystemPrompt = nil
        } else {
            effectiveSystemPrompt = systemPrompt
        }

        let isLfmModel = loadedModelName.contains("LFM2.5-8B-A1B") || loadedModelName.contains("LFM-2.5 2.6B") || loadedModelName.contains("LFM-2.5 1.2B Thinking")
        let isPhi4MiniModel = loadedModelName.lowercased().contains("phi-4") || loadedModelName.lowercased().contains("phi 4") || loadedModelName.lowercased().contains("phi4")
        var usePrompt: String
        do {
            let strippedPrompt: String
            if rawPrompt.hasPrefix("__RAW_PROMPT__\n") {
                strippedPrompt = String(rawPrompt.dropFirst("__RAW_PROMPT__\n".count))
            } else if rawPrompt.hasPrefix("__RAW_PROMPT__") {
                strippedPrompt = String(rawPrompt.dropFirst("__RAW_PROMPT__".count))
            } else {
                strippedPrompt = rawPrompt
            }

            if isLfmModel {
                if !strippedPrompt.contains("<|im_start|>") && !strippedPrompt.contains("[INST]") {
                    let sys = (effectiveSystemPrompt ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
                    if !sys.isEmpty {
                        usePrompt = "<|im_start|>system\n\(sys)\n<|im_end|>\n<|im_start|>user\n\(strippedPrompt)\n<|im_end|>\n<|im_start|>assistant\n<think>\n"
                    } else {
                        usePrompt = "<|im_start|>user\n\(strippedPrompt)\n<|im_end|>\n<|im_start|>assistant\n<think>\n"
                    }
                } else if !strippedPrompt.contains("<think>") {
                    usePrompt = strippedPrompt + "\n<think>\n"
                } else {
                    usePrompt = strippedPrompt
                }
            } else if isPhi4MiniModel {
                if !strippedPrompt.contains("<|user|>") && !strippedPrompt.contains("<|system|>") && !strippedPrompt.contains("[INST]") {
                    let sys = (effectiveSystemPrompt ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
                    if !sys.isEmpty {
                        usePrompt = "<|system|>\n\(sys)<|end|>\n<|user|>\n\(strippedPrompt)<|end|>\n<|assistant|>\n"
                    } else {
                        usePrompt = "<|user|>\n\(strippedPrompt)<|end|>\n<|assistant|>\n"
                    }
                } else {
                    usePrompt = strippedPrompt
                }
            } else if isHarmonyModel && strippedPrompt.contains("<|start|>") && !strippedPrompt.contains("<|begin_of_text|>") {
                usePrompt = "<|begin_of_text|>" + strippedPrompt
            } else if isMuseGlimmerModel {
                if strippedPrompt.contains("<|start|>") && strippedPrompt.contains("<|begin_of_text|>") {
                    usePrompt = strippedPrompt
                } else {
                    usePrompt = Self.buildMuseGlimmerPrompt(
                        prompt: strippedPrompt,
                        systemPrompt: effectiveSystemPrompt,
                        thinkingEnabled: useMuseGlimmerThinking,
                        includeRawPrefix: false
                    )
                }
            } else if isGranite42Model {
                if !strippedPrompt.contains("<|im_start|>") {
                    let sys = (effectiveSystemPrompt ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
                    let thinkTag = enableThinking ? "<think>\n" : "<think></think>"
                    if !sys.isEmpty {
                        usePrompt = "<|im_start|>system\n\(sys)\n<|im_end|>\n<|im_start|>user\n\(strippedPrompt)\n<|im_end|>\n<|im_start|>assistant\n\(thinkTag)"
                    } else {
                        usePrompt = "<|im_start|>user\n\(strippedPrompt)\n<|im_end|>\n<|im_start|>assistant\n\(thinkTag)"
                    }
                } else {
                    usePrompt = strippedPrompt
                }
            } else {
                usePrompt = strippedPrompt
            }
        }

        if let model = loadedModel, model.modelFormat == .gguf {
            var imageData: Data?
            if let imageURL, enableVision, model.supportsVision {
                guard let projector = resolveVisionProjectorPath(for: model) else {
                    throw NSError(domain: "LLMBackend", code: -111, userInfo: [
                        NSLocalizedDescriptionKey: "The vision projector is not downloaded"
                    ])
                }
                let path = try resolveModelGGUFPath(for: model)
                let isCPU = selectedBackend.caseInsensitiveCompare("CPU") == .orderedSame
                let layers = UserDefaults.standard.object(forKey: "gpu_layers_\(model.id)") != nil
                    ? UserDefaults.standard.integer(forKey: "gpu_layers_\(model.id)") : 999
                try await DirectLlamaCppBackend.shared.load(
                    path: path, projector: projector,
                    contextSize: clampedContextWindow(contextWindow, for: model),
                    gpuLayers: isCPU ? 0 : max(0, layers)
                )
                #if canImport(UIKit)
                // Keep vision input at 448 pixels. Sending the full-resolution photo makes
                // llama.cpp split tall images into many costly 512px vision slices.
                if let thumbnail = downsampledUIImage(from: imageURL),
                   let encoded = thumbnail.jpegData(compressionQuality: 0.95) {
                    imageData = encoded
                } else {
                    imageData = try Data(contentsOf: imageURL)
                }
                #else
                imageData = try Data(contentsOf: imageURL)
                #endif
            }

            let effectiveStops = stopSequences.isEmpty && isPhi4MiniModel
                ? ["<|end|>", "<|user|>", "<|system|>"]
                : stopSequences.isEmpty && isMuseGlimmerModel ? ["<|eot|>"]
                : stopSequences.isEmpty && isGranite42Model
                    ? ["<|im_end|>", "<|im_start|>user", "<|im_start|>system"]
                    : stopSequences
            let stream = await DirectLlamaCppBackend.shared.stream(
                prompt: usePrompt, imageData: imageData,
                maxTokens: effectiveMaxTokens, temperature: temperature,
                topK: topK, topP: topP, stopSequences: effectiveStops
            )
            for try await update in stream {
                try Task.checkCancellation()
                let gemmaText = (loadedModelName.range(of: "gemma 4", options: .caseInsensitive) != nil ||
                                 loadedModelName.range(of: "gemma-4", options: .caseInsensitive) != nil)
                    ? Self.cleanGemma4Output(update.text) : update.text
                let museText = isMuseGlimmerModel
                    ? Self.normalizeMuseGlimmerOutput(gemmaText, thinkingEnabled: enableThinking)
                    : gemmaText
                let display = isHarmonyModel ? Self.normalizeHarmonyOutput(museText).0 : museText
                onUpdate(display, update.completionTokens, update.tokensPerSecond)
            }
            return
        }

        throw NSError(domain: "LLMBackend", code: -122, userInfo: [
            NSLocalizedDescriptionKey: "Unsupported language model format"
        ])
    }
}
