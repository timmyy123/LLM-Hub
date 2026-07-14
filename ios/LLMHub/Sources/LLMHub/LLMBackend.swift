import Foundation
import LlamaCPPRuntime
import RunAnywhere
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
    private static let runAnywhereFoundationModelId = "foundation-models-default"

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

    private var isSDKInitialized = false
    private var areModelsRegistered = false
    private var loadedLLMModelId: String?
    private var loadedVLMModelId: String?
    private var loadedVLMProjectorPath: String?

    private init() {}

    private struct HarmonyPromptMessage {
        let role: String
        let content: String
    }

    private static func isHarmonyModelName(_ modelName: String?) -> Bool {
        guard let normalized = modelName?.lowercased() else { return false }
        return normalized.contains("gpt-oss") || normalized.contains("gpt_oss")
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

        if let assistantRange = raw.range(of: Self.harmonyAssistantHeader) {
            let remainder = String(raw[assistantRange.upperBound...])
            if remainder.isEmpty {
                return ("", true)
            }
            return (remainder, true)
        }

        return (raw, false)
    }

    private static func cleanGemma4Output(_ raw: String) -> String {
        let startTokens = ["<|channel|>thought", "<|channel>thought"]
        
        // 1. If the raw string is a prefix of any start token, hide it (prevent flashing)
        if !raw.isEmpty {
            for pfx in startTokens {
                if pfx.hasPrefix(raw) && raw.count < pfx.count {
                    return ""
                }
            }
        }
        
        // 2. Look for the start of the thought channel
        var hasStartTag = false
        var startTagEndIndex: String.Index? = nil
        for tag in startTokens {
            if let range = raw.range(of: tag) {
                hasStartTag = true
                startTagEndIndex = range.upperBound
                break
            }
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
            "<|turn>model"
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

    private func runAnywhereModelDirectory(for model: AIModel) -> URL? {
        try? SimplifiedFileManager.shared.getModelFolderURL(modelId: model.id, framework: model.inferenceFramework)
    }

    private func isModelAvailableLocally(_ model: AIModel) -> Bool {
        // Custom imported models store their file at model.url (an absolute file path).
        if model.source == "Custom", FileManager.default.fileExists(atPath: model.url) {
            return true
        }

        if let runAnywhereDir = runAnywhereModelDirectory(for: model),
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

    private func migrateLegacyModelIfNeeded(_ model: AIModel) throws -> Bool {
        if isModelAvailableLocally(model),
           let runAnywhereDir = runAnywhereModelDirectory(for: model),
           FileManager.default.fileExists(atPath: runAnywhereDir.path),
           hasAllRequiredFiles(in: runAnywhereDir, for: model) {
            return false
        }

        guard let legacyDir = legacyModelDirectory(for: model),
              FileManager.default.fileExists(atPath: legacyDir.path),
              hasAllRequiredFiles(in: legacyDir, for: model) else {
            return false
        }

        let destinationDir = try SimplifiedFileManager.shared.getModelFolderURL(modelId: model.id, framework: model.inferenceFramework)
        try FileManager.default.createDirectory(at: destinationDir, withIntermediateDirectories: true)

        for fileName in model.requiredFileNames {
            let sourceURL = legacyDir.appendingPathComponent(fileName)
            let destinationURL = destinationDir.appendingPathComponent(fileName)

            if FileManager.default.fileExists(atPath: destinationURL.path) {
                try? FileManager.default.removeItem(at: destinationURL)
            }
            try FileManager.default.copyItem(at: sourceURL, to: destinationURL)
        }

        print("ℹ️ [LLMBackend] Migrated legacy model files for \(model.id)")
        return true
    }

    private func migrateCustomModelIfNeeded(_ model: AIModel) throws -> Bool {
        guard model.source == "Custom",
              let destinationDir = runAnywhereModelDirectory(for: model) else {
            return false
        }

        try FileManager.default.createDirectory(at: destinationDir, withIntermediateDirectories: true)

        var copiedAny = false
        let mainSourceURL = URL(fileURLWithPath: model.url)
        let mainDestinationURL = destinationDir.appendingPathComponent(mainSourceURL.lastPathComponent)
        if FileManager.default.fileExists(atPath: mainSourceURL.path),
           !FileManager.default.fileExists(atPath: mainDestinationURL.path) {
            try FileManager.default.copyItem(at: mainSourceURL, to: mainDestinationURL)
            copiedAny = true
        }

        for filePath in model.additionalFiles {
            let sourceURL = URL(fileURLWithPath: filePath)
            let destinationURL = destinationDir.appendingPathComponent(sourceURL.lastPathComponent)
            if FileManager.default.fileExists(atPath: sourceURL.path),
               !FileManager.default.fileExists(atPath: destinationURL.path) {
                try FileManager.default.copyItem(at: sourceURL, to: destinationURL)
                copiedAny = true
            }
        }

        if copiedAny {
            print("ℹ️ [LLMBackend] Migrated custom model files into RunAnywhere storage for \(model.id)")
        }
        return copiedAny
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

    private func framework(for model: AIModel) -> InferenceFramework {
        model.inferenceFramework
    }

    private func isAppleFoundationAlias(_ model: AIModel) -> Bool {
        model.id == Self.appleFoundationAliasId
    }

    private func activeRunAnywhereModelId(for model: AIModel) -> String {
        isAppleFoundationAlias(model) ? Self.runAnywhereFoundationModelId : model.id
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

    private func ensureVLMLoaded(for model: AIModel) async throws {
        let modelPath = try resolveModelGGUFPath(for: model)
        let mmprojPath = resolveVisionProjectorPath(for: model)

        guard let mmprojPath, !mmprojPath.isEmpty else {
            throw NSError(
                domain: "LLMBackend",
                code: -102,
                userInfo: [NSLocalizedDescriptionKey: "Vision projector (mmproj) is missing for \(model.name)"]
            )
        }

        let modelSummary = fileSummary(at: modelPath)
        let projectorSummary = fileSummary(at: mmprojPath)
        let modelHeader = ggufHeaderSummary(at: modelPath)
        let projectorHeader = ggufHeaderSummary(at: mmprojPath)

        print("ℹ️ [LLMBackend] VLM prepare model=\(model.id) main={\(modelSummary)} header={\(modelHeader)} mmproj={\(projectorSummary)} header={\(projectorHeader)}")

        guard modelHeader == "GGUF" else {
            throw NSError(
                domain: "LLMBackend",
                code: -103,
                userInfo: [NSLocalizedDescriptionKey: "Main model file is invalid: \(modelSummary) header=\(modelHeader)"]
            )
        }

        guard projectorHeader == "GGUF" else {
            throw NSError(
                domain: "LLMBackend",
                code: -104,
                userInfo: [NSLocalizedDescriptionKey: "Vision projector file is invalid: \(projectorSummary) header=\(projectorHeader)"]
            )
        }

        let shouldReload = !(loadedVLMModelId == model.id && loadedVLMProjectorPath == mmprojPath)

        guard shouldReload else { return }

        if let loadedLLMModelId {
            var unloadRequest = RAModelUnloadRequest()
            unloadRequest.modelID = loadedLLMModelId
            unloadRequest.category = .language
            _ = await RunAnywhere.unloadModel(unloadRequest)
            self.loadedLLMModelId = nil
            print("ℹ️ [LLMBackend] Unloaded text LLM before VLM load to avoid duplicate model residency")
        }

        if let loadedVLMModelId {
            var unloadRequest = RAModelUnloadRequest()
            unloadRequest.modelID = loadedVLMModelId
            unloadRequest.category = .multimodal
            _ = await RunAnywhere.unloadModel(unloadRequest)
        }
        self.loadedVLMModelId = nil
        self.loadedVLMProjectorPath = nil

        await syncGpuLayersToRegistry(for: model)
        do {
            var loadRequest = RAModelLoadRequest()
            loadRequest.modelID = model.id
            loadRequest.category = .multimodal
            loadRequest.framework = framework(for: model)
            let loadResult = await RunAnywhere.loadModel(loadRequest)
            guard loadResult.success else {
                throw NSError(
                    domain: "LLMBackend",
                    code: -111,
                    userInfo: [NSLocalizedDescriptionKey: loadResult.errorMessage.isEmpty ? "VLM load failed" : loadResult.errorMessage]
                )
            }
        } catch {
            let details = "main={\(modelSummary)} header={\(modelHeader)} mmproj={\(projectorSummary)} header={\(projectorHeader)}"
            throw NSError(
                domain: "LLMBackend",
                code: -111,
                userInfo: [NSLocalizedDescriptionKey: "VLM load failed for \(model.name): \(error.localizedDescription). \(details)"]
            )
        }
        loadedVLMModelId = model.id
        loadedVLMProjectorPath = mmprojPath
    }

    private func ensureTextModelLoaded(for model: AIModel) async throws {
        if let loadedVLMModelId {
            var unloadRequest = RAModelUnloadRequest()
            unloadRequest.modelID = loadedVLMModelId
            unloadRequest.category = .multimodal
            _ = await RunAnywhere.unloadModel(unloadRequest)
            self.loadedVLMModelId = nil
            self.loadedVLMProjectorPath = nil
            print("ℹ️ [LLMBackend] Unloaded VLM before text generation to avoid duplicate model residency")
        }

        let shouldLoad = loadedLLMModelId != model.id
        guard shouldLoad else { return }

        var loadRequest = RAModelLoadRequest()
        loadRequest.modelID = model.id
        loadRequest.category = .language
        loadRequest.framework = framework(for: model)
        let loadResult = await RunAnywhere.loadModel(loadRequest)
        guard loadResult.success else {
            throw NSError(
                domain: "LLMBackend",
                code: -112,
                userInfo: [NSLocalizedDescriptionKey: loadResult.errorMessage.isEmpty ? "Text model load failed" : loadResult.errorMessage]
            )
        }
        loadedLLMModelId = model.id
    }

    private func syncGpuLayersToRegistry(for model: AIModel) async {
        let key = "gpu_layers_\(model.id)"
        let stored = UserDefaults.standard.object(forKey: key) != nil ?
            Int32(UserDefaults.standard.integer(forKey: key)) : 999
        let targetValue: Int32 = (stored == 99) ? 999 : stored

        await CppBridge.ModelRegistry.shared.setGpuLayers(modelId: model.id, gpuLayers: targetValue)
        let runAnywhereModelId = activeRunAnywhereModelId(for: model)
        await CppBridge.ModelRegistry.shared.setGpuLayers(modelId: runAnywhereModelId, gpuLayers: targetValue)

        if let folderURL = try? SimplifiedFileManager.shared.getModelFolderURL(modelId: model.id, framework: framework(for: model)) {
            await CppBridge.ModelRegistry.shared.setGpuLayers(modelId: folderURL.path, gpuLayers: targetValue)
            for ggufFile in listGGUFFiles(in: folderURL) {
                await CppBridge.ModelRegistry.shared.setGpuLayers(modelId: ggufFile.path, gpuLayers: targetValue)
            }
        }
        if let folderURL = runAnywhereModelDirectory(for: model) {
            await CppBridge.ModelRegistry.shared.setGpuLayers(modelId: folderURL.path, gpuLayers: targetValue)
            for ggufFile in listGGUFFiles(in: folderURL) {
                await CppBridge.ModelRegistry.shared.setGpuLayers(modelId: ggufFile.path, gpuLayers: targetValue)
            }
        }
    }

    private func fileSummary(at path: String) -> String {
        let url = URL(fileURLWithPath: path)
        let name = url.lastPathComponent

        guard FileManager.default.fileExists(atPath: path) else {
            return "\(name) missing"
        }

        let attributes = try? FileManager.default.attributesOfItem(atPath: path)
        let size = (attributes?[.size] as? NSNumber)?.int64Value ?? -1
        let formatter = ByteCountFormatter()
        formatter.allowedUnits = [.useGB, .useMB, .useKB]
        formatter.countStyle = .file
        let sizeLabel = size >= 0 ? formatter.string(fromByteCount: size) : "unknown"

        return "\(name) \(sizeLabel)"
    }

    private func ggufHeaderSummary(at path: String) -> String {
        guard let handle = FileHandle(forReadingAtPath: path) else { return "unreadable" }
        defer {
            try? handle.close()
        }

        let data = handle.readData(ofLength: 4)
        guard data.count == 4, let header = String(data: data, encoding: .ascii) else {
            return "invalid"
        }
        return header
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

    private func vlmImage(from imageURL: URL) -> RAVLMImage {
        #if canImport(UIKit)
        if let uiImage = downsampledUIImage(from: imageURL) {
            return RAVLMImage.fromUIImage(uiImage) ?? RAVLMImage.fromFilePath(imageURL.path)
        }
        #endif
        return RAVLMImage.fromFilePath(imageURL.path)
    }

    private func modelMaxContextWindow(for model: AIModel) -> Int {
        let advertised = model.contextWindowSize > 0 ? model.contextWindowSize : 2048
        return max(1, advertised)
    }

    func clampedContextWindow(_ requested: Int, for model: AIModel) -> Int {
        min(max(1, requested), modelMaxContextWindow(for: model))
    }

    private func registerModel(_ model: AIModel, contextLengthOverride: Int? = nil) async throws {
        guard model.modelFormat == .gguf || model.modelFormat == .onnx else { return }
        
        // Custom models store an absolute file path in model.url — use file:// URL.
        let primaryURL: URL
        if model.source == "Custom" {
            primaryURL = URL(fileURLWithPath: model.url)
        } else {
            guard let url = URL(string: model.url) else { return }
            primaryURL = url
        }
        let contextLength = contextLengthOverride ?? model.contextWindowSize

        let urls: [URL]
        if model.source == "Custom" {
            urls = ([model.url] + model.additionalFiles).map { URL(fileURLWithPath: $0) }
        } else {
            urls = model.allDownloadURLs
        }

        if urls.count <= 1 {
            try await RunAnywhere.registerModel(
                id: model.id,
                name: model.name,
                url: primaryURL.absoluteString,
                framework: framework(for: model),
                modality: model.supportsVision ? .multimodal : .language,
                contextLength: contextLength,
                memoryRequirement: model.sizeBytes,
                supportsThinking: model.supportsThinking
            )
            return
        }

        let descriptors = urls.map {
            RAModelFileDescriptor(url: $0, filename: filename(from: $0), isRequired: true)
        }

        try await RunAnywhere.registerModel(
            multiFile: descriptors,
            id: model.id,
            name: model.name,
            framework: framework(for: model),
            modality: model.supportsVision ? .multimodal : .language,
            memoryRequirement: model.sizeBytes,
            contextLength: contextLength,
            supportsThinking: model.supportsThinking
        )
    }

    private func ensureSDKReady() async throws {
        if !isSDKInitialized {
            try RunAnywhere.initialize(environment: .development)
            LlamaCPP.register()
            isSDKInitialized = true
        }

        if !areModelsRegistered {
            for model in ModelData.allModels() {
                guard model.modelFormat == .gguf || model.modelFormat == .onnx else { continue }
                try await registerModel(model)
            }
            areModelsRegistered = true
        }

        // Ensure model path APIs are configured before storage checks/migration.
        try await RunAnywhere.completeServicesInitialization()
    }

    func loadModel(_ model: AIModel) async throws {
        isBackendLoading = true
        defer { isBackendLoading = false }

        print("ℹ️ [LLMBackend] loadModel name=\(model.name) visionEnabled=\(enableVision) audioEnabled=\(enableAudio)")

        #if canImport(LiteRTLM)
        await LiteRTLMBackend.shared.unload()
        #endif
        self.isLoaded = false
        self.currentlyLoadedModel = nil
        self.loadedContextWindow = nil
        self.loadedLLMModelId = nil
        self.loadedVLMModelId = nil
        self.loadedVLMProjectorPath = nil

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
                supportsVision: model.supportsVision && enableVision,
                supportsAudio: model.supportsAudio && enableAudio,
                maxTokens: effectiveContext
            )
            isLoaded = true
            currentlyLoadedModel = model.name
            loadedContextWindow = effectiveContext
            return
        }
        #endif
        // ────────────────────────────────────────────────────────────────────

        try await ensureSDKReady()
        let effectiveContext = clampedContextWindow(contextWindow, for: model)
        let runAnywhereModelId = activeRunAnywhereModelId(for: model)

        if let loadedLLMModelId {
            var unloadRequest = RAModelUnloadRequest()
            unloadRequest.modelID = loadedLLMModelId
            unloadRequest.category = .language
            _ = await RunAnywhere.unloadModel(unloadRequest)
        }
        if let loadedVLMModelId {
            var unloadRequest = RAModelUnloadRequest()
            unloadRequest.modelID = loadedVLMModelId
            unloadRequest.category = .multimodal
            _ = await RunAnywhere.unloadModel(unloadRequest)
        }

        if isAppleFoundationAlias(model) {
            // Apple Foundation model is built in; no download or registration required.
            var loadRequest = RAModelLoadRequest()
            loadRequest.modelID = runAnywhereModelId
            loadRequest.category = .language
            loadRequest.framework = .foundationModels
            let loadResult = await RunAnywhere.loadModel(loadRequest)
            guard loadResult.success else {
                throw NSError(domain: "LLMBackend", code: -101, userInfo: [NSLocalizedDescriptionKey: loadResult.errorMessage.isEmpty ? "Model load failed" : loadResult.errorMessage])
            }
        } else {
            try await registerModel(model, contextLengthOverride: effectiveContext)
            _ = try? migrateCustomModelIfNeeded(model)
            _ = try? migrateLegacyModelIfNeeded(model)

            // Only local load here. Downloads are handled by the model download screen.
            guard isModelAvailableLocally(model) else {
                throw NSError(domain: "LLMBackend", code: -100, userInfo: [NSLocalizedDescriptionKey: "Model is not downloaded locally"])
            }

            // The C++ backend looks up context_length from the registry using the absolute
            // file path as the identifier. Re-register the model under its absolute path so
            // the C++ ID lookup succeeds and uses effectiveContext (e.g. 2048) instead of
            // auto-detecting a larger value (e.g. 4096) which causes OOM on <8 GB devices.
            let useMultimodal = model.supportsVision && enableVision && isVisionProjectorAvailable(for: model)
            let loadCategory: RAModelCategory = useMultimodal ? .multimodal : .language

            // The C++ backend looks up context_length from the registry using the absolute
            // file path as the identifier. Re-register the model under its absolute path so
            // the C++ ID lookup succeeds and uses effectiveContext (e.g. 2048) instead of
            // auto-detecting a larger value (e.g. 4096) which causes OOM on <8 GB devices.
            if model.source == "Custom",
               let folderURL = runAnywhereModelDirectory(for: model),
               let ggufFile = listGGUFFiles(in: folderURL).first(where: {
                   let name = $0.lastPathComponent.lowercased()
                   return !name.contains("mmproj") && !name.contains("projector")
               }) {
                let ggufURL = ggufFile
                var registeredModelInfo = await CppBridge.ModelRegistry.shared.get(modelId: runAnywhereModelId) ?? ModelInfo(
                    id: runAnywhereModelId,
                    name: model.name,
                    category: loadCategory,
                    format: .gguf,
                    framework: framework(for: model),
                    downloadURL: ggufURL,
                    localPath: folderURL,
                    contextLength: Int32(effectiveContext),
                    supportsThinking: model.supportsThinking
                )
                registeredModelInfo.contextLength = Int32(effectiveContext)
                registeredModelInfo.category = loadCategory
                registeredModelInfo.setLocalPath(folderURL)
                try? await CppBridge.ModelRegistry.shared.save(registeredModelInfo)

                var pathModelInfo = await CppBridge.ModelRegistry.shared.get(modelId: ggufURL.path) ?? registeredModelInfo
                pathModelInfo.id = ggufURL.path
                pathModelInfo.contextLength = Int32(effectiveContext)
                pathModelInfo.category = loadCategory
                pathModelInfo.setLocalPath(folderURL)
                try? await CppBridge.ModelRegistry.shared.save(pathModelInfo)
            } else if let folderURL = try? SimplifiedFileManager.shared.getModelFolderURL(
                modelId: runAnywhereModelId,
                framework: framework(for: model)
            ), let ggufFile = listGGUFFiles(in: folderURL).first(where: {
                let name = $0.lastPathComponent.lowercased()
                return !name.contains("mmproj") && !name.contains("projector")
            }) {
                var registeredModelInfo = await CppBridge.ModelRegistry.shared.get(modelId: runAnywhereModelId) ?? ModelInfo(
                    id: runAnywhereModelId,
                    name: model.name,
                    category: loadCategory,
                    format: .gguf,
                    framework: framework(for: model),
                    downloadURL: URL(string: model.url),
                    localPath: folderURL,
                    contextLength: Int32(effectiveContext),
                    supportsThinking: model.supportsThinking
                )
                registeredModelInfo.contextLength = Int32(effectiveContext)
                registeredModelInfo.category = loadCategory
                registeredModelInfo.setLocalPath(folderURL)
                try? await CppBridge.ModelRegistry.shared.save(registeredModelInfo)

                var pathModelInfo = await CppBridge.ModelRegistry.shared.get(modelId: ggufFile.path) ?? registeredModelInfo
                pathModelInfo.id = ggufFile.path
                pathModelInfo.contextLength = Int32(effectiveContext)
                pathModelInfo.category = loadCategory
                pathModelInfo.setLocalPath(folderURL)
                try? await CppBridge.ModelRegistry.shared.save(pathModelInfo)
            }

            if useMultimodal,
               let folderURL = (model.source == "Custom" ? runAnywhereModelDirectory(for: model) : try? SimplifiedFileManager.shared.getModelFolderURL(modelId: runAnywhereModelId, framework: framework(for: model))),
               let projectorPath = resolveVisionProjectorPath(for: model) {
                let projectorSourceURL = URL(fileURLWithPath: projectorPath)
                let projectorDestURL = folderURL.appendingPathComponent(projectorSourceURL.lastPathComponent)
                print("🔍 [LLMBackend] Symlink check: source=\(projectorSourceURL.path), dest=\(projectorDestURL.path)")
                if projectorSourceURL.path != projectorDestURL.path {
                    do {
                        try FileManager.default.removeItem(at: projectorDestURL)
                        print("🔍 [LLMBackend] Removed old symlink/file at \(projectorDestURL.path)")
                    } catch {
                        print("🔍 [LLMBackend] Failed to remove at \(projectorDestURL.path): \(error)")
                    }
                    do {
                        let relativePath = "../" + projectorSourceURL.deletingLastPathComponent().lastPathComponent + "/" + projectorSourceURL.lastPathComponent
                        print("🔍 [LLMBackend] Creating symlink to relativePath=\(relativePath)")
                        try FileManager.default.createSymbolicLink(atPath: projectorDestURL.path, withDestinationPath: relativePath)
                        print("🔍 [LLMBackend] Symlink created successfully!")
                    } catch {
                        print("🔍 [LLMBackend] Failed to create symlink: \(error)")
                    }
                }
            }

            await syncGpuLayersToRegistry(for: model)
            var loadRequest = RAModelLoadRequest()
            loadRequest.modelID = runAnywhereModelId
            loadRequest.category = loadCategory
            loadRequest.framework = framework(for: model)
            let loadResult = await RunAnywhere.loadModel(loadRequest)
            guard loadResult.success else {
                throw NSError(domain: "LLMBackend", code: -101, userInfo: [NSLocalizedDescriptionKey: loadResult.errorMessage.isEmpty ? "Model load failed" : loadResult.errorMessage])
            }

            if useMultimodal {
                loadedVLMModelId = runAnywhereModelId
                loadedVLMProjectorPath = resolveVisionProjectorPath(for: model)
                loadedLLMModelId = nil
            } else {
                loadedLLMModelId = runAnywhereModelId
                loadedVLMModelId = nil
                loadedVLMProjectorPath = nil
            }
        }

        isLoaded = true
        currentlyLoadedModel = model.name
        loadedContextWindow = effectiveContext
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
            if let loadedLLMModelId {
                var unloadRequest = RAModelUnloadRequest()
                unloadRequest.modelID = loadedLLMModelId
                unloadRequest.category = .language
                _ = await RunAnywhere.unloadModel(unloadRequest)
            }
            if let loadedVLMModelId {
                var unloadRequest = RAModelUnloadRequest()
                unloadRequest.modelID = loadedVLMModelId
                unloadRequest.category = .multimodal
                _ = await RunAnywhere.unloadModel(unloadRequest)
            }
            #if canImport(LiteRTLM)
            await LiteRTLMBackend.shared.unload()
            #endif
            await MainActor.run {
                self.isLoaded = false
                self.currentlyLoadedModel = nil
                self.loadedContextWindow = nil
                self.loadedLLMModelId = nil
                self.loadedVLMModelId = nil
                self.loadedVLMProjectorPath = nil
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
        onUpdate: @escaping (String, Int, Double) -> Void
    ) async throws {
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
                enableAgentTools: enableAgentTools && model.name.contains("Gemma 4") && !model.name.contains("Translate") && model.modelFormat == .litertlm,
                onUpdate: onUpdate
            )
            return
        }
        #endif
        // ────────────────────────────────────────────────────────────────────

        _ = audioURL

        try await ensureSDKReady()

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
        let usePrompt: String
        if isHarmonyModel && !prompt.hasPrefix("__RAW_PROMPT__") {
            usePrompt = Self.buildHarmonyPrompt(prompt: prompt, systemPrompt: systemPrompt, thinkingEnabled: enableThinking)
        } else {
            usePrompt = prompt
        }

        let effectiveSystemPrompt: String?
        if prompt.hasPrefix("__RAW_PROMPT__") {
            // System prompt is already embedded in the formatted multi-turn prompt — don't pass it
            // again via options or the SDK will prepend it a second time, breaking conversation history.
            effectiveSystemPrompt = nil
        } else if isHarmonyModel {
            effectiveSystemPrompt = nil
        } else {
            effectiveSystemPrompt = systemPrompt
        }

        var options = RALLMGenerationOptions.defaults()
        options.maxTokens = Int32(effectiveMaxTokens)
        options.temperature = Float(temperature)
        options.topP = Float(topP)
        options.stopSequences = stopSequences
        options.streamingEnabled = true
        options.systemPrompt = effectiveSystemPrompt ?? ""

        do {

        if let imageURL,
           enableVision,
           let model = loadedAIModel(),
           model.supportsVision,
           isVisionProjectorAvailable(for: model) {
            try await ensureVLMLoaded(for: model)

            let image = vlmImage(from: imageURL)
            var vlmOptions = RAVLMGenerationOptions.defaults()
            vlmOptions.maxTokens = Int32(effectiveMaxTokens)
            vlmOptions.temperature = Float(temperature)
            vlmOptions.topP = Float(topP)
            vlmOptions.systemPrompt = effectiveSystemPrompt ?? ""
            let stream = try await RunAnywhere.processImageStream(image, prompt: usePrompt, options: vlmOptions)

            let isGemma4 = (loadedModelName.range(of: "gemma 4", options: .caseInsensitive) != nil ||
                            loadedModelName.range(of: "gemma-4", options: .caseInsensitive) != nil) &&
                           loadedModelName.range(of: "translate", options: .caseInsensitive) == nil

            var currentOutput = ""
            var finalResult: RAVLMResult?
            for try await event in stream {
                try Task.checkCancellation()
                if !event.token.isEmpty {
                    currentOutput += event.token
                    let displayOutput = isGemma4 ? Self.cleanGemma4Output(currentOutput) : currentOutput
                    onUpdate(displayOutput, 0, 0)
                }
                if event.isFinal {
                    finalResult = event.hasResult ? event.result : nil
                    break
                }
            }

            let result = finalResult ?? RAVLMResult()
            let finalText = result.text.isEmpty ? currentOutput : result.text
            let finalOutput = isGemma4 ? Self.cleanGemma4Output(finalText) : finalText
            onUpdate(finalOutput, Int(result.completionTokens), Double(result.tokensPerSecond))
            return
        }

        if let model = loadedAIModel(), model.id == Self.appleFoundationAliasId || loadedLLMModelId == Self.runAnywhereFoundationModelId {
            // Foundation models may not stream in exact per-token order; generate non-stream and emulate incremental updates for UX.
            let result = try await RunAnywhere.generate(prompt: usePrompt, options: options)
            let fullText = result.text

            // If the SDK provides separate thinking content, wrap it in sentinels so the
            // thinking drawer shows the real reasoning and the answer streams below it —
            // matching the same overlay path used for other models.
            let sdkThinking = result.thinkingContent.trimmingCharacters(in: .whitespacesAndNewlines)
            let answerText = fullText.trimmingCharacters(in: .whitespacesAndNewlines)
            let hasSdkThinking = !sdkThinking.isEmpty || result.thinkingTokens > 0

            if hasSdkThinking {
                // First, surface the thinking content immediately so the drawer opens.
                let thinkingDisplay = Self.thinkingSentinelOpen + sdkThinking + Self.thinkingSentinelClose
                onUpdate(thinkingDisplay, 0, 0)

                // Then stream the answer word-by-word after the thinking sentinels.
                var currentOutput = thinkingDisplay
                let answerTokens = answerText.split(separator: " ", omittingEmptySubsequences: false)
                for (index, token) in answerTokens.enumerated() {
                    if index > 0 { currentOutput += " " }
                    currentOutput += String(token)
                    onUpdate(currentOutput, 0, 0)
                    try? await Task.sleep(nanoseconds: 10_000_000) // 10ms for smoother perception
                }

                let finalDisplay = currentOutput.isEmpty ? thinkingDisplay : currentOutput
                onUpdate(finalDisplay, Int(result.responseTokens), result.tokensPerSecond)
            } else {
                // No thinking content — stream the answer directly (heuristic drawer handled by UI).
                var currentOutput = ""
                let tokens = fullText.split(separator: " ", omittingEmptySubsequences: false)
                for (index, token) in tokens.enumerated() {
                    if index > 0 { currentOutput += " " }
                    currentOutput += String(token)
                    onUpdate(currentOutput, Int(result.responseTokens), result.tokensPerSecond)
                    try? await Task.sleep(nanoseconds: 10_000_000) // 10ms for smoother perception
                }

                if currentOutput.isEmpty {
                    onUpdate(fullText, Int(result.responseTokens), result.tokensPerSecond)
                }
            }
            return
        }

        if let model = loadedAIModel() {
            try await ensureTextModelLoaded(for: model)
        }

        print("ℹ️ [LLMBackend] generate visionEnabled=\(enableVision) audioEnabled=\(enableAudio) images=0 videos=0")

        // === Harmony two-phase generation (GPT-OSS, thinking enabled) ===
        // <|end|> is a stop token built into the GGUF, so phase 1 ends after the analysis
        // section. We immediately start phase 2 with the thinking as context to get the
        // final answer — mirroring Android's Harmony state machine but as two sequential calls.
        if isHarmonyModel && enableThinking {
            // Phase 1 — get thinking content (generation stops at <|end|>)
            print("🧠 [ThinkingDebug][harmony-phase1] starting")
            print("🧠 [ThinkingDebug][gate] model=\(loadedModelName) supportsThinking=\(modelSupportsThinking) enableThinking=\(enableThinking)")
            let streamResult1 = try await RunAnywhere.generateStream(prompt: usePrompt, options: options)
            var thinkingRaw = ""
            var phase1Chunks = 0

            for try await event in streamResult1 {
                try Task.checkCancellation()
                thinkingRaw += event.token
                phase1Chunks += 1
                let pureThinking = Self.extractHarmonyThinking(thinkingRaw)
                if phase1Chunks == 1 {
                    print("🧠 [ThinkingDebug][harmony-phase1] firstChunk=\(String(event.token.prefix(80)))")
                }
                onUpdate(Self.thinkingSentinelOpen + pureThinking, 0, 0)
                if event.isFinal { break }
            }

            let pureThinking = Self.extractHarmonyThinking(thinkingRaw)
            print("🧠 [ThinkingDebug][harmony-phase1] complete thinkingChars=\(pureThinking.count) chunks=\(phase1Chunks)")

            // Phase 2 — inject thinking as context, get the final answer
            // Prompt = original (ending with <|start|>assistant) +
            //          <|channel|>analysis<|message|>THINKING<|end|><|start|>assistant<|channel|>final<|message|>
            // thinkingRaw starts with "analysis<|message|>..." so prepending "<|channel|>" reconstructs
            // the full Harmony structure with the correct special tokens.
            let phase2Prompt = usePrompt + "<|channel|>" + thinkingRaw + Self.harmonyEndTag + Self.harmonyFinalHeader
            print("🧠 [ThinkingDebug][harmony-phase2] starting phase2PromptLen=\(phase2Prompt.count)")

            let streamResult2 = try await RunAnywhere.generateStream(prompt: phase2Prompt, options: options)
            var finalOutput = ""
            var phase2Chunks = 0
            var phase2Final: RALLMStreamFinalResult?
            // The model may echo a channel prefix before the actual answer ("final<|message|>" or
            // "analysis<|message|>" variants). Strip it in-place so the answer stays clean.
            let finalChannelPrefixes = ["final<|message|>", "analysis<|message|>",
                                        "<|channel|>final<|message|>", "<|channel|>analysis<|message|>"]

            for try await event in streamResult2 {
                try Task.checkCancellation()
                finalOutput += event.token
                phase2Chunks += 1
                if phase2Chunks == 1 {
                    print("🧠 [ThinkingDebug][harmony-phase2] firstChunk=\(String(event.token.prefix(80)))")
                }
                if event.isFinal {
                    phase2Final = event.hasResult ? event.result : nil
                }
                // Strip any leading channel prefix before surfacing to UI.
                var displayFinal = finalOutput
                for pfx in finalChannelPrefixes {
                    if displayFinal.hasPrefix(pfx) {
                        displayFinal = String(displayFinal.dropFirst(pfx.count))
                        break
                    }
                    // Prefix still assembling — hide until complete
                    if pfx.hasPrefix(displayFinal) {
                        displayFinal = ""
                        break
                    }
                }
                onUpdate(Self.thinkingSentinelOpen + pureThinking + Self.thinkingSentinelClose + displayFinal, 0, 0)
                if event.isFinal { break }
            }

            // Strip channel prefix from the fully accumulated answer before saving.
            var cleanFinal = finalOutput.trimmingCharacters(in: CharacterSet.whitespacesAndNewlines)
            for pfx in finalChannelPrefixes {
                if cleanFinal.hasPrefix(pfx) {
                    cleanFinal = String(cleanFinal.dropFirst(pfx.count))
                        .trimmingCharacters(in: CharacterSet.whitespacesAndNewlines)
                    break
                }
            }
            print("🧠 [ThinkingDebug][harmony-phase2] complete finalChars=\(cleanFinal.count) chunks=\(phase2Chunks)")
            let result2 = phase2Final ?? RALLMStreamFinalResult()
            onUpdate(
                Self.thinkingSentinelOpen + pureThinking + Self.thinkingSentinelClose + cleanFinal,
                Int(result2.completionTokens),
                Double(result2.tokensPerSecond)
            )
            return
        }

        let streamResult = try await RunAnywhere.generateStream(prompt: usePrompt, options: options)
        var currentOutput = ""
        var chunkCount = 0
        var loggedRealThinkingMarker = false

        print(
            "🧠 [ThinkingDebug][gate] model=\(loadedModelName) supportsThinking=\(modelSupportsThinking) enableThinking=\(enableThinking) harmony=\(isHarmonyModel)"
        )

        let isGemma4 = (loadedModelName.range(of: "gemma 4", options: .caseInsensitive) != nil ||
                        loadedModelName.range(of: "gemma-4", options: .caseInsensitive) != nil) &&
                       loadedModelName.range(of: "translate", options: .caseInsensitive) == nil

        var finalStreamResult: RALLMStreamFinalResult?
        for try await event in streamResult {
            // Respect Task cancellation — stop consuming tokens when the caller
            // cancels (e.g. user taps stop, or turn-leak auto-stop).
            try Task.checkCancellation()

            chunkCount += 1
            currentOutput += event.token
            // Strip <unusedN> thinking tokens (Gemma 4 emits these when thinking mode activates)
            if currentOutput.contains("<unused") {
                currentOutput = currentOutput.replacingOccurrences(
                    of: #"<unused\d+>"#, with: "", options: .regularExpression)
            }
            let displayOutput = isGemma4 ? Self.cleanGemma4Output(currentOutput) : currentOutput
            let normalizedOutput = isHarmonyModel ? Self.normalizeHarmonyOutput(displayOutput) : (displayOutput, false)

            if chunkCount == 1 {
                print("🧠 [ThinkingDebug][stream] firstChunk preview=\(String(event.token.prefix(120)))")
            }

            let hasRealThinkingMarkers = normalizedOutput.0.contains(Self.thinkingSentinelOpen)
                || currentOutput.contains(Self.thinkingSentinelClose)
                || currentOutput.contains("<think>")
                || currentOutput.contains("</think>")
                || normalizedOutput.1

            if hasRealThinkingMarkers && !loggedRealThinkingMarker {
                loggedRealThinkingMarker = true
                print("🧠 [ThinkingDebug][stream] first real thinking marker chunk=\(chunkCount) preview=\(String(normalizedOutput.0.prefix(160)))")
            }

            if chunkCount == 1 || chunkCount % 40 == 0 {
                print("🧠 [ThinkingDebug][stream] chunk=\(chunkCount) chars=\(currentOutput.count) hasRealMarkers=\(hasRealThinkingMarkers) preview=\(String(currentOutput.prefix(120)))")
            }

            // Always surface the real cumulative stream text. If the backend emits actual
            // thinking markers, the UI parser will split them. If it doesn't, the text is
            // just the answer and should remain visible while streaming.
            onUpdate(normalizedOutput.0, 0, 0)

            if event.isFinal {
                finalStreamResult = event.hasResult ? event.result : nil
                break
            }
        }

        let result = finalStreamResult ?? RALLMStreamFinalResult()
        // Strip trailing stop tokens that leak through when generation ends on EOG
        let gemmaTrailingTokens = ["<end_of_turn>", "</s>", "<eos>", "<|eot_id|>"]
        for tok in gemmaTrailingTokens {
            if currentOutput.hasSuffix(tok) {
                currentOutput = String(currentOutput.dropLast(tok.count))
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                break
            }
        }
        if isGemma4 {
            currentOutput = Self.cleanGemma4Output(currentOutput)
        }
        let normalizedFinalOutput = isHarmonyModel ? Self.normalizeHarmonyOutput(currentOutput) : (currentOutput, false)
        let sdkThinking = result.hasThinkingContent ? result.thinkingContent.trimmingCharacters(in: CharacterSet.whitespacesAndNewlines) : ""
        let sdkHasThinking = !sdkThinking.isEmpty || !result.thinkingContent.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        let streamHasThinkingMarkers = normalizedFinalOutput.0.contains(Self.thinkingSentinelOpen)
            || currentOutput.contains(Self.thinkingSentinelClose)
            || currentOutput.contains("<think>")
            || currentOutput.contains("</think>")
            || normalizedFinalOutput.1
        let shouldUseFinalThinkingOverlay = sdkHasThinking && !streamHasThinkingMarkers

        if shouldUseFinalThinkingOverlay {
            let response = result.text.trimmingCharacters(in: CharacterSet.whitespacesAndNewlines)
            let displayText: String
            if !sdkThinking.isEmpty {
                displayText = Self.thinkingSentinelOpen + sdkThinking + Self.thinkingSentinelClose + response
            } else {
                displayText = response.isEmpty ? currentOutput : response
            }

            print(
                "🧠 [ThinkingDebug][final] rawChars=\(currentOutput.count) responseChars=\(result.text.count) thinkingChars=\(result.thinkingContent.count) tokens=\(result.completionTokens) responseTokens=\(result.completionTokens) sdkHasThinking=\(sdkHasThinking) rawPreview=\(String(currentOutput.prefix(120))) responsePreview=\(String(result.text.prefix(120))) thinkingPreview=\(String(result.thinkingContent.prefix(120)))"
            )

            onUpdate(displayText, Int(result.completionTokens), Double(result.tokensPerSecond))
        } else {
            print(
                "🧠 [ThinkingDebug][final] raw-only rawChars=\(currentOutput.count) responseChars=\(result.text.count) thinkingChars=\(result.thinkingContent.count) tokens=\(result.completionTokens) responseTokens=\(result.completionTokens) streamHasMarkers=\(streamHasThinkingMarkers) rawPreview=\(String(currentOutput.prefix(120)))"
            )
            onUpdate(normalizedFinalOutput.0, Int(result.completionTokens), Double(result.tokensPerSecond))
        }
        } catch {
            let isRawPrompt = prompt.hasPrefix("__RAW_PROMPT__")
            print(
                "❌ [LLMBackend] generate failed model=\(loadedModelName) maxTokens=\(effectiveMaxTokens) contextWindow=\(loadedContextWindow ?? contextWindow) promptChars=\(usePrompt.count) rawPrompt=\(isRawPrompt) error=\(error)"
            )
            throw error
        }
    }
}
