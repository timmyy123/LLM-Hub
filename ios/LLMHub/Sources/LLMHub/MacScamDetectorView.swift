//
//  MacScamDetectorView.swift
//  LLMHub
//
//  Native macOS Scam Detector. Same persisted settings, prompts, model handling
//  and localized strings as `ScamDetectorScreen` on iOS.
//

#if os(macOS)
import AppKit
import PhotosUI
import SwiftUI

struct MacScamDetectorView: View {
    @EnvironmentObject var settings: AppSettings
    @ObservedObject private var ttsManager = OnDeviceTtsManager.shared
    @ObservedObject private var llm = LLMBackend.shared

    @AppStorage("feature_scam_model_name") private var selectedModelName: String = ""
    @AppStorage("feature_scam_enable_thinking") private var enableThinking: Bool = true
    @AppStorage("feature_scam_enable_vision") private var enableVision: Bool = true

    @State private var maxTokens: Double = 4096
    @State private var inputText: String = ""
    @State private var outputText: String = ""
    @State private var isLoading = false
    @State private var isAnalyzing = false
    @State private var isFetchingURL = false
    @State private var showInspector = false
    @State private var errorMessage: String?
    @State private var selectedImageItem: PhotosPickerItem?
    @State private var selectedImageURL: URL?
    @State private var generationTask: Task<Void, Never>?

    private let ttsKey = "scam-detector-output"

    private var hasOutput: Bool {
        !outputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    var body: some View {
        Group {
            if selectedModelName.isEmpty {
                ContentUnavailableView {
                    Label(settings.localized("scam_detector_load_model"), systemImage: "shield.lefthalf.filled")
                } description: {
                    Text(settings.localized("scam_detector_load_model_desc"))
                } actions: {
                    Button(settings.localized("feature_settings_title")) { showInspector = true }
                        .buttonStyle(.borderedProminent)
                }
            } else {
                editor
                    .safeAreaInset(edge: .bottom, spacing: 0) {
                        MacPrimaryActionBar(
                            title: isLoading ? settings.localized("model_loading")
                                : isAnalyzing ? settings.localized("content_description_stop")
                                : settings.localized("scam_detector_analyze"),
                            systemImage: (isAnalyzing || isLoading) ? "stop.fill" : "shield.lefthalf.filled",
                            isBusy: isLoading,
                            isEnabled: !(selectedModelName.isEmpty || isLoading),
                            tint: isAnalyzing ? .red : ApolloPalette.accentStrong,
                            action: toggleAnalyze
                        )
                    }
            }
        }
        .navigationTitle(settings.localized("scam_detector_title"))
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                Button {
                    showInspector.toggle()
                } label: {
                    Label(settings.localized("feature_settings_title"), systemImage: "slider.horizontal.3")
                }
            }
        }
        .inspector(isPresented: $showInspector) {
            MacFeatureModelInspector(
                selectedModelName: $selectedModelName,
                maxTokens: $maxTokens,
                enableThinking: $enableThinking,
                enableVision: $enableVision,
                isLoading: $isLoading,
                errorMessage: $errorMessage,
                supportsVisionToggle: true,
                visionToggleTitleKey: "scam_detector_enable_vision",
                visionAvailableCheck: hasDownloadedVisionProjector,
                modelFilter: isNonTranslatorFeatureModel,
                onLoad: { await ensureModelLoaded(force: false) },
                onUnload: { llm.unloadModel() },
                showsThinkingToggle: true
            )
            .inspectorColumnWidth(min: 280, ideal: 320, max: 420)
        }
        .task {
            await refreshDownloadedModelStatus()
            let available = downloadableFeatureModels().filter(isNonTranslatorFeatureModel)
            if selectedModelName.isEmpty || !available.contains(where: { $0.name == selectedModelName }) {
                selectedModelName = available.first?.name ?? ""
            }
            maxTokens = loadScamContextWindow(for: selectedModelName)
        }
        .onChange(of: selectedModelName) { _, newModelName in
            maxTokens = loadScamContextWindow(for: newModelName)
        }
        .onChange(of: maxTokens) { _, newValue in
            saveScamContextWindow(value: newValue, for: selectedModelName)
        }
        .onChange(of: selectedImageItem) { _, item in
            guard let item else { selectedImageURL = nil; return }
            Task {
                if let sourceURL = try? await item.loadTransferable(type: URL.self) {
                    selectedImageURL = sourceURL
                    return
                }
                if let data = try? await item.loadTransferable(type: Data.self) {
                    let temp = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString).appendingPathExtension("jpg")
                    try? data.write(to: temp)
                    selectedImageURL = temp
                }
            }
        }
        .onChange(of: enableVision) { _, isEnabled in
            if !isEnabled {
                selectedImageItem = nil
                selectedImageURL = nil
            }
        }
        .onDisappear {
            generationTask?.cancel()
            llm.unloadModel()
        }
    }

    private var editor: some View {
        HStack(spacing: 0) {
            VStack(alignment: .leading, spacing: 8) {
                MacTextPanel(title: settings.localized("scam_detector_input_label"), text: $inputText)

                HStack(spacing: 6) {
                    Button {
                        if let clip = NSPasteboard.general.string(forType: .string), !clip.isEmpty {
                            inputText += clip
                        }
                    } label: {
                        Image(systemName: "doc.on.clipboard")
                    }

                    if enableVision {
                        PhotosPicker(selection: $selectedImageItem, matching: .images) {
                            Image(systemName: "photo")
                        }
                    }

                    Button {
                        ttsManager.toggleSpeaking(
                            getDisplayContentWithoutThinking(outputText),
                            fallbackLanguage: settings.selectedLanguage,
                            key: ttsKey
                        )
                    } label: {
                        Image(systemName: ttsManager.isSpeaking(key: ttsKey) ? "stop.fill" : "speaker.wave.2")
                    }
                    .disabled(!hasOutput)

                    Button {
                        copyOutput()
                    } label: {
                        Image(systemName: "doc.on.doc")
                    }
                    .disabled(!hasOutput)

                    Spacer()
                }
                .buttonStyle(.bordered)

                if enableVision,
                   let selectedImageURL,
                   let nsImage = NSImage(contentsOf: selectedImageURL) {
                    ZStack(alignment: .topTrailing) {
                        Image(nsImage: nsImage)
                            .resizable()
                            .scaledToFit()
                            .frame(maxHeight: 220)
                            .frame(maxWidth: .infinity)
                            .clipShape(RoundedRectangle(cornerRadius: 8))
                        Button {
                            self.selectedImageURL = nil
                            self.selectedImageItem = nil
                        } label: {
                            Image(systemName: "xmark.circle.fill")
                                .font(.title3)
                                .foregroundStyle(.white, .black.opacity(0.5))
                        }
                        .buttonStyle(.plain)
                        .padding(8)
                    }
                }
            }
            .padding()
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)

            Divider()

            VStack(alignment: .leading, spacing: 8) {
                Text(settings.localized("scam_detector_result")).font(.headline)

                if isFetchingURL {
                    HStack(spacing: 8) {
                        ProgressView().controlSize(.small)
                        Text(settings.localized("scam_detector_fetching_url"))
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                        Spacer()
                    }
                }

                ScrollViewReader { proxy in
                    ScrollView {
                        VStack(alignment: .leading, spacing: 0) {
                            if outputText.isEmpty {
                                Text("-")
                                    .foregroundStyle(.secondary)
                                    .textSelection(.enabled)
                                    .frame(maxWidth: .infinity, alignment: .leading)
                            } else {
                                ThinkingAwareResultContent(
                                    content: outputText,
                                    isGenerating: isAnalyzing,
                                    preferThinkingWhileStreaming: enableThinking
                                        && (selectedFeatureModel(named: selectedModelName)?.supportsThinking == true)
                                        && supportsUnmarkedStreamingThinkingHeuristic(forModelNamed: selectedModelName)
                                )
                            }
                            Color.clear.frame(height: 1).id("scam_detector_bottom")
                        }
                        .padding(10)
                    }
                    .background(Color(nsColor: .textBackgroundColor).opacity(0.2), in: RoundedRectangle(cornerRadius: 8))
                    .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color(nsColor: .separatorColor)))
                    .onChange(of: outputText) { _, _ in
                        if isAnalyzing { proxy.scrollTo("scam_detector_bottom", anchor: .bottom) }
                    }
                }

                HStack(spacing: 6) {
                    if hasOutput {
                        Button {
                            copyOutput()
                        } label: {
                            Image(systemName: "doc.on.doc")
                        }
                        .buttonStyle(.bordered)
                    }

                    Spacer()

                    if let errorMessage {
                        Text(errorMessage)
                            .font(.caption)
                            .foregroundStyle(.red)
                            .lineLimit(2)
                    }
                }
            }
            .padding()
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
        }
    }

    private func copyOutput() {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(getDisplayContentWithoutThinking(outputText), forType: .string)
    }

    // MARK: - Logic (mirrors ScamDetectorScreen)

    private func loadScamContextWindow(for modelName: String) -> Double {
        let key = "feature_scam_max_tokens_\(modelName)"
        let val = UserDefaults.standard.double(forKey: key)
        if val > 0 {
            return val
        }
        if let model = selectedFeatureModel(named: modelName) {
            let cap = contextLimitForFeatureModel(model)
            return Double(min(featureDefaultContextWindow, cap))
        }
        return 4096
    }

    private func saveScamContextWindow(value: Double, for modelName: String) {
        let key = "feature_scam_max_tokens_\(modelName)"
        UserDefaults.standard.set(value, forKey: key)
    }

    private func buildAnalysisRequest(content: String, hasImage: Bool) -> (systemPrompt: String, prompt: String) {
        if hasImage && !content.isEmpty {
            let systemPrompt = """
            You are a scam detection expert. Analyze BOTH the provided image AND the text content below for potential scams, fraud, phishing attempts, or suspicious activity.

            **Instructions:**
            - Carefully examine the image for any suspicious elements, fake logos, misleading graphics, or scam indicators
            - Cross-reference the text content with what's shown in the image
            - Look for inconsistencies between the image and text
            - Check if the image appears to be a screenshot of a phishing message, fake website, or fraudulent offer

            Please provide a comprehensive analysis covering:
            1. **Risk Level**: Low, Medium, High, or Critical
            2. **Red Flags in Image**: List any suspicious visual elements (fake logos, poor quality graphics, misleading layouts, etc.)
            3. **Red Flags in Text**: List any suspicious text elements (urgency tactics, too-good-to-be-true offers, suspicious links, impersonation, poor grammar, etc.)
            4. **Consistency Check**: Do the image and text align? Are there contradictions?
            5. **Legitimacy Indicators**: Any signs suggesting it might be legitimate
            6. **Verdict**: Is this likely a scam? Explain your reasoning based on BOTH the image and text.
            7. **Recommendations**: What should the user do?

            Be thorough and specific in your analysis. If you detect a scam, clearly state it. If it appears legitimate, explain why.
            """
            let prompt = """
            **Text content to analyze:**
            \(content)
            """
            return (systemPrompt, prompt)
        }

        if hasImage {
            let systemPrompt = """
            You are a scam detection expert. Analyze the provided image for potential scams, fraud, phishing attempts, or suspicious activity.

            **Instructions:**
            - Carefully examine the image for any suspicious elements, fake logos, misleading graphics, or scam indicators
            - Check if the image appears to be a screenshot of a phishing message, fake website, or fraudulent offer
            - Look for common scam tactics in the visual content

            Please provide a comprehensive analysis covering:
            1. **Risk Level**: Low, Medium, High, or Critical
            2. **Visual Red Flags**: List any suspicious elements in the image (fake logos, poor quality graphics, misleading layouts, urgency messages, too-good-to-be-true offers, etc.)
            3. **Legitimacy Indicators**: Any visual signs suggesting it might be legitimate
            4. **Verdict**: Is this likely a scam? Explain your reasoning based on the image.
            5. **Recommendations**: What should the user do?

            Be thorough and specific in your analysis. If you detect a scam, clearly state it. If it appears legitimate, explain why.
            """
            return (systemPrompt, "Analyze the provided image for potential scams, fraud, phishing attempts, or suspicious activity.")
        }

        let systemPrompt = """
        You are a scam detection expert. Analyze the following content for potential scams, fraud, phishing attempts, or suspicious activity.

        IMPORTANT: Respond in the same language as the input content. Match the language of the content in the image.

        Please provide a comprehensive analysis covering:
        1. **Risk Level**: Low, Medium, High, or Critical
        2. **Red Flags**: List any suspicious elements (urgency tactics, too-good-to-be-true offers, suspicious links, impersonation, poor grammar, etc.)
        3. **Legitimacy Indicators**: Any signs suggesting it might be legitimate
        4. **Verdict**: Is this likely a scam? Explain your reasoning.
        5. **Recommendations**: What should the user do?

        Be thorough and specific in your analysis. If you detect a scam, clearly state it. If it appears legitimate, explain why.
        """
        let prompt = """
        Content to analyze:
        \(content)
        """
        return (systemPrompt, prompt)
    }

    private func detectFirstURL(in text: String) -> String? {
        let pattern = #"https?://[^\s]+"#
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return nil }
        let range = NSRange(text.startIndex..<text.endIndex, in: text)
        guard let match = regex.firstMatch(in: text, range: range),
              let urlRange = Range(match.range, in: text) else { return nil }
        return String(text[urlRange])
    }

    private func extractTextFromHTML(_ html: String) -> String {
        var cleaned = html.replacingOccurrences(of: #"<script[^>]*>.*?</script>"#, with: "", options: [.regularExpression, .caseInsensitive])
        cleaned = cleaned.replacingOccurrences(of: #"<style[^>]*>.*?</style>"#, with: "", options: [.regularExpression, .caseInsensitive])
        cleaned = cleaned.replacingOccurrences(of: #"<[^>]*>"#, with: " ", options: .regularExpression)
        cleaned = cleaned
            .replacingOccurrences(of: "&amp;", with: "&")
            .replacingOccurrences(of: "&lt;", with: "<")
            .replacingOccurrences(of: "&gt;", with: ">")
            .replacingOccurrences(of: "&quot;", with: "\"")
            .replacingOccurrences(of: "&#39;", with: "'")
            .replacingOccurrences(of: "&nbsp;", with: " ")
            .replacingOccurrences(of: #"\s+"#, with: " ", options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return String(cleaned.prefix(3000))
    }

    private func fetchURLContent(_ urlString: String) async -> String {
        guard let url = URL(string: urlString) else { return "" }
        var request = URLRequest(url: url)
        request.timeoutInterval = 15
        request.setValue("Mozilla/5.0", forHTTPHeaderField: "User-Agent")
        do {
            let (data, _) = try await URLSession.shared.data(for: request)
            let html = String(data: data, encoding: .utf8) ?? ""
            return extractTextFromHTML(html)
        } catch {
            return ""
        }
    }

    private func ensureModelLoaded(force: Bool) async {
        guard let model = selectedFeatureModel(named: selectedModelName) else {
            errorMessage = settings.localized("scam_detector_no_model")
            return
        }
        isLoading = true
        defer { isLoading = false }

        let modelContextCap = contextLimitForFeatureModel(model)
        let effectiveContext = min(max(1, Int(maxTokens)), modelContextCap)
        let shouldReload = force
            || llm.currentlyLoadedModel != model.name
            || llm.loadedContextWindow != effectiveContext

        llm.maxTokens = min(Int(maxTokens), effectiveContext)
        llm.contextWindow = effectiveContext
        llm.enableVision = enableVision
        llm.enableAudio = false
        let isGranite42 = selectedModelName.lowercased().contains("granite-4.2") || selectedModelName.lowercased().contains("granite 4.2")
        llm.enableThinking = isGranite42 ? false : enableThinking

        do {
            if shouldReload {
                try await llm.loadModel(model)
            }
            errorMessage = nil
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    private func toggleAnalyze() {
        if isAnalyzing {
            generationTask?.cancel()
            generationTask = nil
            isAnalyzing = false
            return
        }

        generationTask = Task {
            await ensureModelLoaded(force: false)
            guard llm.isLoaded else { return }

            isAnalyzing = true
            outputText = ""

            var contentToAnalyze = inputText.trimmingCharacters(in: .whitespacesAndNewlines)

            if let url = detectFirstURL(in: contentToAnalyze) {
                isFetchingURL = true
                let fetchedContent = await fetchURLContent(url)
                isFetchingURL = false
                if !fetchedContent.isEmpty {
                    let additionalContext = contentToAnalyze.replacingOccurrences(of: url, with: "").trimmingCharacters(in: .whitespacesAndNewlines)
                    contentToAnalyze = """
                    URL: \(url)

                    Content from URL:
                    \(fetchedContent)

                    \(!additionalContext.isEmpty ? "Additional context: \(additionalContext)" : "")
                    """
                }
            }

            if contentToAnalyze.isEmpty && selectedImageURL == nil {
                errorMessage = settings.localized("scam_detector_input_hint")
                isAnalyzing = false
                generationTask = nil
                return
            }

            do {
                let hasImage = selectedImageURL != nil && enableVision
                let analysisRequest = buildAnalysisRequest(content: contentToAnalyze, hasImage: hasImage)
                let effectiveImageURL = enableVision ? selectedImageURL : nil
                try await llm.generate(
                    prompt: analysisRequest.prompt,
                    imageURL: effectiveImageURL,
                    systemPrompt: analysisRequest.systemPrompt
                ) { text, _, _ in
                    Task { @MainActor in
                        outputText = sanitizeModelOutputText(text)
                    }
                }
            } catch {
                errorMessage = error.localizedDescription
            }

            isAnalyzing = false
            isFetchingURL = false
            generationTask = nil
        }
    }
}
#endif
