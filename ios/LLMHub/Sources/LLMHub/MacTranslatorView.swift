//
//  MacTranslatorView.swift
//  LLMHub
//
//  Native macOS Translator. Same persisted settings, prompts, model handling
//  and localized strings as `TranslatorScreen` on iOS.
//

#if os(macOS)
import AppKit
import PhotosUI
import SwiftUI
import UniformTypeIdentifiers

struct MacTranslatorView: View {
    @EnvironmentObject var settings: AppSettings
    @ObservedObject private var ttsManager = OnDeviceTtsManager.shared
    @ObservedObject private var llm = LLMBackend.shared

    @AppStorage("feature_translator_model_name") private var selectedModelName: String = ""
    @AppStorage("feature_translator_enable_vision") private var enableVision: Bool = true
    @AppStorage("feature_translator_enable_audio") private var enableAudio: Bool = true
    @AppStorage("feature_translator_source_lang") private var sourceLanguageCode: String = "en"
    @AppStorage("feature_translator_target_lang") private var targetLanguageCode: String = "es"
    @AppStorage("feature_translator_auto_detect") private var autoDetectSource: Bool = false

    @State private var maxTokens: Double = 2048
    @State private var inputText: String = ""
    @State private var outputText: String = ""
    @State private var isLoading = false
    @State private var isTranslating = false
    @State private var showInspector = false
    @State private var errorMessage: String?
    @State private var selectedImageItem: PhotosPickerItem?
    @State private var selectedImageURL: URL?
    @State private var selectedAudioURL: URL?
    @State private var showAudioImporter = false
    @State private var generationTask: Task<Void, Never>?
    @State private var availableTranslatorModels: [AIModel] = []
    @StateObject private var audioRecorder = AudioRecorder()

    let onNavigateToModels: () -> Void

    /// Picker tag used for the "auto detect" source entry.
    private let autoDetectTag = "__auto_detect__"

    private var selectedModel: AIModel? {
        selectedFeatureModel(named: selectedModelName).flatMap { isTranslatorSupportedModel($0) ? $0 : nil }
    }

    private var sourceLanguage: TranslatorLanguage {
        translatorLanguages.first(where: { $0.code == sourceLanguageCode })
            ?? translatorLanguages.first(where: { $0.code == "en" })
            ?? translatorLanguages[0]
    }

    private var targetLanguage: TranslatorLanguage {
        translatorLanguages.first(where: { $0.code == targetLanguageCode })
            ?? translatorLanguages.first(where: { $0.code == "es" })
            ?? translatorLanguages[0]
    }

    private var sourceSelection: Binding<String> {
        Binding(
            get: { autoDetectSource ? autoDetectTag : sourceLanguage.code },
            set: { newValue in
                if newValue == autoDetectTag {
                    autoDetectSource = true
                } else {
                    autoDetectSource = false
                    sourceLanguageCode = newValue
                }
            }
        )
    }

    private var targetSelection: Binding<String> {
        Binding(
            get: { targetLanguage.code },
            set: { targetLanguageCode = $0 }
        )
    }

    private var inputHasContent: Bool {
        !inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            || selectedImageURL != nil
            || selectedAudioURL != nil
    }

    private var canUseAudioInput: Bool {
        guard let model = selectedModel else { return false }
        return enableAudio && model.isGemma4LiteRTLM
    }

    private var hasOutput: Bool {
        !outputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    var body: some View {
        Group {
            if selectedModelName.isEmpty {
                unloadedStateView
            } else {
                loadedStateView
                    .safeAreaInset(edge: .bottom, spacing: 0) {
                        MacPrimaryActionBar(
                            title: isLoading ? settings.localized("model_loading")
                                : isTranslating ? settings.localized("content_description_stop")
                                : settings.localized("translator_translate"),
                            systemImage: (isTranslating || isLoading) ? "stop.fill" : "network",
                            isBusy: isLoading,
                            isEnabled: !(selectedModelName.isEmpty || isLoading || (!isTranslating && !inputHasContent)),
                            tint: isTranslating ? .red : ApolloPalette.accentStrong,
                            action: toggleTranslate
                        )
                    }
            }
        }
        .navigationTitle(settings.localized("translator_title"))
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
                enableThinking: .constant(false),
                enableVision: $enableVision,
                enableAudio: $enableAudio,
                isLoading: $isLoading,
                errorMessage: $errorMessage,
                supportsVisionToggle: true,
                visionToggleTitleKey: "translator_enable_vision",
                audioToggleTitleKey: "translator_enable_audio",
                visionAvailableCheck: translatorHasDownloadedVisionProjector,
                modelFilter: isTranslatorSupportedModel,
                onLoad: { await ensureModelLoaded(force: false) },
                onUnload: { llm.unloadModel() },
                showsThinkingToggle: false
            )
            .inspectorColumnWidth(min: 280, ideal: 320, max: 420)
        }
        .onChange(of: showInspector) { _, isPresented in
            if !isPresented {
                Task {
                    await refreshDownloadedModelStatus()
                    let available = downloadableFeatureModels().filter(isTranslatorSupportedModel)
                    availableTranslatorModels = available
                    if selectedModelName.isEmpty || !available.contains(where: { $0.name == selectedModelName }) {
                        selectedModelName = available.first?.name ?? ""
                    }
                }
            }
        }
        .task {
            await refreshDownloadedModelStatus()
            let available = downloadableFeatureModels().filter(isTranslatorSupportedModel)
            availableTranslatorModels = available
            if selectedModelName.isEmpty || !available.contains(where: { $0.name == selectedModelName }) {
                selectedModelName = available.first?.name ?? ""
            }
            maxTokens = loadTranslatorContextWindow(for: selectedModelName)
        }
        .onChange(of: selectedModelName) { _, newModelName in
            maxTokens = loadTranslatorContextWindow(for: newModelName)
        }
        .onChange(of: maxTokens) { _, newValue in
            saveTranslatorContextWindow(value: newValue, for: selectedModelName)
        }
        .onChange(of: selectedImageItem) { _, item in
            guard let item else { selectedImageURL = nil; return }
            Task {
                if let sourceURL = try? await item.loadTransferable(type: URL.self) {
                    selectedImageURL = sourceURL
                    inputText = ""
                    selectedAudioURL = nil
                    return
                }
                if let data = try? await item.loadTransferable(type: Data.self) {
                    let temp = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString).appendingPathExtension("jpg")
                    try? data.write(to: temp)
                    selectedImageURL = temp
                    inputText = ""
                    selectedAudioURL = nil
                }
            }
        }
        .onChange(of: enableVision) { _, isEnabled in
            if !isEnabled {
                selectedImageItem = nil
                selectedImageURL = nil
            }
        }
        .onChange(of: enableAudio) { _, isEnabled in
            if !isEnabled {
                selectedAudioURL = nil
            }
        }
        .fileImporter(
            isPresented: $showAudioImporter,
            allowedContentTypes: [.audio, .mpeg4Audio, .mp3],
            allowsMultipleSelection: false
        ) { result in
            switch result {
            case .success(let urls):
                if let first = urls.first {
                    selectedAudioURL = sandboxCopyOfImportedFile(first)
                }
            case .failure(let error):
                NSLog("[LLMHub][Translator] Audio import failed: \(error.localizedDescription)")
            }
        }
        .onChange(of: selectedAudioURL) { _, url in
            if url != nil {
                inputText = ""
                selectedImageItem = nil
                selectedImageURL = nil
            }
        }
        .onDisappear {
            generationTask?.cancel()
            audioRecorder.cancelRecording()
            llm.unloadModel()
        }
    }

    // MARK: - Empty state

    private var unloadedStateView: some View {
        let requiresDownload = availableTranslatorModels.isEmpty

        return ContentUnavailableView {
            Label(
                settings.localized(requiresDownload ? "load_model_to_start" : "scam_detector_load_model"),
                systemImage: "network"
            )
        } description: {
            Text(settings.localized(requiresDownload ? "translator_load_model_desc" : "scam_detector_load_model_desc"))
        } actions: {
            Button(settings.localized(requiresDownload ? "download_models" : "feature_settings_title")) {
                if requiresDownload {
                    onNavigateToModels()
                } else {
                    showInspector = true
                }
            }
            .buttonStyle(.borderedProminent)
        }
    }

    // MARK: - Loaded state

    private var loadedStateView: some View {
        VStack(spacing: 0) {
            languageBar
                .padding(.horizontal)
                .padding(.vertical, 10)

            Divider()

            HStack(spacing: 0) {
                inputPanel
                    .padding()
                    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)

                Divider()

                outputPanel
                    .padding()
                    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
            }
        }
    }

    private var languageBar: some View {
        HStack(spacing: 10) {
            Picker(settings.localized("translator_source_lang"), selection: sourceSelection) {
                Text(settings.localized("lang_auto_detect")).tag(autoDetectTag)
                Divider()
                ForEach(translatorLanguages) { language in
                    Text(settings.localized(language.localizationKey)).tag(language.code)
                }
            }
            .frame(maxWidth: 320)

            Button {
                let oldSource = sourceLanguageCode
                sourceLanguageCode = targetLanguageCode
                targetLanguageCode = oldSource
            } label: {
                Image(systemName: "arrow.left.arrow.right")
            }
            .buttonStyle(.bordered)
            .disabled(autoDetectSource)

            Picker(settings.localized("translator_target_lang"), selection: targetSelection) {
                ForEach(translatorLanguages) { language in
                    Text(settings.localized(language.localizationKey)).tag(language.code)
                }
            }
            .frame(maxWidth: 320)

            Spacer(minLength: 0)
        }
    }

    private var inputPanel: some View {
        let hasSelectedImage = selectedImageURL != nil

        return VStack(alignment: .leading, spacing: 8) {
            if let selectedImageURL,
               let nsImage = NSImage(contentsOf: selectedImageURL) {
                Text(settings.localized("translator_input_label")).font(.headline)
                ZStack(alignment: .topTrailing) {
                    Image(nsImage: nsImage)
                        .resizable()
                        .scaledToFit()
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                        .clipShape(RoundedRectangle(cornerRadius: 8))
                    Button {
                        self.selectedImageItem = nil
                        self.selectedImageURL = nil
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                            .font(.title3)
                            .foregroundStyle(.white, .black.opacity(0.5))
                    }
                    .buttonStyle(.plain)
                    .padding(8)
                }
                .frame(maxHeight: .infinity)
            } else {
                MacTextPanel(title: settings.localized("translator_input_label"), text: $inputText)
            }

            HStack(spacing: 6) {
                Button {
                    if let clip = NSPasteboard.general.string(forType: .string), !clip.isEmpty {
                        inputText += clip
                        selectedImageItem = nil
                        selectedImageURL = nil
                        selectedAudioURL = nil
                    }
                } label: {
                    Image(systemName: "doc.on.clipboard")
                }

                Button {
                    ttsManager.toggleSpeaking(
                        inputText,
                        fallbackLanguage: settings.selectedLanguage,
                        key: "translator-input"
                    )
                } label: {
                    Image(systemName: ttsManager.isSpeaking(key: "translator-input") ? "stop.fill" : "speaker.wave.2")
                }
                .disabled(inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)

                if canUseAudioInput {
                    Button {
                        if audioRecorder.isRecording {
                            _ = audioRecorder.stopRecording()
                        } else {
                            Task { @MainActor in
                                let isGemma4 = selectedModel?.isGemma4LiteRTLM ?? false
                                let ext = isGemma4 ? "wav" : "m4a"
                                let destination = persistentAudioStorageDirectory()
                                    .appendingPathComponent("translator_audio_\(UUID().uuidString)")
                                    .appendingPathExtension(ext)
                                _ = await audioRecorder.startRecording(
                                    outputURL: destination,
                                    autoStopAfterSilence: false,
                                    isFloat32Wav: isGemma4
                                ) { url in
                                    Task { @MainActor in
                                        selectedAudioURL = url
                                    }
                                }
                            }
                        }
                    } label: {
                        Image(systemName: audioRecorder.isRecording ? "stop.fill" : "mic.fill")
                            .foregroundStyle(audioRecorder.isRecording ? Color.red : Color.primary)
                    }

                    Button {
                        showAudioImporter = true
                    } label: {
                        Image(systemName: "waveform.badge.plus")
                    }
                }

                if enableVision && (selectedModel?.supportsVision == true) {
                    PhotosPicker(selection: $selectedImageItem, matching: .images) {
                        Image(systemName: hasSelectedImage ? "photo.badge.plus" : "photo")
                    }
                }

                Spacer()
            }
            .buttonStyle(.bordered)

            if let selectedAudioURL {
                MacTranslatorAudioFileRow(url: selectedAudioURL) {
                    self.selectedAudioURL = nil
                }
            }
        }
    }

    private var outputPanel: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(settings.localized("translator_result")).font(.headline)
            ScrollViewReader { proxy in
                ScrollView {
                    VStack(alignment: .leading, spacing: 0) {
                        if outputText.isEmpty {
                            Text("-")
                                .foregroundStyle(.secondary)
                                .textSelection(.enabled)
                                .frame(maxWidth: .infinity, alignment: .leading)
                        } else {
                            RenderMessageSegments(displayContent: outputText)
                                .frame(maxWidth: .infinity, alignment: .leading)
                        }
                        Color.clear.frame(height: 1).id("translator_bottom")
                    }
                    .padding(10)
                }
                .background(Color(nsColor: .textBackgroundColor).opacity(0.2), in: RoundedRectangle(cornerRadius: 8))
                .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color(nsColor: .separatorColor)))
                .onChange(of: outputText) { _, _ in
                    if isTranslating { proxy.scrollTo("translator_bottom", anchor: .bottom) }
                }
            }

            HStack(spacing: 6) {
                Button {
                    ttsManager.toggleSpeaking(
                        outputText,
                        fallbackLanguage: settings.selectedLanguage,
                        key: "translator-output"
                    )
                } label: {
                    Image(systemName: ttsManager.isSpeaking(key: "translator-output") ? "stop.fill" : "speaker.wave.2")
                }
                .disabled(!hasOutput)

                Button {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(outputText, forType: .string)
                } label: {
                    Image(systemName: "doc.on.doc")
                }
                .disabled(!hasOutput)

                Spacer()

                if let errorMessage {
                    Text(errorMessage)
                        .font(.caption)
                        .foregroundStyle(.red)
                        .lineLimit(2)
                }
            }
            .buttonStyle(.bordered)
        }
    }

    // MARK: - Logic (mirrors TranslatorScreen)

    private func loadTranslatorContextWindow(for modelName: String) -> Double {
        let key = "feature_translator_max_tokens_\(modelName)"
        let val = UserDefaults.standard.double(forKey: key)
        if val > 0 {
            return val
        }
        if let model = selectedFeatureModel(named: modelName) {
            let cap = contextLimitForFeatureModel(model, fallback: 4096)
            return Double(min(featureDefaultContextWindow, cap))
        }
        return 4096
    }

    private func saveTranslatorContextWindow(value: Double, for modelName: String) {
        let key = "feature_translator_max_tokens_\(modelName)"
        UserDefaults.standard.set(value, forKey: key)
    }

    /// The macOS app is sandboxed: file importer URLs are only readable while
    /// security-scoped access is held, so keep a private copy for playback and inference.
    private func sandboxCopyOfImportedFile(_ url: URL) -> URL {
        let accessing = url.startAccessingSecurityScopedResource()
        defer { if accessing { url.stopAccessingSecurityScopedResource() } }
        let ext = url.pathExtension.isEmpty ? "m4a" : url.pathExtension
        let dest = persistentAudioStorageDirectory()
            .appendingPathComponent("translator_upload_\(UUID().uuidString)")
            .appendingPathExtension(ext)
        if (try? FileManager.default.copyItem(at: url, to: dest)) != nil {
            return dest
        }
        return url
    }

    private func englishName(for language: TranslatorLanguage) -> String {
        translatorLanguageEnglishNames[language.code] ?? language.code
    }

    private func rawTranslateGemmaPrompt(source: TranslatorLanguage?, target: TranslatorLanguage, text: String) -> String {
        let trimmedText = text.trimmingCharacters(in: .whitespacesAndNewlines)
        let targetName = englishName(for: target)
        let targetCode = target.code.replacingOccurrences(of: "_", with: "-")

        if let source = source {
            let sourceName = englishName(for: source)
            let sourceCode = source.code.replacingOccurrences(of: "_", with: "-")
            return "You are a professional \(sourceName) (\(sourceCode)) to \(targetName) (\(targetCode)) translator. Respond ONLY with the translation, no preamble or commentary.\n\n\(trimmedText)"
        }

        return "You are a professional translator. Detect the source language and translate the following into \(targetName) (\(targetCode)). Respond ONLY with the translation, no preamble or commentary.\n\n\(trimmedText)"
    }

    private func buildPrompt() -> String {
        let trimmedInput = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        let source = autoDetectSource ? nil : sourceLanguage
        let targetCode = targetLanguage.code.replacingOccurrences(of: "_", with: "-")
        let targetName = englishName(for: targetLanguage)
        let isTranslateGemma = selectedModel.map(isTranslateGemmaModel) ?? false

        let hasAudio = selectedAudioURL != nil && canUseAudioInput
        let hasImage = selectedImageURL != nil && enableVision && !hasAudio

        let rawPromptText: String

        if hasAudio {
            if let source = source {
                let sourceName = englishName(for: source)
                let sourceCode = source.code.replacingOccurrences(of: "_", with: "-")
                rawPromptText = "Transcribe the spoken audio in \(sourceName) (\(sourceCode)) and translate it into \(targetName) (\(targetCode)). Respond ONLY with the translated \(targetName) text and no commentary."
            } else {
                rawPromptText = "Transcribe the spoken audio and translate it into \(targetName) (\(targetCode)). Respond ONLY with the translated \(targetName) text and no commentary."
            }
        } else if hasImage {
            let srcPart: String
            if let source = source {
                let srcName = englishName(for: source)
                let srcCode = source.code.replacingOccurrences(of: "_", with: "-")
                srcPart = "You are a professional \(srcName) (\(srcCode)) to \(targetName) (\(targetCode)) translator. Your goal is to accurately convey the meaning and nuances of the original \(srcName) text while adhering to \(targetName) grammar, vocabulary, and cultural sensitivities.\nPlease translate the \(srcName) text in the provided image into \(targetName). Produce only the \(targetName) translation, without any additional explanations, alternatives or commentary. Focus only on the text, do not output where the text is located, surrounding objects or any other explanation about the picture. Ignore symbols, pictogram, and arrows!"
            } else {
                srcPart = "You are a professional translator. Your goal is to accurately convey the meaning and nuances of the original text while adhering to \(targetName) grammar, vocabulary, and cultural sensitivities.\nPlease translate the text in the provided image into \(targetName). Produce only the \(targetName) translation, without any additional explanations, alternatives or commentary. Focus only on the text, do not output where the text is located, surrounding objects or any other explanation about the picture. Ignore symbols, pictogram, and arrows!"
            }
            let extra = trimmedInput.isEmpty ? "" : "\n\(trimmedInput)"
            rawPromptText = srcPart + extra
        } else if !isTranslateGemma {
            if let source = source {
                let sourceName = englishName(for: source)
                let sourceCode = source.code.replacingOccurrences(of: "_", with: "-")
                rawPromptText = "You are a professional translator. Translate the following \(sourceName) (\(sourceCode)) text into \(targetName) (\(targetCode)). Preserve meaning and nuance. Respond with only the translated \(targetName) text and no commentary.\n\n\(trimmedInput)"
            } else {
                rawPromptText = "You are a professional translator. Detect the source language and translate the following text into \(targetName) (\(targetCode)). Preserve meaning and nuance. Respond with only the translated \(targetName) text and no commentary.\n\n\(trimmedInput)"
            }
        } else {
            rawPromptText = rawTranslateGemmaPrompt(source: source, target: targetLanguage, text: trimmedInput)
        }

        if let model = selectedModel, model.name.localizedCaseInsensitiveContains("gemma") {
            if !rawPromptText.contains("<start_of_turn>") {
                return "<start_of_turn>user\n\(rawPromptText)<end_of_turn>\n<start_of_turn>model\n"
            }
        }

        return rawPromptText
    }

    private func ensureModelLoaded(force: Bool) async {
        guard let model = selectedModel else {
            errorMessage = settings.localized("scam_detector_load_model")
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
        llm.temperature = 0.2
        llm.topP = 0.8
        llm.enableVision = enableVision
        llm.enableAudio = enableAudio && (selectedModel?.supportsAudio == true)
        llm.enableThinking = false

        do {
            if shouldReload {
                try await llm.loadModel(model)
            }
            errorMessage = nil
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    private func sanitizeTranslatorOutput(_ text: String) -> String {
        let base = sanitizeModelOutputText(text)
        let markers = [
            "\nuser:",
            "\nassistant:",
            "\nUser:",
            "\nAssistant:",
            "\n\nuser:",
            "\n\nassistant:",
            "\n\nUser:",
            "\n\nAssistant:",
            "<start_of_turn>",
            "<end_of_turn>",
            "<|turn|>",
            "<|im_start|>",
            "<|im_end|>"
        ]
        var result = base
        for marker in markers {
            if let range = result.range(of: marker) {
                result = String(result[..<range.lowerBound])
            }
        }
        if result.hasPrefix("assistant:\n") {
            result = String(result.dropFirst("assistant:\n".count))
        } else if result.hasPrefix("Assistant:\n") {
            result = String(result.dropFirst("Assistant:\n".count))
        } else if result.hasPrefix("assistant:") {
            result = String(result.dropFirst("assistant:".count))
        } else if result.hasPrefix("Assistant:") {
            result = String(result.dropFirst("Assistant:".count))
        }
        return result.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func toggleTranslate() {
        if isTranslating {
            generationTask?.cancel()
            generationTask = nil
            isTranslating = false
            return
        }

        generationTask = Task {
            guard inputHasContent else { return }

            if selectedImageURL != nil,
               enableVision,
               let model = selectedModel,
               !translatorHasDownloadedVisionProjector(for: model) {
                errorMessage = String(format: settings.localized("translator_missing_vision_projector"), model.name)
                return
            }

            await ensureModelLoaded(force: false)
            guard llm.isLoaded else { return }

            isTranslating = true
            outputText = ""

            do {
                let effectiveAudioURL = canUseAudioInput ? selectedAudioURL : nil
                let effectiveImageURL = (enableVision && effectiveAudioURL == nil) ? selectedImageURL : nil
                try await llm.generate(
                    prompt: buildPrompt(),
                    imageURL: effectiveImageURL,
                    audioURL: effectiveAudioURL,
                    maxTokensOverride: 512,
                    stopSequences: [
                        "<turn|>",
                        "<|turn>user\n",
                        "<|turn>system\n",
                        "<|turn>model\n",
                        "<end_of_turn>",
                        "<start_of_turn>",
                        "<|im_start|>",
                        "<|im_end|>",
                        "<|eot_id|>",
                        "</s>",
                        "<eos>"
                    ]
                ) { text, _, _ in
                    Task { @MainActor in
                        outputText = sanitizeTranslatorOutput(text)
                    }
                }
            } catch is CancellationError {
                // User cancelled translation.
            } catch {
                errorMessage = error.localizedDescription
            }

            isTranslating = false
            generationTask = nil
        }
    }
}

/// Selected/recorded audio clip row: play, file name, remove.
private struct MacTranslatorAudioFileRow: View {
    let url: URL
    let onRemove: () -> Void
    @StateObject private var controller = AudioPlaybackController()

    var body: some View {
        HStack(spacing: 8) {
            Button {
                controller.toggle(url: url)
            } label: {
                Image(systemName: controller.isPlaying ? "stop.fill" : "play.fill")
            }
            .buttonStyle(.bordered)

            Text(url.lastPathComponent)
                .font(.subheadline)
                .lineLimit(1)
                .truncationMode(.middle)
                .frame(maxWidth: .infinity, alignment: .leading)

            Button(role: .destructive, action: onRemove) {
                Image(systemName: "trash")
            }
            .buttonStyle(.borderless)
        }
        .padding(8)
        .background(Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: 8))
        .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color(nsColor: .separatorColor)))
    }
}
#endif
