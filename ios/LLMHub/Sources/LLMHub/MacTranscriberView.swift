//
//  MacTranscriberView.swift
//  LLMHub
//
//  Native macOS Transcriber. Mirrors `TranscriberScreen` / `IOS26TranscriberScreen`
//  on iOS: system speech recognizer by default, Gemma 4 LiteRT-LM audio input
//  when such a model is loaded, and Whisper when a Whisper model is loaded.
//  Same persisted settings, prompts and localized strings.
//

#if os(macOS)
import AppKit
import SwiftUI
import UniformTypeIdentifiers

struct MacTranscriberView: View {
    @EnvironmentObject var settings: AppSettings
    @StateObject private var transcriber = IOSSpeechTranscriber()
    @StateObject private var audioRecorder = AudioRecorder()
    @ObservedObject private var ttsManager = OnDeviceTtsManager.shared
    @ObservedObject private var llm = LLMBackend.shared
    @ObservedObject private var whisper = WhisperBackend.shared

    @AppStorage("feature_transcriber_model_name") private var selectedModelName: String = ""
    @AppStorage("feature_transcriber_max_tokens") private var maxTokens: Double = 4096

    @State private var showAudioImporter = false
    @State private var showInspector = false
    @State private var audioTranscript: String = ""
    @State private var audioHistory: [TranscriptionSession] = []
    @State private var isAudioTranscribing = false
    @State private var selectedAudioURL: URL?
    @State private var audioTranscriptionTask: Task<Void, Never>?
    @State private var isModelLoading = false
    @State private var modelLoadError: String? = nil
    @State private var whisperHistory: [TranscriptionSession] = []

    private var canStartRecording: Bool {
        !transcriber.isRecording && !transcriber.isTranscribing && !transcriber.isPreparing
    }

    private var canUploadAudio: Bool {
        !transcriber.isRecording && !transcriber.isTranscribing && !transcriber.isPreparing
    }

    private var canTranscribeUploadedAudio: Bool {
        transcriber.selectedAudioURL != nil && !transcriber.isRecording && !transcriber.isTranscribing && !transcriber.isPreparing
    }

    private var selectedModel: AIModel? {
        selectedFeatureModel(named: selectedModelName)
    }

    /// True only when the user has loaded a Gemma 4 LiteRT-LM model (same rule as iOS).
    private var useModelAudioInput: Bool {
        guard let model = selectedModel else { return false }
        return model.isGemma4LiteRTLM && llm.isLoaded && llm.currentlyLoadedModel == model.name
    }

    private var useWhisperTranscription: Bool {
        guard let model = selectedModel else { return false }
        return model.isWhisperModel && whisper.isLoaded && whisper.currentModelName == model.name
    }

    var body: some View {
        Group {
            if useWhisperTranscription {
                whisperTranscriberView
            } else if useModelAudioInput {
                gemmaAudioTranscriberView
            } else {
                systemTranscriberView
            }
        }
        .safeAreaInset(edge: .bottom, spacing: 0) {
            primaryActionBar
        }
        .navigationTitle(settings.localized("transcriber_title"))
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
                enableVision: .constant(false),
                isLoading: $isModelLoading,
                errorMessage: $modelLoadError,
                supportsVisionToggle: false,
                visionToggleTitleKey: "scam_detector_enable_vision",
                modelFilter: {
                    ($0.modelFormat == .litertlm && $0.supportsAudio && $0.isLanguageModel) || $0.isWhisperModel
                },
                onLoad: {
                    isModelLoading = true
                    modelLoadError = nil
                    if let model = selectedFeatureModel(named: selectedModelName), model.isWhisperModel {
                        await loadWhisperModel(model)
                    } else {
                        llm.isLoaded = false
                        llm.currentlyLoadedModel = nil
                        await ensureAudioModelLoaded(force: true)
                    }
                    isModelLoading = false
                },
                onUnload: {
                    whisper.unload()
                    llm.isLoaded = false
                    llm.currentlyLoadedModel = nil
                    llm.unloadModel()
                },
                showsThinkingToggle: false
            )
            .inspectorColumnWidth(min: 280, ideal: 320, max: 420)
        }
        .fileImporter(
            isPresented: $showAudioImporter,
            allowedContentTypes: [.audio],
            allowsMultipleSelection: false
        ) { result in
            switch result {
            case .success(let urls):
                if let first = urls.first {
                    if useWhisperTranscription {
                        // Copy to sandbox — fileImporter URLs lose security scope after the handler returns
                        let accessing = first.startAccessingSecurityScopedResource()
                        let ext = first.pathExtension.isEmpty ? "m4a" : first.pathExtension
                        let dest = persistentAudioStorageDirectory()
                            .appendingPathComponent("whisper_upload_\(UUID().uuidString)")
                            .appendingPathExtension(ext)
                        if let _ = try? FileManager.default.copyItem(at: first, to: dest) {
                            selectedAudioURL = dest
                        } else {
                            selectedAudioURL = first
                        }
                        if accessing { first.stopAccessingSecurityScopedResource() }
                    } else if useModelAudioInput {
                        selectedAudioURL = prepareGemmaAudioInput(
                            from: first,
                            destinationDirectory: persistentAudioStorageDirectory(),
                            filePrefix: "transcriber_audio"
                        )
                    } else {
                        transcriber.setSelectedAudioURL(first)
                    }
                }
            case .failure(let error):
                NSLog("[LLMHub][Transcriber] Audio import failed: \(error.localizedDescription)")
            }
        }
        .onDisappear {
            transcriber.cleanup()
            audioRecorder.cancelRecording()
            audioTranscriptionTask?.cancel()
            audioTranscriptionTask = nil
            if llm.isLoaded {
                llm.unloadModel()
                llm.isLoaded = false
                llm.currentlyLoadedModel = nil
            }
        }
        .onAppear {
            // Don't reset selectedModelName — preserve last-used model across visits.
            Task { await refreshDownloadedModelStatus() }
            if maxTokens < 4096 {
                maxTokens = 4096
            }
        }
    }

    // MARK: - Bottom primary action (the iOS bottom button)

    @ViewBuilder
    private var primaryActionBar: some View {
        if useWhisperTranscription {
            MacPrimaryActionBar(
                title: whisperBottomButtonTitle,
                systemImage: "waveform",
                isBusy: whisper.isTranscribing || isAudioTranscribing,
                isEnabled: !whisperBottomButtonDisabled
            ) {
                guard let url = selectedAudioURL else { return }
                audioTranscriptionTask?.cancel()
                audioTranscriptionTask = Task {
                    isAudioTranscribing = true
                    defer { isAudioTranscribing = false }
                    do {
                        let text = try await whisper.transcribeFile(url: url)
                        if !text.isEmpty {
                            whisperHistory.append(TranscriptionSession(id: UUID(), text: text, timestamp: Date()))
                        }
                        selectedAudioURL = nil
                    } catch {
                        NSLog("[WhisperBackend] transcription error: \(error)")
                    }
                }
            }
        } else if useModelAudioInput {
            MacPrimaryActionBar(
                title: gemmaButtonTitle,
                systemImage: audioRecorder.isRecording ? "stop.fill" : "waveform",
                isBusy: audioRecorder.isPreparing || isAudioTranscribing,
                isEnabled: !gemmaButtonDisabled,
                tint: (audioRecorder.isRecording || isAudioTranscribing) ? .red : ApolloPalette.accentStrong
            ) {
                if audioRecorder.isRecording {
                    if let url = audioRecorder.stopRecording() {
                        selectedAudioURL = url
                        Task { await transcribeAudio(url) }
                    }
                } else if isAudioTranscribing {
                    audioTranscriptionTask?.cancel()
                    audioTranscriptionTask = nil
                    isAudioTranscribing = false
                } else if let selectedAudioURL {
                    Task { await transcribeAudio(selectedAudioURL) }
                }
            }
        } else {
            MacPrimaryActionBar(
                title: buttonTitle,
                systemImage: transcriber.isRecording ? "stop.fill" : "waveform",
                isBusy: transcriber.isPreparing || transcriber.isTranscribing,
                isEnabled: !isBottomButtonDisabled,
                tint: (transcriber.isRecording || transcriber.isTranscribing) ? .red : ApolloPalette.accentStrong
            ) {
                if transcriber.isRecording {
                    Task { await transcriber.stopLiveTranscription() }
                } else if transcriber.isTranscribing {
                    transcriber.cancelTranscription()
                } else if canTranscribeUploadedAudio {
                    Task { @MainActor in
                        await transcriber.transcribeSelectedAudio()
                    }
                }
            }
        }
    }

    // MARK: - Shared building blocks

    private func recorderPanel(
        isRecording: Bool,
        isPreparing: Bool,
        recordDisabled: Bool,
        uploadDisabled: Bool,
        audioURL: URL?,
        onRecord: @escaping () -> Void,
        onClearAudio: @escaping () -> Void
    ) -> some View {
        HStack(alignment: .center, spacing: 16) {
            Button(action: onRecord) {
                ZStack {
                    Circle()
                        .fill(isRecording ? Color.red.opacity(0.18) : Color(nsColor: .controlBackgroundColor))
                        .overlay(Circle().stroke(Color(nsColor: .separatorColor)))
                        .frame(width: 72, height: 72)
                    if isPreparing {
                        ProgressView().controlSize(.small)
                    } else {
                        Image(systemName: isRecording ? "stop.fill" : "mic.fill")
                            .font(.system(size: 26, weight: .bold))
                            .foregroundStyle(isRecording ? Color.red : Color.primary)
                    }
                }
                .contentShape(Circle())
            }
            .buttonStyle(.plain)
            .disabled(recordDisabled)

            VStack(alignment: .leading, spacing: 8) {
                Text(
                    isPreparing
                        ? settings.localized("processing")
                        : isRecording
                        ? settings.localized("transcriber_recording")
                        : settings.localized("transcriber_record")
                )
                .font(.headline)

                Button {
                    showAudioImporter = true
                } label: {
                    Label(settings.localized("transcriber_upload"), systemImage: "waveform.badge.plus")
                }
                .buttonStyle(.bordered)
                .disabled(uploadDisabled)
            }

            if let audioURL {
                MacTranscriberAudioFileRow(url: audioURL, onRemove: onClearAudio)
                    .frame(maxWidth: 420)
            }

            Spacer(minLength: 0)
        }
        .padding()
    }

    @ViewBuilder
    private func transcriptionBoxView(for text: String) -> some View {
        let speechKey = "transcriber-\(text)"
        let isLive = transcriber.isRecording || audioRecorder.isRecording
        VStack(alignment: .leading, spacing: 8) {
            Text(text.isEmpty ? (isLive ? "..." : "-") : text)
                .textSelection(.enabled)
                .foregroundStyle(text.isEmpty ? .secondary : .primary)
                .frame(maxWidth: .infinity, alignment: .leading)
                .frame(minHeight: 80, alignment: .topLeading)

            if !text.isEmpty {
                HStack(spacing: 6) {
                    Spacer()
                    Button {
                        ttsManager.toggleSpeaking(
                            text,
                            fallbackLanguage: settings.selectedLanguage,
                            key: speechKey
                        )
                    } label: {
                        Image(systemName: ttsManager.isSpeaking(key: speechKey) ? "stop.fill" : "speaker.wave.2")
                    }

                    Button {
                        NSPasteboard.general.clearContents()
                        NSPasteboard.general.setString(text, forType: .string)
                    } label: {
                        Image(systemName: "doc.on.doc")
                    }
                }
                .buttonStyle(.bordered)
            }
        }
        .padding(12)
        .background(Color(nsColor: .textBackgroundColor).opacity(0.2), in: RoundedRectangle(cornerRadius: 8))
        .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color(nsColor: .separatorColor)))
    }

    // MARK: - System speech transcriber

    private var systemTranscriberView: some View {
        VStack(spacing: 0) {
            recorderPanel(
                isRecording: transcriber.isRecording,
                isPreparing: transcriber.isPreparing,
                recordDisabled: !transcriber.isRecording && !canStartRecording,
                uploadDisabled: !canUploadAudio,
                audioURL: transcriber.selectedAudioURL,
                onRecord: {
                    if transcriber.isRecording {
                        Task { await transcriber.stopLiveTranscription() }
                    } else if canStartRecording {
                        Task { @MainActor in
                            await transcriber.startLiveTranscription()
                        }
                    }
                },
                onClearAudio: { transcriber.clearSelectedAudio() }
            )

            Divider()

            ScrollViewReader { proxy in
                ScrollView {
                    VStack(alignment: .leading, spacing: 12) {
                        ForEach(transcriber.history) { session in
                            transcriptionBoxView(for: session.text)
                                .id(session.id)
                        }

                        if !transcriber.transcript.isEmpty || transcriber.isRecording || transcriber.isPreparing || transcriber.isTranscribing {
                            transcriptionBoxView(for: transcriber.transcript)
                                .id("current_box")
                        }
                    }
                    .padding()
                }
                .onChange(of: transcriber.transcript) { _, _ in
                    withAnimation {
                        proxy.scrollTo("current_box", anchor: .bottom)
                    }
                }
                .onChange(of: transcriber.history.count) { _, _ in
                    if let lastId = transcriber.history.last?.id {
                        withAnimation {
                            proxy.scrollTo(lastId, anchor: .bottom)
                        }
                    }
                }
            }
        }
    }

    private var buttonTitle: String {
        if transcriber.isRecording {
            return settings.localized("transcriber_stop")
        }
        if transcriber.isPreparing {
            return settings.localized("processing")
        }
        if transcriber.isTranscribing {
            return settings.localized("transcribing_tap_to_cancel")
        }
        if transcriber.selectedAudioURL != nil {
            return settings.localized("transcriber_transcribe")
        }
        return settings.localized("transcriber_record")
    }

    private var isBottomButtonDisabled: Bool {
        if transcriber.isRecording || transcriber.isTranscribing || transcriber.isPreparing {
            return false
        }
        return transcriber.selectedAudioURL == nil
    }

    // MARK: - Gemma 4 audio transcriber

    private var gemmaAudioTranscriberView: some View {
        VStack(spacing: 0) {
            recorderPanel(
                isRecording: audioRecorder.isRecording,
                isPreparing: audioRecorder.isPreparing,
                recordDisabled: audioRecorder.isPreparing || isAudioTranscribing,
                uploadDisabled: audioRecorder.isRecording || audioRecorder.isPreparing || isAudioTranscribing,
                audioURL: selectedAudioURL,
                onRecord: {
                    if audioRecorder.isRecording {
                        _ = audioRecorder.stopRecording()
                    } else {
                        Task { @MainActor in
                            let isGemma4 = useModelAudioInput
                            let ext = isGemma4 ? "wav" : "m4a"
                            let destination = persistentAudioStorageDirectory()
                                .appendingPathComponent("transcriber_audio_\(UUID().uuidString)")
                                .appendingPathExtension(ext)
                            _ = await audioRecorder.startRecording(
                                outputURL: destination,
                                autoStopAfterSilence: false,
                                isFloat32Wav: isGemma4
                            ) { url in
                                Task { @MainActor in
                                    selectedAudioURL = url
                                }
                                Task { await transcribeAudio(url) }
                            }
                        }
                    }
                },
                onClearAudio: { self.selectedAudioURL = nil }
            )

            Divider()

            ScrollViewReader { proxy in
                ScrollView {
                    VStack(alignment: .leading, spacing: 12) {
                        ForEach(audioHistory) { session in
                            transcriptionBoxView(for: session.text)
                                .id(session.id)
                        }

                        if !audioTranscript.isEmpty || audioRecorder.isRecording || audioRecorder.isPreparing || isAudioTranscribing {
                            transcriptionBoxView(for: audioTranscript)
                                .id("current_box")
                        }
                    }
                    .padding()
                }
                .onChange(of: audioTranscript) { _, _ in
                    withAnimation {
                        proxy.scrollTo("current_box", anchor: .bottom)
                    }
                }
                .onChange(of: audioHistory.count) { _, _ in
                    if let lastId = audioHistory.last?.id {
                        withAnimation {
                            proxy.scrollTo(lastId, anchor: .bottom)
                        }
                    }
                }
            }
        }
    }

    private var gemmaButtonTitle: String {
        if audioRecorder.isRecording {
            return settings.localized("transcriber_stop")
        }
        if audioRecorder.isPreparing {
            return settings.localized("processing")
        }
        if isAudioTranscribing {
            return settings.localized("transcribing_tap_to_cancel")
        }
        if selectedAudioURL != nil {
            return settings.localized("transcriber_transcribe")
        }
        return settings.localized("transcriber_record")
    }

    private var gemmaButtonDisabled: Bool {
        if audioRecorder.isRecording || isAudioTranscribing || audioRecorder.isPreparing {
            return false
        }
        return selectedAudioURL == nil
    }

    private func transcribeAudio(_ url: URL) async {
        audioTranscriptionTask?.cancel()
        audioTranscriptionTask = nil

        await ensureAudioModelLoaded(force: false)
        guard llm.isLoaded else { return }

        isAudioTranscribing = true
        audioTranscript = ""

        let audioInputURL: URL? = useModelAudioInput
            ? prepareGemmaAudioInput(
                from: url,
                destinationDirectory: FileManager.default.temporaryDirectory,
                filePrefix: "transcribe_audio"
            )
            : url

        guard let audioInputURL else {
            isAudioTranscribing = false
            return
        }

        audioTranscriptionTask = Task {
            var latest = ""
            do {
                try await llm.generate(
                    prompt: "Transcribe this audio.",
                    audioURL: audioInputURL,
                    maxTokensOverride: Int(max(maxTokens, 4096))
                ) { text, _, _ in
                    Task { @MainActor in
                        latest = sanitizeModelOutputText(text)
                        audioTranscript = latest
                    }
                }
            } catch is CancellationError {
                // User cancelled.
            } catch {
                NSLog("[LLMHub][Transcriber] Audio transcription failed: \(error.localizedDescription)")
            }

            await MainActor.run {
                let final = latest.trimmingCharacters(in: .whitespacesAndNewlines)
                if !final.isEmpty {
                    audioHistory.append(TranscriptionSession(id: UUID(), text: final, timestamp: Date()))
                }
                audioTranscript = ""
                isAudioTranscribing = false
                audioTranscriptionTask = nil
            }
        }
    }

    private func ensureAudioModelLoaded(force: Bool) async {
        guard let model = selectedModel else { return }
        let modelContextCap = contextLimitForFeatureModel(model)
        let effectiveTokens = maxTokens < 4096 ? 4096 : maxTokens
        let effectiveContext = min(max(1, Int(effectiveTokens)), modelContextCap)
        let shouldReload = force
            || llm.currentlyLoadedModel != model.name
            || llm.loadedContextWindow != effectiveContext

        llm.maxTokens = min(Int(effectiveTokens), effectiveContext)
        llm.contextWindow = effectiveContext
        llm.enableVision = false
        // Auto-enable audio when a Gemma4 LiteRT-LM model is selected; no toggle needed.
        llm.enableAudio = model.isGemma4LiteRTLM
        llm.enableThinking = false

        if shouldReload {
            do {
                try await llm.loadModel(model)
            } catch {
                modelLoadError = error.localizedDescription
            }
        }
    }

    private func loadWhisperModel(_ model: AIModel) async {
        guard model.isWhisperModel else { return }
        do {
            let modelDir = try SimplifiedFileManager.shared.getModelFolderURL(modelId: model.id, framework: .llamaCpp)
            let fileName = URL(string: model.url)?.lastPathComponent ?? "ggml-model.bin"
            let modelPath = modelDir.appendingPathComponent(fileName).path
            try await whisper.load(modelPath: modelPath, modelName: model.name)
        } catch {
            modelLoadError = error.localizedDescription
        }
    }

    // MARK: - Whisper transcriber

    private var whisperTranscriberView: some View {
        VStack(spacing: 0) {
            recorderPanel(
                isRecording: audioRecorder.isRecording,
                isPreparing: audioRecorder.isPreparing,
                recordDisabled: audioRecorder.isPreparing || whisper.isTranscribing || isAudioTranscribing,
                uploadDisabled: audioRecorder.isRecording || whisper.isTranscribing || isAudioTranscribing,
                audioURL: selectedAudioURL,
                onRecord: {
                    if audioRecorder.isRecording {
                        _ = audioRecorder.stopRecording()
                    } else {
                        Task { @MainActor in
                            let destination = persistentAudioStorageDirectory()
                                .appendingPathComponent("whisper_audio_\(UUID().uuidString)")
                                .appendingPathExtension("m4a")
                            _ = await audioRecorder.startRecording(
                                outputURL: destination,
                                autoStopAfterSilence: false,
                                isFloat32Wav: false
                            ) { url in
                                selectedAudioURL = url
                            }
                        }
                    }
                },
                onClearAudio: { selectedAudioURL = nil }
            )

            Divider()

            ScrollView {
                VStack(alignment: .leading, spacing: 12) {
                    ForEach(whisperHistory) { session in
                        transcriptionBoxView(for: session.text)
                    }
                    if whisper.isTranscribing || isAudioTranscribing {
                        transcriptionBoxView(for: "")
                    }
                }
                .padding()
            }
        }
    }

    private var whisperBottomButtonTitle: String {
        if whisper.isTranscribing || isAudioTranscribing { return settings.localized("whisper_transcribing") }
        if selectedAudioURL != nil { return settings.localized("transcriber_transcribe") }
        return settings.localized("transcriber_record")
    }

    private var whisperBottomButtonDisabled: Bool {
        if whisper.isTranscribing || isAudioTranscribing { return false }
        return selectedAudioURL == nil
    }
}

/// Selected/recorded audio clip row: play, file name, remove.
private struct MacTranscriberAudioFileRow: View {
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
