//
//  MacVibeVoiceView.swift
//  LLMHub
//
//  Native macOS VibeVoice. Mirrors `VibeVoiceScreen` / `IOS17VibeVoiceScreen`
//  on iOS: same persisted settings, voice loop, prompt templates, TTS
//  streaming and localized strings. The logic section is kept identical to iOS.
//

#if os(macOS)
import AppKit
import SwiftUI

private struct MacVibeVoiceReplyHeightKey: PreferenceKey {
    static let defaultValue: CGFloat = 0
    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) {
        value = max(value, nextValue())
    }
}

struct MacVibeVoiceView: View {
    @EnvironmentObject var settings: AppSettings
    @ObservedObject private var llm = LLMBackend.shared
    @ObservedObject private var ttsManager = OnDeviceTtsManager.shared
    @AppStorage("feature_vibevoice_model_name") private var selectedModelName: String = ""
    @AppStorage("feature_vibevoice_max_tokens") private var maxTokens: Double = 512
    @AppStorage("feature_vibevoice_enable_model_audio") private var enableModelAudio: Bool = true
    @State private var voiceState: VibeVoiceState = .idle
    @State private var isChatActive = false
    @State private var latestReply = ""
    @State private var isLoading = false
    @State private var errorMessage: String?
    @State private var showInspector = false
    @State private var generationTask: Task<Void, Never>?
    @State private var conversationHistory: [(role: String, content: String)] = []
    @StateObject private var transcriber = IOSVibeVoiceTranscriber()
    @StateObject private var audioRecorder = AudioRecorder()
    @State private var lastRecordedAudioURL: URL?
    @State private var replyContentHeight: CGFloat = 0
    @State private var ttsReadCursor = 0

    private let ttsKey = "vibevoice-reply"

    private var isCurrentModelLoaded: Bool {
        llm.isLoaded && llm.currentlyLoadedModel == selectedModelName
    }

    private var useModelAudioInput: Bool {
        guard enableModelAudio else { return false }
        guard let model = selectedFeatureModel(named: selectedModelName) else { return false }
        return model.isGemma4LiteRTLM
    }

    var body: some View {
        Group {
            if !isCurrentModelLoaded {
                MacFeatureLoadModelPrompt { showInspector = true }
            } else {
                chatView
                    .safeAreaInset(edge: .bottom, spacing: 0) {
                        MacPrimaryActionBar(
                            title: isChatActive ? settings.localized("vibevoice_tap_to_stop")
                                : settings.localized("vibevoice_tap_to_start"),
                            systemImage: isChatActive ? "stop.fill" : "mic",
                            tint: isChatActive ? .red : ApolloPalette.accentStrong,
                            action: handleGlobeTap
                        )
                    }
            }
        }
        .navigationTitle(settings.localized("feature_vibevoice"))
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
                enableAudio: $enableModelAudio,
                isLoading: $isLoading,
                errorMessage: $errorMessage,
                supportsVisionToggle: false,
                visionToggleTitleKey: "scam_detector_enable_vision",
                modelFilter: isNonTranslatorFeatureModel,
                onLoad: { await ensureModelLoaded(force: false) },
                onUnload: { llm.unloadModel() },
                showsThinkingToggle: false
            )
            .inspectorColumnWidth(min: 280, ideal: 320, max: 420)
        }
        .onAppear {
            Task {
                let available = downloadableFeatureModels().filter(isNonTranslatorFeatureModel)
                // Preserve a valid last-used LLM, but clear any stale selection
                // left by older builds that exposed dedicated media models here.
                if selectedModelName.isEmpty || !available.contains(where: { $0.name == selectedModelName }) {
                    selectedModelName = available.first?.name ?? ""
                }
            }
        }
        .onDisappear {
            stopAll()
            generationTask?.cancel()
            generationTask = nil
            llm.unloadModel()
        }
        .onChange(of: ttsManager.isSpeaking) { oldValue, newValue in
            NSLog("[LLMHub][VibeVoice] onChange(ttsManager.isSpeaking): \(oldValue) -> \(newValue), current voiceState: \(voiceState)")
            guard isChatActive else { return }
            if !oldValue && newValue {
                if voiceState == .responding || voiceState == .idle {
                    voiceState = .speaking
                }
            } else if oldValue && !newValue && voiceState == .speaking {
                NSLog("[LLMHub][VibeVoice] TTS finished speaking. Setting voiceState to .idle and scheduling startListeningCycle()")
                voiceState = .idle
                Task { @MainActor in
                    // 700ms: give audio hardware time to switch from playback → record
                    try? await Task.sleep(nanoseconds: 700_000_000)
                    guard isChatActive && isCurrentModelLoaded else { return }
                    NSLog("[LLMHub][VibeVoice] Triggering startListeningCycle() from TTS onChange")
                    await startListeningCycle()
                }
            }
        }
        .onChange(of: transcriber.isRecording) { oldValue, newValue in
            NSLog("[LLMHub][VibeVoice] onChange(transcriber.isRecording): \(oldValue) -> \(newValue), voiceState: \(voiceState), isPreparing: \(transcriber.isPreparing)")
            guard !useModelAudioInput else { return }
            if oldValue && !newValue && !transcriber.isPreparing {
                if voiceState == .listening {
                    NSLog("[LLMHub][VibeVoice] Recording stopped while in .listening. Setting voiceState to .idle")
                    voiceState = .idle
                }
                // Auto-restart the listen cycle if chat is still active
                // (covers "No speech detected" and other engine errors)
                guard isChatActive && (voiceState == .idle || voiceState == .listening) else { return }
                Task { @MainActor in
                    try? await Task.sleep(nanoseconds: 700_000_000)
                    guard isChatActive && isCurrentModelLoaded && voiceState == .idle else { return }
                    NSLog("[LLMHub][VibeVoice] Auto-restarting startListeningCycle() from recording onChange")
                    await startListeningCycle()
                }
            }
        }
    }

    // MARK: - Chat View

    @ViewBuilder
    private var chatView: some View {
        VStack(spacing: 0) {
            Spacer()

            // Animated globe (start / stop the voice conversation)
            Button {
                handleGlobeTap()
            } label: {
                TimelineView(.animation) { timeline in
                    let t = timeline.date.timeIntervalSinceReferenceDate
                    let period: Double = voiceState == .listening ? 0.52 : voiceState == .responding ? 0.9 : 1.8
                    let phase = (sin(t * .pi / period) + 1.0) / 2.0
                    let maxGlow: Double = voiceState == .listening ? 0.75 : voiceState == .responding ? 0.48 : 0.28
                    let glow = 0.22 + (maxGlow - 0.22) * phase
                    let maxScale: CGFloat = voiceState == .listening ? 1.08 : voiceState == .responding ? 1.04 : 1.0
                    let scale = 1.0 + (maxScale - 1.0) * CGFloat(phase)

                    ZStack {
                        Circle()
                            .fill(
                                RadialGradient(
                                    colors: [
                                        Color(hex: "F7EFE1").opacity(glow),
                                        Color(hex: "69C6FF").opacity(glow),
                                        Color(hex: "1478F4").opacity(glow * 0.4)
                                    ],
                                    center: .center,
                                    startRadius: 0,
                                    endRadius: 124
                                )
                            )
                            .frame(width: 248, height: 248)

                        Circle()
                            .fill(
                                RadialGradient(
                                    colors: [
                                        Color(hex: "E5F8FF"),
                                        Color(hex: "4FAAF8"),
                                        Color(hex: "0E67E8")
                                    ],
                                    center: .center,
                                    startRadius: 0,
                                    endRadius: 109
                                )
                            )
                            .frame(width: 218, height: 218)
                            .scaleEffect(scale)
                            .overlay {
                                Image(systemName: globeIcon)
                                    .font(.system(size: 72, weight: .semibold))
                                    .foregroundStyle(Color.white.opacity(0.92))
                            }
                    }
                }
                .frame(width: 248, height: 248)
                .contentShape(Circle())
            }
            .buttonStyle(.plain)
            .accessibilityLabel(statusText)

            // Status text
            Text(statusText)
                .font(.headline)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .padding(.top, 24)

            // Live partial transcript while listening
            if !useModelAudioInput && !transcriber.transcript.isEmpty && voiceState == .listening {
                Text(transcriber.transcript)
                    .font(.footnote)
                    .foregroundStyle(.tertiary)
                    .lineLimit(2)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, 32)
                    .padding(.top, 8)
                    .transition(.opacity)
                    .animation(.easeInOut(duration: 0.2), value: transcriber.transcript)
            }

            if let lastRecordedAudioURL, !useModelAudioInput && lastRecordedAudioURL.pathExtension.lowercased() != "wav" {
                HStack(spacing: 8) {
                    MacVibeVoicePlaybackButton(url: lastRecordedAudioURL)
                    Text(lastRecordedAudioURL.lastPathComponent)
                        .font(.caption)
                        .lineLimit(1)
                        .truncationMode(.middle)
                }
                .padding(.horizontal, 24)
                .padding(.top, 8)
            }

            // Latest AI reply card
            if !latestReply.isEmpty {
                ScrollViewReader { proxy in
                    ScrollView {
                        Text(latestReply)
                            .font(.body)
                            .textSelection(.enabled)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .background(
                                GeometryReader { geo in
                                    Color.clear
                                        .preference(key: MacVibeVoiceReplyHeightKey.self, value: geo.size.height)
                                }
                            )
                            .id("bottom")
                    }
                    .onPreferenceChange(MacVibeVoiceReplyHeightKey.self) { height in
                        replyContentHeight = height
                    }
                    .onChange(of: latestReply) { _, _ in
                        withAnimation {
                            proxy.scrollTo("bottom", anchor: .bottom)
                        }
                    }
                    .frame(height: min(replyContentHeight, 250))
                    .padding(14)
                    .background(Color(nsColor: .textBackgroundColor).opacity(0.2), in: RoundedRectangle(cornerRadius: 12))
                    .overlay(RoundedRectangle(cornerRadius: 12).stroke(Color(nsColor: .separatorColor)))
                    .frame(maxWidth: 640)
                    .padding(.horizontal, 24)
                    .padding(.top, 20)
                    .transition(.opacity.combined(with: .move(edge: .bottom)))
                    .animation(.spring(duration: 0.3), value: latestReply)
                }
            }

            Spacer()
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding(.bottom, 20)
    }

    // MARK: - Computed

    private var globeIcon: String {
        if !isChatActive { return "mic" }
        switch voiceState {
        case .listening: return "mic.fill"
        case .responding: return "ellipsis"
        case .speaking: return "speaker.wave.2.fill"
        case .idle: return "mic"
        }
    }

    private var statusText: String {
        if !isChatActive { return settings.localized("vibevoice_tap_to_start") }
        switch voiceState {
        case .listening: return settings.localized("vibevoice_listening")
        case .responding: return settings.localized("vibevoice_responding")
        case .speaking: return settings.localized("vibevoice_speaking")
        case .idle: return settings.localized("vibevoice_tap_to_start")
        }
    }

    // MARK: - Actions

    private func handleGlobeTap() {
        if isChatActive {
            isChatActive = false
            stopAll()
        } else {
            isChatActive = true
            Task { @MainActor in
                await startListeningCycle()
            }
        }
    }

    private func startListeningCycle() async {
        NSLog("[LLMHub][VibeVoice] startListeningCycle() entered. isChatActive: \(isChatActive), isCurrentModelLoaded: \(isCurrentModelLoaded)")
        guard isChatActive && isCurrentModelLoaded else { return }

        voiceState = .listening
        if useModelAudioInput {
            NSLog("[LLMHub][VibeVoice] using model audio input")
            let isGemma4 = useModelAudioInput
            let ext = isGemma4 ? "wav" : "m4a"
            let destination = persistentAudioStorageDirectory()
                .appendingPathComponent("vibevoice_audio_\(UUID().uuidString)")
                .appendingPathExtension(ext)
            _ = await audioRecorder.startRecording(
                outputURL: destination,
                autoStopAfterSilence: true,
                isFloat32Wav: isGemma4
            ) { url in
                Task { @MainActor in
                    self.lastRecordedAudioURL = url
                }
                Task {
                    await self.handleAudioTranscript(url)
                }
            }

            if !audioRecorder.isRecording && !audioRecorder.isPreparing {
                NSLog("[LLMHub][VibeVoice] AudioRecorder failed to start. Resetting to idle.")
                voiceState = .idle
            }
        } else {
            NSLog("[LLMHub][VibeVoice] calling transcriber.startListening()")
            await transcriber.startListening { text in
                Task { @MainActor in
                    guard self.isChatActive else { return }
                    await self.handleTranscript(text)
                }
            }
            NSLog("[LLMHub][VibeVoice] transcriber.startListening() returned. isRecording: \(transcriber.isRecording), isPreparing: \(transcriber.isPreparing)")

            if !transcriber.isRecording && !transcriber.isPreparing {
                NSLog("[LLMHub][VibeVoice] Transcriber failed to start. Resetting to idle.")
                voiceState = .idle
            }
        }
    }

    private func handleTranscript(_ text: String) async {
        guard isChatActive else { return }
        voiceState = .responding
        latestReply = ""
        ttsReadCursor = 0

        await ensureModelLoaded(force: false)
        guard llm.isLoaded && isChatActive else {
            voiceState = .idle
            return
        }

        let systemPrompt = """
            You are VibeVoice, a natural real-time voice conversation assistant.
            Keep responses short, conversational, and useful.
            Match response length to the user's request: brief for simple questions, fuller for detail requests.
            Do not repeat the user's words verbatim.
            Do not output role labels like 'assistant:' or 'user:'.
            If the input is unclear, ask one brief clarification question.
            """

        // 1. Record the newest turn
        conversationHistory.append((role: "user", content: text))

        // 2. Context Management (Sliding Window)
        // Same smarter budget as AI Chat: reserve min(maxTokens, ctx/4) for response
        // so history keeps room for prior assistant turns. Reserving the full
        // maxTokens cap starves history and makes the model "forget" its own replies.
        let effectiveCtxTokens = llm.loadedContextWindow ?? 2048
        let reservedForResponse = max(256, min(Int(maxTokens), effectiveCtxTokens / 4))
        let reservedForCurrent = max(32, text.count / 3) + 64
        let reservedSafety = 128
        let availableHistoryTokens = max(128, effectiveCtxTokens - reservedForResponse - reservedForCurrent - reservedSafety)
        let maxHistoryChars = availableHistoryTokens * 3
        var currentChars = 0
        var truncatedHistory: [(role: String, content: String)] = []
        for msg in conversationHistory.reversed() {
            let msgLen = msg.content.count
            let remaining = maxHistoryChars - currentChars
            if msgLen <= remaining {
                truncatedHistory.insert(msg, at: 0)
                currentChars += msgLen
            } else if remaining > 300 {
                // Message exceeds remaining budget — truncate its MIDDLE so both
                // the opening and closing of a long reply are preserved.
                let half = remaining / 2
                let elided = (role: msg.role, content: String(msg.content.prefix(half)) + "\n…\n" + String(msg.content.suffix(half)))
                truncatedHistory.insert(elided, at: 0)
                currentChars = maxHistoryChars
                break
            } else {
                break
            }
        }

        // 3. Family Detection
        let modelName = selectedModelName.lowercased()
        let modelSupportsThinking = selectedFeatureModel(named: selectedModelName)?.supportsThinking == true
        let isGemma      = modelName.contains("gemma")
        let isGemma4     = isGemma && (modelName.contains("gemma 4") || modelName.contains("gemma-4")) && !modelName.contains("translate")
        let isLlama      = modelName.contains("llama") || modelName.contains("mistral")
        let isLlama3     = isLlama && (modelName.contains("llama-3") || modelName.contains("llama 3") || modelName.contains("llama-3."))
        let isHarmonyModel = modelName.contains("gpt-oss") || modelName.contains("gpt_oss")
        let isMuseGlimmer = selectedFeatureModel(named: selectedModelName)?.chatTemplateFamily == .museGlimmer
        let isGranite42  = modelName.contains("granite-4.2") || modelName.contains("granite 4.2")
        let isGranite    = modelName.contains("granite") && !isGranite42
        let isPhi4       = modelName.contains("phi-4") || modelName.contains("phi 4") || modelName.contains("phi4")
        // LFM (Liquid AI) and similar ChatML-style models
        let isChatML     = modelName.contains("lfm") || modelName.contains("liquid")

        // 4. Build Raw Prompt (Prepend __RAW_PROMPT__ to bypass SDK auto-formatting)
        var parts: [String] = ["__RAW_PROMPT__"]

        if isHarmonyModel {
            var harmonyParts: [String] = []
            harmonyParts.append("<|start|>system<|message|>\(systemPrompt)<|end|>")

            for msg in truncatedHistory {
                let content = msg.content.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !content.isEmpty else { continue }
                let role = msg.role == "user" ? "user" : "assistant"
                harmonyParts.append("<|start|>\(role)<|message|>\(content)<|end|>")
            }

            // VibeVoice always disables thinking — no reasoning should be generated or shown
            harmonyParts.append("<|start|>assistant<|channel|>analysis<|message|><|end|><|start|>assistant<|channel|>final<|message|>")
            parts.append(contentsOf: harmonyParts)
            let multiTurnPrompt = parts.joined()

            generationTask = Task {
                do {
                    let savedThinking = self.llm.enableThinking
                    self.llm.enableThinking = false
                    defer { self.llm.enableThinking = savedThinking }
                    try await llm.generate(
                        prompt: multiTurnPrompt,
                        maxTokensOverride: Int(maxTokens)
                    ) { content, _, _ in
                        Task { @MainActor in
                            let sanitized = sanitizeModelOutputText(content)
                            let answerSoFar = getDisplayContentWithoutThinking(sanitized)
                            self.latestReply = answerSoFar

                            let delta = String(answerSoFar.dropFirst(self.ttsReadCursor))
                            if !delta.isEmpty {
                                self.ttsReadCursor = answerSoFar.count
                                self.ttsManager.addStreamingToken(delta, fallbackLanguage: self.settings.selectedLanguage, key: self.ttsKey)
                            }
                        }
                    }
                } catch {
                    NSLog("[LLMHub][VibeVoice] LLM error: \(error.localizedDescription)")
                }
                await MainActor.run {
                    self.generationTask = nil
                    guard self.isChatActive else { return }

                    let displayContent = getDisplayContentWithoutThinking(self.latestReply)
                    if displayContent.count > self.ttsReadCursor {
                        let delta = String(displayContent.dropFirst(self.ttsReadCursor))
                        if !delta.isEmpty {
                            self.ttsManager.addStreamingToken(delta, fallbackLanguage: self.settings.selectedLanguage, key: self.ttsKey)
                        }
                    }
                    self.ttsManager.flushStreamingBuffer(fallbackLanguage: self.settings.selectedLanguage, key: self.ttsKey)
                    self.ttsReadCursor = 0

                    let rawReply = self.latestReply.trimmingCharacters(in: .whitespacesAndNewlines)
                    let replyForHistory = getDisplayContentWithoutThinking(rawReply)
                    let finalReply = replyForHistory.isEmpty ? rawReply : replyForHistory
                    if !finalReply.isEmpty && !contentHasThinkingMarkers(finalReply) {
                        self.latestReply = finalReply
                        self.conversationHistory.append((role: "assistant", content: finalReply))
                        if self.ttsManager.isSpeaking(key: self.ttsKey) {
                            self.voiceState = .speaking
                        } else {
                            self.voiceState = .idle
                            Task { @MainActor in
                                try? await Task.sleep(nanoseconds: 350_000_000)
                                if self.isChatActive {
                                    await self.startListeningCycle()
                                }
                            }
                        }
                    } else {
                        self.latestReply = ""
                        self.voiceState = .idle
                        Task { @MainActor in
                            try? await Task.sleep(nanoseconds: 350_000_000)
                            if self.isChatActive {
                                await self.startListeningCycle()
                            }
                        }
                    }
                }
            }
            return
        }

        if isMuseGlimmer {
            let dateFormatter = DateFormatter()
            dateFormatter.calendar = Calendar(identifier: .gregorian)
            dateFormatter.locale = Locale(identifier: "en_US_POSIX")
            dateFormatter.dateFormat = "yyyy-MM-dd"

            var museParts: [String] = []
            museParts.append("<|begin_of_text|>")
            museParts.append("<|start|>system<|message|>\(systemPrompt)\nKnowledge cutoff: 2026-01-04.\nCurrent date: \(dateFormatter.string(from: Date())).\n\nReasoning strength: high.\n\n# Valid recipients: \"user\".<|eot|>")

            for msg in truncatedHistory {
                let content = msg.content.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !content.isEmpty else { continue }
                if msg.role == "user" {
                    museParts.append("<|start|>user<|message|>\(content)<|eot|>")
                } else {
                    museParts.append("<|start|>assistant to=user<|message|>\(content)<|eot|>")
                }
            }

            museParts.append("<|start|>user<|message|>\(text)<|eot|>")
            museParts.append("<|start|>assistant to=user<|message|>")
            parts.append(contentsOf: museParts)
            let multiTurnPrompt = parts.joined()

            generationTask = Task {
                do {
                    let savedThinking = self.llm.enableThinking
                    self.llm.enableThinking = false
                    defer { self.llm.enableThinking = savedThinking }
                    try await llm.generate(
                        prompt: multiTurnPrompt,
                        maxTokensOverride: Int(maxTokens)
                    ) { content, _, _ in
                        Task { @MainActor in
                            let sanitized = sanitizeModelOutputText(content)
                            let answerSoFar = getDisplayContentWithoutThinking(sanitized)
                            self.latestReply = answerSoFar.isEmpty ? sanitized : answerSoFar

                            let delta = String(self.latestReply.dropFirst(self.ttsReadCursor))
                            if !delta.isEmpty {
                                self.ttsReadCursor = self.latestReply.count
                                self.ttsManager.addStreamingToken(delta, fallbackLanguage: self.settings.selectedLanguage, key: self.ttsKey)
                            }
                        }
                    }
                } catch {
                    NSLog("[LLMHub][VibeVoice] LLM error: \(error.localizedDescription)")
                }
                await MainActor.run {
                    self.generationTask = nil
                    guard self.isChatActive else { return }

                    let finalReply = self.latestReply.trimmingCharacters(in: .whitespacesAndNewlines)
                    if !finalReply.isEmpty {
                        self.conversationHistory.append((role: "assistant", content: finalReply))
                        self.ttsManager.flushStreamingBuffer(fallbackLanguage: self.settings.selectedLanguage, key: self.ttsKey)
                        self.ttsReadCursor = 0
                        self.voiceState = self.ttsManager.isSpeaking(key: self.ttsKey) ? .speaking : .idle
                    } else {
                        self.voiceState = .idle
                    }
                }
            }
            return
        }

        // When using RAW_PROMPT, the SDK's systemPrompt argument is ignored,
        // so we must inject it manually into our sequence.
        if isPhi4 {
            parts.append("<|system|>\n\(systemPrompt)<|end|>")
        } else if isGemma4 {
            parts.append("<|turn>system\n\(systemPrompt)<turn|>")
        } else if isGemma {
            parts.append("<start_of_turn>system\n\(systemPrompt)<end_of_turn>")
        } else if isLlama3 {
            parts.append("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n\(systemPrompt)<|eot_id|>")
        } else if isLlama {
            parts.append("<<SYS>>\n\(systemPrompt)\n<</SYS>>")
        } else if isGranite42 {
            parts.append("<|im_start|>system\n\(systemPrompt)<|im_end|>")
        } else if isGranite {
            parts.append("<|start_of_role|>system<|end_of_role|>\(systemPrompt)<|end_of_text|>")
        } else if isChatML {
            parts.append("<|startoftext|><|im_start|>system\n\(systemPrompt)<|im_end|>")
        } else {
            parts.append("System: \(systemPrompt)")
        }

        for msg in truncatedHistory {
            let content = msg.content.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !content.isEmpty else { continue }

            if isPhi4 {
                let role = (msg.role == "user") ? "user" : "assistant"
                parts.append("<|\(role)|>\n\(content)<|end|>")
            } else if isGemma4 {
                let role = (msg.role == "user") ? "user" : "model"
                parts.append("<|turn>\(role)\n\(content)<turn|>")
            } else if isGemma {
                let role = (msg.role == "user") ? "user" : "model"
                parts.append("<start_of_turn>\(role)\n\(content)<end_of_turn>")
            } else if isLlama3 {
                let role = (msg.role == "user") ? "user" : "assistant"
                parts.append("<|start_header_id|>\(role)<|end_header_id|>\n\n\(content)<|eot_id|>")
            } else if isLlama {
                if msg.role == "user" {
                    parts.append("[INST] \(content) [/INST]")
                } else {
                    parts.append(content)
                }
            } else if isGranite42 {
                let role = (msg.role == "user") ? "user" : "assistant"
                let prefix = (role == "assistant") ? "<think></think>" : ""
                parts.append("<|im_start|>\(role)\n\(prefix)\(content)<|im_end|>")
            } else if isGranite {
                let role = (msg.role == "user") ? "user" : "assistant"
                parts.append("<|start_of_role|>\(role)<|end_of_role|>\(content)<|end_of_text|>")
            } else if isChatML {
                let role = (msg.role == "user") ? "user" : "assistant"
                parts.append("<|im_start|>\(role)\n\(content)<|im_end|>")
            } else {
                let prefix = (msg.role == "user") ? "User" : "Assistant"
                parts.append("\(prefix): \(content)")
            }
        }

        // Final Open Turn (Assistant)
        if isPhi4 {
            parts.append("<|user|>\n\(text)<|end|>")
            parts.append("<|assistant|>\n")
        } else if isGemma4 {
            parts.append("<|turn>user\n\(text)<turn|>")
            parts.append("<|turn>model\n")
        } else if isGemma {
            parts.append("<start_of_turn>user\n\(text)<end_of_turn>")
            parts.append("<start_of_turn>model\n")
        } else if isLlama3 {
            parts.append("<|start_header_id|>user<|end_header_id|>\n\n\(text)<|eot_id|>")
            parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
        } else if isGranite42 {
            parts.append("<|im_start|>user\n\(text)<|im_end|>")
            parts.append("<|im_start|>assistant\n<think></think>")
        } else if isGranite {
            parts.append("<|start_of_role|>user<|end_of_role|>\(text)<|end_of_text|>")
            parts.append("<|start_of_role|>assistant<|end_of_role|>")
        } else if isChatML {
            parts.append("<|im_start|>user\n\(text)<|im_end|>")
            parts.append("<|im_start|>assistant\n")
        } else {
            parts.append("Assistant:")
        }

        let multiTurnPrompt = parts.joined(separator: "\n")

        generationTask = Task {
            do {
                let savedThinking = self.llm.enableThinking
                self.llm.enableThinking = false
                defer { self.llm.enableThinking = savedThinking }
                try await llm.generate(
                    prompt: multiTurnPrompt,
                    systemPrompt: nil,
                    maxTokensOverride: Int(maxTokens)
                ) { content, _, _ in
                    Task { @MainActor in
                        let sanitized = sanitizeModelOutputText(content)
                        let answerSoFar = getDisplayContentWithoutThinking(sanitized)
                        self.latestReply = answerSoFar

                        let delta = String(answerSoFar.dropFirst(self.ttsReadCursor))
                        if !delta.isEmpty {
                            self.ttsReadCursor = answerSoFar.count
                            self.ttsManager.addStreamingToken(delta, fallbackLanguage: self.settings.selectedLanguage, key: self.ttsKey)
                        }
                    }
                }
            } catch {
                NSLog("[LLMHub][VibeVoice] LLM error: \(error.localizedDescription)")
            }
            await MainActor.run {
                self.generationTask = nil
                guard self.isChatActive else { return }

                let displayContent = getDisplayContentWithoutThinking(self.latestReply)
                if displayContent.count > self.ttsReadCursor {
                    let delta = String(displayContent.dropFirst(self.ttsReadCursor))
                    if !delta.isEmpty {
                        self.ttsManager.addStreamingToken(delta, fallbackLanguage: self.settings.selectedLanguage, key: self.ttsKey)
                    }
                }
                self.ttsManager.flushStreamingBuffer(fallbackLanguage: self.settings.selectedLanguage, key: self.ttsKey)
                self.ttsReadCursor = 0

                let rawReply = self.latestReply.trimmingCharacters(in: .whitespacesAndNewlines)
                let replyForHistory = getDisplayContentWithoutThinking(rawReply)
                let finalReply = replyForHistory.isEmpty ? rawReply : replyForHistory
                if !finalReply.isEmpty && !contentHasThinkingMarkers(finalReply) {
                    self.latestReply = finalReply
                    self.conversationHistory.append((role: "assistant", content: finalReply))
                    if self.ttsManager.isSpeaking(key: self.ttsKey) {
                        self.voiceState = .speaking
                    } else {
                        self.voiceState = .idle
                        Task { @MainActor in
                            try? await Task.sleep(nanoseconds: 300_000_000)
                            guard self.isChatActive && self.isCurrentModelLoaded else { return }
                            await self.startListeningCycle()
                        }
                    }
                } else {
                    self.latestReply = ""
                    self.voiceState = .idle
                    Task { @MainActor in
                        try? await Task.sleep(nanoseconds: 300_000_000)
                        guard self.isChatActive && self.isCurrentModelLoaded else { return }
                        await self.startListeningCycle()
                    }
                }
            }
        }
    }

    private func handleAudioTranscript(_ url: URL) async {
        guard isChatActive else { return }
        voiceState = .responding
        latestReply = ""

        await ensureModelLoaded(force: false)
        guard llm.isLoaded && isChatActive else {
            voiceState = .idle
            return
        }

        let transcribed = await transcribeAudioWithModel(url)
        let cleaned = transcribed.trimmingCharacters(in: .whitespacesAndNewlines)
        if cleaned.isEmpty {
            voiceState = .idle
            Task { @MainActor in
                try? await Task.sleep(nanoseconds: 300_000_000)
                guard self.isChatActive && self.isCurrentModelLoaded else { return }
                await self.startListeningCycle()
            }
            return
        }

        await handleTranscript(cleaned)
    }

    private func transcribeAudioWithModel(_ url: URL) async -> String {
        final class TextHolder: @unchecked Sendable {
            var val = ""
        }
        let holder = TextHolder()
        do {
            try await llm.generate(
                prompt: "Transcribe this audio.",
                audioURL: url,
                maxTokensOverride: 512
            ) { text, _, _ in
                holder.val = text
            }
        } catch is CancellationError {
            return ""
        } catch {
            NSLog("[LLMHub][VibeVoice] Audio transcription failed: \(error.localizedDescription)")
            return ""
        }
        return sanitizeModelOutputText(holder.val)
    }

    private func ensureModelLoaded(force: Bool) async {
        guard let model = selectedFeatureModel(named: selectedModelName) else {
            errorMessage = settings.localized("writing_aid_no_model")
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
        llm.enableVision = false
        llm.enableAudio = model.supportsAudio
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

    private func stopAll() {
        generationTask?.cancel()
        generationTask = nil
        ttsManager.stop()
        voiceState = .idle
        conversationHistory = []
        transcriber.cancelListening()
        audioRecorder.cancelRecording()
    }
}

/// Play / stop button for the last recorded clip (native bordered style).
private struct MacVibeVoicePlaybackButton: View {
    let url: URL
    @StateObject private var controller = AudioPlaybackController()

    var body: some View {
        Button {
            controller.toggle(url: url)
        } label: {
            Image(systemName: controller.isPlaying ? "stop.fill" : "play.fill")
        }
        .buttonStyle(.bordered)
    }
}
#endif
