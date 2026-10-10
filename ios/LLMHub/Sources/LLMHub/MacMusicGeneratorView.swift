//
//  MacMusicGeneratorView.swift
//  LLMHub
//
//  Native macOS Music Generator. Same persisted settings, backend flow and
//  localized strings as `MusicGeneratorScreen` on iOS.
//

#if os(macOS)
import SwiftUI

struct MacMusicGeneratorView: View {
    @EnvironmentObject var settings: AppSettings
    let onNavigateToModels: () -> Void

    @AppStorage("feature_music_model_name") private var selectedModelName: String = ""
    @State private var maxTokens: Double = 2048
    @State private var prompt: String = ""
    @FocusState private var promptFocused: Bool
    @State private var isGenerating: Bool = false
    @State private var generationStartedAt: Date?
    @State private var generatedTracks: [GeneratedMusicTrack] = []
    @State private var durationSeconds: Double = 10.0
    @AppStorage("feature_music_live_generation") private var liveGeneration: Bool = false
    @AppStorage("feature_music_unlimited_duration") private var unlimitedDuration: Bool = false
    @AppStorage("feature_music_random_seed") private var randomSeed: Bool = true
    @AppStorage("feature_music_seed") private var musicSeed: Double = 0
    @State private var showInspector: Bool = false
    @State private var isLoading: Bool = false
    @State private var errorMessage: String? = nil

    @ObservedObject private var musicBackend = MusicGeneratorBackend.shared

    private var isCurrentModelLoaded: Bool {
        musicBackend.isLoaded && musicBackend.loadedModelName == selectedModelName
    }

    private let presetPrompts = [
        "Upbeat 80s Synthwave synth bass & drums",
        "Ambient relaxing acoustic piano & warm pads",
        "Epic cinematic trailer orchestral battle motif",
        "Chill Lo-Fi hip hop beat with rain sounds",
        "Energetic rock guitar riff with upbeat rhythm",
        "Smooth jazz saxophone melody with acoustic bass"
    ]

    init(onNavigateToModels: @escaping () -> Void) {
        self.onNavigateToModels = onNavigateToModels
    }

    private func ensureModelLoaded(force _: Bool = false) async -> Bool {
        guard !selectedModelName.isEmpty else { return false }
        guard let model = selectedFeatureModel(named: selectedModelName) else { return false }
        let managerReportsDownloaded: Bool
        if case .downloaded? = ModelManager.shared.modelStatuses[model.id] {
            managerReportsDownloaded = true
        } else {
            managerReportsDownloaded = false
        }
        guard ModelData.isModelFullyAvailableLocally(model) || managerReportsDownloaded else { return false }
        return await musicBackend.loadModel(modelName: selectedModelName)
    }

    var body: some View {
        Group {
            if selectedModelName.isEmpty {
                ContentUnavailableView {
                    Label(settings.localized("music_generator_download_model"), systemImage: "music.note")
                } description: {
                    Text(settings.localized("music_generator_download_model_desc"))
                } actions: {
                    Button(action: onNavigateToModels) {
                        Label(settings.localized("download_models"), systemImage: "arrow.down.circle")
                    }
                    .buttonStyle(.borderedProminent)
                }
            } else {
                mainView
                    .safeAreaInset(edge: .bottom, spacing: 0) {
                        if isGenerating {
                            TimelineView(.periodic(from: .now, by: 1)) { context in
                                let elapsed = max(0, Int(context.date.timeIntervalSince(generationStartedAt ?? context.date)))
                                primaryActionBar(
                                    title: "\(settings.localized("music_stop_generation")) · \(String(format: "%02d:%02d", elapsed / 60, elapsed % 60))"
                                )
                            }
                        } else {
                            primaryActionBar(
                                title: isLoading ? settings.localized("model_loading") : settings.localized("generate_music")
                            )
                        }
                    }
            }
        }
        .navigationTitle(settings.localized("feature_music_generator"))
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                Button {
                    showInspector.toggle()
                } label: {
                    Label(settings.localized("feature_settings_title"), systemImage: "slider.horizontal.3")
                }
                // iOS blocks opening settings while busy; closing stays possible here.
                .disabled(!showInspector && (isGenerating || isLoading))
            }
        }
        .inspector(isPresented: $showInspector) {
            MacFeatureModelInspector(
                selectedModelName: $selectedModelName,
                maxTokens: $maxTokens,
                enableThinking: .constant(false),
                enableVision: .constant(false),
                enableAudio: nil,
                isLoading: $isLoading,
                errorMessage: $errorMessage,
                supportsVisionToggle: false,
                visionToggleTitleKey: "",
                audioToggleTitleKey: nil,
                visionAvailableCheck: nil,
                writingMode: nil,
                modelFilter: isMusicGenerationFeatureModel,
                onLoad: {
                    isLoading = true
                    defer { isLoading = false }
                    _ = await ensureModelLoaded(force: true)
                },
                onUnload: { musicBackend.unloadModel() },
                showsThinkingToggle: false,
                extraModelConfigsContent: AnyView(musicGenerationSettings.environmentObject(settings))
            )
            .inspectorColumnWidth(min: 280, ideal: 320, max: 420)
        }
        .onAppear {
            Task {
                await refreshDownloadedModelStatus()
                let available = downloadableFeatureModels().filter(isMusicGenerationFeatureModel)
                if selectedModelName.isEmpty || !available.contains(where: { $0.name == selectedModelName }) {
                    selectedModelName = available.first?.name ?? ""
                }
            }
        }
        .onDisappear {
            musicBackend.unloadModel()
        }
    }

    // MARK: - Primary Action

    private func primaryActionBar(title: String) -> some View {
        MacPrimaryActionBar(
            title: title,
            systemImage: isGenerating ? "stop.fill" : isLoading ? "hourglass" : "sparkles",
            isBusy: isLoading,
            isEnabled: !(isLoading || (!isGenerating && prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)),
            tint: isGenerating ? .red : ApolloPalette.accentStrong
        ) {
            if isGenerating {
                musicBackend.stopLiveGeneration()
            } else {
                generateMusic()
            }
        }
    }

    // MARK: - Main

    private var mainView: some View {
        VStack(spacing: 0) {
            ScrollViewReader { scrollProxy in
                ScrollView {
                    VStack(alignment: .leading, spacing: 16) {
                        promptInputPanel

                        if !liveGeneration || !unlimitedDuration {
                            VStack(alignment: .leading, spacing: 6) {
                                LabeledContent(settings.localized("music_duration_label")) {
                                    Text("\(Int(durationSeconds))s")
                                        .bold()
                                        .monospacedDigit()
                                        .foregroundStyle(ApolloPalette.accentStrong)
                                }
                                ApolloSlider(value: $durationSeconds, in: 1...600, step: 1)
                                    .labelsHidden()
                            }
                        }

                        if let errorMessage {
                            Text(errorMessage)
                                .foregroundStyle(.red)
                                .font(.caption)
                        }

                        if !generatedTracks.isEmpty {
                            VStack(spacing: 0) {
                                ForEach(generatedTracks) { track in
                                    trackRow(track)
                                        .id(track.id)
                                    if track.id != generatedTracks.last?.id {
                                        Divider()
                                    }
                                }
                            }
                            .background(Color(nsColor: .textBackgroundColor).opacity(0.2), in: RoundedRectangle(cornerRadius: 8))
                            .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color(nsColor: .separatorColor)))
                        }
                    }
                    .padding()
                    .frame(maxWidth: 820)
                    .frame(maxWidth: .infinity)
                }
                .onChange(of: generatedTracks.count) { _, _ in
                    if let latest = generatedTracks.last {
                        withAnimation { scrollProxy.scrollTo(latest.id, anchor: .bottom) }
                    }
                }
            }

            if isGenerating && !(liveGeneration && unlimitedDuration) {
                ProgressView(value: musicBackend.progress)
                    .padding(.horizontal)
                    .padding(.vertical, 10)
            }
        }
    }

    private func trackRow(_ track: GeneratedMusicTrack) -> some View {
        HStack(spacing: 10) {
            Image(systemName: "waveform.circle.fill")
                .font(.title2)
                .foregroundStyle(ApolloPalette.accentStrong)
            VStack(alignment: .leading, spacing: 2) {
                Text(track.prompt)
                    .font(.subheadline)
                    .bold()
                    .lineLimit(1)
                Text("\(track.requestedDurationSeconds)s Audio Clip")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            ShareLink(item: track.url) {
                Image(systemName: "square.and.arrow.down")
            }
            .buttonStyle(.bordered)
            .accessibilityLabel(settings.localized("save"))
            MacAudioPlaybackButton(url: track.url)
        }
        .padding(10)
    }

    private var musicGenerationSettings: some View {
        VStack(alignment: .leading, spacing: 8) {
            Toggle(settings.localized("music_live_generation"), isOn: $liveGeneration)
                .disabled(isGenerating || isLoading)

            if liveGeneration {
                Toggle(settings.localized("music_unlimited_duration"), isOn: $unlimitedDuration)
                    .disabled(isGenerating || isLoading)
            }
            Toggle(settings.localized("image_generator_random_seed"), isOn: $randomSeed)
                .disabled(isGenerating || isLoading)

            if !randomSeed {
                LabeledContent(settings.localized("image_generator_seed")) {
                    Text("\(Int(musicSeed))")
                        .monospacedDigit()
                }
                ApolloSlider(value: $musicSeed, in: 0...999_999, step: 1)
                    .labelsHidden()
                    .disabled(isGenerating || isLoading)
            }
        }
    }

    private var promptInputPanel: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(settings.localized("music_prompt_label"))
                .font(.headline)
            TextEditor(text: $prompt)
                .focused($promptFocused)
                .font(.body)
                .scrollContentBackground(.hidden)
                .padding(6)
                .frame(minHeight: 120)
                .background(Color(nsColor: .textBackgroundColor).opacity(0.35), in: RoundedRectangle(cornerRadius: 8))
                .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color(nsColor: .separatorColor)))
                .overlay(alignment: .topLeading) {
                    if prompt.isEmpty {
                        Text(settings.localized("prompt_hint_music"))
                            .foregroundStyle(.secondary)
                            .padding(.horizontal, 11)
                            .padding(.vertical, 6)
                            .allowsHitTesting(false)
                    }
                }

            Text(settings.localized("music_style_presets"))
                .font(.caption)
                .foregroundStyle(.secondary)
                .padding(.top, 4)

            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 8) {
                    ForEach(presetPrompts, id: \.self) { preset in
                        Button {
                            prompt = preset
                        } label: {
                            Label(preset, systemImage: "waveform")
                                .font(.caption)
                        }
                        .buttonStyle(.bordered)
                        .buttonBorderShape(.capsule)
                    }
                }
            }
        }
    }

    private func generateMusic() {
        promptFocused = false
        let requestedPrompt = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !requestedPrompt.isEmpty else { return }
        guard !isGenerating, !isLoading else { return }
        let requestedDuration = Int(durationSeconds)
        let requestedModel = selectedModelName
        let requestedLive = liveGeneration
        let requestedUnlimited = liveGeneration && unlimitedDuration
        let requestedSeed: UInt64? = randomSeed ? nil : UInt64(min(999_999, max(0, musicSeed)))
        isLoading = true
        Task {
            if !isCurrentModelLoaded {
                let success = await ensureModelLoaded(force: false)
                guard success else {
                    isLoading = false
                    errorMessage = musicBackend.errorMessage
                    return
                }
            }
            await MainActor.run {
                isLoading = false
                generationStartedAt = Date()
                isGenerating = true
                errorMessage = nil
            }

            let backend = MusicGeneratorBackend.shared
            if let outputURL = await backend.generateMusic(
                modelName: requestedModel,
                prompt: requestedPrompt,
                durationSeconds: Double(requestedDuration),
                live: requestedLive,
                unlimited: requestedUnlimited,
                seed: requestedSeed
            ) {
                await MainActor.run {
                    generatedTracks.append(
                        GeneratedMusicTrack(
                            prompt: requestedPrompt,
                            requestedDurationSeconds: Int(ceil(backend.generatedDurationSeconds)),
                            url: outputURL
                        )
                    )
                    isGenerating = false
                }
            } else {
                await MainActor.run {
                    isGenerating = false
                    errorMessage = backend.errorMessage
                }
            }
        }
    }
}

/// Native-styled equivalent of `AudioPlaybackButton` (same controller).
private struct MacAudioPlaybackButton: View {
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
