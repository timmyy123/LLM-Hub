//
//  MacVideoGeneratorView.swift
//  LLMHub
//
//  Native macOS Video Generator. Same persisted settings, backend flow and
//  localized strings as `VideoGeneratorScreen` on iOS.
//

#if os(macOS)
import SwiftUI
import PhotosUI
import AVKit

@MainActor
struct MacVideoGeneratorView: View {
    @EnvironmentObject var settings: AppSettings
    @AppStorage("sd_video_steps") private var storedSteps: Double = 20
    @AppStorage("sd_video_motion_strength") private var storedMotionStrength: Double = 0.7
    @AppStorage("sd_video_model_id") private var selectedModelId: String = ""

    @State private var promptText = ""
    @FocusState private var promptFocused: Bool
    @State private var seed: Int = Int.random(in: 0..<1_000_000)
    @State private var generatedVideoURL: URL?
    @State private var player: AVPlayer?
    @State private var isGenerating = false
    @State private var isSaving = false
    @State private var showInspector = false
    @State private var errorMessage: String?
    @State private var inputImage: UIImage?
    @State private var selectedImageItem: PhotosPickerItem?
    @State private var generateTask: Task<Void, Never>?
    @State private var videoSaver = VideoSaver()
    @State private var showSaveAlert = false
    @State private var saveAlertTitle = ""
    @State private var saveAlertMessage = ""
    @ObservedObject private var videoBackend = VideoGeneratorBackend.shared

    let onNavigateToModels: () -> Void

    // Only real video generation drawthings models — never image gen models
    private var availableModels: [AIModel] {
        ModelData.allModels().filter { $0.isDrawThingsVideoGeneration && ModelData.isModelFullyAvailableLocally($0) }
    }

    private var selectedModel: AIModel? {
        availableModels.first(where: { $0.id == selectedModelId }) ?? availableModels.first
    }

    private var isModelDownloaded: Bool {
        guard let model = selectedModel else { return false }
        return ModelData.isModelFullyAvailableLocally(model)
    }

    private var canGenerate: Bool {
        !(promptText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
          || (inputImage == nil && selectedModel?.supportsPromptOnlyVideoGeneration != true))
    }

    var body: some View {
        Group {
            if availableModels.isEmpty {
                noModelView
            } else if !isModelDownloaded {
                loadModelView
            } else {
                mainGenerationView
                    .safeAreaInset(edge: .bottom, spacing: 0) {
                        MacPrimaryActionBar(
                            title: videoBackend.isLoading ? settings.localized("model_loading")
                                : isGenerating ? "\(settings.localized("video_generator_generating")) (\(videoBackend.generationStep)/\(videoBackend.generationTotalSteps))"
                                : settings.localized("video_generator_generate"),
                            systemImage: (isGenerating || videoBackend.isLoading) ? "stop.fill" : "sparkles",
                            isBusy: videoBackend.isLoading,
                            isEnabled: isGenerating || canGenerate,
                            tint: isGenerating ? .red : ApolloPalette.accentStrong
                        ) {
                            if isGenerating {
                                generateTask?.cancel()
                                videoBackend.cancelGeneration()
                            } else {
                                startGeneration()
                            }
                        }
                    }
            }
        }
        .navigationTitle(settings.localized("video_generator_title"))
        .toolbar {
            if !availableModels.isEmpty {
                ToolbarItem(placement: .primaryAction) {
                    Button {
                        showInspector.toggle()
                    } label: {
                        Label(settings.localized("feature_settings_title"), systemImage: "slider.horizontal.3")
                    }
                }
            }
        }
        .inspector(isPresented: $showInspector) {
            settingsInspector
                .inspectorColumnWidth(min: 280, ideal: 320, max: 420)
        }
        .onChange(of: selectedImageItem) { _, item in
            guard let item else { inputImage = nil; return }
            Task {
                if let data = try? await item.loadTransferable(type: Data.self),
                   let img = UIImage(data: data) {
                    inputImage = img
                } else {
                    inputImage = nil
                }
            }
        }
        .onAppear {
            if selectedModelId.isEmpty || !availableModels.contains(where: { $0.id == selectedModelId }) {
                selectedModelId = availableModels.first?.id ?? ""
            }
        }
        .onDisappear {
            generateTask?.cancel()
            player?.pause()
        }
        .overlay(alignment: .bottom) {
            if let msg = errorMessage {
                Text(msg)
                    .font(.caption)
                    .foregroundStyle(.red)
                    .padding(.horizontal)
                    .padding(.bottom, 72) // stay above the bottom action bar
                    .onTapGesture { errorMessage = nil }
            }
        }
        .alert(saveAlertTitle, isPresented: $showSaveAlert) {
            Button(settings.localized("ok"), role: .cancel) {}
        } message: {
            Text(saveAlertMessage)
        }
    }

    // MARK: - Empty States

    private var noModelView: some View {
        ContentUnavailableView {
            Label(settings.localized("video_generator_download_model"), systemImage: "video.fill")
        } description: {
            Text(settings.localized("video_generator_download_model_desc"))
        } actions: {
            Button(settings.localized("download"), action: onNavigateToModels)
                .buttonStyle(.borderedProminent)
        }
    }

    private var loadModelView: some View {
        Group {
            if videoBackend.isLoading {
                VStack(spacing: 16) {
                    ProgressView()
                    Text(settings.localized("image_generator_loading_model"))
                        .font(.title3.bold())
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                ContentUnavailableView {
                    Label(settings.localized("video_generator_load_model_title"), systemImage: "cpu.fill")
                } description: {
                    Text(settings.localized("video_generator_load_model_desc"))
                } actions: {
                    Button(settings.localized("feature_settings_title")) { showInspector = true }
                        .buttonStyle(.borderedProminent)
                }
            }
        }
    }

    // MARK: - Main

    private var mainGenerationView: some View {
        HStack(spacing: 0) {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    promptPanel
                    Divider()
                    img2vidPanel
                }
                .padding()
            }
            .frame(minWidth: 300, idealWidth: 360, maxWidth: 420)

            Divider()

            Group {
                if generatedVideoURL != nil, let player {
                    VStack(spacing: 10) {
                        VideoPlayer(player: player)
                            .frame(minHeight: 300)
                            .clipShape(RoundedRectangle(cornerRadius: 12))
                        HStack {
                            Button {
                                if let url = generatedVideoURL {
                                    saveVideoToPhotos(url)
                                }
                            } label: {
                                if isSaving {
                                    HStack(spacing: 6) {
                                        ProgressView().controlSize(.small)
                                        Text("Saving...")
                                    }
                                } else {
                                    Label(settings.localized("video_generator_save"), systemImage: "square.and.arrow.down")
                                }
                            }
                            .buttonStyle(.bordered)
                            .disabled(isSaving)
                        }
                    }
                    .padding()
                } else if isGenerating {
                    VStack(spacing: 12) {
                        ProgressView(
                            value: Double(videoBackend.generationStep),
                            total: Double(max(1, videoBackend.generationTotalSteps))
                        )
                        .frame(width: 220)
                        Text("\(settings.localized("video_generator_generating")) (\(videoBackend.generationStep)/\(videoBackend.generationTotalSteps))")
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                            .monospacedDigit()
                    }
                } else {
                    Image(systemName: "film")
                        .font(.system(size: 56, weight: .light))
                        .foregroundStyle(.tertiary)
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
    }

    private var promptPanel: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(settings.localized("video_generator_prompt_label"))
                .font(.headline)
            TextEditor(text: $promptText)
                .focused($promptFocused)
                .font(.body)
                .scrollContentBackground(.hidden)
                .padding(6)
                .frame(minHeight: 120)
                .background(Color(nsColor: .textBackgroundColor).opacity(0.35), in: RoundedRectangle(cornerRadius: 8))
                .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color(nsColor: .separatorColor)))
                .overlay(alignment: .topLeading) {
                    if promptText.isEmpty {
                        Text(settings.localized("video_generator_prompt_hint"))
                            .foregroundStyle(.secondary)
                            .padding(.horizontal, 11)
                            .padding(.vertical, 6)
                            .allowsHitTesting(false)
                    }
                }
                .disabled(isGenerating)
        }
    }

    private var img2vidPanel: some View {
        let labelText = settings.localized(inputImage != nil ? "video_generator_change_image" : "video_generator_select_image")
        return VStack(alignment: .leading, spacing: 10) {
            Text(settings.localized("video_generator_input_image"))
                .font(.headline)
            HStack(spacing: 12) {
                PhotosPicker(selection: $selectedImageItem, matching: .images) {
                    Label(labelText, systemImage: "photo")
                        .lineLimit(1)
                }
                .buttonStyle(.bordered)
                .disabled(isGenerating)

                if let thumb = inputImage {
                    ZStack(alignment: .topTrailing) {
                        Image(uiImage: thumb)
                            .resizable()
                            .scaledToFill()
                            .frame(width: 48, height: 48)
                            .clipShape(RoundedRectangle(cornerRadius: 8))
                        Button {
                            inputImage = nil
                            selectedImageItem = nil
                        } label: {
                            Image(systemName: "xmark.circle.fill")
                                .font(.caption)
                                .foregroundStyle(.white, .black.opacity(0.5))
                        }
                        .buttonStyle(.plain)
                    }
                }
                Spacer(minLength: 0)
            }
        }
    }

    // MARK: - Inspector (VideoGeneratorSettingsSheet)

    private var settingsInspector: some View {
        Form {
            Section(settings.localized("select_model_title")) {
                if availableModels.isEmpty {
                    Label(settings.localized("no_models_available"), systemImage: "exclamationmark.triangle")
                        .foregroundStyle(.secondary)
                } else {
                    Picker(settings.localized("select_model"), selection: $selectedModelId) {
                        ForEach(availableModels) { model in
                            Text(model.name).tag(model.id)
                        }
                    }
                    .labelsHidden()
                }
            }

            Section {
                LabeledContent(settings.localized("video_generator_iterations")) {
                    Text("\(Int(storedSteps))").monospacedDigit()
                }
                ApolloSlider(value: $storedSteps, in: 4...50, step: 1)
                    .labelsHidden()
            }

            Section {
                Text(String(format: settings.localized("video_generator_motion_strength"), storedMotionStrength))
                Text(settings.localized("video_generator_motion_strength_desc"))
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Slider(value: $storedMotionStrength, in: 0.1...1.0)
                    .labelsHidden()
            }

            Section {
                VStack(spacing: 8) {
                    Button {
                        guard let model = selectedModel else { return }
                        Task {
                            do {
                                try await videoBackend.loadModel(url: URL(fileURLWithPath: "/"), modelId: model.id)
                            } catch {
                                errorMessage = error.localizedDescription
                            }
                        }
                    } label: {
                        Group {
                            if videoBackend.isLoading {
                                HStack(spacing: 6) {
                                    ProgressView().controlSize(.small)
                                    Text(settings.localized("image_generator_loading_model"))
                                }
                            } else {
                                Text(settings.localized(videoBackend.isLoaded ? "reload_model" : "image_generator_load_model"))
                                    .fontWeight(.semibold)
                            }
                        }
                        .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.large)
                    .disabled(availableModels.isEmpty || videoBackend.isLoading)

                    if videoBackend.isLoaded {
                        Button(role: .destructive) {
                            videoBackend.unloadModel()
                        } label: {
                            Text(settings.localized("unload_model")).frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.large)
                    }
                }
            }
        }
        .formStyle(.grouped)
    }

    // MARK: - Generation Logic

    private func startGeneration() {
        guard !promptText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
        guard inputImage != nil || selectedModel?.supportsPromptOnlyVideoGeneration == true else {
            errorMessage = VideoError.inputImageRequired.localizedDescription
            return
        }
        isGenerating = true
        promptFocused = false
        errorMessage = nil

        let prompt = promptText
        let steps = Int(storedSteps)
        let strength = Float(storedMotionStrength)
        let startingImage = inputImage

        generateTask?.cancel()
        generateTask = Task {
            do {
                guard let model = selectedModel else {
                    errorMessage = settings.localized("video_generator_no_model")
                    isGenerating = false
                    return
                }
                if !videoBackend.isLoaded || videoBackend.loadedModelId != model.id {
                    try await videoBackend.loadModel(url: URL(fileURLWithPath: "/"), modelId: model.id)
                }
                let url = try await videoBackend.generateVideo(
                    prompt: prompt,
                    steps: steps,
                    seed: UInt32(seed),
                    inputImage: startingImage,
                    motionStrength: strength
                )
                generatedVideoURL = url
                // A fresh player per result (iOS builds a new AVPlayer for each render).
                player?.pause()
                player = AVPlayer(url: url)
            } catch is CancellationError {
                // ignore
            } catch {
                errorMessage = error.localizedDescription
            }
            isGenerating = false
        }
    }

    private func saveVideoToPhotos(_ fileURL: URL) {
        guard !isSaving else { return }
        isSaving = true
        errorMessage = nil

        let tempCopyURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString + ".mp4")

        do {
            if FileManager.default.fileExists(atPath: tempCopyURL.path) {
                try? FileManager.default.removeItem(at: tempCopyURL)
            }
            try FileManager.default.copyItem(at: fileURL, to: tempCopyURL)
        } catch {
            self.saveAlertTitle = self.settings.localized("error")
            self.saveAlertMessage = String(format: self.settings.localized("video_generator_save_failed") + ": %@", error.localizedDescription)
            self.showSaveAlert = true
            self.isSaving = false
            return
        }

        videoSaver.writeToPhotoAlbum(videoURL: tempCopyURL) { error in
            DispatchQueue.main.async {
                try? FileManager.default.removeItem(at: tempCopyURL)
                self.isSaving = false
                if let error = error {
                    self.saveAlertTitle = self.settings.localized("error")
                    self.saveAlertMessage = String(format: self.settings.localized("video_generator_save_failed") + ": %@", error.localizedDescription)
                } else {
                    self.saveAlertTitle = self.settings.localized("success")
                    self.saveAlertMessage = self.settings.localized("video_generator_saved")
                }
                self.showSaveAlert = true
            }
        }
    }
}
#endif
