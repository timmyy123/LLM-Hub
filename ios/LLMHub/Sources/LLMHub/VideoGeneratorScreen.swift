import SwiftUI
import PhotosUI
import Photos
import AVKit

@MainActor
struct VideoGeneratorScreen: View {
    @EnvironmentObject var settings: AppSettings
    @AppStorage("sd_video_steps") private var storedSteps: Double = 20
    @AppStorage("sd_video_motion_strength") private var storedMotionStrength: Double = 0.7
    @AppStorage("sd_video_model_id") private var selectedModelId: String = ""

    @State private var promptText = ""
    @FocusState private var promptFocused: Bool
    @State private var seed: Int = Int.random(in: 0..<1_000_000)
    @State private var generatedVideoURL: URL?
    @State private var isGenerating = false
    @State private var isUnfolded = false
    // Portrait: heights of the controls above and below the video, so the video fills the rest.
    @State private var portraitTopHeight: CGFloat = 0
    @State private var portraitBottomHeight: CGFloat = 0
    @State private var hasHinge = false
    @State private var isSaving = false
    @State private var showSettings = false
    @State private var errorMessage: String?
    @State private var inputImage: UIImage?
    @State private var selectedImageItem: PhotosPickerItem?
    @State private var generateTask: Task<Void, Never>?
    @State private var videoSaver = VideoSaver()
    @State private var showSaveAlert = false
    @State private var saveAlertTitle = ""
    @State private var saveAlertMessage = ""
    @ObservedObject private var videoBackend = VideoGeneratorBackend.shared

    let onNavigateBack: () -> Void
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

    /// Screenshot automation (Debug only): show an injected video as the result.
    private var isInjectingResult: Bool {
        #if DEBUG
        return ScreenshotHooks.injectVideoURL != nil
        #else
        return false
        #endif
    }

    var body: some View {
        Group {
            if availableModels.isEmpty && !isInjectingResult {
                noModelView
            } else if !isModelDownloaded && !isInjectingResult {
                loadModelView
            } else {
                mainGenerationView
            }
        }
        .navigationTitle(settings.localized("video_generator_title"))
        .navigationBarTitleDisplayMode(.inline)
        .apolloScreenBackground()
        .apolloNavigationBackground()
        .toolbar {
            ToolbarItem(placement: .navigationBarLeading) {
                Button {
                    generateTask?.cancel()
                    videoBackend.unloadModel()
                    onNavigateBack()
                } label: { Image(systemName: "arrow.left") }
                .accessibilityIdentifier("nav_back_button")
            }
            if !availableModels.isEmpty || isInjectingResult {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button { showSettings = true } label: { Image(systemName: "slider.horizontal.3") }
                        .accessibilityIdentifier("feature_settings_button")
                }
            }
        }
        .apolloSheet(isPresented: $showSettings) {
            VideoGeneratorSettingsSheet(
                steps: $storedSteps,
                motionStrength: $storedMotionStrength,
                selectedModelId: $selectedModelId,
                availableModels: availableModels,
                isLoaded: videoBackend.isLoaded,
                isLoading: videoBackend.isLoading,
                onLoad: {
                    guard let model = selectedModel else { return }
                    Task {
                        do {
                            try await videoBackend.loadModel(url: URL(fileURLWithPath: "/"), modelId: model.id)
                        } catch {
                            errorMessage = error.localizedDescription
                        }
                    }
                },
                onUnload: { videoBackend.unloadModel() }
            )
            .environmentObject(settings)
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
            #if DEBUG
            // Screenshot automation: prefill the prompt.
            if promptText.isEmpty, let prompt = ScreenshotHooks.videoPrompt { promptText = prompt }
            // Input image only (real generation) or together with an injected result.
            if inputImage == nil, let injected = ScreenshotHooks.image(at: ScreenshotHooks.injectVideoInputURL) ?? ScreenshotHooks.environmentInputImage {
                inputImage = injected
            }
            #endif
        }
        .onDisappear {
            generateTask?.cancel()
        }
        .overlay(alignment: .bottom) {
            if let msg = errorMessage {
                Text(msg)
                    .font(.caption)
                    .foregroundStyle(.red)
                    .padding(.horizontal)
                    .padding(.bottom, 8)
                    .onTapGesture { errorMessage = nil }
            }
        }
        .alert(saveAlertTitle, isPresented: $showSaveAlert) {
            Button(settings.localized("ok"), role: .cancel) {}
        } message: {
            Text(saveAlertMessage)
        }
    }

    // MARK: - No-Model State

    private var noModelView: some View {
        VStack(spacing: 20) {
            Image(systemName: "video.fill")
                .font(.system(size: 56, weight: .semibold))
                .foregroundStyle(.secondary)

            Text(settings.localized("video_generator_download_model"))
                .font(.title3.bold())
                .multilineTextAlignment(.center)

            Text(settings.localized("video_generator_download_model_desc"))
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)

            Button {
                onNavigateToModels()
            } label: {
                Text(settings.localized("download"))
                    .frame(maxWidth: .infinity)
                    .frame(height: 52)
            }
            .foregroundStyle(.white)
            .liquidGlassPrimaryButton(cornerRadius: 12)
            .padding(.horizontal, 32)
        }
        .padding()
    }

    private var loadModelView: some View {
        VStack(spacing: 20) {
            if videoBackend.isLoading {
                ProgressView()
                    .scaleEffect(1.4)
                Text(settings.localized("image_generator_loading_model"))
                    .font(.title3.bold())
            } else {
                Image(systemName: "cpu.fill")
                    .font(.system(size: 56, weight: .semibold))
                    .foregroundStyle(.secondary)
                Text(settings.localized("video_generator_load_model_title"))
                    .font(.title3.bold())
                    .multilineTextAlignment(.center)
                Text(settings.localized("video_generator_load_model_desc"))
                    .font(.subheadline)
                    .foregroundStyle(.white.opacity(0.7))
                    .multilineTextAlignment(.center)
                    .padding(.horizontal)
                Button {
                    showSettings = true
                } label: {
                    HStack {
                        Spacer()
                        Text(settings.localized("feature_settings_title"))
                        Spacer()
                    }
                    .frame(height: 50)
                    .contentShape(Rectangle())
                }
                .frame(maxWidth: 260)
                .liquidGlassPrimaryButton(cornerRadius: 12)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    // MARK: - Main Generation View

    @ViewBuilder
    private var mainGenerationView: some View {
        GeometryReader { geo in
            let isLandscape = geo.size.width > geo.size.height
            if isLandscape {
                landscapeLayout
            } else if isUnfolded, let url = generatedVideoURL {
                hingePortraitLayout(url: url)
            } else {
                portraitLayout(height: geo.size.height)
            }
        }
        .apolloDetectsUnfoldedHinge($isUnfolded)
        .apolloDetectsHinge($hasHinge)
    }

    // MARK: - Hinge Portrait Layout (stacked, divided at the iPhone Duo hinge)

    private func hingePortraitLayout(url: URL) -> some View {
        ApolloHingeVSplit {
            ScrollView {
                generationControls
                    .padding(.vertical, 12)
            }
        } bottom: {
            videoPlayerCard(url: url)
                .padding(.bottom, 12)
        }
        .padding(.horizontal)
    }

    private var generationControls: some View {
        VStack(spacing: 14) {
            promptCard
            img2vidCard
            generateButton
            if generatedVideoURL != nil {
                saveButton
            }
        }
    }

    // MARK: - Portrait Layout

    private func portraitLayout(height: CGFloat) -> some View {
        // The video takes all the height left below the controls (at least 300 pt), instead of
        // a fixed 300 pt that left the bottom of a tall screen (iPad) empty. iPhone Duo keeps
        // its own layouts unchanged (hinge splits; 300 pt here).
        let spacing: CGFloat = 14
        let playerHeight: CGFloat? = hasHinge ? nil : max(300, height - 24 - portraitTopHeight - portraitBottomHeight - spacing * 2)
        return ScrollView {
            VStack(spacing: spacing) {
                VStack(spacing: spacing) {
                    promptCard
                    img2vidCard
                }
                .onGeometryChange(for: CGFloat.self) { $0.size.height } action: { portraitTopHeight = $0 }
                if let url = generatedVideoURL {
                    videoPlayerCard(url: url)
                        .frame(height: playerHeight)
                    VStack(spacing: spacing) {
                        saveButton
                        generateButton
                    }
                    .onGeometryChange(for: CGFloat.self) { $0.size.height } action: { portraitBottomHeight = $0 }
                } else {
                    generateButton
                }
            }
            .padding(.horizontal)
            .padding(.vertical, 12)
        }
    }

    // MARK: - Landscape Layout

    private var landscapeLayout: some View {
        ApolloHingeSplit {
            ScrollView {
                generationControls
                    .padding(.vertical, 12)
            }
        } trailing: {
            // Landscape always uses two columns; the output side waits empty until
            // a video exists.
            if let url = generatedVideoURL {
                videoPlayerCard(url: url)
            } else {
                ApolloOutputPlaceholder(systemImage: "film")
                    .padding(.vertical, 12)
            }
        }
        .padding(.horizontal)
    }

    // MARK: - Prompt Card

    private var promptCard: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(settings.localized("video_generator_prompt_label"))
                .font(.headline)
            TextEditor(text: $promptText)
                .accessibilityIdentifier("video_prompt")
                .screenshotAutocorrectionDisabled()
                .focused($promptFocused)
                .frame(minHeight: 100)
                .scrollContentBackground(.hidden)
                .background(Color.white.opacity(0.02))
                .clipShape(RoundedRectangle(cornerRadius: 12))
                .disabled(isGenerating)
                .overlay(
                    Group {
                        if promptText.isEmpty {
                            Text(settings.localized("video_generator_prompt_hint"))
                                .foregroundStyle(.secondary)
                                .padding(.horizontal, 12)
                                .padding(.vertical, 16)
                                .allowsHitTesting(false)
                                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
                        }
                    }
                )
        }
        .padding(.horizontal)
        .padding(.vertical, 12)
        .background(.ultraThinMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 16))
        .overlay(RoundedRectangle(cornerRadius: 16).stroke(Color.white.opacity(0.14), lineWidth: 1))
    }

    // MARK: - Image-to-Video Card

    private var img2vidCard: some View {
        let labelText = settings.localized(inputImage != nil ? "video_generator_change_image" : "video_generator_select_image")
        return VStack(alignment: .leading, spacing: 12) {
            Text(settings.localized("video_generator_input_image"))
                .font(.headline)
            HStack(spacing: 12) {
                PhotosPicker(
                    selection: $selectedImageItem,
                    matching: .images
                ) {
                    HStack {
                        Image(systemName: "photo")
                        Text(labelText)
                            .lineLimit(1)
                    }
                    .frame(maxWidth: .infinity)
                    .frame(height: 40)
                    .contentShape(Rectangle())
                }
                .foregroundStyle(.white)
                .background(.ultraThinMaterial)
                .clipShape(RoundedRectangle(cornerRadius: 10))
                .overlay(RoundedRectangle(cornerRadius: 10).stroke(Color.white.opacity(0.18), lineWidth: 1))
                .disabled(isGenerating)
                .accessibilityIdentifier("video_input_picker")

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
                    }
                }
            }
        }
        .padding(.horizontal)
        .padding(.vertical, 12)
        .background(.ultraThinMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 16))
        .overlay(RoundedRectangle(cornerRadius: 16).stroke(Color.white.opacity(0.14), lineWidth: 1))
    }

    // MARK: - Video Player

    private func videoPlayerCard(url: URL) -> some View {
        VStack {
            VideoPlayer(player: AVPlayer(url: url))
                .frame(minHeight: 300)
                .clipShape(RoundedRectangle(cornerRadius: 12))
                .padding(4)
        }
        .background(.ultraThinMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 16))
        .overlay(RoundedRectangle(cornerRadius: 16).stroke(Color.white.opacity(0.14), lineWidth: 1))
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("video_result")
    }

    // MARK: - Generate Button

    private var generateButton: some View {
        Button {
            if isGenerating {
                generateTask?.cancel()
                videoBackend.cancelGeneration()
            } else {
                startGeneration()
            }
        } label: {
            HStack(spacing: 8) {
                if isGenerating || videoBackend.isLoading {
                    ProgressView()
                        .tint(.white)
                        .scaleEffect(0.85)
                    if videoBackend.isLoading {
                        Text(settings.localized("model_loading"))
                            .lineLimit(1)
                    } else {
                        Text("\(settings.localized("video_generator_generating")) (\(videoBackend.generationStep)/\(videoBackend.generationTotalSteps))")
                            .lineLimit(1)
                    }
                } else {
                    Image(systemName: "sparkles")
                        .font(.system(size: 13, weight: .bold))
                    Text(settings.localized("video_generator_generate"))
                        .lineLimit(1)
                }
            }
            .frame(maxWidth: .infinity)
            .frame(height: 52)
        }
        .foregroundStyle(.white)
        .liquidGlassPrimaryButton(cornerRadius: 12)
        .disabled(!isGenerating && (promptText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || (inputImage == nil && selectedModel?.supportsPromptOnlyVideoGeneration != true)))
        .accessibilityIdentifier("video_generate_button")
    }

    private var saveButton: some View {
        Button {
            if let url = generatedVideoURL {
                saveVideoToPhotos(url)
            }
        } label: {
            HStack {
                Spacer()
                if isSaving {
                    ProgressView()
                        .tint(.white)
                        .scaleEffect(0.85)
                        .padding(.trailing, 8)
                    Text("Saving...")
                } else {
                    Text(settings.localized("video_generator_save"))
                }
                Spacer()
            }
            .frame(height: 44)
            .contentShape(Rectangle())
        }
        .foregroundStyle(.white)
        .liquidGlassPrimaryButton(cornerRadius: 12)
        .disabled(isSaving)
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
            #if DEBUG
            if let injected = ScreenshotHooks.injectVideoURL {
                try? await Task.sleep(nanoseconds: ScreenshotHooks.injectedResultDelay)
                generatedVideoURL = injected
                isGenerating = false
                return
            }
            #endif
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
                #if DEBUG
                ScreenshotHooks.saveOutput(copyOf: url, as: "video.\(url.pathExtension.isEmpty ? "mp4" : url.pathExtension)")
                ScreenshotHooks.saveOutput(startingImage, as: "video_input.png")
                #endif
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

// MARK: - VideoGeneratorSettingsSheet

struct VideoGeneratorSettingsSheet: View {
    @EnvironmentObject var settings: AppSettings
    @Binding var steps: Double
    @Binding var motionStrength: Double
    @Binding var selectedModelId: String
    let availableModels: [AIModel]
    let isLoaded: Bool
    let isLoading: Bool
    let onLoad: () -> Void
    let onUnload: () -> Void

    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 20) {
                    // Model Picker Card
                    VStack(alignment: .leading, spacing: 12) {
                        Text(settings.localized("select_model_title"))
                            .font(.headline)
                        if availableModels.isEmpty {
                            Label(settings.localized("no_models_available"), systemImage: "exclamationmark.triangle")
                                .foregroundStyle(.secondary)
                        } else {
                            Picker(settings.localized("select_model"), selection: $selectedModelId) {
                                ForEach(availableModels) { model in
                                    Text(model.name).tag(model.id)
                                }
                            }
                            .pickerStyle(.menu)
                            .tint(ApolloPalette.accentStrong)
                            .frame(maxWidth: .infinity, alignment: .leading)
                        }
                    }
                    .padding()
                    .background(.regularMaterial)
                    .clipShape(RoundedRectangle(cornerRadius: 14))

                    // Steps Slider Card
                    VStack(alignment: .leading, spacing: 8) {
                        Text("\(settings.localized("video_generator_iterations")): \(Int(steps))")
                            .font(.headline)
                        Slider(value: $steps, in: 4...50, step: 1)
                            .tint(ApolloPalette.accentStrong)
                    }
                    .padding()
                    .background(.regularMaterial)
                    .clipShape(RoundedRectangle(cornerRadius: 14))

                    // Motion Strength Slider Card
                    VStack(alignment: .leading, spacing: 8) {
                        Text(String(format: settings.localized("video_generator_motion_strength"), motionStrength))
                            .font(.headline)
                        Text(settings.localized("video_generator_motion_strength_desc"))
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        Slider(value: $motionStrength, in: 0.1...1.0)
                            .tint(ApolloPalette.accentStrong)
                    }
                    .padding()
                    .background(.regularMaterial)
                    .clipShape(RoundedRectangle(cornerRadius: 14))

                    // Load / Unload Actions
                    VStack(spacing: 12) {
                        Button {
                            onLoad()
                            dismiss()
                        } label: {
                            if isLoading {
                                HStack {
                                    ProgressView()
                                    Text(settings.localized("image_generator_loading_model"))
                                }
                                .frame(maxWidth: .infinity)
                                .frame(height: 50)
                            } else {
                                Text(settings.localized(isLoaded ? "reload_model" : "image_generator_load_model"))
                                    .frame(maxWidth: .infinity)
                                    .frame(height: 50)
                            }
                        }
                        .liquidGlassPrimaryButton(cornerRadius: 12)
                        .disabled(availableModels.isEmpty || isLoading)

                        if isLoaded {
                            Button(role: .destructive) {
                                onUnload()
                                dismiss()
                            } label: {
                                Text(settings.localized("unload_model"))
                                    .frame(maxWidth: .infinity)
                                    .frame(height: 50)
                            }
                            .background(
                                RoundedRectangle(cornerRadius: 12)
                                    .fill(ApolloPalette.destructive.opacity(0.10))
                            )
                            .overlay(
                                RoundedRectangle(cornerRadius: 12)
                                    .stroke(ApolloPalette.destructive.opacity(0.9), lineWidth: 1)
                            )
                            .foregroundStyle(ApolloPalette.destructive.opacity(0.98))
                        }
                    }
                }
                .padding()
            }
            .apolloScreenBackground()
            .navigationTitle(settings.localized("feature_settings_title"))
            .navigationBarTitleDisplayMode(.inline)
            .apolloNavigationBackground()
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button(settings.localized("close")) { dismiss() }
                }
            }
        }
    }
}

// MARK: - Video Saver
class VideoSaver: NSObject {
    var onComplete: ((Error?) -> Void)?

    func writeToPhotoAlbum(videoURL: URL, completion: @escaping (Error?) -> Void) {
        self.onComplete = completion
        UISaveVideoAtPathToSavedPhotosAlbum(videoURL.path, self, #selector(saveCompleted(_:didFinishSavingWithError:contextInfo:)), nil)
    }

    @objc func saveCompleted(_ videoPath: String, didFinishSavingWithError error: Error?, contextInfo: UnsafeRawPointer) {
        onComplete?(error)
    }
}
