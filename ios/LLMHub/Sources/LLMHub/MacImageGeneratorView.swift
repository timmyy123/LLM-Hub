//
//  MacImageGeneratorView.swift
//  LLMHub
//
//  Native macOS Image Generator. Same persisted settings, backend flow and
//  localized strings as `ImageGeneratorScreen` on iOS.
//

#if os(macOS)
import SwiftUI
import PhotosUI

struct MacImageGeneratorView: View {
    @EnvironmentObject var settings: AppSettings
    @AppStorage("sd_selected_model_id") private var selectedModelId: String = ""
    @AppStorage("sd_steps") private var storedSteps: Double = 20
    @AppStorage("sd_denoise_strength") private var storedDenoiseStrength: Double = 0.7

    @State private var promptText = ""
    @FocusState private var promptFocused: Bool
    @State private var seed: Int = Int.random(in: 0..<1_000_000)
    @State private var generatedImages: [UIImage] = []
    @State private var isGenerating = false
    @State private var currentPage = 0
    @State private var showInspector = false
    @State private var errorMessage: String?
    @State private var inputImage: UIImage?
    @State private var selectedImageItem: PhotosPickerItem?
    @State private var generateTask: Task<Void, Never>?
    @State private var imageSaver = ImageSaver()
    @State private var showSaveAlert = false
    @State private var saveAlertTitle = ""
    @State private var saveAlertMessage = ""

    @ObservedObject private var sdBackend = StableDiffusionBackend.shared

    let onNavigateToModels: () -> Void

    private var availableModels: [AIModel] {
        ModelData.allModels().filter { $0.isDrawThingsImageGeneration && StableDiffusionBackend.isModelDownloaded(modelId: $0.id) }
    }

    private var selectedModel: AIModel? {
        availableModels.first(where: { $0.id == selectedModelId }) ?? availableModels.first
    }

    private var selectedModelSupportsImageToImage: Bool {
        guard let selectedModel else { return false }
        return StableDiffusionBackend.supportsImageToImage(modelId: selectedModel.id)
    }

    private var isModelDownloaded: Bool {
        guard let model = selectedModel else { return false }
        return StableDiffusionBackend.isModelDownloaded(modelId: model.id)
    }

    private var isPromptEmpty: Bool {
        promptText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
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
                            title: sdBackend.isLoading ? settings.localized("model_loading")
                                : isGenerating ? settings.localized("image_generator_generating")
                                : settings.localized("image_generator_generate"),
                            systemImage: (isGenerating || sdBackend.isLoading) ? "stop.fill" : "sparkles",
                            isBusy: sdBackend.isLoading,
                            isEnabled: isGenerating || !isPromptEmpty,
                            tint: isGenerating ? .red : ApolloPalette.accentStrong
                        ) {
                            if isGenerating {
                                generateTask?.cancel()
                                isGenerating = false
                            } else {
                                startGeneration(clearAll: true)
                            }
                        }
                    }
            }
        }
        .navigationTitle(settings.localized("image_generator_title"))
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
            settingsInspector
                .inspectorColumnWidth(min: 280, ideal: 320, max: 420)
        }
        .onChange(of: selectedImageItem) { _, item in
            guard selectedModelSupportsImageToImage else {
                selectedImageItem = nil
                inputImage = nil
                return
            }
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
            if !selectedModelSupportsImageToImage {
                selectedImageItem = nil
                inputImage = nil
            }
        }
        .onChange(of: selectedModelId) { _, _ in
            if !selectedModelSupportsImageToImage {
                selectedImageItem = nil
                inputImage = nil
            }
        }
        .onDisappear {
            generateTask?.cancel()
            sdBackend.unloadModel()
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
            Label(settings.localized("image_generator_download_model"), systemImage: "paintpalette.fill")
        } description: {
            Text(settings.localized("image_generator_download_model_desc"))
        } actions: {
            Button(settings.localized("download_models"), action: onNavigateToModels)
                .buttonStyle(.borderedProminent)
        }
    }

    private var loadModelView: some View {
        Group {
            if sdBackend.isLoading {
                VStack(spacing: 16) {
                    ProgressView()
                    Text(settings.localized("image_generator_loading_model"))
                        .font(.title3.weight(.bold))
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                ContentUnavailableView {
                    Label(settings.localized("image_generator_load_model_title"), systemImage: "cpu.fill")
                } description: {
                    Text(settings.localized("image_generator_load_model_desc"))
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
                    if selectedModelSupportsImageToImage {
                        Divider()
                        img2imgPanel
                    }
                }
                .padding()
            }
            .frame(minWidth: 300, idealWidth: 360, maxWidth: 420)

            Divider()

            Group {
                if !generatedImages.isEmpty || isGenerating {
                    imageSwipeView
                        .padding()
                } else {
                    Image(systemName: "photo.on.rectangle.angled")
                        .font(.system(size: 56, weight: .light))
                        .foregroundStyle(.tertiary)
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
    }

    private var promptPanel: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(settings.localized("image_generator_prompt_label"))
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
                        Text(settings.localized("image_generator_prompt_hint"))
                            .foregroundStyle(.secondary)
                            .padding(.horizontal, 11)
                            .padding(.vertical, 6)
                            .allowsHitTesting(false)
                    }
                }
                .disabled(isGenerating)
        }
    }

    private var img2imgPanel: some View {
        let labelText = settings.localized(inputImage != nil ? "image_generator_change_image" : "image_generator_select_image")
        return VStack(alignment: .leading, spacing: 10) {
            Text(settings.localized("image_generator_img2img"))
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

            if inputImage != nil {
                VStack(alignment: .leading, spacing: 4) {
                    Text(String(format: settings.localized("image_generator_denoise_strength"), storedDenoiseStrength))
                        .font(.caption)
                    Text(settings.localized("image_generator_denoise_strength_desc"))
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    Slider(value: $storedDenoiseStrength, in: 0.1...1.0)
                        .disabled(isGenerating)
                }
            }
        }
    }

    // MARK: - Pager

    @ViewBuilder
    private func imageSwipePage(_ page: Int) -> some View {
        if page < generatedImages.count {
            Image(uiImage: generatedImages[page])
                .resizable()
                .aspectRatio(1, contentMode: .fit)
                .clipShape(RoundedRectangle(cornerRadius: 12))
                .padding(.horizontal, 4)
                .contextMenu {
                    Button {
                        saveImageToPhotos(generatedImages[page])
                    } label: {
                        Label(settings.localized("image_generator_save"), systemImage: "square.and.arrow.down")
                    }
                }
        } else {
            placeholderPage
        }
    }

    private var imagePager: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            LazyHStack(spacing: 0) {
                ForEach(0..<(generatedImages.count + 1), id: \.self) { page in
                    imageSwipePage(page)
                        .containerRelativeFrame(.horizontal)
                        .id(page)
                }
            }
            .scrollTargetLayout()
        }
        .scrollTargetBehavior(.paging)
        .scrollPosition(id: Binding(get: { currentPage }, set: { currentPage = $0 ?? currentPage }))
        .overlay {
            HStack {
                Button {
                    withAnimation { currentPage = max(0, currentPage - 1) }
                } label: {
                    Image(systemName: "chevron.left.circle.fill").font(.title)
                }
                .buttonStyle(.plain)
                .opacity(currentPage > 0 ? 0.85 : 0)
                Spacer()
                Button {
                    withAnimation { currentPage = min(generatedImages.count, currentPage + 1) }
                } label: {
                    Image(systemName: "chevron.right.circle.fill").font(.title)
                }
                .buttonStyle(.plain)
                .opacity(currentPage < generatedImages.count ? 0.85 : 0)
            }
            .foregroundStyle(.white)
            .padding(.horizontal, 10)
        }
        .aspectRatio(1, contentMode: .fit)
    }

    private var imageSwipeView: some View {
        VStack(spacing: 10) {
            imagePager
                .onChange(of: currentPage) { _, page in
                    checkForPrefetch(page: page)
                }
                .onChange(of: isGenerating) { _, generating in
                    if !generating {
                        checkForPrefetch(page: currentPage)
                    }
                }

            HStack(spacing: 6) {
                ForEach(0..<generatedImages.count, id: \.self) { i in
                    Circle()
                        .fill(currentPage == i ? Color.white : Color.white.opacity(0.35))
                        .frame(width: 7, height: 7)
                }
                Circle()
                    .fill(currentPage == generatedImages.count ? Color.white : Color.white.opacity(0.35))
                    .frame(width: 7, height: 7)
            }

            if !generatedImages.isEmpty && currentPage < generatedImages.count {
                Button {
                    if currentPage < generatedImages.count {
                        saveImageToPhotos(generatedImages[currentPage])
                    }
                } label: {
                    Label(settings.localized("image_generator_save"), systemImage: "square.and.arrow.down")
                }
                .buttonStyle(.bordered)
            }
        }
    }

    private var placeholderPage: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 12)
                .fill(Color.white.opacity(0.05))
                .overlay(RoundedRectangle(cornerRadius: 12).stroke(Color(nsColor: .separatorColor), lineWidth: 1))
            if isGenerating {
                VStack(spacing: 16) {
                    ProgressView()
                    Text(String(format: settings.localized("image_generator_variation"), generatedImages.count + 1))
                        .font(.subheadline.bold())
                        .multilineTextAlignment(.center)
                    if sdBackend.generationTotalSteps > 0 {
                        VStack(spacing: 8) {
                            ProgressView(value: Double(sdBackend.generationStep), total: Double(sdBackend.generationTotalSteps))
                                .frame(width: 180)
                            Text(String(format: settings.localized("image_generator_step_of"), sdBackend.generationStep, sdBackend.generationTotalSteps))
                                .font(.caption.bold())
                                .foregroundStyle(.secondary)
                                .monospacedDigit()
                        }
                    }
                }
            } else {
                VStack(spacing: 12) {
                    Image(systemName: "plus.circle")
                        .font(.system(size: 44, weight: .light))
                        .foregroundStyle(.secondary)
                    Text(settings.localized("image_generator_swipe_more"))
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)
                }
            }
        }
        .aspectRatio(1, contentMode: .fit)
        .clipShape(RoundedRectangle(cornerRadius: 12))
        .padding(.horizontal, 4)
    }

    // MARK: - Inspector (ImageGeneratorSettingsSheet)

    private var settingsInspector: some View {
        Form {
            Section(settings.localized("select_model_title")) {
                if availableModels.isEmpty {
                    Label(settings.localized("image_generator_no_models"), systemImage: "exclamationmark.triangle")
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
                LabeledContent(settings.localized("image_generator_iterations")) {
                    Text("\(Int(storedSteps))").monospacedDigit()
                }
                ApolloSlider(value: $storedSteps, in: 1...50, step: 1)
                    .labelsHidden()
            }

            Section {
                VStack(spacing: 8) {
                    Button {
                        guard let model = selectedModel else { return }
                        Task {
                            do {
                                try await sdBackend.loadModel(model)
                            } catch {
                                errorMessage = error.localizedDescription
                            }
                        }
                    } label: {
                        Group {
                            if sdBackend.isLoading {
                                HStack(spacing: 6) {
                                    ProgressView().controlSize(.small)
                                    Text(settings.localized("image_generator_loading_model"))
                                }
                            } else {
                                Text(settings.localized(sdBackend.isLoaded ? "reload_model" : "image_generator_load_model"))
                                    .fontWeight(.semibold)
                            }
                        }
                        .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.large)
                    .disabled(availableModels.isEmpty || sdBackend.isLoading)

                    if sdBackend.isLoaded {
                        Button(role: .destructive) {
                            sdBackend.unloadModel()
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

    private func startGeneration(clearAll: Bool) {
        guard !isPromptEmpty else { return }
        if clearAll {
            generatedImages.removeAll()
            currentPage = 0
            seed = Int.random(in: 0..<1_000_000)
        }
        runGeneration(usingSeed: UInt32(bitPattern: Int32(truncatingIfNeeded: seed)))
    }

    private func checkForPrefetch(page: Int) {
        guard !isGenerating, sdBackend.isLoaded, !generatedImages.isEmpty else { return }
        let shouldPrefetch = page < generatedImages.count && page == generatedImages.count - 1
        let swipedToPlaceholder = page == generatedImages.count
        if shouldPrefetch || swipedToPlaceholder {
            triggerVariation()
        }
    }

    private func triggerVariation() {
        guard !isGenerating, sdBackend.isLoaded else { return }
        let varSeed = UInt32.random(in: 0..<UInt32.max)
        runGeneration(usingSeed: varSeed)
    }

    private func runGeneration(usingSeed genSeed: UInt32) {
        isGenerating = true
        promptFocused = false
        let prompt = promptText
        let steps = Int(storedSteps)
        let denoiseStrength = Float(storedDenoiseStrength)
        let startingImage = selectedModelSupportsImageToImage ? inputImage : nil

        generateTask?.cancel()
        generateTask = Task {
            defer { isGenerating = false }
            do {
                if !sdBackend.isLoaded {
                    guard let model = selectedModel else {
                        errorMessage = settings.localized("image_generator_no_model")
                        return
                    }
                    try await sdBackend.loadModel(model)
                }
                let img = try await sdBackend.generateImage(
                    prompt: prompt,
                    steps: steps,
                    seed: genSeed,
                    inputImage: startingImage,
                    denoiseStrength: denoiseStrength
                )
                if let img {
                    let oldCount = generatedImages.count
                    generatedImages.append(img)
                    if currentPage >= oldCount {
                        currentPage = generatedImages.count - 1
                    }
                }
            } catch is CancellationError {
                // ignore
            } catch {
                errorMessage = error.localizedDescription
            }
        }
    }

    private func saveImageToPhotos(_ image: UIImage) {
        imageSaver.writeToPhotoAlbum(image: image) { error in
            DispatchQueue.main.async {
                if let error = error {
                    self.saveAlertTitle = self.settings.localized("error")
                    self.saveAlertMessage = String(format: self.settings.localized("image_generator_save_failed") + ": %@", error.localizedDescription)
                } else {
                    self.saveAlertTitle = self.settings.localized("success")
                    self.saveAlertMessage = self.settings.localized("image_generator_saved")
                }
                self.showSaveAlert = true
            }
        }
    }
}
#endif
