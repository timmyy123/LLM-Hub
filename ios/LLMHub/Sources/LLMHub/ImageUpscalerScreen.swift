import SwiftUI
import PhotosUI
import ModelZoo

struct ImageUpscalerScreen: View {
    let onNavigateBack: () -> Void
    let onNavigateToModels: () -> Void

    @EnvironmentObject var settings: AppSettings
    @StateObject private var backend = ImageUpscalerBackend.shared

    @AppStorage("selected_upscaler_model_id") private var selectedModelId: String = ""
    @State private var inputImage: UIImage? = nil
    @State private var outputImage: UIImage? = nil
    @State private var selectedImageItem: PhotosPickerItem? = nil
    @State private var showSettings = false
    @State private var errorMessage: String? = nil
    @State private var showSaveAlert = false
    @State private var saveAlertTitle = ""
    @State private var saveAlertMessage = ""
    @State private var upscaleTask: Task<Void, Never>? = nil
    @State private var zoomScale: CGFloat = 1.0
    @State private var zoomOffset: CGSize = .zero

    // All upscale models
    private var allUpscaleModels: [AIModel] {
        ModelData.allModels().filter { $0.category == .imageUpscale }
    }

    // Only locally downloaded upscale models
    private var downloadedModels: [AIModel] {
        allUpscaleModels.filter { ModelData.isModelFullyAvailableLocally($0) }
    }

    private var selectedModel: AIModel? {
        downloadedModels.first(where: { $0.id == selectedModelId }) ?? downloadedModels.first
    }

    // "model selected" = a downloaded model is chosen
    private var isModelSelected: Bool {
        selectedModel != nil
    }

    var body: some View {
        Group {
            if downloadedModels.isEmpty {
                noModelView
            } else if !isModelSelected {
                selectModelView
            } else {
                mainUpscaleView
            }
        }
        .navigationTitle(settings.localized("image_upscale_title"))
        .navigationBarTitleDisplayMode(.inline)
        .apolloScreenBackground()
        .safeAreaInset(edge: .bottom, spacing: 0) { BannerAdContainer() }
        .toolbarBackground(.hidden, for: .navigationBar)
        .toolbar {
            ToolbarItem(placement: .navigationBarLeading) {
                Button {
                    upscaleTask?.cancel()
                    onNavigateBack()
                } label: { Image(systemName: "arrow.left") }
            }
            if !downloadedModels.isEmpty {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button { showSettings = true } label: {
                        Image(systemName: "slider.horizontal.3")
                    }
                }
            }
        }
        .sheet(isPresented: $showSettings) {
            ImageUpscalerSettingsSheet(
                selectedModelId: $selectedModelId,
                availableModels: downloadedModels
            )
            .environmentObject(settings)
        }
        .onChange(of: selectedImageItem) { _, item in
            guard let item else { inputImage = nil; return }
            Task {
                if let data = try? await item.loadTransferable(type: Data.self),
                   let img = UIImage(data: data) {
                    inputImage = img
                    outputImage = nil
                    resetZoom()
                } else {
                    inputImage = nil
                    outputImage = nil
                    resetZoom()
                }
            }
        }
        .onAppear {
            // Auto-select first downloaded model if stored selection is gone
            if !selectedModelId.isEmpty && !downloadedModels.contains(where: { $0.id == selectedModelId }) {
                selectedModelId = downloadedModels.first?.id ?? ""
            }
            if selectedModelId.isEmpty {
                selectedModelId = downloadedModels.first?.id ?? ""
            }
        }
        .onDisappear {
            upscaleTask?.cancel()
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

    // MARK: - No Downloaded Models State

    private var noModelView: some View {
        VStack(spacing: 20) {
            Image(systemName: "wand.and.stars")
                .font(.system(size: 56, weight: .semibold))
                .foregroundStyle(.secondary)

            Text(settings.localized("image_upscale_download_model"))
                .font(.title3.bold())
                .multilineTextAlignment(.center)

            Text(settings.localized("image_upscale_download_model_desc"))
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)

            Button {
                onNavigateToModels()
            } label: {
                Text(settings.localized("download_models"))
                    .frame(maxWidth: .infinity)
                    .frame(height: 52)
            }
            .foregroundStyle(.white)
            .liquidGlassPrimaryButton(cornerRadius: 12)
            .padding(.horizontal, 32)
        }
        .padding()
    }

    // MARK: - Model Downloaded but None Selected State

    private var selectModelView: some View {
        VStack(spacing: 20) {
            Image(systemName: "cpu.fill")
                .font(.system(size: 56, weight: .semibold))
                .foregroundStyle(.secondary)

            Text(settings.localized("image_upscale_load_model_title"))
                .font(.title3.bold())
                .multilineTextAlignment(.center)

            Text(settings.localized("image_upscale_load_model_desc"))
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
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    // MARK: - Main Upscale View

    @ViewBuilder
    private var mainUpscaleView: some View {
        GeometryReader { geo in
            let isLandscape = geo.size.width > geo.size.height
            if isLandscape {
                landscapeLayout
            } else {
                portraitLayout
            }
        }
    }

    // MARK: - Portrait Layout

    private var portraitLayout: some View {
        VStack(spacing: 14) {
            // Model chip
            if let model = selectedModel {
                HStack {
                    Text(model.name)
                        .font(.footnote.bold())
                        .foregroundStyle(ApolloPalette.accentStrong)
                    Spacer()
                }
                .padding(.horizontal, 16)
                .padding(.top, 8)
            }

            GeometryReader { imageArea in
                let cardSpacing: CGFloat = 12
                let hasOutput = outputImage != nil
                let cardHeight = hasOutput
                    ? max(1, (imageArea.size.height - cardSpacing) / 2)
                    : max(1, imageArea.size.height / 2)

                VStack(spacing: cardSpacing) {
                    // Input image card
                    ZoomableImageCard(
                        image: inputImage,
                        label: inputImage.map { "\(Int($0.size.width))×\(Int($0.size.height))" } ?? "",
                        placeholderIcon: "photo.on.rectangle",
                        placeholderText: settings.localized("image_upscale_select_image"),
                        showClose: !backend.isUpscaling && inputImage != nil,
                        zoomScale: $zoomScale,
                        zoomOffset: $zoomOffset,
                        onClose: {
                            inputImage = nil
                            outputImage = nil
                            resetZoom()
                        },
                        onTap: {
                            // trigger via overlay PhotosPicker button
                        }
                    )
                    .overlay {
                        if inputImage == nil {
                            PhotosPicker(selection: $selectedImageItem, matching: .images) {
                                Color.clear
                            }
                        }
                    }
                    .frame(maxWidth: .infinity)
                    .frame(height: cardHeight)

                    // Output image card
                    if let out = outputImage {
                        ZoomableImageCard(
                            image: out,
                            label: "\(Int(out.size.width))×\(Int(out.size.height))",
                            placeholderIcon: "sparkles",
                            placeholderText: "",
                            showClose: false,
                            zoomScale: $zoomScale,
                            zoomOffset: $zoomOffset,
                            onClose: {},
                            onTap: nil
                        )
                        .frame(maxWidth: .infinity)
                        .frame(height: cardHeight)
                    } else {
                        Spacer(minLength: 0)
                    }
                }
            }

            // Buttons
            upscaleButton
            if outputImage != nil { saveButton }
        }
        .padding(.horizontal, 16)
        .padding(.bottom, 16)
    }

    // MARK: - Landscape Layout

    private var landscapeLayout: some View {
        VStack(spacing: 12) {
            if let model = selectedModel {
                HStack {
                    Text(model.name)
                        .font(.footnote.bold())
                        .foregroundStyle(ApolloPalette.accentStrong)
                    Spacer()
                }
            }

            HStack(spacing: 12) {
                ZoomableImageCard(
                    image: inputImage,
                    label: inputImage.map { "\(Int($0.size.width))×\(Int($0.size.height))" } ?? "",
                    placeholderIcon: "photo.on.rectangle",
                    placeholderText: settings.localized("image_upscale_select_image"),
                    showClose: !backend.isUpscaling && inputImage != nil,
                    zoomScale: $zoomScale,
                    zoomOffset: $zoomOffset,
                    onClose: {
                        inputImage = nil
                        outputImage = nil
                        resetZoom()
                    },
                    onTap: nil
                )
                .overlay {
                    if inputImage == nil {
                        PhotosPicker(selection: $selectedImageItem, matching: .images) {
                            Color.clear
                        }
                    }
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)

                if let out = outputImage {
                    ZoomableImageCard(
                        image: out,
                        label: "\(Int(out.size.width))×\(Int(out.size.height))",
                        placeholderIcon: "sparkles",
                        placeholderText: "",
                        showClose: false,
                        zoomScale: $zoomScale,
                        zoomOffset: $zoomOffset,
                        onClose: {},
                        onTap: nil
                    )
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else {
                    Spacer()
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                }
            }
            .frame(maxWidth: .infinity)

            HStack(spacing: 12) {
                upscaleButton
                if outputImage != nil { saveButton }
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
    }

    // MARK: - Action Buttons

    private var upscaleButton: some View {
        Button {
            guard let model = selectedModel, let image = inputImage else { return }
            upscaleTask = Task {
                outputImage = nil
                resetZoom()
                do {
                    outputImage = try await backend.upscale(image: image, model: model)
                } catch {
                    if !(error is CancellationError) {
                        errorMessage = error.localizedDescription
                    }
                }
            }
        } label: {
            Group {
                if backend.isUpscaling {
                    HStack(spacing: 8) {
                        ProgressView()
                            .tint(.white)
                            .scaleEffect(0.85)
                        Text(settings.localized("image_upscale_upscaling"))
                    }
                } else {
                    HStack(spacing: 8) {
                        Image(systemName: "wand.and.stars")
                        Text(settings.localized("image_upscale_button"))
                    }
                }
            }
            .frame(maxWidth: .infinity)
            .frame(height: 50)
        }
        .disabled(inputImage == nil || backend.isUpscaling)
        .liquidGlassPrimaryButton(cornerRadius: 12)
    }

    private var saveButton: some View {
        Button {
            guard let out = outputImage else { return }
            UIImageWriteToSavedPhotosAlbum(out, nil, nil, nil)
            saveAlertTitle = settings.localized("image_generator_saved")
            saveAlertMessage = ""
            showSaveAlert = true
        } label: {
            HStack(spacing: 8) {
                Image(systemName: "square.and.arrow.down")
                Text(settings.localized("image_upscale_save"))
            }
            .frame(maxWidth: .infinity)
            .frame(height: 50)
        }
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(ApolloPalette.accentStrong.opacity(0.10))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 12)
                .stroke(ApolloPalette.accentStrong.opacity(0.6), lineWidth: 1)
        )
        .foregroundStyle(ApolloPalette.accentStrong)
    }

    private func resetZoom() {
        zoomScale = 1
        zoomOffset = .zero
    }
}

// MARK: - Zoomable Image Card

private struct ZoomableImageCard: View {
    let image: UIImage?
    let label: String
    let placeholderIcon: String
    let placeholderText: String
    let showClose: Bool
    @Binding var zoomScale: CGFloat
    @Binding var zoomOffset: CGSize
    let onClose: () -> Void
    let onTap: (() -> Void)?

    @State private var scaleAtGestureStart: CGFloat? = nil
    @State private var offsetAtGestureStart: CGSize? = nil

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            if !label.isEmpty {
                Text(label)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            ZStack(alignment: .topTrailing) {
                RoundedRectangle(cornerRadius: 16)
                    .fill(.ultraThinMaterial)
                    .overlay {
                        if let img = image {
                            Image(uiImage: img)
                                .resizable()
                                .scaledToFit()
                                .scaleEffect(zoomScale)
                                .offset(zoomOffset)
                                .gesture(transformGesture)
                                .onTapGesture(count: 2) {
                                    withAnimation(.spring()) {
                                        zoomScale = 1
                                        zoomOffset = .zero
                                    }
                                }
                        } else {
                            VStack(spacing: 10) {
                                Image(systemName: placeholderIcon)
                                    .font(.system(size: 40))
                                    .foregroundStyle(.secondary)
                                if !placeholderText.isEmpty {
                                    Text(placeholderText)
                                        .font(.subheadline)
                                        .foregroundStyle(.secondary)
                                }
                            }
                            .frame(maxWidth: .infinity, maxHeight: .infinity)
                            .contentShape(Rectangle())
                            .onTapGesture { onTap?() }
                        }
                    }
                    .clipShape(RoundedRectangle(cornerRadius: 16))
                    .onChange(of: image) { _, _ in
                        zoomScale = 1
                        zoomOffset = .zero
                    }

                // Close button
                if showClose {
                    Button {
                        onClose()
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                            .font(.title2)
                            .foregroundStyle(.secondary)
                            .background(Color.black.opacity(0.001)) // hit target
                    }
                    .padding(8)
                }
            }
        }
    }

    private var transformGesture: some Gesture {
        SimultaneousGesture(
            MagnificationGesture()
                .onChanged { value in
                    if scaleAtGestureStart == nil {
                        scaleAtGestureStart = zoomScale
                    }
                    zoomScale = ((scaleAtGestureStart ?? 1) * value).clamped(to: 1...8)
                    if zoomScale == 1 {
                        zoomOffset = .zero
                    }
                }
                .onEnded { _ in
                    scaleAtGestureStart = nil
                    if zoomScale == 1 {
                        zoomOffset = .zero
                    }
                },
            DragGesture()
                .onChanged { value in
                    guard zoomScale > 1 else { return }
                    if offsetAtGestureStart == nil {
                        offsetAtGestureStart = zoomOffset
                    }
                    let start = offsetAtGestureStart ?? .zero
                    zoomOffset = CGSize(
                        width: start.width + value.translation.width,
                        height: start.height + value.translation.height
                    )
                }
                .onEnded { _ in
                    offsetAtGestureStart = nil
                }
        )
    }
}

private extension Comparable {
    func clamped(to range: ClosedRange<Self>) -> Self {
        min(max(self, range.lowerBound), range.upperBound)
    }
}

// MARK: - Settings Sheet

struct ImageUpscalerSettingsSheet: View {
    @EnvironmentObject var settings: AppSettings
    @Binding var selectedModelId: String
    let availableModels: [AIModel]

    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 20) {
                    // Model Picker Card
                    VStack(alignment: .leading, spacing: 12) {
                        Text(settings.localized("image_upscale_load_model_title"))
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

                    // Load / confirm button
                    Button {
                        dismiss()
                    } label: {
                        Text(settings.localized("image_generator_load_model"))
                            .frame(maxWidth: .infinity)
                            .frame(height: 50)
                    }
                    .liquidGlassPrimaryButton(cornerRadius: 12)
                    .disabled(availableModels.isEmpty)
                }
                .padding()
            }
            .navigationTitle(settings.localized("feature_settings_title"))
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button(settings.localized("close")) { dismiss() }
                }
            }
        }
    }
}
