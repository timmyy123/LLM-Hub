//
//  MacImageUpscalerView.swift
//  LLMHub
//
//  Native macOS Image Upscaler. Same persisted settings, backend flow and
//  localized strings as `ImageUpscalerScreen` on iOS.
//

#if os(macOS)
import SwiftUI
import PhotosUI

struct MacImageUpscalerView: View {
    let onNavigateToModels: () -> Void

    @EnvironmentObject var settings: AppSettings
    @StateObject private var backend = ImageUpscalerBackend.shared

    @AppStorage("selected_upscaler_model_id") private var selectedModelId: String = ""
    @State private var inputImage: UIImage? = nil
    @State private var outputImage: UIImage? = nil
    @State private var selectedImageItem: PhotosPickerItem? = nil
    @State private var showInspector = false
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
                    .safeAreaInset(edge: .bottom, spacing: 0) {
                        MacPrimaryActionBar(
                            title: backend.isUpscaling ? settings.localized("image_upscale_upscaling")
                                : settings.localized("image_upscale_button"),
                            systemImage: backend.isUpscaling ? "hourglass" : "wand.and.stars",
                            isBusy: backend.isUpscaling,
                            isEnabled: !(inputImage == nil || backend.isUpscaling)
                        ) {
                            startUpscale()
                        }
                    }
            }
        }
        .navigationTitle(settings.localized("image_upscale_title"))
        .toolbar {
            if !downloadedModels.isEmpty && isModelSelected {
                if outputImage != nil {
                    ToolbarItem(placement: .primaryAction) {
                        Button {
                            saveOutput()
                        } label: {
                            Label(settings.localized("image_upscale_save"), systemImage: "square.and.arrow.down")
                        }
                    }
                }
            }
            if !downloadedModels.isEmpty {
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
            Label(settings.localized("image_upscale_download_model"), systemImage: "wand.and.stars")
        } description: {
            Text(settings.localized("image_upscale_download_model_desc"))
        } actions: {
            Button(settings.localized("download_models"), action: onNavigateToModels)
                .buttonStyle(.borderedProminent)
        }
    }

    private var selectModelView: some View {
        ContentUnavailableView {
            Label(settings.localized("image_upscale_load_model_title"), systemImage: "cpu.fill")
        } description: {
            Text(settings.localized("image_upscale_load_model_desc"))
        } actions: {
            Button(settings.localized("feature_settings_title")) { showInspector = true }
                .buttonStyle(.borderedProminent)
        }
    }

    // MARK: - Main

    private var mainUpscaleView: some View {
        VStack(alignment: .leading, spacing: 12) {
            if let model = selectedModel {
                Text(model.name)
                    .font(.footnote.bold())
                    .foregroundStyle(ApolloPalette.accentStrong)
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
                            Color.clear.contentShape(Rectangle())
                        }
                        .buttonStyle(.plain)
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
                } else if backend.isUpscaling {
                    VStack(spacing: 10) {
                        ProgressView()
                        Text(settings.localized("image_upscale_upscaling"))
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else {
                    Spacer()
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .padding()
    }

    // MARK: - Inspector (ImageUpscalerSettingsSheet)

    private var settingsInspector: some View {
        Form {
            Section(settings.localized("image_upscale_load_model_title")) {
                if downloadedModels.isEmpty {
                    Label(settings.localized("no_models_available"), systemImage: "exclamationmark.triangle")
                        .foregroundStyle(.secondary)
                } else {
                    Picker(settings.localized("select_model"), selection: $selectedModelId) {
                        ForEach(downloadedModels) { model in
                            Text(model.name).tag(model.id)
                        }
                    }
                    .labelsHidden()
                }
            }

            Section {
                // Load / confirm button: the iOS sheet only dismisses itself here.
                VStack(spacing: 8) {
                    Button {
                        showInspector = false
                    } label: {
                        Text(settings.localized("image_generator_load_model"))
                            .fontWeight(.semibold)
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.large)
                    .disabled(downloadedModels.isEmpty)
                }
            }
        }
        .formStyle(.grouped)
    }

    // MARK: - Actions

    private func startUpscale() {
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
    }

    private func saveOutput() {
        guard let out = outputImage else { return }
        UIImageWriteToSavedPhotosAlbum(out, nil, nil, nil)
        saveAlertTitle = settings.localized("image_generator_saved")
        saveAlertMessage = ""
        showSaveAlert = true
    }

    private func resetZoom() {
        zoomScale = 1
        zoomOffset = .zero
    }
}
#endif
