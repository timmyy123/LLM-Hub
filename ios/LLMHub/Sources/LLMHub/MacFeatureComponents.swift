//
//  MacFeatureComponents.swift
//  LLMHub
//
//  Shared building blocks for the native macOS feature screens. They drive the
//  same backends, persisted keys and localized strings as the iOS screens.
//

#if os(macOS)
import SwiftUI

/// Native macOS counterpart of `FeatureModelSettingsSheet`, shown in an
/// inspector column. Logic (model filtering, context caps, GPU layers, toggle
/// normalization, load/unload) mirrors the iOS sheet exactly.
struct MacFeatureModelInspector: View {
    @EnvironmentObject var settings: AppSettings
    @Binding var selectedModelName: String
    @Binding var maxTokens: Double
    @Binding var enableThinking: Bool
    @Binding var enableVision: Bool
    var enableAudio: Binding<Bool>? = nil
    @Binding var isLoading: Bool
    @Binding var errorMessage: String?
    var supportsVisionToggle: Bool = false
    var visionToggleTitleKey: String = "scam_detector_enable_vision"
    var audioToggleTitleKey: String? = nil
    var visionAvailableCheck: ((AIModel) -> Bool)? = nil
    var writingMode: Binding<WritingAidMode>? = nil
    var modelFilter: ((AIModel) -> Bool)? = nil
    let onLoad: () async -> Void
    let onUnload: () -> Void
    var showsThinkingToggle: Bool = false
    var extraModelConfigsContent: AnyView? = nil

    @ObservedObject private var llm = LLMBackend.shared
    @ObservedObject private var whisperBackend = WhisperBackend.shared
    @ObservedObject private var musicBackend = MusicGeneratorBackend.shared
    @State private var models: [AIModel] = []
    @State private var isRefreshingModels = false
    @State private var gpuLayersTemp: Double = 999
    @State private var gpuLayerLimit: Double = 999

    private var selectedModel: AIModel? {
        models.first(where: { $0.name == selectedModelName })
            ?? selectedFeatureModel(named: selectedModelName)
    }

    private var selectedModelSupportsThinking: Bool {
        guard let model = selectedModel else { return false }
        let name = model.name.lowercased()
        if name.contains("lfm") { return false }
        if name.contains("granite-4.2") || name.contains("granite 4.2") { return false }
        return model.supportsThinking
    }

    private var selectedModelSupportsVision: Bool {
        guard supportsVisionToggle, let model = selectedModel, model.supportsVision else { return false }
        return visionAvailableCheck?(model) ?? true
    }

    private var selectedModelSupportsAudio: Bool {
        selectedModel?.isGemma4LiteRTLM ?? false
    }

    private var maxContextCap: Double {
        guard let selectedModel else { return 4096 }
        let cap = selectedModel.modelFormat == .gguf
            ? llm.modelMaxContextWindow(for: selectedModel)
            : (selectedModel.contextWindowSize > 0 ? selectedModel.contextWindowSize : 4096)
        return Double(max(2, cap))
    }

    private var isSelectedModelLoaded: Bool {
        if let model = selectedModel, model.isWhisperModel {
            return whisperBackend.isLoaded && whisperBackend.currentModelName == model.name
        }
        if selectedModel?.category == .musicGeneration {
            return musicBackend.isLoaded && musicBackend.loadedModelName == selectedModelName
        }
        return llm.isLoaded && llm.currentlyLoadedModel == selectedModelName
    }

    var body: some View {
        Form {
            Section(settings.localized("select_model")) {
                if isRefreshingModels {
                    HStack(spacing: 8) {
                        ProgressView().controlSize(.small)
                        Text(settings.localized("loading"))
                            .foregroundStyle(.secondary)
                    }
                } else {
                    Picker(settings.localized("select_model"), selection: $selectedModelName) {
                        ForEach(models, id: \.id) { model in
                            Text(model.name).tag(model.name)
                        }
                    }
                    .labelsHidden()
                }
            }

            if selectedModel?.isWhisperModel != true && selectedModel?.category != .musicGeneration {
                Section {
                    LabeledContent(settings.localized("context_window_size")) {
                        Text("\(Int(maxTokens))").monospacedDigit()
                    }
                    if maxContextCap > 1 {
                        ApolloSlider(value: $maxTokens, in: 1...maxContextCap, step: 1) { editing in
                            if !editing {
                                maxTokens = min(max(1, maxTokens), maxContextCap)
                            }
                        }
                        .labelsHidden()
                    }
                    if let model = selectedModel, model.modelFormat == .gguf {
                        LabeledContent(settings.localized("gpu_layers_label").replacingOccurrences(of: "%1$d", with: "\(Int(gpuLayersTemp))")) {
                            Text("\(Int(gpuLayersTemp))").monospacedDigit()
                        }
                        ApolloSlider(value: $gpuLayersTemp, in: 0...gpuLayerLimit, step: 1) { editing in
                            if !editing { saveGpuLayers(gpuLayersTemp) }
                        }
                        .labelsHidden()
                    }
                }
            }

            let showsThinking = showsThinkingToggle && selectedModelSupportsThinking
            let showsAudio = enableAudio != nil && selectedModelSupportsAudio
            if showsThinking || selectedModelSupportsVision || showsAudio || writingMode != nil {
                Section {
                    if showsThinking {
                        Toggle(settings.localized("enable_thinking"), isOn: $enableThinking)
                    }
                    if selectedModelSupportsVision {
                        Toggle(settings.localized(visionToggleTitleKey), isOn: $enableVision)
                    }
                    if let enableAudio, showsAudio {
                        Toggle(settings.localized(audioToggleTitleKey ?? "enable_audio"), isOn: enableAudio)
                    }
                    if let writingMode {
                        Picker(settings.localized("writing_aid_select_mode"), selection: writingMode) {
                            ForEach(WritingAidMode.allCases, id: \.rawValue) { mode in
                                Text(settings.localized(mode.rawValue)).tag(mode)
                            }
                        }
                    }
                }
            }

            if let extraModelConfigsContent {
                Section { extraModelConfigsContent }
            }

            Section {
                VStack(spacing: 8) {
                    Button {
                        Task { await onLoad() }
                    } label: {
                        Group {
                            if isLoading {
                                ProgressView().controlSize(.small)
                            } else {
                                Text(settings.localized(isSelectedModelLoaded ? "reload_model" : "load_model"))
                                    .fontWeight(.semibold)
                            }
                        }
                        .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.large)
                    .disabled(isLoading || selectedModelName.isEmpty || isRefreshingModels)

                    if isSelectedModelLoaded {
                        Button(role: .destructive) {
                            onUnload()
                        } label: {
                            Text(settings.localized("unload_model")).frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.large)
                        .disabled(isLoading)
                    }
                }
                if let errorMessage, !errorMessage.isEmpty {
                    Text(errorMessage)
                        .font(.caption)
                        .foregroundStyle(.red)
                }
            }
        }
        .formStyle(.grouped)
        .task {
            await refreshModelsIfNeeded()
            normalizeToggleStatesForSelectedModel()
            loadInitialGpuLayers()
        }
        .onChange(of: selectedModelName) { _, _ in
            normalizeToggleStatesForSelectedModel()
            loadInitialGpuLayers()
        }
    }

    private func refreshModelsIfNeeded() async {
        if !models.isEmpty { return }
        isRefreshingModels = true
        var loaded: [AIModel]
        if let modelFilter {
            loaded = ModelData.allModels().filter { model in
                guard !model.isDependencyOnly else { return false }
                let cat = model.category
                guard cat != .embedding && cat != .imageGeneration && cat != .videoGeneration && cat != .imageUpscale else { return false }
                guard model.name.lowercased().contains("vision projector") == false,
                      model.name.lowercased().contains("mmproj") == false,
                      model.name.lowercased().contains("projector") == false else { return false }
                guard ModelData.isModelFullyAvailableLocally(model) else { return false }
                return modelFilter(model)
            }
        } else {
            loaded = downloadableFeatureModels()
        }
        if let appleModel = appleFoundationModelIfAvailable(),
           modelFilter?(appleModel) ?? true,
           !loaded.contains(where: { $0.id == appleModel.id }) {
            loaded.append(appleModel)
        }
        models = loaded
        if selectedModelName.isEmpty || !loaded.contains(where: { $0.name == selectedModelName }) {
            selectedModelName = loaded.first?.name ?? ""
        }
        maxTokens = min(max(1, maxTokens), Double(maxContextCap))
        isRefreshingModels = false
    }

    private func normalizeToggleStatesForSelectedModel() {
        enableThinking = false
        if supportsVisionToggle && !selectedModelSupportsVision {
            enableVision = false
        }
    }

    private func loadInitialGpuLayers() {
        guard let selectedModel else { return }
        let limit = LLMBackend.shared.modelMaxGpuLayers(for: selectedModel)
        let key = "gpu_layers_\(selectedModel.id)"
        let hasStored = UserDefaults.standard.object(forKey: key) != nil
        let stored = hasStored ? UserDefaults.standard.integer(forKey: key) : -1

        if limit != GGUFLayerLimits.unknown {
            gpuLayerLimit = Double(limit)
            if !hasStored || stored == 999 || stored == 99 || stored > limit {
                gpuLayersTemp = Double(limit)
                saveGpuLayers(Double(limit))
            } else {
                gpuLayersTemp = min(max(0, Double(stored)), Double(limit))
            }
        } else {
            gpuLayerLimit = Double(GGUFLayerLimits.unknown)
            gpuLayersTemp = hasStored ? Double(stored) : Double(GGUFLayerLimits.unknown)
        }
    }

    private func saveGpuLayers(_ value: Double) {
        guard let selectedModel else { return }
        let key = "gpu_layers_\(selectedModel.id)"
        let intValue = Int32(min(max(0, value), gpuLayerLimit))
        UserDefaults.standard.set(intValue, forKey: key)
    }
}

/// Full-width primary action docked under a feature's content (the macOS
/// counterpart of the large bottom button on iOS). Triggered with ⌘↩.
struct MacPrimaryActionBar: View {
    let title: String
    let systemImage: String
    var isBusy: Bool = false
    var isEnabled: Bool = true
    var tint: Color = ApolloPalette.accentStrong
    let action: () -> Void

    var body: some View {
        VStack(spacing: 0) {
            Divider()
            Button(action: action) {
                HStack(spacing: 8) {
                    if isBusy {
                        ProgressView().controlSize(.small)
                    } else {
                        Image(systemName: systemImage)
                    }
                    Text(title).fontWeight(.semibold)
                }
                .frame(maxWidth: .infinity)
                .frame(height: 30)
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)
            .tint(tint)
            .keyboardShortcut(.return, modifiers: .command)
            .disabled(!isEnabled)
            .padding(.horizontal, 16)
            .padding(.vertical, 12)
        }
        .background(.bar)
    }
}

/// Empty state shown when a feature has no model selected (same strings as iOS).
struct MacFeatureLoadModelPrompt: View {
    @EnvironmentObject var settings: AppSettings
    let onOpenSettings: () -> Void

    var body: some View {
        ContentUnavailableView {
            Label(settings.localized("scam_detector_load_model"), systemImage: "cpu")
        } description: {
            Text(settings.localized("scam_detector_load_model_desc"))
        } actions: {
            Button(settings.localized("feature_settings_title"), action: onOpenSettings)
                .buttonStyle(.borderedProminent)
        }
    }
}

/// A titled, bordered text editing area in the native macOS style.
struct MacTextPanel: View {
    let title: String
    @Binding var text: String
    var minHeight: CGFloat = 140

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title).font(.headline)
            TextEditor(text: $text)
                .font(.body)
                .scrollContentBackground(.hidden)
                .padding(6)
                .frame(minHeight: minHeight)
                .background(Color(nsColor: .textBackgroundColor).opacity(0.35), in: RoundedRectangle(cornerRadius: 8))
                .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color(nsColor: .separatorColor)))
        }
    }
}
#endif
