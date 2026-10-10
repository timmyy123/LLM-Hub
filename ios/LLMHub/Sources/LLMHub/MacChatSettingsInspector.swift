//
//  MacChatSettingsInspector.swift
//  LLMHub
//
//  Native macOS counterpart of `ChatSettingsSheet`, shown in an inspector
//  column next to the chat. Draft state, commit-on-release behavior, model
//  list computation, GPU layer limits, context caps, modality rules, reset and
//  per-model system prompt persistence mirror the iOS sheet exactly.
//

#if os(macOS)
import SwiftUI
#if canImport(FoundationModels)
import FoundationModels
#endif

struct MacChatSettingsInspector: View {
    @ObservedObject var vm: ChatViewModel
    @EnvironmentObject var settings: AppSettings
    /// Called after the Load button finishes loading (the iOS sheet dismisses
    /// itself here; the inspector stays open and only invokes this if set).
    var onLoaded: (() -> Void)? = nil
    /// Called after Unload (the iOS sheet dismisses itself here).
    var onUnloaded: (() -> Void)? = nil

    @State private var draftContextWindow: Double = 2048
    @State private var draftTopK: Double = 64
    @State private var draftTopP: Double = 0.95
    @State private var draftTemperature: Double = 1.0
    @State private var draftSystemPrompt: String = ""
    @State private var gpuLayersTemp: Double = 999
    @State private var gpuLayerLimit: Double = 999
    @State private var cachedModels: [AIModel] = []

    var body: some View {
        Form {
            // Model Selection
            Section {
                Picker(settings.localized("select_model"), selection: $vm.selectedModelName) {
                    if vm.selectedModelName == settings.localized("no_model_selected") {
                        Text(settings.localized("no_model_selected")).tag(settings.localized("no_model_selected"))
                    }
                    ForEach(cachedModels) { model in
                        Text(model.name).tag(model.name)
                    }
                }
                .pickerStyle(.menu)
            } header: {
                Label(settings.localized("select_model_title"), systemImage: "cpu")
            }

            // Model Configurations
            Section {
                MacChatConfigSlider(title: settings.localized("context_window_size"), value: $draftContextWindow, range: 1...modelMaxContextWindow, format: "%.0f", subtitle: "max \(Int(modelMaxContextWindow))", step: 1, onCommit: applyDraftToViewModel)
                MacChatConfigSlider(title: settings.localized("top_k"), value: $draftTopK, range: 1...256, format: "%.0f", onCommit: applyDraftToViewModel)
                MacChatConfigSlider(title: settings.localized("top_p"), value: $draftTopP, range: 0...1, format: "%.2f", onCommit: applyDraftToViewModel)
                MacChatConfigSlider(title: settings.localized("temperature"), value: $draftTemperature, range: 0...2, format: "%.2f", onCommit: applyDraftToViewModel)

                if let model = currentModel, model.modelFormat == .gguf {
                    MacChatGPULayersSlider(
                        value: $gpuLayersTemp,
                        maxLayers: gpuLayerLimit,
                        label: settings.localized("gpu_layers_label"),
                        onCommit: { saveGpuLayers(gpuLayersTemp) }
                    )
                }

                HStack {
                    Spacer()
                    Button(settings.localized("reset_to_defaults")) {
                        resetAllConfigsToDefaults()
                    }
                    .buttonStyle(.bordered)
                }
            } header: {
                Label(settings.localized("model_configs_title"), systemImage: "slider.horizontal.3")
            }

            // System Prompt
            Section {
                TextEditor(text: $draftSystemPrompt)
                    .font(.body)
                    .frame(minHeight: 100)
                    .scrollContentBackground(.hidden)
                    .padding(4)
                    .background(Color(nsColor: .textBackgroundColor).opacity(0.35), in: RoundedRectangle(cornerRadius: 6))
                    .overlay(RoundedRectangle(cornerRadius: 6).stroke(Color(nsColor: .separatorColor)))
                    .labelsHidden()
            } header: {
                Label(settings.localized("model_system_prompt_label"), systemImage: "text.justify.left")
            } footer: {
                Text(settings.localized("model_system_prompt_hint"))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            // Modality Options
            let visionAvailable = currentModel.map { $0.supportsVision && LLMBackend.shared.isVisionProjectorAvailable(for: $0) } ?? false
            if let model = currentModel, (visionAvailable || model.supportsAudio || model.supportsThinking) {
                Section {
                    if visionAvailable {
                        MacChatToggleRow(title: settings.localized("enable_vision"), isOn: $vm.enableVision, icon: "eye.fill")
                    }
                    if model.supportsAudio {
                        MacChatToggleRow(title: settings.localized("enable_audio"), isOn: $vm.enableAudio, icon: "mic.fill")
                    }
                    if model.supportsThinking {
                        MacChatToggleRow(title: settings.localized("enable_thinking"), isOn: $vm.enableThinking, icon: "brain")
                    }
                    if model.name.contains("Gemma 4") && model.modelFormat == .litertlm {
                        MacChatToggleRow(
                            title: settings.localized("agent_tools_label"),
                            subtitle: settings.localized("agent_tools_description"),
                            isOn: $vm.enableAgentTools,
                            icon: "square.stack.3d.up.fill"
                        )
                    }
                } header: {
                    Label(settings.localized("modality_options"), systemImage: "sparkles")
                }
            }

            // Actions
            Section {
                VStack(spacing: 8) {
                    Button {
                        applyDraftToViewModel()
                        Task {
                            await vm.loadModelIfNecessary(force: true)
                            onLoaded?()
                        }
                    } label: {
                        HStack(spacing: 6) {
                            if vm.isBackendLoading {
                                ProgressView().controlSize(.small)
                            } else {
                                Image(systemName: "arrow.clockwise.circle.fill")
                            }
                            Text(settings.localized("load_model")).fontWeight(.semibold)
                        }
                        .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.large)
                    .disabled(vm.isBackendLoading || vm.selectedModelName == settings.localized("no_model_selected"))

                    if vm.loadedModelName != nil {
                        Button(role: .destructive) {
                            vm.unloadModel()
                            onUnloaded?()
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
        .onAppear {
            syncDraftFromViewModel()
        }
        .onChange(of: vm.selectedModelName) { _, _ in
            syncDraftFromViewModel()
        }
        .onChange(of: draftSystemPrompt) { _, newValue in
            // iOS commits the prompt on Done; the inspector has no Done button,
            // so commit edits as they happen (same per-model persistence path).
            if vm.systemPrompt != newValue {
                vm.systemPrompt = newValue
            }
        }
        .onDisappear {
            // Equivalent of the iOS Done button.
            applyDraftToViewModel()
        }
    }

    private var downloadedModels: [AIModel] {
        var models = ModelData.allModels().filter { model in
            if model.isDependencyOnly { return false }
            if model.name.hasPrefix("Translate Gemma") { return false }
            if !model.isLanguageModel { return false }

            guard ModelData.isModelFullyAvailableLocally(model) else { return false }
            return true
        }

        if let appleModel = macAppleFoundationModelIfAvailable(),
           !models.contains(where: { $0.id == appleModel.id }) {
            models.append(appleModel)
        }

        return models
    }

    private var currentModel: AIModel? {
        if let model = ModelData.allModels().first(where: { $0.name == vm.selectedModelName }) {
            return model
        }
        if let appleModel = macAppleFoundationModelIfAvailable(), appleModel.name == vm.selectedModelName {
            return appleModel
        }
        return nil
    }

    private var modelMaxContextWindow: Double {
        guard let currentModel else { return 4096 }
        let cap = LLMBackend.shared.modelMaxContextWindow(for: currentModel)
        if cap > 0 { return Double(cap) }
        let advertised = currentModel.contextWindowSize > 0 ? currentModel.contextWindowSize : 4096
        return Double(max(1, advertised))
    }

    private func syncDraftFromViewModel() {
        cachedModels = downloadedModels
        if vm.selectedModelName != settings.localized("no_model_selected"),
           !cachedModels.contains(where: { $0.name == vm.selectedModelName }) {
            vm.selectedModelName = cachedModels.first?.name ?? settings.localized("no_model_selected")
        }
        draftContextWindow = min(max(1, vm.contextWindow), modelMaxContextWindow)
        draftTopK = min(max(1, vm.topK), 256)
        draftTopP = min(max(0, vm.topP), 1)
        draftTemperature = min(max(0, vm.temperature), 2)
        draftSystemPrompt = vm.systemPrompt
        loadInitialGpuLayers()
    }

    private func loadInitialGpuLayers() {
        guard let currentModel = currentModel else { return }
        let limit = LLMBackend.shared.modelMaxGpuLayers(for: currentModel)
        gpuLayerLimit = Double(limit)
        let key = "gpu_layers_\(currentModel.id)"
        if UserDefaults.standard.object(forKey: key) != nil {
            let stored = UserDefaults.standard.integer(forKey: key)
            if stored == 999 || stored == 99 || (limit != GGUFLayerLimits.unknown && stored > limit) {
                gpuLayersTemp = Double(limit)
                saveGpuLayers(Double(limit))
            } else {
                gpuLayersTemp = min(max(0, Double(stored)), gpuLayerLimit)
            }
        } else {
            gpuLayersTemp = gpuLayerLimit
            saveGpuLayers(gpuLayerLimit)
        }
    }

    private func saveGpuLayers(_ value: Double) {
        guard let currentModel = currentModel else { return }
        let key = "gpu_layers_\(currentModel.id)"
        let intValue = Int32(min(max(0, value), gpuLayerLimit))
        UserDefaults.standard.set(intValue, forKey: key)
    }

    private func applyDraftToViewModel() {
        let clampedContext = min(max(1, draftContextWindow), modelMaxContextWindow)
        let clampedTopK = min(max(1, draftTopK), 256)
        let clampedTopP = min(max(0, draftTopP), 1)
        let clampedTemperature = min(max(0, draftTemperature), 2)

        draftContextWindow = clampedContext
        draftTopK = clampedTopK
        draftTopP = clampedTopP
        draftTemperature = clampedTemperature

        vm.contextWindow = clampedContext
        // maxTokens is set to full context — no artificial cap on output length
        vm.maxTokens = clampedContext
        vm.topK = clampedTopK
        vm.topP = clampedTopP
        vm.temperature = clampedTemperature
        vm.systemPrompt = draftSystemPrompt
    }

    private func resetAllConfigsToDefaults() {
        draftContextWindow = min(2048, modelMaxContextWindow)
        draftTopK = 64
        draftTopP = 0.95
        draftTemperature = 1.0
        draftSystemPrompt = ""
        applyDraftToViewModel()
    }

    @MainActor
    private func macAppleFoundationModelIfAvailable() -> AIModel? {
        #if canImport(FoundationModels)
        if #available(iOS 26.0, macOS 26.0, *) {
            let model = SystemLanguageModel.default
            guard model.isAvailable else { return nil }

            return AIModel(
                id: "apple.foundation.system",
                name: "Apple Foundation Model",
                description: "On-device Apple Intelligence foundation model.",
                url: "apple://foundation-model",
                category: .text,
                sizeBytes: 0,
                source: "Apple",
                supportsVision: false,
                supportsAudio: false,
                supportsThinking: false,
                supportsGpu: true,
                requirements: ModelRequirements(minRamGB: 8, recommendedRamGB: 8),
                contextWindowSize: max(1, model.contextSize),
                modelFormat: .platform,
                additionalFiles: []
            )
        }
        #endif

        return nil
    }
}

/// Native counterpart of `ConfigSlider` (same snap-to-max and commit-on-release logic).
private struct MacChatConfigSlider: View {
    let title: String
    @Binding var value: Double
    let range: ClosedRange<Double>
    let format: String
    var subtitle: String? = nil
    var step: Double? = nil
    let onCommit: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            LabeledContent {
                Text(String(format: format, value))
                    .monospacedDigit()
            } label: {
                HStack(spacing: 4) {
                    Text(title)
                    if let subtitle = subtitle {
                        Text("(\(subtitle))")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
            }
            if let step {
                ApolloSlider(value: $value, in: range, step: step) { editing in
                    if !editing {
                        // Snap to max if within one step — fixes slider not reaching max
                        // due to step not dividing evenly from lower bound
                        if value >= range.upperBound - step {
                            value = range.upperBound
                        }
                        onCommit()
                    }
                }
                .labelsHidden()
            } else {
                Slider(value: $value, in: range) { editing in
                    if !editing {
                        onCommit()
                    }
                }
                .labelsHidden()
            }
        }
    }
}

/// Native counterpart of `GPULayersSlider`.
private struct MacChatGPULayersSlider: View {
    @Binding var value: Double
    let maxLayers: Double
    let label: String
    let onCommit: () -> Void

    private var displayValue: String {
        "\(Int(value))"
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            LabeledContent(label.replacingOccurrences(of: "%1$d", with: displayValue)) {
                Text(displayValue).monospacedDigit()
            }
            ApolloSlider(
                value: $value,
                in: 0...maxLayers,
                step: 1,
                onEditingChanged: { editing in
                    if !editing {
                        onCommit()
                    }
                }
            )
            .labelsHidden()
        }
    }
}

/// Native counterpart of `ToggleTile`.
private struct MacChatToggleRow: View {
    let title: String
    var subtitle: String? = nil
    @Binding var isOn: Bool
    let icon: String

    var body: some View {
        Toggle(isOn: $isOn) {
            Label {
                VStack(alignment: .leading, spacing: 2) {
                    Text(title)
                    if let subtitle = subtitle {
                        Text(subtitle)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
            } icon: {
                Image(systemName: icon)
                    .foregroundStyle(isOn ? Color.accentColor : Color.secondary)
            }
        }
    }
}
#endif
