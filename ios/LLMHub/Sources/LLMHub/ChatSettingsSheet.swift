import SwiftUI
import RunAnywhere
#if canImport(FoundationModels)
import FoundationModels
#endif

struct ChatSettingsSheet: View {
    @ObservedObject var vm: ChatViewModel
    @EnvironmentObject var settings: AppSettings
    @Environment(\.dismiss) var dismiss
    @State private var draftContextWindow: Double = 2048
    @State private var draftTopK: Double = 64
    @State private var draftTopP: Double = 0.95
    @State private var draftTemperature: Double = 1.0
    @State private var draftSystemPrompt: String = ""
    
    var body: some View {
        NavigationView {
            ZStack {
                ApolloLiquidBackground()
                
                ScrollView {
                    VStack(spacing: 20) {
                        // Model Selection Header
                        VStack(spacing: 16) {
                            HStack(alignment: .center, spacing: 14) {
                                Image(systemName: "cpu")
                                    .font(.system(size: 26))
                                    .foregroundColor(ApolloPalette.accentStrong)
                                    .frame(width: 32)
                                
                                VStack(alignment: .leading, spacing: 2) {
                                    Text(settings.localized("select_model_title"))
                                        .font(.headline)
                                        .foregroundColor(.white)
                                    Text(settings.localized("select_model"))
                                        .font(.caption)
                                        .foregroundColor(.white.opacity(0.68))
                                }
                                
                                Spacer()
                                
                                Menu {
                                    Picker("", selection: $vm.selectedModelName) {
                                        if vm.selectedModelName == settings.localized("no_model_selected") {
                                            Text(settings.localized("no_model_selected")).tag(settings.localized("no_model_selected"))
                                        }
                                        ForEach(downloadedModels) { model in
                                            Text(model.name).tag(model.name)
                                        }
                                    }
                                } label: {
                                    HStack(spacing: 6) {
                                        Text(vm.selectedModelName)
                                            .lineLimit(1)
                                            .truncationMode(.tail)
                                        Image(systemName: "chevron.up.chevron.down")
                                            .font(.system(size: 10, weight: .bold))
                                    }
                                    .foregroundColor(ApolloPalette.accentStrong)
                                    // ensure it aligns properly and can shrink
                                    .multilineTextAlignment(.trailing)
                                }
                            }
                            .frame(maxWidth: .infinity)
                        }
                        .padding()
                        .background(.ultraThinMaterial)
                        .clipShape(RoundedRectangle(cornerRadius: 16))
                        .overlay(
                            RoundedRectangle(cornerRadius: 16)
                                .stroke(Color.white.opacity(0.14), lineWidth: 1)
                        )
                        .shadow(color: .black.opacity(0.3), radius: 12, x: 0, y: 8)

                        // Model Configurations
                        VStack(alignment: .leading, spacing: 20) {
                            HStack {
                                Image(systemName: "slider.horizontal.3")
                                    .foregroundColor(ApolloPalette.accentStrong)
                                Text(settings.localized("model_configs_title"))
                                    .font(.headline)
                                    .foregroundColor(.white)
                            }
                            .padding(.bottom, 8)
                            
                            ConfigSlider(title: settings.localized("context_window_size"), value: $draftContextWindow, range: 1...modelMaxContextWindow, format: "%.0f", subtitle: "max \(Int(modelMaxContextWindow))", step: 1, onCommit: applyDraftToViewModel)
                            ConfigSlider(title: settings.localized("top_k"), value: $draftTopK, range: 1...256, format: "%.0f", onCommit: applyDraftToViewModel)
                            ConfigSlider(title: settings.localized("top_p"), value: $draftTopP, range: 0...1, format: "%.2f", onCommit: applyDraftToViewModel)
                            ConfigSlider(title: settings.localized("temperature"), value: $draftTemperature, range: 0...2, format: "%.2f", onCommit: applyDraftToViewModel)

                            if let model = currentModel, model.modelFormat == .gguf {
                                VStack(alignment: .leading, spacing: 8) {
                                    HStack {
                                        Text(settings.localized("gpu_layers_label").replacingOccurrences(of: "%1$d", with: gpuLayersBinding.wrappedValue == 99 ? settings.localized("max") : "\(Int(gpuLayersBinding.wrappedValue))"))
                                            .font(.subheadline)
                                            .foregroundColor(.white)
                                        Spacer()
                                        Text(gpuLayersBinding.wrappedValue == 99 ? settings.localized("max") : "\(Int(gpuLayersBinding.wrappedValue))")
                                            .font(.system(.subheadline, design: .monospaced))
                                            .fontWeight(.bold)
                                            .foregroundColor(.white.opacity(0.92))
                                    }
                                    Slider(value: gpuLayersBinding, in: 0...99, step: 1)
                                        .tint(ApolloPalette.accentStrong)
                                }
                            }

                            HStack {
                                Spacer()
                                Button(settings.localized("reset_to_defaults")) {
                                    resetAllConfigsToDefaults()
                                }
                                .font(.subheadline.weight(.semibold))
                                .foregroundColor(ApolloPalette.accentStrong)
                            }
                            
                        }
                        .padding()
                        .background(.ultraThinMaterial)
                        .clipShape(RoundedRectangle(cornerRadius: 16))
                        .overlay(
                            RoundedRectangle(cornerRadius: 16)
                                .stroke(Color.white.opacity(0.14), lineWidth: 1)
                        )
                        .shadow(color: .black.opacity(0.3), radius: 12, x: 0, y: 8)

                        // System Prompt Section
                        VStack(alignment: .leading, spacing: 12) {
                            HStack {
                                Image(systemName: "text.justify.left")
                                    .foregroundColor(ApolloPalette.accentStrong)
                                Text(settings.localized("model_system_prompt_label"))
                                    .font(.headline)
                                    .foregroundColor(.white)
                            }
                            
                            VStack(alignment: .leading, spacing: 4) {
                                TextEditor(text: $draftSystemPrompt)
                                    .frame(minHeight: 100)
                                    .padding(8)
                                    .scrollContentBackground(.hidden)
                                    .background(Color.white.opacity(0.06))
                                    .foregroundColor(.white)
                                    .font(.body)
                                    .clipShape(RoundedRectangle(cornerRadius: 12))
                                    .overlay(RoundedRectangle(cornerRadius: 12).stroke(Color.white.opacity(0.14), lineWidth: 1))
                                
                                Text(settings.localized("model_system_prompt_hint"))
                                    .font(.caption)
                                    .foregroundColor(.white.opacity(0.42))
                                    .padding(.horizontal, 4)
                            }
                        }
                        .padding()
                        .background(.ultraThinMaterial)
                        .clipShape(RoundedRectangle(cornerRadius: 16))
                        .overlay(
                            RoundedRectangle(cornerRadius: 16)
                                .stroke(Color.white.opacity(0.14), lineWidth: 1)
                        )
                        .shadow(color: .black.opacity(0.3), radius: 12, x: 0, y: 8)

                        // Modality Toggle Tiles
                        let visionAvailable = currentModel.map { $0.supportsVision && LLMBackend.shared.isVisionProjectorAvailable(for: $0) } ?? false
                        if let model = currentModel, (visionAvailable || model.supportsAudio || model.supportsThinking) {
                            VStack(alignment: .leading, spacing: 16) {
                                HStack {
                                    Image(systemName: "sparkles")
                                        .foregroundColor(ApolloPalette.accentStrong)
                                    Text(settings.localized("modality_options"))
                                        .font(.headline)
                                        .foregroundColor(.white)
                                }
                                .padding(.bottom, 4)
                                
                                if visionAvailable {
                                    ToggleTile(title: settings.localized("enable_vision"), isOn: $vm.enableVision, icon: "eye.fill")
                                }
                                if model.supportsAudio {
                                    ToggleTile(title: settings.localized("enable_audio"), isOn: $vm.enableAudio, icon: "mic.fill")
                                }
                                if model.supportsThinking {
                                    ToggleTile(title: settings.localized("enable_thinking"), isOn: $vm.enableThinking, icon: "brain")
                                }
                                if model.name.contains("Gemma 4") && model.modelFormat == .litertlm {
                                    ToggleTile(
                                        title: settings.localized("agent_tools_label"),
                                        subtitle: settings.localized("agent_tools_description"),
                                        isOn: $vm.enableAgentTools,
                                        icon: "square.stack.3d.up.fill"
                                    )
                                }
                            }
                            .padding()
                            .background(.ultraThinMaterial)
                            .clipShape(RoundedRectangle(cornerRadius: 16))
                            .overlay(
                                RoundedRectangle(cornerRadius: 16)
                                    .stroke(Color.white.opacity(0.14), lineWidth: 1)
                            )
                            .shadow(color: .black.opacity(0.3), radius: 12, x: 0, y: 8)
                        }
                        
                        // Action Buttons
                        VStack(spacing: 12) {
                            Button(action: {
                                applyDraftToViewModel()
                                Task {
                                    await vm.loadModelIfNecessary(force: true)
                                    dismiss()
                                }
                            }) {
                                HStack {
                                    if vm.isBackendLoading {
                                        ProgressView().padding(.trailing, 8)
                                    } else {
                                        Image(systemName: "arrow.clockwise.circle.fill")
                                            .padding(.trailing, 4)
                                    }
                                    Text(settings.localized("load_model"))
                                        .fontWeight(.bold)
                                }
                                .frame(maxWidth: .infinity)
                                .padding()
                                .foregroundColor(.white)
                                .contentShape(Rectangle())
                            }
                            .liquidGlassPrimaryButton(cornerRadius: 14)
                            .disabled(vm.isBackendLoading || vm.selectedModelName == settings.localized("no_model_selected"))
                            
                            if vm.loadedModelName != nil {
                                Button(action: {
                                    vm.unloadModel()
                                    dismiss()
                                }) {
                                    Text(settings.localized("unload_model"))
                                        .fontWeight(.medium)
                                        .foregroundColor(ApolloPalette.destructive.opacity(0.98))
                                        .frame(maxWidth: .infinity)
                                        .padding()
                                        .background(.ultraThinMaterial)
                                        .clipShape(RoundedRectangle(cornerRadius: 14))
                                        .overlay(
                                            RoundedRectangle(cornerRadius: 14)
                                                .stroke(ApolloPalette.destructive.opacity(0.42), lineWidth: 1)
                                        )
                                }
                            }
                        }
                    }
                    .padding()
                }
            }
            .navigationTitle(settings.localized("feature_settings_title"))
            .navigationBarTitleDisplayMode(.inline)
            .toolbarBackground(.hidden, for: .navigationBar)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button(settings.localized("done")) {
                        applyDraftToViewModel()
                        dismiss()
                    }
                }
            }
            .onAppear {
                syncDraftFromViewModel()
            }
            .onChange(of: vm.selectedModelName) { _, _ in
                syncDraftFromViewModel()
            }
        }
    }
    
    private var downloadedModels: [AIModel] {
        let legacyModelsDir: URL? = {
            guard let documentsDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else { return nil }
            return documentsDir.appendingPathComponent("models")
        }()

        var models = ModelData.allModels().filter { model in
            if model.isDependencyOnly { return false }
            if model.name.hasPrefix("Translate Gemma") { return false }
            if model.category == .embedding || model.category == .imageGeneration || model.category == .videoGeneration || model.category == .imageUpscale || model.category == .asr { return false }

            guard ModelData.isModelFullyAvailableLocally(model) else { return false }
            return true
        }

        if let appleModel = appleFoundationModelIfAvailable(),
           !models.contains(where: { $0.id == appleModel.id }) {
            models.append(appleModel)
        }

        return models
    }
    
    private var currentModel: AIModel? {
        if let model = ModelData.allModels().first(where: { $0.name == vm.selectedModelName }) {
            return model
        }
        if let appleModel = appleFoundationModelIfAvailable(), appleModel.name == vm.selectedModelName {
            return appleModel
        }
        return nil
    }

    private var modelMaxContextWindow: Double {
        guard let currentModel else { return 4096 }
        let advertised = currentModel.contextWindowSize > 0 ? currentModel.contextWindowSize : 4096
        return Double(max(1, advertised))
    }


    private var contextWindowStep: Double {
        let maxWindow = max(1, Int(modelMaxContextWindow))
        let raw = max(1, maxWindow / 1024)
        return Double(raw)
    }

    private var gpuLayersBinding: Binding<Double> {
        Binding<Double>(
            get: {
                guard let currentModel = currentModel else { return 99 }
                let key = "gpu_layers_\(currentModel.id)"
                if UserDefaults.standard.object(forKey: key) != nil {
                    return Double(UserDefaults.standard.integer(forKey: key))
                }
                return 99 // Default to max
            },
            set: { newValue in
                guard let currentModel = currentModel else { return }
                let key = "gpu_layers_\(currentModel.id)"
                let intValue = Int32(newValue)
                UserDefaults.standard.set(intValue, forKey: key)
                
                // Synchronize to C++ registry immediately
                let targetValue: Int32 = (intValue == 99) ? 999 : intValue
                CppBridge.ModelRegistry.shared.setGpuLayers(modelId: currentModel.id, gpuLayers: targetValue)
                if let folderURL = try? SimplifiedFileManager.shared.getModelFolderURL(modelId: currentModel.id, framework: currentModel.inferenceFramework),
                   let ggufFile = LLMBackend.shared.listGGUFFiles(in: folderURL).first(where: { !$0.lastPathComponent.lowercased().contains("mmproj") }) {
                    CppBridge.ModelRegistry.shared.setGpuLayers(modelId: ggufFile.path, gpuLayers: targetValue)
                }
            }
        )
    }

    private func syncDraftFromViewModel() {
        draftContextWindow = min(max(1, vm.contextWindow), modelMaxContextWindow)
        draftTopK = min(max(1, vm.topK), 256)
        draftTopP = min(max(0, vm.topP), 1)
        draftTemperature = min(max(0, vm.temperature), 2)
        draftSystemPrompt = vm.systemPrompt
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

    private var draftedModels: [AIModel] {
        downloadedModels
    }

    @MainActor
    private func appleFoundationModelIfAvailable() -> AIModel? {
        #if canImport(FoundationModels)
        if #available(iOS 26.0, *) {
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
                modelFormat: .gguf,
                additionalFiles: []
            )
        }
        #endif

        return nil
    }
}

struct ConfigSlider: View {
    let title: String
    @Binding var value: Double
    let range: ClosedRange<Double>
    let format: String
    var subtitle: String? = nil
    var step: Double? = nil
    let onCommit: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(title)
                    .font(.subheadline)
                    .foregroundColor(.white)
                if let subtitle = subtitle {
                    Text("(\(subtitle))")
                        .font(.caption2)
                        .foregroundColor(.white.opacity(0.58))
                }
                Spacer()
                Text(String(format: format, value))
                    .font(.system(.subheadline, design: .monospaced))
                    .fontWeight(.bold)
                    .foregroundColor(.white.opacity(0.92))
            }
            Group {
                if let step {
                    Slider(value: $value, in: range, step: step) { editing in
                        if !editing {
                            // Snap to max if within one step — fixes slider not reaching max
                            // due to step not dividing evenly from lower bound
                            if value >= range.upperBound - step {
                                value = range.upperBound
                            }
                            onCommit()
                        }
                    }
                } else {
                    Slider(value: $value, in: range) { editing in
                        if !editing {
                            onCommit()
                        }
                    }
                }
            }
            .tint(ApolloPalette.accentStrong)
        }
    }
}

struct ToggleTile: View {
    let title: String
    var subtitle: String? = nil
    @Binding var isOn: Bool
    let icon: String
    
    var body: some View {
        HStack(alignment: subtitle == nil ? .center : .top) {
            Image(systemName: icon)
                .foregroundColor(isOn ? ApolloPalette.accentStrong : .white.opacity(0.58))
                .frame(width: 24)
                .padding(.top, subtitle == nil ? 0 : 2)
            VStack(alignment: .leading, spacing: 4) {
                Text(title)
                    .font(.subheadline)
                    .foregroundColor(.white)
                if let subtitle = subtitle {
                    Text(subtitle)
                        .font(.caption2)
                        .foregroundColor(.white.opacity(0.58))
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            Spacer()
            Toggle("", isOn: $isOn)
                .labelsHidden()
                .tint(ApolloPalette.accentStrong)
                .padding(.top, subtitle == nil ? 0 : 2)
        }
    }
}


