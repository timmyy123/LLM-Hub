import SwiftUI
#if canImport(FoundationModels)
import FoundationModels
#endif

/// Reads only GGUF header metadata, without mapping or loading model tensors.
enum GGUFLayerLimits {
    static let unknown = 999

    static func read(from url: URL) -> Int? {
        readInteger(from: url, suffix: "block_count", offset: 1)
    }

    static func readContextLength(from url: URL) -> Int? {
        readInteger(from: url, suffix: "context_length", offset: 0)
    }

    private static func readInteger(from url: URL, suffix: String, offset: Int) -> Int? {
        guard let handle = try? FileHandle(forReadingFrom: url),
              let size = (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? NSNumber)?.uint64Value else {
            return nil
        }
        defer { try? handle.close() }
        do {
            let reader = Reader(handle: handle, size: size)
            guard try reader.bytes(4) == Data("GGUF".utf8),
                  (2...3).contains(try reader.number(4)) else { return nil }
            _ = try reader.number(8) // tensor count
            let keyCount = try reader.number(8)
            guard keyCount <= 1_000_000 else { return nil }
            var architecture: String?
            var metadataValues: [String: UInt64] = [:]
            for _ in 0..<keyCount {
                let key = try reader.string()
                let type = try reader.number(4)
                if key == "general.architecture", type == 8 {
                    architecture = try reader.string()
                } else if key.hasSuffix(".\(suffix)"), type == 4 || type == 10 {
                    metadataValues[key] = try reader.number(type == 4 ? 4 : 8)
                } else {
                    try reader.skipValue(type)
                }
                if let architecture,
                   let count = metadataValues["\(architecture).\(suffix)"],
                   count > 0, count <= UInt64(Int.max - offset) {
                    // Tokenizer arrays can occupy megabytes of metadata after this point.
                    // The slider needs only this model limit, not the rest of the header.
                    return Int(count) + offset
                }
            }
            guard let architecture,
                  let count = metadataValues["\(architecture).\(suffix)"],
                  count > 0, count <= UInt64(Int.max - offset) else { return nil }
            // Full layer offload includes the output layer; context length needs no offset.
            return Int(count) + offset
        } catch {
            return nil
        }
    }

    private struct Reader {
        let handle: FileHandle
        let size: UInt64

        func bytes(_ count: Int) throws -> Data {
            let data = try handle.read(upToCount: count) ?? Data()
            guard data.count == count else { throw CocoaError(.fileReadCorruptFile) }
            return data
        }

        func number(_ width: Int) throws -> UInt64 {
            let data = try bytes(width)
            return data.enumerated().reduce(UInt64(0)) { value, byte in
                value | (UInt64(byte.element) << (byte.offset * 8))
            }
        }

        func skip(_ count: UInt64) throws {
            let offset = handle.offsetInFile
            guard offset <= size, count <= size - offset else { throw CocoaError(.fileReadCorruptFile) }
            try handle.seek(toOffset: offset + count)
        }

        func string() throws -> String {
            let length = try number(8)
            guard length <= 1_048_576,
                  let value = String(data: try bytes(Int(length)), encoding: .utf8) else {
                throw CocoaError(.fileReadCorruptFile)
            }
            return value
        }

        func skipValue(_ type: UInt64) throws {
            switch type {
            case 0, 1, 7: try skip(1)
            case 2, 3: try skip(2)
            case 4, 5, 6: try skip(4)
            case 8: try skip(number(8))
            case 10, 11, 12: try skip(8)
            case 9:
                let itemType = try number(4)
                let count = try number(8)
                let width: UInt64
                switch itemType {
                case 0, 1, 7: width = 1
                case 2, 3: width = 2
                case 4, 5, 6: width = 4
                case 10, 11, 12: width = 8
                case 8: width = 0
                default: throw CocoaError(.fileReadCorruptFile)
                }
                if width > 0 {
                    guard count <= (size - handle.offsetInFile) / width else { throw CocoaError(.fileReadCorruptFile) }
                    try skip(count * width)
                } else {
                    guard count <= 1_000_000 else { throw CocoaError(.fileReadCorruptFile) }
                    for _ in 0..<count { try skip(number(8)) }
                }
            default: throw CocoaError(.fileReadCorruptFile)
            }
        }
    }
}

struct ChatSettingsSheet: View {
    @ObservedObject var vm: ChatViewModel
    @EnvironmentObject var settings: AppSettings
    @Environment(\.dismiss) var dismiss
    @State private var draftContextWindow: Double = 2048
    @State private var draftTopK: Double = 64
    @State private var draftTopP: Double = 0.95
    @State private var draftTemperature: Double = 1.0
    @State private var draftSystemPrompt: String = ""
    @State private var gpuLayersTemp: Double = 999
    @State private var gpuLayerLimit: Double = 999
    @State private var cachedModels: [AIModel] = []
    
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
                                        ForEach(cachedModels) { model in
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
                                GPULayersSlider(
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
                .apolloTopScrollEdgeFade()
            }
            .navigationTitle(settings.localized("feature_settings_title"))
            .navigationBarTitleDisplayMode(.inline)
            .apolloNavigationBackground()
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
            if !model.isLanguageModel { return false }

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
        if currentModel.modelFormat == .gguf {
            return Double(LLMBackend.shared.modelMaxContextWindow(for: currentModel))
        }
        let advertised = currentModel.contextWindowSize > 0 ? currentModel.contextWindowSize : 4096
        return Double(max(1, advertised))
    }


    private var contextWindowStep: Double {
        let maxWindow = max(1, Int(modelMaxContextWindow))
        let raw = max(1, maxWindow / 1024)
        return Double(raw)
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
        // Read the small GGUF metadata header before presenting the slider. Updating its
        // range asynchronously made SwiftUI briefly draw the thumb at the left edge.
        let limit = LLMBackend.shared.ggufFileURL(for: currentModel)
            .flatMap { GGUFLayerLimits.read(from: $0) } ?? GGUFLayerLimits.unknown
        gpuLayerLimit = Double(limit)
        let key = "gpu_layers_\(currentModel.id)"
        if UserDefaults.standard.object(forKey: key) != nil {
            let stored = UserDefaults.standard.integer(forKey: key)
            gpuLayersTemp = min(max(0, Double(stored == 99 ? 999 : stored)), gpuLayerLimit)
        } else {
            gpuLayersTemp = gpuLayerLimit
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

    private var draftedModels: [AIModel] {
        cachedModels
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
                modelFormat: .platform,
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

struct GPULayersSlider: View {
    @Binding var value: Double
    let maxLayers: Double
    let label: String
    let onCommit: () -> Void

    private var displayValue: String {
        "\(Int(value))"
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(label.replacingOccurrences(of: "%1$d", with: displayValue))
                    .font(.subheadline)
                    .foregroundColor(.white)
                Spacer()
                Text(displayValue)
                    .font(.system(.subheadline, design: .monospaced))
                    .fontWeight(.bold)
                    .foregroundColor(.white.opacity(0.92))
            }
            Slider(
                value: $value,
                in: 0...maxLayers,
                step: 1,
                onEditingChanged: { editing in
                    if !editing {
                        onCommit()
                    }
                }
            )
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
