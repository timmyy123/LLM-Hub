import SwiftUI

struct ChatSettingsSheet: View {
    @ObservedObject var vm: ChatViewModel
    @EnvironmentObject var settings: AppSettings
    @Environment(\.dismiss) var dismiss
    
    var body: some View {
        NavigationView {
            ZStack {
                Color(uiColor: .systemGroupedBackground).ignoresSafeArea()
                
                ScrollView {
                    VStack(spacing: 20) {
                        // Model Selection Header
                        VStack(spacing: 16) {
                            HStack(alignment: .center, spacing: 14) {
                                Image(systemName: "chip.fill")
                                    .font(.system(size: 26))
                                    .foregroundColor(.indigo)
                                    .frame(width: 32)
                                
                                VStack(alignment: .leading, spacing: 2) {
                                    Text(settings.localized("select_model_title"))
                                        .font(.headline)
                                    Text(settings.localized("select_model"))
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                }
                                .padding(.vertical, 4)
                                
                                Spacer()
                                
                                Picker("", selection: $vm.selectedModelName) {
                                    if vm.selectedModelName == settings.localized("no_model_selected") {
                                        Text(settings.localized("no_model_selected")).tag(settings.localized("no_model_selected"))
                                    }
                                    ForEach(downloadedModels) { model in
                                        Text(model.name).tag(model.name)
                                    }
                                }
                                .pickerStyle(.menu)
                                .accentColor(.indigo)
                                .labelsHidden()
                            }
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 10)
                            
                            if currentModel != nil {
                                Divider()
                                HStack {
                                    Label(settings.localized("currently_loaded"), systemImage: "bolt.circle.fill")
                                        .font(.subheadline)
                                        .foregroundColor(.secondary)
                                    Spacer()
                                    if vm.selectedModelName == vm.loadedModelName {
                                        Text("✅ " + settings.localized("ready_to_chat"))
                                            .font(.caption.bold())
                                            .foregroundColor(.green)
                                    }
                                }
                            }
                        }
                        .padding()
                        .background(Color(uiColor: .secondarySystemGroupedBackground))
                        .clipShape(RoundedRectangle(cornerRadius: 16))
                        .shadow(color: .black.opacity(0.05), radius: 10, x: 0, y: 4)

                        // Model Configurations
                        VStack(alignment: .leading, spacing: 20) {
                            HStack {
                                Image(systemName: "slider.horizontal.3")
                                    .foregroundColor(.indigo)
                                Text(settings.localized("model_configs_title"))
                                    .font(.headline)
                            }
                            .padding(.bottom, 8)
                            
                            configSlider(title: settings.localized("max_tokens"), value: $vm.maxTokens, range: 1...4096, format: "%.0f")
                            configSlider(title: settings.localized("top_k"), value: $vm.topK, range: 1...256, format: "%.0f")
                            configSlider(title: settings.localized("top_p"), value: $vm.topP, range: 0...1, format: "%.2f")
                            configSlider(title: settings.localized("temperature"), value: $vm.temperature, range: 0...2, format: "%.2f")
                            
                        }
                        .padding()
                        .background(Color(uiColor: .secondarySystemGroupedBackground))
                        .clipShape(RoundedRectangle(cornerRadius: 16))
                        .shadow(color: .black.opacity(0.05), radius: 10, x: 0, y: 4)

                        // Modality Toggle Tiles
                        if let model = currentModel, (model.supportsVision || model.supportsAudio || model.supportsThinking) {
                            VStack(alignment: .leading, spacing: 16) {
                                HStack {
                                    Image(systemName: "sparkles")
                                        .foregroundColor(.indigo)
                                    Text(settings.localized("modality_options"))
                                        .font(.headline)
                                }
                                .padding(.bottom, 4)
                                
                                if model.supportsVision {
                                    ToggleTile(title: settings.localized("enable_vision"), isOn: $vm.enableVision, icon: "eye.fill")
                                }
                                if model.supportsAudio {
                                    ToggleTile(title: settings.localized("enable_audio"), isOn: $vm.enableAudio, icon: "mic.fill")
                                }
                                if model.supportsThinking {
                                    ToggleTile(title: settings.localized("enable_thinking"), isOn: $vm.enableThinking, icon: "brain")
                                }
                            }
                            .padding()
                            .background(Color(uiColor: .secondarySystemGroupedBackground))
                            .clipShape(RoundedRectangle(cornerRadius: 16))
                            .shadow(color: .black.opacity(0.05), radius: 10, x: 0, y: 4)
                        }
                        
                        // Action Buttons
                        VStack(spacing: 12) {
                            Button(action: {
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
                                .background(vm.selectedModelName == settings.localized("no_model_selected") ? Color.gray : Color.indigo)
                                .foregroundColor(.white)
                                .clipShape(RoundedRectangle(cornerRadius: 14))
                            }
                            .disabled(vm.isBackendLoading || vm.selectedModelName == settings.localized("no_model_selected"))
                            
                            if vm.loadedModelName != nil {
                                Button(action: {
                                    vm.unloadModel()
                                    dismiss()
                                }) {
                                    Text(settings.localized("unload_model"))
                                        .fontWeight(.medium)
                                        .foregroundColor(.red)
                                        .frame(maxWidth: .infinity)
                                        .padding()
                                        .background(Color.red.opacity(0.1))
                                        .clipShape(RoundedRectangle(cornerRadius: 14))
                                }
                            }
                        }
                    }
                    .padding()
                }
            }
            .navigationTitle(settings.localized("feature_settings_title"))
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button(settings.localized("done")) { dismiss() }
                }
            }
        }
    }
    
    @ViewBuilder
    private func configSlider(title: String, value: Binding<Double>, range: ClosedRange<Double>, format: String, subtitle: String? = nil) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(title)
                    .font(.subheadline)
                    .foregroundColor(.primary)
                if let subtitle = subtitle {
                    Text("(\(subtitle))")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                }
                Spacer()
                Text(String(format: format, value.wrappedValue))
                    .font(.system(.subheadline, design: .monospaced))
                    .fontWeight(.bold)
                    .foregroundColor(.indigo)
            }
            Slider(value: value, in: range)
                .accentColor(.indigo)
        }
    }
    
    private var downloadedModels: [AIModel] {
        guard let documentsDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else { return [] }
        let modelsDir = documentsDir.appendingPathComponent("models")
        return ModelData.models.filter { model in
            let weightsFile = modelsDir.appendingPathComponent(model.id).appendingPathComponent("model.safetensors")
            return FileManager.default.fileExists(atPath: weightsFile.path)
        }
    }
    
    private var currentModel: AIModel? {
        ModelData.models.first(where: { $0.name == vm.selectedModelName })
    }
}

struct ToggleTile: View {
    let title: String
    @Binding var isOn: Bool
    let icon: String
    
    var body: some View {
        HStack {
            Image(systemName: icon)
                .foregroundColor(.indigo.opacity(0.6))
                .frame(width: 24)
            Text(title)
                .font(.subheadline)
            Spacer()
            Toggle("", isOn: $isOn)
                .labelsHidden()
                .tint(.green)
        }
    }
}
