//
//  ModelSelectorView.swift
//  LLMHub
//
//  Model selection and download interface
//

import SwiftUI

struct ModelSelectorView: View {
    @EnvironmentObject var inferenceService: InferenceService
    @State private var models: [LLMModel] = LLMModel.sampleModels
    @State private var selectedModel: LLMModel?
    @State private var isDownloading = false
    @State private var downloadProgress: Float = 0.0
    @State private var errorMessage: String?
    
    var body: some View {
        NavigationView {
            List {
                Section(header: Text("Available Models")) {
                    ForEach(models) { model in
                        ModelRow(
                            model: model,
                            isDownloaded: inferenceService.isModelDownloaded(model),
                            isLoaded: inferenceService.currentModel?.id == model.id,
                            onLoad: {
                                Task {
                                    do {
                                        try await inferenceService.loadModel(model)
                                    } catch {
                                        errorMessage = error.localizedDescription
                                    }
                                }
                            },
                            onDownload: {
                                Task {
                                    do {
                                        isDownloading = true
                                        try await inferenceService.downloadModel(model) { progress in
                                            downloadProgress = progress
                                        }
                                        isDownloading = false
                                    } catch {
                                        errorMessage = error.localizedDescription
                                        isDownloading = false
                                    }
                                }
                            }
                        )
                    }
                }
                
                Section(header: Text("Model Information")) {
                    if let model = inferenceService.currentModel {
                        VStack(alignment: .leading, spacing: 8) {
                            InfoRow(label: "Name", value: model.name)
                            InfoRow(label: "Description", value: model.description)
                            InfoRow(label: "Category", value: model.category)
                            InfoRow(label: "Source", value: model.source)
                            InfoRow(label: "Size", value: ByteCountFormatter.string(fromByteCount: model.sizeBytes, countStyle: .file))
                            InfoRow(label: "Context Window", value: "\(model.contextWindowSize) tokens")
                            InfoRow(label: "Vision Support", value: model.supportsVision ? "Yes" : "No")
                            InfoRow(label: "Audio Support", value: model.supportsAudio ? "Yes" : "No")
                            InfoRow(label: "GPU Support", value: model.supportsGpu ? "Yes" : "No")
                        }
                    } else {
                        Text("No model loaded")
                            .foregroundColor(.secondary)
                    }
                }
            }
            .navigationTitle("Models")
            .alert("Error", isPresented: .constant(errorMessage != nil)) {
                Button("OK") {
                    errorMessage = nil
                }
            } message: {
                if let error = errorMessage {
                    Text(error)
                }
            }
        }
    }
}

struct ModelRow: View {
    let model: LLMModel
    let isDownloaded: Bool
    let isLoaded: Bool
    let onLoad: () -> Void
    let onDownload: () -> Void
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text(model.name)
                        .font(.headline)
                    Text(model.description)
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .lineLimit(2)
                }
                
                Spacer()
                
                if isLoaded {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundColor(.green)
                        .font(.title3)
                }
            }
            
            HStack {
                Text(ByteCountFormatter.string(fromByteCount: model.sizeBytes, countStyle: .file))
                    .font(.caption)
                    .foregroundColor(.secondary)
                
                Spacer()
                
                if isDownloaded {
                    Button(action: onLoad) {
                        Label("Load", systemImage: "play.circle")
                            .font(.caption)
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(isLoaded)
                } else {
                    Button(action: onDownload) {
                        Label("Download", systemImage: "arrow.down.circle")
                            .font(.caption)
                    }
                    .buttonStyle(.bordered)
                }
            }
        }
        .padding(.vertical, 4)
    }
}

struct InfoRow: View {
    let label: String
    let value: String
    
    var body: some View {
        HStack {
            Text(label)
                .foregroundColor(.secondary)
            Spacer()
            Text(value)
                .multilineTextAlignment(.trailing)
        }
        .font(.subheadline)
    }
}

#Preview {
    ModelSelectorView()
        .environmentObject(InferenceService())
}
