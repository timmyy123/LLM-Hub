//
//  SettingsView.swift
//  LLMHub
//
//  Settings and configuration interface
//

import SwiftUI

struct SettingsView: View {
    @EnvironmentObject var inferenceService: InferenceService
    @AppStorage("maxTokens") private var maxTokens: Int = 512
    @AppStorage("temperature") private var temperature: Double = 0.8
    @AppStorage("topK") private var topK: Int = 40
    @AppStorage("topP") private var topP: Double = 0.95
    @State private var showingClearDataAlert = false
    
    var body: some View {
        NavigationView {
            Form {
                Section(header: Text("Generation Parameters")) {
                    VStack(alignment: .leading) {
                        Text("Max Tokens: \(maxTokens)")
                        Slider(value: Binding(
                            get: { Double(maxTokens) },
                            set: { maxTokens = Int($0) }
                        ), in: 128...2048, step: 128)
                    }
                    
                    VStack(alignment: .leading) {
                        Text("Temperature: \(String(format: "%.2f", temperature))")
                        Slider(value: $temperature, in: 0.0...2.0, step: 0.1)
                    }
                    
                    VStack(alignment: .leading) {
                        Text("Top K: \(topK)")
                        Slider(value: Binding(
                            get: { Double(topK) },
                            set: { topK = Int($0) }
                        ), in: 1...100, step: 1)
                    }
                    
                    VStack(alignment: .leading) {
                        Text("Top P: \(String(format: "%.2f", topP))")
                        Slider(value: $topP, in: 0.0...1.0, step: 0.05)
                    }
                }
                
                Section(header: Text("Model Status")) {
                    HStack {
                        Text("Current Model")
                        Spacer()
                        Text(inferenceService.currentModel?.name ?? "None")
                            .foregroundColor(.secondary)
                    }
                    
                    HStack {
                        Text("Model Loaded")
                        Spacer()
                        Image(systemName: inferenceService.isModelLoaded ? "checkmark.circle.fill" : "xmark.circle")
                            .foregroundColor(inferenceService.isModelLoaded ? .green : .red)
                    }
                }
                
                Section(header: Text("Data Management")) {
                    Button(role: .destructive, action: {
                        showingClearDataAlert = true
                    }) {
                        Label("Clear Downloaded Models", systemImage: "trash")
                    }
                    
                    Button(action: {
                        inferenceService.unloadModel()
                    }) {
                        Label("Unload Current Model", systemImage: "eject")
                    }
                    .disabled(!inferenceService.isModelLoaded)
                }
                
                Section(header: Text("About")) {
                    HStack {
                        Text("Version")
                        Spacer()
                        Text("1.0.0")
                            .foregroundColor(.secondary)
                    }
                    
                    HStack {
                        Text("iOS Version")
                        Spacer()
                        Text("Native Swift")
                            .foregroundColor(.secondary)
                    }
                    
                    Link(destination: URL(string: "https://ai.google.dev/edge/mediapipe/solutions/genai/llm_inference/ios")!) {
                        HStack {
                            Text("MediaPipe Documentation")
                            Spacer()
                            Image(systemName: "arrow.up.right.square")
                        }
                    }
                }
            }
            .navigationTitle("Settings")
            .alert("Clear Downloaded Models", isPresented: $showingClearDataAlert) {
                Button("Cancel", role: .cancel) {}
                Button("Clear", role: .destructive) {
                    clearDownloadedModels()
                }
            } message: {
                Text("This will delete all downloaded models. You will need to download them again to use them.")
            }
        }
    }
    
    private func clearDownloadedModels() {
        let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first
        if let modelsDir = documentsPath?.appendingPathComponent("models") {
            try? FileManager.default.removeItem(at: modelsDir)
        }
        inferenceService.unloadModel()
    }
}

#Preview {
    SettingsView()
        .environmentObject(InferenceService())
}
