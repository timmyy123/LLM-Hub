import Foundation
import MLX
import MLXNN
import MLXRandom
import MLXLLM
import MLXLMCommon
import Hub
import Tokenizers

@MainActor
class LLMBackend: ObservableObject {
    static let shared = LLMBackend()
    
    @Published var isLoaded: Bool = false
    @Published var currentlyLoadedModel: String? = nil
    @Published var isBackendLoading: Bool = false
    
    // Generation Parameters
    var maxTokens: Int = 2048
    var topK: Int = 64
    var topP: Float = 0.95
    var temperature: Float = 1.0
    var enableVision: Bool = true
    var enableAudio: Bool = true
    var enableThinking: Bool = true
    
    // MLX Model Container
    private var modelContainer: ModelContainer?
    
    private init() {}
    
    func loadModel(_ model: AIModel) async throws {
        isBackendLoading = true
        defer { isBackendLoading = false }
        
        let fileManager = FileManager.default
        let documentsDir = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first!
        let modelsDir = documentsDir.appendingPathComponent("models")
        let modelDir = modelsDir.appendingPathComponent(model.id)
        
        // 1. Verify files exist
        for file in model.files {
            let path = modelDir.appendingPathComponent(file).path
            if !fileManager.fileExists(atPath: path) {
                throw NSError(domain: "LLMBackend", code: 404, userInfo: [NSLocalizedDescriptionKey: "Missing model file: \(file)"])
            }
        }
        
        // 2. Load with MLX LLM
        let modelConfiguration = ModelConfiguration(directory: modelDir)
        let hub = HubApi()
        let factory = LLMModelFactory.shared
        self.modelContainer = try await factory.loadContainer(hub: hub, configuration: modelConfiguration)
        
        isLoaded = true
        currentlyLoadedModel = model.name
    }
    
    func unloadModel() {
        modelContainer = nil
        isLoaded = false
        currentlyLoadedModel = nil
    }
    
    func generate(prompt: String, onUpdate: @escaping (String, Int, Double) -> Void) async throws {
        guard let container = self.modelContainer else { 
            throw NSError(domain: "LLMBackend", code: 403, userInfo: [NSLocalizedDescriptionKey: "Model not loaded"]) 
        }
        
        let input = LMInput(text: prompt)
        let params = GenerateParameters(maxTokens: self.maxTokens, temperature: self.temperature, topP: self.topP)
        
        let startTime = Date()
        var currentOutput = ""
        var tokens = 0
        
        for await item in try MLXLMCommon.generate(input: input, parameters: params, context: container.context) {
            if Task.isCancelled { break }
            switch item {
            case .chunk(let text):
                currentOutput += text
                tokens += 1
                let elapsed = Date().timeIntervalSince(startTime)
                let tps = elapsed > 0 ? Double(tokens) / elapsed : 0
                onUpdate(currentOutput, tokens, tps)
            case .info, .toolCall:
                break
            }
        }
    }
}
