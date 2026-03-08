import Foundation

public enum ModelFormat: String, Codable, Sendable {
    case task
    case litertlm
    case gguf
    case onnx
    case mlx // Added for MLX specific models
}

public enum DownloadState: Equatable, Sendable {
    case notDownloaded
    case downloading(progress: Double, downloaded: String, speed: String)
    case paused
    case downloaded
    case error(message: String)
}

public enum ModelCategory: String, Codable, CaseIterable, Sendable {
    case text = "Text Models"
    case multimodal = "Multimodal Models"
    case embedding = "Embedding Models"
    case imageGeneration = "Image Generation"

    public var icon: String {
        switch self {
        case .text:      return "text.bubble.fill"
        case .multimodal: return "eye.fill"
        case .embedding: return "link.circle.fill"
        case .imageGeneration: return "photo.fill"
        }
    }

    public var titleKey: String {
        switch self {
        case .text: return "text_models"
        case .multimodal: return "vision_models"
        case .embedding: return "embedding_models"
        case .imageGeneration: return "image_generation_models"
        }
    }

    public var descriptionKey: String {
        switch self {
        case .text: return "text_models_description"
        case .multimodal: return "vision_models_description"
        case .embedding: return "embedding_models_description"
        case .imageGeneration: return "image_generation_models_description"
        }
    }
}

public struct ModelRequirements: Codable, Sendable {
    public let minRamGB: Int
    public let recommendedRamGB: Int
}

public struct AIModel: Identifiable, Codable, Sendable {
    public let id: String
    public let name: String
    public let description: String
    public let repoId: String // Hugging Face repo ID (e.g. mlx-community/gemma-3n-E2B-it-4bit)
    public let category: ModelCategory
    public let sizeBytes: Int64
    public let source: String
    public let supportsVision: Bool
    public let supportsAudio: Bool
    public let supportsGpu: Bool
    public let requirements: ModelRequirements
    public let contextWindowSize: Int
    public let modelFormat: ModelFormat
    public let files: [String] // List of files in the repo needed for inference
    
    public init(
        id: String? = nil,
        name: String,
        description: String,
        repoId: String,
        category: ModelCategory,
        sizeBytes: Int64,
        source: String,
        supportsVision: Bool = false,
        supportsAudio: Bool = false,
        supportsGpu: Bool = true,
        requirements: ModelRequirements,
        contextWindowSize: Int = 2048,
        modelFormat: ModelFormat = .mlx,
        files: [String] = []
    ) {
        self.id = id ?? name.lowercased().replacingOccurrences(of: " ", with: "_")
        self.name = name
        self.description = description
        self.repoId = repoId
        self.category = category
        self.sizeBytes = sizeBytes
        self.source = source
        self.supportsVision = supportsVision
        self.supportsAudio = supportsAudio
        self.supportsGpu = supportsGpu
        self.requirements = requirements
        self.contextWindowSize = contextWindowSize
        self.modelFormat = modelFormat
        self.files = files
    }

    public var sizeLabel: String {
        let formatter = ByteCountFormatter()
        formatter.allowedUnits = [.useGB, .useMB]
        formatter.countStyle = .file
        return formatter.string(fromByteCount: sizeBytes)
    }

    public var ramLabel: String {
        "\(requirements.minRamGB)GB RAM"
    }
}

public struct ModelData {
    public static let models: [AIModel] = [
        // --- Multimodal Models (User's priority) ---
        AIModel(
            name: "Gemma-3n E2B (4-bit)",
            description: "Google's powerful multimodal model for text and vision. Optimized for efficiency. (MLX 4-bit)",
            repoId: "mlx-community/gemma-3n-E2B-it-4bit",
            category: .multimodal,
            sizeBytes: 4460000000,
            source: "Google via MLX Community",
            supportsVision: true,
            supportsAudio: false,
            requirements: ModelRequirements(minRamGB: 8, recommendedRamGB: 12),
            contextWindowSize: 4096,
            modelFormat: .mlx,
            files: [
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "model.safetensors",
                "chat_template.jinja",
                "special_tokens_map.json",
                "preprocessor_config.json",
                "processor_config.json"
            ]
        ),
        AIModel(
            name: "Gemma-3n E4B (4-bit)",
            description: "Larger multimodal model from Google. High performance for vision and text tasks.",
            repoId: "mlx-community/gemma-3n-E4B-it-4bit",
            category: .multimodal,
            sizeBytes: 5860000000, 
            source: "Google via MLX Community",
            supportsVision: true,
            supportsAudio: true,
            requirements: ModelRequirements(minRamGB: 12, recommendedRamGB: 16),
            contextWindowSize: 4096,
            modelFormat: .mlx,
            files: [
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors",
                "model.safetensors.index.json",
                "chat_template.jinja",
                "special_tokens_map.json",
                "preprocessor_config.json",
                "processor_config.json"
            ]
        ),



        // --- Text Models ---
        AIModel(
            name: "Gemma-3 1B (4-bit)",
            description: "Ultra-compact and efficient language model from Google.",
            repoId: "mlx-community/gemma-3-1b-it-4bit",
            category: .text,
            sizeBytes: 733000000,
            source: "Google via MLX Community",
            requirements: ModelRequirements(minRamGB: 2, recommendedRamGB: 4),
            contextWindowSize: 2048,
            modelFormat: .mlx,
            files: [
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "model.safetensors",
                "chat_template.jinja"
            ]
        ),
        AIModel(
            name: "Phi-4 Mini (4-bit)",
            description: "Microsoft's efficient small language model with high performance.",
            repoId: "mlx-community/phi-4-mini-instruct-4bit",
            category: .text,
            sizeBytes: 2160000000,
            source: "Microsoft via MLX Community",
            requirements: ModelRequirements(minRamGB: 4, recommendedRamGB: 6),
            contextWindowSize: 4096,
            modelFormat: .mlx,
            files: [
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "model.safetensors",
                "generation_config.json"
            ]
        ),
        AIModel(
            name: "Llama-3.2 3B (4-bit)",
            description: "Meta's most capable small model for mobile devices.",
            repoId: "mlx-community/Llama-3.2-3B-Instruct-4bit",
            category: .text,
            sizeBytes: 1810000000,
            source: "Meta via MLX Community",
            requirements: ModelRequirements(minRamGB: 4, recommendedRamGB: 8),
            contextWindowSize: 4096,
            modelFormat: .mlx,
            files: [
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "model.safetensors"
            ]
        )

    ]
}

