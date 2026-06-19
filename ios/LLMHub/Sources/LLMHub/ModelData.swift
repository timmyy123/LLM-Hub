import Foundation
import RunAnywhere
import ModelZoo

public enum ModelFormat: String, Codable, Sendable {
    case task
    case litertlm
    case gguf
    case onnx
    case coreml
    case drawthings
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
    case videoGeneration = "Video Generation"

    public var icon: String {
        switch self {
        case .text: return "text.bubble.fill"
        case .multimodal: return "eye.fill"
        case .embedding: return "link.circle.fill"
        case .imageGeneration: return "photo.fill"
        case .videoGeneration: return "video.fill"
        }
    }

    public var titleKey: String {
        switch self {
        case .text: return "text_models"
        case .multimodal: return "vision_models"
        case .embedding: return "embedding_models"
        case .imageGeneration: return "image_generation_models"
        case .videoGeneration: return "video_generation_models"
        }
    }

    public var descriptionKey: String {
        switch self {
        case .text: return "text_models_description"
        case .multimodal: return "vision_models_description"
        case .embedding: return "embedding_models_description"
        case .imageGeneration: return "image_generation_models_description"
        case .videoGeneration: return "video_generation_models_description"
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
    public let url: String
    public let category: ModelCategory
    public let sizeBytes: Int64
    public let source: String
    public let supportsVision: Bool
    public let supportsAudio: Bool
    public let supportsThinking: Bool
    public let supportsGpu: Bool
    public let requirements: ModelRequirements
    public let contextWindowSize: Int
    public let modelFormat: ModelFormat
    public let additionalFiles: [String]

    public init(
        id: String? = nil,
        name: String,
        description: String,
        url: String,
        category: ModelCategory,
        sizeBytes: Int64,
        source: String,
        supportsVision: Bool = false,
        supportsAudio: Bool = false,
        supportsThinking: Bool = false,
        supportsGpu: Bool = true,
        requirements: ModelRequirements,
        contextWindowSize: Int = 2048,
        modelFormat: ModelFormat = .gguf,
        additionalFiles: [String] = []
    ) {
        self.id = id ?? name.lowercased().replacingOccurrences(of: " ", with: "_")
        self.name = name
        self.description = description
        self.url = url
        self.category = category
        self.sizeBytes = sizeBytes
        self.source = source
        self.supportsVision = supportsVision
        self.supportsAudio = supportsAudio
        self.supportsThinking = supportsThinking
        self.supportsGpu = supportsGpu
        self.requirements = requirements
        self.contextWindowSize = contextWindowSize
        self.modelFormat = modelFormat
        self.additionalFiles = additionalFiles
    }

    public var sizeLabel: String {
        let formatter = ByteCountFormatter()
        formatter.allowedUnits = [.useGB, .useMB]
        formatter.countStyle = .file
        return formatter.string(fromByteCount: sizeBytes)
    }

    public var isGemma4LiteRTLM: Bool {
        modelFormat == .litertlm
            && supportsAudio
            && name.lowercased().contains("gemma 4")
    }

    public var ramLabel: String {
        "\(requirements.minRamGB)GB RAM"
    }

    public var allDownloadURLs: [URL] {
        ([url] + additionalFiles).compactMap { URL(string: $0) }
    }

    public var requiredFileNames: [String] {
        if modelFormat == .coreml {
            // CoreML models are downloaded as ZIPs and extracted; sentinel marks completion
            return ["_downloaded"]
        }
        return allDownloadURLs.compactMap { downloadURL in
            URLComponents(url: downloadURL, resolvingAgainstBaseURL: false)?.path.split(separator: "/").last.map(String.init)
        }
    }

    public var inferenceFramework: InferenceFramework {
        switch modelFormat {
        case .onnx: return .onnx
        case .coreml: return .llamaCpp // CoreML models manage their own directories
        default: return .llamaCpp
        }
    }

    public var isCoreMLImageGeneration: Bool {
        modelFormat == .coreml && category == .imageGeneration
    }

    public var isCoreMLVideoGeneration: Bool {
        modelFormat == .coreml && category == .videoGeneration
    }

    public var isDrawThingsVideoGeneration: Bool {
        modelFormat == .drawthings && category == .videoGeneration
    }

    public var imageGenerationResolution: Int? {
        guard isCoreMLImageGeneration else { return nil }
        return id.contains("sdxl") ? 768 : 512
    }

    public var isDependencyOnly: Bool {
        let lowerName = name.lowercased()
        let lowerURL = url.lowercased()
        return lowerName.contains("vision projector")
            || lowerName.contains("mmproj")
            || lowerURL.contains("mmproj")
    }
}

public struct ModelData {
    private static let localCompletionThresholdRatio = 0.98

    private static func rerootedPath(_ storedPath: String) -> String {
        guard let docsDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else {
            return storedPath
        }
        if let marker = storedPath.range(of: "/Documents/ImportedModels/") {
            let relativeSuffix = String(storedPath[marker.upperBound...])
            return docsDir.appendingPathComponent("ImportedModels").appendingPathComponent(relativeSuffix).path
        }
        if let marker = storedPath.range(of: "/Documents/RunAnywhere/") {
            let relativeSuffix = String(storedPath[marker.upperBound...])
            return docsDir.appendingPathComponent("RunAnywhere").appendingPathComponent(relativeSuffix).path
        }
        return storedPath
    }

    private static func ggufFiles(in directory: URL) -> [URL] {
        guard let contents = try? FileManager.default.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        ) else {
            return []
        }

        return contents
            .filter { $0.pathExtension.lowercased() == "gguf" }
            .sorted { $0.lastPathComponent.localizedCaseInsensitiveCompare($1.lastPathComponent) == .orderedAscending }
    }

    private static func localFileStatus(in directory: URL, for model: AIModel) -> (allExist: Bool, totalBytes: Int64) {
        if model.modelFormat == .drawthings {
            let fileURL = directory.appendingPathComponent(model.id)
            let exists = FileManager.default.fileExists(atPath: fileURL.path)
            return (exists, exists ? model.sizeBytes : 0)
        }

        if model.isCoreMLImageGeneration || model.isCoreMLVideoGeneration {
            let sentinel = directory.appendingPathComponent("_downloaded")
            let exists = FileManager.default.fileExists(atPath: sentinel.path)
            return (exists, exists ? model.sizeBytes : 0)
        }

        guard !model.requiredFileNames.isEmpty else {
            return (false, 0)
        }

        var totalBytes: Int64 = 0
        for fileName in model.requiredFileNames {
            let fileURL = directory.appendingPathComponent(fileName)
            guard FileManager.default.fileExists(atPath: fileURL.path) else {
                return (false, totalBytes)
            }
            let size = (try? FileManager.default.attributesOfItem(atPath: fileURL.path)[.size] as? Int64) ?? 0
            totalBytes += size
        }

        return (true, totalBytes)
    }

    public static func normalizeCustomModel(_ model: AIModel) -> AIModel {
        guard model.source == "Custom" else { return model }

        var fixedURL = rerootedPath(model.url)
        var fixedAdditional = model.additionalFiles.map(rerootedPath)
        let modelDirectory = URL(fileURLWithPath: fixedURL).deletingLastPathComponent()
        let ggufFilesInDirectory = ggufFiles(in: modelDirectory)

        if fixedURL.lowercased().contains("mmproj"),
           let mainModel = ggufFilesInDirectory.first(where: { !$0.lastPathComponent.lowercased().contains("mmproj") }) {
            if !fixedAdditional.contains(fixedURL) {
                fixedAdditional.append(fixedURL)
            }
            fixedURL = mainModel.path
        }

        if model.supportsVision,
           !fixedAdditional.contains(where: { $0.lowercased().contains("mmproj") && FileManager.default.fileExists(atPath: $0) }),
           let projector = ggufFilesInDirectory.first(where: { $0.lastPathComponent.lowercased().contains("mmproj") }) {
            fixedAdditional.append(projector.path)
        }

        var seen = Set<String>()
        fixedAdditional = fixedAdditional.filter { path in
            guard path != fixedURL else { return false }
            let inserted = seen.insert(path).inserted
            return inserted
        }

        guard fixedURL != model.url || fixedAdditional != model.additionalFiles else { return model }
        return AIModel(
            id: model.id, name: model.name, description: model.description,
            url: fixedURL, category: model.category, sizeBytes: model.sizeBytes,
            source: model.source, supportsVision: model.supportsVision,
            supportsAudio: model.supportsAudio, supportsThinking: model.supportsThinking,
            supportsGpu: model.supportsGpu, requirements: model.requirements,
            contextWindowSize: model.contextWindowSize, modelFormat: model.modelFormat,
            additionalFiles: fixedAdditional
        )
    }

    public static func isModelFullyAvailableLocally(_ model: AIModel) -> Bool {
        if model.modelFormat == .drawthings {
            guard let docsDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else {
                return false
            }
            let drawThingsDir = docsDir.appendingPathComponent("Models")
            let allFiles: [String]
            if let specification = ModelZoo.specificationForModel(model.id) {
                var seen = Set<String>()
                allFiles = ModelZoo.filesToDownload(specification).map(\.file).filter { seen.insert($0).inserted }
            } else {
                allFiles = [model.id]
            }
            for f in allFiles {
                let fileURL = drawThingsDir.appendingPathComponent(f)
                if !FileManager.default.fileExists(atPath: fileURL.path) {
                    return false
                }
            }
            return true
        }

        if model.source == "Custom" {
            let normalized = normalizeCustomModel(model)
            guard FileManager.default.fileExists(atPath: normalized.url),
                  !normalized.url.lowercased().contains("mmproj") else {
                return false
            }
            if normalized.supportsVision {
                return normalized.additionalFiles.contains {
                    let lower = $0.lowercased()
                    return lower.contains("mmproj") && FileManager.default.fileExists(atPath: $0)
                }
            }
            return true
        }

        if let runAnywhereDir = try? SimplifiedFileManager.shared.getModelFolderURL(
            modelId: model.id,
            framework: model.inferenceFramework
        ), FileManager.default.fileExists(atPath: runAnywhereDir.path) {
            let status = localFileStatus(in: runAnywhereDir, for: model)
            let minimumExpectedBytes = Int64(Double(model.sizeBytes) * localCompletionThresholdRatio)
            if status.allExist && (minimumExpectedBytes <= 0 || status.totalBytes >= minimumExpectedBytes) {
                return true
            }
        }

        guard let documentsDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else {
            return false
        }
        let legacyDir = documentsDir.appendingPathComponent("models").appendingPathComponent(model.id)
        guard FileManager.default.fileExists(atPath: legacyDir.path) else {
            return false
        }

        let status = localFileStatus(in: legacyDir, for: model)
        let minimumExpectedBytes = Int64(Double(model.sizeBytes) * localCompletionThresholdRatio)
        return status.allExist && (minimumExpectedBytes <= 0 || status.totalBytes >= minimumExpectedBytes)
    }

    // Returns catalog models merged with any user-imported custom models from UserDefaults.
    public static func allModels() -> [AIModel] {
        var all = models
        if let data = UserDefaults.standard.data(forKey: "imported_models_ios"),
           let imported = try? JSONDecoder().decode([AIModel].self, from: data) {
            for model in imported where !all.contains(where: { $0.id == model.id }) {
                all.append(normalizeCustomModel(model))
            }
        }
        return all
    }

    /// iOS may relocate the app container between launches, changing the UUID in
    /// absolute paths. Re-root stored paths to the current Documents directory so
    /// imported models survive across restarts.
    private static func fixUpCustomModelPaths(_ model: AIModel) -> AIModel {
        normalizeCustomModel(model)
    }

// AUTO-GENERATED from android ModelData.kt: GGUF + ONNX models only
public static let models: [AIModel] = [
    AIModel(
        name: "Llama-3.2 1B (IQ3_M)",
        description: "Llama 3.2 1B with IQ3_M quantization. Smallest size, great for low-memory devices. 128k context. (657MB)",
        url: "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-IQ3_M.gguf?download=true",
        category: .text,
        sizeBytes: 657000000,
        source: "Meta via bartowski",
        supportsVision: false,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 2, recommendedRamGB: 3),
        contextWindowSize: 131072,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Llama-3.2 1B (IQ4_XS)",
        description: "Llama 3.2 1B with IQ4_XS quantization. Optimized 4-bit. 128k context. (743MB)",
        url: "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-IQ4_XS.gguf?download=true",
        category: .text,
        sizeBytes: 743000000,
        source: "Meta via bartowski",
        supportsVision: false,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 2, recommendedRamGB: 3),
        contextWindowSize: 131072,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Llama-3.2 1B (Q3_K_L)",
        description: "Llama 3.2 1B with Q3_K_L quantization. Large 3-bit variant. 128k context. (733MB)",
        url: "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q3_K_L.gguf?download=true",
        category: .text,
        sizeBytes: 733000000,
        source: "Meta via bartowski",
        supportsVision: false,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 2, recommendedRamGB: 3),
        contextWindowSize: 131072,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Llama-3.2 1B (Q3_K_XL)",
        description: "Llama 3.2 1B with Q3_K_XL quantization. Extra-large 3-bit variant. 128k context. (796MB)",
        url: "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q3_K_XL.gguf?download=true",
        category: .text,
        sizeBytes: 796000000,
        source: "Meta via bartowski",
        supportsVision: false,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 2, recommendedRamGB: 3),
        contextWindowSize: 131072,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Llama-3.2 1B (Q4_0)",
        description: "Llama 3.2 1B with Q4_0 quantization. Standard 4-bit, good balance. 128k context. (773MB)",
        url: "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_0.gguf?download=true",
        category: .text,
        sizeBytes: 773000000,
        source: "Meta via bartowski",
        supportsVision: false,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 2, recommendedRamGB: 3),
        contextWindowSize: 131072,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Llama-3.2 1B (Q4_0_4_4)",
        description: "Llama 3.2 1B with Q4_0_4_4 quantization. 4-bit ARM-optimized. 128k context. (771MB)",
        url: "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_0_4_4.gguf?download=true",
        category: .text,
        sizeBytes: 771000000,
        source: "Meta via bartowski",
        supportsVision: false,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 2, recommendedRamGB: 3),
        contextWindowSize: 131072,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Llama-3.2 1B (Q4_0_4_8)",
        description: "Llama 3.2 1B with Q4_0_4_8 quantization. 4-bit ARM-optimized. 128k context. (771MB)",
        url: "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_0_4_8.gguf?download=true",
        category: .text,
        sizeBytes: 771000000,
        source: "Meta via bartowski",
        supportsVision: false,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 2, recommendedRamGB: 3),
        contextWindowSize: 131072,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Llama-3.2 1B (Q4_0_8_8)",
        description: "Llama 3.2 1B with Q4_0_8_8 quantization. 4-bit ARM-optimized. 128k context. (771MB)",
        url: "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_0_8_8.gguf?download=true",
        category: .text,
        sizeBytes: 771000000,
        source: "Meta via bartowski",
        supportsVision: false,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 2, recommendedRamGB: 3),
        contextWindowSize: 131072,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Llama-3.2 1B (Q4_K_L)",
        description: "Llama 3.2 1B with Q4_K_L quantization. Large K-quant 4-bit. 128k context. (871MB)",
        url: "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_L.gguf?download=true",
        category: .text,
        sizeBytes: 871000000,
        source: "Meta via bartowski",
        supportsVision: false,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 2, recommendedRamGB: 3),
        contextWindowSize: 131072,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Llama-3.2 1B (Q4_K_M)",
        description: "Llama 3.2 1B with Q4_K_M quantization. Medium K-quant 4-bit, recommended. 128k context. (808MB)",
        url: "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf?download=true",
        category: .text,
        sizeBytes: 808000000,
        source: "Meta via bartowski",
        supportsVision: false,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 2, recommendedRamGB: 3),
        contextWindowSize: 131072,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Llama-3.2 1B (Q4_K_S)",
        description: "Llama 3.2 1B with Q4_K_S quantization. Small K-quant 4-bit. 128k context. (776MB)",
        url: "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_S.gguf?download=true",
        category: .text,
        sizeBytes: 776000000,
        source: "Meta via bartowski",
        supportsVision: false,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 2, recommendedRamGB: 3),
        contextWindowSize: 131072,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Llama-3.2 1B (Q5_K_L)",
        description: "Llama 3.2 1B with Q5_K_L quantization. Large K-quant 5-bit. 128k context. (975MB)",
        url: "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q5_K_L.gguf?download=true",
        category: .text,
        sizeBytes: 975000000,
        source: "Meta via bartowski",
        supportsVision: false,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 2, recommendedRamGB: 3),
        contextWindowSize: 131072,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Llama-3.2 1B (Q5_K_M)",
        description: "Llama 3.2 1B with Q5_K_M quantization. Medium K-quant 5-bit, high quality. 128k context. (912MB)",
        url: "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q5_K_M.gguf?download=true",
        category: .text,
        sizeBytes: 912000000,
        source: "Meta via bartowski",
        supportsVision: false,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 2, recommendedRamGB: 3),
        contextWindowSize: 131072,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Llama-3.2 1B (Q5_K_S)",
        description: "Llama 3.2 1B with Q5_K_S quantization. Small K-quant 5-bit. 128k context. (893MB)",
        url: "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q5_K_S.gguf?download=true",
        category: .text,
        sizeBytes: 893000000,
        source: "Meta via bartowski",
        supportsVision: false,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 2, recommendedRamGB: 3),
        contextWindowSize: 131072,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Llama-3.2 1B (Q6_K)",
        description: "Llama 3.2 1B with Q6_K quantization. 6-bit, very high quality. 128k context. (1.02GB)",
        url: "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q6_K.gguf?download=true",
        category: .text,
        sizeBytes: 1020000000,
        source: "Meta via bartowski",
        supportsVision: false,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 3, recommendedRamGB: 4),
        contextWindowSize: 131072,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Llama-3.2 1B (Q6_K_L)",
        description: "Llama 3.2 1B with Q6_K_L quantization. Large 6-bit, highest quality. 128k context. (1.09GB)",
        url: "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q6_K_L.gguf?download=true",
        category: .text,
        sizeBytes: 1090000000,
        source: "Meta via bartowski",
        supportsVision: false,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 3, recommendedRamGB: 4),
        contextWindowSize: 131072,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Llama-3.2 1B (Q8_0)",
        description: "Llama 3.2 1B with Q8_0 quantization. 8-bit, near-original quality. 128k context. (1.32GB)",
        url: "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q8_0.gguf?download=true",
        category: .text,
        sizeBytes: 1320000000,
        source: "Meta via bartowski",
        supportsVision: false,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 3, recommendedRamGB: 4),
        contextWindowSize: 131072,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Llama-3.2 1B (f16)",
        description: "Llama 3.2 1B with f16 (full precision). Maximum quality, largest size. 128k context. (2.48GB)",
        url: "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-f16.gguf?download=true",
        category: .text,
        sizeBytes: 2480000000,
        source: "Meta via bartowski",
        supportsVision: false,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 4, recommendedRamGB: 6),
        contextWindowSize: 131072,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Llama-3.2 3B (IQ3_M)",
        description: "Llama 3.2 3B with IQ3_M quantization. Smallest size, great for low-memory devices. 128k context. (1.6GB)",
        url: "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-IQ3_M.gguf?download=true",
        category: .text,
        sizeBytes: 1600000000,
        source: "Meta via bartowski",
        supportsVision: false,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 3, recommendedRamGB: 4),
        contextWindowSize: 131072,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Llama-3.2 3B (IQ4_XS)",
        description: "Llama 3.2 3B with IQ4_XS quantization. Optimized 4-bit. 128k context. (1.83GB)",
        url: "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-IQ4_XS.gguf?download=true",
        category: .text,
        sizeBytes: 1830000000,
        source: "Meta via bartowski",
        supportsVision: false,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 3, recommendedRamGB: 4),
        contextWindowSize: 131072,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Llama-3.2 3B (Q3_K_L)",
        description: "Llama 3.2 3B with Q3_K_L quantization. Large 3-bit variant. 128k context. (1.82GB)",
        url: "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q3_K_L.gguf?download=true",
        category: .text,
        sizeBytes: 1820000000,
        source: "Meta via bartowski",
        supportsVision: false,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 3, recommendedRamGB: 4),
        contextWindowSize: 131072,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Llama-3.2 3B (Q3_K_XL)",
        description: "Llama 3.2 3B with Q3_K_XL quantization. Extra-large 3-bit variant. 128k context. (1.91GB)",
        url: "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q3_K_XL.gguf?download=true",
        category: .text,
        sizeBytes: 1910000000,
        source: "Meta via bartowski",
        supportsVision: false,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 3, recommendedRamGB: 4),
        contextWindowSize: 131072,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Llama-3.2 3B (Q4_0)",
        description: "Llama 3.2 3B with Q4_0 quantization. Standard 4-bit, good balance. 128k context. (1.92GB)",
        url: "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_0.gguf?download=true",
        category: .text,
        sizeBytes: 1920000000,
        source: "Meta via bartowski",
        supportsVision: false,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 3, recommendedRamGB: 4),
        contextWindowSize: 131072,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Llama-3.2 3B (Q4_0_4_4)",
        description: "Llama 3.2 3B with Q4_0_4_4 quantization. 4-bit ARM-optimized. 128k context. (1.92GB)",
        url: "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_0_4_4.gguf?download=true",
        category: .text,
        sizeBytes: 1920000000,
        source: "Meta via bartowski",
        supportsVision: false,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 3, recommendedRamGB: 4),
        contextWindowSize: 131072,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Llama-3.2 3B (Q4_0_4_8)",
        description: "Llama 3.2 3B with Q4_0_4_8 quantization. 4-bit ARM-optimized. 128k context. (1.92GB)",
        url: "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_0_4_8.gguf?download=true",
        category: .text,
        sizeBytes: 1920000000,
        source: "Meta via bartowski",
        supportsVision: false,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 3, recommendedRamGB: 4),
        contextWindowSize: 131072,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Llama-3.2 3B (Q4_0_8_8)",
        description: "Llama 3.2 3B with Q4_0_8_8 quantization. 4-bit ARM-optimized. 128k context. (1.92GB)",
        url: "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_0_8_8.gguf?download=true",
        category: .text,
        sizeBytes: 1920000000,
        source: "Meta via bartowski",
        supportsVision: false,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 3, recommendedRamGB: 4),
        contextWindowSize: 131072,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Llama-3.2 3B (Q4_K_L)",
        description: "Llama 3.2 3B with Q4_K_L quantization. Large K-quant 4-bit. 128k context. (2.11GB)",
        url: "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_L.gguf?download=true",
        category: .text,
        sizeBytes: 2110000000,
        source: "Meta via bartowski",
        supportsVision: false,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 3, recommendedRamGB: 5),
        contextWindowSize: 131072,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Llama-3.2 3B (Q4_K_M)",
        description: "Llama 3.2 3B with Q4_K_M quantization. Medium K-quant 4-bit, recommended. 128k context. (2.02GB)",
        url: "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf?download=true",
        category: .text,
        sizeBytes: 2020000000,
        source: "Meta via bartowski",
        supportsVision: false,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 3, recommendedRamGB: 5),
        contextWindowSize: 131072,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Llama-3.2 3B (Q4_K_S)",
        description: "Llama 3.2 3B with Q4_K_S quantization. Small K-quant 4-bit. 128k context. (1.93GB)",
        url: "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_S.gguf?download=true",
        category: .text,
        sizeBytes: 1930000000,
        source: "Meta via bartowski",
        supportsVision: false,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 3, recommendedRamGB: 4),
        contextWindowSize: 131072,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Llama-3.2 3B (Q5_K_L)",
        description: "Llama 3.2 3B with Q5_K_L quantization. Large K-quant 5-bit. 128k context. (2.42GB)",
        url: "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q5_K_L.gguf?download=true",
        category: .text,
        sizeBytes: 2420000000,
        source: "Meta via bartowski",
        supportsVision: false,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 4, recommendedRamGB: 5),
        contextWindowSize: 131072,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Llama-3.2 3B (Q5_K_M)",
        description: "Llama 3.2 3B with Q5_K_M quantization. Medium K-quant 5-bit, high quality. 128k context. (2.32GB)",
        url: "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q5_K_M.gguf?download=true",
        category: .text,
        sizeBytes: 2320000000,
        source: "Meta via bartowski",
        supportsVision: false,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 4, recommendedRamGB: 5),
        contextWindowSize: 131072,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Llama-3.2 3B (Q5_K_S)",
        description: "Llama 3.2 3B with Q5_K_S quantization. Small K-quant 5-bit. 128k context. (2.27GB)",
        url: "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q5_K_S.gguf?download=true",
        category: .text,
        sizeBytes: 2270000000,
        source: "Meta via bartowski",
        supportsVision: false,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 4, recommendedRamGB: 5),
        contextWindowSize: 131072,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Llama-3.2 3B (Q6_K)",
        description: "Llama 3.2 3B with Q6_K quantization. 6-bit, very high quality. 128k context. (2.64GB)",
        url: "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q6_K.gguf?download=true",
        category: .text,
        sizeBytes: 2640000000,
        source: "Meta via bartowski",
        supportsVision: false,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 4, recommendedRamGB: 6),
        contextWindowSize: 131072,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Llama-3.2 3B (Q6_K_L)",
        description: "Llama 3.2 3B with Q6_K_L quantization. Large 6-bit, highest quality. 128k context. (2.74GB)",
        url: "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q6_K_L.gguf?download=true",
        category: .text,
        sizeBytes: 2740000000,
        source: "Meta via bartowski",
        supportsVision: false,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 4, recommendedRamGB: 6),
        contextWindowSize: 131072,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Llama-3.2 3B (Q8_0)",
        description: "Llama 3.2 3B with Q8_0 quantization. 8-bit, near-original quality. 128k context. (3.42GB)",
        url: "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q8_0.gguf?download=true",
        category: .text,
        sizeBytes: 3420000000,
        source: "Meta via bartowski",
        supportsVision: false,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 5, recommendedRamGB: 7),
        contextWindowSize: 131072,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Llama-3.2 3B (f16)",
        description: "Llama 3.2 3B with f16 (full precision). Maximum quality, largest size. 128k context. (6.43GB)",
        url: "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-f16.gguf?download=true",
        category: .text,
        sizeBytes: 6430000000,
        source: "Meta via bartowski",
        supportsVision: false,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 8, recommendedRamGB: 10),
        contextWindowSize: 131072,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Granite 4.0 H-Tiny (Q2_K)",
        description: "IBM Granite 4.0 H-Tiny with Q2_K quantization. Smallest size. 128k context. (2.59GB)",
        url: "https://huggingface.co/ibm-granite/granite-4.0-h-tiny-GGUF/resolve/main/granite-4.0-h-tiny-Q2_K.gguf?download=true",
        category: .text,
        sizeBytes: 2590000000,
        source: "IBM Granite",
        supportsVision: false,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 3, recommendedRamGB: 4),
        contextWindowSize: 131072,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Granite 4.0 H-Tiny (Q3_K_S)",
        description: "IBM Granite 4.0 H-Tiny with Q3_K_S quantization. Balanced size. 128k context. (3.06GB)",
        url: "https://huggingface.co/ibm-granite/granite-4.0-h-tiny-GGUF/resolve/main/granite-4.0-h-tiny-Q3_K_S.gguf?download=true",
        category: .text,
        sizeBytes: 3060000000,
        source: "IBM Granite",
        supportsVision: false,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 4, recommendedRamGB: 5),
        contextWindowSize: 131072,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Granite 4.0 H-Tiny (Q3_K_M)",
        description: "IBM Granite 4.0 H-Tiny with Q3_K_M quantization. Good quality. 128k context. (3.35GB)",
        url: "https://huggingface.co/ibm-granite/granite-4.0-h-tiny-GGUF/resolve/main/granite-4.0-h-tiny-Q3_K_M.gguf?download=true",
        category: .text,
        sizeBytes: 3350000000,
        source: "IBM Granite",
        supportsVision: false,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 4, recommendedRamGB: 5),
        contextWindowSize: 131072,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Granite 4.0 H-Tiny (Q3_K_L)",
        description: "IBM Granite 4.0 H-Tiny with Q3_K_L quantization. Better quality. 128k context. (3.6GB)",
        url: "https://huggingface.co/ibm-granite/granite-4.0-h-tiny-GGUF/resolve/main/granite-4.0-h-tiny-Q3_K_L.gguf?download=true",
        category: .text,
        sizeBytes: 3600000000,
        source: "IBM Granite",
        supportsVision: false,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 4, recommendedRamGB: 5),
        contextWindowSize: 131072,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Granite 4.0 H-Tiny (Q4_0)",
        description: "IBM Granite 4.0 H-Tiny with Q4_0 quantization. Good balance. 128k context. (3.96GB)",
        url: "https://huggingface.co/ibm-granite/granite-4.0-h-tiny-GGUF/resolve/main/granite-4.0-h-tiny-Q4_0.gguf?download=true",
        category: .text,
        sizeBytes: 3960000000,
        source: "IBM Granite",
        supportsVision: false,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 5, recommendedRamGB: 6),
        contextWindowSize: 131072,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Granite 4.0 H-Tiny (Q4_K_S)",
        description: "IBM Granite 4.0 H-Tiny with Q4_K_S quantization. High quality. 128k context. (4GB)",
        url: "https://huggingface.co/ibm-granite/granite-4.0-h-tiny-GGUF/resolve/main/granite-4.0-h-tiny-Q4_K_S.gguf?download=true",
        category: .text,
        sizeBytes: 4000000000,
        source: "IBM Granite",
        supportsVision: false,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 5, recommendedRamGB: 6),
        contextWindowSize: 131072,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Granite 4.0 H-Tiny (Q4_K_M)",
        description: "IBM Granite 4.0 H-Tiny with Q4_K_M quantization. Very high quality. 128k context. (4.23GB)",
        url: "https://huggingface.co/ibm-granite/granite-4.0-h-tiny-GGUF/resolve/main/granite-4.0-h-tiny-Q4_K_M.gguf?download=true",
        category: .text,
        sizeBytes: 4230000000,
        source: "IBM Granite",
        supportsVision: false,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 5, recommendedRamGB: 6),
        contextWindowSize: 131072,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Granite 4.0 H-Tiny (Q4_1)",
        description: "IBM Granite 4.0 H-Tiny with Q4_1 quantization. Enhanced quality. 128k context. (4.39GB)",
        url: "https://huggingface.co/ibm-granite/granite-4.0-h-tiny-GGUF/resolve/main/granite-4.0-h-tiny-Q4_1.gguf?download=true",
        category: .text,
        sizeBytes: 4390000000,
        source: "IBM Granite",
        supportsVision: false,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 5, recommendedRamGB: 6),
        contextWindowSize: 131072,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Granite 4.0 H-Tiny (Q5_K_S)",
        description: "IBM Granite 4.0 H-Tiny with Q5_K_S quantization. Excellent quality. 128k context. (4.81GB)",
        url: "https://huggingface.co/ibm-granite/granite-4.0-h-tiny-GGUF/resolve/main/granite-4.0-h-tiny-Q5_K_S.gguf?download=true",
        category: .text,
        sizeBytes: 4810000000,
        source: "IBM Granite",
        supportsVision: false,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 6, recommendedRamGB: 7),
        contextWindowSize: 131072,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Granite 4.0 H-Tiny (Q5_0)",
        description: "IBM Granite 4.0 H-Tiny with Q5_0 quantization. Near-lossless. 128k context. (4.81GB)",
        url: "https://huggingface.co/ibm-granite/granite-4.0-h-tiny-GGUF/resolve/main/granite-4.0-h-tiny-Q5_0.gguf?download=true",
        category: .text,
        sizeBytes: 4810000000,
        source: "IBM Granite",
        supportsVision: false,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 6, recommendedRamGB: 7),
        contextWindowSize: 131072,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Granite 4.0 H-Tiny (Q5_K_M)",
        description: "IBM Granite 4.0 H-Tiny with Q5_K_M quantization. Superior quality. 128k context. (4.95GB)",
        url: "https://huggingface.co/ibm-granite/granite-4.0-h-tiny-GGUF/resolve/main/granite-4.0-h-tiny-Q5_K_M.gguf?download=true",
        category: .text,
        sizeBytes: 4950000000,
        source: "IBM Granite",
        supportsVision: false,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 6, recommendedRamGB: 7),
        contextWindowSize: 131072,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Granite 4.0 H-Tiny (Q5_1)",
        description: "IBM Granite 4.0 H-Tiny with Q5_1 quantization. Premium quality. 128k context. (5.23GB)",
        url: "https://huggingface.co/ibm-granite/granite-4.0-h-tiny-GGUF/resolve/main/granite-4.0-h-tiny-Q5_1.gguf?download=true",
        category: .text,
        sizeBytes: 5230000000,
        source: "IBM Granite",
        supportsVision: false,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 6, recommendedRamGB: 7),
        contextWindowSize: 131072,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Granite 4.0 H-Tiny (Q6_K)",
        description: "IBM Granite 4.0 H-Tiny with Q6_K quantization. Outstanding quality. 128k context. (5.71GB)",
        url: "https://huggingface.co/ibm-granite/granite-4.0-h-tiny-GGUF/resolve/main/granite-4.0-h-tiny-Q6_K.gguf?download=true",
        category: .text,
        sizeBytes: 5710000000,
        source: "IBM Granite",
        supportsVision: false,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 7, recommendedRamGB: 8),
        contextWindowSize: 131072,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Granite 4.0 H-Tiny (Q8_0)",
        description: "IBM Granite 4.0 H-Tiny with Q8_0 quantization. Ultimate quality. 128k context. (7.39GB)",
        url: "https://huggingface.co/ibm-granite/granite-4.0-h-tiny-GGUF/resolve/main/granite-4.0-h-tiny-Q8_0.gguf?download=true",
        category: .text,
        sizeBytes: 7390000000,
        source: "IBM Granite",
        supportsVision: false,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 8, recommendedRamGB: 10),
        contextWindowSize: 131072,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Granite 4.0 H-Tiny (f16)",
        description: "IBM Granite 4.0 H-Tiny with f16 (full precision). Maximum quality, largest size. 128k context. (13.9GB)",
        url: "https://huggingface.co/ibm-granite/granite-4.0-h-tiny-GGUF/resolve/main/granite-4.0-h-tiny-f16.gguf?download=true",
        category: .text,
        sizeBytes: 13900000000,
        source: "IBM Granite",
        supportsVision: false,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 14, recommendedRamGB: 16),
        contextWindowSize: 131072,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Granite 4.0 H-Small (Q2_K)",
        description: "IBM Granite 4.0 H-Small with Q2_K quantization. Smallest size. 128k context. (11.8GB)",
        url: "https://huggingface.co/ibm-granite/granite-4.0-h-small-GGUF/resolve/main/granite-4.0-h-small-Q2_K.gguf?download=true",
        category: .text,
        sizeBytes: 11800000000,
        source: "IBM Granite",
        supportsVision: false,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 12, recommendedRamGB: 14),
        contextWindowSize: 131072,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Granite 4.0 H-Small (Q3_K_S)",
        description: "IBM Granite 4.0 H-Small with Q3_K_S quantization. Balanced size. 128k context. (14.1GB)",
        url: "https://huggingface.co/ibm-granite/granite-4.0-h-small-GGUF/resolve/main/granite-4.0-h-small-Q3_K_S.gguf?download=true",
        category: .text,
        sizeBytes: 14100000000,
        source: "IBM Granite",
        supportsVision: false,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 14, recommendedRamGB: 16),
        contextWindowSize: 131072,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Granite 4.0 H-Small (Q3_K_M)",
        description: "IBM Granite 4.0 H-Small with Q3_K_M quantization. Good quality. 128k context. (15.4GB)",
        url: "https://huggingface.co/ibm-granite/granite-4.0-h-small-GGUF/resolve/main/granite-4.0-h-small-Q3_K_M.gguf?download=true",
        category: .text,
        sizeBytes: 15400000000,
        source: "IBM Granite",
        supportsVision: false,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 15, recommendedRamGB: 18),
        contextWindowSize: 131072,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Granite 4.0 H-Small (Q3_K_L)",
        description: "IBM Granite 4.0 H-Small with Q3_K_L quantization. Better quality. 128k context. (16.5GB)",
        url: "https://huggingface.co/ibm-granite/granite-4.0-h-small-GGUF/resolve/main/granite-4.0-h-small-Q3_K_L.gguf?download=true",
        category: .text,
        sizeBytes: 16500000000,
        source: "IBM Granite",
        supportsVision: false,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 16, recommendedRamGB: 20),
        contextWindowSize: 131072,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Granite 4.0 H-Small (Q4_0)",
        description: "IBM Granite 4.0 H-Small with Q4_0 quantization. Good balance. 128k context. (18.3GB)",
        url: "https://huggingface.co/ibm-granite/granite-4.0-h-small-GGUF/resolve/main/granite-4.0-h-small-Q4_0.gguf?download=true",
        category: .text,
        sizeBytes: 18300000000,
        source: "IBM Granite",
        supportsVision: false,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 18, recommendedRamGB: 22),
        contextWindowSize: 131072,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Granite 4.0 H-Small (Q4_K_S)",
        description: "IBM Granite 4.0 H-Small with Q4_K_S quantization. High quality. 128k context. (18.4GB)",
        url: "https://huggingface.co/ibm-granite/granite-4.0-h-small-GGUF/resolve/main/granite-4.0-h-small-Q4_K_S.gguf?download=true",
        category: .text,
        sizeBytes: 18400000000,
        source: "IBM Granite",
        supportsVision: false,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 18, recommendedRamGB: 22),
        contextWindowSize: 131072,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Granite 4.0 H-Small (Q4_K_M)",
        description: "IBM Granite 4.0 H-Small with Q4_K_M quantization. Very high quality. 128k context. (19.5GB)",
        url: "https://huggingface.co/ibm-granite/granite-4.0-h-small-GGUF/resolve/main/granite-4.0-h-small-Q4_K_M.gguf?download=true",
        category: .text,
        sizeBytes: 19500000000,
        source: "IBM Granite",
        supportsVision: false,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 19, recommendedRamGB: 24),
        contextWindowSize: 131072,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Granite 4.0 H-Small (Q4_1)",
        description: "IBM Granite 4.0 H-Small with Q4_1 quantization. Enhanced quality. 128k context. (20.3GB)",
        url: "https://huggingface.co/ibm-granite/granite-4.0-h-small-GGUF/resolve/main/granite-4.0-h-small-Q4_1.gguf?download=true",
        category: .text,
        sizeBytes: 20300000000,
        source: "IBM Granite",
        supportsVision: false,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 20, recommendedRamGB: 24),
        contextWindowSize: 131072,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Granite 4.0 H-Small (Q5_0)",
        description: "IBM Granite 4.0 H-Small with Q5_0 quantization. Near-lossless. 128k context. (22.2GB)",
        url: "https://huggingface.co/ibm-granite/granite-4.0-h-small-GGUF/resolve/main/granite-4.0-h-small-Q5_0.gguf?download=true",
        category: .text,
        sizeBytes: 22200000000,
        source: "IBM Granite",
        supportsVision: false,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 22, recommendedRamGB: 26),
        contextWindowSize: 131072,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Granite 4.0 H-Small (Q5_K_S)",
        description: "IBM Granite 4.0 H-Small with Q5_K_S quantization. Excellent quality. 128k context. (22.2GB)",
        url: "https://huggingface.co/ibm-granite/granite-4.0-h-small-GGUF/resolve/main/granite-4.0-h-small-Q5_K_S.gguf?download=true",
        category: .text,
        sizeBytes: 22200000000,
        source: "IBM Granite",
        supportsVision: false,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 22, recommendedRamGB: 26),
        contextWindowSize: 131072,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Granite 4.0 H-Small (Q5_K_M)",
        description: "IBM Granite 4.0 H-Small with Q5_K_M quantization. Superior quality. 128k context. (22.9GB)",
        url: "https://huggingface.co/ibm-granite/granite-4.0-h-small-GGUF/resolve/main/granite-4.0-h-small-Q5_K_M.gguf?download=true",
        category: .text,
        sizeBytes: 22900000000,
        source: "IBM Granite",
        supportsVision: false,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 23, recommendedRamGB: 26),
        contextWindowSize: 131072,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Granite 4.0 H-Small (Q5_1)",
        description: "IBM Granite 4.0 H-Small with Q5_1 quantization. Premium quality. 128k context. (24.2GB)",
        url: "https://huggingface.co/ibm-granite/granite-4.0-h-small-GGUF/resolve/main/granite-4.0-h-small-Q5_1.gguf?download=true",
        category: .text,
        sizeBytes: 24200000000,
        source: "IBM Granite",
        supportsVision: false,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 24, recommendedRamGB: 28),
        contextWindowSize: 131072,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Granite 4.0 H-Small (Q6_K)",
        description: "IBM Granite 4.0 H-Small with Q6_K quantization. Outstanding quality. 128k context. (26.5GB)",
        url: "https://huggingface.co/ibm-granite/granite-4.0-h-small-GGUF/resolve/main/granite-4.0-h-small-Q6_K.gguf?download=true",
        category: .text,
        sizeBytes: 26500000000,
        source: "IBM Granite",
        supportsVision: false,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 26, recommendedRamGB: 30),
        contextWindowSize: 131072,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Granite 4.0 H-Small (Q8_0)",
        description: "IBM Granite 4.0 H-Small with Q8_0 quantization. Ultimate quality. 128k context. (34.3GB)",
        url: "https://huggingface.co/ibm-granite/granite-4.0-h-small-GGUF/resolve/main/granite-4.0-h-small-Q8_0.gguf?download=true",
        category: .text,
        sizeBytes: 34300000000,
        source: "IBM Granite",
        supportsVision: false,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 34, recommendedRamGB: 40),
        contextWindowSize: 131072,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Granite 4.0 H-Small (f16)",
        description: "IBM Granite 4.0 H-Small with f16 (full precision). Maximum quality, largest size. 128k context. (64.4GB)",
        url: "https://huggingface.co/ibm-granite/granite-4.0-h-small-GGUF/resolve/main/granite-4.0-h-small-f16.gguf?download=true",
        category: .text,
        sizeBytes: 64400000000,
        source: "IBM Granite",
        supportsVision: false,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 64, recommendedRamGB: 80),
        contextWindowSize: 131072,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "LFM-2.5 1.2B Instruct (Q4_0)",
        description: "LiquidAI's 1.2B instruct model. Q4_0 quantization. 128k context.",
        url: "https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF/resolve/main/LFM2.5-1.2B-Instruct-Q4_0.gguf?download=true",
        category: .text,
        sizeBytes: 696000000,
        source: "LiquidAI",
        supportsVision: false,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 2, recommendedRamGB: 3),
        contextWindowSize: 128000,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "LFM-2.5 1.2B Instruct (Q4_K_M)",
        description: "LiquidAI's 1.2B instruct model. Q4_K_M quantization. 128k context.",
        url: "https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF/resolve/main/LFM2.5-1.2B-Instruct-Q4_K_M.gguf?download=true",
        category: .text,
        sizeBytes: 731000000,
        source: "LiquidAI",
        supportsVision: false,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 2, recommendedRamGB: 3),
        contextWindowSize: 128000,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "LFM-2.5 1.2B Instruct (Q8_0)",
        description: "LiquidAI's 1.2B instruct model. Q8_0 quantization. 128k context.",
        url: "https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF/resolve/main/LFM2.5-1.2B-Instruct-Q8_0.gguf?download=true",
        category: .text,
        sizeBytes: 1250000000,
        source: "LiquidAI",
        supportsVision: false,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 3, recommendedRamGB: 4),
        contextWindowSize: 128000,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "LFM-2.5 1.2B Thinking (Q4_0)",
        description: "LiquidAI's 1.2B thinking model. Q4_0 quantization. 128k context. Supports 'thinking' mode.",
        url: "https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking-GGUF/resolve/main/LFM2.5-1.2B-Thinking-Q4_0.gguf?download=true",
        category: .text,
        sizeBytes: 696000000,
        source: "LiquidAI",
        supportsVision: false,
        supportsAudio: false,
        supportsThinking: true,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 2, recommendedRamGB: 3),
        contextWindowSize: 128000,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "LFM-2.5 1.2B Thinking (Q4_K_M)",
        description: "LiquidAI's 1.2B thinking model. Q4_K_M quantization. 128k context. Supports 'thinking' mode.",
        url: "https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking-GGUF/resolve/main/LFM2.5-1.2B-Thinking-Q4_K_M.gguf?download=true",
        category: .text,
        sizeBytes: 731000000,
        source: "LiquidAI",
        supportsVision: false,
        supportsAudio: false,
        supportsThinking: true,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 2, recommendedRamGB: 3),
        contextWindowSize: 128000,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "LFM-2.5 1.2B Thinking (Q8_0)",
        description: "LiquidAI's 1.2B thinking model. Q8_0 quantization. 128k context. Supports 'thinking' mode.",
        url: "https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking-GGUF/resolve/main/LFM2.5-1.2B-Thinking-Q8_0.gguf?download=true",
        category: .text,
        sizeBytes: 1250000000,
        source: "LiquidAI",
        supportsVision: false,
        supportsAudio: false,
        supportsThinking: true,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 3, recommendedRamGB: 4),
        contextWindowSize: 128000,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    
    // MARK: - LFM-2.5 8B Models (LiquidAI GGUF - MoE, 8.3B total / 1.5B active params)
    AIModel(
        name: "LFM2.5-8B-A1B (Q4_0)",
        description: "Liquid's 8.3B parameter MoE model (1.5B active). Q4_0 quantization. 131k context. (4.84GB)",
        url: "https://huggingface.co/LiquidAI/LFM2.5-8B-A1B-GGUF/resolve/dfd5fdcad7a1c0d31473fb4ca443b8befbacddf0/LFM2.5-8B-A1B-Q4_0.gguf?download=true",
        category: .text,
        sizeBytes: 4844678368,
        source: "LiquidAI",
        supportsVision: false,
        supportsAudio: false,
        supportsThinking: true,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 6, recommendedRamGB: 8),
        contextWindowSize: 131072,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "LFM2.5-8B-A1B (Q4_K_M)",
        description: "Liquid's 8.3B parameter MoE model (1.5B active). Q4_K_M quantization, recommended. 131k context. (5.16GB)",
        url: "https://huggingface.co/LiquidAI/LFM2.5-8B-A1B-GGUF/resolve/dfd5fdcad7a1c0d31473fb4ca443b8befbacddf0/LFM2.5-8B-A1B-Q4_K_M.gguf?download=true",
        category: .text,
        sizeBytes: 5155564768,
        source: "LiquidAI",
        supportsVision: false,
        supportsAudio: false,
        supportsThinking: true,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 7, recommendedRamGB: 9),
        contextWindowSize: 131072,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "LFM2.5-8B-A1B (Q5_K_M)",
        description: "Liquid's 8.3B parameter MoE model (1.5B active). Q5_K_M quantization. 131k context. (6.03GB)",
        url: "https://huggingface.co/LiquidAI/LFM2.5-8B-A1B-GGUF/resolve/dfd5fdcad7a1c0d31473fb4ca443b8befbacddf0/LFM2.5-8B-A1B-Q5_K_M.gguf?download=true",
        category: .text,
        sizeBytes: 6030339296,
        source: "LiquidAI",
        supportsVision: false,
        supportsAudio: false,
        supportsThinking: true,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 8, recommendedRamGB: 10),
        contextWindowSize: 131072,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "LFM2.5-8B-A1B (Q6_K)",
        description: "Liquid's 8.3B parameter MoE model (1.5B active). Q6_K quantization. 131k context. (6.96GB)",
        url: "https://huggingface.co/LiquidAI/LFM2.5-8B-A1B-GGUF/resolve/dfd5fdcad7a1c0d31473fb4ca443b8befbacddf0/LFM2.5-8B-A1B-Q6_K.gguf?download=true",
        category: .text,
        sizeBytes: 6959787232,
        source: "LiquidAI",
        supportsVision: false,
        supportsAudio: false,
        supportsThinking: true,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 9, recommendedRamGB: 11),
        contextWindowSize: 131072,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "LFM2.5-8B-A1B (Q8_0)",
        description: "Liquid's 8.3B parameter MoE model (1.5B active). Q8_0 quantization. 131k context. (9.01GB)",
        url: "https://huggingface.co/LiquidAI/LFM2.5-8B-A1B-GGUF/resolve/dfd5fdcad7a1c0d31473fb4ca443b8befbacddf0/LFM2.5-8B-A1B-Q8_0.gguf?download=true",
        category: .text,
        sizeBytes: 9010195680,
        source: "LiquidAI",
        supportsVision: false,
        supportsAudio: false,
        supportsThinking: true,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 11, recommendedRamGB: 14),
        contextWindowSize: 131072,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "LFM2.5-8B-A1B (BF16)",
        description: "Liquid's 8.3B parameter MoE model (1.5B active). BF16 (bfloat16 precision) variant. 131k context. (16.95GB)",
        url: "https://huggingface.co/LiquidAI/LFM2.5-8B-A1B-GGUF/resolve/dfd5fdcad7a1c0d31473fb4ca443b8befbacddf0/LFM2.5-8B-A1B-BF16.gguf?download=true",
        category: .text,
        sizeBytes: 16947260640,
        source: "LiquidAI",
        supportsVision: false,
        supportsAudio: false,
        supportsThinking: true,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 20, recommendedRamGB: 24),
        contextWindowSize: 131072,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "LFM2.5-8B-A1B (F16)",
        description: "Liquid's 8.3B parameter MoE model (1.5B active). F16 (float16 precision) variant. 131k context. (16.95GB)",
        url: "https://huggingface.co/LiquidAI/LFM2.5-8B-A1B-GGUF/resolve/dfd5fdcad7a1c0d31473fb4ca443b8befbacddf0/LFM2.5-8B-A1B-F16.gguf?download=true",
        category: .text,
        sizeBytes: 16947260640,
        source: "LiquidAI",
        supportsVision: false,
        supportsAudio: false,
        supportsThinking: true,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 20, recommendedRamGB: 24),
        contextWindowSize: 131072,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "LFM2-24B-A2B (Q4_0)",
        description: "LiquidAI's MoE model with 24B total parameters but only 2B active per token. Q4_0 quantization. 32k context. Multilingual (9 languages).",
        url: "https://huggingface.co/LiquidAI/LFM2-24B-A2B-GGUF/resolve/main/LFM2-24B-A2B-Q4_0.gguf?download=true",
        category: .text,
        sizeBytes: 13500000000,
        source: "LiquidAI",
        supportsVision: false,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 16, recommendedRamGB: 18),
        contextWindowSize: 32768,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "LFM2-24B-A2B (Q4_K_M)",
        description: "LiquidAI's MoE model with 24B total parameters but only 2B active per token. Q4_K_M quantization. 32k context. Multilingual (9 languages).",
        url: "https://huggingface.co/LiquidAI/LFM2-24B-A2B-GGUF/resolve/main/LFM2-24B-A2B-Q4_K_M.gguf?download=true",
        category: .text,
        sizeBytes: 14400000000,
        source: "LiquidAI",
        supportsVision: false,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 16, recommendedRamGB: 18),
        contextWindowSize: 32768,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "LFM2-24B-A2B (Q5_K_M)",
        description: "LiquidAI's MoE model with 24B total parameters but only 2B active per token. Q5_K_M quantization for better quality. 32k context. Multilingual (9 languages).",
        url: "https://huggingface.co/LiquidAI/LFM2-24B-A2B-GGUF/resolve/main/LFM2-24B-A2B-Q5_K_M.gguf?download=true",
        category: .text,
        sizeBytes: 16900000000,
        source: "LiquidAI",
        supportsVision: false,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 18, recommendedRamGB: 24),
        contextWindowSize: 32768,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "LFM2-24B-A2B (Q6_K)",
        description: "LiquidAI's MoE model with 24B total parameters but only 2B active per token. Q6_K quantization for high quality. 32k context. Multilingual (9 languages).",
        url: "https://huggingface.co/LiquidAI/LFM2-24B-A2B-GGUF/resolve/main/LFM2-24B-A2B-Q6_K.gguf?download=true",
        category: .text,
        sizeBytes: 19600000000,
        source: "LiquidAI",
        supportsVision: false,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 20, recommendedRamGB: 24),
        contextWindowSize: 32768,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "LFM2-24B-A2B (Q8_0)",
        description: "LiquidAI's MoE model with 24B total parameters but only 2B active per token. Q8_0 quantization - near full quality. 32k context. Multilingual (9 languages).",
        url: "https://huggingface.co/LiquidAI/LFM2-24B-A2B-GGUF/resolve/main/LFM2-24B-A2B-Q8_0.gguf?download=true",
        category: .text,
        sizeBytes: 25400000000,
        source: "LiquidAI",
        supportsVision: false,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 28, recommendedRamGB: 32),
        contextWindowSize: 32768,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "LFM-2.5 VL 1.6B (BF16)",
        description: "LiquidAI's 1.6B vision-language model. BF16 precision. Supports vision + text. Requires mmproj for vision.",
        url: "https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B-GGUF/resolve/main/LFM2.5-VL-1.6B-BF16.gguf?download=true",
        category: .multimodal,
        sizeBytes: 2340000000,
        source: "LiquidAI",
        supportsVision: true,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 4, recommendedRamGB: 6),
        contextWindowSize: 128000,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "LFM-2.5 VL 1.6B (F16)",
        description: "LiquidAI's 1.6B vision-language model. F16 precision. Supports vision + text. Requires mmproj for vision.",
        url: "https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B-GGUF/resolve/main/LFM2.5-VL-1.6B-F16.gguf?download=true",
        category: .multimodal,
        sizeBytes: 2340000000,
        source: "LiquidAI",
        supportsVision: false,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 4, recommendedRamGB: 6),
        contextWindowSize: 128000,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "LFM-2.5 VL 1.6B (Q4_0)",
        description: "LiquidAI's 1.6B vision-language model. Q4_0 quantization. Supports vision + text. Requires mmproj for vision.",
        url: "https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B-GGUF/resolve/main/LFM2.5-VL-1.6B-Q4_0.gguf?download=true",
        category: .multimodal,
        sizeBytes: 696000000,
        source: "LiquidAI",
        supportsVision: false,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 3, recommendedRamGB: 4),
        contextWindowSize: 128000,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "LFM-2.5 VL 1.6B (Q8_0)",
        description: "LiquidAI's 1.6B vision-language model. Q8_0 quantization. Supports vision + text. Requires mmproj for vision.",
        url: "https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B-GGUF/resolve/main/LFM2.5-VL-1.6B-Q8_0.gguf?download=true",
        category: .multimodal,
        sizeBytes: 1250000000,
        source: "LiquidAI",
        supportsVision: true,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 3, recommendedRamGB: 5),
        contextWindowSize: 128000,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "LFM-2.5 VL 1.6B (Vision Projector, BF16)",
        description: "Vision Projector for LFM-2.5 VL models. BF16 variant required for image input. Download this to enable vision.",
        url: "https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B-GGUF/resolve/main/mmproj-LFM2.5-VL-1.6b-BF16.gguf?download=true",
        category: .multimodal,
        sizeBytes: 856000000,
        source: "LiquidAI",
        supportsVision: true,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 1, recommendedRamGB: 2),
        contextWindowSize: 0,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "LFM-2.5 VL 1.6B (Vision Projector, Q8_0)",
        description: "Vision Projector for LFM-2.5 VL models. Q8_0 quantized variant for smaller size. Download this to enable vision.",
        url: "https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B-GGUF/resolve/main/mmproj-LFM2.5-VL-1.6b-Q8_0.gguf?download=true",
        category: .multimodal,
        sizeBytes: 583000000,
        source: "LiquidAI",
        supportsVision: true,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 1, recommendedRamGB: 2),
        contextWindowSize: 0,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Ministral-3 3B Instruct (Q4_K_M)",
        description: "MistralAI's 3B instruct model. Q4_K_M quantization. 32k context. Supports Vision (Requires mmproj).",
        url: "https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-GGUF/resolve/main/Ministral-3-3B-Instruct-2512-Q4_K_M.gguf?download=true",
        category: .multimodal,
        sizeBytes: 2150000000,
        source: "MistralAI",
        supportsVision: true,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 4, recommendedRamGB: 6),
        contextWindowSize: 262144,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Ministral-3 3B Instruct (Q5_K_M)",
        description: "MistralAI's 3B instruct model. Q5_K_M quantization. 32k context. Supports Vision (Requires mmproj).",
        url: "https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-GGUF/resolve/main/Ministral-3-3B-Instruct-2512-Q5_K_M.gguf?download=true",
        category: .multimodal,
        sizeBytes: 2470000000,
        source: "MistralAI",
        supportsVision: true,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 4, recommendedRamGB: 7),
        contextWindowSize: 262144,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Ministral-3 3B Instruct (Q8_0)",
        description: "MistralAI's 3B instruct model. Q8_0 quantization. 32k context. Supports Vision (Requires mmproj).",
        url: "https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-GGUF/resolve/main/Ministral-3-3B-Instruct-2512-Q8_0.gguf?download=true",
        category: .multimodal,
        sizeBytes: 3650000000,
        source: "MistralAI",
        supportsVision: true,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 6, recommendedRamGB: 8),
        contextWindowSize: 262144,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Ministral-3 3B Instruct (Vision Projector, BF16)",
        description: "Multimodal Vision Projector for Ministral-3 3B models. Specifically the BF16 variant required for image input capabilities. Download this if you want to enable Vision for Ministral models.",
        url: "https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-GGUF/resolve/main/Ministral-3-3B-Instruct-2512-BF16-mmproj.gguf?download=true",
        category: .multimodal,
        sizeBytes: 842000000,
        source: "MistralAI",
        supportsVision: true,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 1, recommendedRamGB: 2),
        contextWindowSize: 0,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "GPT-OSS 20B (Q4_K_M)",
        description: "OpenAI's open-weight MoE model (21B total params, 3.6B active). Q4_K_M quantization. 128k context. Uses Harmony chat format. Supports reasoning modes (low/medium/high).",
        url: "https://huggingface.co/unsloth/gpt-oss-20b-GGUF/resolve/main/gpt-oss-20b-Q4_K_M.gguf?download=true",
        category: .text,
        sizeBytes: 11600000000,
        source: "OpenAI",
        supportsVision: false,
        supportsAudio: false,
        supportsThinking: true,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 14, recommendedRamGB: 16),
        contextWindowSize: 131072,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "GPT-OSS 20B (Q5_K_M)",
        description: "OpenAI's open-weight MoE model (21B total params, 3.6B active). Q5_K_M quantization for better quality. 128k context. Uses Harmony chat format. Supports reasoning modes.",
        url: "https://huggingface.co/unsloth/gpt-oss-20b-GGUF/resolve/main/gpt-oss-20b-Q5_K_M.gguf?download=true",
        category: .text,
        sizeBytes: 11700000000,
        source: "OpenAI",
        supportsVision: false,
        supportsAudio: false,
        supportsThinking: true,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 14, recommendedRamGB: 16),
        contextWindowSize: 131072,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "GPT-OSS 20B (Q6_K)",
        description: "OpenAI's open-weight MoE model (21B total params, 3.6B active). Q6_K quantization for high quality. 128k context. Uses Harmony chat format. Supports reasoning modes.",
        url: "https://huggingface.co/unsloth/gpt-oss-20b-GGUF/resolve/main/gpt-oss-20b-Q6_K.gguf?download=true",
        category: .text,
        sizeBytes: 12000000000,
        source: "OpenAI",
        supportsVision: false,
        supportsAudio: false,
        supportsThinking: true,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 14, recommendedRamGB: 16),
        contextWindowSize: 131072,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "GPT-OSS 20B (Q8_0)",
        description: "OpenAI's open-weight MoE model (21B total params, 3.6B active). Q8_0 quantization - near full quality. 128k context. Uses Harmony chat format. Supports reasoning modes.",
        url: "https://huggingface.co/unsloth/gpt-oss-20b-GGUF/resolve/main/gpt-oss-20b-Q8_0.gguf?download=true",
        category: .text,
        sizeBytes: 12100000000,
        source: "OpenAI",
        supportsVision: false,
        supportsAudio: false,
        supportsThinking: true,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 14, recommendedRamGB: 16),
        contextWindowSize: 131072,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Translate Gemma 4B (Q2_K)",
        description: "Translate Gemma 4B with Q2_K quantization. Smallest Translate Gemma option. Supports text + vision and requires a separate Vision Projector (mmproj). 2k context.",
        url: "https://huggingface.co/mradermacher/translategemma-4b-it-GGUF/resolve/main/translategemma-4b-it.Q2_K.gguf?download=true",
        category: .multimodal,
        sizeBytes: 1729180160,
        source: "Google via mradermacher",
        supportsVision: true,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 4, recommendedRamGB: 5),
        contextWindowSize: 2048,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Translate Gemma 4B (Q3_K_L)",
        description: "Translate Gemma 4B with Q3_K_L quantization. Larger 3-bit variant. Supports text + vision and requires a separate Vision Projector (mmproj). 2k context.",
        url: "https://huggingface.co/mradermacher/translategemma-4b-it-GGUF/resolve/main/translategemma-4b-it.Q3_K_L.gguf?download=true",
        category: .multimodal,
        sizeBytes: 2236101120,
        source: "Google via mradermacher",
        supportsVision: true,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 4, recommendedRamGB: 6),
        contextWindowSize: 2048,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Translate Gemma 4B (Q3_K_M)",
        description: "Translate Gemma 4B with Q3_K_M quantization. Balanced 3-bit variant. Supports text + vision and requires a separate Vision Projector (mmproj). 2k context.",
        url: "https://huggingface.co/mradermacher/translategemma-4b-it-GGUF/resolve/main/translategemma-4b-it.Q3_K_M.gguf?download=true",
        category: .multimodal,
        sizeBytes: 2098475520,
        source: "Google via mradermacher",
        supportsVision: true,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 4, recommendedRamGB: 5),
        contextWindowSize: 2048,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Translate Gemma 4B (Q3_K_S)",
        description: "Translate Gemma 4B with Q3_K_S quantization. Small 3-bit variant. Supports text + vision and requires a separate Vision Projector (mmproj). 2k context.",
        url: "https://huggingface.co/mradermacher/translategemma-4b-it-GGUF/resolve/main/translategemma-4b-it.Q3_K_S.gguf?download=true",
        category: .multimodal,
        sizeBytes: 1937379840,
        source: "Google via mradermacher",
        supportsVision: true,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 4, recommendedRamGB: 5),
        contextWindowSize: 2048,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Translate Gemma 4B (IQ4_XS)",
        description: "Translate Gemma 4B with IQ4_XS quantization. Efficient 4-bit variant. Supports text + vision and requires a separate Vision Projector (mmproj). 2k context.",
        url: "https://huggingface.co/mradermacher/translategemma-4b-it-GGUF/resolve/main/translategemma-4b-it.IQ4_XS.gguf?download=true",
        category: .multimodal,
        sizeBytes: 2279641600,
        source: "Google via mradermacher",
        supportsVision: true,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 5, recommendedRamGB: 6),
        contextWindowSize: 2048,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Translate Gemma 4B (Q4_K_S)",
        description: "Translate Gemma 4B with Q4_K_S quantization. Smaller 4-bit variant. Supports text + vision and requires a separate Vision Projector (mmproj). 2k context.",
        url: "https://huggingface.co/mradermacher/translategemma-4b-it-GGUF/resolve/main/translategemma-4b-it.Q4_K_S.gguf?download=true",
        category: .multimodal,
        sizeBytes: 2377945600,
        source: "Google via mradermacher",
        supportsVision: true,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 5, recommendedRamGB: 6),
        contextWindowSize: 2048,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Translate Gemma 4B (Q4_K_M)",
        description: "Translate Gemma 4B with Q4_K_M quantization. Recommended balance for quality and size. Supports text + vision and requires a separate Vision Projector (mmproj). 2k context.",
        url: "https://huggingface.co/mradermacher/translategemma-4b-it-GGUF/resolve/main/translategemma-4b-it.Q4_K_M.gguf?download=true",
        category: .multimodal,
        sizeBytes: 2489909760,
        source: "Google via mradermacher",
        supportsVision: true,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 5, recommendedRamGB: 6),
        contextWindowSize: 2048,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Translate Gemma 4B (Q5_K_S)",
        description: "Translate Gemma 4B with Q5_K_S quantization. Higher-quality 5-bit variant. Supports text + vision and requires a separate Vision Projector (mmproj). 2k context.",
        url: "https://huggingface.co/mradermacher/translategemma-4b-it-GGUF/resolve/main/translategemma-4b-it.Q5_K_S.gguf?download=true",
        category: .multimodal,
        sizeBytes: 2764608000,
        source: "Google via mradermacher",
        supportsVision: true,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 6, recommendedRamGB: 8),
        contextWindowSize: 2048,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Translate Gemma 4B (Q5_K_M)",
        description: "Translate Gemma 4B with Q5_K_M quantization. High-quality 5-bit variant. Supports text + vision and requires a separate Vision Projector (mmproj). 2k context.",
        url: "https://huggingface.co/mradermacher/translategemma-4b-it-GGUF/resolve/main/translategemma-4b-it.Q5_K_M.gguf?download=true",
        category: .multimodal,
        sizeBytes: 2829713920,
        source: "Google via mradermacher",
        supportsVision: true,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 6, recommendedRamGB: 8),
        contextWindowSize: 2048,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Translate Gemma 4B (Q6_K)",
        description: "Translate Gemma 4B with Q6_K quantization. High-precision 6-bit variant. Supports text + vision and requires a separate Vision Projector (mmproj). 2k context.",
        url: "https://huggingface.co/mradermacher/translategemma-4b-it-GGUF/resolve/main/translategemma-4b-it.Q6_K.gguf?download=true",
        category: .multimodal,
        sizeBytes: 3190755840,
        source: "Google via mradermacher",
        supportsVision: true,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 7, recommendedRamGB: 8),
        contextWindowSize: 2048,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Translate Gemma 4B (Q8_0)",
        description: "Translate Gemma 4B with Q8_0 quantization. Near full quality. Supports text + vision and requires a separate Vision Projector (mmproj). 2k context.",
        url: "https://huggingface.co/mradermacher/translategemma-4b-it-GGUF/resolve/main/translategemma-4b-it.Q8_0.gguf?download=true",
        category: .multimodal,
        sizeBytes: 4130417920,
        source: "Google via mradermacher",
        supportsVision: true,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 8, recommendedRamGB: 10),
        contextWindowSize: 2048,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Translate Gemma 4B (F16)",
        description: "Translate Gemma 4B with F16 precision. Highest quality and largest download. Supports text + vision and requires a separate Vision Projector (mmproj). 2k context.",
        url: "https://huggingface.co/mradermacher/translategemma-4b-it-GGUF/resolve/main/translategemma-4b-it.f16.gguf?download=true",
        category: .multimodal,
        sizeBytes: 7767819520,
        source: "Google via mradermacher",
        supportsVision: true,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 12, recommendedRamGB: 16),
        contextWindowSize: 2048,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Translate Gemma 4B (Vision Projector, Q8_0)",
        description: "Vision Projector (mmproj) required to enable image input for Translate Gemma 4B quantized variants.",
        url: "https://huggingface.co/mradermacher/translategemma-4b-it-GGUF/resolve/main/translategemma-4b-it.mmproj-Q8_0.gguf?download=true",
        category: .multimodal,
        sizeBytes: 591377600,
        source: "Google via mradermacher",
        supportsVision: true,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 1, recommendedRamGB: 2),
        contextWindowSize: 0,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Translate Gemma 4B (Vision Projector, F16)",
        description: "Vision Projector (mmproj) required to enable image input for the F16 Translate Gemma 4B variant.",
        url: "https://huggingface.co/mradermacher/translategemma-4b-it-GGUF/resolve/main/translategemma-4b-it.mmproj-f16.gguf?download=true",
        category: .multimodal,
        sizeBytes: 851251520,
        source: "Google via mradermacher",
        supportsVision: true,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 1, recommendedRamGB: 2),
        contextWindowSize: 0,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Gemma-3 4B (Q4_0, GGUF)",
        description: "Google Gemma-3 4B quantized GGUF (Q4_0). Supports text + vision — download the Vision Projector (mmproj) to enable image input.",
        url: "https://huggingface.co/unsloth/gemma-3-4b-it-GGUF/resolve/main/gemma-3-4b-it-Q4_0.gguf?download=true",
        category: .multimodal,
        sizeBytes: 2370065536,
        source: "Google via Unsloth",
        supportsVision: true,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 6, recommendedRamGB: 8),
        contextWindowSize: 4096,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Gemma-3 4B (Vision Projector, BF16)",
        description: "Vision Projector (mmproj) required to enable image input for Gemma-3 4B. BF16 variant for accurate visual encodings.",
        url: "https://huggingface.co/unsloth/gemma-3-4b-it-GGUF/resolve/main/mmproj-F16.gguf?download=true",
        category: .multimodal,
        sizeBytes: 851251328,
        source: "Google via Unsloth",
        supportsVision: true,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 1, recommendedRamGB: 2),
        contextWindowSize: 0,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Gemma-3 12B (Q4_0, GGUF)",
        description: "Google Gemma-3 12B quantized GGUF (Q4_0). Supports text + vision — download the Vision Projector (mmproj) to enable image input.",
        url: "https://huggingface.co/unsloth/gemma-3-12b-it-GGUF/resolve/main/gemma-3-12b-it-Q4_0.gguf?download=true",
        category: .multimodal,
        sizeBytes: 6909282656,
        source: "Google via Unsloth",
        supportsVision: true,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 12, recommendedRamGB: 16),
        contextWindowSize: 4096,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Gemma-3 12B (Vision Projector, BF16)",
        description: "Vision Projector (mmproj) required to enable image input for Gemma-3 12B. BF16 variant for accurate visual encodings.",
        url: "https://huggingface.co/unsloth/gemma-3-12b-it-GGUF/resolve/main/mmproj-F16.gguf?download=true",
        category: .multimodal,
        sizeBytes: 854200448,
        source: "Google via Unsloth",
        supportsVision: true,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 1, recommendedRamGB: 2),
        contextWindowSize: 0,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    // MARK: - Ternary Bonsai Models (prism-ml)
    AIModel(
        name: "Ternary Bonsai 1.7B (F16)",
        description: "Ternary Bonsai 1.7B in full float16 precision. Highest quality variant for high-end devices with ample RAM. (3.21GB)",
        url: "https://huggingface.co/prism-ml/Ternary-Bonsai-1.7B-gguf/resolve/main/Ternary-Bonsai-1.7B-F16.gguf?download=true",
        category: .text,
        sizeBytes: 3446249408,
        source: "Prism ML",
        supportsVision: false,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 5, recommendedRamGB: 6),
        contextWindowSize: 4096,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Ternary Bonsai 4B (F16)",
        description: "Ternary Bonsai 4B in full float16 precision. High quality, large size. (7.50GB)",
        url: "https://huggingface.co/prism-ml/Ternary-Bonsai-4B-gguf/resolve/main/Ternary-Bonsai-4B-F16.gguf?download=true",
        category: .text,
        sizeBytes: 8049911840,
        source: "Prism ML",
        supportsVision: false,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 10, recommendedRamGB: 12),
        contextWindowSize: 4096,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    // MARK: - Granite 4.1 3B Models (Unsloth)
    AIModel(
        name: "Granite 4.1 3B (Q2_K_XL)",
        description: "IBM Granite 4.1 3B with Q2_K_XL quantization. Ultra-compact 2-bit variant for low-memory devices. 512K context. (1.32GB)",
        url: "https://huggingface.co/unsloth/granite-4.1-3b-GGUF/resolve/main/granite-4.1-3b-UD-Q2_K_XL.gguf?download=true",
        category: .text,
        sizeBytes: 1414548800,
        source: "IBM via Unsloth",
        supportsVision: false,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 3, recommendedRamGB: 4),
        contextWindowSize: 524288,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Granite 4.1 3B (IQ3_XXS)",
        description: "IBM Granite 4.1 3B with IQ3_XXS quantization. Ultra-compact 3-bit variant. 512K context. (1.32GB)",
        url: "https://huggingface.co/unsloth/granite-4.1-3b-GGUF/resolve/main/granite-4.1-3b-UD-IQ3_XXS.gguf?download=true",
        category: .text,
        sizeBytes: 1416576320,
        source: "IBM via Unsloth",
        supportsVision: false,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 3, recommendedRamGB: 4),
        contextWindowSize: 524288,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Granite 4.1 3B (Q3_K_XL)",
        description: "IBM Granite 4.1 3B with Q3_K_XL quantization. Balanced 3-bit variant. 512K context. (1.67GB)",
        url: "https://huggingface.co/unsloth/granite-4.1-3b-GGUF/resolve/main/granite-4.1-3b-UD-Q3_K_XL.gguf?download=true",
        category: .text,
        sizeBytes: 1794667840,
        source: "IBM via Unsloth",
        supportsVision: false,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 3, recommendedRamGB: 4),
        contextWindowSize: 524288,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Granite 4.1 3B (Q4_K_XL)",
        description: "IBM Granite 4.1 3B with Q4_K_XL quantization. Standard 4-bit variant. 512K context. (2.00GB)",
        url: "https://huggingface.co/unsloth/granite-4.1-3b-GGUF/resolve/main/granite-4.1-3b-UD-Q4_K_XL.gguf?download=true",
        category: .text,
        sizeBytes: 2152381760,
        source: "IBM via Unsloth",
        supportsVision: false,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 4, recommendedRamGB: 5),
        contextWindowSize: 524288,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Granite 4.1 3B (Q5_K_XL)",
        description: "IBM Granite 4.1 3B with Q5_K_XL quantization. High-quality 5-bit variant. 512K context. (2.28GB)",
        url: "https://huggingface.co/unsloth/granite-4.1-3b-GGUF/resolve/main/granite-4.1-3b-UD-Q5_K_XL.gguf?download=true",
        category: .text,
        sizeBytes: 2453939520,
        source: "IBM via Unsloth",
        supportsVision: false,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 4, recommendedRamGB: 6),
        contextWindowSize: 524288,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Granite 4.1 3B (Q6_K_XL)",
        description: "IBM Granite 4.1 3B with Q6_K_XL quantization. High-quality 6-bit variant. 512K context. (2.84GB)",
        url: "https://huggingface.co/unsloth/granite-4.1-3b-GGUF/resolve/main/granite-4.1-3b-UD-Q6_K_XL.gguf?download=true",
        category: .text,
        sizeBytes: 3048299840,
        source: "IBM via Unsloth",
        supportsVision: false,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 5, recommendedRamGB: 7),
        contextWindowSize: 524288,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Granite 4.1 3B (Q8_K_XL)",
        description: "IBM Granite 4.1 3B with Q8_K_XL quantization. Maximum quality 8-bit variant. 512K context. (3.99GB)",
        url: "https://huggingface.co/unsloth/granite-4.1-3b-GGUF/resolve/main/granite-4.1-3b-UD-Q8_K_XL.gguf?download=true",
        category: .text,
        sizeBytes: 4284472640,
        source: "IBM via Unsloth",
        supportsVision: false,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 6, recommendedRamGB: 8),
        contextWindowSize: 524288,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    // MARK: - Gemma 4 E2B (2B active params)
    AIModel(
        name: "Gemma 4 E2B (Q3_K_M)",
        description: "Google Gemma 4 E2B with Q3_K_M quantization (2.54 GB). 2B effective active params. Supports text + vision — download the Vision Projector (mmproj) to enable image input. 128k context.",
        url: "https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/f7c65a52de0efed3b8ab461e02e4448b3f760a01/gemma-4-E2B-it-Q3_K_M.gguf?download=true",
        category: .multimodal,
        sizeBytes: 2536779136,
        source: "Google via Unsloth",
        supportsVision: true,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 4, recommendedRamGB: 6),
        contextWindowSize: 131072,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Gemma 4 E2B (Q4_K_M)",
        description: "Google Gemma 4 E2B with Q4_K_M quantization (3.11 GB). Recommended balance of quality and size. Supports text + vision — download the Vision Projector (mmproj) to enable image input. 128k context.",
        url: "https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/f7c65a52de0efed3b8ab461e02e4448b3f760a01/gemma-4-E2B-it-Q4_K_M.gguf?download=true",
        category: .multimodal,
        sizeBytes: 3106731392,
        source: "Google via Unsloth",
        supportsVision: true,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 5, recommendedRamGB: 7),
        contextWindowSize: 131072,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Gemma 4 E2B (Q5_K_M)",
        description: "Google Gemma 4 E2B with Q5_K_M quantization (3.36 GB). Higher-quality 5-bit variant. Supports text + vision — download the Vision Projector (mmproj) to enable image input. 128k context.",
        url: "https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/f7c65a52de0efed3b8ab461e02e4448b3f760a01/gemma-4-E2B-it-Q5_K_M.gguf?download=true",
        category: .multimodal,
        sizeBytes: 3356030336,
        source: "Google via Unsloth",
        supportsVision: true,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 5, recommendedRamGB: 7),
        contextWindowSize: 131072,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Gemma 4 E2B (Q8_0)",
        description: "Google Gemma 4 E2B with Q8_0 quantization (5.05 GB). Near full quality. Supports text + vision — download the Vision Projector (mmproj) to enable image input. 128k context.",
        url: "https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/f7c65a52de0efed3b8ab461e02e4448b3f760a01/gemma-4-E2B-it-Q8_0.gguf?download=true",
        category: .multimodal,
        sizeBytes: 5048345984,
        source: "Google via Unsloth",
        supportsVision: true,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 7, recommendedRamGB: 9),
        contextWindowSize: 131072,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Gemma 4 E2B (Vision Projector, F16)",
        description: "Vision Projector (mmproj) required to enable image input for Gemma 4 E2B. F16 variant for accurate visual encodings. (986 MB)",
        url: "https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/f7c65a52de0efed3b8ab461e02e4448b3f760a01/mmproj-F16.gguf?download=true",
        category: .multimodal,
        sizeBytes: 985654208,
        source: "Google via Unsloth",
        supportsVision: true,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 1, recommendedRamGB: 2),
        contextWindowSize: 0,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    // MARK: - Gemma 4 E4B (4B active params)
    AIModel(
        name: "Gemma 4 E4B (Q3_K_M)",
        description: "Google Gemma 4 E4B with Q3_K_M quantization (4.06 GB). 4B effective active params. Supports text + vision — download the Vision Projector (mmproj) to enable image input. 128k context.",
        url: "https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/960a8cd001a5ec7a679e2c5d93f9916238e76d10/gemma-4-E4B-it-Q3_K_M.gguf?download=true",
        category: .multimodal,
        sizeBytes: 4058130816,
        source: "Google via Unsloth",
        supportsVision: true,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 6, recommendedRamGB: 8),
        contextWindowSize: 131072,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Gemma 4 E4B (Q4_K_M)",
        description: "Google Gemma 4 E4B with Q4_K_M quantization (4.98 GB). Recommended balance of quality and size. Supports text + vision — download the Vision Projector (mmproj) to enable image input. 128k context.",
        url: "https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/960a8cd001a5ec7a679e2c5d93f9916238e76d10/gemma-4-E4B-it-Q4_K_M.gguf?download=true",
        category: .multimodal,
        sizeBytes: 4977164672,
        source: "Google via Unsloth",
        supportsVision: true,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 7, recommendedRamGB: 10),
        contextWindowSize: 131072,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Gemma 4 E4B (Q5_K_M)",
        description: "Google Gemma 4 E4B with Q5_K_M quantization (5.48 GB). Higher-quality 5-bit variant. Supports text + vision — download the Vision Projector (mmproj) to enable image input. 128k context.",
        url: "https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/960a8cd001a5ec7a679e2c5d93f9916238e76d10/gemma-4-E4B-it-Q5_K_M.gguf?download=true",
        category: .multimodal,
        sizeBytes: 5481791872,
        source: "Google via Unsloth",
        supportsVision: true,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 8, recommendedRamGB: 12),
        contextWindowSize: 131072,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Gemma 4 E4B (Q8_0)",
        description: "Google Gemma 4 E4B with Q8_0 quantization (8.19 GB). Near full quality. Supports text + vision — download the Vision Projector (mmproj) to enable image input. 128k context.",
        url: "https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/960a8cd001a5ec7a679e2c5d93f9916238e76d10/gemma-4-E4B-it-Q8_0.gguf?download=true",
        category: .multimodal,
        sizeBytes: 8192946560,
        source: "Google via Unsloth",
        supportsVision: true,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 10, recommendedRamGB: 14),
        contextWindowSize: 131072,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Gemma 4 E4B (Vision Projector, F16)",
        description: "Vision Projector (mmproj) required to enable image input for Gemma 4 E4B. F16 variant for accurate visual encodings. (990 MB)",
        url: "https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/960a8cd001a5ec7a679e2c5d93f9916238e76d10/mmproj-F16.gguf?download=true",
        category: .multimodal,
        sizeBytes: 990372800,
        source: "Google via Unsloth",
        supportsVision: true,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 1, recommendedRamGB: 2),
        contextWindowSize: 0,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    // MARK: - Gemma 4 LiteRT-LM (Google native on-device, GPU/Metal)
    AIModel(
        name: "Gemma 4 E2B (LiteRT-LM)",
        description: "Google Gemma 4 E2B via LiteRT-LM — Google's native on-device runtime with GPU/Metal acceleration. Multimodal: supports text + vision + audio. 32k context. (2.41 GB)",
        url: "https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/a4a831c060880f3733135ad22f10e0e9f758f45d/gemma-4-E2B-it.litertlm?download=true",
        category: .multimodal,
        sizeBytes: 2588147712,
        source: "Google via LiteRT Community",
        supportsVision: true,
        supportsAudio: true,
        supportsThinking: true,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 4, recommendedRamGB: 6),
        contextWindowSize: 32768,
        modelFormat: .litertlm,
        additionalFiles: []
    ),
    AIModel(
        name: "Gemma 4 E4B (LiteRT-LM)",
        description: "Google Gemma 4 E4B via LiteRT-LM — Google's native on-device runtime with GPU/Metal acceleration. Multimodal: supports text + vision + audio. 32k context. (3.41 GB)",
        url: "https://huggingface.co/litert-community/gemma-4-E4B-it-litert-lm/resolve/28299f30ee4d43294517a4ac93abd6163412f07f/gemma-4-E4B-it.litertlm?download=true",
        category: .multimodal,
        sizeBytes: 3659530240,
        source: "Google via LiteRT Community",
        supportsVision: true,
        supportsAudio: true,
        supportsThinking: true,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 6, recommendedRamGB: 8),
        contextWindowSize: 32768,
        modelFormat: .litertlm,
        additionalFiles: []
    ),
    AIModel(
        name: "Gemma 4 12B (LiteRT-LM)",
        description: "Google Gemma 4 12B via LiteRT-LM — Google's native on-device runtime with GPU/Metal acceleration. 32k context. (6.54 GB)",
        url: "https://huggingface.co/litert-community/gemma-4-12B-it-litert-lm/resolve/44cf85a326f79b814fa86a60af414c042755b43a/gemma-4-12B-it.litertlm?download=true",
        category: .text,
        sizeBytes: 6547589312,
        source: "Google via LiteRT Community",
        supportsVision: false,
        supportsAudio: false,
        supportsThinking: true,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 12, recommendedRamGB: 16),
        contextWindowSize: 4096,
        modelFormat: .litertlm,
        additionalFiles: []
    ),
    // MARK: - Gemma 4 12B (12B active params)
    AIModel(
        name: "Gemma 4 12B (Q3_K_M)",
        description: "Google Gemma 4 12B with Q3_K_M quantization (5.69 GB). 12B parameters. Supports text + vision — download the Vision Projector (mmproj) to enable image input. 128k context.",
        url: "https://huggingface.co/unsloth/gemma-4-12b-it-GGUF/resolve/b2f1b1f5549253e499db75d34f94acb233d8b7ca/gemma-4-12b-it-Q3_K_M.gguf?download=true",
        category: .multimodal,
        sizeBytes: 5693871520,
        source: "Google via Unsloth",
        supportsVision: true,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 10, recommendedRamGB: 14),
        contextWindowSize: 131072,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Gemma 4 12B (Q4_K_M)",
        description: "Google Gemma 4 12B with Q4_K_M quantization (7.12 GB). Recommended balance of quality and size. Supports text + vision — download the Vision Projector (mmproj) to enable image input. 128k context.",
        url: "https://huggingface.co/unsloth/gemma-4-12b-it-GGUF/resolve/b2f1b1f5549253e499db75d34f94acb233d8b7ca/gemma-4-12b-it-Q4_K_M.gguf?download=true",
        category: .multimodal,
        sizeBytes: 7121860000,
        source: "Google via Unsloth",
        supportsVision: true,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 12, recommendedRamGB: 16),
        contextWindowSize: 131072,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Gemma 4 12B (Q5_K_M)",
        description: "Google Gemma 4 12B with Q5_K_M quantization (8.41 GB). Higher-quality 5-bit variant. Supports text + vision — download the Vision Projector (mmproj) to enable image input. 128k context.",
        url: "https://huggingface.co/unsloth/gemma-4-12b-it-GGUF/resolve/b2f1b1f5549253e499db75d34f94acb233d8b7ca/gemma-4-12b-it-Q5_K_M.gguf?download=true",
        category: .multimodal,
        sizeBytes: 8413574560,
        source: "Google via Unsloth",
        supportsVision: true,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 14, recommendedRamGB: 18),
        contextWindowSize: 131072,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Gemma 4 12B (Q8_0)",
        description: "Google Gemma 4 12B with Q8_0 quantization (12.67 GB). Near full quality. Supports text + vision — download the Vision Projector (mmproj) to enable image input. 128k context.",
        url: "https://huggingface.co/unsloth/gemma-4-12b-it-GGUF/resolve/b2f1b1f5549253e499db75d34f94acb233d8b7ca/gemma-4-12b-it-Q8_0.gguf?download=true",
        category: .multimodal,
        sizeBytes: 12669646240,
        source: "Google via Unsloth",
        supportsVision: true,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 16, recommendedRamGB: 20),
        contextWindowSize: 131072,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Gemma 4 12B (Vision Projector, F16)",
        description: "Vision Projector (mmproj) required to enable image input for Gemma 4 12B. F16 variant for accurate visual encodings. (122 MB)",
        url: "https://huggingface.co/unsloth/gemma-4-12b-it-GGUF/resolve/b2f1b1f5549253e499db75d34f94acb233d8b7ca/mmproj-F16.gguf?download=true",
        category: .multimodal,
        sizeBytes: 122031680,
        source: "Google via Unsloth",
        supportsVision: true,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 1, recommendedRamGB: 2),
        contextWindowSize: 0,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    
    // MARK: - Phi-4 Mini Models (Microsoft via unsloth GGUF)
    AIModel(
        name: "Phi-4 Mini (Q2_K)",
        description: "Microsoft Phi-4 Mini (3.8B) instruct model with Q2_K quantization. Smallest size. 4k context. (1.68GB)",
        url: "https://huggingface.co/unsloth/Phi-4-mini-instruct-GGUF/resolve/main/Phi-4-mini-instruct-Q2_K.gguf?download=true",
        category: .text,
        sizeBytes: 1682635744,
        source: "Microsoft via unsloth",
        supportsVision: false,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 3, recommendedRamGB: 4),
        contextWindowSize: 4096,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Phi-4 Mini (Q2_K_L)",
        description: "Microsoft Phi-4 Mini (3.8B) instruct model with Q2_K_L quantization. Large 2-bit. 4k context. (1.68GB)",
        url: "https://huggingface.co/unsloth/Phi-4-mini-instruct-GGUF/resolve/main/Phi-4-mini-instruct-Q2_K_L.gguf?download=true",
        category: .text,
        sizeBytes: 1682635744,
        source: "Microsoft via unsloth",
        supportsVision: false,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 3, recommendedRamGB: 4),
        contextWindowSize: 4096,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Phi-4 Mini (Q3_K_M)",
        description: "Microsoft Phi-4 Mini (3.8B) instruct model with Q3_K_M quantization. Balanced 3-bit. 4k context. (2.12GB)",
        url: "https://huggingface.co/unsloth/Phi-4-mini-instruct-GGUF/resolve/main/Phi-4-mini-instruct-Q3_K_M.gguf?download=true",
        category: .text,
        sizeBytes: 2117532640,
        source: "Microsoft via unsloth",
        supportsVision: false,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 3, recommendedRamGB: 4),
        contextWindowSize: 4096,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Phi-4 Mini (Q4_K_M)",
        description: "Microsoft Phi-4 Mini (3.8B) instruct model with Q4_K_M quantization. Recommended. 4k context. (2.49GB)",
        url: "https://huggingface.co/unsloth/Phi-4-mini-instruct-GGUF/resolve/main/Phi-4-mini-instruct-Q4_K_M.gguf?download=true",
        category: .text,
        sizeBytes: 2491874272,
        source: "Microsoft via unsloth",
        supportsVision: false,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 3, recommendedRamGB: 4),
        contextWindowSize: 4096,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Phi-4 Mini (Q5_K_M)",
        description: "Microsoft Phi-4 Mini (3.8B) instruct model with Q5_K_M quantization. High quality. 4k context. (2.85GB)",
        url: "https://huggingface.co/unsloth/Phi-4-mini-instruct-GGUF/resolve/main/Phi-4-mini-instruct-Q5_K_M.gguf?download=true",
        category: .text,
        sizeBytes: 2848127968,
        source: "Microsoft via unsloth",
        supportsVision: false,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 4, recommendedRamGB: 5),
        contextWindowSize: 4096,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Phi-4 Mini (Q6_K)",
        description: "Microsoft Phi-4 Mini (3.8B) instruct model with Q6_K quantization. Near-lossless. 4k context. (3.16GB)",
        url: "https://huggingface.co/unsloth/Phi-4-mini-instruct-GGUF/resolve/main/Phi-4-mini-instruct-Q6_K.gguf?download=true",
        category: .text,
        sizeBytes: 3155622880,
        source: "Microsoft via unsloth",
        supportsVision: false,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 4, recommendedRamGB: 5),
        contextWindowSize: 4096,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Phi-4 Mini (Q8_0)",
        description: "Microsoft Phi-4 Mini (3.8B) instruct model with Q8_0 quantization. Excellent quality. 4k context. (4.08GB)",
        url: "https://huggingface.co/unsloth/Phi-4-mini-instruct-GGUF/resolve/main/Phi-4-mini-instruct.Q8_0.gguf?download=true",
        category: .text,
        sizeBytes: 4084611040,
        source: "Microsoft via unsloth",
        supportsVision: false,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 5, recommendedRamGB: 6),
        contextWindowSize: 4096,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    AIModel(
        name: "Phi-4 Mini (BF16)",
        description: "Microsoft Phi-4 Mini (3.8B) instruct model in BF16 (full precision) format. Largest size. 4k context. (7.68GB)",
        url: "https://huggingface.co/unsloth/Phi-4-mini-instruct-GGUF/resolve/main/Phi-4-mini-instruct.BF16.gguf?download=true",
        category: .text,
        sizeBytes: 7680694240,
        source: "Microsoft via unsloth",
        supportsVision: false,
        supportsAudio: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 8, recommendedRamGB: 10),
        contextWindowSize: 4096,
        modelFormat: .gguf,
        additionalFiles: []
    ),
    
    // MARK: - Image Generation (CoreML Stable Diffusion)
    AIModel(
        id: "sd-v1-4-apple-palettized-split-einsum-v2",
        name: "Stable Diffusion v1.4 (ANE)",
        description: "Stable Diffusion v1.4 — Apple official palettized split-einsum v2 bundle for iPhone/iPad Neural Engine deployment. Generates 512×512 images. Requires iOS 17+. (1.57 GB)",
        url: "https://huggingface.co/apple/coreml-stable-diffusion-1-4-palettized/resolve/main/coreml-stable-diffusion-1-4-palettized_split_einsum_v2_compiled.zip?download=true",
        category: .imageGeneration,
        sizeBytes: 1685774664,
        source: "Apple / Hugging Face",
        supportsVision: false,
        supportsAudio: false,
        supportsThinking: false,
        supportsGpu: false,
        requirements: ModelRequirements(minRamGB: 4, recommendedRamGB: 6),
        contextWindowSize: 0,
        modelFormat: .coreml,
        additionalFiles: []
    ),
    AIModel(
        id: "sd-v1-5-apple-palettized-split-einsum-v2",
        name: "Stable Diffusion v1.5 (ANE)",
        description: "Stable Diffusion v1.5 — Apple official palettized split-einsum v2 bundle for iPhone/iPad Neural Engine deployment. Generates 512×512 images. Requires iOS 17+. (1.57 GB)",
        url: "https://huggingface.co/apple/coreml-stable-diffusion-v1-5-palettized/resolve/main/coreml-stable-diffusion-v1-5-palettized_split_einsum_v2_compiled.zip?download=true",
        category: .imageGeneration,
        sizeBytes: 1685774664,
        source: "Apple / Hugging Face",
        supportsVision: false,
        supportsAudio: false,
        supportsThinking: false,
        supportsGpu: false,
        requirements: ModelRequirements(minRamGB: 4, recommendedRamGB: 6),
        contextWindowSize: 0,
        modelFormat: .coreml,
        additionalFiles: []
    ),
    AIModel(
        id: "sd-v2-base-apple-palettized-split-einsum-v2",
        name: "Stable Diffusion 2 Base (ANE)",
        description: "Stable Diffusion 2 Base — Apple official palettized split-einsum v2 bundle for iPhone/iPad Neural Engine deployment. Generates 512×512 images. Requires iOS 17+. (1.14 GB)",
        url: "https://huggingface.co/apple/coreml-stable-diffusion-2-base-palettized/resolve/main/coreml-stable-diffusion-2-base-palettized_split_einsum_v2_compiled.zip?download=true",
        category: .imageGeneration,
        sizeBytes: 1224065679,
        source: "Apple / Hugging Face",
        supportsVision: false,
        supportsAudio: false,
        supportsThinking: false,
        supportsGpu: false,
        requirements: ModelRequirements(minRamGB: 4, recommendedRamGB: 6),
        contextWindowSize: 0,
        modelFormat: .coreml,
        additionalFiles: []
    ),
    AIModel(
        id: "sd-v2-1-base-apple-palettized-split-einsum-v2",
        name: "Stable Diffusion 2.1 Base (ANE)",
        description: "Stable Diffusion 2.1 Base — Apple official palettized split-einsum v2 bundle for iPhone/iPad Neural Engine deployment. Generates 512×512 images. Requires iOS 17+. (1.14 GB)",
        url: "https://huggingface.co/apple/coreml-stable-diffusion-2-1-base-palettized/resolve/main/coreml-stable-diffusion-2-1-base-palettized_split_einsum_v2_compiled.zip?download=true",
        category: .imageGeneration,
        sizeBytes: 1224065679,
        source: "Apple / Hugging Face",
        supportsVision: false,
        supportsAudio: false,
        supportsThinking: false,
        supportsGpu: false,
        requirements: ModelRequirements(minRamGB: 4, recommendedRamGB: 6),
        contextWindowSize: 0,
        modelFormat: .coreml,
        additionalFiles: []
    ),
    AIModel(
        id: "sdxl-base-ios-apple-split-einsum",
        name: "Stable Diffusion XL Base iOS (ANE)",
        description: "Stable Diffusion XL Base iOS — Apple official split-einsum bundle for iPhone/iPad Neural Engine deployment. Generates 768×768 images. Requires iOS 17+. Much slower and heavier than SD 1.x/2.x. (3.05 GB)",
        url: "https://huggingface.co/apple/coreml-stable-diffusion-xl-base-ios/resolve/main/coreml-stable-diffusion-xl-base-ios_split_einsum_compiled.zip?download=true",
        category: .imageGeneration,
        sizeBytes: 3274912563,
        source: "Apple / Hugging Face",
        supportsVision: false,
        supportsAudio: false,
        supportsThinking: false,
        supportsGpu: false,
        requirements: ModelRequirements(minRamGB: 6, recommendedRamGB: 8),
        contextWindowSize: 0,
        modelFormat: .coreml,
        additionalFiles: []
    ),

    // MARK: - EmbeddingGemma 300M ONNX — Embedding Models for RAG

    AIModel(
        id: "embeddinggemma-300m-onnx-q4",
        name: "EmbeddingGemma 300M (Q4)",
        description: "Google EmbeddingGemma 300M Q4 quantized ONNX. Lightweight semantic embedding model for RAG and memory. (219 MB)",
        url: "https://huggingface.co/onnx-community/embeddinggemma-300m-ONNX/resolve/main/onnx/model_q4.onnx?download=true",
        category: .embedding,
        sizeBytes: 218725224,
        source: "Google / ONNX Community",
        supportsVision: false,
        supportsAudio: false,
        supportsThinking: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 1, recommendedRamGB: 2),
        contextWindowSize: 512,
        modelFormat: .onnx,
        additionalFiles: [
            "https://huggingface.co/onnx-community/embeddinggemma-300m-ONNX/resolve/main/onnx/model_q4.onnx_data?download=true",
            "https://huggingface.co/onnx-community/embeddinggemma-300m-ONNX/resolve/main/tokenizer.json?download=true",
            "https://huggingface.co/onnx-community/embeddinggemma-300m-ONNX/resolve/main/tokenizer_config.json?download=true"
        ]
    ),
    AIModel(
        id: "embeddinggemma-300m-onnx-int8",
        name: "EmbeddingGemma 300M (INT8)",
        description: "Google EmbeddingGemma 300M INT8 quantized ONNX. Balanced size and accuracy for RAG and memory. (331 MB)",
        url: "https://huggingface.co/onnx-community/embeddinggemma-300m-ONNX/resolve/main/onnx/model_quantized.onnx?download=true",
        category: .embedding,
        sizeBytes: 330938640,
        source: "Google / ONNX Community",
        supportsVision: false,
        supportsAudio: false,
        supportsThinking: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 1, recommendedRamGB: 2),
        contextWindowSize: 512,
        modelFormat: .onnx,
        additionalFiles: [
            "https://huggingface.co/onnx-community/embeddinggemma-300m-ONNX/resolve/main/onnx/model_quantized.onnx_data?download=true",
            "https://huggingface.co/onnx-community/embeddinggemma-300m-ONNX/resolve/main/tokenizer.json?download=true",
            "https://huggingface.co/onnx-community/embeddinggemma-300m-ONNX/resolve/main/tokenizer_config.json?download=true"
        ]
    ),
    AIModel(
        id: "embeddinggemma-300m-onnx-fp16",
        name: "EmbeddingGemma 300M (FP16)",
        description: "Google EmbeddingGemma 300M FP16 ONNX. High-quality semantic embeddings for RAG and memory. Best accuracy. (640 MB)",
        url: "https://huggingface.co/onnx-community/embeddinggemma-300m-ONNX/resolve/main/onnx/model_fp16.onnx?download=true",
        category: .embedding,
        sizeBytes: 639569517,
        source: "Google / ONNX Community",
        supportsVision: false,
        supportsAudio: false,
        supportsThinking: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 2, recommendedRamGB: 3),
        contextWindowSize: 512,
        modelFormat: .onnx,
        additionalFiles: [
            "https://huggingface.co/onnx-community/embeddinggemma-300m-ONNX/resolve/main/onnx/model_fp16.onnx_data?download=true",
            "https://huggingface.co/onnx-community/embeddinggemma-300m-ONNX/resolve/main/tokenizer.json?download=true",
            "https://huggingface.co/onnx-community/embeddinggemma-300m-ONNX/resolve/main/tokenizer_config.json?download=true"
        ]
    ),
    AIModel(
        id: "svd_i2v_xt_1.0_q6p_q8p.ckpt",
        name: "Stable Video Diffusion (XT)",
        description: "High quality image-to-video model. Requires 12GB RAM for stable performance on iOS. (2.8 GB)",
        url: "https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt/resolve/main/svd_xt.safetensors?download=true",
        category: .videoGeneration,
        sizeBytes: 2_765_619_200,
        source: "Stability AI",
        supportsVision: false,
        supportsAudio: false,
        supportsThinking: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 10, recommendedRamGB: 12),
        contextWindowSize: 512,
        modelFormat: .drawthings
    ),
    AIModel(
        id: "svd_i2v_xt_1.1_q6p_q8p.ckpt",
        name: "Stable Video Diffusion (XT v1.1)",
        description: "Updated image-to-video checkpoint from Draw Things. Requires 12GB RAM for stable performance on iOS.",
        url: "https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt",
        category: .videoGeneration,
        sizeBytes: 2_765_619_200,
        source: "Stability AI / Draw Things",
        supportsVision: false,
        supportsAudio: false,
        supportsThinking: false,
        supportsGpu: true,
        requirements: ModelRequirements(minRamGB: 10, recommendedRamGB: 12),
        contextWindowSize: 512,
        modelFormat: .drawthings
    )

]
}
