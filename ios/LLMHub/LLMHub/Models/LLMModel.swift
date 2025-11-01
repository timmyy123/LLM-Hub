//
//  LLMModel.swift
//  LLMHub
//
//  Data model for LLM models
//

import Foundation

struct ModelRequirements: Codable {
    let minRamGB: Int
    let recommendedRamGB: Int
}

struct LLMModel: Identifiable, Codable {
    let id: String
    let name: String
    let description: String
    let url: String
    let category: String
    let sizeBytes: Int64
    let source: String
    let supportsVision: Bool
    let supportsAudio: Bool
    let supportsGpu: Bool
    let requirements: ModelRequirements
    let contextWindowSize: Int
    let modelFormat: String
    
    var isDownloaded: Bool = false
    var isDownloading: Bool = false
    var downloadProgress: Float = 0.0
    
    init(id: String = UUID().uuidString,
         name: String,
         description: String,
         url: String,
         category: String,
         sizeBytes: Int64,
         source: String,
         supportsVision: Bool = false,
         supportsAudio: Bool = false,
         supportsGpu: Bool = false,
         requirements: ModelRequirements,
         contextWindowSize: Int = 2048,
         modelFormat: String = "task") {
        self.id = id
        self.name = name
        self.description = description
        self.url = url
        self.category = category
        self.sizeBytes = sizeBytes
        self.source = source
        self.supportsVision = supportsVision
        self.supportsAudio = supportsAudio
        self.supportsGpu = supportsGpu
        self.requirements = requirements
        self.contextWindowSize = contextWindowSize
        self.modelFormat = modelFormat
    }
    
    var localFileName: String {
        return url.components(separatedBy: "/").last ?? "\(name).task"
    }
    
    var localFilePath: URL? {
        let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first
        return documentsPath?.appendingPathComponent("models").appendingPathComponent(localFileName)
    }
}

// Sample models similar to Android version
extension LLMModel {
    static let sampleModels: [LLMModel] = [
        LLMModel(
            name: "Gemma-3 1B INT4 2k",
            description: "Compact model optimized for mobile devices with 2k context",
            url: "https://huggingface.co/google/gemma-3-1b-int4-2k/resolve/main/gemma-3-1b-int4-2k.task",
            category: "Text",
            sizeBytes: 700_000_000,
            source: "Google",
            requirements: ModelRequirements(minRamGB: 2, recommendedRamGB: 4),
            contextWindowSize: 2048
        ),
        LLMModel(
            name: "Gemma-3 1B INT8 2k",
            description: "Higher quality 1B model with 2k context window",
            url: "https://huggingface.co/google/gemma-3-1b-int8-2k/resolve/main/gemma-3-1b-int8-2k.task",
            category: "Text",
            sizeBytes: 1_200_000_000,
            source: "Google",
            supportsGpu: true,
            requirements: ModelRequirements(minRamGB: 4, recommendedRamGB: 6),
            contextWindowSize: 2048
        ),
        LLMModel(
            name: "Gemma-3n E2B 4k",
            description: "Multimodal model supporting text, images, and audio",
            url: "https://huggingface.co/google/gemma-3n-e2b-4k/resolve/main/gemma-3n-e2b-4k.task",
            category: "Multimodal",
            sizeBytes: 2_500_000_000,
            source: "Google",
            supportsVision: true,
            supportsAudio: true,
            supportsGpu: true,
            requirements: ModelRequirements(minRamGB: 6, recommendedRamGB: 8),
            contextWindowSize: 4096
        )
    ]
}
