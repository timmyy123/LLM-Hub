import Foundation

#if canImport(FoundationModels)
import FoundationModels
#endif

let appleFoundationModelId = "apple.foundation.system"

func isAppleFoundationModel(_ model: AIModel?) -> Bool {
    model?.id == appleFoundationModelId
}

@MainActor
func appleFoundationModelIfAvailable() -> AIModel? {
    #if canImport(FoundationModels)
    if #available(iOS 26.0, *) {
        let model = SystemLanguageModel.default
        guard model.isAvailable else { return nil }

        return AIModel(
            id: appleFoundationModelId,
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
