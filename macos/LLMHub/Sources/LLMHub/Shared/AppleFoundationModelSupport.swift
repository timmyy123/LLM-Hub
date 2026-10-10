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

/// The caller already supplies conversation history in `prompt`. A new session
/// prevents duplicate turns and carries the current model's instructions verbatim.
@MainActor
func generateAppleFoundationResponse(
    prompt: String,
    systemPrompt: String?,
    temperature: Double,
    maxTokens: Int,
    onUpdate: @escaping (String, Int, Double) -> Void
) async throws {
    #if canImport(FoundationModels)
    if #available(iOS 26.0, *) {
        try Task.checkCancellation()
        let model = SystemLanguageModel.default
        guard model.isAvailable else {
            throw NSError(domain: "AppleFoundationModel", code: 1, userInfo: [
                NSLocalizedDescriptionKey: "Apple Intelligence is unavailable. Check Apple Intelligence settings and model download status."
            ])
        }
        let session = LanguageModelSession(model: model, instructions: systemPrompt)
        let options = GenerationOptions(
            temperature: temperature,
            maximumResponseTokens: max(1, maxTokens)
        )
        let stream = session.streamResponse(to: prompt, options: options)
        for try await snapshot in stream {
            try Task.checkCancellation()
            // Apple streams cumulative text. Do not append snapshots or invent token counts.
            onUpdate(snapshot.content, 0, 0)
        }
        try Task.checkCancellation()
        return
    }
    #endif
    throw NSError(domain: "AppleFoundationModel", code: 2, userInfo: [
        NSLocalizedDescriptionKey: "Apple Foundation Models requires iOS 26 or later."
    ])
}
