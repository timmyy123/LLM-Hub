import Foundation

public enum InferenceFramework: String, Sendable {
    case onnx = "ONNX"
    case llamaCpp = "LlamaCpp"
    case foundationModels = "FoundationModels"
}

final class SimplifiedFileManager: @unchecked Sendable {
    static let shared = SimplifiedFileManager()

    private init() {}

    func getModelFolderURL(modelId: String, framework: InferenceFramework) throws -> URL {
        guard !modelId.isEmpty, modelId != ".", modelId != "..",
              !modelId.contains("/"), !modelId.contains("\\"), !modelId.contains("\0") else {
            throw CocoaError(.fileReadInvalidFileName)
        }

        let documentsDir = try FileManager.default.url(
            for: .documentDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        )
        // Preserve the installed-model location used by earlier app versions.
        let url = documentsDir.appendingPathComponent("RunAnywhere/Models", isDirectory: true)
            .appendingPathComponent(framework.rawValue, isDirectory: true)
            .appendingPathComponent(modelId, isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }
}
