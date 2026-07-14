import Foundation
import RunAnywhere

final class SimplifiedFileManager {
    static let shared = SimplifiedFileManager()

    private init() {}

    func getModelFolderURL(modelId: String, framework: InferenceFramework) throws -> URL {
        if let url = try? CppBridge.ModelPaths.getModelFolder(modelId: modelId, framework: framework) {
            try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
            return url
        }

        let documentsDir = try FileManager.default.url(
            for: .documentDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        )
        let url = documentsDir.appendingPathComponent("models", isDirectory: true)
            .appendingPathComponent(modelId, isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }
}
