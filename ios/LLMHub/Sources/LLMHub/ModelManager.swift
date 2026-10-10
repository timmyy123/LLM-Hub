import Foundation
import Combine

public enum DownloadProgress: Sendable {
    case notDownloaded
    case downloading(progress: Double, downloadedLabel: String, speedLabel: String)
    case paused(progress: Double, downloadedLabel: String)
    case downloaded
}

@MainActor
public class ModelManager: ObservableObject {
    public static let shared = ModelManager()
    
    @Published public var modelStatuses: [String: DownloadProgress] = [:]
    
    private let modelsDirectory: URL
    private var downloadTasks: [String: URLSessionDownloadTask] = [:]
    
    private init() {
        let fileManager = FileManager.default
        if let documentsDirectory = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first {
            modelsDirectory = documentsDirectory.appendingPathComponent("models", isDirectory: true)
            
            if !fileManager.fileExists(atPath: modelsDirectory.path) {
                try? fileManager.createDirectory(at: modelsDirectory, withIntermediateDirectories: true)
            }
        } else {
            // Fallback to a temp directory or handle error
            modelsDirectory = fileManager.temporaryDirectory.appendingPathComponent("models")
        }
        
        refreshStatuses()
    }
    
    public func refreshStatuses() {
        for model in ModelData.allModels() {
            let modelDir = modelsDirectory.appendingPathComponent(model.id)
            if FileManager.default.fileExists(atPath: modelDir.path) {
                // Simple check: if a known model artifact exists, assume downloaded.
                let weightsFile = modelDir.appendingPathComponent("model.safetensors")
                if FileManager.default.fileExists(atPath: weightsFile.path) {
                    modelStatuses[model.id] = .downloaded
                } else {
                    modelStatuses[model.id] = .notDownloaded
                }
            } else {
                modelStatuses[model.id] = .notDownloaded
            }
        }
    }
    
    public func downloadModel(_ model: AIModel, hfToken: String?) async {
        // Implementation for downloading
        // We'll use ModelDownloader for the actual work
    }
    
    public func deleteModel(_ model: AIModel) {
        let modelDir = modelsDirectory.appendingPathComponent(model.id)
        try? FileManager.default.removeItem(at: modelDir)
        modelStatuses[model.id] = .notDownloaded
    }

    public func isDownloaded(_ model: AIModel) -> Bool {
        return ModelData.isModelFullyAvailableLocally(model)
    }

    public func isDownloading(_ model: AIModel) -> Bool {
        if case .downloading = modelStatuses[model.id] {
            return true
        }
        return false
    }

    public func progress(for model: AIModel) -> Double {
        if case .downloading(let p, _, _) = modelStatuses[model.id] {
            return p
        }
        return 0.0
    }

    public func statusText(for model: AIModel) -> String {
        if case .downloading(_, let downloaded, let speed) = modelStatuses[model.id] {
            return "\(downloaded) (\(speed))"
        }
        return ""
    }

    public func cancel(_ model: AIModel) {
        downloadTasks[model.id]?.cancel()
        downloadTasks.removeValue(forKey: model.id)
        modelStatuses[model.id] = .notDownloaded
    }

    public func startDownload(_ model: AIModel) {
        Task {
            await downloadModel(model, hfToken: nil)
        }
    }

    public var modelsDirectoryURL: URL? {
        return modelsDirectory
    }

    public var downloadedModels: [AIModel] {
        return ModelData.allModels().filter { isDownloaded($0) }
    }

    public func fileSize(for model: AIModel) -> Int64? {
        return model.sizeBytes
    }

    public func importCustomModel(from url: URL) {
        let dest = modelsDirectory.appendingPathComponent(url.lastPathComponent)
        try? FileManager.default.copyItem(at: url, to: dest)
        refreshStatuses()
    }

    public func localURL(for model: AIModel) -> URL? {
        return modelsDirectory.appendingPathComponent(model.id)
    }
}

public struct AIModelQuantization: RawRepresentable, Sendable {
    public let rawValue: String
    public init(rawValue: String) { self.rawValue = rawValue }
}

extension AIModel {
    public var formattedSize: String {
        ByteCountFormatter.string(fromByteCount: sizeBytes, countStyle: .file)
    }

    public var requiredRAM: String? {
        "\(requirements.recommendedRamGB)GB"
    }

    public var quantization: AIModelQuantization? {
        return AIModelQuantization(rawValue: "Q4_K_M")
    }

    public var tags: [String] {
        var result: [String] = []
        if supportsVision { result.append("vision") }
        if supportsAudio { result.append("audio") }
        if supportsThinking { result.append("thinking") }
        if supportsGpu { result.append("gpu") }
        result.append(category.rawValue.lowercased())
        return result
    }
}
