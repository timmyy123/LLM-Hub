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
        let documentsDirectory = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first!
        modelsDirectory = documentsDirectory.appendingPathComponent("models", isDirectory: true)
        
        if !fileManager.fileExists(atPath: modelsDirectory.path) {
            try? fileManager.createDirectory(at: modelsDirectory, withIntermediateDirectories: true)
        }
        
        refreshStatuses()
    }
    
    public func refreshStatuses() {
        for model in ModelData.models {
            let modelDir = modelsDirectory.appendingPathComponent(model.id)
            if FileManager.default.fileExists(atPath: modelDir.path) {
                // simple check: if the main model file exists, assume downloaded
                // For MLX, we should check if model.safetensors or similar exists
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
}
