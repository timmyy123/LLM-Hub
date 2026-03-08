import Foundation

public struct DownloadUpdate: Sendable {
    public let bytesDownloaded: Int64
    public let totalBytes: Int64
    public let speedBytesPerSecond: Double
}

public actor ModelDownloader {
    public static let shared = ModelDownloader()
    
    private let urlSession: URLSession
    
    private init() {
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 60
        config.timeoutIntervalForResource = 3600 // 1 hour for large models
        self.urlSession = URLSession(configuration: config)
    }
    
    public func downloadModel(
        _ model: AIModel,
        hfToken: String?,
        destinationDir: URL,
        onProgress: @Sendable @escaping (DownloadUpdate) -> Void
    ) async throws {
        let totalSize = model.sizeBytes
        var downloadedBytesPerFile: [String: Int64] = [:]
        let startTime = Date()
        
        try FileManager.default.createDirectory(at: destinationDir, withIntermediateDirectories: true)
        
        for fileName in model.files {
            guard let fileURL = URL(string: "https://huggingface.co/\(model.repoId)/resolve/main/\(fileName)") else { continue }
            let destinationFileURL = destinationDir.appendingPathComponent(fileName)
            
            // Check if file already exists
            if FileManager.default.fileExists(atPath: destinationFileURL.path) {
                let attributes = try FileManager.default.attributesOfItem(atPath: destinationFileURL.path)
                let fileSize = attributes[.size] as? Int64 ?? 0
                if fileSize > 0 {
                    downloadedBytesPerFile[fileName] = fileSize
                    let currentTotal = downloadedBytesPerFile.values.reduce(0, +)
                    let elapsed = Date().timeIntervalSince(startTime)
                    let speed = elapsed > 0 ? Double(currentTotal) / elapsed : 0
                    onProgress(DownloadUpdate(bytesDownloaded: currentTotal, totalBytes: totalSize, speedBytesPerSecond: speed))
                    continue
                }
            }
            
            // Real-time download
            var request = URLRequest(url: fileURL)
            if let token = hfToken, !token.isEmpty {
                request.addValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
            }
            
            let (bytes, response) = try await urlSession.bytes(for: request)
            guard let httpResponse = response as? HTTPURLResponse, (200...299).contains(httpResponse.statusCode) else {
                throw NSError(domain: "ModelDownloader", code: 1, userInfo: [NSLocalizedDescriptionKey: "Download failed"])
            }
            
            // Create empty file
            if FileManager.default.fileExists(atPath: destinationFileURL.path) {
                try FileManager.default.removeItem(at: destinationFileURL)
            }
            FileManager.default.createFile(atPath: destinationFileURL.path, contents: nil)
            let fileHandle = try FileHandle(forWritingTo: destinationFileURL)
            defer { try? fileHandle.close() }
            
            var byteCount: Int64 = 0
            for try await byte in bytes {
                try fileHandle.write(contentsOf: [byte])
                byteCount += 1
                
                // Update progress every 1MB
                if byteCount % (1024 * 1024) == 0 {
                    downloadedBytesPerFile[fileName] = byteCount
                    let currentTotal = downloadedBytesPerFile.values.reduce(0, +)
                    let elapsed = Date().timeIntervalSince(startTime)
                    let speed = elapsed > 0 ? Double(currentTotal) / elapsed : 0
                    onProgress(DownloadUpdate(bytesDownloaded: currentTotal, totalBytes: totalSize, speedBytesPerSecond: speed))
                }
            }
            
            downloadedBytesPerFile[fileName] = byteCount
            let currentTotal = downloadedBytesPerFile.values.reduce(0, +)
            let elapsed = Date().timeIntervalSince(startTime)
            let speed = elapsed > 0 ? Double(currentTotal) / elapsed : 0
            onProgress(DownloadUpdate(bytesDownloaded: currentTotal, totalBytes: totalSize, speedBytesPerSecond: speed))
        }
    }
}
