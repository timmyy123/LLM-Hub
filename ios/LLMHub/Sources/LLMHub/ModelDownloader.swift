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
        var downloadedSoFar: Int64 = 0
        let startTime = Date()
        
        // Ensure destination exists
        try FileManager.default.createDirectory(at: destinationDir, withIntermediateDirectories: true)
        
        for fileName in model.files {
            let fileURL = URL(string: "https://huggingface.co/\(model.repoId)/resolve/main/\(fileName)")!
            let destinationFileURL = destinationDir.appendingPathComponent(fileName)
            
            // Check if file already exists and is complete
            if FileManager.default.fileExists(atPath: destinationFileURL.path) {
                let attributes = try FileManager.default.attributesOfItem(atPath: destinationFileURL.path)
                let fileSize = attributes[.size] as? Int64 ?? 0
                
                // This is a simple check. A better one would be to compare with remote content-length.
                // But for now, if it's there, we skip it to save time in this demo/context.
                if fileSize > 0 {
                    downloadedSoFar += fileSize
                    continue
                }
            }
            
            // Download the file
            try await downloadFile(
                from: fileURL,
                to: destinationFileURL,
                hfToken: hfToken
            )
            
            // Report progress after each file
            let attributes = try FileManager.default.attributesOfItem(atPath: destinationFileURL.path)
            let fileSize = attributes[.size] as? Int64 ?? 0
            downloadedSoFar += fileSize
            
            let elapsed = Date().timeIntervalSince(startTime)
            let speed = elapsed > 0 ? Double(downloadedSoFar) / elapsed : 0
            onProgress(DownloadUpdate(
                bytesDownloaded: downloadedSoFar,
                totalBytes: totalSize,
                speedBytesPerSecond: speed
            ))
        }
    }
    
    private func downloadFile(
        from url: URL,
        to destination: URL,
        hfToken: String?
    ) async throws {
        var request = URLRequest(url: url)
        if let token = hfToken, !token.isEmpty {
            request.addValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
        
        // Note: URLSession.download(for:delegate:) is better for backgrounding,
        // but for a simple reactive UI we can use bytes(for:delegate:) or similar.
        // However, and for large files, we should use a proper delegate-based download.
        
        let (localURL, response) = try await urlSession.download(for: request)
        
        guard let httpResponse = response as? HTTPURLResponse, (200...299).contains(httpResponse.statusCode) else {
            throw NSError(domain: "ModelDownloader", code: 1, userInfo: [NSLocalizedDescriptionKey: "Download failed with status code \((response as? HTTPURLResponse)?.statusCode ?? 0)"])
        }
        
        // Move the file to destination
        if FileManager.default.fileExists(atPath: destination.path) {
            try FileManager.default.removeItem(at: destination)
        }
        try FileManager.default.moveItem(at: localURL, to: destination)
    }
}
