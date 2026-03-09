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
        
        // Ensure clean destination
        if !FileManager.default.fileExists(atPath: destinationDir.path) {
            try FileManager.default.createDirectory(at: destinationDir, withIntermediateDirectories: true)
        }
        
        for fileName in model.files {
            // Encode repoId and fileName separately to avoid corrupting URL structure
            // Use urlPathAllowed but carefully since repoId has a slash
            let encodedRepoId = model.repoId.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? model.repoId
            let encodedFileName = fileName.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? fileName
            let urlString = "https://huggingface.co/\(encodedRepoId)/resolve/main/\(encodedFileName)"
            
            guard let fileURL = URL(string: urlString) else { continue }
            
            let destinationFileURL = destinationDir.appendingPathComponent(fileName)
            
            // Check if file exists and is already downloaded fully
            if FileManager.default.fileExists(atPath: destinationFileURL.path) {
                if let attrs = try? FileManager.default.attributesOfItem(atPath: destinationFileURL.path),
                   let fileSize = attrs[.size] as? Int64,
                   fileSize > 0 {
                    downloadedBytesPerFile[fileName] = fileSize
                    let currentTotal = downloadedBytesPerFile.values.reduce(0, +)
                    let elapsed = Date().timeIntervalSince(startTime)
                    let speed = elapsed > 0 ? Double(currentTotal) / elapsed : 0
                    onProgress(DownloadUpdate(bytesDownloaded: currentTotal, totalBytes: totalSize, speedBytesPerSecond: speed))
                    continue
                }
            }
            
            var request = URLRequest(url: fileURL, cachePolicy: .reloadIgnoringLocalCacheData)
            if let token = hfToken, !token.isEmpty {
                request.addValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
            }
            
            let (bytes, response) = try await urlSession.bytes(for: request)
            guard let httpResponse = response as? HTTPURLResponse else {
                throw NSError(domain: "ModelDownloader", code: -1, userInfo: [NSLocalizedDescriptionKey: "No Response"])
            }
            
            // Critical 404/403 Handling
            if !(200...299).contains(httpResponse.statusCode) {
                // Ignore missing optional files (like chat_template.jinja in some older MLX repos)
                if httpResponse.statusCode == 404 && fileName == "chat_template.jinja" {
                    continue
                }
                
                let reason = HTTPURLResponse.localizedString(forStatusCode: httpResponse.statusCode)
                throw NSError(domain: "ModelDownloader", code: httpResponse.statusCode, userInfo: [NSLocalizedDescriptionKey: "HTTP \(httpResponse.statusCode): \(reason)"])
            }
            
            // Efficient Buffered Write
            if FileManager.default.fileExists(atPath: destinationFileURL.path) {
                try? FileManager.default.removeItem(at: destinationFileURL)
            }
            FileManager.default.createFile(atPath: destinationFileURL.path, contents: nil)
            let fileHandle = try FileHandle(forWritingTo: destinationFileURL)
            defer { try? fileHandle.close() }
            
            var byteCountPerFile: Int64 = 0
            var buffer = Data()
            let chunkSize = 64 * 1024 // 64KB buffer
            
            for try await byte in bytes {
                buffer.append(byte)
                byteCountPerFile += 1
                
                if buffer.count >= chunkSize {
                    try fileHandle.write(contentsOf: buffer)
                    buffer.removeAll(keepingCapacity: true)
                    
                    // Periodic Progress Update
                    downloadedBytesPerFile[fileName] = byteCountPerFile
                    let currentTotal = downloadedBytesPerFile.values.reduce(0, +)
                    let elapsed = Date().timeIntervalSince(startTime)
                    let speed = elapsed > 0 ? Double(currentTotal) / elapsed : 0
                    onProgress(DownloadUpdate(bytesDownloaded: currentTotal, totalBytes: totalSize, speedBytesPerSecond: speed))
                }
            }
            
            if !buffer.isEmpty {
                try fileHandle.write(contentsOf: buffer)
                buffer.removeAll()
            }
            
            downloadedBytesPerFile[fileName] = byteCountPerFile
            let currentTotal = downloadedBytesPerFile.values.reduce(0, +)
            let elapsed = Date().timeIntervalSince(startTime)
            let speed = elapsed > 0 ? Double(currentTotal) / elapsed : 0
            onProgress(DownloadUpdate(bytesDownloaded: currentTotal, totalBytes: totalSize, speedBytesPerSecond: speed))
        }
    }
}
