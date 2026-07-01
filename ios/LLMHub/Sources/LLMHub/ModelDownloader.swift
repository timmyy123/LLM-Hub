import Foundation
import ZIPFoundation
@preconcurrency import MediaGenerationKit
import ModelZoo

private extension URLError.Code {
    var isTransientDownloadFailure: Bool {
        switch self {
        case .networkConnectionLost,
            .notConnectedToInternet,
            .timedOut,
            .cannotConnectToHost,
            .cannotFindHost,
            .dnsLookupFailed,
            .resourceUnavailable,
            .internationalRoamingOff,
            .callIsActive,
            .dataNotAllowed:
            return true
        default:
            return false
        }
    }
}

public struct DownloadUpdate: Sendable {
    public let bytesDownloaded: Int64
    public let totalBytes: Int64
    public let speedBytesPerSecond: Double
}

private struct ModelInstallMarker: Codable {
    let version: Int
    let modelId: String
    let totalBytes: Int64
    let fileNames: [String]
}

private final class ThroughputTracker: @unchecked Sendable {
    private let lock = NSLock()
    private var throughputSamples: [(time: Date, bytes: Int64)] = []
    private var lastBytesWritten: Int64 = 0
    private let window: TimeInterval = 3.0

    func recordTransfer(bytesWritten: Int64) -> Double {
        lock.lock()
        defer { lock.unlock() }
        let now = Date()
        let diff = bytesWritten - lastBytesWritten
        if diff > 0 {
            throughputSamples.append((time: now, bytes: diff))
            lastBytesWritten = bytesWritten
        } else if bytesWritten < lastBytesWritten {
            lastBytesWritten = bytesWritten
        }
        
        let cutoff = now.addingTimeInterval(-window)
        throughputSamples.removeAll { $0.time < cutoff }
        
        guard !throughputSamples.isEmpty else { return 0 }
        guard let firstTime = throughputSamples.first?.time else { return 0 }
        let bytes = throughputSamples.reduce(0) { $0 + $1.bytes }
        let span = max(0.1, now.timeIntervalSince(firstTime))
        return Double(bytes) / span
    }
}

public actor ModelDownloader {
    public static let shared = ModelDownloader()
    
    private let urlSession: URLSession
    private let completionThresholdRatio: Double = 0.98
    private let optionalModelFiles: Set<String> = []

    nonisolated static func installMarkerURL(for destinationDir: URL) -> URL {
        destinationDir.appendingPathComponent("_downloaded")
    }
    
    private init() {
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 60
        config.timeoutIntervalForResource = 3600 // 1 hour for large models
        self.urlSession = URLSession(configuration: config)
    }

    private func remoteFileSize(fileURL: URL, hfToken: String?) async -> Int64? {
        func authorizedRequest(method: String) -> URLRequest {
            var request = URLRequest(url: fileURL, cachePolicy: .reloadIgnoringLocalCacheData)
            request.httpMethod = method
            if let token = hfToken, !token.isEmpty {
                request.addValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
            }
            return request
        }

        // First try HEAD for content length.
        do {
            let request = authorizedRequest(method: "HEAD")
            let (_, response) = try await urlSession.data(for: request)
            if let httpResponse = response as? HTTPURLResponse,
               (200...299).contains(httpResponse.statusCode),
               let contentLength = httpResponse.value(forHTTPHeaderField: "Content-Length"),
               let size = Int64(contentLength),
               size > 0 {
                return size
            }
        } catch {
            // Fall through to range probe.
        }

        // Some endpoints block HEAD; probe with GET Range to parse total size from Content-Range.
        do {
            var request = authorizedRequest(method: "GET")
            request.addValue("bytes=0-0", forHTTPHeaderField: "Range")
            let (_, response) = try await urlSession.data(for: request)
            if let httpResponse = response as? HTTPURLResponse {
                if let total = totalSizeFromContentRange(httpResponse.value(forHTTPHeaderField: "Content-Range")), total > 0 {
                    return total
                }
                if let contentLength = httpResponse.value(forHTTPHeaderField: "Content-Length"),
                   let size = Int64(contentLength),
                   size > 0,
                   httpResponse.statusCode == 200 {
                    return size
                }
            }
        } catch {
            // Give up and treat as unknown size.
        }
        return nil
    }

    private func localFileSize(at url: URL) -> Int64 {
        guard let attrs = try? FileManager.default.attributesOfItem(atPath: url.path),
            let fileSize = attrs[.size] as? Int64
        else {
            return 0
        }
        return max(0, fileSize)
    }

    private func totalSizeFromContentRange(_ contentRange: String?) -> Int64? {
        guard let contentRange else { return nil }
        // 416 responses often include: "bytes */123456"
        guard let slash = contentRange.lastIndex(of: "/") else { return nil }
        let tail = contentRange[contentRange.index(after: slash)...].trimmingCharacters(in: .whitespaces)
        guard tail != "*", let value = Int64(tail), value > 0 else { return nil }
        return value
    }
    
    public func downloadModel(
        _ model: AIModel,
        hfToken: String?,
        destinationDir: URL,
        onProgress: @Sendable @escaping (DownloadUpdate) -> Void
    ) async throws {
        if model.modelFormat == .drawthings {
            // Upscaler models are standalone .ckpt files hosted at static.libnnc.org.
            // They are NOT in the Draw Things model catalog, so env.ensure() would fail
            // immediately with "unresolved model reference". Download them directly instead.
            if model.category == .imageUpscale {
                let destPath = ModelZoo.filePathForModelDownloaded(model.id)
                let destURL = URL(fileURLWithPath: destPath)
                let sourceURL = URL(string: model.url)!
                
                // Ensure the parent directory exists
                try FileManager.default.createDirectory(
                    at: destURL.deletingLastPathComponent(),
                    withIntermediateDirectories: true
                )
                
                // Check if already fully downloaded
                if let attrs = try? FileManager.default.attributesOfItem(atPath: destPath),
                   let existingSize = attrs[.size] as? Int64,
                   existingSize == model.sizeBytes {
                    onProgress(DownloadUpdate(bytesDownloaded: model.sizeBytes, totalBytes: model.sizeBytes, speedBytesPerSecond: 0))
                    return
                }
                
                let tracker = ThroughputTracker()
                let maxRetries = 6
                var attempt = 0
                var finished = false
                var restartedAfter416 = false
                
                while !finished {
                    do {
                        var existingBytes = localFileSize(at: destURL)
                        var request = URLRequest(url: sourceURL, cachePolicy: .reloadIgnoringLocalCacheData)
                        if existingBytes > 0 {
                            request.addValue("bytes=\(existingBytes)-", forHTTPHeaderField: "Range")
                        }
                        
                        let (bytes, response) = try await urlSession.bytes(for: request)
                        guard let httpResponse = response as? HTTPURLResponse else {
                            throw NSError(domain: "ModelDownloader", code: -1, userInfo: [NSLocalizedDescriptionKey: "No HTTP response"])
                        }
                        
                        if !(200...299).contains(httpResponse.statusCode) {
                            if httpResponse.statusCode == 416 {
                                if !restartedAfter416 {
                                    try? FileManager.default.removeItem(at: destURL)
                                    restartedAfter416 = true
                                    continue
                                }
                            }
                            let reason = HTTPURLResponse.localizedString(forStatusCode: httpResponse.statusCode)
                            throw NSError(domain: "ModelDownloader", code: httpResponse.statusCode, userInfo: [NSLocalizedDescriptionKey: "HTTP \(httpResponse.statusCode): \(reason)"])
                        }
                        
                        if !FileManager.default.fileExists(atPath: destPath) {
                            FileManager.default.createFile(atPath: destPath, contents: nil)
                        }
                        if existingBytes > 0 && httpResponse.statusCode == 200 {
                            try? FileManager.default.removeItem(at: destURL)
                            FileManager.default.createFile(atPath: destPath, contents: nil)
                            existingBytes = 0
                        }
                        
                        let fileHandle = try FileHandle(forWritingTo: destURL)
                        defer { try? fileHandle.close() }
                        if existingBytes > 0 {
                            try fileHandle.seekToEnd()
                        } else {
                            try fileHandle.truncate(atOffset: 0)
                        }
                        
                        var byteCount: Int64 = existingBytes
                        var buffer = Data()
                        let chunkSize = 64 * 1024
                        
                        for try await byte in bytes {
                            buffer.append(byte)
                            byteCount += 1
                            if buffer.count >= chunkSize {
                                let flushed = Int64(buffer.count)
                                try fileHandle.write(contentsOf: buffer)
                                buffer.removeAll(keepingCapacity: true)
                                let speed = tracker.recordTransfer(bytesWritten: flushed)
                                onProgress(DownloadUpdate(bytesDownloaded: byteCount, totalBytes: model.sizeBytes, speedBytesPerSecond: speed))
                            }
                        }
                        if !buffer.isEmpty {
                            let flushed = Int64(buffer.count)
                            try fileHandle.write(contentsOf: buffer)
                            let speed = tracker.recordTransfer(bytesWritten: flushed)
                            onProgress(DownloadUpdate(bytesDownloaded: byteCount, totalBytes: model.sizeBytes, speedBytesPerSecond: speed))
                        }
                        finished = true
                    } catch {
                        attempt += 1
                        let urlErr = error as? URLError
                        if attempt >= maxRetries || !(urlErr?.code.isTransientDownloadFailure ?? false) {
                            throw error
                        }
                        let delay = min(pow(2.0, Double(attempt)), 30.0)
                        try await Task.sleep(nanoseconds: UInt64(delay * 1_000_000_000))
                    }
                }
                return
            }
            
            // Image/video generation drawthings models: resolved through env.ensure which
            // handles catalog lookup + dependency downloading for multi-file models.
            struct EnvWrapper {
                static let env = MediaGenerationEnvironment.default
            }
            let tracker = ThroughputTracker()
            let drawThingsFileSizes: [String: Int64] = [
                "svd_i2v_xt_1.0_q6p_q8p.ckpt": 1334681600,
                "svd_i2v_xt_1.1_q6p_q8p.ckpt": 1525776384,
                "svd_i2v_1.0_q6p_q8p.ckpt": 1334681600,
                "animatelcm_svd_xt_v1.1_q6p_q8p.ckpt": 1376313344,
                "open_clip_vit_h14_vision_model_f16.ckpt": 1263398912,
                "open_clip_vit_h14_visual_proj_f16.ckpt": 2633728,
                "vae_ft_mse_840000_f16.ckpt": 167538688,
                "sd_v1.5_f16.ckpt": 1721266176,
                "clip_vit_l14_f16.ckpt": 246517760,
                "sd_v2.1_f16.ckpt": 1734078464,
                "open_clip_vit_h14_f16.ckpt": 682082304,
                "sd_xl_base_1.0_q6p_q8p.ckpt": 2099683328,
                "open_clip_vit_bigg14_f16.ckpt": 1391304704,
                "sdxl_vae_v1.0_f16.ckpt": 167534592,
                "flux_1_schnell_q5p.ckpt": 9349296128,
                "flux_1_vae_f16.ckpt": 167870464,
                "t5_xxl_encoder_q6p.ckpt": 3884990464,
                "realesrgan_x4plus_f16.ckpt": 33697792,
                "realesrgan_x2plus_f16.ckpt": 33710080,
                "realesrgan_x4plus_anime_6b_f16.ckpt": 9027584,
                "esrgan_4x_universal_upscaler_v2_sharp_f16.ckpt": 33697792,
                "remacri_4x_f16.ckpt": 33697792,
                "4x_ultrasharp_f16.ckpt": 33697792
            ]
            let allFiles: [String]
            if let specification = ModelZoo.specificationForModel(model.id) {
                var seen = Set<String>()
                allFiles = ModelZoo.filesToDownload(specification).map(\.file).filter { seen.insert($0).inserted }
            } else {
                allFiles = [model.id]
            }
            
            _ = try await EnvWrapper.env.ensure(model.id, offline: false) { state in
                switch state {
                case .resolving:
                    onProgress(DownloadUpdate(bytesDownloaded: 0, totalBytes: model.sizeBytes, speedBytesPerSecond: 0))
                case .verifying:
                    var cumulativeBytes: Int64 = 0
                    for f in allFiles {
                        let path = ModelZoo.filePathForModelDownloaded(f)
                        if FileManager.default.fileExists(atPath: path) {
                            let attrs = try? FileManager.default.attributesOfItem(atPath: path)
                            let localSize = attrs?[.size] as? Int64 ?? 0
                            if localSize > 0 {
                                cumulativeBytes += localSize
                            } else if let knownSize = drawThingsFileSizes[f] {
                                cumulativeBytes += knownSize
                            }
                        }
                    }
                    let trueTotalSize = allFiles.reduce(Int64(0)) { sum, f in
                        sum + (drawThingsFileSizes[f] ?? 0)
                    }
                    let finalTotalSize = trueTotalSize > 0 ? trueTotalSize : model.sizeBytes
                    let progressProportion = Double(cumulativeBytes) / Double(max(1, finalTotalSize))
                    let reportedBytes = min(model.sizeBytes, Int64(progressProportion * Double(model.sizeBytes)))
                    onProgress(DownloadUpdate(bytesDownloaded: reportedBytes, totalBytes: model.sizeBytes, speedBytesPerSecond: 0))
                case .downloading(let file, _, _, let bytesWritten, _):
                    let speed = tracker.recordTransfer(bytesWritten: bytesWritten)
                    
                    var cumulativeBytes: Int64 = 0
                    for f in allFiles {
                        if f == file {
                            cumulativeBytes += bytesWritten
                        } else {
                            let path = ModelZoo.filePathForModelDownloaded(f)
                            if FileManager.default.fileExists(atPath: path) {
                                let attrs = try? FileManager.default.attributesOfItem(atPath: path)
                                let localSize = attrs?[.size] as? Int64 ?? 0
                                if localSize > 0 {
                                    cumulativeBytes += localSize
                                } else if let knownSize = drawThingsFileSizes[f] {
                                    cumulativeBytes += knownSize
                                }
                            }
                        }
                    }
                    
                    let trueTotalSize = allFiles.reduce(Int64(0)) { sum, f in
                        sum + (drawThingsFileSizes[f] ?? 0)
                    }
                    let finalTotalSize = trueTotalSize > 0 ? trueTotalSize : model.sizeBytes
                    let progressProportion = Double(cumulativeBytes) / Double(max(1, finalTotalSize))
                    let reportedBytes = min(model.sizeBytes, Int64(progressProportion * Double(model.sizeBytes)))
                    
                    onProgress(DownloadUpdate(
                        bytesDownloaded: reportedBytes,
                        totalBytes: model.sizeBytes,
                        speedBytesPerSecond: speed
                    ))
                }
            }
            return
        }


        // CoreML models (Stable Diffusion) are distributed as ZIP archives;
        // download, extract, and store a sentinel to mark completion.
        if model.modelFormat == .coreml {
            try await downloadAndExtractCoreML(
                model: model,
                hfToken: hfToken,
                destinationDir: destinationDir,
                onProgress: onProgress
            )
            return
        }
        let totalSize = model.sizeBytes
        var downloadedBytesPerFile: [String: Int64] = [:]
        var expectedBytesPerFile: [String: Int64] = [:]
        let realtimeWindowSeconds: TimeInterval = 3.0
        var throughputSamples: [(time: Date, bytes: Int64)] = []

        func recordTransfer(_ bytes: Int64) {
            guard bytes > 0 else { return }
            let now = Date()
            throughputSamples.append((time: now, bytes: bytes))
            let cutoff = now.addingTimeInterval(-realtimeWindowSeconds)
            throughputSamples.removeAll { $0.time < cutoff }
        }

        func realtimeSpeed() -> Double {
            guard !throughputSamples.isEmpty else { return 0 }
            let now = Date()
            let cutoff = now.addingTimeInterval(-realtimeWindowSeconds)
            throughputSamples.removeAll { $0.time < cutoff }
            guard let firstTime = throughputSamples.first?.time else { return 0 }
            let bytes = throughputSamples.reduce(Int64(0)) { $0 + $1.bytes }
            let span = max(0.1, now.timeIntervalSince(firstTime))
            return Double(bytes) / span
        }
        
        // Ensure clean destination
        if !FileManager.default.fileExists(atPath: destinationDir.path) {
            try FileManager.default.createDirectory(at: destinationDir, withIntermediateDirectories: true)
        }

        let markerURL = Self.installMarkerURL(for: destinationDir)
        try? FileManager.default.removeItem(at: markerURL)
        
        let downloadItems = Array(zip(model.requiredFileNames, model.allDownloadURLs))

        for (fileName, fileURL) in downloadItems {
            
            let destinationFileURL = destinationDir.appendingPathComponent(fileName)
            let expectedSize = await remoteFileSize(fileURL: fileURL, hfToken: hfToken)
            if let expectedSize {
                expectedBytesPerFile[fileName] = expectedSize
            }
            
            // Check if file exists and is already downloaded fully.
            // We only skip when local size matches remote Content-Length.
            if FileManager.default.fileExists(atPath: destinationFileURL.path) {
                if let attrs = try? FileManager.default.attributesOfItem(atPath: destinationFileURL.path),
                   let fileSize = attrs[.size] as? Int64,
                   fileSize > 0 {
                    if let expectedSize, expectedSize == fileSize {
                        downloadedBytesPerFile[fileName] = fileSize
                        let currentTotal = downloadedBytesPerFile.values.reduce(0, +)
                        let speed = realtimeSpeed()
                        onProgress(DownloadUpdate(bytesDownloaded: currentTotal, totalBytes: totalSize, speedBytesPerSecond: speed))
                        continue
                    }
                }
            }
            
            let maxRetries = 6
            var attempt = 0
            var finishedFile = false
            var restartedAfter416 = false

            while !finishedFile {
                do {
                    var existingBytes = localFileSize(at: destinationFileURL)
                    var request = URLRequest(url: fileURL, cachePolicy: .reloadIgnoringLocalCacheData)
                    if let token = hfToken, !token.isEmpty {
                        request.addValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
                    }
                    if existingBytes > 0 {
                        request.addValue("bytes=\(existingBytes)-", forHTTPHeaderField: "Range")
                    }

                    let (bytes, response) = try await urlSession.bytes(for: request)
                    guard let httpResponse = response as? HTTPURLResponse else {
                        throw NSError(domain: "ModelDownloader", code: -1, userInfo: [NSLocalizedDescriptionKey: "No Response"])
                    }

                    // Critical 404/403 Handling
                    if !(200...299).contains(httpResponse.statusCode) {
                        if httpResponse.statusCode == 416 {
                            let rangeHeaderTotal = totalSizeFromContentRange(httpResponse.value(forHTTPHeaderField: "Content-Range"))
                            let refreshedExpected: Int64?
                            if let expectedSize {
                                refreshedExpected = expectedSize
                            } else if let rangeHeaderTotal {
                                refreshedExpected = rangeHeaderTotal
                            } else {
                                refreshedExpected = await remoteFileSize(fileURL: fileURL, hfToken: hfToken)
                            }
                            if let refreshedExpected, existingBytes >= refreshedExpected {
                                downloadedBytesPerFile[fileName] = refreshedExpected
                                let currentTotal = downloadedBytesPerFile.values.reduce(0, +)
                                let speed = realtimeSpeed()
                                onProgress(DownloadUpdate(bytesDownloaded: currentTotal, totalBytes: totalSize, speedBytesPerSecond: speed))
                                finishedFile = true
                                break
                            }

                            // Range is invalid for current server file length; restart this file once from zero.
                            if !restartedAfter416 {
                                try? FileManager.default.removeItem(at: destinationFileURL)
                                downloadedBytesPerFile[fileName] = 0
                                restartedAfter416 = true
                                continue
                            }
                        }

                        // Ignore missing optional files.
                        if httpResponse.statusCode == 404 && optionalModelFiles.contains(fileName) {
                            finishedFile = true
                            break
                        }

                        let reason = HTTPURLResponse.localizedString(forStatusCode: httpResponse.statusCode)
                        throw NSError(domain: "ModelDownloader", code: httpResponse.statusCode, userInfo: [NSLocalizedDescriptionKey: "HTTP \(httpResponse.statusCode): \(reason)"])
                    }

                    // Efficient Buffered Write with resume support.
                    // 206 means server accepted Range and we should append.
                    // 200 means full content, so restart file from zero.
                    if !FileManager.default.fileExists(atPath: destinationFileURL.path) {
                        FileManager.default.createFile(atPath: destinationFileURL.path, contents: nil)
                    }
                    if existingBytes > 0 && httpResponse.statusCode == 200 {
                        try? FileManager.default.removeItem(at: destinationFileURL)
                        FileManager.default.createFile(atPath: destinationFileURL.path, contents: nil)
                        existingBytes = 0
                    }
                    let fileHandle = try FileHandle(forWritingTo: destinationFileURL)
                    defer { try? fileHandle.close() }
                    if existingBytes > 0 {
                        try fileHandle.seekToEnd()
                    } else {
                        try fileHandle.truncate(atOffset: 0)
                    }

                    var byteCountPerFile: Int64 = existingBytes
                    var buffer = Data()
                    let chunkSize = 64 * 1024 // 64KB buffer

                    for try await byte in bytes {
                        buffer.append(byte)
                        byteCountPerFile += 1

                        if buffer.count >= chunkSize {
                            let flushedBytes = Int64(buffer.count)
                            try fileHandle.write(contentsOf: buffer)
                            buffer.removeAll(keepingCapacity: true)
                            recordTransfer(flushedBytes)

                            // Periodic Progress Update
                            downloadedBytesPerFile[fileName] = byteCountPerFile
                            let currentTotal = downloadedBytesPerFile.values.reduce(0, +)
                            let speed = realtimeSpeed()
                            onProgress(DownloadUpdate(bytesDownloaded: currentTotal, totalBytes: totalSize, speedBytesPerSecond: speed))
                        }
                    }

                    if !buffer.isEmpty {
                        let flushedBytes = Int64(buffer.count)
                        try fileHandle.write(contentsOf: buffer)
                        buffer.removeAll()
                        recordTransfer(flushedBytes)
                    }

                    downloadedBytesPerFile[fileName] = byteCountPerFile
                    let currentTotal = downloadedBytesPerFile.values.reduce(0, +)
                    let speed = realtimeSpeed()
                    onProgress(DownloadUpdate(bytesDownloaded: currentTotal, totalBytes: totalSize, speedBytesPerSecond: speed))
                    finishedFile = true
                } catch let error as URLError where error.code.isTransientDownloadFailure && attempt < maxRetries {
                    attempt += 1
                    let delaySeconds = min(pow(2.0, Double(attempt - 1)), 30.0)
                    try await Task.sleep(for: .seconds(delaySeconds))
                }
            }
        }

        var finalBytes: Int64 = 0
        for fileName in model.requiredFileNames {
            if optionalModelFiles.contains(fileName) {
                continue
            }

            let localURL = destinationDir.appendingPathComponent(fileName)
            let localBytes = localFileSize(at: localURL)
            finalBytes += localBytes

            if let expectedBytes = expectedBytesPerFile[fileName], expectedBytes > 0 {
                if localBytes != expectedBytes {
                    throw NSError(
                        domain: "ModelDownloader",
                        code: -2,
                        userInfo: [
                            NSLocalizedDescriptionKey:
                                "Incomplete file: \(fileName) (\(localBytes) / \(expectedBytes) bytes)"
                        ]
                    )
                }
            } else if localBytes <= 0 {
                throw NSError(
                    domain: "ModelDownloader",
                    code: -2,
                    userInfo: [
                        NSLocalizedDescriptionKey:
                            "Missing downloaded file: \(fileName)"
                    ]
                )
            }
        }

        let minimumExpectedBytes = Int64(Double(totalSize) * completionThresholdRatio)
        if finalBytes < minimumExpectedBytes {
            // Keep this guard as a soft sanity check for metadata-based progress,
            // but by this point each file has already been validated above.
            onProgress(DownloadUpdate(bytesDownloaded: finalBytes, totalBytes: totalSize, speedBytesPerSecond: 0))
        }

        let marker = ModelInstallMarker(
            version: 1,
            modelId: model.id,
            totalBytes: finalBytes,
            fileNames: model.requiredFileNames
        )
        let markerData = try JSONEncoder().encode(marker)
        FileManager.default.createFile(atPath: markerURL.path, contents: markerData)
    }

    // MARK: - CoreML ZIP Download + Extraction

    private func downloadAndExtractCoreML(
        model: AIModel,
        hfToken: String?,
        destinationDir: URL,
        onProgress: @Sendable @escaping (DownloadUpdate) -> Void
    ) async throws {
        guard let zipURL = model.allDownloadURLs.first else {
            throw NSError(domain: "ModelDownloader", code: -4, userInfo: [NSLocalizedDescriptionKey: "No download URL for CoreML model"])
        }

        let totalSize = model.sizeBytes
        let sentinelURL = destinationDir.appendingPathComponent("_downloaded")

        // Already extracted
        if FileManager.default.fileExists(atPath: sentinelURL.path) {
            onProgress(DownloadUpdate(bytesDownloaded: totalSize, totalBytes: totalSize, speedBytesPerSecond: 0))
            return
        }

        if !FileManager.default.fileExists(atPath: destinationDir.path) {
            try FileManager.default.createDirectory(at: destinationDir, withIntermediateDirectories: true)
        }

        let tempZipURL = destinationDir.appendingPathComponent("_temp_download.zip")

        // Download the ZIP with resume support
        let realtimeWindowSeconds: TimeInterval = 3.0
        var throughputSamples: [(time: Date, bytes: Int64)] = []
        func recordTransfer(_ bytes: Int64) {
            let now = Date()
            throughputSamples.append((time: now, bytes: bytes))
            let cutoff = now.addingTimeInterval(-realtimeWindowSeconds)
            throughputSamples.removeAll { $0.time < cutoff }
        }
        func realtimeSpeed() -> Double {
            let now = Date()
            let cutoff = now.addingTimeInterval(-realtimeWindowSeconds)
            throughputSamples.removeAll { $0.time < cutoff }
            guard let firstTime = throughputSamples.first?.time else { return 0 }
            let bytes = throughputSamples.reduce(Int64(0)) { $0 + $1.bytes }
            let span = max(0.1, now.timeIntervalSince(firstTime))
            return Double(bytes) / span
        }

        let maxRetries = 6
        var attempt = 0
        var downloadComplete = false
        var downloadedBytes: Int64 = 0

        while !downloadComplete {
            do {
                var existingBytes = localFileSize(at: tempZipURL)
                var request = URLRequest(url: zipURL, cachePolicy: .reloadIgnoringLocalCacheData)
                if let token = hfToken, !token.isEmpty {
                    request.addValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
                }
                if existingBytes > 0 {
                    request.addValue("bytes=\(existingBytes)-", forHTTPHeaderField: "Range")
                }

                let (bytes, response) = try await urlSession.bytes(for: request)
                guard let httpResponse = response as? HTTPURLResponse else {
                    throw NSError(domain: "ModelDownloader", code: -1, userInfo: [NSLocalizedDescriptionKey: "No HTTP response"])
                }

                if !(200...299).contains(httpResponse.statusCode) {
                    if httpResponse.statusCode == 416 {
                        // Already complete
                        existingBytes = localFileSize(at: tempZipURL)
                        downloadedBytes = existingBytes
                        onProgress(DownloadUpdate(bytesDownloaded: existingBytes, totalBytes: totalSize, speedBytesPerSecond: 0))
                        downloadComplete = true
                        break
                    }
                    let reason = HTTPURLResponse.localizedString(forStatusCode: httpResponse.statusCode)
                    throw NSError(domain: "ModelDownloader", code: httpResponse.statusCode, userInfo: [NSLocalizedDescriptionKey: "HTTP \(httpResponse.statusCode): \(reason)"])
                }

                if !FileManager.default.fileExists(atPath: tempZipURL.path) {
                    FileManager.default.createFile(atPath: tempZipURL.path, contents: nil)
                }
                if existingBytes > 0 && httpResponse.statusCode == 200 {
                    try? FileManager.default.removeItem(at: tempZipURL)
                    FileManager.default.createFile(atPath: tempZipURL.path, contents: nil)
                    existingBytes = 0
                }
                let fileHandle = try FileHandle(forWritingTo: tempZipURL)
                defer { try? fileHandle.close() }
                if existingBytes > 0 {
                    try fileHandle.seekToEnd()
                } else {
                    try fileHandle.truncate(atOffset: 0)
                }

                var byteCount: Int64 = existingBytes
                var buffer = Data()
                let chunkSize = 64 * 1024

                for try await byte in bytes {
                    buffer.append(byte)
                    byteCount += 1
                    if buffer.count >= chunkSize {
                        let flushed = Int64(buffer.count)
                        try fileHandle.write(contentsOf: buffer)
                        buffer.removeAll(keepingCapacity: true)
                        recordTransfer(flushed)
                        downloadedBytes = byteCount
                        // Report 90% of progress for the download phase
                        let reportedBytes = Int64(Double(byteCount) * 0.9)
                        onProgress(DownloadUpdate(bytesDownloaded: reportedBytes, totalBytes: totalSize, speedBytesPerSecond: realtimeSpeed()))
                    }
                }
                if !buffer.isEmpty {
                    let flushed = Int64(buffer.count)
                    try fileHandle.write(contentsOf: buffer)
                    buffer.removeAll()
                    recordTransfer(flushed)
                }
                downloadedBytes = byteCount
                downloadComplete = true
            } catch let error as URLError where error.code.isTransientDownloadFailure && attempt < maxRetries {
                attempt += 1
                let delay = min(pow(2.0, Double(attempt - 1)), 30.0)
                try await Task.sleep(for: .seconds(delay))
            }
        }

        // Report 90% — now extract
        onProgress(DownloadUpdate(bytesDownloaded: Int64(Double(totalSize) * 0.9), totalBytes: totalSize, speedBytesPerSecond: 0))

        // Extract ZIP into a temp subdirectory, then flatten contents into destinationDir
        let extractTempDir = destinationDir.appendingPathComponent("_extract_temp")
        try? FileManager.default.removeItem(at: extractTempDir)
        try FileManager.default.createDirectory(at: extractTempDir, withIntermediateDirectories: true)

        try extractZip(at: tempZipURL, to: extractTempDir)

        // Flatten: if ZIP contained a single top-level folder, move its contents up
        let extractedItems = (try? FileManager.default.contentsOfDirectory(at: extractTempDir, includingPropertiesForKeys: nil)) ?? []
        var sourceDir = extractTempDir
        if extractedItems.count == 1, let single = extractedItems.first {
            var isDir: ObjCBool = false
            if FileManager.default.fileExists(atPath: single.path, isDirectory: &isDir), isDir.boolValue {
                sourceDir = single
            }
        }

        // Move extracted model files to destinationDir
        let modelFiles = (try? FileManager.default.contentsOfDirectory(at: sourceDir, includingPropertiesForKeys: nil)) ?? []
        for item in modelFiles {
            let dest = destinationDir.appendingPathComponent(item.lastPathComponent)
            try? FileManager.default.removeItem(at: dest)
            try FileManager.default.moveItem(at: item, to: dest)
        }

        // Strip pre-compiled ANE (E5) binaries from all .mlmodelc bundles.
        // These binaries were compiled for a different chip (e.g. M-series Mac).
        // Removing them forces CoreML to compile fresh ANE kernels for this device
        // on first load, eliminating the "Must re-compile E5 bundle" ANE runtime error.
        stripStaleArtifacts(in: destinationDir)

        // Clean up
        try? FileManager.default.removeItem(at: extractTempDir)
        try? FileManager.default.removeItem(at: tempZipURL)

        // Write sentinel to mark successful extraction
        FileManager.default.createFile(atPath: sentinelURL.path, contents: Data())
        onProgress(DownloadUpdate(bytesDownloaded: totalSize, totalBytes: totalSize, speedBytesPerSecond: 0))
    }

    private func extractZip(at zipURL: URL, to destinationURL: URL) throws {
        try FileManager.default.createDirectory(at: destinationURL, withIntermediateDirectories: true)
        let fm = FileManager.default
        try fm.unzipItem(at: zipURL, to: destinationURL)
    }

    /// Recursively walks `directory`, finds every `.mlmodelc` bundle, and removes
    /// the pre-compiled ANE / espresso artifacts inside them.
    /// Those binaries are chip-specific (often compiled for M-series Mac ANE).
    /// Stripping them forces CoreML to JIT-compile correct kernels for this device
    /// on first load, eliminating "Must re-compile E5 bundle" errors.
    private func stripStaleArtifacts(in directory: URL) {
        let fm = FileManager.default
        guard let enumerator = fm.enumerator(
            at: directory,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        ) else { return }

        // ANE / espresso file name patterns compiled by coremltools
        let stalePatterns = [
            "model.espresso.net",
            "model.espresso.shape",
            "model.espresso.weights",
            "coreml_model.espresso.net",
            "coreml_model.espresso.shape",
            "coreml_model.espresso.weights",
        ]

        for case let url as URL in enumerator {
            guard url.pathExtension == "mlmodelc",
                  (try? url.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true
            else { continue }

            // Don't recurse into the mlmodelc itself through the enumerator
            enumerator.skipDescendants()

            for pattern in stalePatterns {
                let target = url.appendingPathComponent(pattern)
                if fm.fileExists(atPath: target.path) {
                    try? fm.removeItem(at: target)
                }
            }
        }
    }
}

private extension Data {
    // No longer needed — extraction handled by ZIPFoundation FileManager extension.
}
