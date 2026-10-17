import Foundation
import CryptoKit
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

struct ChunkedDownloadResult: Sendable {
    let statusCode: Int
    let contentRange: String?
    let fileBytes: Int64
}

/// The OS owns these download tasks, so transfer continues while our process is
/// suspended. Task descriptions recover destinations after a system relaunch.
final class BackgroundModelTransfers: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
    static let shared = BackgroundModelTransfers()
    static let identifier = "\(Bundle.main.bundleIdentifier ?? "com.llmhub").model-downloads"
    @TaskLocal static var owner: String = ""

    private struct Location: Codable {
        let path: String
        let offset: Int64
        var owners: Set<String>
        var url: URL { URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent(path) }
    }
    private struct Waiter {
        let owner: String
        let continuation: CheckedContinuation<ChunkedDownloadResult, Error>
        let progress: @Sendable (Int64, Double) -> Void
    }
    private final class Entry {
        let task: URLSessionDownloadTask
        var location: Location
        var waiters: [UUID: Waiter] = [:]
        var result: Result<ChunkedDownloadResult, Error>?
        var lastTime = Date()
        var lastBytes: Int64 = 0
        init(task: URLSessionDownloadTask, location: Location) { self.task = task; self.location = location }
    }
    private final class Ticket: @unchecked Sendable {
        let id = UUID()
        let lock = NSLock()
        var cancelled = false
        func cancel() { lock.lock(); cancelled = true; lock.unlock() }
        var isCancelled: Bool { lock.lock(); defer { lock.unlock() }; return cancelled }
    }
    private let queue: OperationQueue = {
        let queue = OperationQueue()
        queue.name = "LLMHub.BackgroundModelDownloads"
        queue.maxConcurrentOperationCount = 1
        return queue
    }()
    private var session: URLSession!
    // All entries, results, and event callbacks are confined to queue.
    private var entries: [String: Entry] = [:]
    private var recovered: [String: Result<ChunkedDownloadResult, Error>] = [:]
    private var backgroundCompletion: (() -> Void)?
    private var eventsFinished = false
    private let storage: URL

    override private init() {
        storage = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("BackgroundModelDownloads")
        super.init()
        try? FileManager.default.createDirectory(at: storage, withIntermediateDirectories: true,
            attributes: [.protectionKey: FileProtectionType.completeUntilFirstUserAuthentication])
        let config = URLSessionConfiguration.background(withIdentifier: Self.identifier)
        config.isDiscretionary = false
        config.sessionSendsLaunchEvents = true
        config.waitsForConnectivity = true
        config.timeoutIntervalForResource = 7 * 24 * 60 * 60
        config.httpMaximumConnectionsPerHost = 3
        session = URLSession(configuration: config, delegate: self, delegateQueue: queue)
        session.getAllTasks { tasks in
            self.queue.addOperation {
                for task in tasks {
                    guard let task = task as? URLSessionDownloadTask, task.state != .canceling && task.state != .completed,
                          let location = self.location(task), self.entries[location.path] == nil else { continue }
                    self.entries[location.path] = Entry(task: task, location: location)
                }
            }
        }
    }

    private func path(_ url: URL) -> String {
        let prefix = NSHomeDirectory() + "/"
        precondition(url.path.hasPrefix(prefix), "Model downloads must stay inside the app container")
        return String(url.path.dropFirst(prefix.count))
    }
    private func location(_ task: URLSessionTask) -> Location? {
        guard let text = task.taskDescription, let data = text.data(using: .utf8) else { return nil }
        return try? JSONDecoder().decode(Location.self, from: data)
    }
    private func persist(_ entry: Entry) {
        entry.task.taskDescription = (try? JSONEncoder().encode(entry.location)).flatMap { String(data: $0, encoding: .utf8) }
    }
    private func resumeURL(_ path: String) -> URL {
        let digest = SHA256.hash(data: Data(path.utf8)).map { String(format: "%02x", $0) }.joined()
        return storage.appendingPathComponent(digest)
    }
    private func ensure(request: URLRequest, destination: URL, offset: Int64, owner: String) -> Entry {
        let key = path(destination)
        if let entry = entries[key], entry.task.state != .canceling && entry.task.state != .completed {
            entry.location.owners.insert(owner)
            persist(entry)
            return entry
        }
        var location = Location(path: key, offset: offset, owners: [owner])
        let resume = resumeURL(key)
        let task: URLSessionDownloadTask
        if let data = try? Data(contentsOf: resume),
           let saved = try? Data(contentsOf: resume.appendingPathExtension("json")),
           let old = try? JSONDecoder().decode(Location.self, from: saved), old.offset == offset {
            location = Location(path: key, offset: old.offset, owners: [owner])
            task = session.downloadTask(withResumeData: data)
        } else {
            task = session.downloadTask(with: request)
        }
        let entry = Entry(task: task, location: location)
        entries[key] = entry
        persist(entry)
        task.resume()
        return entry
    }

    /// Queue the complete file set before the app can be suspended between files.
    func prepare(_ items: [(URLRequest, URL, Int64)], owner: String) async {
        // Wait for reconnection to existing system tasks before creating any.
        let ticket = Ticket()
        await withTaskCancellationHandler {
            guard !Task.isCancelled else { return }
            let tasks = await session.allTasks
            guard !Task.isCancelled else { return }
            await withCheckedContinuation { continuation in
                queue.addOperation {
                    guard !ticket.isCancelled else { continuation.resume(); return }
                    for task in tasks {
                        guard let task = task as? URLSessionDownloadTask, task.state != .canceling && task.state != .completed,
                              let location = self.location(task), self.entries[location.path] == nil else { continue }
                        self.entries[location.path] = Entry(task: task, location: location)
                    }
                    for (request, destination, offset) in items {
                        let key = self.path(destination)
                        if case .success(let result) = self.recovered[key], (200...299).contains(result.statusCode),
                           FileManager.default.fileExists(atPath: destination.path), offset == result.fileBytes {
                            continue
                        }
                        self.recovered.removeValue(forKey: key)
                        _ = self.ensure(request: request, destination: destination, offset: offset, owner: owner)
                    }
                    continuation.resume()
                }
        }
        } onCancel: {
            ticket.cancel()
            self.cancel(owner: owner)
        }
    }

    func download(request: URLRequest, destination: URL, offset: Int64,
                  progress: @Sendable @escaping (Int64, Double) -> Void) async throws -> ChunkedDownloadResult {
        let owner = Self.owner
        let ticket = Ticket()
        return try await withTaskCancellationHandler {
            try Task.checkCancellation()
            return try await withCheckedThrowingContinuation { continuation in
                queue.addOperation {
                    if ticket.isCancelled { continuation.resume(throwing: CancellationError()); return }
                    let key = self.path(destination)
                    if let result = self.recovered.removeValue(forKey: key) {
                        continuation.resume(with: result)
                        return
                    }
                    let entry = self.ensure(request: request, destination: destination, offset: offset, owner: owner)
                    entry.waiters[ticket.id] = Waiter(owner: owner, continuation: continuation, progress: progress)
                }
            }
        } onCancel: {
            ticket.cancel()
            self.queue.addOperation {
                let key = self.path(destination)
                if let waiter = self.entries[key]?.waiters.removeValue(forKey: ticket.id) {
                    waiter.continuation.resume(throwing: CancellationError())
                }
            }
        }
    }

    func cancel(owner: String) {
        queue.addOperation {
            for entry in Array(self.entries.values) where entry.location.owners.contains(owner) {
                entry.location.owners.remove(owner)
                for (id, waiter) in entry.waiters where waiter.owner == owner {
                    entry.waiters.removeValue(forKey: id)
                    waiter.continuation.resume(throwing: CancellationError())
                }
                self.persist(entry)
                if entry.location.owners.isEmpty {
                    self.entries.removeValue(forKey: entry.location.path)
                    entry.task.cancel { data in
                        guard let data else { return }
                        let file = self.resumeURL(entry.location.path)
                        try? data.write(to: file, options: .atomic)
                        try? JSONEncoder().encode(entry.location).write(to: file.appendingPathExtension("json"), options: .atomic)
                    }
                }
            }
        }
    }

    func handleEvents(completion: @escaping () -> Void) {
        queue.addOperation {
            self.backgroundCompletion = completion
            self.completeEventsIfReady()
        }
    }

    private func completeEventsIfReady() {
        guard eventsFinished, let completion = backgroundCompletion else { return }
        backgroundCompletion = nil
        eventsFinished = false
        DispatchQueue.main.async { completion() }
    }

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask,
                    didWriteData bytesWritten: Int64, totalBytesWritten: Int64, totalBytesExpectedToWrite: Int64) {
        guard let location = location(downloadTask), let entry = entries[location.path] else { return }
        let now = Date()
        let interval = now.timeIntervalSince(entry.lastTime)
        guard interval >= 0.2 else { return }
        let speed = Double(max(0, totalBytesWritten - entry.lastBytes)) / interval
        entry.lastTime = now
        entry.lastBytes = totalBytesWritten
        for waiter in entry.waiters.values { waiter.progress(location.offset + totalBytesWritten, speed) }
    }

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didFinishDownloadingTo temporaryURL: URL) {
        guard let location = location(downloadTask), let response = downloadTask.response as? HTTPURLResponse else { return }
        if let current = entries[location.path], current.task.taskIdentifier != downloadTask.taskIdentifier { return }
        let entry = entries[location.path] ?? Entry(task: downloadTask, location: location)
        entries[location.path] = entry
        do {
            let fm = FileManager.default
            if (200...299).contains(response.statusCode) {
                let destination = location.url
                try fm.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
                if response.statusCode == 206 && location.offset > 0 {
                    let total = response.value(forHTTPHeaderField: "Content-Range")?
                        .split(separator: "/").last.flatMap { Int64($0) }
                    let partSize = (try? fm.attributesOfItem(atPath: temporaryURL.path)[.size] as? Int64) ?? 0
                    let size = (try? fm.attributesOfItem(atPath: destination.path)[.size] as? Int64) ?? 0
                    // On a native resume, the response range may describe only
                    // the last segment; URLSession has reassembled the complete
                    // remainder in its temp file. Validate its full length.
                    guard total == location.offset + partSize, size == location.offset else {
                        throw URLError(.badServerResponse)
                    }
                    let input = try FileHandle(forReadingFrom: temporaryURL)
                    defer { try? input.close() }
                    let output = try FileHandle(forWritingTo: destination)
                    defer { try? output.close() }
                    try output.seekToEnd()
                    while let data = try input.read(upToCount: 4 * 1024 * 1024), !data.isEmpty {
                        try output.write(contentsOf: data)
                    }
                } else {
                    if fm.fileExists(atPath: destination.path) { try fm.removeItem(at: destination) }
                    try fm.moveItem(at: temporaryURL, to: destination)
                }
                try fm.setAttributes([.protectionKey: FileProtectionType.completeUntilFirstUserAuthentication],
                                     ofItemAtPath: destination.path)
            }
            let bytes = (try? FileManager.default.attributesOfItem(atPath: location.url.path)[.size] as? Int64) ?? 0
            entry.result = .success(ChunkedDownloadResult(statusCode: response.statusCode,
                contentRange: response.value(forHTTPHeaderField: "Content-Range"), fileBytes: bytes))
        } catch { entry.result = .failure(error) }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        guard let location = location(task) else { return }
        if let current = entries[location.path], current.task.taskIdentifier != task.taskIdentifier { return }
        let entry = entries.removeValue(forKey: location.path)
        let resume = resumeURL(location.path)
        if let data = (error as NSError?)?.userInfo[NSURLSessionDownloadTaskResumeData] as? Data {
            try? data.write(to: resume, options: .atomic)
            try? JSONEncoder().encode(location).write(to: resume.appendingPathExtension("json"), options: .atomic)
        } else if error == nil {
            try? FileManager.default.removeItem(at: resume)
            try? FileManager.default.removeItem(at: resume.appendingPathExtension("json"))
        }
        let result: Result<ChunkedDownloadResult, Error>
        if let finished = entry?.result {
            result = finished
        } else if let response = task.response as? HTTPURLResponse, response.statusCode >= 400 {
            let bytes = (try? FileManager.default.attributesOfItem(atPath: location.url.path)[.size] as? Int64) ?? 0
            result = .success(ChunkedDownloadResult(statusCode: response.statusCode,
                contentRange: response.value(forHTTPHeaderField: "Content-Range"), fileBytes: bytes))
        } else {
            result = .failure(error ?? URLError(.badServerResponse))
        }
        if let entry, !entry.waiters.isEmpty {
            for waiter in entry.waiters.values { waiter.continuation.resume(with: result) }
        } else if (error as? URLError)?.code != .cancelled {
            recovered[location.path] = result
        }
    }

    func urlSessionDidFinishEvents(forBackgroundURLSession session: URLSession) {
        eventsFinished = true
        completeEventsIfReady()
    }
}

private final class ChunkedFileDownloader: @unchecked Sendable {
    let destinationURL: URL
    let requestedOffset: Int64
    let progressHandler: @Sendable (Int64, Double) -> Void
    init(destinationURL: URL, requestedOffset: Int64, progressHandler: @Sendable @escaping (Int64, Double) -> Void) {
        self.destinationURL = destinationURL
        self.requestedOffset = requestedOffset
        self.progressHandler = progressHandler
    }
    func start(request: URLRequest, configuration: URLSessionConfiguration) async throws -> ChunkedDownloadResult {
        try await BackgroundModelTransfers.shared.download(request: request, destination: destinationURL,
            offset: requestedOffset, progress: progressHandler)
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
        func authorizedRequest(method: String, token: String?) -> URLRequest {
            var request = URLRequest(url: fileURL, cachePolicy: .reloadIgnoringLocalCacheData)
            request.httpMethod = method
            if let token, !token.isEmpty {
                request.addValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
            }
            return request
        }

        func tryFetchSize(token: String?) async -> (size: Int64?, authFailed: Bool) {
            // First try HEAD for content length.
            do {
                let request = authorizedRequest(method: "HEAD", token: token)
                let (_, response) = try await urlSession.data(for: request)
                if let httpResponse = response as? HTTPURLResponse {
                    if (200...299).contains(httpResponse.statusCode),
                       let contentLength = httpResponse.value(forHTTPHeaderField: "Content-Length"),
                       let size = Int64(contentLength),
                       size > 0 {
                        return (size, false)
                    }
                    if httpResponse.statusCode == 401 || httpResponse.statusCode == 403 {
                        return (nil, true)
                    }
                }
            } catch {
                // Fall through to range probe.
            }

            // Some endpoints block HEAD; probe with GET Range to parse total size from Content-Range.
            do {
                var request = authorizedRequest(method: "GET", token: token)
                request.addValue("bytes=0-0", forHTTPHeaderField: "Range")
                let (_, response) = try await urlSession.data(for: request)
                if let httpResponse = response as? HTTPURLResponse {
                    if (200...299).contains(httpResponse.statusCode) || httpResponse.statusCode == 206 {
                        if let total = totalSizeFromContentRange(httpResponse.value(forHTTPHeaderField: "Content-Range")), total > 0 {
                            return (total, false)
                        }
                        if let contentLength = httpResponse.value(forHTTPHeaderField: "Content-Length"),
                           let size = Int64(contentLength),
                           size > 0 {
                            return (size, false)
                        }
                    }
                    if httpResponse.statusCode == 401 || httpResponse.statusCode == 403 {
                        return (nil, true)
                    }
                }
            } catch {
                // Give up and treat as unknown size.
            }
            return (nil, false)
        }

        // Always attempt unauthenticated request first so public models don't use token quota
        let first = await tryFetchSize(token: nil)
        if let size = first.size {
            return size
        }
        // If auth failed (401/403) and a token is available, retry with token for gated models
        if first.authFailed, let hfToken = hfToken, !hfToken.isEmpty {
            let retry = await tryFetchSize(token: hfToken)
            if let size = retry.size {
                return size
            }
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
        try await withTaskCancellationHandler {
            do {
                try Task.checkCancellation()
                try await BackgroundModelTransfers.$owner.withValue(model.id) {
                    try await downloadModelFiles(model, hfToken: hfToken, destinationDir: destinationDir, onProgress: onProgress)
                }
            } catch {
                BackgroundModelTransfers.shared.cancel(owner: model.id)
                throw error
            }
        } onCancel: {
            BackgroundModelTransfers.shared.cancel(owner: model.id)
        }
    }

    private func downloadModelFiles(
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
                        
                        let result = try await BackgroundModelTransfers.shared.download(request: request,
                            destination: destURL, offset: existingBytes) { bytes, speed in
                                onProgress(DownloadUpdate(bytesDownloaded: bytes, totalBytes: model.sizeBytes, speedBytesPerSecond: speed))
                            }
                        guard (200...299).contains(result.statusCode) else {
                            if result.statusCode == 416 && !restartedAfter416 {
                                try? FileManager.default.removeItem(at: destURL)
                                restartedAfter416 = true
                                continue
                            }
                            throw URLError(.badServerResponse)
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
            // Resolve community metadata so progress and dependency checks include encoders and VAE.
            _ = await EnvWrapper.env.resolveModel(model.id, offline: true)
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
            
            let needed = allFiles.filter { file in
                let size = localFileSize(at: URL(fileURLWithPath: ModelZoo.filePathForModelDownloaded(file)))
                if let expected = drawThingsFileSizes[file] { return size != expected }
                return !ModelZoo.isModelDownloaded(file)
            }
            let items: [(URLRequest, URL, Int64)] = needed.map { file in
                let destination = URL(fileURLWithPath: ModelZoo.filePathForModelDownloaded(file))
                if let expected = drawThingsFileSizes[file], localFileSize(at: destination) > expected {
                    try? FileManager.default.removeItem(at: destination)
                }
                let offset = localFileSize(at: destination)
                var request = URLRequest(url: URL(string: "https://static.libnnc.org/\(file)")!)
                if offset > 0 { request.setValue("bytes=\(offset)-", forHTTPHeaderField: "Range") }
                return (request, destination, offset)
            }
            await BackgroundModelTransfers.shared.prepare(items, owner: model.id)
            var completed = allFiles.filter { !needed.contains($0) }.reduce(Int64(0)) { sum, file in
                sum + localFileSize(at: URL(fileURLWithPath: ModelZoo.filePathForModelDownloaded(file)))
            }
            let expectedTotal = allFiles.reduce(Int64(0)) { $0 + (drawThingsFileSizes[$1] ?? 0) }
            let progressTotal = max(model.sizeBytes, expectedTotal)
            for (request, destination, offset) in items {
                let base = completed
                let result = try await BackgroundModelTransfers.shared.download(request: request,
                    destination: destination, offset: offset) { bytes, speed in
                        onProgress(DownloadUpdate(bytesDownloaded: base + bytes, totalBytes: progressTotal,
                                                  speedBytesPerSecond: speed))
                    }
                guard (200...299).contains(result.statusCode) || result.statusCode == 416 else {
                    throw URLError(.badServerResponse)
                }
                let file = destination.lastPathComponent
                if let expected = drawThingsFileSizes[file], result.fileBytes != expected {
                    throw URLError(.downloadDecodingFailedMidStream)
                }
                if let expectedSHA = ModelZoo.fileSHA256ForModelDownloaded(file) {
                    let handle = try FileHandle(forReadingFrom: destination)
                    defer { try? handle.close() }
                    var hash = SHA256()
                    while let data = try handle.read(upToCount: 4 * 1024 * 1024), !data.isEmpty { hash.update(data: data) }
                    let actual = hash.finalize().map { String(format: "%02x", $0) }.joined()
                    guard actual == expectedSHA.lowercased() else {
                        try? FileManager.default.removeItem(at: destination)
                        throw URLError(.cannotDecodeContentData)
                    }
                }
                completed += result.fileBytes
            }
            // Keep the library's dependency verification/catalog integration;
            // all network payloads above were handled by the native session.
            _ = try await EnvWrapper.env.ensure(model.id, offline: true)
            onProgress(DownloadUpdate(bytesDownloaded: model.sizeBytes, totalBytes: model.sizeBytes, speedBytesPerSecond: 0))
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
        
        // Ensure clean destination
        if !FileManager.default.fileExists(atPath: destinationDir.path) {
            try FileManager.default.createDirectory(at: destinationDir, withIntermediateDirectories: true)
        }

        let markerURL = Self.installMarkerURL(for: destinationDir)
        try? FileManager.default.removeItem(at: markerURL)
        
        let downloadItems = Array(zip(model.requiredFileNames, model.allDownloadURLs))

        await BackgroundModelTransfers.shared.prepare(downloadItems.map { name, url in
            let destination = destinationDir.appendingPathComponent(name)
            let offset = localFileSize(at: destination)
            var request = URLRequest(url: url)
            if offset > 0 { request.setValue("bytes=\(offset)-", forHTTPHeaderField: "Range") }
            return (request, destination, offset)
        }, owner: model.id)

        var currentHfToken: String? = nil // Always attempt unauthenticated download first
        for (fileName, fileURL) in downloadItems {
            
            let destinationFileURL = destinationDir.appendingPathComponent(fileName)
            var expectedSize = await remoteFileSize(fileURL: fileURL, hfToken: hfToken)
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
                        onProgress(DownloadUpdate(bytesDownloaded: currentTotal, totalBytes: totalSize, speedBytesPerSecond: 0))
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
                    let existingBytes = localFileSize(at: destinationFileURL)
                    var request = URLRequest(url: fileURL, cachePolicy: .reloadIgnoringLocalCacheData)
                    if let token = currentHfToken, !token.isEmpty {
                        request.addValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
                    }
                    if existingBytes > 0 {
                        request.addValue("bytes=\(existingBytes)-", forHTTPHeaderField: "Range")
                    }

                    let completedBytes = downloadedBytesPerFile
                        .filter { $0.key != fileName }
                        .values
                        .reduce(0, +)
                    let downloader = ChunkedFileDownloader(
                        destinationURL: destinationFileURL,
                        requestedOffset: existingBytes,
                        progressHandler: { fileBytes, speed in
                            onProgress(
                                DownloadUpdate(
                                    bytesDownloaded: completedBytes + fileBytes,
                                    totalBytes: totalSize,
                                    speedBytesPerSecond: speed
                                )
                            )
                        }
                    )
                    let result = try await downloader.start(
                        request: request,
                        configuration: urlSession.configuration
                    )

                    // Critical 404/403 Handling
                    if !(200...299).contains(result.statusCode) {
                        // If auth failed (401 or 403) on unauthenticated attempt, retry with token for gated models
                        if (result.statusCode == 401 || result.statusCode == 403) && currentHfToken == nil, let hfToken = hfToken, !hfToken.isEmpty {
                            currentHfToken = hfToken
                            if expectedSize == nil {
                                expectedSize = await remoteFileSize(fileURL: fileURL, hfToken: currentHfToken)
                                if let expectedSize {
                                    expectedBytesPerFile[fileName] = expectedSize
                                }
                            }
                            continue
                        }

                        if result.statusCode == 416 {
                            let rangeHeaderTotal = totalSizeFromContentRange(result.contentRange)
                            let refreshedExpected: Int64?
                            if let expectedSize {
                                refreshedExpected = expectedSize
                            } else if let rangeHeaderTotal {
                                refreshedExpected = rangeHeaderTotal
                            } else {
                                refreshedExpected = await remoteFileSize(fileURL: fileURL, hfToken: currentHfToken)
                            }
                            if let refreshedExpected, existingBytes >= refreshedExpected {
                                downloadedBytesPerFile[fileName] = refreshedExpected
                                let currentTotal = downloadedBytesPerFile.values.reduce(0, +)
                                onProgress(DownloadUpdate(bytesDownloaded: currentTotal, totalBytes: totalSize, speedBytesPerSecond: 0))
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
                        if result.statusCode == 404 && optionalModelFiles.contains(fileName) {
                            finishedFile = true
                            break
                        }

                        let reason = HTTPURLResponse.localizedString(forStatusCode: result.statusCode)
                        throw NSError(domain: "ModelDownloader", code: result.statusCode, userInfo: [NSLocalizedDescriptionKey: "HTTP \(result.statusCode): \(reason)"])
                    }

                    downloadedBytesPerFile[fileName] = result.fileBytes
                    let currentTotal = downloadedBytesPerFile.values.reduce(0, +)
                    onProgress(DownloadUpdate(bytesDownloaded: currentTotal, totalBytes: totalSize, speedBytesPerSecond: 0))
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
            // ZIP files are extracted then deleted; skip post-extraction verification
            if fileName.hasSuffix(".zip") {
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

        // Extract any .zip files downloaded as additionalFiles (e.g. CoreML encoder bundles)
        let fm = FileManager.default
        for fileName in model.requiredFileNames where fileName.hasSuffix(".zip") {
            let zipURL = destinationDir.appendingPathComponent(fileName)
            guard fm.fileExists(atPath: zipURL.path) else { continue }
            try fm.unzipItem(at: zipURL, to: destinationDir)
            try? fm.removeItem(at: zipURL)
        }

        // Store marker without transient zip names (zips were extracted and deleted)
        let persistedFileNames = model.requiredFileNames.filter { !$0.hasSuffix(".zip") }
        let marker = ModelInstallMarker(
            version: 1,
            modelId: model.id,
            totalBytes: finalBytes,
            fileNames: persistedFileNames
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

        // Native background download with resume support.
        let maxRetries = 6
        var attempt = 0
        var downloadComplete = false
        var currentHfToken: String? = nil // Try unauthenticated first
        while !downloadComplete {
            do {
                var existingBytes = localFileSize(at: tempZipURL)
                var request = URLRequest(url: zipURL, cachePolicy: .reloadIgnoringLocalCacheData)
                if let token = currentHfToken, !token.isEmpty {
                    request.addValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
                }
                if existingBytes > 0 {
                    request.addValue("bytes=\(existingBytes)-", forHTTPHeaderField: "Range")
                }

                let result = try await BackgroundModelTransfers.shared.download(request: request,
                    destination: tempZipURL, offset: existingBytes) { bytes, speed in
                        onProgress(DownloadUpdate(bytesDownloaded: Int64(Double(bytes) * 0.9), totalBytes: totalSize,
                            speedBytesPerSecond: speed))
                    }
                if !(200...299).contains(result.statusCode) {
                    if (result.statusCode == 401 || result.statusCode == 403) && currentHfToken == nil,
                       let hfToken, !hfToken.isEmpty {
                        currentHfToken = hfToken
                        continue
                    }
                    if result.statusCode != 416 { throw URLError(.badServerResponse) }
                }
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
