import CryptoKit
import Foundation
import Observation

struct InstalledModel: Codable, Identifiable, Hashable {
    var id: String
    var name: String
    var catalogId: String?
    var repo: String
    var format = "gguf"
    var quant: String
    var files: [String]
    var mmproj: String?
    var sizeBytes: Int64
    var contextLength: Int
    var tags: [String]
    var installedAt: Date
    /// Whether the .litertlm file carries an MTP drafter; nil until the engine has looked (on first load).
    var hasMtpDrafter: Bool?

    var folder: URL { AppPaths.models.appendingPathComponent(id, isDirectory: true) }
    /// Downloads live in `folder`; a local file added from disk keeps its full path and stays where it is.
    var primaryPath: URL { path(files[0]) }
    var mmprojPath: URL? { mmproj.map(path) }
    var isLocalFile: Bool { repo == "local" }
    var displayName: String { "\(name) · \(quant)" }
    /// EmbeddingGemma and the like: served on /v1/embeddings, never offered for chat.
    var isEmbedding: Bool { tags.contains("embedding") }

    /// LiteRT-LM model whose catalog entry has an MTP drafter (looked up live so older downloads get it too), unless the
    /// downloaded file turned out to have none: Google's GPU-only builds leave it out.
    @MainActor var supportsMtp: Bool {
        format == "litertlm" && hasMtpDrafter != false && AppModel.shared.catalog.models.first { $0.id == catalogId }?.sources.contains { $0.format == "litertlm" && $0.mtp == true } == true
    }

    @MainActor var mtpEnabled: Bool { supportsMtp && !AppSettings.current.mtpOff.contains(id) }

    private func path(_ file: String) -> URL {
        file.hasPrefix("/") ? URL(fileURLWithPath: file) : folder.appendingPathComponent((file as NSString).lastPathComponent)
    }
}

/// Why a file picked with "Add local file" was refused; the message is shown as is.
struct LocalModelError: LocalizedError {
    let errorDescription: String?
}

/// Models on disk. Each lives in models/<id>/ with a manifest.json written after a verified download.
@MainActor @Observable
final class ModelStore {
    private(set) var installed: [InstalledModel] = []
    /// Installed models that can chat (everything but embedding models), for the Chat, Agent and Design pickers.
    var chatModels: [InstalledModel] { installed.filter { !$0.isEmbedding } }

    init() { reload() }

    func reload() {
        let dirs = (try? FileManager.default.contentsOfDirectory(at: AppPaths.models, includingPropertiesForKeys: nil)) ?? []
        let decoder = JSONDecoder()
        installed = dirs.compactMap { dir in
            guard let data = try? Data(contentsOf: dir.appendingPathComponent("manifest.json")),
                  let model = try? decoder.decode(InstalledModel.self, from: data),
                  !model.files.isEmpty,
                  FileManager.default.fileExists(atPath: model.primaryPath.path) else { return nil }
            return model
        }.sorted { $0.installedAt < $1.installedAt }
    }

    func find(_ id: String) -> InstalledModel? { installed.first { $0.id == id } }

    func add(_ model: InstalledModel) {
        save(model)
        installed.removeAll { $0.id == model.id }
        installed.append(model)
    }

    /// Records what the engine learned about a file (see `hasMtpDrafter`) in memory and in its manifest.
    func setHasMtpDrafter(_ value: Bool, for id: String) {
        guard let index = installed.firstIndex(where: { $0.id == id }) else { return }
        installed[index].hasMtpDrafter = value
        save(installed[index])
    }

    private func save(_ model: InstalledModel) {
        let encoder = JSONEncoder()
        encoder.outputFormatting = .prettyPrinted
        try? encoder.encode(model).write(to: model.folder.appendingPathComponent("manifest.json"), options: .atomic)
    }

    /// Adds a .gguf or .litertlm from disk without copying it: the manifest keeps its absolute path, and a sibling
    /// mmproj*.gguf comes along as the vision projector. Same rules as the Windows and Linux apps.
    func addLocalFile(_ url: URL) throws -> InstalledModel {
        let fm = FileManager.default
        guard let size = (try? fm.attributesOfItem(atPath: url.path)[.size] as? NSNumber)?.int64Value else {
            throw LocalModelError(errorDescription: CocoaError(.fileNoSuchFile).localizedDescription)
        }
        let format: String
        switch url.pathExtension.lowercased() {
        case "gguf": format = "gguf"
        case "litertlm": format = "litertlm"
        default: throw LocalModelError(errorDescription: L.s("ModelsLocalUnsupported"))
        }
        let fileName = url.lastPathComponent
        if format == "gguf" && fileName.lowercased().hasPrefix("mmproj") {
            throw LocalModelError(errorDescription: L.s("ModelsLocalProjector"))
        }
        let stem = url.deletingPathExtension().lastPathComponent
        let quant = stem.range(of: #"(?:UD-)?(?:IQ\d_[A-Z0-9]+|Q\d(?:_[A-Z0-9]+)*|BF16|F16|F32|MXFP4)$"#,
                               options: [.regularExpression, .caseInsensitive]).map { String(stem[$0]) } ?? ""
        let name = String(stem.dropLast(quant.count)).trimmingCharacters(in: CharacterSet(charactersIn: "-_. "))
        let folder = url.deletingLastPathComponent()
        let mmproj = format == "gguf"
            ? ((try? fm.contentsOfDirectory(atPath: folder.path)) ?? [])
                .filter { $0.lowercased().hasPrefix("mmproj") && $0.lowercased().hasSuffix(".gguf") }
                .map { folder.appendingPathComponent($0).path }
                .min { $0.count < $1.count }
            : nil
        var tags: [String] = []
        if mmproj != nil { tags.append("vision") }
        if name.localizedCaseInsensitiveContains("embedding") { tags.append("embedding") }
        let mmprojSize = mmproj.flatMap { (try? fm.attributesOfItem(atPath: $0)[.size] as? NSNumber)?.int64Value } ?? 0
        let variant = GgufVariant(repo: "local/" + stem, quant: quant.isEmpty ? format : quant, files: [], mmproj: nil)
        let model = InstalledModel(
            id: variant.modelId, name: name.isEmpty ? stem : name, catalogId: nil, repo: "local", format: format,
            quant: !quant.isEmpty ? quant.uppercased() : format == "gguf" ? "GGUF" : "LiteRT",
            files: [url.standardizedFileURL.path], mmproj: mmproj, sizeBytes: size + mmprojSize,
            // The file's own limit is not read; the Settings context length still caps what is used.
            contextLength: 32768, tags: tags, installedAt: Date())
        // Only the manifest lives in models/<id>/; the weights stay where they are.
        try fm.createDirectory(at: model.folder, withIntermediateDirectories: true)
        add(model)
        Log.info("Added local model \(url.path)\(mmproj.map { " with \(($0 as NSString).lastPathComponent)" } ?? "")")
        return model
    }

    /// Removes models/<id>/. For a local file that is only the manifest (and LiteRT-LM's cache): the file itself stays.
    func delete(_ model: InstalledModel, engine: EngineHub) {
        if engine.loadedModel?.id == model.id { engine.stop() }
        if AppModel.shared.embeddings.loadedModel?.id == model.id { AppModel.shared.embeddings.stop() }
        do { try FileManager.default.removeItem(at: model.folder) } catch { Log.error("Delete \(model.id)", error) }
        installed.removeAll { $0.id == model.id }
    }
}

@MainActor @Observable
final class DownloadJob: Identifiable {
    let id: String
    let title: String
    var percent = 0.0
    var status = ""
    var task: Task<Void, Never>?

    init(id: String, title: String) {
        self.id = id
        self.title = title
    }
}

/// Resumable, hash-checked downloads from Hugging Face. Partial files survive restarts as *.part.
@MainActor @Observable
final class DownloadManager {
    private(set) var jobs: [DownloadJob] = []
    /// Last error per model id, shown next to the download button until the next attempt.
    private(set) var errors: [String: String] = [:]
    private let hub: HuggingFaceClient
    private let store: ModelStore

    init(hub: HuggingFaceClient, store: ModelStore) {
        self.hub = hub
        self.store = store
    }

    func find(_ id: String) -> DownloadJob? { jobs.first { $0.id == id } }

    func start(_ variant: GgufVariant, name: String, catalogId: String?, tags: [String], contextLength: Int) {
        let id = variant.modelId
        guard find(id) == nil else { return }
        errors[id] = nil
        let job = DownloadJob(id: id, title: "\(name) · \(variant.quant)")
        jobs.append(job)
        job.task = Task { await run(job, variant, name: name, catalogId: catalogId, tags: tags, contextLength: contextLength) }
    }

    func cancel(_ job: DownloadJob) { job.task?.cancel() }

    private func run(_ job: DownloadJob, _ variant: GgufVariant, name: String, catalogId: String?, tags: [String], contextLength: Int) async {
        // Embedding models serve text on /v1/embeddings: a vision/audio projector (EmbeddingGemma 2's, 555 MB) is not needed.
        let variant = tags.contains("embedding") ? GgufVariant(repo: variant.repo, quant: variant.quant, files: variant.files, mmproj: nil, format: variant.format) : variant
        let folder = AppPaths.models.appendingPathComponent(job.id, isDirectory: true)
        try? FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let files = variant.files + (variant.mmproj.map { [$0] } ?? [])
        let total = files.reduce(Int64(0)) { $0 + $1.size }
        var before: Int64 = 0
        do {
            for file in files {
                let target = folder.appendingPathComponent(file.fileName)
                if !FileManager.default.fileExists(atPath: target.path) {
                    let done = before
                    try await FileDownloader.download(
                        request: hub.request(HuggingFaceClient.resolveURL(repo: variant.repo, path: file.path)),
                        to: target, expectedSize: file.size, sha256: file.sha256
                    ) { written in
                        Task { @MainActor in
                            job.percent = total > 0 ? Double(done + written) * 100 / Double(total) : 0
                            job.status = "\(Fmt.bytes(done + written)) / \(Fmt.bytes(total))"
                        }
                    }
                }
                before += file.size
            }
            store.add(InstalledModel(
                id: job.id, name: name, catalogId: catalogId, repo: variant.repo, format: variant.format, quant: variant.quant,
                files: variant.files.map(\.fileName), mmproj: variant.mmproj?.fileName, sizeBytes: total,
                contextLength: contextLength, tags: tags, installedAt: Date()))
            Log.info("Installed \(job.id)")
        } catch is CancellationError {
            Log.info("Download \(job.id) paused")
        } catch let error as URLError where error.code == .cancelled {
            Log.info("Download \(job.id) paused")
        } catch HubError.status(let code) where code == 401 || code == 403 {
            errors[job.id] = L.s("NeedsToken")
            Log.error("Download \(job.id) needs access (HTTP \(code))")
        } catch {
            errors[job.id] = L.f("DownloadFailed", error.localizedDescription)
            Log.error("Download \(job.id) failed", error)
        }
        // A download refused before any data arrived (gated, no access) would otherwise leave an empty folder behind.
        if errors[job.id] != nil, (try? FileManager.default.contentsOfDirectory(atPath: folder.path))?.isEmpty == true {
            try? FileManager.default.removeItem(at: folder)
        }
        jobs.removeAll { $0 === job }
    }
}

/// Streams one URL to `<target>.part`, resuming from what is already there, then verifies SHA-256 and renames.
final class FileDownloader: NSObject, URLSessionDataDelegate, @unchecked Sendable {
    private let part: URL
    private let progress: (Int64) -> Void
    private var handle: FileHandle?
    private var written: Int64 = 0
    private var existing: Int64
    private var continuation: CheckedContinuation<Void, Error>?
    private var lastReport = Date.distantPast

    private init(part: URL, existing: Int64, progress: @escaping (Int64) -> Void) {
        self.part = part
        self.existing = existing
        self.progress = progress
    }

    static func download(request: URLRequest, to target: URL, expectedSize: Int64, sha256: String?, progress: @escaping (Int64) -> Void) async throws {
        let part = target.appendingPathExtension("part")
        // Only one writer per file, even across app instances; a second one would corrupt the partial file.
        let lockFD = open(part.path + ".lock", O_CREAT | O_RDWR, 0o644)
        guard lockFD >= 0, flock(lockFD, LOCK_EX | LOCK_NB) == 0 else {
            if lockFD >= 0 { close(lockFD) }
            throw NSError(domain: "LLMHub", code: 7, userInfo: [NSLocalizedDescriptionKey: "\(target.lastPathComponent) is already downloading in another LLM Hub window"])
        }
        defer {
            flock(lockFD, LOCK_UN)
            close(lockFD)
            unlink(part.path + ".lock")
        }
        var existing = ((try? FileManager.default.attributesOfItem(atPath: part.path)[.size]) as? NSNumber)?.int64Value ?? 0
        if expectedSize > 0, existing > expectedSize {
            try? FileManager.default.removeItem(at: part)
            existing = 0
        }
        if existing < expectedSize || expectedSize == 0 {
            var request = request
            if existing > 0 { request.setValue("bytes=\(existing)-", forHTTPHeaderField: "Range") }
            let downloader = FileDownloader(part: part, existing: existing, progress: progress)
            let session = URLSession(configuration: .default, delegate: downloader, delegateQueue: nil)
            defer { session.finishTasksAndInvalidate() }
            let task = session.dataTask(with: request)
            try await withTaskCancellationHandler {
                try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
                    downloader.continuation = continuation
                    task.resume()
                }
            } onCancel: {
                task.cancel()
            }
        }

        if let sha256, sha256.count == 64 {
            let actual = try await Task.detached { try Self.sha256(of: part) }.value
            guard actual.caseInsensitiveCompare(sha256) == .orderedSame else {
                try? FileManager.default.removeItem(at: part)
                throw NSError(domain: "LLMHub", code: 1, userInfo: [NSLocalizedDescriptionKey: "Checksum mismatch for \(target.lastPathComponent)"])
            }
        }
        try? FileManager.default.removeItem(at: target)
        try FileManager.default.moveItem(at: part, to: target)
        progress(expectedSize)
    }

    private static func sha256(of url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hasher = SHA256()
        while let chunk = try handle.read(upToCount: 8 << 20), !chunk.isEmpty {
            hasher.update(data: chunk)
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive response: URLResponse,
                    completionHandler: @escaping (URLSession.ResponseDisposition) -> Void) {
        let status = (response as? HTTPURLResponse)?.statusCode ?? 0
        if status == 416 {
            completionHandler(.cancel) // Already complete.
            finish(nil)
            return
        }
        guard (200..<300).contains(status) else {
            completionHandler(.cancel)
            finish(HubError.status(status))
            return
        }
        if status == 200 { existing = 0 } // Server ignored the range; start over.
        if existing == 0 { FileManager.default.createFile(atPath: part.path, contents: nil) }
        handle = try? FileHandle(forWritingTo: part)
        handle?.seekToEndOfFile()
        completionHandler(.allow)
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
        handle?.write(data)
        written += Int64(data.count)
        if Date().timeIntervalSince(lastReport) > 0.25 {
            lastReport = Date()
            progress(existing + written)
        }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        finish(error)
    }

    private func finish(_ error: Error?) {
        try? handle?.close()
        handle = nil
        guard let continuation else { return }
        self.continuation = nil
        if let error { continuation.resume(throwing: error) } else { continuation.resume() }
    }
}
