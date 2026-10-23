import Foundation
import Observation

/// Studio's image and video engine: ComfyUI, installed by the app (a standalone CPython from python-build-standalone,
/// ComfyUI's release source, PyTorch from PyPI, which has the Metal (MPS) backend on Apple Silicon) so the user never
/// sets up Python, then run hidden on a loopback port and driven through its HTTP and WebSocket API. Outputs are copied
/// to ~/Pictures/LLM Hub. A port of core/LLMHub.Core/Services/Studio/ComfyService.cs.
@MainActor @Observable
final class ComfyService {
    private static let pythonTag = "3.12."
    private static let userAgent = "LLMHub-Desktop/0.1"

    /// Install or download progress, or the last error.
    var status = ""
    /// "ComfyUI connected · 127.0.0.1:8188" in the Studio header.
    var statusLine = ""
    var isRunning = false
    private(set) var url: String?

    @ObservationIgnored private var process: Process?
    @ObservationIgnored private var starting: Task<Void, Error>?
    @ObservationIgnored private var outputTail = ""
    @ObservationIgnored private let session: URLSession = {
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 600
        config.httpAdditionalHeaders = ["User-Agent": ComfyService.userAgent]
        return URLSession(configuration: config)
    }()

    init() {
        statusLine = L.s(isInstalled ? "ComfyReady" : "ComfyNotInstalled")
        NotificationCenter.default.addObserver(forName: NSNotification.Name("NSApplicationWillTerminateNotification"), object: nil, queue: .main) { [weak self] _ in
            MainActor.assumeIsolated { self?.stop() }
        }
    }

    nonisolated static var root: URL { AppPaths.engines.appendingPathComponent("comfyui", isDirectory: true) }
    nonisolated private static var dataDir: URL { root.appendingPathComponent("data", isDirectory: true) }
    nonisolated static var modelsDir: URL { dataDir.appendingPathComponent("models", isDirectory: true) }
    nonisolated private static var pythonExe: URL { root.appendingPathComponent("python/bin/python3") }
    nonisolated private static var marker: URL { root.appendingPathComponent("installed.txt") }

    /// ~/Pictures/LLM Hub, where every Studio output is saved.
    nonisolated static var gallery: URL {
        let pictures = FileManager.default.urls(for: .picturesDirectory, in: .userDomainMask).first ?? AppPaths.root
        let url = pictures.appendingPathComponent("LLM Hub", isDirectory: true)
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    var isInstalled: Bool { FileManager.default.fileExists(atPath: Self.marker.path) }

    // MARK: - Install and start

    func ensureRunning() async throws {
        if let process, process.isRunning, url != nil { return }
        if let starting { return try await starting.value }
        let task = Task { @MainActor in
            if !isInstalled { try await install() }
            try await start()
        }
        starting = task
        defer { starting = nil }
        try await task.value
    }

    private func install() async throws {
        let fm = FileManager.default
        try fm.createDirectory(at: Self.root, withIntermediateDirectories: true)

        // 1. A standalone Python (the relocatable CPython builds ComfyUI's own desktop app uses too).
        if !fm.fileExists(atPath: Self.pythonExe.path) {
            #if arch(arm64)
            let triple = "aarch64-apple-darwin"
            #else
            let triple = "x86_64-apple-darwin"
            #endif
            guard let asset = try await gitHubAsset("astral-sh/python-build-standalone", scan: 3, where: {
                $0.hasPrefix("cpython-" + Self.pythonTag) && $0.hasSuffix("-\(triple)-install_only.tar.gz")
            }) else { throw Self.error("No standalone Python for this Mac") }
            let archive = Self.root.appendingPathComponent("python.tar.gz")
            try await download(asset, to: archive, size: 0) { [weak self] done, total in
                self?.status = L.f("ComfyStepPython", total > 0 ? String(format: "%.0f", Double(done) * 100 / Double(total)) : "…")
            }
            try await Self.run("/usr/bin/tar", ["-xzf", archive.path, "-C", Self.root.path])
            try? fm.removeItem(at: archive)
        }

        // 2. ComfyUI's latest release source.
        let source = Self.root.appendingPathComponent("ComfyUI", isDirectory: true)
        if !fm.fileExists(atPath: source.appendingPathComponent("main.py").path) {
            let (data, _) = try await session.data(from: URL(string: "https://api.github.com/repos/Comfy-Org/ComfyUI/releases/latest")!)
            guard let tag = (try JSONSerialization.jsonObject(with: data) as? [String: Any])?["tag_name"] as? String else {
                throw Self.error("Could not find ComfyUI's latest release")
            }
            let zip = Self.root.appendingPathComponent("comfyui.zip")
            try await download("https://github.com/Comfy-Org/ComfyUI/archive/refs/tags/\(tag).zip", to: zip, size: 0) { [weak self] done, total in
                self?.status = L.f("ComfyStepSource", total > 0 ? String(format: "%.0f", Double(done) * 100 / Double(total)) : "…")
            }
            let unpacked = Self.root.appendingPathComponent("src", isDirectory: true)
            try? fm.removeItem(at: unpacked)
            try await Self.run("/usr/bin/ditto", ["-x", "-k", zip.path, unpacked.path])
            try? fm.removeItem(at: zip)
            try? fm.removeItem(at: source)
            guard let folder = try fm.contentsOfDirectory(at: unpacked, includingPropertiesForKeys: nil).first(where: { $0.hasDirectoryPath }) else {
                throw Self.error("ComfyUI archive was empty")
            }
            try fm.moveItem(at: folder, to: source)
            try? fm.removeItem(at: unpacked)
            Log.info("ComfyUI \(tag) unpacked")
        }

        // 3. PyTorch (PyPI's macOS wheels include the Metal backend), then ComfyUI's requirements.
        status = L.s("ComfyStepTorch")
        try await pip(["install", "--upgrade", "pip"])
        try await pip(["install", "torch", "torchvision", "torchaudio"])
        status = L.s("ComfyStepRequirements")
        try await pip(["install", "-r", source.appendingPathComponent("requirements.txt").path])
        try "pypi".write(to: Self.marker, atomically: true, encoding: .utf8)
        status = ""
        Log.info("ComfyUI installed with torch from PyPI (MPS)")
    }

    private func pip(_ args: [String]) async throws {
        let process = Process()
        process.executableURL = Self.pythonExe
        process.arguments = ["-m", "pip"] + args + ["--disable-pip-version-check", "--no-input"]
        let out = Pipe(), err = Pipe()
        process.standardOutput = out
        process.standardError = err
        let tail = TailBuffer()
        out.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let text = String(decoding: handle.availableData, as: UTF8.self)
            guard let line = text.split(separator: "\n").last(where: { $0.hasPrefix("Collecting") || $0.hasPrefix("Downloading") || $0.hasPrefix("Installing") }) else { return }
            let short = String(line.trimmingCharacters(in: .whitespaces).prefix(90))
            Task { @MainActor in self?.status = short }
        }
        err.fileHandleForReading.readabilityHandler = { handle in tail.append(String(decoding: handle.availableData, as: UTF8.self)) }
        try process.run()
        await Self.wait(process)
        out.fileHandleForReading.readabilityHandler = nil
        err.fileHandleForReading.readabilityHandler = nil
        guard process.terminationStatus == 0 else {
            throw Self.error("pip \(args.prefix(3).joined(separator: " ")) failed: \(tail.text.suffix(600))")
        }
    }

    private func start() async throws {
        // --base-directory expects its folders to exist (ComfyUI scans custom_nodes before creating anything).
        for folder in ["models", "custom_nodes", "input", "output", "temp", "user"] {
            try FileManager.default.createDirectory(at: Self.dataDir.appendingPathComponent(folder), withIntermediateDirectories: true)
        }
        let port = Self.freePort()
        let process = Process()
        process.executableURL = Self.pythonExe
        process.currentDirectoryURL = Self.root.appendingPathComponent("ComfyUI", isDirectory: true)
        process.arguments = ["-s", "main.py", "--listen", "127.0.0.1", "--port", "\(port)", "--base-directory", Self.dataDir.path,
                             "--disable-auto-launch", "--preview-method", "none"]
        var environment = ProcessInfo.processInfo.environment
        environment["PYTHONUNBUFFERED"] = "1"
        // A few PyTorch operators have no Metal kernel yet; run those on the CPU instead of failing.
        environment["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        process.environment = environment
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        outputTail = ""
        pipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let text = String(decoding: handle.availableData, as: UTF8.self)
            Task { @MainActor in
                guard let self else { return }
                self.outputTail = String((self.outputTail + text).suffix(8000))
            }
        }
        process.terminationHandler = { [weak self] ended in
            Task { @MainActor in
                guard let self, self.process === ended else { return }
                Log.error("ComfyUI exited with \(ended.terminationStatus): \(self.outputTail.suffix(1500))")
                self.process = nil
                self.url = nil
                self.isRunning = false
                self.statusLine = L.s(self.isInstalled ? "ComfyReady" : "ComfyNotInstalled")
            }
        }
        try process.run()
        self.process = process
        startWatchdog(pid: process.processIdentifier)
        status = L.s("ComfyStarting")

        let base = "http://127.0.0.1:\(port)"
        let deadline = Date().addingTimeInterval(240)
        while Date() < deadline {
            guard process.isRunning else { throw Self.error("ComfyUI exited with \(process.terminationStatus): \(outputTail.suffix(1200))") }
            if let (data, response) = try? await session.data(from: URL(string: base + "/system_stats")!),
               (response as? HTTPURLResponse)?.statusCode == 200 {
                url = base
                let stats = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
                let device = ((stats?["devices"] as? [[String: Any]])?.first?["name"] as? String) ?? ""
                Log.info("ComfyUI ready on \(port) (\(device))")
                isRunning = true
                statusLine = L.f("ComfyConnected", "127.0.0.1:\(port)")
                status = ""
                return
            }
            try await Task.sleep(for: .milliseconds(500))
        }
        throw Self.error("ComfyUI did not start: \(outputTail.suffix(1200))")
    }

    /// If the app dies without cleaning up (crash, force quit), this shell loop stops ComfyUI instead of leaving
    /// gigabytes of models in memory. It exits by itself when ComfyUI stops first.
    private func startWatchdog(pid: Int32) {
        let app = ProcessInfo.processInfo.processIdentifier
        let watchdog = Process()
        watchdog.executableURL = URL(fileURLWithPath: "/bin/sh")
        watchdog.arguments = ["-c", "while kill -0 \(app) 2>/dev/null && kill -0 \(pid) 2>/dev/null; do sleep 2; done; "
            + "kill -0 \(app) 2>/dev/null || { ps -p \(pid) -o command= | grep -q main.py && kill \(pid); }"]
        do { try watchdog.run() } catch { Log.error("ComfyUI watchdog did not start", error) }
    }

    func stop() {
        if let process, process.isRunning { process.terminate() }
        process = nil
        url = nil
        isRunning = false
        statusLine = L.s(isInstalled ? "ComfyReady" : "ComfyNotInstalled")
    }

    // MARK: - Models

    nonisolated static func hasModels(_ workflow: ComfyWorkflow) -> Bool {
        workflow.files.allSatisfy { FileManager.default.fileExists(atPath: modelsDir.appendingPathComponent("\($0.folder)/\($0.name)").path) }
    }

    func ensureModels(_ workflow: ComfyWorkflow) async throws {
        let total = Double(max(workflow.downloadSize, 1))
        var done: Int64 = 0
        for file in workflow.files {
            let target = Self.modelsDir.appendingPathComponent("\(file.folder)/\(file.name)")
            if !FileManager.default.fileExists(atPath: target.path) {
                try FileManager.default.createDirectory(at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
                let before = done
                try await download(file.url, to: target, size: file.size) { [weak self] bytes, _ in
                    self?.status = L.f("ComfyDownloadingModel", file.name, String(format: "%.0f", Double(before + bytes) * 100 / total))
                }
                Log.info("Studio downloaded \(file.name)")
            }
            done += file.size
        }
        status = ""
    }

    // MARK: - Run

    /// Uploads an input image (for video and upscale) and returns the name ComfyUI knows it by.
    func upload(_ path: URL) async throws -> String {
        guard let url else { throw Self.error("ComfyUI is not running") }
        let boundary = "LLMHub-\(UUID().uuidString)"
        let ext = path.pathExtension.lowercased()
        var body = Data()
        func field(_ text: String) { body.append(Data(text.utf8)) }
        field("--\(boundary)\r\nContent-Disposition: form-data; name=\"image\"; filename=\"\(path.lastPathComponent)\"\r\n")
        field("Content-Type: image/\(ext == "jpg" ? "jpeg" : ext)\r\n\r\n")
        body.append(try Data(contentsOf: path))
        field("\r\n--\(boundary)\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n--\(boundary)--\r\n")
        var request = URLRequest(url: URL(string: url + "/upload/image")!)
        request.httpMethod = "POST"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        let (data, response) = try await session.upload(for: request, from: body)
        guard (response as? HTTPURLResponse)?.statusCode == 200,
              let name = (try JSONSerialization.jsonObject(with: data) as? [String: Any])?["name"] as? String else {
            throw Self.error("Upload failed: \(String(decoding: data, as: UTF8.self).prefix(300))")
        }
        return name
    }

    /// Queues a graph, reports progress (0...1), and returns the output files copied into the gallery.
    func run(_ graph: [String: Any], progress: @escaping (Double, String) -> Void) async throws -> [URL] {
        guard let url else { throw Self.error("ComfyUI is not running") }
        let clientId = UUID().uuidString.replacingOccurrences(of: "-", with: "").lowercased()
        let socket = session.webSocketTask(with: URL(string: url.replacingOccurrences(of: "http://", with: "ws://") + "/ws?clientId=\(clientId)")!)
        socket.maximumMessageSize = 64 << 20
        socket.resume()
        defer { socket.cancel(with: .normalClosure, reason: nil) }

        var request = URLRequest(url: URL(string: url + "/prompt")!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: ["prompt": graph, "client_id": clientId])
        let (data, response) = try await session.data(for: request)
        guard (response as? HTTPURLResponse)?.statusCode == 200,
              let promptId = (try? JSONSerialization.jsonObject(with: data) as? [String: Any])?["prompt_id"] as? String else {
            throw Self.error(Self.describeError(data))
        }

        let interrupt = { [session] in
            var stop = URLRequest(url: URL(string: url + "/interrupt")!)
            stop.httpMethod = "POST"
            session.dataTask(with: stop).resume()
        }
        return try await withTaskCancellationHandler {
            while true {
                try Task.checkCancellation()
                let message: URLSessionWebSocketTask.Message
                do { message = try await socket.receive() } catch {
                    if Task.isCancelled { throw CancellationError() }
                    break // Socket closed: fall back to the history.
                }
                guard case .string(let text) = message,
                      let event = try? JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any] else { continue }
                let payload = event["data"] as? [String: Any] ?? [:]
                if let id = payload["prompt_id"] as? String, id != promptId { continue }
                switch event["type"] as? String {
                case "progress":
                    let value = (payload["value"] as? NSNumber)?.doubleValue ?? 0
                    let max = Swift.max(1, (payload["max"] as? NSNumber)?.doubleValue ?? 1)
                    progress(value / max, L.f("ComfyStep", String(format: "%.0f", value), String(format: "%.0f", max)))
                case "executing" where payload["node"] == nil || payload["node"] is NSNull:
                    return try await collect(promptId)
                case "execution_success":
                    return try await collect(promptId)
                case "execution_error":
                    throw Self.error(payload["exception_message"] as? String ?? "ComfyUI error")
                case "execution_interrupted":
                    throw CancellationError()
                default:
                    break
                }
            }
            return try await collect(promptId)
        } onCancel: {
            interrupt()
        }
    }

    private func collect(_ promptId: String) async throws -> [URL] {
        guard let url else { throw Self.error("ComfyUI is not running") }
        let (data, _) = try await session.data(from: URL(string: "\(url)/history/\(promptId)")!)
        let history = ((try? JSONSerialization.jsonObject(with: data) as? [String: Any])?[promptId] as? [String: Any])?["outputs"] as? [String: Any] ?? [:]
        let stamp = DateFormatter()
        stamp.dateFormat = "yyyyMMdd-HHmmss"
        var saved: [URL] = []
        for nodeId in history.keys.sorted() {
            guard let output = history[nodeId] as? [String: Any] else { continue }
            for key in ["images", "gifs", "videos"] {
                for file in output[key] as? [[String: Any]] ?? [] {
                    let name = file["filename"] as? String ?? ""
                    let subfolder = file["subfolder"] as? String ?? ""
                    let type = file["type"] as? String ?? "output"
                    guard type == "output", !name.isEmpty else { continue }
                    var components = URLComponents(string: "\(url)/view")!
                    components.queryItems = [URLQueryItem(name: "filename", value: name), URLQueryItem(name: "subfolder", value: subfolder),
                                             URLQueryItem(name: "type", value: type)]
                    let (bytes, _) = try await session.data(from: components.url!)
                    let target = Self.gallery.appendingPathComponent("\(stamp.string(from: Date()))-\(name.replacingOccurrences(of: "/", with: "_"))")
                    try bytes.write(to: target)
                    saved.append(target)
                }
            }
        }
        Log.info("ComfyUI \(promptId): \(saved.count) output(s)")
        return saved
    }

    private static func describeError(_ body: Data) -> String {
        guard let node = try? JSONSerialization.jsonObject(with: body) as? [String: Any] else { return String(decoding: body, as: UTF8.self) }
        if let errors = node["node_errors"] as? [String: Any], let first = (errors.values.first as? [String: Any])?["errors"] as? [[String: Any]],
           let error = first.first, let details = error["details"] as? String, !details.isEmpty {
            return "\(error["message"] as? String ?? ""): \(details)"
        }
        return ((node["error"] as? [String: Any])?["message"] as? String) ?? String(decoding: body, as: UTF8.self)
    }

    // MARK: - Helpers

    private func gitHubAsset(_ repo: String, scan: Int, where match: (String) -> Bool) async throws -> String? {
        let (data, _) = try await session.data(from: URL(string: "https://api.github.com/repos/\(repo)/releases?per_page=\(scan)")!)
        for release in (try JSONSerialization.jsonObject(with: data) as? [[String: Any]]) ?? [] {
            for asset in release["assets"] as? [[String: Any]] ?? [] {
                if let name = asset["name"] as? String, match(name) { return asset["browser_download_url"] as? String }
            }
        }
        return nil
    }

    /// Resumable download through the app's FileDownloader; `report` gets (bytes, total) on the main actor.
    private func download(_ source: String, to target: URL, size: Int64, report: @escaping @MainActor (Int64, Int64) -> Void) async throws {
        var request = URLRequest(url: URL(string: source)!)
        request.setValue(Self.userAgent, forHTTPHeaderField: "User-Agent")
        var total = size
        if total == 0 {
            var head = request
            head.httpMethod = "HEAD"
            if let (_, response) = try? await session.data(for: head) { total = max(0, response.expectedContentLength) }
        }
        let expected = total
        try await Self.resumableDownload(request: request, to: target, expectedSize: size) { bytes in
            Task { @MainActor in report(bytes, expected) }
        }
    }

    /// FileDownloader with up to six attempts: multi-GB model downloads over a flaky connection resume where they stopped.
    nonisolated static func resumableDownload(request: URLRequest, to target: URL, expectedSize: Int64,
                                              progress: @escaping (Int64) -> Void) async throws {
        var attempt = 0
        while true {
            do {
                try await FileDownloader.download(request: request, to: target, expectedSize: expectedSize, sha256: nil, progress: progress)
                return
            } catch let error as URLError where error.code != .cancelled && attempt < 5 && !Task.isCancelled {
                attempt += 1
                Log.info("Download of \(target.lastPathComponent) interrupted (\(error.code.rawValue)); retry \(attempt) of 5")
                try await Task.sleep(for: .seconds(Double(attempt * 3)))
            }
        }
    }

    nonisolated static func run(_ tool: String, _ args: [String]) async throws {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: tool)
        process.arguments = args
        try process.run()
        await wait(process)
        guard process.terminationStatus == 0 else { throw error("\(tool) failed with \(process.terminationStatus)") }
    }

    nonisolated static func wait(_ process: Process) async {
        await withCheckedContinuation { continuation in
            DispatchQueue.global().async {
                process.waitUntilExit()
                continuation.resume()
            }
        }
    }

    nonisolated static func error(_ message: String) -> NSError {
        NSError(domain: "LLMHub.Studio", code: 1, userInfo: [NSLocalizedDescriptionKey: message])
    }

    private static func freePort() -> Int {
        let fd = socket(AF_INET, SOCK_STREAM, 0)
        defer { close(fd) }
        var addr = sockaddr_in()
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_addr.s_addr = inet_addr("127.0.0.1")
        addr.sin_port = 0
        var len = socklen_t(MemoryLayout<sockaddr_in>.size)
        _ = withUnsafeMutablePointer(to: &addr) { $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { bind(fd, $0, len) } }
        _ = withUnsafeMutablePointer(to: &addr) { $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { getsockname(fd, $0, &len) } }
        return Int(UInt16(bigEndian: addr.sin_port))
    }
}

/// The last few KB of a process's stderr, appended from a pipe handler thread.
final class TailBuffer: @unchecked Sendable {
    private let lock = NSLock()
    private var value = ""
    func append(_ text: String) {
        lock.lock()
        value = String((value + text).suffix(8000))
        lock.unlock()
    }
    var text: String {
        lock.lock()
        defer { lock.unlock() }
        return value
    }
}
