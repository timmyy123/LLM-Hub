import Foundation
import Observation

enum EngineState { case idle, installing, loading, ready, error }

/// Runs models with the official prebuilt llama.cpp release for macOS (Metal), downloaded on first use,
/// as a llama-server child process on a random localhost port.
@MainActor @Observable
final class LlamaEngine {
    private static let releasesAPI = URL(string: "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=20")!

    private(set) var state = EngineState.idle
    private(set) var statusText = L.s("EngineIdle")
    private(set) var loadedModel: InstalledModel?
    private(set) var baseURL: URL?

    private var process: Process?
    private var outputTail = ""
    private var loading: Task<Void, Error>?

    private var engineRoot: URL { AppPaths.engines.appendingPathComponent("llama.cpp", isDirectory: true) }
    /// Each server keeps its own pid file, so the embedding server never stops the chat model's and vice versa.
    private var pidFile: URL { engineRoot.appendingPathComponent(embeddings ? "embeddings.pid" : "server.pid") }
    /// A second, small server for /v1/embeddings (EmbeddingGemma): started with --embeddings, alongside the chat model.
    let embeddings: Bool

    /// If the app dies without cleaning up (crash, force quit), this tiny shell loop stops llama-server instead of
    /// leaving the model in memory. It exits on its own when the server stops first, and checks the process name so a
    /// reused PID is never touched.
    private func startWatchdog(serverPid: Int32) {
        let app = ProcessInfo.processInfo.processIdentifier
        let watchdog = Process()
        watchdog.executableURL = URL(fileURLWithPath: "/bin/sh")
        watchdog.arguments = ["-c", "while kill -0 \(app) 2>/dev/null && kill -0 \(serverPid) 2>/dev/null; do sleep 2; done; "
            + "kill -0 \(app) 2>/dev/null || { ps -p \(serverPid) -o comm= | grep -q llama-server && kill \(serverPid); rm -f '\(pidFile.path)'; }"]
        do { try watchdog.run() } catch { Log.error("llama-server watchdog did not start", error) }
    }
    private var variant: String {
        #if arch(arm64)
        "macos-arm64"
        #else
        "macos-x64"
        #endif
    }

    init(embeddings: Bool = false) {
        self.embeddings = embeddings
        killStaleServer()
    }

    /// Context window of the running server; asking for a larger one reloads the model.
    private(set) var loadedContext = 0

    static func context(for model: InstalledModel, wanted: Int) -> Int { min(model.contextLength > 0 ? model.contextLength : 8192, wanted) }

    /// Loads <paramref name="model"/> with the chat context, or <paramref name="contextLength"/> for the Agent, Design
    /// and the Local API (Claude Code's own prompt is about 18K tokens).
    func load(_ model: InstalledModel, contextLength: Int? = nil) async throws {
        let wanted = Self.context(for: model, wanted: contextLength ?? AppSettings.current.contextLength)
        if loadedModel?.id == model.id, process?.isRunning == true, loadedContext >= wanted { return }
        stopProcess()
        do {
            let server = try await ensureInstalled()
            set(.loading, L.f("LoadingModel", model.name))
            try await startServer(server, model, context: wanted)
            loadedContext = wanted
            loadedModel = model
            set(.ready, L.f("EngineReady", model.name))
        } catch is CancellationError {
            stopProcess()
            set(.idle, L.s("EngineIdle"))
            throw CancellationError()
        } catch {
            Log.error("Engine load failed", error)
            stopProcess()
            set(.error, L.f("EngineError", error.localizedDescription))
            throw error
        }
    }

    func stop() {
        stopProcess()
        set(.idle, L.s("EngineIdle"))
    }

    var isServerRunning: Bool { process?.isRunning == true }

    private func set(_ state: EngineState, _ text: String) {
        self.state = state
        statusText = text
    }

    private func stopProcess() {
        if let process, process.isRunning { process.terminate() }
        process = nil
        baseURL = nil
        loadedModel = nil
        try? FileManager.default.removeItem(at: pidFile)
    }

    /// A crash can leave llama-server running; the PID file lets the next launch clean it up.
    private func killStaleServer() {
        guard let text = try? String(contentsOf: pidFile, encoding: .utf8), let pid = Int32(text.trimmingCharacters(in: .whitespacesAndNewlines)) else { return }
        var buffer = [CChar](repeating: 0, count: 1024)
        if proc_pidpath(pid, &buffer, UInt32(buffer.count)) > 0, String(cString: buffer).hasSuffix("llama-server") {
            kill(pid, SIGTERM)
            Log.info("Stopped stale llama-server \(pid)")
        }
        try? FileManager.default.removeItem(at: pidFile)
    }

    /// The chat and embedding servers share one llama.cpp install; a second caller waits for the first download.
    private static var installing: Task<URL, Error>?

    private func ensureInstalled() async throws -> URL {
        if let running = Self.installing { return try await running.value }
        let task = Task { try await self.install() }
        Self.installing = task
        defer { Self.installing = nil }
        return try await task.value
    }

    private func install() async throws -> URL {
        try? FileManager.default.createDirectory(at: engineRoot, withIntermediateDirectories: true)
        let dirs = ((try? FileManager.default.contentsOfDirectory(at: engineRoot, includingPropertiesForKeys: nil)) ?? [])
            .filter { $0.lastPathComponent.hasSuffix("-\(variant)") }
            .sorted { $0.lastPathComponent > $1.lastPathComponent }
        for dir in dirs {
            if let server = Self.findServer(in: dir) { return server }
        }

        set(.installing, L.s("InstallingEngine"))
        var request = URLRequest(url: Self.releasesAPI)
        request.setValue("LLMHub-Desktop/0.1", forHTTPHeaderField: "User-Agent")
        let (data, _) = try await URLSession.shared.data(for: request)
        let releases = try JSONSerialization.jsonObject(with: data) as? [[String: Any]] ?? []
        for release in releases {
            guard let tag = release["tag_name"] as? String, tag.hasPrefix("b") else { continue } // Build tags look like b11836.
            let assets = release["assets"] as? [[String: Any]] ?? []
            guard let asset = assets.first(where: { $0["name"] as? String == "llama-\(tag)-bin-\(variant).tar.gz" }),
                  let urlString = asset["browser_download_url"] as? String, let url = URL(string: urlString) else { continue }

            let archive = engineRoot.appendingPathComponent("\(tag)-\(variant).tar.gz")
            let target = engineRoot.appendingPathComponent("\(tag)-\(variant)", isDirectory: true)
            set(.installing, L.f("DownloadingEngine", "0"))
            let (tmp, response) = try await URLSession.shared.download(from: url)
            try HuggingFaceClient.check(response)
            try? FileManager.default.removeItem(at: archive)
            try FileManager.default.moveItem(at: tmp, to: archive)
            try FileManager.default.createDirectory(at: target, withIntermediateDirectories: true)
            try await Self.run("/usr/bin/tar", ["-xzf", archive.path, "-C", target.path])
            try? FileManager.default.removeItem(at: archive)
            guard let server = Self.findServer(in: target) else { throw NSError(domain: "LLMHub", code: 2, userInfo: [NSLocalizedDescriptionKey: "llama-server not found in release"]) }
            Log.info("Installed llama.cpp \(tag) (\(variant))")
            return server
        }
        throw NSError(domain: "LLMHub", code: 3, userInfo: [NSLocalizedDescriptionKey: "No llama.cpp build found for this Mac"])
    }

    private static func findServer(in dir: URL) -> URL? {
        let enumerator = FileManager.default.enumerator(at: dir, includingPropertiesForKeys: nil)
        while let url = enumerator?.nextObject() as? URL {
            if url.lastPathComponent == "llama-server" { return url }
        }
        return nil
    }

    private static func run(_ tool: String, _ args: [String]) async throws {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: tool)
        process.arguments = args
        try process.run()
        await withCheckedContinuation { continuation in
            DispatchQueue.global().async {
                process.waitUntilExit()
                continuation.resume()
            }
        }
        guard process.terminationStatus == 0 else {
            throw NSError(domain: "LLMHub", code: Int(process.terminationStatus), userInfo: [NSLocalizedDescriptionKey: "\(tool) failed"])
        }
    }

    private func startServer(_ server: URL, _ model: InstalledModel, context: Int) async throws {
        let port = Self.freePort()
        var args = ["-m", model.primaryPath.path, "--host", "127.0.0.1", "--port", "\(port)", "--alias", model.id,
                    "-ngl", "999", "-c", "\(context)", embeddings ? "--embeddings" : "--jinja"]
        if !embeddings, let mmproj = model.mmprojPath, FileManager.default.fileExists(atPath: mmproj.path) {
            args += ["--mmproj", mmproj.path]
        }

        let process = Process()
        process.executableURL = server
        process.currentDirectoryURL = server.deletingLastPathComponent()
        process.arguments = args
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
                Log.error("llama-server exited with \(ended.terminationStatus): \(self.outputTail.suffix(2000))")
                self.process = nil
                self.loadedModel = nil
                self.baseURL = nil
                self.set(.error, L.f("EngineError", "exit \(ended.terminationStatus)"))
            }
        }
        try process.run()
        self.process = process
        try? "\(process.processIdentifier)".write(to: pidFile, atomically: true, encoding: .utf8)
        startWatchdog(serverPid: process.processIdentifier)
        Log.info("Started llama-server on \(port) for \(model.id)")

        let base = URL(string: "http://127.0.0.1:\(port)")!
        let deadline = Date().addingTimeInterval(600)
        while Date() < deadline {
            try Task.checkCancellation()
            guard process.isRunning else {
                let last = outputTail.split(separator: "\n").last.map(String.init) ?? "llama-server exited"
                throw NSError(domain: "LLMHub", code: 4, userInfo: [NSLocalizedDescriptionKey: last])
            }
            var request = URLRequest(url: base.appendingPathComponent("health"))
            request.timeoutInterval = 5
            if let (_, response) = try? await URLSession.shared.data(for: request), (response as? HTTPURLResponse)?.statusCode == 200 {
                baseURL = base
                return
            }
            try await Task.sleep(for: .milliseconds(400))
        }
        throw NSError(domain: "LLMHub", code: 5, userInfo: [NSLocalizedDescriptionKey: "Model took too long to load"])
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

/// One message. Images (vision models) are sent as OpenAI image_url parts with data URLs.
struct ChatTurn: Encodable {
    let role: String
    let content: String
    var images: [URL] = []

    init(role: String, content: String, images: [URL] = []) {
        self.role = role
        self.content = content
        self.images = images
    }

    private enum CodingKeys: String, CodingKey { case role, content }

    func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(role, forKey: .role)
        guard !images.isEmpty else {
            try c.encode(content, forKey: .content)
            return
        }
        struct Part: Encodable {
            struct Image: Encodable { let url: String }
            let type: String
            var text: String?
            var image_url: Image?
        }
        var parts = images.compactMap { url -> Part? in
            guard let data = try? Data(contentsOf: url) else { return nil }
            let mime = url.pathExtension.lowercased() == "png" ? "image/png" : url.pathExtension.lowercased() == "webp" ? "image/webp" : "image/jpeg"
            return Part(type: "image_url", image_url: .init(url: "data:\(mime);base64,\(data.base64EncodedString())"))
        }
        parts.append(Part(type: "text", text: content))
        try c.encode(parts, forKey: .content)
    }
}

/// Per-request switches from the composer.
struct ChatOptions {
    /// Extended thinking: models with a reasoning switch in their chat template think first.
    var thinking = false
}

/// Streams replies from llama-server's OpenAI-compatible /v1/chat/completions endpoint.
enum ChatClient {
    struct Delta {
        var content: String?
        var reasoning: String?
        var tokensPerSecond: Double?
        /// Prompt plus generated tokens after this reply, for the composer's context meter.
        var contextTokens: Int?
    }

    static func stream(baseURL: URL, model: String, turns: [ChatTurn], options: ChatOptions = .init()) -> AsyncThrowingStream<Delta, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    var request = URLRequest(url: baseURL.appendingPathComponent("v1/chat/completions"))
                    request.httpMethod = "POST"
                    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
                    request.timeoutInterval = 3600
                    struct Kwargs: Encodable {
                        let enable_thinking: Bool
                        let thinking: Bool
                    }
                    struct Body: Encodable {
                        let model: String
                        let stream = true
                        let messages: [ChatTurn]
                        let chat_template_kwargs: Kwargs
                    }
                    request.httpBody = try JSONEncoder().encode(Body(model: model, messages: turns,
                                                                     chat_template_kwargs: .init(enable_thinking: options.thinking, thinking: options.thinking)))
                    let (bytes, response) = try await URLSession.shared.bytes(for: request)
                    try HuggingFaceClient.check(response)
                    for try await line in bytes.lines {
                        guard line.hasPrefix("data: ") else { continue }
                        let payload = line.dropFirst(6)
                        if payload == "[DONE]" { break }
                        guard let json = try JSONSerialization.jsonObject(with: Data(payload.utf8)) as? [String: Any] else { continue }
                        if let error = json["error"] { throw NSError(domain: "LLMHub", code: 6, userInfo: [NSLocalizedDescriptionKey: "\(error)"]) }
                        if let timings = json["timings"] as? [String: Any], let speed = timings["predicted_per_second"] as? Double {
                            let used = ["prompt_n", "cache_n", "predicted_n"].compactMap { timings[$0] as? Int }.reduce(0, +)
                            continuation.yield(Delta(tokensPerSecond: speed, contextTokens: used > 0 ? used : nil))
                        }
                        guard let choices = json["choices"] as? [[String: Any]], let delta = choices.first?["delta"] as? [String: Any] else { continue }
                        let d = Delta(content: delta["content"] as? String, reasoning: delta["reasoning_content"] as? String)
                        if !(d.content ?? "").isEmpty || !(d.reasoning ?? "").isEmpty { continuation.yield(d) }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }
}
