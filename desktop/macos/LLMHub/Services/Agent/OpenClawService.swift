import CryptoKit
import Foundation
import Observation

/// The OpenClaw agent engine for the Agent tab. OpenClaw (MIT, openclaw.ai) is installed by the app: a private Node runtime
/// from nodejs.org (checksum verified), then a pinned openclaw from npm with every extension and skill of Chinese origin
/// deleted. Its Gateway runs warm on a loopback port, pointed only at LLM Hub's Agent bridge, with channels, Bonjour,
/// update checks and other plugins off. Each session talks to it through `openclaw acp` (the Agent Client Protocol,
/// JSON-RPC over stdio). Same behavior as core Services/Agent/OpenClawService.cs.
@MainActor @Observable
final class OpenClawService {
    static let version = "2026.10.7"
    private static let nodeVersion = "24.22.1"

    /// Removed after install. In the npm package: alibaba (DashScope/Wan), minimax, kie (Kling, Hailuo, Seedance, Wan),
    /// senseaudio. The rest are separate @openclaw packages that are never installed, deleted too should a version ship them.
    private static let strippedExtensions = [
        "alibaba", "minimax", "kie", "senseaudio", "byteplus", "volcengine", "feishu", "deepseek", "kimi-coding", "moonshot", "longcat",
        "qianfan", "qwen", "stepfun", "tencent", "xiaomi", "zai", "pixverse", "fish-audio-speech", "novita", "vydra",
    ]

    /// sherpa-onnx (k2-fsa, Next-gen Kaldi) text to speech.
    private static let strippedSkills = ["sherpa-onnx-tts"]

    private(set) var status = ""
    private(set) var isInstalling = false
    /// Bumped after an install so views re-check `isInstalled`.
    private(set) var installGeneration = 0

    @ObservationIgnored private var gateway: Process?
    @ObservationIgnored private var starting: Task<Void, Error>?
    @ObservationIgnored private var port = 0
    @ObservationIgnored private var token = ""

    static var root: URL { AppPaths.engines.appendingPathComponent("openclaw", isDirectory: true) }
    private static var nodeDir: URL { root.appendingPathComponent("node", isDirectory: true) }
    static var nodeExe: URL { nodeDir.appendingPathComponent("bin/node") }
    private static var npmCli: URL { nodeDir.appendingPathComponent("lib/node_modules/npm/bin/npm-cli.js") }
    private static var appDir: URL { root.appendingPathComponent("app", isDirectory: true) }
    private static var package: URL { appDir.appendingPathComponent("node_modules/openclaw", isDirectory: true) }
    static var entry: URL { package.appendingPathComponent("openclaw.mjs") }
    /// OPENCLAW_HOME: its config, sessions (SQLite) and workspaces.
    private static var home: URL { root.appendingPathComponent("home", isDirectory: true) }
    private static var configFile: URL { home.appendingPathComponent(".openclaw/openclaw.json") }
    private static var tokenFile: URL { root.appendingPathComponent("gateway.token") }
    private static var markerFile: URL { root.appendingPathComponent("installed.txt") }
    private static var pidFile: URL { root.appendingPathComponent("gateway.pid") }

    static var filesPresent: Bool {
        [markerFile, nodeExe, entry].allSatisfy { FileManager.default.fileExists(atPath: $0.path) }
    }

    var isInstalled: Bool {
        _ = installGeneration
        return Self.filesPresent
    }

    // MARK: Install

    func install() async throws {
        guard !isInstalling else { return }
        isInstalling = true
        defer {
            isInstalling = false
            installGeneration += 1
        }
        do {
            try FileManager.default.createDirectory(at: Self.root, withIntermediateDirectories: true)
            try await installNode()
            status = L.s("OpenClawInstallingPackage")
            try FileManager.default.createDirectory(at: Self.appDir, withIntermediateDirectories: true)
            // A private npm cache, deleted afterwards, so ~/.npm is never touched.
            let cache = Self.root.appendingPathComponent("npm-cache", isDirectory: true)
            let (code, output) = try await Self.run(Self.nodeExe, [
                Self.npmCli.path, "install", "--prefix", Self.appDir.path, "openclaw@\(Self.version)", "--omit=dev", "--no-audit", "--no-fund",
                "--cache", cache.path, "--loglevel", "error",
            ], in: Self.appDir)
            try? FileManager.default.removeItem(at: cache)
            guard code == 0, FileManager.default.fileExists(atPath: Self.entry.path) else {
                throw Self.error("npm install openclaw@\(Self.version) failed (\(code)): \(output.suffix(600))")
            }
            Self.strip()
            try "openclaw \(Self.version), node \(Self.nodeVersion)".write(to: Self.markerFile, atomically: true, encoding: .utf8)
            status = ""
            Log.info("OpenClaw \(Self.version) installed with Node \(Self.nodeVersion)")
        } catch {
            Log.error("OpenClaw install failed", error)
            status = L.f("ErrorPrefix", error.localizedDescription)
            throw error
        }
    }

    private func installNode() async throws {
        guard !FileManager.default.fileExists(atPath: Self.nodeExe.path) else { return }
        #if arch(arm64)
        let name = "node-v\(Self.nodeVersion)-darwin-arm64.tar.gz"
        #else
        let name = "node-v\(Self.nodeVersion)-darwin-x64.tar.gz"
        #endif
        let base = URL(string: "https://nodejs.org/dist/v\(Self.nodeVersion)/")!
        let (sums, response) = try await URLSession.shared.data(from: base.appendingPathComponent("SHASUMS256.txt"))
        try HuggingFaceClient.check(response)
        let expected = String(decoding: sums, as: UTF8.self).split(separator: "\n")
            .map { $0.trimmingCharacters(in: .whitespaces).components(separatedBy: "  ") }
            .first { $0.count == 2 && $0[1] == name }?[0]
        guard let expected else { throw Self.error("\(name) is not listed in Node's checksums") }

        let url = base.appendingPathComponent(name)
        var head = URLRequest(url: url)
        head.httpMethod = "HEAD"
        let size = (try? await URLSession.shared.data(for: head).1)?.expectedContentLength ?? 0
        status = L.f("OpenClawDownloadingNode", "0")
        let archive = Self.root.appendingPathComponent(name)
        // FileDownloader verifies the SHA-256 against SHASUMS256.txt and deletes the file on a mismatch.
        try await FileDownloader.download(request: URLRequest(url: url), to: archive, expectedSize: max(size, 0), sha256: expected) { bytes in
            guard size > 0 else { return }
            let percent = String(format: "%.0f", min(100, Double(bytes) * 100 / Double(size)))
            Task { @MainActor in self.status = L.f("OpenClawDownloadingNode", percent) }
        }
        let unpacked = Self.root.appendingPathComponent("node-unpacked", isDirectory: true)
        try? FileManager.default.removeItem(at: unpacked)
        try FileManager.default.createDirectory(at: unpacked, withIntermediateDirectories: true)
        let (code, output) = try await Self.run(URL(fileURLWithPath: "/usr/bin/tar"), ["-xzf", archive.path, "-C", unpacked.path], in: Self.root)
        try? FileManager.default.removeItem(at: archive)
        guard code == 0, let folder = try FileManager.default.contentsOfDirectory(at: unpacked, includingPropertiesForKeys: nil).first else {
            throw Self.error("Node could not be unpacked: \(output.suffix(300))")
        }
        try? FileManager.default.removeItem(at: Self.nodeDir)
        try FileManager.default.moveItem(at: folder, to: Self.nodeDir)
        try? FileManager.default.removeItem(at: unpacked)
    }

    /// Deletes the Chinese-origin extensions and skills from the installed package.
    private static func strip() {
        var removed: [String] = []
        let fm = FileManager.default
        for (roots, ids) in [(["dist/extensions", "extensions"], strippedExtensions), (["skills", "dist/skills"], strippedSkills)] {
            for root in roots {
                for id in ids {
                    let dir = package.appendingPathComponent(root, isDirectory: true).appendingPathComponent(id, isDirectory: true)
                    guard fm.fileExists(atPath: dir.path) else { continue }
                    try? fm.removeItem(at: dir)
                    if !removed.contains(id) { removed.append(id) }
                }
            }
        }
        Log.info("OpenClaw: removed \(removed.joined(separator: ", "))")
    }

    func remove() {
        stopGateway()
        try? FileManager.default.removeItem(at: Self.root)
        installGeneration += 1
    }

    // MARK: Config and gateway

    /// openclaw.json: one provider, LLM Hub's Agent bridge, offering the installed chat models; one agent per session with
    /// the session's folder as its workspace and the session's model. Plugins other than memory are off; commands ask first.
    /// The Gateway reloads this file when it changes.
    private func writeConfig(_ sessions: [AgentSession], bridge: LocalApiServer, models chatModels: [InstalledModel]) throws {
        let models: [[String: Any]] = chatModels.map {
            ["id": bridge.apiName($0), "name": $0.name, "reasoning": $0.tags.contains("thinking"), "input": ["text"],
             "cost": ["input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0],
             "contextWindow": AppSettings.current.apiContextLength, "maxTokens": 4096]
        }
        // One agent per project folder in use: the Gateway prepares a database for every listed agent when it starts, so
        // folders that are not open stay out (their history stays on disk). Sessions in a folder differ by session key.
        var entries: [String: Any] = [:]
        for session in sessions where entries[Self.agentId(session)] == nil { entries[Self.agentId(session)] = ["workspace": session.folder] }
        let primary = chatModels.first { $0.id == sessions.last?.info.modelId } ?? chatModels.first
        let workspace = Self.root.appendingPathComponent("workspace", isDirectory: true)
        // Keys OpenClaw adds itself (meta, skills, wizard) are kept: a file that loses them looks clobbered to the Gateway,
        // which then restores its last-known-good copy, with an old port and token.
        var config = (try? JSONSerialization.jsonObject(with: Data(contentsOf: Self.configFile))) as? [String: Any] ?? [:]
        let ours: [String: Any] = [
            "gateway": ["mode": "local", "bind": "loopback", "port": port, "auth": ["mode": "token", "token": token]],
            "update": ["checkOnStart": false],
            "agents": [
                "defaults": [
                    "model": ["primary": primary.map { "llmhub/" + bridge.apiName($0) } ?? "llmhub/none"],
                    "skipBootstrap": true,
                    "workspace": workspace.path,
                ],
                "entries": entries,
                // One agent per folder is a multi-agent roster, which OpenClaw only accepts with explicit ownership.
                "ownership": "explicit",
            ],
            "models": [
                "mode": "replace",
                "providers": ["llmhub": [
                    "baseUrl": bridge.baseURL + "/v1",
                    "apiKey": bridge.key,
                    "api": "openai-completions",
                    "timeoutSeconds": 600,
                    "models": models,
                ]],
            ],
            // Commands run on this Mac through the Gateway, which asks first (as Claude Code does) for anything unknown.
            "tools": ["exec": ["mode": "ask", "host": "gateway"]],
            "plugins": ["allow": ["memory-core"]],
        ]
        config.merge(ours) { _, new in new }
        if config["meta"] == nil { config["meta"] = ["lastTouchedVersion": Self.version] }
        try FileManager.default.createDirectory(at: Self.configFile.deletingLastPathComponent(), withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: workspace, withIntermediateDirectories: true)
        let json = try JSONSerialization.data(withJSONObject: config, options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])
        if (try? Data(contentsOf: Self.configFile)) != json { try json.write(to: Self.configFile, options: .atomic) }
    }

    /// "f-" and a hash of the folder: the OpenClaw agent whose workspace is that folder.
    static func agentId(_ session: AgentSession) -> String {
        var folder = URL(fileURLWithPath: session.folder).standardizedFileURL.path
        while folder.count > 1 && folder.hasSuffix("/") { folder.removeLast() }
        return "f-" + SHA256.hash(data: Data(folder.utf8)).map { String(format: "%02x", $0) }.joined().prefix(12)
    }

    /// Starts the Gateway once (it takes 15 to 20 seconds) and keeps it running while the app is open.
    func ensureGateway(_ sessions: [AgentSession], bridge: LocalApiServer, models: [InstalledModel]) async throws {
        guard Self.filesPresent else { throw Self.error(L.s("OpenClawNotInstalled")) }
        // One start at a time; a second caller waits for it and then only rewrites the config.
        while let starting { _ = try? await starting.value }
        if let gateway, gateway.isRunning {
            try writeConfig(sessions, bridge: bridge, models: models)
            return
        }
        let start = Task { try await startGateway(sessions, bridge: bridge, models: models) }
        starting = start
        defer { starting = nil }
        try await start.value
    }

    private func startGateway(_ sessions: [AgentSession], bridge: LocalApiServer, models: [InstalledModel]) async throws {
        await Self.stopOrphan()
        port = LocalApiServer.freePort()
        token = SymmetricKey(size: .bits192).withUnsafeBytes { $0.map { String(format: "%02x", $0) }.joined() }
        try token.write(to: Self.tokenFile, atomically: true, encoding: .utf8)
        try writeConfig(sessions, bridge: bridge, models: models)
        status = L.s("OpenClawStarting")
        defer { status = "" }
        let clock = Date()
        for attempt in 0..<2 {
            let process = node(["gateway"], in: Self.root)
            let tail = OutputTail()
            let output = Pipe()
            process.standardOutput = output
            process.standardError = output
            output.fileHandleForReading.readabilityHandler = { handle in
                let data = handle.availableData
                if data.isEmpty { handle.readabilityHandler = nil } else { tail.append(data) }
            }
            try process.run()
            gateway = process
            try? "\(process.processIdentifier)".write(to: Self.pidFile, atomically: true, encoding: .utf8)
            let deadline = Date().addingTimeInterval(240)
            let port = self.port
            while Date() < deadline, process.isRunning, !(await Self.portOpen(port)) {
                try await Task.sleep(for: .milliseconds(300))
            }
            if process.isRunning, await Self.portOpen(port) {
                Log.info("OpenClaw gateway ready on \(port) in \(String(format: "%.0f", Date().timeIntervalSince(clock)))s")
                return
            }
            // 78 is a config error: OpenClaw's doctor repairs it, then one more try.
            if !process.isRunning, process.terminationStatus == 78, attempt == 0 {
                Log.info("OpenClaw gateway config error, running doctor: \(tail.text)")
                _ = try await Self.run(Self.nodeExe, [Self.entry.path, "doctor", "--fix", "--yes", "--non-interactive"], in: Self.root,
                                       environment: Self.childEnvironment())
                continue
            }
            if process.isRunning { AgentService.kill(process) }
            gateway = nil
            throw Self.error("OpenClaw gateway did not start: \(tail.text)")
        }
    }

    /// `openclaw acp` for one session, connected to the warm Gateway (not started yet).
    func acp(folder: String) -> Process {
        node(["acp", "--url", "ws://127.0.0.1:\(port)", "--token-file", Self.tokenFile.path], in: URL(fileURLWithPath: folder, isDirectory: true))
    }

    private func node(_ args: [String], in folder: URL) -> Process {
        let process = Process()
        process.executableURL = Self.nodeExe
        process.arguments = [Self.entry.path] + args
        process.currentDirectoryURL = folder
        process.environment = Self.childEnvironment()
        return process
    }

    /// Keeps OpenClaw local and quiet: its own home, no update or telemetry checks, no channels, no Bonjour.
    private static func childEnvironment() -> [String: String] {
        var env = AgentService.childEnvironment()
        env["OPENCLAW_HOME"] = home.path
        env["OPENCLAW_CONFIG_PATH"] = configFile.path
        env["OPENCLAW_NO_AUTO_UPDATE"] = "1"
        env["OPENCLAW_OFFLINE"] = "1"
        env["OPENCLAW_DISABLE_BONJOUR"] = "1"
        env["OPENCLAW_SKIP_CHANNELS"] = "1"
        env["OPENCLAW_NO_RESPAWN"] = "1"
        env["OPENCLAW_EXEC_SHELL_SNAPSHOT"] = "0"
        env["OPENCLAW_HIDE_BANNER"] = "1"
        env["OPENCLAW_SUPPRESS_NOTES"] = "1"
        env["NO_COLOR"] = "1"
        // The Gateway runs `npm root -g` at startup; npm's debug logs then go to a private cache, not ~/.npm/_logs.
        env["npm_config_cache"] = root.appendingPathComponent("npm-cache", isDirectory: true).path
        env["npm_config_update_notifier"] = "false"
        env["PATH"] = nodeExe.deletingLastPathComponent().path + ":" + (env["PATH"] ?? "/usr/bin:/bin")
        return env
    }

    func stopGateway() {
        if let gateway, gateway.isRunning { AgentService.kill(gateway) }
        gateway = nil
        try? FileManager.default.removeItem(at: Self.pidFile)
    }

    /// A Gateway left running by an app that did not quit cleanly owns the state folder, and a new one refuses to start
    /// (exit 78) until it is gone. It is ours if its executable is our private node.
    private static func stopOrphan() async {
        guard let text = try? String(contentsOf: pidFile, encoding: .utf8), let pid = pid_t(text.trimmingCharacters(in: .whitespacesAndNewlines)),
              pid > 0, kill(pid, 0) == 0 else { return }
        var buffer = [CChar](repeating: 0, count: Int(MAXPATHLEN))
        guard proc_pidpath(pid, &buffer, UInt32(buffer.count)) > 0,
              URL(fileURLWithPath: String(cString: buffer)).resolvingSymlinksInPath().path == nodeExe.resolvingSymlinksInPath().path else { return }
        Log.info("OpenClaw: stopping a Gateway left from an earlier run (pid \(pid))")
        kill(pid, SIGTERM)
        for _ in 0..<50 where kill(pid, 0) == 0 { try? await Task.sleep(for: .milliseconds(100)) }
        if kill(pid, 0) == 0 { kill(pid, SIGKILL) }
        try? FileManager.default.removeItem(at: pidFile)
    }

    // MARK: Helpers

    private static func run(_ exe: URL, _ args: [String], in folder: URL, environment: [String: String]? = nil) async throws -> (Int32, String) {
        let process = Process()
        process.executableURL = exe
        process.arguments = args
        process.currentDirectoryURL = folder
        var env = environment ?? AgentService.childEnvironment()
        env["PATH"] = nodeExe.deletingLastPathComponent().path + ":" + (env["PATH"] ?? "/usr/bin:/bin")
        process.environment = env
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        try process.run()
        let data = await Task.detached { pipe.fileHandleForReading.readDataToEndOfFile() }.value
        while process.isRunning { try? await Task.sleep(for: .milliseconds(50)) }
        return (process.terminationStatus, String(decoding: data, as: UTF8.self))
    }

    private nonisolated static func portOpen(_ port: Int) async -> Bool {
        await Task.detached {
            let fd = socket(AF_INET, SOCK_STREAM, 0)
            guard fd >= 0 else { return false }
            defer { close(fd) }
            var addr = sockaddr_in()
            addr.sin_family = sa_family_t(AF_INET)
            addr.sin_addr.s_addr = inet_addr("127.0.0.1")
            addr.sin_port = in_port_t(UInt16(port).bigEndian)
            return withUnsafePointer(to: &addr) {
                $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { connect(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size)) == 0 }
            }
        }.value
    }

    private static func error(_ message: String) -> NSError {
        NSError(domain: "LLMHub.OpenClaw", code: 1, userInfo: [NSLocalizedDescriptionKey: message])
    }
}

/// The last few KB of a child's output, for error messages.
final class OutputTail: @unchecked Sendable {
    private let lock = NSLock()
    private var data = Data()

    func append(_ chunk: Data) {
        lock.lock()
        defer { lock.unlock() }
        data.append(chunk)
        if data.count > 6000 { data.removeFirst(data.count - 6000) }
    }

    var text: String {
        lock.lock()
        defer { lock.unlock() }
        return String(decoding: data.suffix(1200), as: UTF8.self)
    }
}
