import Foundation
import os

enum AppPaths {
    static let root: URL = {
        var url = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("LLMHub", isDirectory: true)
        #if DEBUG
        // Lets a second debug copy run with its own data while tests use the real folder.
        if let home = ProcessInfo.processInfo.environment["LLMHUB_HOME"] { url = URL(fileURLWithPath: home, isDirectory: true) }
        #endif
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }()

    static var models: URL { directory("models") }
    static var attachments: URL { directory("attachments") }
    static var design: URL { directory("design") }
    static var agent: URL { directory("agent") }
    static var engines: URL { directory("engines") }
    static var logs: URL { directory("logs") }
    static var database: URL { root.appendingPathComponent("llmhub.sqlite") }
    static var settings: URL { root.appendingPathComponent("settings.json") }
    static var bundledCatalog: URL? { Bundle.main.url(forResource: "catalog", withExtension: "json") }

    private static func directory(_ name: String) -> URL {
        let url = root.appendingPathComponent(name, isDirectory: true)
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }
}

enum Log {
    private static let logger = Logger(subsystem: "app.llm-hub.desktop", category: "app")
    private static let queue = DispatchQueue(label: "llmhub.log")
    private static let file: URL = {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyyMMdd"
        return AppPaths.logs.appendingPathComponent("llmhub-\(formatter.string(from: Date())).log")
    }()

    static func info(_ message: String) { write("INFO", message) }
    static func error(_ message: String, _ error: Error? = nil) {
        write("ERROR", error.map { "\(message): \($0)" } ?? message)
    }

    private static func write(_ level: String, _ message: String) {
        logger.log("\(level, privacy: .public) \(message, privacy: .public)")
        queue.async {
            let stamp = ISO8601DateFormatter.string(from: Date(), timeZone: .current, formatOptions: [.withTime, .withColonSeparatorInTime])
            let line = "\(stamp) \(level) \(message)\n"
            if let handle = try? FileHandle(forWritingTo: file) {
                handle.seekToEndOfFile()
                handle.write(Data(line.utf8))
                try? handle.close()
            } else {
                try? Data(line.utf8).write(to: file)
            }
        }
    }
}

/// Localized strings from Localizable.xcstrings. Keys match the Windows app's Resources.resw.
enum L {
    static let languages: [(tag: String, name: String)] = [
        ("en", "English"), ("es", "Español"), ("pt-BR", "Português"), ("fr", "Français"),
        ("de", "Deutsch"), ("it", "Italiano"), ("ru", "Русский"),
    ]

    static func s(_ key: String) -> String { NSLocalizedString(key, comment: "") }
    static func f(_ key: String, _ args: String...) -> String { String(format: s(key), arguments: args) }

    /// The app's language override, or "" to follow macOS. Takes effect on next launch.
    static var override: String {
        get { UserDefaults.standard.string(forKey: "LLMHubLanguage") ?? "" }
        set {
            if newValue.isEmpty {
                UserDefaults.standard.removeObject(forKey: "AppleLanguages")
                UserDefaults.standard.removeObject(forKey: "LLMHubLanguage")
            } else {
                UserDefaults.standard.set([newValue], forKey: "AppleLanguages")
                UserDefaults.standard.set(newValue, forKey: "LLMHubLanguage")
            }
        }
    }
}

struct AppSettings: Codable {
    var huggingFaceToken = ""
    var lastChatModelId: String?
    var contextLength = 16384
    /// Models where the user turned multi-token prediction off (it is on by default where supported).
    var mtpOff: [String] = []
    var apiEnabled = false
    var apiPort = 8642
    var apiAllowLan = false
    var apiRequireKey = true
    var apiKey = ""
    /// dark (the designed look), light or system.
    var appearance = "dark"
    /// New chats start with web search on.
    var webSearch = false
    var thinking = false
    /// Supertonic voice for read aloud, Studio › Voice and the Local API: F1 to F5, M1 to M5 or an imported custom_<uuid>.
    var voice = "F1"
    /// Supertonic ONNX build: fp16 (default) or fp32.
    var ttsBuild = "fp16"
    /// Supertonic language code, or "" to detect it from the text.
    var ttsLanguage = ""
    /// Supertonic speaking speed, 0.5 to 2.
    var ttsSpeed = 1.05
    /// Local API: free the models' memory after this many idle minutes; 0 keeps them loaded.
    var apiUnloadAfterMinutes = 10
    var agentModelId: String?
    var designModelId: String?
    /// Context for the Agent and the Local API (Claude Code's own prompt is about 18K tokens).
    var apiContextLength = 65536

    init() {}

    /// Missing keys keep their defaults, so adding settings never resets a user's file.
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        huggingFaceToken = try c.decodeIfPresent(String.self, forKey: .huggingFaceToken) ?? ""
        lastChatModelId = try c.decodeIfPresent(String.self, forKey: .lastChatModelId)
        contextLength = try c.decodeIfPresent(Int.self, forKey: .contextLength) ?? 16384
        mtpOff = try c.decodeIfPresent([String].self, forKey: .mtpOff) ?? []
        apiEnabled = try c.decodeIfPresent(Bool.self, forKey: .apiEnabled) ?? false
        apiPort = try c.decodeIfPresent(Int.self, forKey: .apiPort) ?? 8642
        apiAllowLan = try c.decodeIfPresent(Bool.self, forKey: .apiAllowLan) ?? false
        apiRequireKey = try c.decodeIfPresent(Bool.self, forKey: .apiRequireKey) ?? true
        apiKey = try c.decodeIfPresent(String.self, forKey: .apiKey) ?? ""
        appearance = try c.decodeIfPresent(String.self, forKey: .appearance) ?? "dark"
        webSearch = try c.decodeIfPresent(Bool.self, forKey: .webSearch) ?? false
        thinking = try c.decodeIfPresent(Bool.self, forKey: .thinking) ?? false
        voice = try c.decodeIfPresent(String.self, forKey: .voice) ?? "F1"
        ttsBuild = try c.decodeIfPresent(String.self, forKey: .ttsBuild) ?? "fp16"
        ttsLanguage = try c.decodeIfPresent(String.self, forKey: .ttsLanguage) ?? ""
        ttsSpeed = try c.decodeIfPresent(Double.self, forKey: .ttsSpeed) ?? 1.05
        apiUnloadAfterMinutes = try c.decodeIfPresent(Int.self, forKey: .apiUnloadAfterMinutes) ?? 10
        agentModelId = try c.decodeIfPresent(String.self, forKey: .agentModelId)
        designModelId = try c.decodeIfPresent(String.self, forKey: .designModelId)
        apiContextLength = try c.decodeIfPresent(Int.self, forKey: .apiContextLength) ?? 65536
    }

    static var current: AppSettings = load()

    private static func load() -> AppSettings {
        guard let data = try? Data(contentsOf: AppPaths.settings),
              let value = try? JSONDecoder().decode(AppSettings.self, from: data) else { return AppSettings() }
        return value
    }

    static func save() {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try? encoder.encode(current).write(to: AppPaths.settings, options: .atomic)
    }
}

enum Fmt {
    static func bytes(_ value: Int64) -> String {
        switch value {
        case (1 << 30)...: return String(format: "%.1f GB", Double(value) / 1_073_741_824)
        case (1 << 20)...: return String(format: "%.0f MB", Double(value) / 1_048_576)
        default: return String(format: "%.0f KB", Double(value) / 1024)
        }
    }
}
