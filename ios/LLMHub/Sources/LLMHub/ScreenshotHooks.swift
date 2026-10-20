import SwiftUI

extension View {
    /// A non-localized accessibility value for UI automation, in Debug builds only, so
    /// VoiceOver output in Release is unchanged.
    @ViewBuilder
    func automationValue(_ value: @autoclosure () -> String) -> some View {
        #if DEBUG
        accessibilityValue(value())
        #else
        self
        #endif
    }
}

extension View {
    /// Turns off autocorrection (and with it the red spell-check underline) on text
    /// inputs while screenshot automation drives the app. No effect otherwise.
    @ViewBuilder
    func screenshotAutocorrectionDisabled() -> some View {
        #if DEBUG
        autocorrectionDisabled(ScreenshotHooks.isActive)
        #else
        self
        #endif
    }
}

extension DownloadState {
    /// Stable state name for UI automation (see `automationValue`).
    var automationValue: String {
        switch self {
        case .notDownloaded: return "notDownloaded"
        case .downloading: return "downloading"
        case .paused: return "paused"
        case .downloaded: return "downloaded"
        case .error: return "error"
        }
    }
}

#if DEBUG
import Darwin
import Foundation
import UIKit

/// Debug-only hooks that let the `LLMHubScreenshotUITests` target capture App Store
/// screenshots. Everything in this file is compiled out of Release builds, and every
/// hook is inert unless the test passes its launch argument.
///
/// Launch arguments (read from the `UserDefaults` argument domain):
/// - `-screenshot_vibevoice_transcript "<text>"`: VibeVoice submits this text as the
///   user's turn instead of listening, and does not listen again afterwards.
/// - `-screenshot_disable_auto_variation YES`: the image generator does not start the
///   next variation by itself.
/// - `-screenshot_inject_image|video|video_input|music|upscale_input|upscale_output <file>`:
///   the screen shows that file (relative to Documents) as its result instead of running
///   a model, through the same result UI.
/// - `-screenshot_vibecode_workspace <folder>`: Vibecode uses Documents/<folder> as its
///   workspace instead of asking for one through the Files picker.
/// - `-screenshot_vibecode_file <name>` / `-screenshot_vibecode_prompt "<text>"`: Vibecode
///   opens that new file with an empty chat and prefills the chat input.
/// - `-screenshot_translator_input "<text>"`: the Translator opens with this text.
/// - `-screenshot_image_prompt` / `-screenshot_video_prompt` / `-screenshot_music_prompt "<text>"`: prefilled prompts.
/// - `-screenshot_autotest music`: unattended music generation test (see `autotest`).
/// - `-screenshot_save_outputs YES`: generated results are copied into
///   Documents/screenshot_outputs/.
/// - `-screenshot_output_port <port>`: serves Documents/screenshot_outputs/ on
///   127.0.0.1:<port> so the UI test runner can attach the files.
///
/// Darwin notifications `com.llmhub.screenshots.lang.<code>` switch the app language live.
enum ScreenshotHooks {
    static let languageNotificationPrefix = "com.llmhub.screenshots.lang."

    private static func string(_ key: String) -> String? {
        guard let value = UserDefaults.standard.string(forKey: key)?
            .trimmingCharacters(in: .whitespacesAndNewlines), !value.isEmpty else { return nil }
        return value
    }

    // MARK: - Flags

    /// True when the app was launched by the screenshot tests (any `-screenshot_` argument).
    static let isActive: Bool = ProcessInfo.processInfo.arguments.contains { $0.hasPrefix("-screenshot_") }

    static let vibeVoiceTranscript: String? = string("screenshot_vibevoice_transcript")
    static let disableAutoVariation: Bool = UserDefaults.standard.bool(forKey: "screenshot_disable_auto_variation")
    static let saveOutputs: Bool = UserDefaults.standard.bool(forKey: "screenshot_save_outputs")

    static var documentsDirectory: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    }

    static let vibeCodeFileName: String? = string("screenshot_vibecode_file")
    static let vibeCodePrompt: String? = string("screenshot_vibecode_prompt")
    /// Translator input text, prefilled instead of typed (typing is unreliable on some iPads).
    static let translatorInput: String? = string("screenshot_translator_input")
    /// Image and video prompts, prefilled for the same reason.
    static let imagePrompt: String? = string("screenshot_image_prompt")
    static let videoPrompt: String? = string("screenshot_video_prompt")
    static let musicPrompt: String? = string("screenshot_music_prompt")
    /// `-screenshot_autotest music`: download the music model if needed, open Music Generator
    /// and generate once, logging each step to the unified log ("[LLMHubAutotest]"). For Mac
    /// "Designed for iPad" runs, where UI tests cannot drive the app.
    static let autotest: String? = string("screenshot_autotest")

    static func autotestLog(_ message: String) {
        NSLog("[LLMHubAutotest] %@", message)
    }

    /// Downloads `modelName` through the regular download path unless it is already installed.
    @MainActor static func ensureModelDownloaded(named modelName: String) async -> Bool {
        let vm = ModelDownloadViewModel.shared
        vm.refreshStatuses()
        guard let model = vm.models.first(where: { $0.name == modelName }) else {
            autotestLog("model not in catalog: \(modelName)"); return false
        }
        if vm.downloadStates[model.id] == .downloaded {
            autotestLog("model already downloaded: \(modelName)"); return true
        }
        autotestLog("downloading \(modelName)")
        vm.startDownload(model)
        var lastLog = Date.distantPast
        while true {
            try? await Task.sleep(nanoseconds: 2_000_000_000)
            switch vm.downloadStates[model.id] {
            case .downloaded:
                autotestLog("download finished: \(modelName)"); return true
            case .error(let message):
                autotestLog("download failed: \(message)"); return false
            case .downloading(let progress, let downloaded, _):
                if Date().timeIntervalSince(lastLog) > 20 {
                    autotestLog("download \(Int(progress * 100))% (\(downloaded))"); lastLog = Date()
                }
            default:
                break
            }
        }
    }
    /// Set once Vibecode has opened the scripted file in this launch.
    @MainActor static var vibeCodePrepared = false

    static let vibeCodeWorkspaceURL: URL? = {
        guard let name = string("screenshot_vibecode_workspace") else { return nil }
        let url = documentsDirectory.appendingPathComponent(name, isDirectory: true)
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }()

    // MARK: - Result injection

    private static func injectedFile(_ key: String) -> URL? {
        guard let name = string(key) else { return nil }
        let url = name.hasPrefix("/") ? URL(fileURLWithPath: name) : documentsDirectory.appendingPathComponent(name)
        guard FileManager.default.fileExists(atPath: url.path) else {
            NSLog("[Screenshots] \(key): no file at \(url.path); injection disabled")
            return nil
        }
        return url
    }

    static let injectImageURL: URL? = injectedFile("screenshot_inject_image")
    static let injectVideoURL: URL? = injectedFile("screenshot_inject_video")
    static let injectVideoInputURL: URL? = injectedFile("screenshot_inject_video_input")
    static let injectMusicURL: URL? = injectedFile("screenshot_inject_music")
    static let injectUpscaleInputURL: URL? = injectedFile("screenshot_inject_upscale_input")
    static let injectUpscaleOutputURL: URL? = injectedFile("screenshot_inject_upscale_output")

    /// How long an injected "generation" shows its progress UI before the result appears.
    static let injectedResultDelay: UInt64 = 1_500_000_000

    /// `SCREENSHOT_INPUT_IMAGE_B64` (launch environment): a base64 image that Video and Upscale
    /// start from, for devices where no file can be copied into the app first.
    static let environmentInputImage: UIImage? = ProcessInfo.processInfo.environment["SCREENSHOT_INPUT_IMAGE_B64"]
        .flatMap { Data(base64Encoded: $0, options: .ignoreUnknownCharacters) }
        .flatMap(UIImage.init(data:))

    static func image(at url: URL?) -> UIImage? {
        guard let url, let data = try? Data(contentsOf: url) else { return nil }
        return UIImage(data: data)
    }

    // MARK: - Saving results

    static var outputsDirectory: URL {
        documentsDirectory.appendingPathComponent("screenshot_outputs", isDirectory: true)
    }

    static func saveOutput(_ image: UIImage?, as name: String) {
        guard saveOutputs, let image, let data = image.pngData() else { return }
        write(to: name) { try data.write(to: $0, options: .atomic) }
    }

    static func saveOutput(copyOf source: URL?, as name: String) {
        guard saveOutputs, let source else { return }
        write(to: name) { try FileManager.default.copyItem(at: source, to: $0) }
    }

    private static func write(to name: String, _ body: (URL) throws -> Void) {
        let destination = outputsDirectory.appendingPathComponent(name)
        do {
            try FileManager.default.createDirectory(at: outputsDirectory, withIntermediateDirectories: true)
            try? FileManager.default.removeItem(at: destination)
            try body(destination)
            NSLog("[Screenshots] saved output \(destination.path)")
        } catch {
            NSLog("[Screenshots] failed to save \(name): \(error.localizedDescription)")
        }
    }

    // MARK: - Live language switching

    private final class ObserverToken: Sendable {}
    private static let observerToken = ObserverToken()

    /// Registers one Darwin notification per app language. Posting
    /// `com.llmhub.screenshots.lang.<code>` from any process (the UI test runner)
    /// sets `AppSettings.selectedLanguage`, which re-renders the UI in place.
    static func installLanguageObservers() {
        let center = CFNotificationCenterGetDarwinNotifyCenter()
        let observer = Unmanaged.passUnretained(observerToken).toOpaque()
        for language in AppLanguage.allCases where language != .systemDefault {
            let name = (languageNotificationPrefix + language.rawValue) as CFString
            CFNotificationCenterAddObserver(center, observer, { _, _, name, _, _ in
                guard let raw = name?.rawValue as String?,
                      raw.hasPrefix(ScreenshotHooks.languageNotificationPrefix) else { return }
                let code = String(raw.dropFirst(ScreenshotHooks.languageNotificationPrefix.count))
                DispatchQueue.main.async {
                    MainActor.assumeIsolated {
                        guard let language = AppLanguage(rawValue: code) else { return }
                        NSLog("[Screenshots] switching language to \(code)")
                        AppSettings.shared.selectedLanguage = language
                    }
                }
            }, name, nil, .deliverImmediately)
        }
    }

    // MARK: - Output server

    /// Loopback-only HTTP server (BSD sockets, like LocalHTMLPreviewServer, so there is
    /// no local network prompt). `GET /` lists the files in Documents/screenshot_outputs,
    /// `GET /<name>` returns one of them.
    static func startOutputServerIfRequested() {
        let port = UserDefaults.standard.integer(forKey: "screenshot_output_port")
        guard port > 0, port < 65536 else { return }
        Thread.detachNewThread { runOutputServer(port: UInt16(port)) }
    }

    private static func runOutputServer(port: UInt16) {
        let fd = socket(AF_INET, SOCK_STREAM, 0)
        guard fd >= 0 else { return }
        var yes: Int32 = 1
        setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, socklen_t(MemoryLayout<Int32>.size))
        var addr = sockaddr_in()
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_port = port.bigEndian
        addr.sin_addr.s_addr = UInt32(INADDR_LOOPBACK).bigEndian
        let bound = withUnsafePointer(to: &addr) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                Darwin.bind(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        guard bound == 0, Darwin.listen(fd, 8) == 0 else {
            NSLog("[Screenshots] output server could not listen on \(port): errno \(errno)")
            Darwin.close(fd)
            return
        }
        NSLog("[Screenshots] output server listening on 127.0.0.1:\(port)")
        while true {
            let client = Darwin.accept(fd, nil, nil)
            if client < 0 { continue }
            serve(client: client)
            Darwin.close(client)
        }
    }

    private static func serve(client: Int32) {
        var buffer = [UInt8](repeating: 0, count: 4096)
        let count = Darwin.recv(client, &buffer, buffer.count, 0)
        guard count > 0 else { return }
        let request = String(decoding: buffer[0..<count], as: UTF8.self)
        let path = request.split(separator: "\r\n").first?.split(separator: " ").dropFirst().first.map(String.init) ?? "/"
        let name = (path.removingPercentEncoding ?? path).trimmingCharacters(in: CharacterSet(charactersIn: "/"))

        let status: String
        let body: Data
        if name.isEmpty {
            let files = (try? FileManager.default.contentsOfDirectory(atPath: outputsDirectory.path)) ?? []
            status = "200 OK"
            body = Data(files.sorted().joined(separator: "\n").utf8)
        } else if !name.contains("/"), !name.hasPrefix("."),
                  let data = try? Data(contentsOf: outputsDirectory.appendingPathComponent(name)) {
            status = "200 OK"
            body = data
        } else {
            status = "404 Not Found"
            body = Data()
        }
        var response = Data("HTTP/1.1 \(status)\r\nContent-Type: application/octet-stream\r\nContent-Length: \(body.count)\r\nConnection: close\r\n\r\n".utf8)
        response.append(body)
        response.withUnsafeBytes { raw in
            guard let base = raw.baseAddress else { return }
            var sent = 0
            while sent < raw.count {
                let result = Darwin.send(client, base + sent, raw.count - sent, 0)
                if result <= 0 { break }
                sent += result
            }
        }
    }
}
#endif
