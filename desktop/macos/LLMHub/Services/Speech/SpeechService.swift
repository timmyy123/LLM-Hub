import Foundation
import Observation

/// Read aloud (Supertonic 3, ONNX Runtime on the CPU) and dictation (whisper.cpp large-v3 turbo, Metal), on this Mac.
/// Both download on first use into the engines folder; nothing is sent anywhere else. Studio › Voice uses `synthesize` too.
@MainActor @Observable
final class SpeechService {
    static let voices = ["F1", "F2", "M1", "M2"]
    /// Setup progress or the last error, for the composer and Settings › Speech and voice.
    var status = ""
    /// Id of the message being read aloud, or nil.
    var speakingId: String?
    var isRecording = false
    var isTranscribing = false

    // Same files and pinned revision as the Windows and Linux apps.
    private static let supertonicRepo = "https://huggingface.co/supertone-oss-archive/supertonic-3/resolve/aafc6e32416a594460b32413efc49d7fe4ce6d46/"
    private static let supertonicFiles = [
        "onnx/duration_predictor.onnx", "onnx/text_encoder.onnx", "onnx/vector_estimator.onnx", "onnx/vocoder.onnx",
        "onnx/tts.json", "onnx/unicode_indexer.json",
        "voice_styles/F1.json", "voice_styles/F2.json", "voice_styles/M1.json", "voice_styles/M2.json",
    ]
    private static let whisperModel = "ggml-large-v3-turbo-q5_0.bin"
    private static let whisperModelUrl = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/" + whisperModel

    static var ttsRoot: URL { AppPaths.engines.appendingPathComponent("supertonic-3", isDirectory: true) }
    static var whisperRoot: URL { AppPaths.engines.appendingPathComponent("whisper.cpp", isDirectory: true) }
    private static var whisperModelPath: URL { whisperRoot.appendingPathComponent("models").appendingPathComponent(whisperModel) }
    static var isTtsInstalled: Bool { supertonicFiles.allSatisfy { FileManager.default.fileExists(atPath: ttsRoot.appendingPathComponent($0).path) } }
    static var isAsrInstalled: Bool { FileManager.default.fileExists(atPath: whisperModelPath.path) }
    static let ttsName = "Supertonic 3"
    static let asrName = "Whisper large-v3 turbo · q5_0"

    private var tts: Supertonic.TextToSpeech?
    private var ttsLoading: Task<Supertonic.TextToSpeech, Error>?
    private let whisper = WhisperTranscriber()
    private var whisperLoading: Task<Void, Error>?
    private var whisperReady = false
    private let audio = AudioIO()
    private var speaking: Task<Void, Never>?

    /// Downloads both engines now (Settings › Speech and voice) instead of on first use.
    func setUp() async throws {
        _ = try await ensureTts()
        try await ensureWhisper()
        status = ""
    }

    /// Stops the engines and deletes their downloads.
    func remove() async {
        stopSpeaking()
        tts = nil
        ttsLoading = nil
        whisperReady = false
        whisperLoading = nil
        await whisper.unload()
        for dir in [Self.ttsRoot, Self.whisperRoot] { try? FileManager.default.removeItem(at: dir) }
    }

    // MARK: Read aloud

    /// Reads a reply aloud; a second call with the same id stops it.
    func toggleSpeak(id: String, text: String) {
        if speakingId == id {
            stopSpeaking()
            return
        }
        stopSpeaking()
        speakingId = id
        speaking = Task {
            do {
                let wav = try await synthesize(Markdown.plainText(text), voice: AppSettings.current.voice)
                try Task.checkCancellation()
                await play(wav.url)
                try? FileManager.default.removeItem(at: wav.url)
            } catch is CancellationError {
            } catch {
                Log.error("Read aloud failed", error)
                status = L.f("ErrorPrefix", error.localizedDescription)
            }
            if speakingId == id { speakingId = nil }
        }
    }

    /// Plays a WAV file; returns when it finishes or is stopped.
    func play(_ url: URL) async {
        await audio.play(url)
    }

    func stopSpeaking() {
        speaking?.cancel()
        speaking = nil
        audio.stopPlayback()
        speakingId = nil
    }

    /// Speaks `text` in `voice` to a WAV file (44.1 kHz) and returns it with its length in seconds.
    /// The language is guessed from the text and the app language unless given.
    func synthesize(_ text: String, voice: String, language: String? = nil) async throws -> (url: URL, seconds: Double) {
        let tts = try await ensureTts()
        let voiceFile = Self.ttsRoot.appendingPathComponent("voice_styles/\(Self.voices.contains(voice) ? voice : "F1").json")
        let lang = language.map { String($0.prefix(2)).lowercased() }.flatMap { Supertonic.languages.contains($0) ? $0 : nil } ?? Self.guessLanguage(text)
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("llmhub-tts-\(UUID().uuidString).wav")
        let seconds = try await Task.detached(priority: .userInitiated) {
            let style = try Supertonic.TextToSpeech.loadStyle(voiceFile)
            let (wav, duration) = try tts.call(text, lang: lang, style: style, totalStep: 8, speed: 1.05)
            let length = min(wav.count, Int(Float(tts.sampleRate) * duration))
            try AudioIO.writeWav(url, samples: wav[0..<length], sampleRate: tts.sampleRate)
            return Double(duration)
        }.value
        return (url, seconds)
    }

    private func ensureTts() async throws -> Supertonic.TextToSpeech {
        if let tts { return tts }
        if let ttsLoading { return try await ttsLoading.value }
        let task = Task<Supertonic.TextToSpeech, Error> {
            let files = Self.supertonicFiles
            for (n, file) in files.enumerated() {
                let target = Self.ttsRoot.appendingPathComponent(file)
                guard !FileManager.default.fileExists(atPath: target.path) else { continue }
                try await Self.download(Self.supertonicRepo + file, to: target) { p in
                    self.status = L.f("DownloadingVoice", String(format: "%.0f", (Double(n) * 100 + p) / Double(files.count)))
                }
            }
            status = ""
            let onnx = Self.ttsRoot.appendingPathComponent("onnx", isDirectory: true)
            let loaded = try await Task.detached(priority: .userInitiated) { try Supertonic.TextToSpeech(onnxDir: onnx) }.value
            Log.info("Supertonic 3 loaded")
            return loaded
        }
        ttsLoading = task
        do {
            let loaded = try await task.value
            tts = loaded
            ttsLoading = nil
            return loaded
        } catch {
            ttsLoading = nil
            status = L.f("ErrorPrefix", error.localizedDescription)
            throw error
        }
    }

    /// App language first, then a cheap script check, so Russian text is read as Russian in an English UI.
    private static func guessLanguage(_ text: String) -> String {
        if text.unicodeScalars.contains(where: { (0x0400...0x04FF).contains($0.value) }) { return "ru" }
        let tag = L.override.isEmpty ? (Locale.preferredLanguages.first ?? "en") : L.override
        let ui = String(tag.prefix(2))
        return ["es", "pt", "fr", "de", "it", "ru"].contains(ui) ? ui : "en"
    }

    // MARK: Dictation

    func startDictation() async throws {
        guard !isRecording else { return }
        stopSpeaking()
        guard await AudioIO.requestMicrophone() else {
            throw NSError(domain: "LLMHub", code: 45, userInfo: [NSLocalizedDescriptionKey: "Microphone access is off for LLM Hub in System Settings › Privacy & Security"])
        }
        try audio.startRecording()
        isRecording = true
        // Warm Whisper up while the user talks.
        Task {
            do { try await ensureWhisper() } catch { Log.error("Whisper setup failed", error) }
        }
    }

    /// Stops recording and returns the transcript ("" on failure, with `status` set).
    func stopDictation() async -> String {
        guard isRecording else { return "" }
        let samples = audio.stopRecording()
        isRecording = false
        isTranscribing = true
        defer { isTranscribing = false }
        do {
            return try await transcribe(samples: samples, language: nil)
        } catch {
            Log.error("Dictation failed", error)
            status = L.f("ErrorPrefix", error.localizedDescription)
            return ""
        }
    }

    /// Speech in an audio file (any rate) to text.
    func transcribe(_ url: URL, language: String? = nil) async throws -> String {
        let samples = try await Task.detached { try AudioIO.readSamples(url) }.value
        let text = try await transcribe(samples: samples, language: language)
        Log.info("Transcribed \(url.lastPathComponent): \(text.count) chars")
        return text
    }

    func transcribe(samples: [Float], language: String?) async throws -> String {
        guard samples.count > Int(AudioIO.whisperRate / 4) else { return "" }
        try await ensureWhisper()
        return try await whisper.transcribe(samples, language: language)
    }

    private func ensureWhisper() async throws {
        if whisperReady { return }
        if let whisperLoading { return try await whisperLoading.value }
        let task = Task<Void, Error> {
            let model = Self.whisperModelPath
            if !FileManager.default.fileExists(atPath: model.path) {
                try await Self.download(Self.whisperModelUrl, to: model) { p in
                    self.status = L.f("DownloadingWhisper", String(format: "%.0f", p))
                }
            }
            status = ""
            let start = Date()
            try await whisper.load(model)
            Log.info("Whisper loaded in \(String(format: "%.1f", Date().timeIntervalSince(start)))s")
        }
        whisperLoading = task
        do {
            try await task.value
            whisperReady = true
            whisperLoading = nil
        } catch {
            whisperLoading = nil
            status = L.f("ErrorPrefix", error.localizedDescription)
            throw error
        }
    }

    // MARK: Downloads

    /// Downloads one file (resumable, via `FileDownloader`), reporting percent on the main actor.
    private static func download(_ url: String, to target: URL, progress: @escaping @MainActor (Double) -> Void) async throws {
        try FileManager.default.createDirectory(at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
        var head = URLRequest(url: URL(string: url)!)
        head.httpMethod = "HEAD"
        let size = (try? await URLSession.shared.data(for: head).1.expectedContentLength).map { max(0, $0) } ?? 0
        Log.info("Downloading \(url) (\(Fmt.bytes(size)))")
        try await FileDownloader.download(request: URLRequest(url: URL(string: url)!), to: target, expectedSize: size, sha256: nil) { bytes in
            let percent = size > 0 ? min(100, Double(bytes) * 100 / Double(size)) : 0
            Task { @MainActor in progress(percent) }
        }
    }
}
