import Foundation
import SwiftUI
import AVFoundation
import NaturalLanguage

private final class LocalizationBundleProbe {}

// MARK: - Supported Languages matching Android locale XMLs
enum AppLanguage: String, CaseIterable, Identifiable, Sendable {
    case systemDefault = "system"
    case english = "en"
    case arabic = "ar"
    case german = "de"
    case spanish = "es"
    case persian = "fa"
    case french = "fr"
    case hebrew = "he"
    case indonesian = "id"
    case italian = "it"
    case japanese = "ja"
    case korean = "ko"
    case polish = "pl"
    case portuguese = "pt"
    case russian = "ru"
    case turkish = "tr"
    case ukrainian = "uk"
    case chinese = "zh"

    var id: String { rawValue }

    var displayNameKey: String {
        switch self {
        case .systemDefault: return "system_default_language"
        case .english: return "language_english"
        case .arabic: return "language_arabic"
        case .german: return "language_german"
        case .spanish: return "language_spanish"
        case .persian: return "language_persian"
        case .french: return "language_french"
        case .hebrew: return "language_hebrew"
        case .indonesian: return "language_indonesian"
        case .italian: return "language_italian"
        case .japanese: return "language_japanese"
        case .korean: return "language_korean"
        case .polish: return "language_polish"
        case .portuguese: return "language_portuguese"
        case .russian: return "language_russian"
        case .turkish: return "language_turkish"
        case .ukrainian: return "language_ukrainian"
        case .chinese: return "language_chinese"
        }
    }

    var isRTL: Bool {
        switch self {
        case .arabic, .persian, .hebrew: return true
        default: return false
        }
    }

    var locale: Locale {
        if self == .systemDefault {
            return Locale.current
        }
        return Locale(identifier: self.rawValue)
    }
}

enum AppTheme: String, CaseIterable, Identifiable, Sendable {
    case light = "light"
    case dark = "dark"
    case system = "system"

    var id: String { rawValue }

    var displayNameKey: String {
        switch self {
        case .light: return "theme_light"
        case .dark: return "theme_dark"
        case .system: return "theme_system"
        }
    }

    var colorScheme: ColorScheme? {
        switch self {
        case .light: return .light
        case .dark: return .dark
        case .system: return nil
        }
    }
}

// MARK: - Settings Store
@MainActor
final class AppSettings: ObservableObject {
    static let shared = AppSettings()

    @Published var selectedLanguage: AppLanguage {
        didSet { UserDefaults.standard.set(selectedLanguage.rawValue, forKey: "app_language") }
    }
    @Published var theme: AppTheme {
        didSet { UserDefaults.standard.set(theme.rawValue, forKey: "app_theme") }
    }
    @Published var streamingEnabled: Bool {
        didSet { UserDefaults.standard.set(streamingEnabled, forKey: "streaming_enabled") }
    }
    @Published var showResultStatus: Bool {
        didSet { UserDefaults.standard.set(showResultStatus, forKey: "show_result_status") }
    }
    @Published var autoReadoutEnabled: Bool {
        didSet { UserDefaults.standard.set(autoReadoutEnabled, forKey: "auto_readout_enabled") }
    }
    @Published var selectedEmbeddingModelId: String? {
        didSet { UserDefaults.standard.set(selectedEmbeddingModelId, forKey: "selected_embedding_model_id") }
    }
    /// RAG is enabled when an embedding model is selected (no separate toggle, mirrors Android).
    var ragEnabled: Bool { selectedEmbeddingModelId != nil }
    @Published var memoryEnabled: Bool {
        didSet { UserDefaults.standard.set(memoryEnabled, forKey: "memory_enabled") }
    }

    private init() {
        let langRaw = UserDefaults.standard.string(forKey: "app_language") ?? "system"
        selectedLanguage = AppLanguage(rawValue: langRaw) ?? .systemDefault
        let themeRaw = UserDefaults.standard.string(forKey: "app_theme") ?? "system"
        theme = AppTheme(rawValue: themeRaw) ?? .system
        streamingEnabled = UserDefaults.standard.bool(forKey: "streaming_enabled")
        showResultStatus = UserDefaults.standard.object(forKey: "show_result_status") as? Bool ?? true
        autoReadoutEnabled = UserDefaults.standard.bool(forKey: "auto_readout_enabled")
        selectedEmbeddingModelId = UserDefaults.standard.string(forKey: "selected_embedding_model_id")
        memoryEnabled = UserDefaults.standard.bool(forKey: "memory_enabled")
    }

    private var activeLocalizationCode: String {
        if selectedLanguage == .systemDefault {
            let preferred = Locale.preferredLanguages.first ?? "en"
            return preferred.components(separatedBy: "-").first ?? "en"
        }
        return selectedLanguage.rawValue
    }

    func localized(_ key: String) -> String {
        let code = activeLocalizationCode
        
        // 1. Gather all bundles to check
        var bundlesToCheck: [Bundle] = [Bundle.main, Bundle(for: LocalizationBundleProbe.self)]
        bundlesToCheck.append(contentsOf: Bundle.allBundles)
        
        // 2. Try target language in all bundles
        for bundle in bundlesToCheck {
            if let path = bundle.path(forResource: code, ofType: "lproj"),
               let langBundle = Bundle(path: path) {
                let val = langBundle.localizedString(forKey: key, value: nil, table: "Localizable")
                if val != key { 
                    return val 
                }
            }
        }
        
        // 3. Fallback to English in all bundles
        for bundle in bundlesToCheck {
            if let path = bundle.path(forResource: "en", ofType: "lproj"),
               let enBundle = Bundle(path: path) {
                let val = enBundle.localizedString(forKey: key, value: nil, table: "Localizable")
                if val != key { 
                    return val 
                }
            }
        }
        
        // 4. Ultimate fallback
        return NSLocalizedString(key, comment: "")
    }
}

@MainActor
final class OnDeviceTtsManager: NSObject, ObservableObject, AVSpeechSynthesizerDelegate {
    static let shared = OnDeviceTtsManager()

    @Published private(set) var isSpeaking = false
    @Published private(set) var currentKey: String?

    private let synthesizer = AVSpeechSynthesizer()
    /// Streaming token buffer — mirrors Android TtsService.textBuffer.
    private var streamingBuffer = ""
    private static let sentenceDelimiters: Set<Character> = [".", "!", "?", "。", "！", "？"]

    private override init() {
        super.init()
        synthesizer.delegate = self
    }

    // MARK: - Non-streaming speak (manual tap)

    func speak(_ text: String, fallbackLanguage: AppLanguage, key: String? = nil) {
        let cleaned = sanitize(text)
        guard !cleaned.isEmpty else { return }

        stop()

        currentKey = key
        isSpeaking = true

        let fallback = fallbackLanguage
        Task { @MainActor [weak self, cleaned, fallback] in
            guard let self = self else { return }
            try? AVAudioSession.sharedInstance().setCategory(.playback, mode: .spokenAudio, options: .duckOthers)
            try? AVAudioSession.sharedInstance().setActive(true)
            let utterance = AVSpeechUtterance(string: cleaned)
            utterance.voice = self.bestVoice(for: cleaned, fallbackLanguage: fallback)
            utterance.rate = AVSpeechUtteranceDefaultSpeechRate
            utterance.prefersAssistiveTechnologySettings = true
            self.synthesizer.speak(utterance)
        }
    }

    // MARK: - Streaming TTS (mirrors Android addStreamingText / flushStreamingBuffer)

    /// Feed one token of new text into the buffer.
    /// Speaks any complete sentence(s) immediately via QUEUE_ADD so nothing is ever skipped.
    func addStreamingToken(_ token: String, fallbackLanguage: AppLanguage, key: String?) {
        guard !token.isEmpty else { return }

        // Activate audio session on first token
        if !isSpeaking && streamingBuffer.isEmpty {
            try? AVAudioSession.sharedInstance().setCategory(.playback, mode: .spokenAudio, options: .duckOthers)
            try? AVAudioSession.sharedInstance().setActive(true)
            currentKey = key
            isSpeaking = true
        }

        streamingBuffer += token

        // Speak each complete sentence as it forms
        while let delimIdx = streamingBuffer.firstIndex(where: { Self.sentenceDelimiters.contains($0) }) {
            // Include the delimiter itself
            let endIdx = streamingBuffer.index(after: delimIdx)
            let sentence = String(streamingBuffer[streamingBuffer.startIndex..<endIdx]).trimmingCharacters(in: .whitespacesAndNewlines)
            streamingBuffer = String(streamingBuffer[endIdx...])

            guard !sentence.isEmpty else { continue }
            let cleaned = sanitize(sentence)
            guard !cleaned.isEmpty else { continue }
            let utterance = AVSpeechUtterance(string: cleaned)
            utterance.voice = bestVoice(for: cleaned, fallbackLanguage: fallbackLanguage)
            utterance.rate = AVSpeechUtteranceDefaultSpeechRate
            utterance.prefersAssistiveTechnologySettings = true
            synthesizer.speak(utterance)  // QUEUE_ADD behaviour — AVSpeechSynthesizer queues by default
        }
    }

    /// Speak any leftover text in the buffer after generation finishes.
    func flushStreamingBuffer(fallbackLanguage: AppLanguage, key: String?) {
        let remaining = streamingBuffer.trimmingCharacters(in: .whitespacesAndNewlines)
        streamingBuffer = ""
        guard !remaining.isEmpty else { return }
        let cleaned = sanitize(remaining)
        guard !cleaned.isEmpty else { return }
        let utterance = AVSpeechUtterance(string: cleaned)
        utterance.voice = bestVoice(for: cleaned, fallbackLanguage: fallbackLanguage)
        utterance.rate = AVSpeechUtteranceDefaultSpeechRate
        utterance.prefersAssistiveTechnologySettings = true
        synthesizer.speak(utterance)
    }

    // MARK: - Controls

    func toggleSpeaking(_ text: String, fallbackLanguage: AppLanguage, key: String) {
        if isSpeaking(key: key) {
            stop()
        } else {
            speak(text, fallbackLanguage: fallbackLanguage, key: key)
        }
    }

    func isSpeaking(key: String) -> Bool {
        isSpeaking && currentKey == key
    }

    func stopIfMatching(key: String) {
        guard isSpeaking(key: key) else { return }
        stop()
    }

    func stop() {
        streamingBuffer = ""
        if synthesizer.isSpeaking || synthesizer.isPaused {
            synthesizer.stopSpeaking(at: .immediate)
        }
        isSpeaking = false
        currentKey = nil
    }

    nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
        // Do NOT call setActive(false) here — calling it tears down the audio server
        // connection (IPCAUClient: can't connect to server). SpeechEngine.start() will
        // switch the category and call setActive(true) itself when the mic restarts.
        Task { @MainActor in
            // Only clear speaking state if the native queue is now empty
            if !self.synthesizer.isSpeaking {
                self.isSpeaking = false
                self.currentKey = nil
            }
        }
    }

    nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didCancel utterance: AVSpeechUtterance) {
        Task { @MainActor in
            self.isSpeaking = false
            self.currentKey = nil
        }
    }

    private func bestVoice(for text: String, fallbackLanguage: AppLanguage) -> AVSpeechSynthesisVoice? {
        let preferredCodes = detectedLocaleIdentifiers(for: text) + localeIdentifiers(for: fallbackLanguage)
        for code in preferredCodes {
            if let voice = AVSpeechSynthesisVoice(language: code) {
                return voice
            }
        }
        return AVSpeechSynthesisVoice(language: Locale.current.identifier)
            ?? AVSpeechSynthesisVoice(language: "en-US")
    }

    private func localeIdentifiers(for language: AppLanguage) -> [String] {
        switch language {
        case .systemDefault:
            let current = Locale.current.identifier
            let short = Locale.current.language.languageCode?.identifier ?? "en"
            return [current, short, "en-US"]
        case .english:
            return ["en-US", "en-GB", "en"]
        case .arabic:
            return ["ar-SA", "ar-AE", "ar"]
        case .german:
            return ["de-DE", "de"]
        case .spanish:
            return ["es-ES", "es-MX", "es"]
        case .persian:
            return ["fa-IR", "fa"]
        case .french:
            return ["fr-FR", "fr-CA", "fr"]
        case .hebrew:
            return ["he-IL", "he"]
        case .indonesian:
            return ["id-ID", "id"]
        case .italian:
            return ["it-IT", "it"]
        case .japanese:
            return ["ja-JP", "ja"]
        case .korean:
            return ["ko-KR", "ko"]
        case .polish:
            return ["pl-PL", "pl"]
        case .portuguese:
            return ["pt-BR", "pt-PT", "pt"]
        case .russian:
            return ["ru-RU", "ru"]
        case .turkish:
            return ["tr-TR", "tr"]
        case .ukrainian:
            return ["uk-UA", "uk"]
        case .chinese:
            return ["zh-CN", "zh-TW", "zh-HK", "zh"]
        }
    }

    private func detectedLocaleIdentifiers(for text: String) -> [String] {
        let recognizer = NLLanguageRecognizer()
        recognizer.processString(text)

        guard let dominantLanguage = recognizer.dominantLanguage else { return [] }

        switch dominantLanguage {
        case .english:
            return ["en-US", "en-GB", "en"]
        case .arabic:
            return ["ar-SA", "ar-AE", "ar"]
        case .german:
            return ["de-DE", "de"]
        case .spanish:
            return ["es-ES", "es-MX", "es"]
        case .persian:
            return ["fa-IR", "fa"]
        case .french:
            return ["fr-FR", "fr-CA", "fr"]
        case .hebrew:
            return ["he-IL", "he"]
        case .indonesian:
            return ["id-ID", "id"]
        case .italian:
            return ["it-IT", "it"]
        case .japanese:
            return ["ja-JP", "ja"]
        case .korean:
            return ["ko-KR", "ko"]
        case .polish:
            return ["pl-PL", "pl"]
        case .portuguese:
            return ["pt-BR", "pt-PT", "pt"]
        case .russian:
            return ["ru-RU", "ru"]
        case .turkish:
            return ["tr-TR", "tr"]
        case .ukrainian:
            return ["uk-UA", "uk"]
        case .simplifiedChinese:
            return ["zh-CN", "zh"]
        case .traditionalChinese:
            return ["zh-TW", "zh-HK", "zh"]
        default:
            let raw = dominantLanguage.rawValue
            return raw.isEmpty ? [] : [raw]
        }
    }

    private func sanitize(_ text: String) -> String {
        text
            .replacingOccurrences(of: "`", with: "")
            .replacingOccurrences(of: "*", with: "")
            .replacingOccurrences(of: "#", with: "")
            .replacingOccurrences(of: ">", with: "")
            .replacingOccurrences(of: "_", with: " ")
            .replacingOccurrences(of: "\\[(.*?)\\]\\((.*?)\\)", with: "$1", options: .regularExpression)
            .replacingOccurrences(of: "\\s+", with: " ", options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
