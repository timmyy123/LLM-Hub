import Foundation
import NaturalLanguage

/// Supertonic 3 voices, model builds and languages, as the Android app offers them (data/SupertonicModel.kt): ten preset
/// voices F1 to F5 and M1 to M5, voices imported from style JSON files, FP32 or FP16 ONNX builds, 31 languages or auto.
enum SupertonicVoices {
    struct Voice: Identifiable, Hashable {
        let key: String
        let label: String
        var id: String { key }
        var isCustom: Bool { key.hasPrefix("custom_") }
    }

    /// The ONNX build read aloud runs: FP16 (half the download, the default) or the original FP32.
    enum Build: String, CaseIterable {
        case fp16, fp32

        /// Supertone's official FP32 release and Kyumdroid's FP16 conversion (inputs and outputs stay FP32), pinned.
        var repo: String {
            switch self {
            case .fp32: "https://huggingface.co/Supertone/supertonic-3/resolve/3cadd1ee6394adea1bd021217a0e650ede09a323/onnx/"
            case .fp16: "https://huggingface.co/Kyumdroid/supertonic-3-quant/resolve/d9bfd0a9384b42a76f9256eafdd3d8a5cc4706ba/fp16/onnx/"
            }
        }

        var files: [(name: String, size: Int64)] {
            switch self {
            case .fp32: [("duration_predictor.onnx", 3_700_147), ("text_encoder.onnx", 36_416_150), ("vector_estimator.onnx", 256_534_781),
                         ("vocoder.onnx", 101_424_195), ("tts.json", 8253), ("unicode_indexer.json", 277_676)]
            case .fp16: [("duration_predictor.onnx", 2_058_262), ("text_encoder.onnx", 18_567_302), ("vector_estimator.onnx", 128_515_913),
                         ("vocoder.onnx", 50_773_381), ("tts.json", 8253), ("unicode_indexer.json", 277_676)]
            }
        }

        var folder: URL { SpeechService.ttsRoot.appendingPathComponent(self == .fp32 ? "onnx" : "onnx-fp16", isDirectory: true) }
        var size: Int64 { files.reduce(0) { $0 + $1.size } }
        var label: String { self == .fp16 ? "FP16" : "FP32" }

        var isInstalled: Bool {
            files.allSatisfy { file in
                let size = (try? FileManager.default.attributesOfItem(atPath: folder.appendingPathComponent(file.name).path)[.size] as? NSNumber)?.int64Value
                return size == file.size
            }
        }

        static var current: Build { Build(rawValue: AppSettings.current.ttsBuild) ?? .fp16 }
    }

    static let presets = ["F1", "F2", "F3", "F4", "F5", "M1", "M2", "M3", "M4", "M5"]
    /// Preset style files come from the FP32 release (both builds read the same styles).
    static let stylesRepo = "https://huggingface.co/Supertone/supertonic-3/resolve/3cadd1ee6394adea1bd021217a0e650ede09a323/voice_styles/"
    static var presetFolder: URL { SpeechService.ttsRoot.appendingPathComponent("voice_styles", isDirectory: true) }
    /// Imported voices: user data, so they live outside the engine folder and survive removing the engine.
    static var customFolder: URL { AppPaths.root.appendingPathComponent("supertonic-voices", isDirectory: true) }

    private static let customKey = try! NSRegularExpression(pattern: "^custom_[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")

    static func isCustomKey(_ key: String) -> Bool {
        customKey.firstMatch(in: key, range: NSRange(location: 0, length: (key as NSString).length)) != nil
    }

    static func isVoice(_ key: String) -> Bool { presets.contains(key) || (isCustomKey(key) && file(key) != nil) }

    /// The style file for a voice key (presets may still need downloading; custom ones exist or nil).
    static func file(_ key: String) -> URL? {
        if presets.contains(key) { return presetFolder.appendingPathComponent("\(key).json") }
        guard isCustomKey(key) else { return nil }
        let url = customFolder.appendingPathComponent("\(key).json")
        return FileManager.default.fileExists(atPath: url.path) ? url : nil
    }

    static func label(_ key: String) -> String {
        guard isCustomKey(key) else { return key }
        let name = (try? String(contentsOf: customFolder.appendingPathComponent("\(key).txt"), encoding: .utf8)) ?? ""
        return name.isEmpty ? key : String(name.prefix(80))
    }

    static func custom() -> [Voice] {
        let files = (try? FileManager.default.contentsOfDirectory(at: customFolder, includingPropertiesForKeys: [.creationDateKey])) ?? []
        return files.filter { $0.pathExtension == "json" && isCustomKey($0.deletingPathExtension().lastPathComponent) }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
            .map { Voice(key: $0.deletingPathExtension().lastPathComponent, label: label($0.deletingPathExtension().lastPathComponent)) }
    }

    /// Presets, then imported voices, for the voice pickers.
    static func all() -> [Voice] { presets.map { Voice(key: $0, label: $0) } + custom() }

    /// Validates a style JSON (2 MiB at most; style_ttl [1, 50, 256], style_dp [1, 8, 16]) and stores it under a fresh
    /// key; the source file name is only kept as the label, never used as a path. Returns the new key.
    static func importVoice(from source: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: source)
        defer { try? handle.close() }
        let data = try handle.read(upToCount: Supertonic.maxVoiceBytes + 1) ?? Data()
        try Supertonic.TextToSpeech.validateStyle(data)
        try FileManager.default.createDirectory(at: customFolder, withIntermediateDirectories: true)
        let key = "custom_" + UUID().uuidString.lowercased()
        let target = customFolder.appendingPathComponent("\(key).json")
        let label = customFolder.appendingPathComponent("\(key).txt")
        do {
            try Data(source.deletingPathExtension().lastPathComponent.prefix(80).utf8).write(to: label, options: .atomic)
            try data.write(to: target, options: .atomic)
        } catch {
            try? FileManager.default.removeItem(at: target)
            try? FileManager.default.removeItem(at: label)
            throw error
        }
        Log.info("Imported Supertonic voice \(key) (\(data.count) bytes)")
        return key
    }

    static func delete(_ key: String) {
        guard isCustomKey(key) else { return }
        try? FileManager.default.removeItem(at: customFolder.appendingPathComponent("\(key).json"))
        try? FileManager.default.removeItem(at: customFolder.appendingPathComponent("\(key).txt"))
        if AppSettings.current.voice == key {
            AppSettings.current.voice = "F1"
            AppSettings.save()
        }
    }

    // MARK: Language

    /// The language name in the app's language, for the pickers.
    static func languageName(_ code: String) -> String {
        Locale.current.localizedString(forLanguageCode: code)?.capitalized(with: Locale.current) ?? code
    }

    /// The setting's language, or for auto ("") the text's own language when Supertonic speaks it (Apple's
    /// NaturalLanguage recognizer), then the app language, then English.
    static func language(for text: String, setting: String = AppSettings.current.ttsLanguage) -> String {
        if Supertonic.languages.contains(setting) { return setting }
        let recognizer = NLLanguageRecognizer()
        recognizer.languageConstraints = Supertonic.spoken.map { NLLanguage(rawValue: $0) }
        recognizer.processString(String(text.prefix(1000)))
        if let best = recognizer.languageHypotheses(withMaximum: 1).first, best.value > 0.5, Supertonic.languages.contains(best.key.rawValue) {
            return best.key.rawValue
        }
        let tag = L.override.isEmpty ? (Locale.preferredLanguages.first ?? "en") : L.override
        let ui = String(tag.prefix(2))
        return Supertonic.languages.contains(ui) ? ui : "en"
    }
}
