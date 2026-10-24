#if DEBUG
import Foundation

extension DevTools {
    /// `--speechtest "<sentence>" [--voice F1] [--wav file]`: an end-to-end Speech run on this Mac, results in the log.
    /// Synthesizes the sentence with Supertonic, then transcribes the WAV with Whisper and compares the words.
    /// `--wav` transcribes an existing recording instead (the dictation path, minus the microphone).
    static func speechTest(_ args: [String]) async {
        let speech = AppModel.shared.speech
        if let path = value("--wav", args) {
            do {
                let start = Date()
                let text = try await speech.transcribe(URL(fileURLWithPath: path))
                Log.info("SPEECHTEST asr \(path) in \(String(format: "%.1f", Date().timeIntervalSince(start)))s: \(text)")
            } catch {
                Log.error("SPEECHTEST failed", error)
            }
        }
        guard let sentence = value("--speechtest", args) else { return }
        do {
            let voice = value("--voice", args) ?? AppSettings.current.voice
            var start = Date()
            let wav = try await speech.synthesize(sentence, voice: voice)
            let ttsTime = Date().timeIntervalSince(start)
            Log.info("SPEECHTEST tts voice=\(voice) \(String(format: "%.1f", wav.seconds))s of audio in \(String(format: "%.1f", ttsTime))s "
                + "(real-time factor \(String(format: "%.2f", ttsTime / max(wav.seconds, 0.01)))) -> \(wav.url.path)")
            // Second run: the first includes download and model load.
            start = Date()
            let again = try await speech.synthesize(sentence, voice: voice)
            let warm = Date().timeIntervalSince(start)
            Log.info("SPEECHTEST tts warm \(String(format: "%.1f", again.seconds))s of audio in \(String(format: "%.1f", warm))s "
                + "(real-time factor \(String(format: "%.2f", warm / max(again.seconds, 0.01))))")
            try? FileManager.default.removeItem(at: again.url)

            start = Date()
            let text = try await speech.transcribe(wav.url, language: nil)
            let asrTime = Date().timeIntervalSince(start)
            start = Date()
            _ = try await speech.transcribe(wav.url, language: nil)
            let asrWarm = Date().timeIntervalSince(start)
            func words(_ s: String) -> [String] {
                s.lowercased().components(separatedBy: CharacterSet.alphanumerics.inverted).filter { !$0.isEmpty }
            }
            let ok = words(text) == words(sentence)
            Log.info("SPEECHTEST ok=\(ok) asr in \(String(format: "%.1f", asrTime))s (warm \(String(format: "%.1f", asrWarm))s): \"\(text)\"")
        } catch {
            Log.error("SPEECHTEST failed", error)
        }
    }
}
#endif
