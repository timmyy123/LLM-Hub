#if DEBUG
import AVFoundation
import AppKit

extension DevTools {
    /// `--dictatetest <seconds> [--noprompt]`: the chat composer's mic button on the live microphone. Starts dictation the
    /// way the first click does, records for the given seconds (play speech near the Mac meanwhile; "DICTATETEST recording"
    /// in the log is the cue), stops the way the second click does, and logs the text that landed in the chat draft.
    /// `--noprompt` only logs the microphone permission when macOS has not asked yet, instead of asking.
    static func dictateTest(_ args: [String]) async {
        guard let text = value("--dictatetest", args), let seconds = Double(text) else { return }
        let app = AppModel.shared
        let chat = app.chat
        for i in 0..<40 where !NSApp.windows.contains(where: { $0.isVisible && $0.frame.width > 600 }) {
            if i == 8 { NSApp.activate() }
            try? await Task.sleep(for: .milliseconds(250))
        }
        app.section = .chat
        let status = AVCaptureDevice.authorizationStatus(for: .audio)
        let names: [AVAuthorizationStatus: String] = [.authorized: "authorized", .denied: "denied", .restricted: "restricted", .notDetermined: "not asked yet"]
        Log.info("DICTATETEST microphone permission: \(names[status] ?? "\(status.rawValue)"), input: \(AVCaptureDevice.default(for: .audio)?.localizedName ?? "none")")
        if status == .notDetermined && args.contains("--noprompt") { return }
        chat.draft = ""
        await chat.toggleDictation()
        guard app.speech.isRecording else {
            Log.info("DICTATETEST did not start: \(chat.attachError)")
            return
        }
        Log.info("DICTATETEST recording for \(text)s")
        try? await Task.sleep(for: .seconds(seconds))
        let start = Date()
        await chat.toggleDictation()
        Log.info("DICTATETEST stopped; transcribed in \(String(format: "%.1f", Date().timeIntervalSince(start)))s: draft='\(chat.draft)' error='\(chat.attachError)'")
    }

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
            // --build fp16|fp32 picks the Supertonic build for this run, --lang the language (else auto),
            // --importvoice <json> imports a style file first and speaks with it (then deletes it unless --keepvoice).
            if let build = value("--build", args) { AppSettings.current.ttsBuild = build }
            let language = value("--lang", args)
            var voice = value("--voice", args) ?? AppSettings.current.voice
            var imported: String?
            if let path = value("--importvoice", args) {
                voice = try speech.importVoice(from: URL(fileURLWithPath: path))
                imported = voice
                Log.info("SPEECHTEST imported \(path) as \(voice) (\(SupertonicVoices.label(voice)))")
            }
            defer {
                if let imported, !args.contains("--keepvoice") { speech.deleteVoice(imported) }
            }
            let detected = language ?? SupertonicVoices.language(for: sentence)
            var start = Date()
            let wav = try await speech.synthesize(sentence, voice: voice, language: language)
            let ttsTime = Date().timeIntervalSince(start)
            Log.info("SPEECHTEST tts \(SpeechService.ttsName) voice=\(voice) lang=\(detected) \(String(format: "%.1f", wav.seconds))s of audio in \(String(format: "%.1f", ttsTime))s "
                + "(real-time factor \(String(format: "%.2f", ttsTime / max(wav.seconds, 0.01)))) -> \(wav.url.path)")
            // Second run: the first includes download and model load.
            start = Date()
            let again = try await speech.synthesize(sentence, voice: voice, language: language)
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
            let heard = Set(words(text))
            let match = Double(words(sentence).filter { heard.contains($0) }.count) / Double(max(1, words(sentence).count))
            Log.info("SPEECHTEST ok=\(ok) words \(String(format: "%.0f", match * 100))% asr in \(String(format: "%.1f", asrTime))s (warm \(String(format: "%.1f", asrWarm))s): \"\(text)\"")
        } catch {
            Log.error("SPEECHTEST failed", error)
        }
    }
}
#endif
