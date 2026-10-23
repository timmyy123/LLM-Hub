import Foundation
import Observation

/// Read aloud (Supertonic) and dictation (whisper.cpp), on this Mac. Studio › Voice uses `synthesize` too.
@MainActor @Observable
final class SpeechService {
    static let voices = ["F1", "F2", "M1", "M2"]
    /// Setup progress or the last error, for the composer and Settings › Speech and voice.
    var status = ""
    /// Id of the message being read aloud, or nil.
    var speakingId: String?
    var isRecording = false
    var isTranscribing = false

    /// Speaks `text` in `voice` to a WAV file (44.1 kHz) and returns it with its length in seconds.
    func synthesize(_ text: String, voice: String, language: String? = nil) async throws -> (url: URL, seconds: Double) {
        throw NSError(domain: "LLMHub", code: 90, userInfo: [NSLocalizedDescriptionKey: "Speech is not set up yet"])
    }

    /// Plays a WAV file; returns when it finishes or is stopped.
    func play(_ url: URL) async {}

    func stopSpeaking() {}
}
