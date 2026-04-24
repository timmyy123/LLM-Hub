import Foundation
import AVFoundation

/// System TTS via AVSpeechSynthesizer, with a custom output tap so we can
/// capture PCM and Opus-encode it for the device.
///
/// iOS 16+ added `write(_:toBufferCallback:)` on AVSpeechSynthesizer which
/// gives us raw PCM without setting up an audio engine — use that.
///
/// TODO(system-tts):
///   1. `AVSpeechUtterance(string:)`, voice = `.init(language:)`.
///   2. `synthesizer.write(utterance) { buffer in ... }` (iOS 16+).
///   3. Resample the voice's native rate (usually 22 050) → 16 kHz with
///      `AVAudioConverter`.
///   4. Chunk into 320-sample Int16 frames for OpusEncoderWrap.
final class SystemTTS: TTS {
    private let synthesizer = AVSpeechSynthesizer()

    func load() async throws { /* no-op — nothing to load */ }

    func speakToPCM(text: String, language: String = "en-US") -> AsyncStream<[Int16]> {
        AsyncStream { continuation in
            // TODO(system-tts): synthesizer.write + AVAudioConverter + 320-frame chunking
            _ = continuation
        }
    }

    func stop() { synthesizer.stopSpeaking(at: .immediate) }
}

/// Back-compat alias. Prefer `SystemTTS` in new code.
typealias TTSEngine = SystemTTS
