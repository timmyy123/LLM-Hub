import Foundation

/// Abstraction over any text-to-speech backend. Implementations emit 16 kHz
/// mono Int16 PCM in 320-sample frames ready for Opus encode.
///
/// v0 ships two implementations:
///   - `SystemTTS` — AVSpeechSynthesizer (free, decent, ships with iOS)
///   - `KokoroTTS` — Kokoro-82M via ONNX Runtime Core ML EP (higher quality)
protocol TTS: AnyObject {
    func load() async throws
    func speakToPCM(text: String, language: String) -> AsyncStream<[Int16]>
    func stop()
}
