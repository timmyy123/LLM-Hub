import Foundation

/// whisper.cpp-backed `SpeechToText`.
///
/// NOT IMPLEMENTED YET — iOS v0 uses `SFSpeechRecognizerSTT`. When this lands,
/// it'll consume PCM from a `MicSource` (or the BLE transport) and run
/// whisper.cpp inference via a SwiftPM dependency.
///
/// TODO(whisper-ios):
///   1. `https://github.com/ggerganov/whisper.cpp` — the `whisper.spm` branch
///      builds cleanly on iOS 17+.
///   2. Transcribe once the mic source reports end-of-utterance (VAD or PTT).
final class WhisperSTT: SpeechToText {
    let modelURL: URL
    init(modelURL: URL) { self.modelURL = modelURL }

    func recognizeTurn(languageHint: String) async -> String {
        fatalError("TODO(whisper-ios): whisper.cpp init + transcribe")
    }

    func cancel() { /* TODO */ }
    func close() { /* TODO */ }
}
