import Foundation

/// whisper.cpp wrapper. Same model as Android — tiny.en.q5_1 — so the model
/// catalog in `ModelData` can be extended with a single entry that targets both
/// platforms.
///
/// TODO(stt):
///   1. Add `https://github.com/ggerganov/whisper.cpp` as a SwiftPM dep. There's
///      a pre-packaged `whisper.spm` branch that builds cleanly on iOS 17+.
///   2. `var ctx = whisper_init_from_file_with_params(modelPath, params)`.
///   3. Transcribe with `whisper_full`; join segments from
///      `whisper_full_get_segment_text(ctx, i)`.
///   4. Metal is enabled by default on Apple Silicon — tiny.en runs ~20x realtime.
final class WhisperSTT {
    let modelURL: URL
    init(modelURL: URL) { self.modelURL = modelURL }

    func load() async throws {
        fatalError("TODO(stt): whisper_init_from_file")
    }

    /// Transcribe a PCM buffer (16 kHz mono, Float32 for whisper.cpp).
    func transcribe(pcm: [Float], languageHint: String? = "en") async throws -> String {
        fatalError("TODO(stt): whisper_full + collect segments")
    }

    func close() { /* TODO: whisper_free */ }
}
