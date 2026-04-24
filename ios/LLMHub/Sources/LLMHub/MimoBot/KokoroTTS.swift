import Foundation

/// Kokoro-82M neural TTS via ONNX Runtime (Core ML execution provider on iOS).
///
/// Why Kokoro on iOS:
///   - 82M params, ~160 MB fp16 ONNX → fits comfortably alongside an LLM.
///   - ONNX Runtime iOS has a Core ML EP that maps compatible ops to the ANE /
///     GPU, so inference is fast on Apple Silicon.
///   - Shares the exact same model files as the Android side — one download,
///     two platforms.
///
/// Model assets (see android/KokoroTts.kt for equivalents):
///   - kokoro-v0_19.onnx
///   - voices.bin
///   - phoneme vocab JSON
///
/// TODO(kokoro):
///   1. Add `https://github.com/microsoft/onnxruntime-swift-package-manager`
///      to Package.swift dependencies when implementing.
///   2. Register the Kokoro asset bundle in ModelData / ModelDownloader so
///      first-run download reuses existing plumbing.
///   3. Create `ORTSession` with Core ML EP:
///        let sessionOptions = try ORTSessionOptions()
///        try sessionOptions.appendCoreMLExecutionProvider(with: opts)
///   4. G2P: espeak-ng via small XCFramework, or ship a cached phoneme dict
///      for English only in v0.
///   5. Output is 24 kHz float; resample to 16 kHz, quantize Int16, emit
///      320-sample frames.
final class KokoroTTS: TTS {
    let modelURL: URL
    let voicesURL: URL
    let voiceId: String

    init(modelURL: URL, voicesURL: URL, voiceId: String = "af_bella") {
        self.modelURL = modelURL
        self.voicesURL = voicesURL
        self.voiceId = voiceId
    }

    func load() async throws {
        fatalError("TODO(kokoro): ORTEnv + ORTSession from modelURL with Core ML EP; load voices.bin")
    }

    func speakToPCM(text: String, language: String = "en-US") -> AsyncStream<[Int16]> {
        AsyncStream { continuation in
            // TODO(kokoro): G2P → ORT run → resample 24k→16k → 320-frame chunks
            _ = continuation
        }
    }

    func stop() { /* TODO: cancel the in-flight run */ }
}
