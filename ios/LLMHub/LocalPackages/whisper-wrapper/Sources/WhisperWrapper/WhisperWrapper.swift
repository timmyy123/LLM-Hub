import Foundation
import WhisperBridge

/// Keeps whisper.cpp's ggml headers inside this package. The app only sees
/// Swift values, so its llama.cpp module cannot collide with whisper's ggml.
public actor WhisperEngine {
    public enum EngineError: Error, LocalizedError {
        case couldNotLoadModel(String)
        case transcriptionFailed

        public var errorDescription: String? {
            switch self {
            case .couldNotLoadModel(let path): "Failed to load Whisper model at \(path)"
            case .transcriptionFailed: "Whisper transcription failed"
            }
        }
    }

    private nonisolated(unsafe) var context: UnsafeMutableRawPointer

    public init(path: String) throws {
        guard let loaded = path.withCString({ LLMHubWhisperCreate($0) }) else {
            throw EngineError.couldNotLoadModel(path)
        }
        context = loaded
    }

    deinit { LLMHubWhisperDestroy(context) }

    public func transcribe(samples: [Float]) throws -> String {
        let threadCount = Int32(max(1, min(8, ProcessInfo.processInfo.processorCount - 2)))
        var resultText: UnsafeMutablePointer<CChar>?
        let status = samples.withUnsafeBufferPointer { buffer in
            LLMHubWhisperTranscribe(context, buffer.baseAddress, Int32(buffer.count),
                                    threadCount, &resultText)
        }
        guard status == 0, let resultText else { throw EngineError.transcriptionFailed }
        defer { LLMHubWhisperFreeText(resultText) }
        let text = String(cString: resultText)
        return text
            .replacingOccurrences(of: "[BLANK_AUDIO]", with: "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
