import Foundation
@preconcurrency import AVFoundation
import WhisperWrapper

private enum WhisperError: Error, LocalizedError {
    case modelNotLoaded
    case transcriptionFailed

    var errorDescription: String? {
        switch self {
        case .modelNotLoaded: return "No Whisper model loaded"
        case .transcriptionFailed: return "Whisper transcription failed"
        }
    }
}

/// Convert any audio file to 16 kHz mono Float32 PCM samples.
private func toWhisperSamples(from url: URL) async throws -> [Float] {
    let file = try AVAudioFile(forReading: url)
    let outFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                  sampleRate: 16000, channels: 1, interleaved: false)!
    guard let converter = AVAudioConverter(from: file.processingFormat, to: outFormat) else {
        throw WhisperError.transcriptionFailed
    }
    let frameCapacity = AVAudioFrameCount(
        outFormat.sampleRate * Double(file.length) / file.processingFormat.sampleRate + 1024)
    guard let outBuf = AVAudioPCMBuffer(pcmFormat: outFormat, frameCapacity: frameCapacity),
          let inBuf  = AVAudioPCMBuffer(pcmFormat: file.processingFormat,
                                        frameCapacity: AVAudioFrameCount(file.length))
    else { throw WhisperError.transcriptionFailed }
    try file.read(into: inBuf)
    nonisolated(unsafe) var inputDone = false
    var convertError: NSError?
    converter.convert(to: outBuf, error: &convertError) { _, status in
        if inputDone { status.pointee = .noDataNow; return nil }
        status.pointee = .haveData; inputDone = true; return inBuf
    }
    if let e = convertError { throw e }
    let count = Int(outBuf.frameLength)
    guard let chan = outBuf.floatChannelData?[0] else { return [] }
    return Array(UnsafeBufferPointer(start: chan, count: count))
}

// MARK: - WhisperBackend

@MainActor
public final class WhisperBackend: ObservableObject {
    public static let shared = WhisperBackend()

    @Published public var isLoaded = false
    @Published public var isTranscribing = false
    @Published public var currentModelName: String?

    private var whisperCtx: WhisperEngine?
    private init() {}

    public func load(modelPath: String, modelName: String) async throws {
        if currentModelName == modelName, isLoaded { return }
        unload()
        let ctx = try await Task.detached(priority: .userInitiated) {
            try WhisperEngine(path: modelPath)
        }.value
        whisperCtx = ctx
        isLoaded = true
        currentModelName = modelName
    }

    public func unload() {
        whisperCtx = nil
        isLoaded = false
        currentModelName = nil
    }

    public func transcribeFile(url: URL) async throws -> String {
        guard let ctx = whisperCtx else { throw WhisperError.modelNotLoaded }
        isTranscribing = true
        defer { isTranscribing = false }
        let samples = try await Task.detached(priority: .userInitiated) {
            try await toWhisperSamples(from: url)
        }.value
        return try await Task.detached(priority: .userInitiated) {
            try await ctx.transcribe(samples: samples)
        }.value
    }
}
