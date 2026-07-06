import Foundation
import AVFoundation
import whisper

private enum WhisperError: Error, LocalizedError {
    case modelNotLoaded
    case couldNotLoadModel(String)
    case transcriptionFailed

    var errorDescription: String? {
        switch self {
        case .modelNotLoaded: return "No Whisper model loaded"
        case .couldNotLoadModel(let p): return "Failed to load Whisper model at \(p)"
        case .transcriptionFailed: return "Whisper transcription failed"
        }
    }
}

private actor WhisperContext {
    private var ctx: OpaquePointer

    init(ctx: OpaquePointer) { self.ctx = ctx }
    deinit { whisper_free(ctx) }

    static func load(path: String) throws -> WhisperContext {
        var params = whisper_context_default_params()
        params.use_gpu = false
        guard let c = whisper_init_from_file_with_params(path, params) else {
            throw WhisperError.couldNotLoadModel(path)
        }
        return WhisperContext(ctx: c)
    }

    func transcribe(samples: [Float]) throws -> String {
        let nThreads = Int32(max(1, min(8, ProcessInfo.processInfo.processorCount - 2)))
        var params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY)
        params.n_threads        = nThreads
        params.print_realtime   = false
        params.print_progress   = false
        params.print_timestamps = false
        params.print_special    = false
        params.translate        = false  // never translate — preserve original language
        params.language         = nil    // nil = auto-detect, do not force English
        params.no_context       = true
        params.single_segment   = false
        let result = samples.withUnsafeBufferPointer { buf in
            whisper_full(ctx, params, buf.baseAddress, Int32(buf.count))
        }
        guard result == 0 else { throw WhisperError.transcriptionFailed }
        var text = ""
        for i in 0..<whisper_full_n_segments(ctx) {
            if let t = whisper_full_get_segment_text(ctx, i) {
                text += String(cString: t)
            }
        }
        return text.trimmingCharacters(in: .whitespacesAndNewlines)
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
    var inputDone = false
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

    private var whisperCtx: WhisperContext?
    private init() {}

    public func load(modelPath: String, modelName: String) async throws {
        if currentModelName == modelName, isLoaded { return }
        unload()
        let ctx = try await Task.detached(priority: .userInitiated) {
            try WhisperContext.load(path: modelPath)
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
