import AVFoundation
import Foundation

/// Microphone recording to 16 kHz mono samples (AVAudioEngine), WAV files, and playback (AVAudioPlayer).
@MainActor
final class AudioIO: NSObject, AVAudioPlayerDelegate {
    private var engine: AVAudioEngine?
    private let recorded = SampleBuffer()
    private var player: AVAudioPlayer?
    private var playback: CheckedContinuation<Void, Never>?

    nonisolated static let whisperRate = 16_000.0

    // MARK: Recording

    /// Asks for the microphone the first time (macOS shows NSMicrophoneUsageDescription).
    static func requestMicrophone() async -> Bool {
        switch AVCaptureDevice.authorizationStatus(for: .audio) {
        case .authorized: return true
        case .notDetermined: return await AVCaptureDevice.requestAccess(for: .audio)
        default: return false
        }
    }

    func startRecording() throws {
        stopEngine()
        recorded.reset()
        let engine = AVAudioEngine()
        let input = engine.inputNode
        let format = input.outputFormat(forBus: 0)
        guard format.sampleRate > 0, let target = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: Self.whisperRate, channels: 1, interleaved: false),
              let converter = AVAudioConverter(from: format, to: target) else {
            throw NSError(domain: "LLMHub", code: 43, userInfo: [NSLocalizedDescriptionKey: "No microphone input"])
        }
        Self.installTap(on: input, format: format, converter: converter, target: target, buffer: recorded)
        engine.prepare()
        try engine.start()
        self.engine = engine
    }

    /// Stops recording and returns what was heard, as 16 kHz mono samples.
    func stopRecording() -> [Float] {
        stopEngine()
        return recorded.take()
    }

    private func stopEngine() {
        engine?.inputNode.removeTap(onBus: 0)
        engine?.stop()
        engine = nil
    }

    /// The tap runs on the audio thread, so its closure must not be main-actor isolated.
    nonisolated private static func installTap(on input: AVAudioInputNode, format: AVAudioFormat, converter: AVAudioConverter,
                                               target: AVAudioFormat, buffer: SampleBuffer) {
        input.installTap(onBus: 0, bufferSize: 4096, format: format) { pcm, _ in
            buffer.append(convert(pcm, with: converter, to: target))
        }
    }

    nonisolated private static func convert(_ pcm: AVAudioPCMBuffer, with converter: AVAudioConverter, to format: AVAudioFormat) -> [Float] {
        let capacity = AVAudioFrameCount(Double(pcm.frameLength) * format.sampleRate / pcm.format.sampleRate) + 64
        guard let out = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: capacity) else { return [] }
        var fed = false
        var error: NSError?
        converter.convert(to: out, error: &error) { _, status in
            if fed {
                status.pointee = .noDataNow
                return nil
            }
            fed = true
            status.pointee = .haveData
            return pcm
        }
        guard error == nil, let data = out.floatChannelData else { return [] }
        return Array(UnsafeBufferPointer(start: data[0], count: Int(out.frameLength)))
    }

    // MARK: Files

    /// Any audio file as 16 kHz mono samples, for Whisper.
    nonisolated static func readSamples(_ url: URL) throws -> [Float] {
        let file = try AVAudioFile(forReading: url)
        guard let source = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(file.length)),
              let target = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: whisperRate, channels: 1, interleaved: false),
              let converter = AVAudioConverter(from: file.processingFormat, to: target) else {
            throw NSError(domain: "LLMHub", code: 44, userInfo: [NSLocalizedDescriptionKey: "Unreadable audio \(url.lastPathComponent)"])
        }
        try file.read(into: source)
        let capacity = AVAudioFrameCount(Double(source.frameLength) * whisperRate / file.processingFormat.sampleRate) + 1024
        guard let out = AVAudioPCMBuffer(pcmFormat: target, frameCapacity: capacity) else { return [] }
        var fed = false
        var error: NSError?
        converter.convert(to: out, error: &error) { _, status in
            if fed {
                status.pointee = .endOfStream
                return nil
            }
            fed = true
            status.pointee = .haveData
            return source
        }
        if let error { throw error }
        guard let data = out.floatChannelData else { return [] }
        return Array(UnsafeBufferPointer(start: data[0], count: Int(out.frameLength)))
    }

    /// Writes 32-bit float samples as a 16-bit PCM mono WAV.
    nonisolated static func writeWav(_ url: URL, samples: ArraySlice<Float>, sampleRate: Int) throws {
        var data = Data(capacity: 44 + samples.count * 2)
        func put<T: FixedWidthInteger>(_ v: T) { withUnsafeBytes(of: v.littleEndian) { data.append(contentsOf: $0) } }
        data.append(contentsOf: Array("RIFF".utf8)); put(UInt32(36 + samples.count * 2))
        data.append(contentsOf: Array("WAVEfmt ".utf8)); put(UInt32(16)); put(UInt16(1)); put(UInt16(1))
        put(UInt32(sampleRate)); put(UInt32(sampleRate * 2)); put(UInt16(2)); put(UInt16(16))
        data.append(contentsOf: Array("data".utf8)); put(UInt32(samples.count * 2))
        for s in samples { put(Int16(max(-1, min(1, s)) * 32767)) }
        try data.write(to: url)
    }

    // MARK: Playback

    /// Plays an audio file; returns when it finishes or `stopPlayback` is called.
    func play(_ url: URL) async {
        stopPlayback()
        guard let player = try? AVAudioPlayer(contentsOf: url) else { return }
        player.delegate = self
        self.player = player
        await withCheckedContinuation { continuation in
            playback = continuation
            if !player.play() { finishPlayback() }
        }
    }

    func stopPlayback() {
        player?.stop()
        player = nil
        finishPlayback()
    }

    private func finishPlayback() {
        let continuation = playback
        playback = nil
        continuation?.resume()
    }

    nonisolated func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        Task { @MainActor in
            if self.player === player { self.player = nil }
            self.finishPlayback()
        }
    }
}

/// Samples collected on the audio thread.
final class SampleBuffer: @unchecked Sendable {
    private let lock = NSLock()
    private var samples: [Float] = []

    func append(_ more: [Float]) {
        lock.lock()
        samples += more
        lock.unlock()
    }

    func reset() {
        lock.lock()
        samples = []
        lock.unlock()
    }

    func take() -> [Float] {
        lock.lock()
        defer { samples = []; lock.unlock() }
        return samples
    }
}
