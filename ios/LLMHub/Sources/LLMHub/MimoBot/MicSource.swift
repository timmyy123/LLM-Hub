import Foundation
import AVFoundation

/// Phone microphone as an `AudioSource` using AVAudioEngine's input tap.
///
/// AVAudioEngine gives us the mic in the device's native format (usually
/// 44 100 or 48 000 Hz float). We install an AVAudioConverter on the fly to
/// downsample + convert to 16 kHz mono Int16 and chunk into
/// `MimoBotIds.frameSamples`-sample frames.
///
/// Caller must request NSMicrophoneUsageDescription permission (host app
/// Info.plist) before starting.
final class MicSource: NSObject, AudioSource {

    private let engine = AVAudioEngine()
    private var continuation: AsyncStream<[Int16]>.Continuation?
    private var converter: AVAudioConverter?
    private var targetFormat: AVAudioFormat?
    private var frameBuffer: [Int16] = []
    private let frameSamples = MimoBotIds.frameSamples

    func frames() -> AsyncStream<[Int16]> {
        AsyncStream { continuation in
            self.continuation = continuation
            continuation.onTermination = { [weak self] _ in self?.stop() }
            do { try start() } catch {
                print("⚠️ MicSource start failed: \(error)")
                continuation.finish()
            }
        }
    }

    private func start() throws {
        let input = engine.inputNode
        let inputFormat = input.inputFormat(forBus: 0)
        let tgt = AVAudioFormat(
            commonFormat: .pcmFormatInt16,
            sampleRate: Double(MimoBotIds.sampleRateHz),
            channels: 1,
            interleaved: true
        )!
        targetFormat = tgt
        converter = AVAudioConverter(from: inputFormat, to: tgt)

        try AVAudioSession.sharedInstance().setCategory(.playAndRecord, mode: .voiceChat, options: [.defaultToSpeaker, .allowBluetooth])
        try AVAudioSession.sharedInstance().setActive(true)

        input.installTap(onBus: 0, bufferSize: 1024, format: inputFormat) { [weak self] buffer, _ in
            self?.handle(buffer: buffer)
        }
        try engine.start()
    }

    private func handle(buffer: AVAudioPCMBuffer) {
        guard let converter, let targetFormat else { return }

        // Size the output buffer generously — the converter may emit up to
        // ratio * inputFrames samples.
        let capacity = AVAudioFrameCount(Double(buffer.frameLength) * targetFormat.sampleRate / buffer.format.sampleRate) + 32
        guard let outBuf = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: capacity) else { return }

        var err: NSError?
        var fedInput = false
        let status = converter.convert(to: outBuf, error: &err) { _, inputStatus in
            if fedInput {
                inputStatus.pointee = .noDataNow
                return nil
            }
            fedInput = true
            inputStatus.pointee = .haveData
            return buffer
        }

        if status == .error { return }

        let framesOut = Int(outBuf.frameLength)
        guard framesOut > 0, let ch = outBuf.int16ChannelData?[0] else { return }

        // Append to the running buffer and emit full 320-sample frames.
        frameBuffer.reserveCapacity(frameBuffer.count + framesOut)
        for i in 0..<framesOut { frameBuffer.append(ch[i]) }
        while frameBuffer.count >= frameSamples {
            let frame = Array(frameBuffer.prefix(frameSamples))
            frameBuffer.removeFirst(frameSamples)
            continuation?.yield(frame)
        }
    }

    func stop() {
        if engine.isRunning {
            engine.inputNode.removeTap(onBus: 0)
            engine.stop()
        }
        converter = nil
        targetFormat = nil
        frameBuffer.removeAll(keepingCapacity: false)
        continuation?.finish()
        continuation = nil
    }
}
