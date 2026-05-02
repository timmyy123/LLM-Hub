import Foundation
import AVFoundation

/// Phone speaker as an `AudioSink` via AVAudioEngine + AVAudioPlayerNode.
///
/// Accepts 16 kHz mono Int16 frames (the pipeline's wire format), packs them
/// into an AVAudioPCMBuffer at the engine's output format, and schedules them
/// on the player node. AVAudioEngine handles resampling to the device's
/// native output rate internally.
final class SpeakerSink: AudioSink {

    private let engine = AVAudioEngine()
    private let player = AVAudioPlayerNode()
    private var inputFormat: AVAudioFormat?
    private var started = false

    func start() {
        if started { return }
        let fmt = AVAudioFormat(
            commonFormat: .pcmFormatInt16,
            sampleRate: Double(MimoBotIds.sampleRateHz),
            channels: 1,
            interleaved: true
        )!
        inputFormat = fmt
        engine.attach(player)
        engine.connect(player, to: engine.mainMixerNode, format: fmt)

        do {
            try AVAudioSession.sharedInstance().setCategory(.playAndRecord, mode: .voiceChat, options: [.defaultToSpeaker, .allowBluetooth])
            try AVAudioSession.sharedInstance().setActive(true)
            try engine.start()
            player.play()
            started = true
        } catch {
            print("⚠️ SpeakerSink start failed: \(error)")
        }
    }

    func write(_ frame: [Int16]) async {
        if !started { start() }
        guard let fmt = inputFormat,
              let buf = AVAudioPCMBuffer(pcmFormat: fmt, frameCapacity: AVAudioFrameCount(frame.count)),
              let ch = buf.int16ChannelData?[0] else { return }
        buf.frameLength = AVAudioFrameCount(frame.count)
        for i in 0..<frame.count { ch[i] = frame[i] }

        await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
            player.scheduleBuffer(buf, at: nil, options: []) { cont.resume() }
        }
    }

    func stop() {
        if !started { return }
        player.stop()
        engine.stop()
        engine.disconnectNodeInput(player)
        engine.detach(player)
        inputFormat = nil
        started = false
    }
}
