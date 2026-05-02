import Foundation
import AVFoundation

/// System TTS via AVSpeechSynthesizer. iOS 16+ exposes a `write` API that
/// hands us an `AVAudioBuffer` as the voice renders — we resample that to
/// 16 kHz mono Int16 and chunk into 320-sample frames for the pipeline.
///
/// Pre-iOS-16 fallback isn't implemented; for now we require iOS 16+ for the
/// Mimo voice test. The rest of the app still works on older iOS.
final class SystemTTS: NSObject, TTS, AVSpeechSynthesizerDelegate {

    private let synthesizer = AVSpeechSynthesizer()

    override init() {
        super.init()
        synthesizer.delegate = self
    }

    func load() async throws { /* no-op */ }

    func speakToPCM(text: String, language: String = "en-US") -> AsyncStream<[Int16]> {
        AsyncStream { continuation in
            guard #available(iOS 16.0, macOS 13.0, *) else {
                print("⚠️ SystemTTS requires iOS 16+ for PCM capture")
                continuation.finish()
                return
            }

            let utter = AVSpeechUtterance(string: text)
            utter.voice = AVSpeechSynthesisVoice(language: language) ?? AVSpeechSynthesisVoice(language: "en-US")

            let targetFormat = AVAudioFormat(
                commonFormat: .pcmFormatInt16,
                sampleRate: Double(MimoBotIds.sampleRateHz),
                channels: 1,
                interleaved: true
            )!
            var converter: AVAudioConverter? = nil
            var buf: [Int16] = []
            let frameSamples = MimoBotIds.frameSamples

            synthesizer.write(utter) { avBuffer in
                guard let pcmBuf = avBuffer as? AVAudioPCMBuffer, pcmBuf.frameLength > 0 else {
                    // End of synthesis — flush any partial frame.
                    if !buf.isEmpty {
                        var tail = buf
                        while tail.count < frameSamples { tail.append(0) }
                        continuation.yield(Array(tail.prefix(frameSamples)))
                        buf.removeAll(keepingCapacity: false)
                    }
                    continuation.finish()
                    return
                }

                if converter == nil {
                    converter = AVAudioConverter(from: pcmBuf.format, to: targetFormat)
                }
                guard let converter else { continuation.finish(); return }

                let capacity = AVAudioFrameCount(Double(pcmBuf.frameLength) * targetFormat.sampleRate / pcmBuf.format.sampleRate) + 32
                guard let outBuf = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: capacity) else { return }

                var err: NSError?
                var fedInput = false
                _ = converter.convert(to: outBuf, error: &err) { _, inputStatus in
                    if fedInput {
                        inputStatus.pointee = .noDataNow
                        return nil
                    }
                    fedInput = true
                    inputStatus.pointee = .haveData
                    return pcmBuf
                }

                let framesOut = Int(outBuf.frameLength)
                guard framesOut > 0, let ch = outBuf.int16ChannelData?[0] else { return }
                buf.reserveCapacity(buf.count + framesOut)
                for i in 0..<framesOut { buf.append(ch[i]) }

                while buf.count >= frameSamples {
                    let frame = Array(buf.prefix(frameSamples))
                    buf.removeFirst(frameSamples)
                    continuation.yield(frame)
                }
            }
        }
    }

    func stop() {
        synthesizer.stopSpeaking(at: .immediate)
    }
}

/// Back-compat alias — prefer `SystemTTS` in new code.
typealias TTSEngine = SystemTTS
