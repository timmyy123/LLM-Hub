import Foundation

/// Opus encode/decode at 16 kHz mono, 20 ms frames.
///
/// iOS doesn't have a system Opus API. Options:
///   - YbridOpus (https://github.com/ybrid/opus-swift) — Swift wrapper over libopus,
///     binary-compatible with the Android side.
///   - Roll a thin XCFramework from the upstream libopus source.
///   - CocoaPods `libopus`.
///
/// Starting choice: YbridOpus as a SwiftPM dep (cleanest). Add to Package.swift
/// dependencies when we're ready to wire it in.
///
/// TODO(opus):
///   1. Add `.package(url: "https://github.com/ybrid/opus-swift", from: "0.8.0")`
///      to ios/LLMHub/Package.swift (only when implementing — avoid extra deps
///      until then so the tree keeps building).
///   2. Replace the bodies below with the real encoder/decoder calls.
///   3. Reuse encoder/decoder instances.
final class OpusEncoderWrap {
    let sampleRateHz: Int
    let channels: Int
    let bitrateBps: Int
    let frameSamples: Int

    init(sampleRateHz: Int = 16_000, channels: Int = 1, bitrateBps: Int = 24_000, frameSamples: Int = 320) {
        self.sampleRateHz = sampleRateHz
        self.channels = channels
        self.bitrateBps = bitrateBps
        self.frameSamples = frameSamples
    }

    func encode(pcm: [Int16]) -> Data {
        precondition(pcm.count == frameSamples, "expected \(frameSamples) samples, got \(pcm.count)")
        fatalError("TODO(opus): encode via libopus")
    }
}

final class OpusDecoderWrap {
    let sampleRateHz: Int
    let channels: Int
    let frameSamples: Int

    init(sampleRateHz: Int = 16_000, channels: Int = 1, frameSamples: Int = 320) {
        self.sampleRateHz = sampleRateHz
        self.channels = channels
        self.frameSamples = frameSamples
    }

    func decode(opusFrame: Data) -> [Int16] {
        fatalError("TODO(opus): decode via libopus")
    }

    func decodePLC() -> [Int16] {
        fatalError("TODO(opus): PLC frame via libopus (data=nil)")
    }
}
