import Foundation

/// Source/sink abstractions so `VoicePipeline` runs against either the phone's
/// own mic+speaker (local dev mode) or a BLE transport + Opus codec (real
/// device). All frames are 16 kHz mono Int16, `MimoBotIds.frameSamples` per frame.
protocol AudioSource: AnyObject {
    /// Yields 320-sample Int16 frames as they arrive. Finishes when `stop()` is called.
    func frames() -> AsyncStream<[Int16]>
    func stop()
}

protocol AudioSink: AnyObject {
    /// Must be called before `write`. Idempotent.
    func start()
    func write(_ frame: [Int16]) async
    func stop()
}
