import Foundation
import MagentaLiteRTBridge

/// Preallocated SPSC audio storage: one generation worker and one render thread.
/// The render path uses lock-free C++ atomics; it never waits or allocates.
public final class MagentaAudioBuffer: @unchecked Sendable {
    private let ring: OpaquePointer

    public init(capacityFrames: Int, startupFrames: Int) throws {
        guard capacityFrames > 0, startupFrames >= 0, startupFrames <= capacityFrames else {
            throw POSIXError(.EINVAL)
        }
        guard let ring = MRTAudioRingCreate(capacityFrames, startupFrames) else {
            throw POSIXError(.ENOMEM)
        }
        self.ring = ring
    }
    deinit { MRTAudioRingDestroy(ring) }

    public var bufferedFrames: Int { MRTAudioRingBuffered(ring) }
    public var underrunFrames: UInt64 { MRTAudioRingUnderruns(ring) }
    public var isStopped: Bool { MRTAudioRingStopped(ring) }

    public func writePCM(_ pcm: Data, frameOffset: Int = 0) -> Int {
        guard frameOffset >= 0, frameOffset < pcm.count / 4 else { return 0 }
        return pcm.withUnsafeBytes { bytes in
            MRTAudioRingWrite(ring, bytes.baseAddress!.assumingMemoryBound(to: UInt8.self)
                .advanced(by: frameOffset * 4), pcm.count / 4 - frameOffset)
        }
    }

    @discardableResult
    public func read(left: UnsafeMutablePointer<Float>, right: UnsafeMutablePointer<Float>, frames: Int) -> Int {
        MRTAudioRingRead(ring, left, right, frames)
    }

    public func finish() { MRTAudioRingFinish(ring) }
    public func stop() { MRTAudioRingStop(ring) }
}
