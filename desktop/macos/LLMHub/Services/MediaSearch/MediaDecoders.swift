import AVFoundation
import Foundation
import ImageIO
import UniformTypeIdentifiers

/// Image and audio decoding for media search (core MediaDecoders.cs): ImageIO scales photos to a JPEG, AVFoundation
/// decodes any audio format it knows to 16 kHz mono.
enum MediaDecoders {
    static let sampleRate = 16_000

    static let imageExtensions: Set<String> = ["jpg", "jpeg", "png", "webp", "bmp", "gif", "heic", "heif", "tif", "tiff"]
    static let audioExtensions: Set<String> = ["wav", "mp3", "flac", "ogg", "oga", "opus", "m4a", "aac", "wma", "aiff", "aif", "caf"]

    /// A JPEG of the image, no larger than maxEdge on its long side and turned upright by its EXIF orientation; nil if
    /// it cannot be read.
    static func image(_ url: URL, maxEdge: Int = 512) -> Data? {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, [kCGImageSourceShouldCache: false] as CFDictionary),
              let image = CGImageSourceCreateThumbnailAtIndex(source, 0, [
                  kCGImageSourceCreateThumbnailFromImageAlways: true,
                  kCGImageSourceThumbnailMaxPixelSize: maxEdge,
                  kCGImageSourceCreateThumbnailWithTransform: true,
              ] as CFDictionary) else { return nil }
        let data = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(data, UTType.jpeg.identifier as CFString, 1, nil) else { return nil }
        CGImageDestinationAddImage(destination, image, [kCGImageDestinationLossyCompressionQuality: 0.9] as CFDictionary)
        return CGImageDestinationFinalize(destination) ? data as Data : nil
    }

    /// The first maxSeconds of an audio file as 16 kHz mono floats; nil if it cannot be decoded. AVAudioFile and
    /// AVAudioConverter first (WAV, AIFF, CAF, MP3, AAC/M4A, FLAC), then AVAssetReader for other containers.
    static func audio(_ url: URL, maxSeconds: Int) async -> [Float]? {
        if let samples = try? readWithAudioFile(url, maxSeconds: maxSeconds) { return samples }
        return await readWithAssetReader(url, maxSeconds: maxSeconds)
    }

    private static var target: AVAudioFormat {
        AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: Double(sampleRate), channels: 1, interleaved: false)!
    }

    private static func readWithAudioFile(_ url: URL, maxSeconds: Int) throws -> [Float]? {
        let file = try AVAudioFile(forReading: url)
        let input = file.processingFormat
        let target = target
        guard let converter = AVAudioConverter(from: input, to: target) else { return nil }
        converter.downmix = true
        let limit = min(file.length, AVAudioFramePosition(Double(maxSeconds) * input.sampleRate))
        let chunk: AVAudioFrameCount = 32_768
        guard let out = AVAudioPCMBuffer(pcmFormat: target, frameCapacity: chunk) else { return nil }
        var samples: [Float] = []
        samples.reserveCapacity(Int(Double(limit) * target.sampleRate / input.sampleRate) + 1024)
        var read: AVAudioFramePosition = 0
        var finished = false
        while true {
            out.frameLength = 0
            var error: NSError?
            let status = converter.convert(to: out, error: &error) { _, inputStatus in
                let remaining = limit - read
                guard !finished, remaining > 0,
                      let buffer = AVAudioPCMBuffer(pcmFormat: input, frameCapacity: AVAudioFrameCount(min(Int64(chunk), remaining))),
                      (try? file.read(into: buffer, frameCount: buffer.frameCapacity)) != nil, buffer.frameLength > 0 else {
                    finished = true
                    inputStatus.pointee = .endOfStream
                    return nil
                }
                read += AVAudioFramePosition(buffer.frameLength)
                inputStatus.pointee = .haveData
                return buffer
            }
            if let error { throw error }
            if let data = out.floatChannelData, out.frameLength > 0 {
                samples.append(contentsOf: UnsafeBufferPointer(start: data[0], count: Int(out.frameLength)))
            }
            if status == .endOfStream || status == .error || (status == .inputRanDry && finished) { break }
        }
        return samples
    }

    /// AVAssetReader resamples and downmixes itself when asked for 16 kHz mono float PCM.
    private static func readWithAssetReader(_ url: URL, maxSeconds: Int) async -> [Float]? {
        let asset = AVURLAsset(url: url)
        guard let track = try? await asset.loadTracks(withMediaType: .audio).first, let reader = try? AVAssetReader(asset: asset) else { return nil }
        reader.timeRange = CMTimeRange(start: .zero, duration: CMTime(seconds: Double(maxSeconds), preferredTimescale: 600))
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: [
            AVFormatIDKey: kAudioFormatLinearPCM, AVSampleRateKey: sampleRate, AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 32, AVLinearPCMIsFloatKey: true, AVLinearPCMIsNonInterleaved: false, AVLinearPCMIsBigEndianKey: false,
        ])
        guard reader.canAdd(output) else { return nil }
        reader.add(output)
        guard reader.startReading() else { return nil }
        var samples: [Float] = []
        while let buffer = output.copyNextSampleBuffer() {
            guard let block = CMSampleBufferGetDataBuffer(buffer) else { continue }
            let length = CMBlockBufferGetDataLength(block)
            var chunk = [Float](repeating: 0, count: length / MemoryLayout<Float>.size)
            chunk.withUnsafeMutableBytes { _ = CMBlockBufferCopyDataBytes(block, atOffset: 0, dataLength: length, destination: $0.baseAddress!) }
            samples += chunk
        }
        return reader.status == .completed && !samples.isEmpty ? samples : nil
    }

    /// A 16 kHz mono PCM16 WAV of samples[from, to).
    static func wav(_ samples: [Float], from: Int, to: Int) -> Data {
        let count = max(0, to - from)
        var data = Data(capacity: 44 + count * 2)
        func put<T: FixedWidthInteger>(_ v: T) { withUnsafeBytes(of: v.littleEndian) { data.append(contentsOf: $0) } }
        data.append(contentsOf: Array("RIFF".utf8)); put(UInt32(36 + count * 2))
        data.append(contentsOf: Array("WAVEfmt ".utf8)); put(UInt32(16)); put(UInt16(1)); put(UInt16(1))
        put(UInt32(sampleRate)); put(UInt32(sampleRate * 2)); put(UInt16(2)); put(UInt16(16))
        data.append(contentsOf: Array("data".utf8)); put(UInt32(count * 2))
        for i in from ..< max(from, to) { put(Int16(max(-1, min(1, samples[i])) * 32767)) }
        return data
    }

    static func rms(_ samples: [Float], from: Int, to: Int) -> Float {
        guard to > from else { return 0 }
        var sum = 0.0
        for i in from ..< to { sum += Double(samples[i] * samples[i]) }
        return Float((sum / Double(to - from)).squareRoot())
    }
}
