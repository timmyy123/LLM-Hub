import AVFoundation
import Foundation
import UIKit

/// First frame and, for clips longer than a second, the frame near the end.
func videoKeyframes(url: URL, maxEdge: CGFloat = 512) -> [Data] {
    let asset = AVURLAsset(url: url)
    let generator = AVAssetImageGenerator(asset: asset)
    generator.appliesPreferredTrackTransform = true
    generator.maximumSize = CGSize(width: maxEdge, height: maxEdge)
    let seconds = CMTimeGetSeconds(asset.duration)
    var times: [CMTime] = [.zero]
    if seconds.isFinite, seconds > 1 {
        times.append(CMTime(seconds: max(0, seconds - 0.5), preferredTimescale: 600))
    }
    var frames: [Data] = []
    for time in times {
        guard let cg = try? generator.copyCGImage(at: time, actualTime: nil),
              let jpeg = mediaSearchJPEG(from: UIImage(cgImage: cg), maxEdge: maxEdge) else { continue }
        frames.append(jpeg)
    }
    return frames
}

// MARK: - Shared engine + storage for Photo Search and Audio Search
// Mirrors AI Edge Gallery's Instant Media Search / Video Moment Finder: EmbeddingGemma 2 with a
// 70-token vision budget and 256-token inputs, vectors stored locally and ranked by cosine.

enum MediaSearchConfig {
    static let visionTokensPerImage = 70
    static let maxInputTokens = 256
    static let audioSampleRate = 16_000

    static var directory: URL {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        let dir = base.appendingPathComponent("media_search", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    /// Downloaded multimodal EmbeddingGemma 2 builds.
    static func downloadedModels() -> [AIModel] {
        ModelData.allModels().filter {
            $0.category == .embedding && $0.modelFormat == .litertlm && $0.supportsVision
                && ModelData.isModelFullyAvailableLocally($0)
        }
    }
}

/// One embedded item: a photo, or an audio moment covering startMs..<endMs of a file.
struct MediaVector: Sendable {
    let id: String
    let startMs: Int32
    let endMs: Int32
    let vector: [Float]
}

/// Binary on-disk vector store. Confine each instance to one actor / the main actor.
final class MediaIndexStore {
    private let fileURL: URL
    private var records: [String: MediaVector] = [:]
    private var dirty = false

    init(name: String) {
        fileURL = MediaSearchConfig.directory.appendingPathComponent("\(name).idx")
    }

    var all: Dictionary<String, MediaVector>.Values { records.values }
    var ids: Set<String> { Set(records.values.map(\.id)) }

    func load() {
        records.removeAll()
        guard let data = try? Data(contentsOf: fileURL), data.count >= 8 else { return }
        var reader = DataReader(data: data)
        guard reader.int32() == Self.magic, let count = reader.int32() else { return }
        for _ in 0..<count {
            guard let id = reader.string(), let start = reader.int32(), let end = reader.int32(),
                  let dim = reader.int32(), let vector = reader.floats(Int(dim)) else {
                records.removeAll()
                return
            }
            records[Self.key(id, start)] = MediaVector(id: id, startMs: start, endMs: end, vector: vector)
        }
    }

    func put(_ item: MediaVector) {
        records[Self.key(item.id, item.startMs)] = item
        dirty = true
    }

    func remove(ids: Set<String>) {
        let before = records.count
        records = records.filter { !ids.contains($0.value.id) }
        if records.count != before { dirty = true }
    }

    func vectors(for id: String) -> [MediaVector] { records.values.filter { $0.id == id } }

    func clear() {
        records.removeAll()
        try? FileManager.default.removeItem(at: fileURL)
        dirty = false
    }

    func saveIfDirty() {
        guard dirty else { return }
        var data = Data()
        data.appendInt32(Self.magic)
        data.appendInt32(Int32(records.count))
        for r in records.values {
            let idBytes = Data(r.id.utf8)
            data.appendInt32(Int32(idBytes.count))
            data.append(idBytes)
            data.appendInt32(r.startMs)
            data.appendInt32(r.endMs)
            data.appendInt32(Int32(r.vector.count))
            r.vector.withUnsafeBufferPointer { data.append(Data(buffer: $0)) }
        }
        try? data.write(to: fileURL, options: .atomic)
        dirty = false
    }

    private static let magic: Int32 = 0x4C484D31
    private static func key(_ id: String, _ start: Int32) -> String { "\(id)#\(start)" }
}

private struct DataReader {
    let data: Data
    var offset = 0

    mutating func int32() -> Int32? {
        guard offset + 4 <= data.count else { return nil }
        defer { offset += 4 }
        return data.subdata(in: offset..<offset + 4).withUnsafeBytes { $0.loadUnaligned(as: Int32.self) }
    }

    mutating func string() -> String? {
        guard let len = int32(), len >= 0, offset + Int(len) <= data.count else { return nil }
        defer { offset += Int(len) }
        return String(data: data.subdata(in: offset..<offset + Int(len)), encoding: .utf8)
    }

    mutating func floats(_ n: Int) -> [Float]? {
        let bytes = n * 4
        guard n >= 0, offset + bytes <= data.count else { return nil }
        defer { offset += bytes }
        return data.subdata(in: offset..<offset + bytes).withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }
}

private extension Data {
    mutating func appendInt32(_ value: Int32) {
        Swift.withUnsafeBytes(of: value.littleEndian) { append(contentsOf: $0) }
    }
}

/// Embeddings are L2-normalized, so the dot product is the cosine similarity.
func mediaDot(_ a: [Float], _ b: [Float]) -> Float {
    guard a.count == b.count else { return -1 }
    var sum: Float = 0
    for i in a.indices { sum += a[i] * b[i] }
    return sum
}

// MARK: - Engine

/// EmbeddingGemma 2 with vision + audio encoders, separate from the text-only memory engine.
actor MediaSearchEngine {
    private let service = EmbeddingService()
    private(set) var loadedModelId: String?

    func load(model: AIModel) async throws {
        if loadedModelId == model.id, await service.isInitialized { return }
        let dir = try SimplifiedFileManager.shared.getModelFolderURL(modelId: model.id, framework: model.inferenceFramework)
        guard let file = (try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil))?
            .first(where: { $0.pathExtension.lowercased() == "litertlm" }) else {
            throw EmbeddingError.modelLoadFailed("EmbeddingGemma 2 file not found")
        }
        let cache = dir.appendingPathComponent("media_search_cache", isDirectory: true)
        try? FileManager.default.createDirectory(at: cache, withIntermediateDirectories: true)
        try await service.initializeLiteRTLM(
            modelID: model.id,
            modelPath: file.path,
            modelName: model.name,
            cacheDir: cache.path,
            supportsImage: model.supportsVision,
            supportsAudio: model.supportsAudio,
            visionTokensPerImage: MediaSearchConfig.visionTokensPerImage,
            maxInputLength: MediaSearchConfig.maxInputTokens
        )
        loadedModelId = model.id
    }

    func unload() async {
        await service.cleanup()
        loadedModelId = nil
    }

    func embedQuery(_ text: String) async -> [Float]? {
        let v = try? await service.embed(text, isQuery: true)
        return (v?.isEmpty ?? true) ? nil : v
    }

    func embedImage(_ jpeg: Data) async -> [Float]? {
        try? await service.embedImage(jpeg)
    }

    func embedImages(_ frames: [Data]) async -> [Float]? {
        try? await service.embedImages(frames)
    }

    func embedAudio(_ wav: Data) async -> [Float]? {
        try? await service.embedAudio(wav)
    }
}

// MARK: - Image / audio helpers

/// Downscale and JPEG-encode an image for the embedder.
func mediaSearchJPEG(from image: UIImage, maxEdge: CGFloat = 512) -> Data? {
    let longest = max(image.size.width, image.size.height)
    guard longest > 0 else { return nil }
    let scale = min(1, maxEdge / longest)
    let size = CGSize(width: (image.size.width * scale).rounded(), height: (image.size.height * scale).rounded())
    let format = UIGraphicsImageRendererFormat.default()
    format.scale = 1
    let resized = UIGraphicsImageRenderer(size: size, format: format).image { _ in
        image.draw(in: CGRect(origin: .zero, size: size))
    }
    return resized.jpegData(compressionQuality: 0.9)
}

/// Decode the first `maxSeconds` of an audio file to 16 kHz mono floats.
func decodeAudio16kMono(url: URL, maxSeconds: Int) -> [Float]? {
    guard let file = try? AVAudioFile(forReading: url),
          let outFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: Double(MediaSearchConfig.audioSampleRate), channels: 1, interleaved: false),
          let converter = AVAudioConverter(from: file.processingFormat, to: outFormat) else { return nil }
    let maxFrames = maxSeconds * MediaSearchConfig.audioSampleRate
    let chunk: AVAudioFrameCount = 8192
    guard let inBuffer = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: chunk),
          let outBuffer = AVAudioPCMBuffer(pcmFormat: outFormat, frameCapacity: chunk) else { return nil }
    var samples: [Float] = []
    samples.reserveCapacity(min(maxFrames, Int(Double(file.length) * outFormat.sampleRate / file.processingFormat.sampleRate)))
    var endOfStream = false
    while samples.count < maxFrames {
        outBuffer.frameLength = 0
        var error: NSError?
        let status = converter.convert(to: outBuffer, error: &error) { _, inputStatus in
            if endOfStream {
                inputStatus.pointee = .endOfStream
                return nil
            }
            do {
                try file.read(into: inBuffer, frameCount: chunk)
            } catch {
                endOfStream = true
                inputStatus.pointee = .endOfStream
                return nil
            }
            if inBuffer.frameLength == 0 {
                endOfStream = true
                inputStatus.pointee = .endOfStream
                return nil
            }
            inputStatus.pointee = .haveData
            return inBuffer
        }
        if let channel = outBuffer.floatChannelData?[0], outBuffer.frameLength > 0 {
            let n = min(Int(outBuffer.frameLength), maxFrames - samples.count)
            samples.append(contentsOf: UnsafeBufferPointer(start: channel, count: n))
        }
        if status == .endOfStream || status == .error || (outBuffer.frameLength == 0 && endOfStream) { break }
    }
    return samples.isEmpty ? nil : samples
}

/// 16 kHz mono PCM16 WAV of samples[range].
func pcm16WAV(_ samples: [Float], _ range: Range<Int>) -> Data {
    let dataSize = range.count * 2
    var data = Data(capacity: 44 + dataSize)
    func u32(_ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { data.append(contentsOf: $0) } }
    func u16(_ v: UInt16) { withUnsafeBytes(of: v.littleEndian) { data.append(contentsOf: $0) } }
    let rate = UInt32(MediaSearchConfig.audioSampleRate)
    data.append(contentsOf: Array("RIFF".utf8)); u32(UInt32(36 + dataSize)); data.append(contentsOf: Array("WAVE".utf8))
    data.append(contentsOf: Array("fmt ".utf8)); u32(16); u16(1); u16(1); u32(rate); u32(rate * 2); u16(2); u16(16)
    data.append(contentsOf: Array("data".utf8)); u32(UInt32(dataSize))
    for i in range {
        let s = Int16(max(-1, min(1, samples[i])) * 32767)
        u16(UInt16(bitPattern: s))
    }
    return data
}

func mediaRMS(_ samples: [Float], _ range: Range<Int>) -> Float {
    guard !range.isEmpty else { return 0 }
    var sum: Float = 0
    for i in range { sum += samples[i] * samples[i] }
    return (sum / Float(range.count)).squareRoot()
}

struct MediaIndexingProgress: Equatable {
    var processed = 0
    var total = 0
    var isComplete: Bool { processed >= total }
    var percent: Int { total == 0 ? 100 : processed * 100 / total }
}
