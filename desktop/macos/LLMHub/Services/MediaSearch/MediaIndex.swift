import Foundation

/// One embedded item: a photo, or an audio moment covering startMs..endMs of a file.
struct MediaVector: Sendable {
    let id: String
    let startMs: Int32
    let endMs: Int32
    let vector: [Float]
}

/// Vectors for one search feature, in the Android app's on-disk format ("LHM1": count, then id, start, end and the
/// floats of each record, big-endian), plus a `.files` list of each file's stamp (modified time and size), so changed
/// files are embedded again. A port of core/LLMHub.Core/Services/MediaSearch/MediaIndex.cs.
final class MediaIndex {
    private static let magic: UInt32 = 0x4C48_4D31
    private let file: URL
    private var records: [String: MediaVector] = [:]
    private var stamps: [String: Int64] = [:]
    /// Each file's record keys, so replacing a file's vectors does not scan the whole index.
    private var keysByFile: [String: [String]] = [:]
    private var dirty = false
    /// The model the vectors came from; vectors from another model are not comparable.
    private(set) var modelId = ""

    init(_ file: URL) { self.file = file }

    var all: Dictionary<String, MediaVector>.Values { records.values }
    var fileCount: Int { stamps.count }
    var files: [String] { Array(stamps.keys) }

    func load() {
        records = [:]
        stamps = [:]
        keysByFile = [:]
        if let data = try? Data(contentsOf: file) {
            var reader = BigEndianReader(data)
            if reader.u32() == Self.magic, let count = reader.u32() {
                for _ in 0 ..< count {
                    guard let id = reader.utf(), let start = reader.u32(), let end = reader.u32(), let length = reader.u32(),
                          let vector = reader.floats(Int(length)) else {
                        Log.error("Media index \(file.path) is truncated; keeping the \(records.count) records read")
                        break
                    }
                    let record = MediaVector(id: id, startMs: Int32(bitPattern: start), endMs: Int32(bitPattern: end), vector: vector)
                    let key = Self.key(id, record.startMs)
                    records[key] = record
                    keysByFile[id, default: []].append(key)
                }
            }
        }
        if let text = try? String(contentsOf: stampsFile, encoding: .utf8) {
            let lines = text.split(separator: "\n", omittingEmptySubsequences: false).map(String.init)
            modelId = lines.first ?? ""
            for line in lines.dropFirst() {
                guard let tab = line.firstIndex(of: "\t"), let stamp = Int64(line[..<tab]) else { continue }
                stamps[String(line[line.index(after: tab)...])] = stamp
            }
        }
    }

    func isCurrent(_ path: String, stamp: Int64) -> Bool { stamps[path] == stamp }

    /// Replaces a file's vectors (one for a photo, one per moment for audio).
    func put(_ path: String, stamp: Int64, vectors: [MediaVector]) {
        remove(path)
        for v in vectors {
            let key = Self.key(v.id, v.startMs)
            records[key] = v
            keysByFile[v.id, default: []].append(key)
        }
        stamps[path] = stamp
        dirty = true
    }

    func remove(_ path: String) {
        for key in keysByFile.removeValue(forKey: path) ?? [] { records[key] = nil }
        if stamps.removeValue(forKey: path) != nil { dirty = true }
    }

    func reset(_ modelId: String) {
        records = [:]
        stamps = [:]
        keysByFile = [:]
        self.modelId = modelId
        dirty = true
    }

    func saveIfDirty() {
        guard dirty else { return }
        do {
            try FileManager.default.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
            var out = Data()
            func u32(_ v: UInt32) { withUnsafeBytes(of: v.bigEndian) { out.append(contentsOf: $0) } }
            u32(Self.magic)
            u32(UInt32(records.count))
            for r in records.values {
                // Java's DataOutputStream.writeUTF: a 2-byte big-endian length, then UTF-8.
                let bytes = Array(r.id.utf8)
                out.append(UInt8((bytes.count >> 8) & 0xFF))
                out.append(UInt8(bytes.count & 0xFF))
                out.append(contentsOf: bytes)
                u32(UInt32(bitPattern: r.startMs))
                u32(UInt32(bitPattern: r.endMs))
                u32(UInt32(r.vector.count))
                for f in r.vector { u32(f.bitPattern) }
            }
            try out.write(to: file, options: .atomic)
            let lines = [modelId] + stamps.map { "\($0.value)\t\($0.key)" }
            try (lines.joined(separator: "\n") + "\n").write(to: stampsFile, atomically: true, encoding: .utf8)
            dirty = false
        } catch {
            Log.error("Saving media index \(file.path) failed", error)
        }
    }

    /// Vectors are L2-normalized, so the dot product is the cosine similarity.
    static func dot(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count else { return -1 }
        var sum: Float = 0
        for i in a.indices { sum += a[i] * b[i] }
        return sum
    }

    private var stampsFile: URL { URL(fileURLWithPath: file.path + ".files") }
    private static func key(_ id: String, _ start: Int32) -> String { "\(id)#\(start)" }
}

private struct BigEndianReader {
    private let data: Data
    private var offset: Int

    init(_ data: Data) {
        self.data = data
        offset = data.startIndex
    }

    mutating func u32() -> UInt32? {
        guard offset + 4 <= data.endIndex else { return nil }
        let value = data[offset ..< offset + 4].reduce(UInt32(0)) { $0 << 8 | UInt32($1) }
        offset += 4
        return value
    }

    mutating func utf() -> String? {
        guard offset + 2 <= data.endIndex else { return nil }
        let length = Int(data[offset]) << 8 | Int(data[offset + 1])
        offset += 2
        guard offset + length <= data.endIndex else { return nil }
        defer { offset += length }
        return String(decoding: data[offset ..< offset + length], as: UTF8.self)
    }

    mutating func floats(_ count: Int) -> [Float]? {
        guard count >= 0, offset + count * 4 <= data.endIndex else { return nil }
        var values = [Float](repeating: 0, count: count)
        data.withUnsafeBytes { raw in
            let base = raw.baseAddress!.advanced(by: offset - data.startIndex)
            for i in 0 ..< count {
                values[i] = Float(bitPattern: UInt32(bigEndian: base.loadUnaligned(fromByteOffset: i * 4, as: UInt32.self)))
            }
        }
        offset += count * 4
        return values
    }
}
