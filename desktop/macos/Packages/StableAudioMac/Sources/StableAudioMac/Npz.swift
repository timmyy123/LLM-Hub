import Foundation
import MLX

/// Reads NumPy .npz archives as `np.savez` writes them (a zip of uncompressed .npy files, zip64 aware) straight into
/// MLX arrays. mlx-swift loads safetensors and .npy but not .npz, and Stability ships the MLX weights as .npz.
enum Npz {
    struct NpzError: LocalizedError {
        let message: String
        var errorDescription: String? { message }
    }

    /// Every array in the archive, keyed by name without ".npy". `keep` filters names before anything is copied.
    static func load(_ url: URL, keep: ((String) -> Bool)? = nil) throws -> [String: MLXArray] {
        let data = try Data(contentsOf: url, options: .alwaysMapped)
        return try data.withUnsafeBytes { raw -> [String: MLXArray] in
            var arrays: [String: MLXArray] = [:]
            for entry in try entries(raw, file: url.lastPathComponent) {
                guard entry.name.hasSuffix(".npy") else { continue }
                let name = String(entry.name.dropLast(4))
                if let keep, !keep(name) { continue }
                guard entry.compression == 0 else { throw NpzError(message: "\(url.lastPathComponent): \(name) is compressed") }
                let local = entry.offset
                guard local + 30 <= raw.count, u32(raw, local) == 0x0403_4b50 else { throw NpzError(message: "\(url.lastPathComponent): bad local header") }
                let start = local + 30 + u16(raw, local + 26) + u16(raw, local + 28)
                guard start + entry.size <= raw.count else { throw NpzError(message: "\(url.lastPathComponent): truncated") }
                arrays[name] = try npy(UnsafeRawBufferPointer(rebasing: raw[start ..< start + entry.size]), name: name)
            }
            return arrays
        }
    }

    private struct Entry {
        let name: String
        let compression: Int
        let size: Int
        let offset: Int
    }

    private static func u16(_ raw: UnsafeRawBufferPointer, _ at: Int) -> Int {
        Int(UInt16(littleEndian: raw.loadUnaligned(fromByteOffset: at, as: UInt16.self)))
    }

    private static func u32(_ raw: UnsafeRawBufferPointer, _ at: Int) -> Int {
        Int(UInt32(littleEndian: raw.loadUnaligned(fromByteOffset: at, as: UInt32.self)))
    }

    private static func u64(_ raw: UnsafeRawBufferPointer, _ at: Int) -> Int {
        Int(UInt64(littleEndian: raw.loadUnaligned(fromByteOffset: at, as: UInt64.self)))
    }

    /// The central directory, found from the end-of-central-directory record (and its zip64 form for big archives).
    private static func entries(_ raw: UnsafeRawBufferPointer, file: String) throws -> [Entry] {
        var eocd = -1
        var i = raw.count - 22
        while i >= max(0, raw.count - 65_557) {
            if u32(raw, i) == 0x0605_4b50 {
                eocd = i
                break
            }
            i -= 1
        }
        guard eocd >= 0 else { throw NpzError(message: "\(file) is not a zip archive") }
        var count = u16(raw, eocd + 10)
        var directory = u32(raw, eocd + 16)
        if (count == 0xFFFF || directory == 0xFFFF_FFFF), eocd >= 20, u32(raw, eocd - 20) == 0x0706_4b50 {
            let record = u64(raw, eocd - 20 + 8)
            guard record + 56 <= raw.count, u32(raw, record) == 0x0606_4b50 else { throw NpzError(message: "\(file): bad zip64 record") }
            count = u64(raw, record + 32)
            directory = u64(raw, record + 48)
        }
        var list: [Entry] = []
        var at = directory
        for _ in 0 ..< count {
            guard at + 46 <= raw.count, u32(raw, at) == 0x0201_4b50 else { throw NpzError(message: "\(file): bad central directory") }
            let compression = u16(raw, at + 10)
            var size = u32(raw, at + 24)
            let nameLength = u16(raw, at + 28), extraLength = u16(raw, at + 30), commentLength = u16(raw, at + 32)
            var offset = u32(raw, at + 42)
            let name = String(decoding: UnsafeRawBufferPointer(rebasing: raw[at + 46 ..< at + 46 + nameLength]), as: UTF8.self)
            // Zip64 extra field: the 64-bit values for whichever of the sizes and the offset overflowed, in that order.
            var extra = at + 46 + nameLength
            let extraEnd = extra + extraLength
            while extra + 4 <= extraEnd {
                let id = u16(raw, extra), length = u16(raw, extra + 2)
                if id == 0x0001 {
                    var field = extra + 4
                    if size == 0xFFFF_FFFF { size = u64(raw, field); field += 8 }
                    if u32(raw, at + 20) == 0xFFFF_FFFF { field += 8 }
                    if offset == 0xFFFF_FFFF { offset = u64(raw, field) }
                }
                extra += 4 + length
            }
            list.append(Entry(name: name, compression: compression, size: size, offset: offset))
            at = extraEnd + commentLength
        }
        return list
    }

    /// One .npy file: the header dict, then little-endian C-order data.
    private static func npy(_ raw: UnsafeRawBufferPointer, name: String) throws -> MLXArray {
        guard raw.count >= 10, raw[0] == 0x93, raw[1] == UInt8(ascii: "N") else { throw NpzError(message: "\(name) is not .npy") }
        let major = raw[6]
        let headerLength = major == 1 ? u16(raw, 8) : u32(raw, 8)
        let headerStart = major == 1 ? 10 : 12
        let header = String(decoding: UnsafeRawBufferPointer(rebasing: raw[headerStart ..< headerStart + headerLength]), as: UTF8.self)
        guard !header.contains("'fortran_order': True") else { throw NpzError(message: "\(name) is Fortran ordered") }
        func field(_ key: String, _ open: Character, _ close: Character) -> String? {
            guard let range = header.range(of: "'\(key)':") else { return nil }
            let rest = header[range.upperBound...]
            guard let a = rest.firstIndex(of: open), let b = rest[rest.index(after: a)...].firstIndex(of: close) else { return nil }
            return String(rest[rest.index(after: a) ..< b])
        }
        guard let descr = field("descr", "'", "'"), let shapeText = field("shape", "(", ")") else { throw NpzError(message: "\(name): bad header") }
        let shape = shapeText.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
        let body = UnsafeRawBufferPointer(rebasing: raw[(headerStart + headerLength)...])
        let count = shape.reduce(1, *)
        func make<T: HasDType>(_ type: T.Type) throws -> MLXArray {
            guard body.count >= count * MemoryLayout<T>.size else { throw NpzError(message: "\(name): truncated data") }
            return MLXArray(UnsafeRawBufferPointer(rebasing: body[..<(count * MemoryLayout<T>.size)]), shape, type: type)
        }
        switch descr {
        case "<f2": return try make(Float16.self)
        case "<f4": return try make(Float.self)
        case "|u1": return try make(UInt8.self)
        case "<i4": return try make(Int32.self)
        case "<i8": return try make(Int64.self)
        default: throw NpzError(message: "\(name): unsupported dtype \(descr)")
        }
    }
}
