import Foundation

/// Manages on-device Kokoro asset files. Downloads from the canonical
/// onnx-community Hugging Face mirror on first use and caches under
/// `Documents/kokoro/`.
///
/// Files (one-time download):
///   - model_fp16.onnx           (≈ 165 MB)
///   - voices/<voice>.bin        (≈ 512 KB each)
enum KokoroAssets {

    struct Progress: Sendable {
        let stage: String
        let bytesDone: Int64
        let bytesTotal: Int64   // -1 if unknown
    }

    private static let modelURL = URL(string:
        "https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/onnx/model_fp16.onnx"
    )!

    private static func voiceURL(_ id: String) -> URL {
        URL(string:
            "https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/voices/\(id).bin"
        )!
    }

    static func root() throws -> URL {
        let docs = try FileManager.default.url(
            for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: true
        )
        let dir = docs.appendingPathComponent("kokoro", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    static func modelFile() throws -> URL {
        try root().appendingPathComponent("model_fp16.onnx")
    }

    static func voiceFile(_ id: String) throws -> URL {
        let voicesDir = try root().appendingPathComponent("voices", isDirectory: true)
        try FileManager.default.createDirectory(at: voicesDir, withIntermediateDirectories: true)
        return voicesDir.appendingPathComponent("\(id).bin")
    }

    static func isReady(voice id: String) -> Bool {
        guard let model = try? modelFile(), let voice = try? voiceFile(id) else { return false }
        return FileManager.default.fileExists(atPath: model.path) &&
               FileManager.default.fileExists(atPath: voice.path)
    }

    /// Download model + voice if missing. Yields progress + a final "Ready" event.
    static func ensure(voice id: String = "af_heart") -> AsyncThrowingStream<Progress, Error> {
        AsyncThrowingStream { continuation in
            Task {
                do {
                    let model = try modelFile()
                    if !FileManager.default.fileExists(atPath: model.path) {
                        continuation.yield(Progress(stage: "Downloading Kokoro model", bytesDone: 0, bytesTotal: -1))
                        try await download(from: modelURL, to: model) { done, total in
                            continuation.yield(Progress(stage: "Downloading Kokoro model", bytesDone: done, bytesTotal: total))
                        }
                    }
                    let voice = try voiceFile(id)
                    if !FileManager.default.fileExists(atPath: voice.path) {
                        continuation.yield(Progress(stage: "Downloading voice \(id)", bytesDone: 0, bytesTotal: -1))
                        try await download(from: voiceURL(id), to: voice) { done, total in
                            continuation.yield(Progress(stage: "Downloading voice \(id)", bytesDone: done, bytesTotal: total))
                        }
                    }
                    continuation.yield(Progress(stage: "Ready", bytesDone: 1, bytesTotal: 1))
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    private static func download(
        from url: URL,
        to target: URL,
        onProgress: @escaping (_ done: Int64, _ total: Int64) -> Void
    ) async throws {
        var request = URLRequest(url: url)
        request.timeoutInterval = 60 * 30  // 30 min for big model
        let (asyncBytes, response) = try await URLSession.shared.bytes(for: request)
        guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            throw NSError(domain: "KokoroAssets", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "HTTP error for \(url)"])
        }
        let total = response.expectedContentLength
        let tmp = target.appendingPathExtension("part")
        FileManager.default.createFile(atPath: tmp.path, contents: nil)
        let handle = try FileHandle(forWritingTo: tmp)
        defer { try? handle.close() }

        var done: Int64 = 0
        var lastEmit: Int64 = 0
        var buf = Data()
        buf.reserveCapacity(64 * 1024)

        for try await byte in asyncBytes {
            buf.append(byte)
            if buf.count >= 64 * 1024 {
                try handle.write(contentsOf: buf)
                done += Int64(buf.count)
                buf.removeAll(keepingCapacity: true)
                if done - lastEmit >= 1_000_000 {
                    onProgress(done, total)
                    lastEmit = done
                }
            }
        }
        if !buf.isEmpty {
            try handle.write(contentsOf: buf)
            done += Int64(buf.count)
        }
        try handle.close()

        // Atomically rename to the final path.
        if FileManager.default.fileExists(atPath: target.path) {
            try FileManager.default.removeItem(at: target)
        }
        try FileManager.default.moveItem(at: tmp, to: target)
        onProgress(done, total)
    }
}
