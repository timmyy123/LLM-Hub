#if DEBUG
import CryptoKit
import Foundation
import LlamaCppRuntime
import ModelZoo
import SQLite3
import UIKit

/// Device checks that download real models and run them on this device's GPU. ImageExtrasDeviceTests runs them
/// through XCTest. Launching a Debug build with `-LLMHubSelfTest encoders,llama` runs them without XCTest, e.g. on a
/// Mac as "Designed for iPad", where hosted test bundles do not bootstrap. Results go to stdout and
/// Documents/llmhub-self-test.log, ending with `SELF_TEST PASS` or `SELF_TEST FAIL`.
@MainActor
enum DeviceSelfTests {
    private static var documents: URL { FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0] }

    static func runFromLaunchArguments() {
        let arguments = ProcessInfo.processInfo.arguments
        guard let index = arguments.firstIndex(of: "-LLMHubSelfTest"), index + 1 < arguments.count else { return }
        let checks = arguments[index + 1].split(separator: ",").map(String.init)
        let logFile = documents.appendingPathComponent("llmhub-self-test.log")
        try? FileManager.default.removeItem(at: logFile)
        Task { @MainActor in
            var failures = [String]()
            for check in checks {
                do {
                    switch check {
                    case "encoders": failures += try await checkImportedClipGAndT5(download: true)
                    case "llama": failures += try await checkGgufOnMetal()
                    default: failures.append("Unknown self-test \(check)")
                    }
                } catch {
                    failures.append("\(check) threw \(error)")
                }
            }
            failures.forEach { log("FAILURE: \($0)") }
            log(failures.isEmpty ? "SELF_TEST PASS" : "SELF_TEST FAIL")
            exit(failures.isEmpty ? 0 : 1)
        }
    }

    static func log(_ line: String) {
        print(line)
        NSLog("%@", line)
        let file = documents.appendingPathComponent("llmhub-self-test.log")
        let data = Data((line + "\n").utf8)
        if let handle = try? FileHandle(forWritingTo: file) {
            handle.seekToEndOfFile(); handle.write(data); try? handle.close()
        } else {
            try? data.write(to: file)
        }
    }

    // MARK: - Imported CLIP-G and T5

    /// SDXL (CLIP-G) and SD3.5 (CLIP-G and T5) through the import path: importing the model's own encoder file keeps
    /// the output, and an altered copy changes it. Returns failures; models that are not downloaded fail unless
    /// [download] is set, which downloads SSD-1B and SD3.5 Medium (about 11.5 GB).
    static func checkImportedClipGAndT5(download: Bool) async throws -> [String] {
        var failures = [String]()
        let models = ModelZoo.persistentModelsDirectory()
        let store = ImageExtrasStore.shared
        let backend = StableDiffusionBackend.shared
        let original = store.options
        defer {
            backend.unloadModel()
            for asset in store.options.assets { store.remove(asset.id) }
            store.options = original
        }
        let cases: [(String, [ImageExtra.Kind])] = [("ssd_1b_q6p_q8p.ckpt", [.clipG]), ("sd3_medium_3.5_q8p.ckpt", [.clipG, .textEncoder])]
        for (modelID, kinds) in cases {
            guard let model = ModelData.allModels().first(where: { $0.id == modelID }),
                  let specification = ModelZoo.specificationForModel(modelID) else {
                failures.append("No model \(modelID)"); continue
            }
            if !StableDiffusionBackend.isModelDownloaded(modelId: modelID) {
                guard download else { failures.append("\(model.name) is not downloaded"); continue }
                let started = Date()
                try await ModelDownloader.shared.downloadModel(model, hfToken: nil, destinationDir: models) { _ in }
                log("ENCODER_TEST: downloaded \(model.name) in \(Int(Date().timeIntervalSince(started))) s")
            }
            backend.unloadModel()
            for asset in store.options.assets { store.remove(asset.id) }
            store.options = ImageGenerationOptions()
            try await backend.loadModel(model)
            let baseline = try await render("\(model.name) baseline")

            for kind in kinds {
                // The encoder file this model downloaded (CLIP-G is OpenCLIP bigG; T5 is T5-XXL), at Draw Things' own path.
                let downloaded = ModelZoo.filesToDownload(specification).map(\.file)
                guard let file = downloaded.first(where: { kind == .textEncoder ? $0.hasPrefix("t5_") : $0.contains("bigg14") }) else {
                    failures.append("No \(kind.rawValue) among \(downloaded)"); continue
                }
                let source = URL(fileURLWithPath: ModelZoo.filePathForModelDownloaded(file))
                guard FileManager.default.fileExists(atPath: source.path) else {
                    let listing = (try? FileManager.default.contentsOfDirectory(atPath: models.path)) ?? []
                    failures.append("\(file) is not at \(source.path); Models contains \(listing.sorted())"); continue
                }
                let altered = documents.appendingPathComponent("llmhub-test-altered-\(kind.rawValue).ckpt")
                defer { try? FileManager.default.removeItem(at: altered) }
                try? FileManager.default.removeItem(at: altered)
                try FileManager.default.copyItem(at: source, to: altered)
                try alterEncoder(altered, kind: kind)

                // The identical encoder is the model's own file, imported as a user would pick it.
                try await store.importFile(source, kind: kind, modelID: modelID)
                let same = try await render("\(model.name) identical \(kind.rawValue)")
                store.options.assets[store.options.assets.count - 1].enabled = false
                try await store.importFile(altered, kind: kind, modelID: modelID)
                let changed = try await render("\(model.name) altered \(kind.rawValue)")
                for asset in store.options.assets where asset.kind == kind { store.remove(asset.id) }

                let sameDistance = try pixelDistance(same, baseline)
                let changedDistance = try pixelDistance(changed, baseline)
                log("ENCODER_TEST: \(model.name) \(kind.rawValue) identical=\(sameDistance) altered=\(changedDistance)")
                if changedDistance <= 0.01 { failures.append("An altered \(kind.rawValue) must change \(model.name) (distance \(changedDistance))") }
                if sameDistance >= changedDistance * 0.25 {
                    failures.append("An identical \(kind.rawValue) must keep \(model.name)'s output (identical \(sameDistance), altered \(changedDistance))")
                }
            }
            backend.unloadModel()
        }
        return failures
    }

    private static func render(_ label: String) async throws -> Data {
        let started = Date()
        let image = try await StableDiffusionBackend.shared.generateImage(prompt: "a red ceramic teapot on a wooden table", steps: 2, seed: 12345)
        guard let png = image?.pngData() else { throw CocoaError(.fileReadCorruptFile) }
        try? png.write(to: documents.appendingPathComponent("llmhub-self-test-\(label.replacingOccurrences(of: " ", with: "_")).png"))
        log("ENCODER_TEST: \(label) rendered in \(Int(Date().timeIntervalSince(started))) s, bytes=\(png.count)")
        return png
    }

    /// Zeroes CLIP-G's text projection (as the Android SDXL test does), or sets T5's final RMSNorm scale to 1.
    static func alterEncoder(_ file: URL, kind: ImageExtra.Kind) throws {
        var database: OpaquePointer?
        guard sqlite3_open(file.path, &database) == SQLITE_OK else { throw CocoaError(.fileReadCorruptFile) }
        defer { sqlite3_close(database) }
        let sql: String
        if kind == .textEncoder {
            // The quantized checkpoint compresses this vector; an uncompressed f16 row (type 1) of the same shape replaces it.
            let ones = String(repeating: "003C", count: 4096) // f16 1.0, little-endian
            sql = "UPDATE tensors SET type = 1, data = X'\(ones)' WHERE name = '__text_model__[t-final_norm-0-0]'"
        } else {
            sql = "UPDATE tensors SET data = zeroblob(length(data)) WHERE name = 'text_projection'"
        }
        guard sqlite3_exec(database, sql, nil, nil, nil) == SQLITE_OK, sqlite3_changes(database) == 1 else {
            throw CocoaError(.fileWriteUnknown)
        }
    }

    /// Mean absolute RGB difference of two PNGs, from 0 to 1.
    static func pixelDistance(_ a: Data, _ b: Data) throws -> Double {
        func pixels(_ data: Data) throws -> [UInt8] {
            guard let image = UIImage(data: data)?.cgImage else { throw CocoaError(.fileReadCorruptFile) }
            var buffer = [UInt8](repeating: 0, count: image.width * image.height * 4)
            guard let context = CGContext(data: &buffer, width: image.width, height: image.height, bitsPerComponent: 8,
                                          bytesPerRow: image.width * 4, space: CGColorSpaceCreateDeviceRGB(),
                                          bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { throw CocoaError(.fileReadCorruptFile) }
            context.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
            return buffer
        }
        let (x, y) = (try pixels(a), try pixels(b))
        guard x.count == y.count else { return 1 }
        var total = 0
        for i in stride(from: 0, to: x.count, by: 4) { for c in 0..<3 { total += abs(Int(x[i + c]) - Int(y[i + c])) } }
        return Double(total) / Double(x.count / 4 * 3 * 255)
    }

    // MARK: - llama.cpp on Metal

    /// Downloads LFM-2.5 1.2B Q4_0 (pinned, SHA-256 checked), loads every layer on the GPU, and asks a one-word question
    /// through the model's chat template. Returns failures.
    static func checkGgufOnMetal() async throws -> [String] {
        let file = documents.appendingPathComponent("llmhub-test-LFM2.5-1.2B-Instruct-Q4_0.gguf")
        defer { try? FileManager.default.removeItem(at: file) }
        let url = URL(string: "https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF/resolve/8ed288026e23958ad9dfa92d53ed773a8eee7125/LFM2.5-1.2B-Instruct-Q4_0.gguf")!
        let (temporary, _) = try await URLSession.shared.download(from: url)
        try? FileManager.default.removeItem(at: file)
        try FileManager.default.moveItem(at: temporary, to: file)
        let digest = try sha256(file)
        guard digest == "2ea801949d760cdf1a2cc04a54262c22c3c0c54f0769d57760c9adeb0e59233f" else { return ["GGUF SHA-256 mismatch: \(digest)"] }

        let backend = DirectLlamaCppBackend.shared
        let started = Date()
        try await backend.load(path: file.path, projector: nil, contextSize: 2048, gpuLayers: 99)
        defer { Task { await backend.unload() } }
        guard let prompt = await backend.applyChatTemplate(systemPrompt: nil, userPrompt: "What is the capital of France? Answer in one word.") else {
            return ["No chat template"]
        }
        var reply = ""
        for try await update in await backend.stream(prompt: prompt, maxTokens: 32, temperature: 0, topK: 1, topP: 1, stopSequences: []) {
            reply = update.text
        }
        log("LLAMA_TEST: reply=\(reply.prefix(200)) seconds=\(Int(Date().timeIntervalSince(started)))")
        return reply.localizedCaseInsensitiveContains("Paris") ? [] : ["Unexpected answer: \(reply)"]
    }

    private static func sha256(_ file: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: file)
        defer { try? handle.close() }
        var hasher = SHA256()
        while let chunk = try handle.read(upToCount: 1 << 20), !chunk.isEmpty { hasher.update(data: chunk) }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }
}
#endif
