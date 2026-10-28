#if DEBUG
import CryptoKit
import Foundation
import LlamaCppRuntime
import ModelZoo
import SQLite3
import UIKit

/// Device checks that download real models and run them on this device's GPU. ImageExtrasDeviceTests runs them
/// through XCTest. Launching a Debug build with `-LLMHubSelfTest encoders,llama` (or `lfm3,lfm3-vl`) runs them without XCTest, e.g. on a
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
                    case "lfm3": failures += try await checkLfm3Text()
                    case "lfm3-vl": failures += try await checkLfm3Vision()
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

    // MARK: - LFM3 (llama.cpp b11946)

    private static let lfm3TextURL = "https://huggingface.co/LiquidAI/LFM3-1.7B-GGUF/resolve/b3f2ae5d73cd3c91bf40e8ba752c648aa05dffdf/LFM3-1.7B-QAD-Q4_0.gguf"
    private static let lfm3TextSHA256 = "43a7d09a0e9cba91b63eb7052488b9564569bddfd65754d2e426465e572a7452"
    private static let lfm3VisionURL = "https://huggingface.co/LiquidAI/LFM3-VL-2B-GGUF/resolve/7c3b4a43c9023e80bf234fec5e826a3365c8dda8/LFM3-VL-2B-QAD-Q4_0.gguf"
    private static let lfm3VisionSHA256 = "621c455376cb96a20891697deb9975c6fb47dd5ea98ed3ad1f3b7e8f24f0b1e6"
    private static let lfm3ProjectorURL = "https://huggingface.co/LiquidAI/LFM3-VL-2B-GGUF/resolve/7c3b4a43c9023e80bf234fec5e826a3365c8dda8/mmproj-LFM3-VL-2B-Q8_0.gguf"
    private static let lfm3ProjectorSHA256 = "c5c5b58d544a18da919172d2d0368090ea1a4a351b6feb7f8a326a264a5918ab"
    /// llama.cpp's own vision test photo: the New York Times front page of July 21, 1969 ("MEN WALK ON MOON").
    private static let newspaperURL = "https://raw.githubusercontent.com/ggml-org/llama.cpp/8ca4060b7e30e76baaa5670ee1ec1b760b7f8552/tools/mtmd/test-1.jpeg"
    private static let newspaperSHA256 = "2dff664c0c8aaea18aff8cbe7e868845b775e90cdd7a0bac98df709b131deaa3"

    /// Downloads LFM3 1.7B Q4_0 (pinned, SHA-256 checked, about 1.05 GB), loads every layer on the GPU, and checks
    /// answers through the app's LFM3 prompt format: a direct answer, a chat history in the app's ChatML shape, and an
    /// answer after `<think>` reasoning. Returns failures.
    static func checkLfm3Text() async throws -> [String] {
        let model = try await downloadPinned(lfm3TextURL, sha256: lfm3TextSHA256)
        defer { try? FileManager.default.removeItem(at: model) }
        let backend = DirectLlamaCppBackend.shared
        let started = Date()
        try await backend.load(path: model.path, projector: nil, contextSize: 4096, gpuLayers: 99)
        defer { Task { await backend.unload() } }

        var failures = [String]()
        let question = "What is the capital of France? Answer in one word."
        let direct = try await lfm3Reply(
            LLMBackend.buildLfm3Prompt(prompt: question, systemPrompt: nil, thinkingEnabled: false), maxTokens: 32)
        log("LFM3_TEST: direct reply=\(direct.prefix(200))")
        if !direct.localizedCaseInsensitiveContains("Paris") { failures.append("LFM3 direct answer: \(direct)") }

        // The shape ChatScreen builds for every "lfm" model, normalized by LLMBackend for LFM3.
        let chat = "<|startoftext|>\n<|im_start|>user\nMy name is Ada.<|im_end|>\n<|im_start|>assistant\nNice to meet you, Ada!<|im_end|>\n<|im_start|>user\nWhat is my name? Answer in one word.<|im_end|>\n<|im_start|>assistant\n"
        let remembered = try await lfm3Reply(
            LLMBackend.buildLfm3Prompt(prompt: chat, systemPrompt: nil, thinkingEnabled: false), maxTokens: 32)
        log("LFM3_TEST: history reply=\(remembered.prefix(200))")
        if !remembered.localizedCaseInsensitiveContains("Ada") { failures.append("LFM3 history answer: \(remembered)") }

        let reasoned = try await lfm3Reply(
            LLMBackend.buildLfm3Prompt(prompt: question, systemPrompt: nil, thinkingEnabled: true), maxTokens: 2048)
        let answer = reasoned.components(separatedBy: "</think>").dropFirst().joined(separator: "</think>")
        log("LFM3_TEST: reasoning chars=\(reasoned.count) answer=\(answer.prefix(200)) seconds=\(Int(Date().timeIntervalSince(started)))")
        if !reasoned.contains("</think>") {
            failures.append("LFM3 reasoning did not reach </think> in 2048 tokens: \(reasoned.suffix(200))")
        } else if !answer.localizedCaseInsensitiveContains("Paris") {
            failures.append("LFM3 answer after reasoning: \(answer)")
        }
        return failures
    }

    /// Downloads LFM3-VL 2B Q4_0 and its Q8_0 projector (pinned, SHA-256 checked, about 1.25 GB), loads every layer on
    /// the GPU, and asks for the color of a drawn red circle and the headline of a photographed newspaper, with the
    /// image inside the user turn. Returns failures.
    static func checkLfm3Vision() async throws -> [String] {
        let model = try await downloadPinned(lfm3VisionURL, sha256: lfm3VisionSHA256)
        defer { try? FileManager.default.removeItem(at: model) }
        let projector = try await downloadPinned(lfm3ProjectorURL, sha256: lfm3ProjectorSHA256)
        defer { try? FileManager.default.removeItem(at: projector) }
        let backend = DirectLlamaCppBackend.shared
        let started = Date()
        try await backend.load(path: model.path, projector: projector.path, contextSize: 4096, gpuLayers: 99)
        defer { Task { await backend.unload() } }

        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        let image = UIGraphicsImageRenderer(size: CGSize(width: 448, height: 448), format: format).image { context in
            UIColor.white.setFill()
            context.fill(CGRect(x: 0, y: 0, width: 448, height: 448))
            UIColor.red.setFill()
            context.cgContext.fillEllipse(in: CGRect(x: 74, y: 74, width: 300, height: 300))
        }
        guard let jpeg = image.jpegData(compressionQuality: 0.95) else { return ["Could not encode the test image"] }
        let prompt = LLMBackend.insertingLfm3ImageMarker(
            DirectLlamaCppBackend.mediaMarker,
            into: LLMBackend.buildLfm3Prompt(
                prompt: "What color is the circle in this image? Answer in one word.",
                systemPrompt: nil, thinkingEnabled: false))
        let reply = try await lfm3Reply(prompt, imageData: jpeg, maxTokens: 32)
        log("LFM3_VL_TEST: reply=\(reply.prefix(200)) seconds=\(Int(Date().timeIntervalSince(started)))")
        var failures = reply.localizedCaseInsensitiveContains("red") ? [String]() : ["LFM3-VL answer: \(reply)"]

        let photo = try await downloadPinned(newspaperURL, sha256: newspaperSHA256)
        defer { try? FileManager.default.removeItem(at: photo) }
        let headlinePrompt = LLMBackend.insertingLfm3ImageMarker(
            DirectLlamaCppBackend.mediaMarker,
            into: LLMBackend.buildLfm3Prompt(
                prompt: "What is the main headline of this newspaper? Reply with the headline only.",
                systemPrompt: nil, thinkingEnabled: false))
        let headline = try await lfm3Reply(headlinePrompt, imageData: try Data(contentsOf: photo), maxTokens: 48)
        log("LFM3_VL_TEST: headline=\(headline.prefix(200)) seconds=\(Int(Date().timeIntervalSince(started)))")
        if !headline.localizedCaseInsensitiveContains("moon") { failures.append("LFM3-VL headline: \(headline)") }
        return failures
    }

    private static func lfm3Reply(_ prompt: String, imageData: Data? = nil, maxTokens: Int) async throws -> String {
        var reply = ""
        let stream = await DirectLlamaCppBackend.shared.stream(
            prompt: prompt, imageData: imageData, maxTokens: maxTokens,
            temperature: 0, topK: 1, topP: 1, stopSequences: [])
        for try await update in stream { reply = update.text }
        return reply
    }

    private struct ChecksumMismatch: LocalizedError {
        let file: String
        let digest: String
        var errorDescription: String? { "\(file) SHA-256 mismatch: \(digest)" }
    }

    /// Downloads a pinned file to Documents/llmhub-test-<name> and checks its SHA-256. The caller deletes it.
    private static func downloadPinned(_ urlString: String, sha256 expected: String) async throws -> URL {
        let url = URL(string: urlString)!
        let file = documents.appendingPathComponent("llmhub-test-" + url.lastPathComponent)
        try? FileManager.default.removeItem(at: file)
        let (temporary, _) = try await URLSession.shared.download(from: url)
        try FileManager.default.moveItem(at: temporary, to: file)
        let digest = try sha256(file)
        guard digest == expected else {
            try? FileManager.default.removeItem(at: file)
            throw ChecksumMismatch(file: url.lastPathComponent, digest: digest)
        }
        return file
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
