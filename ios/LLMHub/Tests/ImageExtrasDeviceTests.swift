import XCTest
import UIKit
import ModelZoo
@testable import LLMHub

/// Runs inside the real app. Weight tests require the explicitly staged fixtures in Documents.
@MainActor
final class ImageExtrasDeviceTests: XCTestCase {
    override func setUp() async throws { UIApplication.shared.isIdleTimerDisabled = true }
    override func tearDown() async throws { UIApplication.shared.isIdleTimerDisabled = false }

    private var documents: URL { FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0] }

    func testPromptResourcesPersistenceAndValidation() async throws {
        let store = ImageExtrasStore.shared
        let original = store.options
        let file = documents.appendingPathComponent("modifier-test-resource.txt")
        defer { store.options = original; try? FileManager.default.removeItem(at: file) }
        store.options = ImageGenerationOptions()
        try "cat=dog\ndog=wolf\n猫=cat\nred cat=$1\\cat".write(to: file, atomically: true, encoding: .utf8)
        try await store.importFile(file, kind: .translations, modelID: nil)
        XCTAssertEqual(store.options.translatedPrompt("cat catfish cat_2 猫咪 猫, dog; red cat"), "dog catfish cat_2 猫咪 cat, wolf; $1\\cat")
        store.options.negativePrompt = "cat"
        try "cinematic\nsoft light".write(to: file, atomically: true, encoding: .utf8)
        try await store.importFile(file, kind: .tags, modelID: nil)
        XCTAssertEqual(store.options.assets.last?.tags, ["cinematic", "soft light"])
        try "{\"portrait\":\"a detailed portrait, soft lighting\"}".write(to: file, atomically: true, encoding: .utf8)
        try await store.importFile(file, kind: .definitions, modelID: nil)
        XCTAssertEqual(store.options.assets.last?.mappings["portrait"], "a detailed portrait, soft lighting")
        store.options.guidance = 9
        store.options.clipSkip = 2
        let data = try XCTUnwrap(UserDefaults.standard.data(forKey: "image_generation_extras_v1"))
        let persisted = try JSONDecoder().decode(ImageGenerationOptions.self, from: data)
        XCTAssertEqual(persisted.negativePrompt, "cat")
        XCTAssertEqual(persisted.guidance, 9)
        XCTAssertEqual(persisted.clipSkip, 2)
        store.options.assets[0].enabled = false
        XCTAssertEqual(store.options.translatedPrompt("cat"), "cat")
        store.remove(store.options.assets[0].id)
        XCTAssertEqual(store.options.assets.count, 2)
        for invalid in ["missing separator", String(repeating: "x", count: 2_000_001)] {
            try invalid.write(to: file, atomically: true, encoding: .utf8)
            do { try await store.importFile(file, kind: .translations, modelID: nil); XCTFail("Invalid resource accepted") }
            catch { /* expected */ }
        }
    }

    func testGenerationWithImportedModifiers() async throws {
        #if targetEnvironment(simulator)
        throw XCTSkip("Run generation coverage on a physical iPhone")
        #endif
        let fm = FileManager.default
        let lora = documents.appendingPathComponent("llmhub-test-lora.ckpt")
        let embedding = documents.appendingPathComponent("llmhub-test-embedding.safetensors")
        guard fm.fileExists(atPath: lora.path), fm.fileExists(atPath: embedding.path) else {
            throw XCTSkip("Stage the compatible LoRA and embedding test fixtures in Documents first")
        }
        let model = try XCTUnwrap(ModelData.allModels().first { $0.isDrawThingsImageGeneration && StableDiffusionBackend.isModelDownloaded(modelId: $0.id) })
        let store = ImageExtrasStore.shared
        let original = store.options
        let backend = StableDiffusionBackend.shared
        backend.unloadModel()
        store.options = ImageGenerationOptions()
        defer {
            backend.unloadModel()
            for asset in store.options.assets { store.remove(asset.id) }
            store.options = original
        }
        func render(_ label: String, input: UIImage? = nil, strength: Float = 0.7) async throws -> Data {
            let image = try await backend.generateImage(prompt: "a red ceramic teapot on a wooden table", steps: 2, seed: 12345, inputImage: input, denoiseStrength: strength)
            let result = try XCTUnwrap(image, label)
            XCTAssertEqual(result.size.width, CGFloat(model.imageGenerationResolution ?? 512), label)
            let png = try XCTUnwrap(result.pngData(), label)
            XCTAssertGreaterThan(png.count, 1000, label)
            let attachment = XCTAttachment(data: png, uniformTypeIdentifier: "public.png")
            attachment.name = label; attachment.lifetime = .keepAlways; add(attachment)
            print("MODIFIER_GENERATION_PASS: \(label), bytes=\(png.count)")
            return png
        }
        try await backend.loadModel(model)
        let baseline = try await render("baseline")
        store.options.guidance = 1
        let guidance = try await render("guidance 1")
        XCTAssertNotEqual(guidance, baseline)
        store.options.guidance = 7
        store.options.clipSkip = 2
        let clipSkip = try await render("CLIP skip 2")
        XCTAssertNotEqual(clipSkip, baseline)
        store.options.clipSkip = 1
        store.options.negativePrompt = "red, teapot, wood"
        let negative = try await render("negative prompt")
        XCTAssertNotEqual(negative, baseline)
        store.options.negativePrompt = ImageGenerationOptions().negativePrompt
        try await store.importFile(lora, kind: .lora, modelID: model.id)
        store.options.assets[0].weight = 0.8
        let weighted = try await render("LoRA weight 0.8")
        XCTAssertNotEqual(weighted, baseline)
        store.options.assets[0].weight = 0.2
        let lowerWeight = try await render("LoRA weight 0.2")
        XCTAssertNotEqual(lowerWeight, weighted)
        store.options.assets[0].enabled = false
        // Identical fixed-seed output proves disabling restores the baseline path.
        let disabled = try await render("LoRA disabled")
        XCTAssertEqual(disabled, baseline)
        let embeddedPrompt = "<llmhub-test-embedding> a red ceramic teapot"
        let withoutEmbedding = try await backend.generateImage(prompt: embeddedPrompt, steps: 2, seed: 12345)
        try await store.importFile(embedding, kind: .embedding, modelID: model.id)
        let token = try XCTUnwrap(store.options.assets.last?.keyword)
        let embedded = try await backend.generateImage(prompt: "<\(token)> a red ceramic teapot", steps: 2, seed: 12345)
        XCTAssertNotNil(embedded)
        XCTAssertNotEqual(embedded?.pngData(), withoutEmbedding?.pngData(), "Imported embedding must change conditioning for the same prompt")
        print("MODIFIER_GENERATION_PASS: textual inversion")
        let encoder = documents.appendingPathComponent("Models/counterfeit_v3.0_clip_vit_l14_f16.ckpt")
        let vae = documents.appendingPathComponent("Models/vae_ft_mse_840000_f16.ckpt")
        try await store.importFile(encoder, kind: .clipL, modelID: model.id)
        try await store.importFile(vae, kind: .vae, modelID: model.id)
        _ = try await render("imported CLIP-L and VAE")
        // The generic encoder selector must also resolve the CLIP encoder for this model.
        store.options.assets.indices.filter { store.options.assets[$0].kind == .clipL }.forEach { store.options.assets[$0].enabled = false }
        try await store.importFile(encoder, kind: .textEncoder, modelID: model.id)
        _ = try await render("imported text encoder")
        let portable = documents.appendingPathComponent("llmhub-test-portable-lora.safetensors")
        if fm.fileExists(atPath: portable.path) {
            try await store.importFile(portable, kind: .lora, modelID: model.id)
            _ = try await render("SafeTensors LoRA import and generation")
            store.options.assets[store.options.assets.count - 1].enabled = false
        }
        let reference = try XCTUnwrap(UIImage(data: baseline))
        let low = try await render("image-to-image strength 0.3", input: reference, strength: 0.3)
        let high = try await render("image-to-image strength 0.8", input: reference, strength: 0.8)
        XCTAssertNotEqual(low, high)
    }
    func testEncoderOverridesFollowSDKMetadataSlots() throws {
        let specifications = ModelZoo.availableSpecifications
        let single = try XCTUnwrap(specifications.first { $0.version == .v1 && $0.clipEncoder == nil })
        let multiple = try XCTUnwrap(specifications.first { $0.version == .sdxlBase && $0.clipEncoder != nil })
        let withT5 = try XCTUnwrap(specifications.first { $0.t5Encoder != nil })
        let clipL = ImageExtra(name: "CLIP-L", file: "test-clip-l.ckpt", kind: .clipL)
        let clipG = ImageExtra(name: "CLIP-G", file: "test-clip-g.ckpt", kind: .clipG)
        let encoder = ImageExtra(name: "T5", file: "test-t5.ckpt", kind: .textEncoder)
        let primary = StableDiffusionBackend.applyingEncoderOverrides([clipL], to: single)
        XCTAssertEqual(primary.textEncoder, clipL.file)
        XCTAssertNil(primary.clipEncoder)
        let supplementary = StableDiffusionBackend.applyingEncoderOverrides([clipL], to: multiple)
        XCTAssertEqual(supplementary.clipEncoder, clipL.file)
        XCTAssertEqual(supplementary.textEncoder, multiple.textEncoder)
        let both = StableDiffusionBackend.applyingEncoderOverrides([clipL, clipG], to: multiple)
        XCTAssertEqual(both.clipEncoder, clipL.file)
        XCTAssertEqual(both.textEncoder, clipG.file)
        let t5 = StableDiffusionBackend.applyingEncoderOverrides([encoder], to: withT5)
        XCTAssertEqual(t5.t5Encoder, encoder.file)
        XCTAssertEqual(t5.textEncoder, withT5.textEncoder)
    }

    func testEmbeddingWithUppercaseFilenameChangesConditioning() async throws {
        #if targetEnvironment(simulator)
        throw XCTSkip("Verify embedding conditioning on a physical iPhone")
        #endif
        let fixture = documents.appendingPathComponent("LLMHUB-TEST-EMBEDDING.safetensors")
        guard FileManager.default.fileExists(atPath: fixture.path) else { throw XCTSkip("Stage uppercase embedding fixture first") }
        let model = try XCTUnwrap(ModelData.allModels().first { $0.isDrawThingsImageGeneration && StableDiffusionBackend.isModelDownloaded(modelId: $0.id) })
        let store = ImageExtrasStore.shared
        let original = store.options
        let backend = StableDiffusionBackend.shared
        backend.unloadModel()
        store.options = ImageGenerationOptions()
        defer {
            backend.unloadModel()
            for asset in store.options.assets { store.remove(asset.id) }
            store.options = original
            try? FileManager.default.removeItem(at: fixture)
        }
        try await backend.loadModel(model)
        let prompt = "<llmhub-test-embedding> a red ceramic teapot"
        let beforeImage = try await backend.generateImage(prompt: prompt, steps: 2, seed: 12345)
        let before = try XCTUnwrap(beforeImage)
        try await store.importFile(fixture, kind: .embedding, modelID: model.id)
        XCTAssertEqual(store.options.assets.last?.keyword, "llmhub-test-embedding")
        let afterImage = try await backend.generateImage(prompt: prompt, steps: 2, seed: 12345)
        let after = try XCTUnwrap(afterImage)
        XCTAssertNotEqual(before.pngData(), after.pngData())
    }

    /// SDXL (CLIP-G) and SD3.5 (CLIP-G and T5) through the import path: importing the model's own encoder file keeps
    /// the output, and an altered copy changes it (see DeviceSelfTests). Skips models that are not downloaded unless
    /// LLMHUB_DOWNLOAD_MODELS=1 is set.
    func testImportedClipGAndT5ChangeConditioning() async throws {
        #if targetEnvironment(simulator)
        throw XCTSkip("Run generation coverage on a physical device")
        #endif
        let download = ProcessInfo.processInfo.environment["LLMHUB_DOWNLOAD_MODELS"] == "1"
        guard download || StableDiffusionBackend.isModelDownloaded(modelId: "ssd_1b_q6p_q8p.ckpt") else {
            throw XCTSkip("Download SSD-1B and SD3.5 Medium first or set LLMHUB_DOWNLOAD_MODELS=1")
        }
        try await DeviceSelfTests.checkImportedClipGAndT5(download: download).forEach { XCTFail($0) }
    }

    /// The same check after downloading SSD-1B and SD3.5 Medium (about 11.5 GB) on the device, for device farms that
    /// cannot set environment variables (select it with a test filter).
    func testImportedClipGAndT5ChangeConditioningAfterDownload() async throws {
        #if targetEnvironment(simulator)
        throw XCTSkip("Run generation coverage on a physical device")
        #endif
        try await DeviceSelfTests.checkImportedClipGAndT5(download: true).forEach { XCTFail($0) }
    }

    /// llama.cpp on Metal: LFM-2.5 1.2B Q4_0 with every layer on the GPU answers a one-word question.
    func testGgufGeneratesOnMetalAfterDownload() async throws {
        #if targetEnvironment(simulator)
        throw XCTSkip("Run llama.cpp Metal coverage on a physical device")
        #endif
        try await DeviceSelfTests.checkGgufOnMetal().forEach { XCTFail($0) }
    }

    func testZZRemoveOwnedTestFixtures() throws {
        let fm = FileManager.default
        let names = ["llmhub-test-lora", "llmhub-test-portable-lora", "llmhub-test-embedding", "LLMHUB-TEST-EMBEDDING"]
        for name in ["llmhub-test-lora.ckpt", "llmhub-test-tags.txt", "llmhub-test-embedding.safetensors", "llmhub-test-portable-lora.safetensors"] {
            let file = documents.appendingPathComponent(name)
            if fm.fileExists(atPath: file.path) { try fm.removeItem(at: file) }
        }
        // Imported weights were removed by the generation test. Remove only their now-stale registry records.
        for name in ["custom_lora.json", "custom_textual_inversions.json"] {
            let models = (fm.urls(for: .applicationSupportDirectory, in: .userDomainMask).first ?? documents).appendingPathComponent("Models")
            let registry = models.appendingPathComponent(name)
            guard let data = try? Data(contentsOf: registry),
                  let entries = try JSONSerialization.jsonObject(with: data) as? [[String: Any]] else { continue }
            let retained = entries.filter { entry in
                guard let name = entry["name"] as? String, names.contains(name),
                      let file = entry["file"] as? String, file.hasPrefix("extra_"), !file.contains("/") else { return true }
                return fm.fileExists(atPath: models.appendingPathComponent(file).path)
            }
            try JSONSerialization.data(withJSONObject: retained).write(to: registry, options: .atomic)
        }
    }

}
