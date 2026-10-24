import Foundation
import LiteRTLM

/// What one part of an embedding input is (LiteRT-LM's input data types).
enum EmbedPart {
    case text(String)
    /// JPEG or PNG bytes, ideally no larger than 512 px on the long edge.
    case image(Data)
    /// A 16 kHz mono PCM16 WAV.
    case audio(Data)

    var kind: String {
        switch self {
        case .text: return "text"
        case .image: return "image"
        case .audio: return "audio"
        }
    }
}

/// An EmbeddingGemma 2 .litertlm file loaded for embeddings (text, images, audio) through LiteRT-LM's EmbeddingEngine,
/// configured like Android's media search: 256 input tokens and 70 vision tokens per image, vectors L2-normalized (so
/// the dot product is the cosine). A port of LiteRtEngine.LiteRtEmbedder in core/LLMHub.Core/Services/LiteRtEngine.cs.
/// The engine is an actor, so calls are serialized.
final class MediaEmbedder: @unchecked Sendable {
    static let maxInputTokens = 256
    static let visionTokens = 70
    private static let queryPrefix = "task: search result | query: "
    private static let documentPrefix = "title: none | text: "
    private static let options = EmbeddingOptions(normalize: true, visionTokensPerImage: visionTokens)

    let model: InstalledModel
    /// "gpu" or "cpu".
    let backend: String
    let supportsImages: Bool
    let supportsAudio: Bool
    private let engine: EmbeddingEngine

    private init(model: InstalledModel, backend: String, engine: EmbeddingEngine, vision: Bool, audio: Bool) {
        self.model = model
        self.backend = backend
        self.engine = engine
        supportsImages = vision
        supportsAudio = audio
    }

    /// GPU (Metal) first, CPU if the GPU engine cannot start. A GPU engine can be created and still fail on the first
    /// input, so each one is probed with a short text before it is used.
    static func open(_ model: InstalledModel) async throws -> MediaEmbedder {
        let vision = model.tags.contains("vision")
        let audio = model.tags.contains("audio")
        for backend in [Backend.gpu, Backend.cpu()] {
            let cache = model.folder.appendingPathComponent("cache", isDirectory: true).appendingPathComponent("embed-" + backend.rawValue, isDirectory: true)
            try? FileManager.default.createDirectory(at: cache, withIntermediateDirectories: true)
            // Audio always runs on the CPU, as on Windows, Linux and Android.
            let config = EmbeddingEngineConfig(modelPath: model.primaryPath.path, backend: backend, visionBackend: vision ? backend : nil,
                                               audioBackend: audio ? .cpu() : nil, cacheDir: cache.path,
                                               maxInputLength: maxInputTokens, visionTokensPerImage: visionTokens)
            let engine = EmbeddingEngine(config: config)
            do {
                try await engine.initialize()
            } catch {
                Log.error("LiteRT-LM embeddings \(model.id): \(backend.rawValue) failed", error)
                continue
            }
            let embedder = MediaEmbedder(model: model, backend: backend.rawValue, engine: engine, vision: vision, audio: audio)
            if let probe = await embedder.embedText("test", query: true), !probe.isEmpty {
                Log.info("LiteRT-LM embeddings \(model.id) on \(backend.rawValue) (\(probe.count) dimensions)")
                return embedder
            }
            Log.info("LiteRT-LM embeddings \(model.id): \(backend.rawValue) probe failed")
            await engine.close()
        }
        throw NSError(domain: "LLMHub", code: 60, userInfo: [NSLocalizedDescriptionKey: L.f("EmbedderFailed", model.name)])
    }

    /// A search query or a document, with EmbeddingGemma's task prompts.
    func embedText(_ text: String, query: Bool) async -> [Float]? {
        var clean = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if clean.isEmpty { return nil }
        if clean.count > 6000 { clean = String(clean.prefix(6000)) }
        return await embed([.text((query ? Self.queryPrefix : Self.documentPrefix) + clean)])
    }

    /// Several frames make one vector.
    func embedImages(_ images: [Data]) async -> [Float]? {
        supportsImages && !images.isEmpty ? await embed(images.map { .image($0) }) : nil
    }

    func embedAudio(_ wav: Data) async -> [Float]? {
        supportsAudio ? await embed([.audio(wav)]) : nil
    }

    /// The parts of one input, embedded together into one vector; nil if the engine refused them.
    func embed(_ parts: [EmbedPart]) async -> [Float]? {
        let contents: [Content] = parts.map {
            switch $0 {
            case .text(let text): return .text(text)
            case .image(let data): return .imageData(data)
            case .audio(let data): return .audioData(data)
            }
        }
        do {
            let vector = try await engine.computeEmbedding(contents: contents, options: Self.options).embedding
            return vector.isEmpty ? nil : vector
        } catch {
            Log.error("LiteRT-LM embedding failed (\(parts.map(\.kind).joined(separator: ", ")))", error)
            return nil
        }
    }

    func close() async { await engine.close() }
}

/// The one EmbeddingGemma 2 (.litertlm) engine in memory, shared by Studio › Search and the /v1/embeddings image and
/// audio inputs, so switching between them never loads a second copy (core MediaEmbedderHost.cs).
@MainActor
final class MediaEmbedderHost {
    private(set) var loaded: MediaEmbedder?
    /// Loads and unloads run one at a time.
    private var chain: Task<Void, Never>?

    func get(_ model: InstalledModel, status: @escaping @MainActor (String) -> Void = { _ in }) async throws -> MediaEmbedder {
        try await serialized { [self] in
            if let loaded, loaded.model.id == model.id { return loaded }
            if let old = loaded {
                loaded = nil
                await old.close()
            }
            status(L.f("LoadingModel", model.name))
            let embedder = try await MediaEmbedder.open(model)
            loaded = embedder
            return embedder
        }
    }

    /// Frees the engine, unless it is `keep`'s (checked when the unload runs, after any load queued before it).
    func unload(keep: String? = nil) async {
        _ = try? await serialized { [self] in
            if let old = loaded, old.model.id != keep {
                loaded = nil
                await old.close()
            }
        }
    }

    private func serialized<T>(_ work: @escaping @MainActor () async throws -> T) async throws -> T {
        let previous = chain
        let task = Task { @MainActor in
            await previous?.value
            return try await work()
        }
        chain = Task { _ = try? await task.value }
        return try await task.value
    }
}
