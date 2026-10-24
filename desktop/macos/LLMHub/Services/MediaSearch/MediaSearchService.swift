import AVFoundation
import Foundation
import Observation

struct PhotoMatch: Identifiable, Hashable {
    let path: String
    let score: Float
    var id: String { path }
    var url: URL { URL(fileURLWithPath: path) }
    var name: String { url.lastPathComponent }
    var scoreText: String { String(format: "%.2f", score) }
}

struct AudioMoment: Identifiable, Hashable {
    let startMs: Int
    let endMs: Int
    let score: Float
    var id: Int { startMs }
    var label: String { String(format: "%d:%02d", startMs / 60000, startMs / 1000 % 60) }
}

struct AudioMatch: Identifiable, Hashable {
    let path: String
    let score: Float
    let moments: [AudioMoment]
    var id: String { path }
    var url: URL { URL(fileURLWithPath: path) }
    var name: String { url.lastPathComponent }
    var scoreText: String { String(format: "%.2f", score) }
}

/// Studio › Search, after the Android app's Photo Search and Audio Search: EmbeddingGemma 2 (440M for photos, 740M for
/// photos and audio) embeds a folder's photos, or 5-second moments of its audio, on this Mac; then a description, or a
/// photo, finds the closest ones. Indexes are kept per mode and redone only for new or changed files.
/// A port of core/LLMHub.Core/ViewModels/MediaSearchViewModel.cs.
@MainActor @Observable
final class MediaSearchService {
    nonisolated private static let photoEdge = 512
    nonisolated private static let audioWindowMs = 5_000
    nonisolated private static let maxAudioSeconds = 600
    nonisolated private static let silenceRms: Float = 0.003
    nonisolated private static let momentsPerFile = 3
    nonisolated private static let maxAudioResults = 40
    nonisolated private static let maxPhotoResults = 60
    nonisolated private static let maxFiles = 20_000

    private weak var app: AppModel?
    @ObservationIgnored private let photoIndex = MediaIndex(AppPaths.root.appendingPathComponent("media_search/photos.idx"))
    @ObservationIgnored private let audioIndex = MediaIndex(AppPaths.root.appendingPathComponent("media_search/audio.idx"))
    @ObservationIgnored private var indexing: Task<Void, Never>?
    @ObservationIgnored private let player = AudioIO()

    /// photos or audio.
    var mode = "photos" {
        didSet {
            guard mode != oldValue else { return }
            // Pick a build that can do this mode.
            if !modelFitsMode, let better = models.first(where: fits) { selectedModelId = better.id }
            status = ""
        }
    }
    var selectedModelId: String? {
        didSet {
            guard selectedModelId != oldValue else { return }
            if let selectedModelId {
                AppSettings.current.mediaModelId = selectedModelId
                AppSettings.save()
            }
            // Free the previous build's memory; the next index or search loads the new one.
            let keep = selectedModel?.id
            Task { await app?.mediaEmbedder.unload(keep: keep) }
        }
    }
    var photoFolder: String {
        didSet {
            AppSettings.current.mediaPhotoFolder = photoFolder
            AppSettings.save()
        }
    }
    var audioFolder: String {
        didSet {
            AppSettings.current.mediaAudioFolder = audioFolder
            AppSettings.save()
        }
    }
    var query = ""
    var status = ""
    var isIndexing = false
    var isSearching = false
    var progress = 0.0
    var photos: [PhotoMatch] = []
    var audio: [AudioMatch] = []
    /// The moment playing now, for its button's stop icon.
    var playing: String?
    /// Bumped when an index changes, so the indexed count refreshes.
    private var indexRevision = 0

    init() {
        let s = AppSettings.current
        let home = FileManager.default.homeDirectoryForCurrentUser
        photoFolder = s.mediaPhotoFolder.isEmpty ? home.appendingPathComponent("Pictures").path : s.mediaPhotoFolder
        audioFolder = s.mediaAudioFolder.isEmpty ? home.appendingPathComponent("Music").path : s.mediaAudioFolder
        selectedModelId = s.mediaModelId
        photoIndex.load()
        audioIndex.load()
    }

    func attach(_ app: AppModel) { self.app = app }

    /// Downloaded EmbeddingGemma 2 builds that understand images (440M) or images and audio (740M), 740M first.
    var models: [InstalledModel] {
        (app?.modelStore.installed ?? []).filter { $0.isEmbedding && $0.format == "litertlm" && $0.tags.contains("vision") }
            .sorted { ($0.tags.contains("audio") ? 0 : 1) < ($1.tags.contains("audio") ? 0 : 1) }
    }

    var selectedModel: InstalledModel? {
        let models = models
        return models.first { $0.id == selectedModelId } ?? models.first
    }

    var isPhotos: Bool { mode == "photos" }
    var folder: String { isPhotos ? photoFolder : audioFolder }
    var hasModel: Bool { selectedModel != nil }
    /// Audio needs the 740M build; photos work with either.
    var modelFitsMode: Bool { selectedModel.map(fits) ?? false }
    var indexedText: String {
        _ = indexRevision
        return L.f(isPhotos ? "SearchIndexedPhotos" : "SearchIndexedAudio", index.fileCount.formatted())
    }
    var placeholder: String { L.s(isPhotos ? "SearchPlaceholderPhotos" : "SearchPlaceholderAudio") }
    private var index: MediaIndex { isPhotos ? photoIndex : audioIndex }

    private func fits(_ model: InstalledModel) -> Bool { model.tags.contains(isPhotos ? "vision" : "audio") }

    func setFolder(_ folder: String) {
        if isPhotos { photoFolder = folder } else { audioFolder = folder }
    }

    private func embedder() async throws -> MediaEmbedder {
        guard let model = selectedModel, let app else {
            throw NSError(domain: "LLMHub", code: 61, userInfo: [NSLocalizedDescriptionKey: L.s("SearchNoModel")])
        }
        return try await app.mediaEmbedder.get(model) { [weak self] in self?.status = $0 }
    }

    func unload() async { await app?.mediaEmbedder.unload() }

    // MARK: Indexing

    func startIndexing() {
        guard !isIndexing else { return }
        indexing = Task { await index() }
    }

    func stopIndexing() { indexing?.cancel() }

    func index() async {
        guard !isIndexing, let model = selectedModel else { return }
        guard modelFitsMode else {
            status = L.s("SearchAudioNeeds740")
            return
        }
        let folder = folder
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: folder, isDirectory: &isDir), isDir.boolValue else {
            status = L.f("SearchNoFolder", folder)
            return
        }
        let photos = isPhotos
        let index = index
        isIndexing = true
        progress = 0
        let clock = Date()
        var added = 0
        defer {
            isIndexing = false
            progress = 0
            indexRevision += 1
            indexing = nil
        }
        do {
            let embedder = try await embedder()
            if index.modelId != model.id { index.reset(model.id) }
            status = L.s("SearchScanning")
            let extensions = photos ? MediaDecoders.imageExtensions : MediaDecoders.audioExtensions
            let files = await Task.detached { Self.scan(folder, extensions: extensions) }.value
            // Files that were deleted or are no longer under the folder leave the index.
            let present = Set(files)
            for gone in index.files where !present.contains(gone) { index.remove(gone) }
            let todo = files.filter { !index.isCurrent($0, stamp: Self.stamp($0)) }
            for (i, file) in todo.enumerated() {
                try Task.checkCancellation()
                let vectors = photos ? await Self.embedPhoto(embedder, file) : try await Self.embedAudio(embedder, file)
                index.put(file, stamp: Self.stamp(file), vectors: vectors)
                added += 1
                if added % 20 == 0 { index.saveIfDirty() }
                progress = Double(i + 1) / Double(todo.count)
                status = L.f("SearchIndexing", (i + 1).formatted(), todo.count.formatted())
            }
            index.saveIfDirty()
            let seconds = Date().timeIntervalSince(clock)
            status = L.f("SearchIndexDone", files.count.formatted(), String(format: "%.0f", seconds))
            Log.info("Media search \(photos ? "photos" : "audio"): \(added) new of \(files.count) in \(folder) with \(model.id) on \(embedder.backend) in \(String(format: "%.1f", seconds))s")
        } catch is CancellationError {
            index.saveIfDirty()
            status = L.s("SearchStopped")
        } catch {
            Log.error("Media indexing failed", error)
            index.saveIfDirty()
            status = L.f("ErrorPrefix", error.localizedDescription)
        }
    }

    nonisolated private static func scan(_ folder: String, extensions: Set<String>) -> [String] {
        guard let walker = FileManager.default.enumerator(at: URL(fileURLWithPath: folder, isDirectory: true),
                                                          includingPropertiesForKeys: [.isRegularFileKey],
                                                          options: [.skipsHiddenFiles, .skipsPackageDescendants]) else { return [] }
        var files: [String] = []
        for case let url as URL in walker where extensions.contains(url.pathExtension.lowercased()) {
            files.append(url.path)
            if files.count >= maxFiles { break }
        }
        return files.sorted()
    }

    /// Modified time and size: a file whose stamp changed is embedded again.
    nonisolated private static func stamp(_ path: String) -> Int64 {
        guard let attributes = try? FileManager.default.attributesOfItem(atPath: path) else { return 0 }
        let modified = (attributes[.modificationDate] as? Date)?.timeIntervalSince1970 ?? 0
        let size = (attributes[.size] as? NSNumber)?.int64Value ?? 0
        return Int64(modified * 10_000_000) ^ size
    }

    /// Decoding and embedding run off the main actor.
    nonisolated private static func embedPhoto(_ embedder: MediaEmbedder, _ file: String) async -> [MediaVector] {
        guard let jpeg = MediaDecoders.image(URL(fileURLWithPath: file), maxEdge: photoEdge), let vector = await embedder.embedImages([jpeg]) else { return [] }
        return [MediaVector(id: file, startMs: 0, endMs: 0, vector: vector)]
    }

    /// One vector per 5-second moment, skipping silence, for the first 10 minutes of the file.
    nonisolated private static func embedAudio(_ embedder: MediaEmbedder, _ file: String) async throws -> [MediaVector] {
        let rate = MediaDecoders.sampleRate
        guard let samples = await MediaDecoders.audio(URL(fileURLWithPath: file), maxSeconds: maxAudioSeconds), samples.count >= rate else { return [] }
        let window = rate * audioWindowMs / 1000
        var vectors: [MediaVector] = []
        for start in stride(from: 0, to: samples.count, by: window) {
            try Task.checkCancellation()
            let end = min(start + window, samples.count)
            if end - start < rate || MediaDecoders.rms(samples, from: start, to: end) < silenceRms { continue }
            if let vector = await embedder.embedAudio(MediaDecoders.wav(samples, from: start, to: end)) {
                vectors.append(MediaVector(id: file, startMs: Int32(start * 1000 / rate), endMs: Int32(end * 1000 / rate), vector: vector))
            }
        }
        return vectors
    }

    // MARK: Search

    func search() async {
        let text = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty, !isSearching else { return }
        await runSearch { await $0.embedText(text, query: true) }
    }

    /// Photos that look like this one.
    func searchByPhoto(_ url: URL) async {
        guard !isSearching else { return }
        mode = "photos"
        query = ""
        await runSearch { embedder in
            let jpeg = await Task.detached(priority: .userInitiated) { MediaDecoders.image(url, maxEdge: Self.photoEdge) }.value
            guard let jpeg else { return nil }
            return await embedder.embedImages([jpeg])
        }
    }

    private func runSearch(_ embed: (MediaEmbedder) async -> [Float]?) async {
        guard let model = selectedModel else {
            status = L.s("SearchNoModel")
            return
        }
        let index = index
        guard index.fileCount > 0, index.modelId == model.id else {
            status = L.s("SearchIndexFirst")
            return
        }
        isSearching = true
        defer { isSearching = false }
        do {
            let clock = Date()
            guard let vector = await embed(try await embedder()) else {
                status = L.s("SearchNoVector")
                return
            }
            let fm = FileManager.default
            if isPhotos {
                photos = index.all.map { PhotoMatch(path: $0.id, score: MediaIndex.dot(vector, $0.vector)) }
                    .filter { fm.fileExists(atPath: $0.path) }
                    .sorted { $0.score > $1.score }
                    .prefix(Self.maxPhotoResults).map { $0 }
            } else {
                audio = Dictionary(grouping: index.all, by: \.id).map { path, vectors in
                    let moments = vectors.map { AudioMoment(startMs: Int($0.startMs), endMs: Int($0.endMs), score: MediaIndex.dot(vector, $0.vector)) }
                        .sorted { $0.score > $1.score }
                    return AudioMatch(path: path, score: moments[0].score, moments: Array(moments.prefix(Self.momentsPerFile)))
                }
                .filter { fm.fileExists(atPath: $0.path) }
                .sorted { $0.score > $1.score }
                .prefix(Self.maxAudioResults).map { $0 }
            }
            status = L.f("SearchFound", "\(isPhotos ? photos.count : audio.count)", String(format: "%.1f", Date().timeIntervalSince(clock)))
        } catch {
            Log.error("Media search failed", error)
            status = L.f("ErrorPrefix", error.localizedDescription)
        }
    }

    /// Plays the matching moment (decoded again and written as a short WAV), so any format plays; again stops it.
    func play(_ match: AudioMatch, _ moment: AudioMoment) async {
        let key = "\(match.path)#\(moment.startMs)"
        if playing == key {
            player.stopPlayback()
            return
        }
        player.stopPlayback()
        guard let samples = await MediaDecoders.audio(match.url, maxSeconds: moment.endMs / 1000 + 1) else { return }
        let rate = MediaDecoders.sampleRate
        let from = min(samples.count, moment.startMs * rate / 1000)
        let to = min(samples.count, moment.endMs * rate / 1000)
        let wav = FileManager.default.temporaryDirectory.appendingPathComponent("llmhub-moment.wav")
        do {
            try MediaDecoders.wav(samples, from: from, to: to).write(to: wav)
        } catch {
            status = L.f("ErrorPrefix", error.localizedDescription)
            return
        }
        playing = key
        await player.play(wav)
        if playing == key { playing = nil }
    }
}
