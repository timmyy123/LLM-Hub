import AVKit
import Photos
import PhotosUI
import SwiftUI

/// AI Edge Gallery's Video Moment Finder. Each video is split into 2 second windows.
/// A window is two frames plus the audio around them, then a text search jumps to that time.
@MainActor
final class VideoMomentModel: ObservableObject {
    struct Video: Identifiable, Hashable, Codable {
        let id: String
        let name: String
        let durationMs: Int
    }

    struct Moment: Identifiable {
        let video: Video
        let startMs: Int
        let endMs: Int
        let score: Float
        var id: String { "\(video.id)#\(startMs)" }
    }

    @Published private(set) var videos: [Video] = []
    @Published private(set) var progress = MediaIndexingProgress()
    @Published private(set) var isPaused = false
    @Published private(set) var openVideo: Video?
    @Published private(set) var query = ""
    @Published private(set) var results: [Moment]?
    @Published private(set) var isSearching = false
    @Published private(set) var isLoadingModel = false
    @Published private(set) var modelError = false

    private let engine = MediaSearchEngine()
    private let store = MediaIndexStore(name: "video_moments")
    private var currentModel: AIModel?
    private var indexTask: Task<Void, Never>?
    private var searchTask: Task<Void, Never>?
    private var failed = Set<String>()
    private var loaded = false
    private static let doneMarker: Int32 = -1
    private static let windowMs = 2000
    private static let maxMs = 600_000

    func start(model: AIModel?) async {
        if !loaded {
            store.load()
            loaded = true
        }
        guard let model else { return }
        if currentModel?.id != model.id {
            indexTask?.cancel()
            await indexTask?.value
            await engine.unload()
            currentModel = model
            store.clear()
            failed.removeAll()
        }
        loadPickedVideos()
        await ensureEngine()
        startIndexing()
    }

    func stop() {
        indexTask?.cancel()
        store.saveIfDirty()
        Task { [engine] in await engine.unload() }
    }

    /// Gallery imports one picked video. It does not scan the library.
    func addPickedVideo(_ url: URL) {
        let accessing = url.startAccessingSecurityScopedResource()
        defer { if accessing { url.stopAccessingSecurityScopedResource() } }
        let ext = url.pathExtension.isEmpty ? "mov" : url.pathExtension
        let id = "\(UUID().uuidString).\(ext)"
        let dest = Self.videosDir.appendingPathComponent(id)
        guard (try? FileManager.default.copyItem(at: url, to: dest)) != nil else { return }
        let seconds = (try? AVAudioPlayer(contentsOf: dest))?.duration ?? 0
        let video = Video(id: id, name: url.deletingPathExtension().lastPathComponent, durationMs: Int(seconds * 1000))
        videos.insert(video, at: 0)
        savePickedVideos()
        let done = Set(store.all.filter { $0.startMs == Self.doneMarker }.map(\.id))
        progress = MediaIndexingProgress(processed: videos.filter { done.contains($0.id) || failed.contains($0.id) }.count, total: videos.count)
        startIndexing()
    }

    func open(_ video: Video) {
        openVideo = video
        query = ""
        results = nil
    }

    func closeVideo() {
        openVideo = nil
        query = ""
        results = nil
    }

    func updateQuery(_ text: String) {
        query = text
        searchTask?.cancel()
        guard !text.trimmingCharacters(in: .whitespaces).isEmpty else {
            results = nil
            isSearching = false
            return
        }
        isSearching = true
        searchTask = Task {
            try? await Task.sleep(nanoseconds: 300_000_000)
            guard !Task.isCancelled else { return }
            await ensureEngine()
            let vector = await engine.embedQuery(text.trimmingCharacters(in: .whitespaces))
            guard !Task.isCancelled else { return }
            results = vector.map(rank) ?? []
            isSearching = false
        }
    }

    func clearAll() {
        indexTask?.cancel()
        query = ""
        results = nil
        store.clear()
        videos = []
        try? FileManager.default.removeItem(at: Self.videosDir)
        try? FileManager.default.createDirectory(at: Self.videosDir, withIntermediateDirectories: true)
        savePickedVideos()
        failed.removeAll()
        progress = MediaIndexingProgress()
    }

    func pause() { isPaused = true }
    func resume() { isPaused = false; startIndexing() }
    func retryModel() { Task { await ensureEngine(); startIndexing() } }

    static func playbackURL(_ video: Video) -> URL {
        Self.videosDir.appendingPathComponent(video.id)
    }

    private static let videosDir: URL = {
        let dir = MediaSearchConfig.directory.appendingPathComponent("moment_videos", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }()
    private static var pickedURL: URL { MediaSearchConfig.directory.appendingPathComponent("moment_videos.json") }

    private func loadPickedVideos() {
        videos = ((try? JSONDecoder().decode([Video].self, from: Data(contentsOf: Self.pickedURL))) ?? [])
            .filter { FileManager.default.fileExists(atPath: Self.playbackURL($0).path) }
        let done = Set(store.all.filter { $0.startMs == Self.doneMarker }.map(\.id))
        progress = MediaIndexingProgress(processed: videos.filter { done.contains($0.id) || failed.contains($0.id) }.count, total: videos.count)
    }

    private func savePickedVideos() {
        if let data = try? JSONEncoder().encode(videos) {
            try? data.write(to: Self.pickedURL, options: .atomic)
        }
    }

    private func ensureEngine() async {
        guard let currentModel else { return }
        if await engine.loadedModelId == currentModel.id { return }
        isLoadingModel = true
        modelError = false
        do {
            try await engine.load(model: currentModel, maxInputTokens: 1024, cacheName: "video_moment_cache")
        } catch {
            modelError = true
        }
        isLoadingModel = false
    }

    private func rank(_ query: [Float]) -> [Moment] {
        let byId = Dictionary(uniqueKeysWithValues: videos.map { ($0.id, $0) })
        let only = openVideo?.id
        return store.all.compactMap { vector -> Moment? in
            guard !vector.vector.isEmpty, vector.id == only, let video = byId[vector.id] else { return nil }
            return Moment(video: video, startMs: Int(vector.startMs), endMs: Int(vector.endMs), score: mediaDot(query, vector.vector))
        }
        .sorted { $0.score > $1.score }
        .prefix(5)
        .map { $0 }
    }

    private func startIndexing() {
        guard indexTask == nil || indexTask?.isCancelled == true, !isPaused, currentModel != nil else { return }
        let engine = engine
        indexTask = Task.detached(priority: .utility) { [weak self] in
            guard let self else { return }
            await self.ensureEngine()
            while !Task.isCancelled {
                if await MainActor.run(body: { self.isPaused }) { break }
                guard let video = await self.nextVideo() else { break }
                await self.index(video, engine: engine)
            }
            await MainActor.run {
                self.store.saveIfDirty()
                self.indexTask = nil
            }
        }
    }

    private func nextVideo() -> Video? {
        let done = Set(store.all.filter { $0.startMs == Self.doneMarker }.map(\.id))
        return videos.first { !done.contains($0.id) && !failed.contains($0.id) }
    }

    private func index(_ video: Video, engine: MediaSearchEngine) async {
        let limit = min(video.durationMs, Self.maxMs)
        var start = 0
        var stored = false
        let samples = await samples16k(video)
        while start < limit {
            if Task.isCancelled { return }
            let end = min(start + Self.windowMs, limit)
            if end - start >= 500 {
                let parts = await windowParts(video, samples: samples, startMs: start, endMs: end)
                if let vector = await engine.embedMixed(parts), !parts.isEmpty {
                    await MainActor.run {
                        self.store.put(MediaVector(id: video.id, startMs: Int32(start), endMs: Int32(end), vector: vector))
                    }
                    stored = true
                }
            }
            start = end
        }
        await MainActor.run {
            if !stored { self.failed.insert(video.id) }
            self.store.put(MediaVector(id: video.id, startMs: Self.doneMarker, endMs: Self.doneMarker, vector: []))
            self.progress.processed = min(self.progress.processed + 1, self.progress.total)
        }
    }

    private func samples16k(_ video: Video) async -> [Float]? {
        let url = Self.playbackURL(video)
        return await Task.detached { decodeAudio16kMono(url: url, maxSeconds: 600) }.value
    }

    private func windowParts(_ video: Video, samples: [Float]?, startMs: Int, endMs: Int) async -> [MediaEmbedPart] {
        let url = Self.playbackURL(video)
        let times = [startMs, max(startMs, endMs - 1)]
        let frames = await Task.detached { times.compactMap { videoFrameJPEG(url: url, timeMs: $0) } }.value
        guard !frames.isEmpty else { return [] }
        let slices = audioSlices(samples, startMs: startMs, endMs: endMs, count: frames.count)
        let hasAudio = slices.contains { $0 != nil }
        if !hasAudio { return frames.map { .image($0) } }
        var parts: [MediaEmbedPart] = []
        for (index, jpeg) in frames.enumerated() {
            let stamp = clock(times[index])
            if let wav = slices[index] {
                parts.append(.text(stamp))
                parts.append(.audio(wav))
            }
            parts.append(.text(stamp))
            parts.append(.image(jpeg))
        }
        return parts
    }

    private func audioSlices(_ samples: [Float]?, startMs: Int, endMs: Int, count: Int) -> [Data?] {
        guard let samples, count > 0 else { return [] }
        let duration = max(1, endMs - startMs)
        return (0..<count).map { index in
            let fromMs = startMs + index * duration / count
            let toMs = startMs + (index + 1) * duration / count
            let from = min(samples.count, fromMs * 16)
            let to = min(samples.count, max(from, toMs * 16))
            guard to - from >= 4000 else { return nil }
            return pcm16WAV(samples, from..<to)
        }
    }

    private func clock(_ ms: Int) -> String {
        let s = max(0, ms / 1000)
        return String(format: "%d:%02d", s / 60, s % 60)
    }
}

private struct MomentMovie: Transferable {
    let url: URL

    static var transferRepresentation: some TransferRepresentation {
        FileRepresentation(contentType: .movie) { movie in
            SentTransferredFile(movie.url)
        } importing: { received in
            let ext = received.file.pathExtension.isEmpty ? "mov" : received.file.pathExtension
            let dest = FileManager.default.temporaryDirectory.appendingPathComponent("\(UUID().uuidString).\(ext)")
            try FileManager.default.copyItem(at: received.file, to: dest)
            return MomentMovie(url: dest)
        }
    }
}

private struct MomentThumb: View {
    let moment: VideoMomentModel.Moment
    let selected: Bool
    let onTap: () -> Void
    @State private var image: UIImage?

    var body: some View {
        Button(action: onTap) {
            ZStack(alignment: .bottom) {
                Group {
                    if let image {
                        Image(uiImage: image).resizable().scaledToFill()
                    } else {
                        Color.white.opacity(0.08)
                    }
                }
                VStack {
                    Text(String(format: "%.2f", moment.score)).font(.caption2).foregroundStyle(.white).padding(.top, 4)
                    Spacer()
                    Text(thumbClock(moment.startMs)).font(.caption2.bold()).foregroundStyle(.white).padding(.bottom, 4)
                }
            }
            .frame(width: 84, height: 120)
            .clipShape(RoundedRectangle(cornerRadius: 8))
            .overlay(RoundedRectangle(cornerRadius: 8).stroke(selected ? Color.accentColor : Color.white, lineWidth: selected ? 3 : 1))
        }
        .task(id: moment.id) {
            let url = VideoMomentModel.playbackURL(moment.video)
            let start = moment.startMs
            if let data = await Task.detached(operation: { videoFrameJPEG(url: url, timeMs: start) }).value {
                image = UIImage(data: data)
            }
        }
    }
}

private struct ClipEditSheet: View {
    let moment: VideoMomentModel.Moment
    let onDone: () -> Void
    @EnvironmentObject var settings: AppSettings
    @State private var start: Double
    @State private var end: Double

    init(moment: VideoMomentModel.Moment, onDone: @escaping () -> Void) {
        self.moment = moment
        self.onDone = onDone
        _start = State(initialValue: Double(moment.startMs))
        _end = State(initialValue: Double(moment.endMs))
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text(settings.localized("video_moment_save_edit")).font(.headline)
            Text(thumbClock(Int(start)))
            Slider(value: $start, in: 0...Double(max(moment.video.durationMs, 1)))
            Text(thumbClock(Int(end)))
            Slider(value: $end, in: 0...Double(max(moment.video.durationMs, 1)))
            Button(settings.localized("video_moment_save"), action: onDone)
                .frame(maxWidth: .infinity).frame(height: 48)
                .foregroundStyle(.white)
                .liquidGlassPrimaryButton(cornerRadius: 12)
        }
        .padding(24)
        .presentationDetents([.medium])
    }
}

private func thumbClock(_ ms: Int) -> String {
    let s = max(0, ms / 1000)
    return String(format: "%d:%02d", s / 60, s % 60)
}

private func videoFrameJPEG(url: URL, timeMs: Int) -> Data? {
    let asset = AVURLAsset(url: url)
    let generator = AVAssetImageGenerator(asset: asset)
    generator.appliesPreferredTrackTransform = true
    generator.maximumSize = CGSize(width: 512, height: 512)
    let time = CMTime(value: CMTimeValue(timeMs), timescale: 1000)
    guard let cg = try? generator.copyCGImage(at: time, actualTime: nil) else { return nil }
    return mediaSearchJPEG(from: UIImage(cgImage: cg), maxEdge: 512)
}

struct VideoMomentScreen: View {
    let onNavigateBack: () -> Void
    let onNavigateToModels: () -> Void

    @EnvironmentObject var settings: AppSettings
    @StateObject private var model = VideoMomentModel()
    @AppStorage("video_moment_model_id") private var selectedModelId = ""
    @State private var showSettings = false
    @State private var player: AVPlayer?
    @State private var playing: VideoMomentModel.Moment?

    private func leave() {
        if model.openVideo != nil {
            model.closeVideo()
            player?.pause()
        } else {
            onNavigateBack()
        }
    }
    @State private var pickedVideo: PhotosPickerItem?
    @State private var selectedMoment: VideoMomentModel.Moment?
    @State private var editing: VideoMomentModel.Moment?

    private var downloadedModels: [AIModel] { MediaSearchConfig.downloadedModels() }
    private var selectedModel: AIModel? {
        downloadedModels.first { $0.id == selectedModelId } ?? downloadedModels.first
    }

    var body: some View {
        Group {
            if downloadedModels.isEmpty {
                MediaSearchGateView(icon: "film.stack", onNavigateToModels: onNavigateToModels)
            } else if model.videos.isEmpty && model.progress.total == 0 {
                MediaSearchOnboardingView(
                    icon: "film.stack",
                    title: settings.localized("video_moment_onboarding_title"),
                    description: settings.localized("video_moment_onboarding_desc")
                ) {
                    PhotosPicker(selection: $pickedVideo, matching: .videos) {
                        Text(settings.localized("video_moment_pick")).frame(maxWidth: .infinity).frame(height: 50)
                    }
                    .foregroundStyle(.white)
                    .liquidGlassPrimaryButton(cornerRadius: 12)
                }
            } else {
                main
            }
        }
        .navigationTitle(model.openVideo?.name ?? settings.localized("feature_video_moment"))
        .navigationBarTitleDisplayMode(.inline)
        .apolloScreenBackground()
        .apolloNavigationBackground()
        .toolbar {
            ToolbarItem(placement: .navigationBarLeading) {
                Button { leave() } label: { Image(systemName: "arrow.left") }
            }
            if !downloadedModels.isEmpty {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button { showSettings = true } label: { Image(systemName: "slider.horizontal.3") }
                }
            }
        }
        .apolloSheet(isPresented: $showSettings) {
            MediaSearchSettingsSheet(
                selectedModelId: Binding(get: { selectedModel?.id ?? "" }, set: { selectedModelId = $0 }),
                availableModels: downloadedModels,
                countText: String(format: settings.localized("media_search_progress"), model.progress.processed, model.videos.count, model.progress.percent),
                onClearAll: { model.clearAll() }
            ) {
                PhotosPicker(selection: $pickedVideo, matching: .videos) {
                    Text(settings.localized("video_moment_pick")).frame(maxWidth: .infinity).frame(height: 50)
                }
                .liquidGlassPrimaryButton(cornerRadius: 12)
            }
            .environmentObject(settings)
        }
        .task(id: selectedModel?.id) { await model.start(model: selectedModel) }
        .onChange(of: pickedVideo) { _, item in
            guard let item else { return }
            pickedVideo = nil
            Task {
                if let movie = try? await item.loadTransferable(type: MomentMovie.self) {
                    model.addPickedVideo(movie.url)
                }
            }
        }
        .onDisappear { model.stop(); player?.pause() }
        .sheet(item: $editing) { moment in
            ClipEditSheet(moment: moment) { editing = nil }
        }
        .fullScreenCover(item: $playing) { moment in
            ZStack {
                Color.black.ignoresSafeArea()
                if let player { VideoPlayer(player: player) }
                VStack {
                    HStack {
                        Spacer()
                        Button { playing = nil; player?.pause() } label: {
                            Image(systemName: "xmark").foregroundStyle(.white).padding()
                        }
                    }
                    Spacer()
                }
            }
            .task {
                let url = VideoMomentModel.playbackURL(moment.video)
                let next = AVPlayer(url: url)
                player = next
                await next.seek(to: CMTime(value: CMTimeValue(moment.startMs), timescale: 1000))
                next.play()
            }
        }
    }

    private var main: some View {
        Group {
            if let open = model.openVideo {
                videoSearch(open)
            } else {
                projectGrid
            }
        }
    }

    private var projectGrid: some View {
        ScrollView {
            LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 16) {
                PhotosPicker(selection: $pickedVideo, matching: .videos) {
                    VStack(spacing: 8) {
                        ZStack {
                            RoundedRectangle(cornerRadius: 16).fill(Color.white.opacity(0.08))
                            Image(systemName: "plus").font(.largeTitle).foregroundStyle(.white)
                        }
                        .aspectRatio(1, contentMode: .fit)
                        Text(settings.localized("video_moment_pick")).font(.caption).foregroundStyle(.white)
                    }
                }
                ForEach(model.videos) { video in
                    Button { model.open(video) } label: {
                        VStack(spacing: 8) {
                            ZStack(alignment: .bottomLeading) {
                                RoundedRectangle(cornerRadius: 16).fill(Color.white.opacity(0.08))
                                Image(systemName: "film").font(.title).foregroundStyle(.white)
                                Text(clock(video.durationMs)).font(.caption2.bold()).foregroundStyle(.white).padding(8)
                            }
                            .aspectRatio(1, contentMode: .fit)
                            Text(video.name).font(.caption).foregroundStyle(.white).lineLimit(1)
                        }
                    }
                }
            }
            .padding(16)
        }
    }

    private func videoSearch(_ video: VideoMomentModel.Video) -> some View {
        ScrollView {
            LazyVStack(alignment: .leading, spacing: 12) {
                MediaSearchField(
                    text: Binding(get: { model.query }, set: { model.updateQuery($0) }),
                    placeholder: settings.localized("video_moment_hint"),
                    isSearching: model.isSearching
                )
                MediaSearchStatusCard(
                    isLoadingModel: model.isLoadingModel,
                    modelError: model.modelError,
                    progress: model.progress,
                    isPaused: model.isPaused,
                    onPause: model.pause,
                    onResume: model.resume,
                    onRetry: model.retryModel
                )
                if model.results?.isEmpty == true && !model.isSearching {
                    Text(settings.localized("media_search_no_results")).foregroundStyle(.white.opacity(0.8))
                }
                if let results = model.results, !results.isEmpty {
                    Text(settings.localized("video_moment_top")).font(.caption).foregroundStyle(.white)
                    ScrollView(.horizontal, showsIndicators: false) {
                        HStack(spacing: 12) {
                            ForEach(results) { moment in
                                MomentThumb(moment: moment, selected: selectedMoment?.id == moment.id) {
                                    selectedMoment = moment
                                    playing = moment
                                }
                            }
                        }
                    }
                }
                if selectedMoment != nil {
                    Button { editing = selectedMoment } label: {
                        Label(settings.localized("video_moment_save_edit"), systemImage: "pencil")
                            .frame(maxWidth: .infinity).frame(height: 44)
                    }
                    .foregroundStyle(.white)
                    .liquidGlassPrimaryButton(cornerRadius: 22)
                }
            }
            .padding(16)
        }
    }

    private func clock(_ ms: Int) -> String {
        let s = max(0, ms / 1000)
        return String(format: "%d:%02d", s / 60, s % 60)
    }
}
