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
    @Published private(set) var isImporting = false

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
    private static let indexedModelKey = "video_moment_indexed_model"

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
        }
        // A fresh screen has no current model. That is not a model switch, so the saved chunks stay.
        let indexedModel = UserDefaults.standard.string(forKey: Self.indexedModelKey)
        if indexedModel != nil, indexedModel != model.id {
            store.clear()
            failed.removeAll()
        }
        if indexedModel != model.id {
            UserDefaults.standard.set(model.id, forKey: Self.indexedModelKey)
        }
        loadPickedVideos()
        await repairStoredDurations()
        await ensureEngine()
        startIndexing()
    }

    func stop() {
        indexTask?.cancel()
        store.saveIfDirty()
        Task { [engine] in await engine.unload() }
    }

    /// Gallery imports one picked video. It does not scan the library.
    func importPickedItem(_ item: PhotosPickerItem) async {
        isImporting = true
        defer { isImporting = false }
        guard let movie = try? await item.loadTransferable(type: MomentMovie.self) else { return }
        await storeImportedVideo(movie.url)
    }

    private func storeImportedVideo(_ url: URL) async {
        let accessing = url.startAccessingSecurityScopedResource()
        defer { if accessing { url.stopAccessingSecurityScopedResource() } }
        let ext = url.pathExtension.isEmpty ? "mov" : url.pathExtension
        let id = "\(UUID().uuidString).\(ext)"
        let dest = Self.videosDir.appendingPathComponent(id)
        let name = url.deletingPathExtension().lastPathComponent
        let stored = await Task.detached {
            if FileManager.default.fileExists(atPath: dest.path) {
                try? FileManager.default.removeItem(at: dest)
            }
            do {
                try FileManager.default.moveItem(at: url, to: dest)
            } catch {
                do { try FileManager.default.copyItem(at: url, to: dest) } catch { return false }
            }
            return true
        }.value
        guard stored else { return }
        let durationMs = await videoDurationMs(url: dest)
        videos.insert(Video(id: id, name: name, durationMs: durationMs), at: 0)
        savePickedVideos()
        refreshProgress()
        startIndexing()
    }

    /// A failed AVAudioPlayer load stored 0 and marked that empty pass finished.
    private func repairStoredDurations() async {
        var changed = false
        for video in videos where video.durationMs < 1000 {
            let ms = await videoDurationMs(url: Self.playbackURL(video))
            guard ms >= 1000, ms > video.durationMs, let index = videos.firstIndex(where: { $0.id == video.id }) else { continue }
            videos[index] = Video(id: video.id, name: video.name, durationMs: ms)
            store.remove(ids: [video.id])
            failed.remove(video.id)
            changed = true
        }
        if changed {
            savePickedVideos()
            refreshProgress()
        }
    }

    private func refreshProgress() {
        let done = Set(store.all.filter { $0.startMs == Self.doneMarker }.map(\.id))
        progress = MediaIndexingProgress(processed: videos.filter { done.contains($0.id) || failed.contains($0.id) }.count, total: videos.count)
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
        results = nil
        isSearching = false
    }

    /// Gallery searches when the query is submitted, not on each keystroke.
    func submitSearch() {
        let text = query.trimmingCharacters(in: .whitespaces)
        searchTask?.cancel()
        guard !text.isEmpty else {
            results = nil
            isSearching = false
            return
        }
        isSearching = true
        searchTask = Task {
            await ensureEngine()
            let vector = await engine.embedQuery(text)
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
        let already = Set(store.all.filter { $0.id == video.id && !$0.vector.isEmpty }.map(\.startMs))
        var start = 0
        var stored = !already.isEmpty
        let samples = already.count * Self.windowMs >= limit ? nil : await samples16k(video)
        while start < limit {
            if Task.isCancelled { return }
            let end = min(start + Self.windowMs, limit)
            if already.contains(Int32(start)) {
                stored = true
            } else if end - start >= 500 {
                let parts = await windowParts(video, samples: samples, startMs: start, endMs: end)
                if let vector = await engine.embedMixed(parts), !parts.isEmpty {
                    store.put(MediaVector(id: video.id, startMs: Int32(start), endMs: Int32(end), vector: vector))
                    store.saveIfDirty()
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
        var frames: [Data] = []
        for time in times {
            if let jpeg = await videoFrameJPEG(url: url, timeMs: time) { frames.append(jpeg) }
        }
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

private struct VideoPoster: View {
    let video: VideoMomentModel.Video
    let clock: String
    @State private var image: UIImage?

    var body: some View {
        Color.white.opacity(0.08)
            .aspectRatio(1, contentMode: .fit)
            .overlay {
                if let image {
                    Image(uiImage: image)
                        .resizable()
                        .scaledToFill()
                }
            }
            .overlay(alignment: .bottomLeading) {
                Text(clock)
                    .font(.caption2.bold())
                    .foregroundStyle(.white)
                    .padding(8)
            }
            .clipped()
            .clipShape(RoundedRectangle(cornerRadius: 16))
        .task(id: video.id) {
            let url = VideoMomentModel.playbackURL(video)
            if let data = await videoFrameJPEG(url: url, timeMs: 500) {
                image = UIImage(data: data)
            }
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
                    Text("\(thumbClock(moment.startMs)) - \(thumbClock(moment.endMs))")
                        .font(.caption2.bold())
                        .foregroundStyle(.white)
                        .padding(.bottom, 4)
                }
            }
            .frame(width: 84, height: 120)
            .clipShape(RoundedRectangle(cornerRadius: 8))
            .overlay(RoundedRectangle(cornerRadius: 8).stroke(selected ? Color.accentColor : Color.white, lineWidth: selected ? 4 : 2))
        }
        .task(id: moment.id) {
            let url = VideoMomentModel.playbackURL(moment.video)
            let start = moment.startMs
            if let data = await videoFrameJPEG(url: url, timeMs: start) {
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
    return String(format: "%02d:%02d", s / 60, s % 60)
}

struct VideoMomentScreen: View {
    let onNavigateBack: () -> Void
    let onNavigateToModels: () -> Void

    @EnvironmentObject var settings: AppSettings
    @StateObject private var model = VideoMomentModel()
    @StateObject private var playback = MomentPlayer()
    @AppStorage("video_moment_model_id") private var selectedModelId = ""
    @State private var showSettings = false

    private func leave() {
        if model.openVideo != nil {
            playback.stop()
            model.closeVideo()
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
            } else if model.videos.isEmpty && !model.isImporting && model.progress.total == 0 {
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
                Group {
                    Button { leave() } label: { Image(systemName: "arrow.left") }
                }
                .apolloToolbarControl()
            }
            if !downloadedModels.isEmpty {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Group {
                        Button { showSettings = true } label: { Image(systemName: "slider.horizontal.3") }
                    }
                    .apolloToolbarControl()
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
            Task { await model.importPickedItem(item) }
        }
        .onChange(of: model.query) { _, _ in selectedMoment = nil }
        .onChange(of: model.openVideo?.id) { _, _ in
            if let video = model.openVideo {
                selectedMoment = nil
                playback.load(video)
            } else {
                playback.stop()
            }
        }
        .onChange(of: editing?.id) { _, id in
            if id != nil { playback.pauseForEditor() }
        }
        .onDisappear { model.stop(); playback.stop() }
        .sheet(item: $editing) { moment in
            ClipEditSheet(moment: moment) { editing = nil }
                .apolloMacSheetSizing()
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
                if model.isImporting {
                    VStack(spacing: 8) {
                        ZStack {
                            RoundedRectangle(cornerRadius: 16).fill(Color.white.opacity(0.08))
                            ProgressView().tint(.white)
                        }
                        .aspectRatio(1, contentMode: .fit)
                        Text(settings.localized("video_moment_pick")).font(.caption).foregroundStyle(.white)
                    }
                }
                ForEach(model.videos) { video in
                    Button { model.open(video) } label: {
                        VStack(spacing: 8) {
                            VideoPoster(video: video, clock: clock(video.durationMs))
                            Text(video.name).font(.caption).foregroundStyle(.white).lineLimit(1)
                        }
                    }
                }
            }
            .padding(16)
        }
    }

    private func videoSearch(_ video: VideoMomentModel.Video) -> some View {
        let results = model.results ?? []
        let intervals = mergedIntervals(results)
        return ZStack {
            Color.black
            MomentPlayerView(player: playback.player)
            VStack(spacing: 0) {
                VStack(alignment: .leading, spacing: 8) {
                    HStack(spacing: 8) {
                        Image(systemName: "magnifyingglass").foregroundStyle(.white.opacity(0.7))
                        TextField(
                            settings.localized("video_moment_hint"),
                            text: Binding(get: { model.query }, set: { model.updateQuery($0) })
                        )
                        .foregroundStyle(.white)
                        .submitLabel(.search)
                        .autocorrectionDisabled()
                        .onSubmit { model.submitSearch() }
                        if model.isSearching {
                            ProgressView().tint(.white)
                        } else if !model.query.isEmpty {
                            Button { model.updateQuery("") } label: {
                                Image(systemName: "xmark.circle.fill").foregroundStyle(.white.opacity(0.7))
                            }
                        }
                    }
                    .padding(.horizontal, 14)
                    .frame(height: 48)
                    .background(Color.white.opacity(0.16))
                    .clipShape(Capsule())
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
                        Text(settings.localized("media_search_no_results")).foregroundStyle(.white)
                    }
                    if !results.isEmpty {
                        Text(settings.localized("video_moment_top")).font(.caption).foregroundStyle(.white)
                        ScrollView(.horizontal, showsIndicators: false) {
                            HStack(spacing: 12) {
                                ForEach(results) { moment in
                                    MomentThumb(moment: moment, selected: selectedMoment?.id == moment.id) {
                                        selectedMoment = moment
                                        let end = intervals.first { $0.ids.contains(moment.id) }?.endMs ?? moment.endMs
                                        playback.playClip(startMs: moment.startMs, endMs: end)
                                    }
                                }
                            }
                        }
                        if selectedMoment != nil {
                            Button { editing = selectedMoment } label: {
                                Label(settings.localized("video_moment_save_edit"), systemImage: "pencil")
                                    .padding(.horizontal, 16)
                                    .frame(height: 40)
                            }
                            .foregroundStyle(.white)
                            .background(.ultraThinMaterial)
                            .clipShape(Capsule())
                            .frame(maxWidth: .infinity)
                        }
                    }
                }
                .padding(.horizontal, 16)
                .padding(.top, 12)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(
                    LinearGradient(colors: [Color.black.opacity(0.9), Color.black.opacity(0)], startPoint: .top, endPoint: .bottom)
                )
                Spacer(minLength: 0)
                MomentTransport(
                    playhead: playback.playhead,
                    durationMs: max(video.durationMs, 1),
                    isPlaying: playback.isPlaying,
                    isMuted: playback.isMuted,
                    intervals: intervals,
                    selectedId: selectedMoment?.id,
                    onToggle: playback.toggle,
                    onMute: playback.toggleMute,
                    onScrub: playback.scrub
                )
            }
        }
    }

    private func clock(_ ms: Int) -> String {
        let s = max(0, ms / 1000)
        return String(format: "%d:%02d", s / 60, s % 60)
    }
}

@MainActor
final class MomentPlayer: ObservableObject {
    let player = AVPlayer()
    @Published private(set) var isPlaying = false
    @Published private(set) var isMuted = false
    @Published private(set) var playhead: Double = 0
    private var timeToken: Any?
    private var clipEnd: Double?
    private var durationMs = 1
    private var scrubbing = false

    func load(_ video: VideoMomentModel.Video) {
        clipEnd = nil
        durationMs = max(video.durationMs, 1)
        playhead = 0
        isPlaying = false
        player.replaceCurrentItem(with: AVPlayerItem(url: VideoMomentModel.playbackURL(video)))
        player.pause()
        guard timeToken == nil else { return }
        timeToken = player.addPeriodicTimeObserver(forInterval: CMTime(value: 1, timescale: 10), queue: .main) { [weak self] time in
            let seconds = time.seconds
            Task { @MainActor in self?.tick(seconds) }
        }
    }

    func playClip(startMs: Int, endMs: Int) {
        clipEnd = nil
        let start = Double(startMs) / 1000
        let end = Double(max(endMs, startMs)) / 1000
        player.pause()
        player.seek(to: CMTime(seconds: start, preferredTimescale: 1000), toleranceBefore: .zero, toleranceAfter: .zero) { [weak self] finished in
            Task { @MainActor in
                guard let self, finished else { return }
                guard end > start + 0.05 else {
                    self.playhead = start * 1000 / Double(self.durationMs)
                    return
                }
                self.clipEnd = end
                self.player.play()
                self.isPlaying = true
            }
        }
    }

    func toggle() {
        clipEnd = nil
        if player.rate > 0 {
            player.pause()
            isPlaying = false
        } else {
            player.play()
            isPlaying = true
        }
    }

    func toggleMute() {
        isMuted.toggle()
        player.isMuted = isMuted
    }

    func scrub(_ fraction: Double) {
        clipEnd = nil
        scrubbing = true
        let clamped = min(1, max(0, fraction))
        playhead = clamped
        let seconds = clamped * Double(durationMs) / 1000
        player.seek(to: CMTime(seconds: seconds, preferredTimescale: 1000), toleranceBefore: .zero, toleranceAfter: .zero) { [weak self] _ in
            Task { @MainActor in self?.scrubbing = false }
        }
    }

    func pauseForEditor() {
        clipEnd = nil
        player.pause()
        isPlaying = false
    }

    func stop() {
        clipEnd = nil
        player.pause()
        isPlaying = false
        if let timeToken {
            player.removeTimeObserver(timeToken)
            self.timeToken = nil
        }
        player.replaceCurrentItem(with: nil)
    }

    private func tick(_ seconds: Double) {
        guard seconds.isFinite else { return }
        if let clipEnd, seconds >= clipEnd - 0.05 {
            self.clipEnd = nil
            player.pause()
            isPlaying = false
            player.seek(to: CMTime(seconds: clipEnd, preferredTimescale: 1000), toleranceBefore: .zero, toleranceAfter: .zero)
            playhead = min(1, clipEnd * 1000 / Double(durationMs))
            return
        }
        if !scrubbing {
            playhead = min(1, max(0, seconds * 1000 / Double(durationMs)))
        }
        isPlaying = player.rate > 0
    }
}

#if os(macOS)
struct MomentPlayerView: NSViewRepresentable {
    let player: AVPlayer

    func makeNSView(context: Context) -> MomentPlayerHost {
        let view = MomentPlayerHost()
        view.playerLayer.player = player
        view.playerLayer.videoGravity = .resizeAspect
        view.playerLayer.backgroundColor = NSColor.black.cgColor
        return view
    }

    func updateNSView(_ nsView: MomentPlayerHost, context: Context) {
        nsView.playerLayer.player = player
    }
}

/// Layer-hosting NSView whose backing layer is an AVPlayerLayer.
final class MomentPlayerHost: NSView {
    let playerLayer = AVPlayerLayer()

    override init(frame frameRect: NSRect) {
        super.init(frame: frameRect)
        layer = playerLayer
        wantsLayer = true
    }

    required init?(coder: NSCoder) {
        super.init(coder: coder)
        layer = playerLayer
        wantsLayer = true
    }
}
#else
private struct MomentPlayerView: UIViewRepresentable {
    let player: AVPlayer

    func makeUIView(context: Context) -> MomentPlayerHost {
        let view = MomentPlayerHost()
        view.playerLayer.player = player
        view.playerLayer.videoGravity = .resizeAspect
        view.backgroundColor = .black
        return view
    }

    func updateUIView(_ uiView: MomentPlayerHost, context: Context) {
        uiView.playerLayer.player = player
    }
}

private final class MomentPlayerHost: UIView {
    override class var layerClass: AnyClass { AVPlayerLayer.self }
    var playerLayer: AVPlayerLayer { layer as! AVPlayerLayer }
}

#endif

struct MergedInterval: Identifiable {
    let startMs: Int
    let endMs: Int
    let ids: Set<String>
    var id: Int { startMs }
}

func mergedIntervals(_ results: [VideoMomentModel.Moment]) -> [MergedInterval] {
    var merged: [MergedInterval] = []
    for result in results.sorted(by: { $0.startMs < $1.startMs }) {
        if let last = merged.last, result.startMs <= last.endMs + 500 {
            merged[merged.count - 1] = MergedInterval(
                startMs: last.startMs,
                endMs: max(last.endMs, result.endMs),
                ids: last.ids.union([result.id])
            )
        } else {
            merged.append(MergedInterval(startMs: result.startMs, endMs: result.endMs, ids: [result.id]))
        }
    }
    return merged
}

private struct MomentTransport: View {
    let playhead: Double
    let durationMs: Int
    let isPlaying: Bool
    let isMuted: Bool
    let intervals: [MergedInterval]
    let selectedId: String?
    let onToggle: () -> Void
    let onMute: () -> Void
    let onScrub: (Double) -> Void

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Button(action: onToggle) {
                    Image(systemName: isPlaying ? "pause.fill" : "play.fill")
                        .foregroundStyle(.white)
                        .frame(width: 44, height: 44)
                }
                Spacer()
                Text("\(thumbClock(Int(playhead * Double(durationMs)))) / \(thumbClock(durationMs))")
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.white)
                Spacer()
                Button(action: onMute) {
                    Image(systemName: isMuted ? "speaker.slash.fill" : "speaker.wave.2.fill")
                        .foregroundStyle(.white)
                        .frame(width: 44, height: 44)
                }
            }
            .padding(.horizontal, 8)
            GeometryReader { geo in
                let width = max(geo.size.width, 1)
                ZStack(alignment: .leading) {
                    Capsule().fill(Color.white.opacity(0.35)).frame(height: 8)
                    Capsule().fill(Color.white).frame(width: width * playhead, height: 8)
                    ForEach(intervals) { interval in
                        let start = CGFloat(interval.startMs) / CGFloat(max(durationMs, 1))
                        let end = CGFloat(interval.endMs) / CGFloat(max(durationMs, 1))
                        let selected = selectedId.map { interval.ids.contains($0) } ?? false
                        let markerWidth = max(6, width * (end - start))
                        let x = min(width - markerWidth, width * start)
                        RoundedRectangle(cornerRadius: 3)
                            .fill(selected ? Color.accentColor : Color.white.opacity(0.85))
                            .frame(width: markerWidth, height: selected ? 18 : 14)
                            .offset(x: x)
                    }
                    RoundedRectangle(cornerRadius: 2)
                        .fill(Color.white)
                        .frame(width: 4, height: 24)
                        .offset(x: min(width - 4, max(0, width * playhead - 2)))
                }
                .frame(maxHeight: .infinity)
                .contentShape(Rectangle())
                .gesture(
                    DragGesture(minimumDistance: 0).onChanged { value in
                        onScrub(min(1, max(0, value.location.x / width)))
                    }
                )
            }
            .frame(height: 48)
            .padding(.horizontal, 24)
        }
        .padding(.bottom, 8)
        .background(
            LinearGradient(colors: [Color.black.opacity(0), Color.black.opacity(0.9)], startPoint: .top, endPoint: .bottom)
        )
    }
}
