import AVFoundation
import SwiftUI
import UniformTypeIdentifiers

// MARK: - Audio Search model

/// Finds moments inside imported audio from a text description. Like AI Edge Gallery's Video
/// Moment Finder, each file is split into short windows that are embedded separately.
@MainActor
final class AudioSearchModel: ObservableObject {
    struct Item: Codable, Identifiable, Hashable {
        let id: String          // stored file name
        let name: String
        let durationMs: Int
    }

    struct Moment: Hashable {
        let startMs: Int
        let endMs: Int
        let score: Float
    }

    struct Match: Identifiable {
        let item: Item
        let score: Float
        let moments: [Moment]
        /// One score per analyzed window, in time order.
        let timeline: [Moment]
        var id: String { item.id }
    }

    @Published private(set) var items: [Item] = []
    @Published private(set) var progress = MediaIndexingProgress()
    @Published private(set) var isPaused = false
    @Published private(set) var query = ""
    @Published private(set) var results: [Match]? = nil
    @Published private(set) var isSearching = false
    @Published private(set) var isLoadingModel = false
    @Published private(set) var modelError = false

    private let engine = MediaSearchEngine()
    private let store = MediaIndexStore(name: "audio")
    private var currentModel: AIModel?
    private var indexTask: Task<Void, Never>?
    private var searchTask: Task<Void, Never>?
    private var failed = Set<String>()
    private var loaded = false

    private static let windowMs = 5_000
    private static let maxSecondsPerFile = 600
    private static let silenceRMS: Float = 0.003
    private static let doneMarker: Int32 = -1

    static let filesDir: URL = {
        let dir = MediaSearchConfig.directory.appendingPathComponent("audio", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }()
    private static var itemsURL: URL { MediaSearchConfig.directory.appendingPathComponent("audio_items.json") }

    func fileURL(_ item: Item) -> URL { Self.filesDir.appendingPathComponent(item.id) }

    func start(model: AIModel?) async {
        if !loaded {
            store.load()
            items = (try? JSONDecoder().decode([Item].self, from: Data(contentsOf: Self.itemsURL))) ?? []
            items = items.filter { FileManager.default.fileExists(atPath: fileURL($0).path) }
            store.remove(ids: store.ids.subtracting(items.map(\.id)))
            loaded = true
        }
        updateProgress()
        guard let model else { return }
        if currentModel?.id != model.id {
            indexTask?.cancel()
            await indexTask?.value
            await engine.unload()
            currentModel = model
        }
        await ensureEngine()
        startIndexing()
    }

    func stop() {
        indexTask?.cancel()
        searchTask?.cancel()
        store.saveIfDirty()
        Task { [engine] in await engine.unload() }
    }

    func importFiles(_ urls: [URL]) async {
        for url in urls {
            let accessing = url.startAccessingSecurityScopedResource()
            defer { if accessing { url.stopAccessingSecurityScopedResource() } }
            let ext = url.pathExtension.isEmpty ? "m4a" : url.pathExtension
            let id = "\(UUID().uuidString).\(ext)"
            let dest = Self.filesDir.appendingPathComponent(id)
            guard (try? FileManager.default.copyItem(at: url, to: dest)) != nil else { continue }
            let duration = (try? await AVURLAsset(url: dest).load(.duration)).map { Int(CMTimeGetSeconds($0) * 1000) } ?? 0
            items.insert(Item(id: id, name: url.lastPathComponent, durationMs: duration), at: 0)
        }
        saveItems()
        updateProgress()
        startIndexing()
    }

    func clearAll() {
        indexTask?.cancel()
        clearSearch()
        store.clear()
        try? FileManager.default.removeItem(at: Self.filesDir)
        try? FileManager.default.createDirectory(at: Self.filesDir, withIntermediateDirectories: true)
        items = []
        failed.removeAll()
        saveItems()
        progress = MediaIndexingProgress()
    }

    func pause() { isPaused = true }

    func resume() {
        isPaused = false
        startIndexing()
    }

    func retryModel() {
        Task {
            await ensureEngine()
            startIndexing()
        }
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

    func clearSearch() {
        searchTask?.cancel()
        query = ""
        results = nil
        isSearching = false
    }

    // MARK: Private

    private func saveItems() {
        if let data = try? JSONEncoder().encode(items) { try? data.write(to: Self.itemsURL, options: .atomic) }
    }

    private func completedIds() -> Set<String> {
        Set(store.all.filter { $0.startMs == Self.doneMarker }.map(\.id))
    }

    private func updateProgress() {
        let done = completedIds()
        progress = MediaIndexingProgress(processed: items.filter { done.contains($0.id) || failed.contains($0.id) }.count, total: items.count)
    }

    private func ensureEngine() async {
        guard let currentModel else { return }
        if await engine.loadedModelId == currentModel.id { return }
        isLoadingModel = true
        modelError = false
        do {
            try await engine.load(model: currentModel)
        } catch {
            modelError = true
        }
        isLoadingModel = false
    }

    private func rank(_ query: [Float]) -> [Match] {
        let byId = Dictionary(uniqueKeysWithValues: items.map { ($0.id, $0) })
        var grouped: [String: [Moment]] = [:]
        for v in store.all where !v.vector.isEmpty && byId[v.id] != nil {
            grouped[v.id, default: []].append(Moment(startMs: Int(v.startMs), endMs: Int(v.endMs), score: mediaDot(query, v.vector)))
        }
        return grouped.compactMap { id, moments -> Match? in
            guard let item = byId[id] else { return nil }
            let best = moments.sorted { $0.score > $1.score }
            return Match(item: item, score: best[0].score, moments: Array(best.prefix(3)), timeline: moments.sorted { $0.startMs < $1.startMs })
        }
        .sorted { $0.score > $1.score }
        .prefix(40)
        .map { $0 }
    }

    private func startIndexing() {
        guard indexTask == nil || indexTask?.isCancelled == true, !isPaused, currentModel != nil else { return }
        indexTask = Task {
            defer { indexTask = nil }
            await ensureEngine()
            guard !modelError else { return }
            while !Task.isCancelled && !isPaused {
                let done = completedIds()
                let pending = items.filter { !done.contains($0.id) && !failed.contains($0.id) }
                if pending.isEmpty { break }
                for item in pending {
                    if Task.isCancelled || isPaused { break }
                    await indexFile(item)
                    progress.processed = min(progress.processed + 1, progress.total)
                }
            }
            store.saveIfDirty()
            if !query.isEmpty { updateQuery(query) }
        }
    }

    private func indexFile(_ item: Item) async {
        let url = fileURL(item)
        let maxSeconds = Self.maxSecondsPerFile
        guard let samples = await Task.detached(priority: .utility, operation: { decodeAudio16kMono(url: url, maxSeconds: maxSeconds) }).value else {
            failed.insert(item.id)
            return
        }
        // Drop partial windows from an interrupted run before re-analyzing the file.
        store.remove(ids: [item.id])
        let window = MediaSearchConfig.audioSampleRate * Self.windowMs / 1000
        let rate = MediaSearchConfig.audioSampleRate
        var start = 0
        while start < samples.count {
            if Task.isCancelled || isPaused { return }
            let end = min(start + window, samples.count)
            if end - start >= rate / 2 && mediaRMS(samples, start..<end) >= Self.silenceRMS,
               let vector = await engine.embedAudio(pcm16WAV(samples, start..<end)) {
                store.put(MediaVector(id: item.id, startMs: Int32(start * 1000 / rate), endMs: Int32(end * 1000 / rate), vector: vector))
            }
            start = end
        }
        store.put(MediaVector(id: item.id, startMs: Self.doneMarker, endMs: Self.doneMarker, vector: []))
        store.saveIfDirty()
    }
}

// MARK: - Playback

@MainActor
final class AudioMomentPlayer: NSObject, ObservableObject, @preconcurrency AVAudioPlayerDelegate {
    @Published private(set) var playingId: String?
    @Published private(set) var isPlaying = false
    @Published private(set) var positionMs = 0
    @Published private(set) var durationMs = 0

    private var player: AVAudioPlayer?
    private var timer: Timer?

    func play(_ item: AudioSearchModel.Item, url: URL, fromMs: Int) {
        if playingId != item.id || player == nil {
            stop()
            guard let p = try? AVAudioPlayer(contentsOf: url) else { return }
            p.delegate = self
            p.prepareToPlay()
            player = p
            playingId = item.id
            durationMs = Int(p.duration * 1000)
        }
        try? AVAudioSession.sharedInstance().setCategory(.playback, mode: .default)
        try? AVAudioSession.sharedInstance().setActive(true)
        player?.currentTime = Double(fromMs) / 1000
        positionMs = fromMs
        player?.play()
        isPlaying = true
        startTimer()
    }

    func toggle(_ item: AudioSearchModel.Item, url: URL) {
        if playingId == item.id, let player {
            if player.isPlaying {
                player.pause()
                isPlaying = false
            } else {
                player.play()
                isPlaying = true
                startTimer()
            }
        } else {
            play(item, url: url, fromMs: 0)
        }
    }

    func seek(_ item: AudioSearchModel.Item, url: URL, fraction: Double) {
        let total = playingId == item.id && durationMs > 0 ? durationMs : item.durationMs
        let target = Int(min(max(fraction, 0), 1) * Double(total))
        if playingId == item.id, let player {
            player.currentTime = Double(target) / 1000
            positionMs = target
        } else {
            play(item, url: url, fromMs: target)
        }
    }

    func stop() {
        player?.stop()
        player = nil
        playingId = nil
        isPlaying = false
        positionMs = 0
        timer?.invalidate()
        timer = nil
    }

    func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        isPlaying = false
        timer?.invalidate()
        timer = nil
    }

    private func startTimer() {
        timer?.invalidate()
        timer = Timer.scheduledTimer(withTimeInterval: 0.05, repeats: true) { [weak self] _ in
            Task { @MainActor in
                guard let self, let player = self.player else { return }
                self.positionMs = Int(player.currentTime * 1000)
            }
        }
    }
}

// MARK: - Audio Search screen

struct AudioSearchScreen: View {
    let onNavigateBack: () -> Void
    let onNavigateToModels: () -> Void

    @EnvironmentObject var settings: AppSettings
    @StateObject private var model = AudioSearchModel()
    @StateObject private var player = AudioMomentPlayer()
    @AppStorage("audio_search_model_id") private var selectedModelId = ""
    @State private var showSettings = false
    @State private var showImporter = false

    private var downloadedModels: [AIModel] { MediaSearchConfig.downloadedModels() }
    private var selectedModel: AIModel? {
        downloadedModels.first(where: { $0.id == selectedModelId }) ?? downloadedModels.first
    }

    var body: some View {
        Group {
            if downloadedModels.isEmpty {
                MediaSearchGateView(icon: "waveform.badge.magnifyingglass", onNavigateToModels: onNavigateToModels)
            } else if model.items.isEmpty {
                MediaSearchOnboardingView(
                    icon: "waveform.badge.magnifyingglass",
                    title: settings.localized("audio_search_onboarding_title"),
                    description: settings.localized("audio_search_onboarding_desc")
                ) {
                    Button { showImporter = true } label: {
                        Text(settings.localized("audio_search_import")).frame(maxWidth: .infinity).frame(height: 50)
                    }
                    .foregroundStyle(.white)
                    .liquidGlassPrimaryButton(cornerRadius: 12)
                }
            } else {
                mainView
            }
        }
        .navigationTitle(settings.localized("feature_audio_search"))
        .navigationBarTitleDisplayMode(.inline)
        .apolloScreenBackground()
        .apolloNavigationBackground()
        .toolbar {
            ToolbarItem(placement: .navigationBarLeading) {
                Button { onNavigateBack() } label: { Image(systemName: "arrow.left") }
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
                countText: String(format: settings.localized("audio_search_count"), model.progress.processed, model.items.count),
                onClearAll: {
                    player.stop()
                    model.clearAll()
                }
            ) {
                Button {
                    showSettings = false
                    showImporter = true
                } label: {
                    Text(settings.localized("audio_search_import"))
                        .frame(maxWidth: .infinity)
                        .frame(height: 50)
                }
                .liquidGlassPrimaryButton(cornerRadius: 12)
            }
            .environmentObject(settings)
        }
        .fileImporter(isPresented: $showImporter, allowedContentTypes: [.audio], allowsMultipleSelection: true) { result in
            guard case .success(let urls) = result, !urls.isEmpty else { return }
            Task { await model.importFiles(urls) }
        }
        .task(id: selectedModel?.id) { await model.start(model: selectedModel) }
        .onDisappear {
            player.stop()
            model.stop()
        }
    }

    private var mainView: some View {
        ScrollView {
            LazyVStack(alignment: .leading, spacing: 12) {
                MediaSearchField(
                    text: Binding(get: { model.query }, set: { model.updateQuery($0) }),
                    placeholder: settings.localized("audio_search_hint"),
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
                if model.results != nil && !model.progress.isComplete {
                    Text(settings.localized("media_search_incomplete_warning")).font(.caption).foregroundStyle(.white.opacity(0.7))
                }
                if let results = model.results {
                    if results.isEmpty && !model.isSearching {
                        Text(settings.localized("media_search_no_results")).font(.subheadline).foregroundStyle(.white.opacity(0.8))
                    }
                    ForEach(results) { match in
                        AudioSearchCard(item: match.item, match: match, url: model.fileURL(match.item), player: player)
                    }
                } else {
                    ForEach(model.items) { item in
                        AudioSearchCard(item: item, match: nil, url: model.fileURL(item), player: player)
                    }
                }
            }
            .padding(16)
        }
        .scrollDismissesKeyboard(.interactively)
    }
}

private func formatMs(_ ms: Int) -> String {
    let s = max(0, ms / 1000)
    return String(format: "%d:%02d", s / 60, s % 60)
}

private struct AudioSearchCard: View {
    let item: AudioSearchModel.Item
    let match: AudioSearchModel.Match?
    let url: URL
    @ObservedObject var player: AudioMomentPlayer
    @EnvironmentObject var settings: AppSettings

    var body: some View {
        let isCurrent = player.playingId == item.id
        let total = isCurrent && player.durationMs > 0 ? player.durationMs : item.durationMs
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 10) {
                Button { player.toggle(item, url: url) } label: {
                    Image(systemName: isCurrent && player.isPlaying ? "pause.fill" : "play.fill")
                        .font(.system(size: 14, weight: .bold))
                        .frame(width: 36, height: 36)
                        .foregroundStyle(.white)
                        .background(RoundedRectangle(cornerRadius: 12).fill(Color.white.opacity(0.08)))
                        .overlay(RoundedRectangle(cornerRadius: 12).stroke(Color.white.opacity(0.16), lineWidth: 1))
                }
                VStack(alignment: .leading, spacing: 2) {
                    Text(item.name).font(.subheadline.bold()).foregroundStyle(.white).lineLimit(1)
                    Text(isCurrent ? "\(formatMs(player.positionMs)) / \(formatMs(total))" : formatMs(total))
                        .font(.caption2.monospacedDigit()).foregroundStyle(.white.opacity(0.7))
                }
                Spacer()
            }
            MomentStrip(
                timeline: match?.timeline ?? [],
                totalMs: total,
                progress: isCurrent && total > 0 ? Double(player.positionMs) / Double(total) : 0
            ) { fraction in
                player.seek(item, url: url, fraction: fraction)
            }
            .frame(height: 36)
            if let moments = match?.moments, !moments.isEmpty {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                        ForEach(moments, id: \.startMs) { moment in
                            Button { player.play(item, url: url, fromMs: moment.startMs) } label: {
                                Label(String(format: settings.localized("audio_search_moment"), formatMs(moment.startMs)), systemImage: "play.circle")
                                    .font(.caption.bold())
                                    .padding(.horizontal, 10).padding(.vertical, 6)
                                    .foregroundStyle(.white)
                                    .background(Capsule().fill(ApolloPalette.accentStrong.opacity(0.35)))
                            }
                        }
                    }
                }
            }
        }
        .padding(12)
        .background(.ultraThinMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 14))
        .overlay(RoundedRectangle(cornerRadius: 14).stroke(Color.white.opacity(0.12), lineWidth: 1))
    }
}

/// Waveform-style strip: one bar per analyzed moment, height = match strength. Tap or drag to seek.
private struct MomentStrip: View {
    let timeline: [AudioSearchModel.Moment]
    let totalMs: Int
    let progress: Double
    let onSeek: (Double) -> Void

    private var bars: [CGFloat] {
        let count = 48
        guard !timeline.isEmpty, totalMs > 0 else { return Array(repeating: 0.3, count: count) }
        let lo = timeline.map(\.score).min() ?? 0
        let hi = timeline.map(\.score).max() ?? 1
        let range = hi - lo > 1e-4 ? hi - lo : 1
        return (0..<count).map { i in
            let t = (Double(i) + 0.5) / Double(count) * Double(totalMs)
            guard let m = timeline.first(where: { t >= Double($0.startMs) && t < Double($0.endMs) }) else { return 0.12 }
            return CGFloat(0.15 + 0.85 * (m.score - lo) / range)
        }
    }

    var body: some View {
        GeometryReader { geo in
            let values = bars
            Canvas { context, size in
                let spacing: CGFloat = 3
                let w = max(1, (size.width - spacing * CGFloat(values.count - 1)) / CGFloat(values.count))
                let played = progress * Double(values.count)
                for (i, v) in values.enumerated() {
                    let h = max(4, v * size.height)
                    let rect = CGRect(x: CGFloat(i) * (w + spacing), y: (size.height - h) / 2, width: w, height: h)
                    let color = Double(i) < played ? Color.white : Color.white.opacity(0.35)
                    context.fill(Path(roundedRect: rect, cornerRadius: w / 2), with: .color(color))
                }
                if progress > 0 {
                    let x = CGFloat(progress) * size.width
                    context.fill(Path(CGRect(x: x - 1, y: 0, width: 2, height: size.height)), with: .color(.white))
                }
            }
            .contentShape(Rectangle())
            .gesture(DragGesture(minimumDistance: 0).onChanged { value in
                onSeek(Double(value.location.x / max(geo.size.width, 1)))
            })
        }
    }
}
