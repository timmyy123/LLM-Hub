import AVKit
import Photos
import PhotosUI
import SwiftUI
import UniformTypeIdentifiers

// MARK: - Photo Search model

/// Semantic text and photo-to-photo search over the user's photos with EmbeddingGemma 2.
@MainActor
final class PhotoSearchModel: ObservableObject {
    enum Source: String { case none, allPhotos, selected }

    /// Cosine floor. Weaker matches stay out of the grid so a search is not padded with unrelated photos.
    static let minimumScore: Float = 0.30

    /// "ph:<PHAsset localIdentifier>" for library items, "file:<name>" for imported copies.
    struct Item: Identifiable, Hashable {
        let id: String
        let isVideo: Bool
    }

    struct Match: Identifiable {
        let item: Item
        let score: Float
        var id: String { item.id }
    }

    @Published private(set) var source: Source
    @Published private(set) var items: [Item] = []
    @Published private(set) var progress = MediaIndexingProgress()
    @Published private(set) var isPaused = false
    @Published private(set) var query = ""
    @Published private(set) var results: [Match]? = nil
    @Published private(set) var isSearching = false
    @Published private(set) var similarTo: Item? = nil
    @Published private(set) var isLoadingModel = false
    @Published private(set) var modelError = false
    @Published private(set) var libraryAccessDenied = false

    let imageManager = PHCachingImageManager()
    private let engine = MediaSearchEngine()
    private let store = MediaIndexStore(name: "photos")
    private var currentModel: AIModel?
    private var indexTask: Task<Void, Never>?
    private var searchTask: Task<Void, Never>?
    private var failed = Set<String>()
    private var storeLoaded = false

    private static let sourceKey = "photo_search_source"
    private static let importedDir: URL = {
        let dir = MediaSearchConfig.directory.appendingPathComponent("photos", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }()

    init() {
        source = Source(rawValue: UserDefaults.standard.string(forKey: Self.sourceKey) ?? "") ?? .none
    }

    func start(model: AIModel?) async {
        if !storeLoaded {
            store.load()
            storeLoaded = true
        }
        refreshItems()
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

    func useAllPhotos() async {
        let status = await PHPhotoLibrary.requestAuthorization(for: .readWrite)
        guard status == .authorized || status == .limited else {
            libraryAccessDenied = true
            return
        }
        libraryAccessDenied = false
        setSource(.allPhotos)
        refreshItems()
        startIndexing()
    }

    func importPhotos(_ pickerItems: [PhotosPickerItem]) async {
        var added = false
        for pickerItem in pickerItems {
            if let data = try? await pickerItem.loadTransferable(type: Data.self),
               let image = UIImage(data: data),
               let jpeg = mediaSearchJPEG(from: image, maxEdge: 1024) {
                let name = "\(UUID().uuidString).jpg"
                if (try? jpeg.write(to: Self.importedDir.appendingPathComponent(name), options: .atomic)) != nil {
                    added = true
                }
                continue
            }
            if let movie = try? await pickerItem.loadTransferable(type: PickedMovie.self) {
                let ext = movie.url.pathExtension.isEmpty ? "mov" : movie.url.pathExtension
                let dest = Self.importedDir.appendingPathComponent("\(UUID().uuidString).\(ext)")
                if (try? FileManager.default.copyItem(at: movie.url, to: dest)) != nil { added = true }
            }
        }
        guard added else { return }
        if source != .allPhotos { setSource(.selected) }
        refreshItems()
        startIndexing()
    }

    func clearAll() {
        indexTask?.cancel()
        clearSearch()
        store.clear()
        try? FileManager.default.removeItem(at: Self.importedDir)
        try? FileManager.default.createDirectory(at: Self.importedDir, withIntermediateDirectories: true)
        failed.removeAll()
        setSource(.none)
        items = []
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
        similarTo = nil
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
            results = vector.map { rank($0, excluding: nil) } ?? []
            isSearching = false
        }
    }

    func findSimilar(_ item: Item) {
        searchTask?.cancel()
        query = ""
        similarTo = item
        isSearching = true
        searchTask = Task {
            var vector = store.vectors(for: item.id).first?.vector
            if vector == nil {
                let frames = await loadFrames(item, maxEdge: 1024)
                if !frames.isEmpty {
                    await ensureEngine()
                    vector = await engine.embedImages(frames)
                }
            }
            guard !Task.isCancelled else { return }
            results = vector.map { rank($0, excluding: item.id) } ?? []
            isSearching = false
        }
    }

    func clearSearch() {
        searchTask?.cancel()
        query = ""
        similarTo = nil
        results = nil
        isSearching = false
    }

    /// Thumbnail / full image for display.
    func playbackURL(_ item: Item) async -> URL? {
        if item.id.hasPrefix("file:") {
            return Self.importedDir.appendingPathComponent(String(item.id.dropFirst(5)))
        }
        let localId = String(item.id.dropFirst(3))
        guard let asset = PHAsset.fetchAssets(withLocalIdentifiers: [localId], options: nil).firstObject else { return nil }
        let options = PHVideoRequestOptions()
        options.isNetworkAccessAllowed = true
        options.deliveryMode = .automatic
        return await withCheckedContinuation { continuation in
            let once = PhotoRequestOnce()
            PHImageManager.default().requestAVAsset(forVideo: asset, options: options) { avAsset, _, _ in
                let url = (avAsset as? AVURLAsset)?.url
                once.run { continuation.resume(returning: url) }
            }
        }
    }

    func loadImage(_ item: Item, targetSize: CGSize, fast: Bool = true) async -> UIImage? {
        if item.id.hasPrefix("file:") {
            let url = Self.importedDir.appendingPathComponent(String(item.id.dropFirst(5)))
            if item.isVideo {
                let edge = max(targetSize.width, targetSize.height)
                if let jpeg = await Task.detached(operation: { videoKeyframes(url: url, maxEdge: edge) }).value.first {
                    return UIImage(data: jpeg)
                }
                return nil
            }
            return UIImage(contentsOfFile: url.path)
        }
        let localId = String(item.id.dropFirst(3))
        guard let asset = PHAsset.fetchAssets(withLocalIdentifiers: [localId], options: nil).firstObject else { return nil }
        let options = PHImageRequestOptions()
        options.deliveryMode = fast ? .fastFormat : .highQualityFormat
        options.resizeMode = .fast
        options.isNetworkAccessAllowed = true
        let once = PhotoRequestOnce()
        return await withCheckedContinuation { continuation in
            imageManager.requestImage(for: asset, targetSize: targetSize, contentMode: .aspectFill, options: options) { image, info in
                // Degraded previews may arrive first; wait for the final image (or a failure).
                if (info?[PHImageResultIsDegradedKey] as? Bool) == true && image != nil { return }
                once.run { continuation.resume(returning: image) }
            }
        }
    }

    // MARK: Private

    private func setSource(_ value: Source) {
        source = value
        UserDefaults.standard.set(value.rawValue, forKey: Self.sourceKey)
    }

    private func refreshItems() {
        var list: [Item] = []
        if source == .allPhotos {
            let status = PHPhotoLibrary.authorizationStatus(for: .readWrite)
            libraryAccessDenied = !(status == .authorized || status == .limited)
            if !libraryAccessDenied {
            let options = PHFetchOptions()
            options.sortDescriptors = [NSSortDescriptor(key: "creationDate", ascending: false)]
            options.predicate = NSPredicate(
                format: "mediaType == %d OR mediaType == %d",
                PHAssetMediaType.image.rawValue,
                PHAssetMediaType.video.rawValue
            )
            let assets = PHAsset.fetchAssets(with: options)
                list.reserveCapacity(assets.count)
                assets.enumerateObjects { asset, _, _ in
                list.append(Item(id: "ph:" + asset.localIdentifier, isVideo: asset.mediaType == .video))
            }
            }
        }
        if source != .none {
            let files = (try? FileManager.default.contentsOfDirectory(at: Self.importedDir, includingPropertiesForKeys: [.creationDateKey])) ?? []
            list += files.sorted { $0.lastPathComponent > $1.lastPathComponent }.map {
                Item(id: "file:" + $0.lastPathComponent, isVideo: !["jpg", "jpeg", "png", "heic"].contains($0.pathExtension.lowercased()))
            }
        }
        items = list
        let live = Set(list.map(\.id))
        store.remove(ids: store.ids.subtracting(live))
        store.saveIfDirty()
        let indexed = store.ids
        progress = MediaIndexingProgress(processed: list.filter { indexed.contains($0.id) || failed.contains($0.id) }.count, total: list.count)
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

    private func rank(_ query: [Float], excluding: String?) -> [Match] {
        let live = Set(items.map(\.id))
            let videoIds = Set(items.filter(\.isVideo).map(\.id))
            return store.all
            .filter { $0.id != excluding && live.contains($0.id) }
            .map { Match(item: Item(id: $0.id, isVideo: videoIds.contains($0.id)), score: mediaDot(query, $0.vector)) }
            .filter { $0.score >= PhotoSearchModel.minimumScore }
            .sorted { $0.score > $1.score }
            .prefix(120)
            .map { $0 }
    }

    private func loadFrames(_ item: Item, maxEdge: CGFloat) async -> [Data] {
        if item.isVideo { return await loadVideoFrames(item, maxEdge: maxEdge) }
        if let jpeg = await loadJPEG(item, maxEdge: maxEdge) { return [jpeg] }
        return []
    }

    private func loadVideoFrames(_ item: Item, maxEdge: CGFloat) async -> [Data] {
        if item.id.hasPrefix("file:") {
            let url = Self.importedDir.appendingPathComponent(String(item.id.dropFirst(5)))
            return await Task.detached { videoKeyframes(url: url, maxEdge: maxEdge) }.value
        }
        let localId = String(item.id.dropFirst(3))
        guard let asset = PHAsset.fetchAssets(withLocalIdentifiers: [localId], options: nil).firstObject else { return [] }
        let options = PHVideoRequestOptions()
        options.isNetworkAccessAllowed = true
        options.deliveryMode = .fastFormat
        let url: URL? = await withCheckedContinuation { continuation in
            let once = PhotoRequestOnce()
            PHImageManager.default().requestAVAsset(forVideo: asset, options: options) { avAsset, _, _ in
                let url = (avAsset as? AVURLAsset)?.url
                once.run { continuation.resume(returning: url) }
            }
        }
        guard let url else { return [] }
        return await Task.detached { videoKeyframes(url: url, maxEdge: maxEdge) }.value
    }

    private func loadJPEG(_ item: Item, maxEdge: CGFloat) async -> Data? {
        guard let image = await loadImage(item, targetSize: CGSize(width: maxEdge, height: maxEdge), fast: false) else { return nil }
        return mediaSearchJPEG(from: image, maxEdge: maxEdge)
    }

    private func startIndexing() {
        guard indexTask == nil || indexTask?.isCancelled == true, !isPaused, currentModel != nil else { return }
        indexTask = Task {
            defer { indexTask = nil }
            await ensureEngine()
            guard !modelError else { return }
            var sinceSave = 0
            while !Task.isCancelled && !isPaused {
                let indexed = store.ids
                let pending = items.filter { !indexed.contains($0.id) && !failed.contains($0.id) }
                if pending.isEmpty { break }
                for item in pending {
                    if Task.isCancelled || isPaused { break }
                    await Task.yield()
                    let frames = await loadFrames(item, maxEdge: 512)
                    if let vector = await engine.embedImages(frames), !frames.isEmpty {
                        store.put(MediaVector(id: item.id, startMs: 0, endMs: 0, vector: vector))
                    } else {
                        failed.insert(item.id)
                    }
                    progress.processed = min(progress.processed + 1, progress.total)
                    sinceSave += 1
                    if sinceSave >= 20 {
                        store.saveIfDirty()
                        sinceSave = 0
                    }
                }
            }
            store.saveIfDirty()
            if !query.isEmpty { updateQuery(query) }
        }
    }
}

private final class PhotoRequestOnce: @unchecked Sendable {
    private let lock = NSLock()
    private var done = false

    func run(_ body: () -> Void) {
        lock.lock()
        defer { lock.unlock() }
        guard !done else { return }
        done = true
        body()
    }
}

// MARK: - Photo Search screen

struct PhotoSearchScreen: View {
    let onNavigateBack: () -> Void
    let onNavigateToModels: () -> Void

    @EnvironmentObject var settings: AppSettings
    @StateObject private var model = PhotoSearchModel()
    @AppStorage("photo_search_model_id") private var selectedModelId = ""
    @State private var showSettings = false
    @State private var pickerItems: [PhotosPickerItem] = []
    @State private var viewing: PhotoSearchModel.Item?

    private var downloadedModels: [AIModel] { MediaSearchConfig.downloadedModels() }
    private var selectedModel: AIModel? {
        downloadedModels.first(where: { $0.id == selectedModelId }) ?? downloadedModels.first
    }

    var body: some View {
        Group {
            if downloadedModels.isEmpty {
                MediaSearchGateView(icon: "photo.on.rectangle.angled", onNavigateToModels: onNavigateToModels)
            } else if model.source == .none {
                MediaSearchOnboardingView(
                    icon: "photo.on.rectangle.angled",
                    title: settings.localized("photo_search_onboarding_title"),
                    description: settings.localized("photo_search_onboarding_desc")
                ) {
                    Button { Task { await model.useAllPhotos() } } label: {
                        Text(settings.localized("photo_search_all_photos")).frame(maxWidth: .infinity).frame(height: 50)
                    }
                    .foregroundStyle(.white)
                    .liquidGlassPrimaryButton(cornerRadius: 12)
                    PhotosPicker(selection: $pickerItems, matching: .any(of: [.images, .videos])) {
                        Text(settings.localized("photo_search_select_photos"))
                            .frame(maxWidth: .infinity).frame(height: 50)
                            .foregroundStyle(.white)
                            .background(RoundedRectangle(cornerRadius: 12).fill(Color.white.opacity(0.08)))
                            .overlay(RoundedRectangle(cornerRadius: 12).stroke(Color.white.opacity(0.16), lineWidth: 1))
                    }
                    if model.libraryAccessDenied {
                        Text(settings.localized("photo_search_permission_denied"))
                            .font(.caption).foregroundStyle(.white.opacity(0.7))
                    }
                }
            } else {
                mainView
            }
        }
        .navigationTitle(settings.localized("feature_photo_search"))
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
                countText: String(format: settings.localized("photo_search_count"), model.progress.processed, model.items.count),
                onClearAll: { model.clearAll() }
            ) {
                PhotosPicker(selection: $pickerItems, matching: .any(of: [.images, .videos])) {
                    Text(settings.localized("photo_search_select_photos"))
                        .frame(maxWidth: .infinity)
                        .frame(height: 50)
                }
                .liquidGlassPrimaryButton(cornerRadius: 12)
                if model.source != .allPhotos {
                    Button { Task { await model.useAllPhotos() } } label: {
                        Text(settings.localized("photo_search_all_photos"))
                            .frame(maxWidth: .infinity)
                            .frame(height: 50)
                    }
                    .liquidGlassPrimaryButton(cornerRadius: 12)
                }
            }
            .environmentObject(settings)
        }
        .onChange(of: pickerItems) { _, items in
            guard !items.isEmpty else { return }
            pickerItems = []
            Task { await model.importPhotos(items) }
        }
        .task(id: selectedModel?.id) { await model.start(model: selectedModel) }
        // Presenting the full-screen viewer also fires onDisappear; keep the engine for that.
        .onDisappear { if viewing == nil { model.stop() } }
        .fullScreenCover(item: $viewing) { item in
            PhotoSearchViewer(item: item, model: model) {
                viewing = nil
                model.findSimilar(item)
            } onClose: {
                viewing = nil
            }
            .environmentObject(settings)
        }
    }

    private var mainView: some View {
        let shown = model.results?.map(\.item) ?? model.items
        return VStack(alignment: .leading, spacing: 14) {
            MediaSearchField(
                text: Binding(get: { model.query }, set: { model.updateQuery($0) }),
                placeholder: settings.localized("photo_search_hint"),
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
            if let seed = model.similarTo {
                HStack(spacing: 12) {
                    PhotoSearchThumbnail(item: seed, model: model).frame(width: 48, height: 48)
                    Text(settings.localized("photo_search_similar_results")).font(.subheadline.bold()).foregroundStyle(.white)
                    Spacer()
                    Button { model.clearSearch() } label: { Image(systemName: "xmark.circle.fill").foregroundStyle(.white.opacity(0.7)) }
                }
            }
            if model.results != nil && !model.progress.isComplete {
                Text(settings.localized("media_search_incomplete_warning")).font(.caption).foregroundStyle(.white.opacity(0.7))
            }
            if model.source == .allPhotos && model.libraryAccessDenied {
                VStack(alignment: .leading, spacing: 8) {
                    Text(settings.localized("photo_search_permission_denied")).font(.subheadline).foregroundStyle(.white)
                    Button(settings.localized("media_search_grant_access")) {
                        if let url = URL(string: UIApplication.openSettingsURLString) { UIApplication.shared.open(url) }
                    }
                    .tint(ApolloPalette.accentStrong)
                }
            }
            if model.results?.isEmpty == true && !model.isSearching {
                Text(settings.localized("media_search_no_results")).font(.subheadline).foregroundStyle(.white.opacity(0.8))
            }
            ScrollView {
                LazyVGrid(columns: [GridItem(.adaptive(minimum: 104), spacing: 4)], spacing: 4) {
                    ForEach(shown) { item in
                        PhotoSearchThumbnail(item: item, model: model)
                            .onTapGesture { viewing = item }
                    }
                }
            }
            .scrollDismissesKeyboard(.interactively)
        }
        .padding(16)
    }
}

private struct PhotoSearchThumbnail: View {
    let item: PhotoSearchModel.Item
    @ObservedObject var model: PhotoSearchModel
    @State private var image: UIImage?

    var body: some View {
        Color.white.opacity(0.06)
            .aspectRatio(1, contentMode: .fit)
            .overlay {
                if let image {
                    Image(uiImage: image).resizable().scaledToFill()
                }
            }
            .overlay {
                if item.isVideo {
                    Image(systemName: "play.circle.fill")
                        .font(.title2)
                        .foregroundStyle(.white)
                        .shadow(radius: 4)
                }
            }
            .clipShape(RoundedRectangle(cornerRadius: 8))
            .contentShape(RoundedRectangle(cornerRadius: 8))
            .task(id: item.id) {
                image = await model.loadImage(item, targetSize: CGSize(width: 300, height: 300))
            }
    }
}

private struct PhotoSearchViewer: View {
    let item: PhotoSearchModel.Item
    @ObservedObject var model: PhotoSearchModel
    let onFindSimilar: () -> Void
    let onClose: () -> Void
    @EnvironmentObject var settings: AppSettings
    @State private var image: UIImage?
    @State private var player: AVPlayer?

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            if item.isVideo {
                if let player {
                    VideoPlayer(player: player)
                } else {
                    ProgressView().tint(.white)
                }
            } else if let image {
                Image(uiImage: image).resizable().scaledToFit()
            } else {
                ProgressView().tint(.white)
            }
            VStack {
                HStack {
                    Spacer()
                    Button(action: onClose) {
                        Image(systemName: "xmark").font(.title3.bold()).foregroundStyle(.white).padding()
                    }
                }
                Spacer()
                Button(action: onFindSimilar) {
                    Label(settings.localized("photo_search_find_similar"), systemImage: "sparkle.magnifyingglass")
                        .padding(.horizontal, 20).frame(height: 50)
                }
                .foregroundStyle(.white)
                .liquidGlassPrimaryButton(cornerRadius: 25)
                .padding(.bottom, 32)
            }
        }
        .task {
            if item.isVideo {
                if let url = await model.playbackURL(item) {
                    let next = AVPlayer(url: url)
                    player = next
                    next.play()
                }
            } else {
                image = await model.loadImage(item, targetSize: CGSize(width: 2048, height: 2048), fast: false)
            }
        }
        .onDisappear { player?.pause() }
    }
}

private struct PickedMovie: Transferable {
    let url: URL

    static var transferRepresentation: some TransferRepresentation {
        FileRepresentation(contentType: .movie) { movie in
            SentTransferredFile(movie.url)
        } importing: { received in
            let ext = received.file.pathExtension.isEmpty ? "mov" : received.file.pathExtension
            let dest = FileManager.default.temporaryDirectory.appendingPathComponent("\(UUID().uuidString).\(ext)")
            try FileManager.default.copyItem(at: received.file, to: dest)
            return PickedMovie(url: dest)
        }
    }
}

// MARK: - Shared Photo / Audio Search views

/// Same layout as the other features' "download a model first" state.
struct MediaSearchGateView: View {
    let icon: String
    let onNavigateToModels: () -> Void
    @EnvironmentObject var settings: AppSettings

    var body: some View {
        VStack(spacing: 20) {
            Image(systemName: icon)
                .font(.system(size: 56, weight: .semibold))
                .foregroundStyle(.secondary)
            Text(settings.localized("media_search_download_model"))
                .font(.title3.bold())
                .multilineTextAlignment(.center)
            Text(settings.localized("media_search_download_model_desc"))
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
            Button { onNavigateToModels() } label: {
                Text(settings.localized("download_models"))
                    .frame(maxWidth: .infinity)
                    .frame(height: 52)
            }
            .foregroundStyle(.white)
            .liquidGlassPrimaryButton(cornerRadius: 12)
            .padding(.horizontal, 32)
        }
        .padding()
    }
}

struct MediaSearchOnboardingView<Actions: View>: View {
    let icon: String
    let title: String
    let description: String
    @ViewBuilder let actions: () -> Actions

    var body: some View {
        ScrollView {
            VStack(spacing: 16) {
                Image(systemName: icon)
                    .font(.system(size: 52, weight: .semibold))
                    .foregroundStyle(ApolloPalette.accentStrong)
                Text(title)
                    .font(.title3.bold())
                    .foregroundStyle(.white)
                    .multilineTextAlignment(.center)
                Text(description)
                    .font(.subheadline)
                    .foregroundStyle(.white.opacity(0.75))
                    .multilineTextAlignment(.center)
                VStack(spacing: 12, content: actions).padding(.top, 8)
            }
            .padding(24)
            .background(.ultraThinMaterial)
            .clipShape(RoundedRectangle(cornerRadius: 20))
            .overlay(RoundedRectangle(cornerRadius: 20).stroke(Color.white.opacity(0.12), lineWidth: 1))
            .padding(20)
            .padding(.top, 40)
        }
    }
}

struct MediaSearchField: View {
    @Binding var text: String
    let placeholder: String
    let isSearching: Bool

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: "magnifyingglass").foregroundStyle(.white.opacity(0.6))
            TextField(placeholder, text: $text)
                .foregroundStyle(.white)
                .submitLabel(.search)
                .autocorrectionDisabled()
            if isSearching {
                ProgressView().tint(.white)
            } else if !text.isEmpty {
                Button { text = "" } label: { Image(systemName: "xmark.circle.fill").foregroundStyle(.white.opacity(0.6)) }
            }
        }
        .padding(.horizontal, 14)
        .frame(height: 48)
        .background(.ultraThinMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 12))
        .overlay(RoundedRectangle(cornerRadius: 12).stroke(Color.white.opacity(0.16), lineWidth: 1))
    }
}

struct MediaSearchStatusCard: View {
    let isLoadingModel: Bool
    let modelError: Bool
    let progress: MediaIndexingProgress
    let isPaused: Bool
    let onPause: () -> Void
    let onResume: () -> Void
    let onRetry: () -> Void
    @EnvironmentObject var settings: AppSettings

    var body: some View {
        if isLoadingModel || modelError || !progress.isComplete {
            VStack(alignment: .leading, spacing: 8) {
                if isLoadingModel {
                    Text(settings.localized("media_search_loading_model")).font(.subheadline).foregroundStyle(.white)
                    ProgressView().progressViewStyle(.linear).tint(ApolloPalette.accentStrong)
                } else if modelError {
                    HStack {
                        Text(settings.localized("media_search_model_failed")).font(.subheadline).foregroundStyle(.red)
                        Spacer()
                        Button(settings.localized("retry"), action: onRetry).tint(ApolloPalette.accentStrong)
                    }
                } else {
                    HStack {
                        VStack(alignment: .leading, spacing: 2) {
                            Text(settings.localized(isPaused ? "media_search_paused" : "media_search_analyzing"))
                                .font(.subheadline.bold()).foregroundStyle(.white)
                            Text(String(format: settings.localized("media_search_progress"), progress.processed, progress.total, progress.percent))
                                .font(.caption).foregroundStyle(.white.opacity(0.7))
                        }
                        Spacer()
                        Button(action: isPaused ? onResume : onPause) {
                            Label(settings.localized(isPaused ? "media_search_resume" : "media_search_pause"),
                                  systemImage: isPaused ? "play.fill" : "pause.fill")
                                .font(.caption.bold())
                        }
                        .tint(ApolloPalette.accentStrong)
                    }
                    ProgressView(value: Double(progress.percent), total: 100).tint(ApolloPalette.accentStrong)
                }
            }
            .padding(14)
            .background(.ultraThinMaterial)
            .clipShape(RoundedRectangle(cornerRadius: 14))
        }
    }
}

/// Settings sheet opened from the top-right icon: model picker + library management.
struct MediaSearchSettingsSheet<LibraryActions: View>: View {
    @Binding var selectedModelId: String
    let availableModels: [AIModel]
    let countText: String
    let onClearAll: () -> Void
    @ViewBuilder let libraryActions: () -> LibraryActions

    @EnvironmentObject var settings: AppSettings
    @Environment(\.dismiss) private var dismiss
    @State private var confirmClear = false

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 20) {
                    VStack(alignment: .leading, spacing: 12) {
                        Text(settings.localized("select_model")).font(.headline).foregroundStyle(.white)
                        Picker(settings.localized("select_model"), selection: $selectedModelId) {
                            ForEach(availableModels) { model in
                                Text(model.name).tag(model.id)
                            }
                        }
                        .pickerStyle(.menu)
                        .tint(ApolloPalette.accentStrong)
                        .frame(maxWidth: .infinity, alignment: .leading)
                    }
                    .padding()
                    .background(.ultraThinMaterial)
                    .clipShape(RoundedRectangle(cornerRadius: 14))
                    .overlay(
                        RoundedRectangle(cornerRadius: 14)
                            .stroke(Color.white.opacity(0.14), lineWidth: 1)
                    )

                    VStack(alignment: .leading, spacing: 12) {
                        Text(settings.localized("media_search_library")).font(.headline).foregroundStyle(.white)
                        Text(countText).font(.caption).foregroundStyle(.white.opacity(0.7))
                        libraryActions()
                        Button(role: .destructive) { confirmClear = true } label: {
                            Text(settings.localized("media_search_clear"))
                                .frame(maxWidth: .infinity)
                                .frame(height: 50)
                        }
                        .background(
                            RoundedRectangle(cornerRadius: 12)
                                .fill(ApolloPalette.destructive.opacity(0.10))
                        )
                        .overlay(
                            RoundedRectangle(cornerRadius: 12)
                                .stroke(ApolloPalette.destructive.opacity(0.9), lineWidth: 1)
                        )
                        .foregroundStyle(ApolloPalette.destructive.opacity(0.98))
                    }
                    .padding()
                    .background(.ultraThinMaterial)
                    .clipShape(RoundedRectangle(cornerRadius: 14))
                    .overlay(
                        RoundedRectangle(cornerRadius: 14)
                            .stroke(Color.white.opacity(0.14), lineWidth: 1)
                    )
                }
                .padding()
            }
            .apolloScreenBackground()
            .navigationTitle(settings.localized("feature_settings_title"))
            .navigationBarTitleDisplayMode(.inline)
            .apolloNavigationBackground()
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button(settings.localized("done")) { dismiss() }
                }
            }
            .confirmationDialog(settings.localized("media_search_clear"), isPresented: $confirmClear, titleVisibility: .visible) {
                Button(settings.localized("media_search_clear"), role: .destructive) {
                    onClearAll()
                    dismiss()
                }
                Button(settings.localized("cancel"), role: .cancel) {}
            }
        }
    }
}
