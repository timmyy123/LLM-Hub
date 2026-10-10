//
//  MacPhotoSearchView.swift
//  LLMHub
//
//  Native macOS Instant Media Search. Same model (`PhotoSearchModel`), persisted
//  keys, library access flow and localized strings as `PhotoSearchScreen` on iOS.
//  Also hosts the Mac building blocks shared by the media search screens
//  (gate, onboarding, status and settings inspector).
//

#if os(macOS)
import AVKit
import Photos
import PhotosUI
import SwiftUI

struct MacPhotoSearchView: View {
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
                MacMediaSearchGate(icon: "photo.on.rectangle.angled", onNavigateToModels: onNavigateToModels)
            } else if model.source == .none {
                MacMediaSearchOnboarding(
                    icon: "photo.on.rectangle.angled",
                    title: settings.localized("photo_search_onboarding_title"),
                    description: settings.localized("photo_search_onboarding_desc")
                ) {
                    Button(settings.localized("photo_search_all_photos")) {
                        Task { await model.useAllPhotos() }
                    }
                    .buttonStyle(.borderedProminent)
                    PhotosPicker(settings.localized("photo_search_select_photos"), selection: $pickerItems, matching: .any(of: [.images, .videos]))
                        .buttonStyle(.bordered)
                    if model.libraryAccessDenied {
                        Text(settings.localized("photo_search_permission_denied"))
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
            } else {
                mainView
            }
        }
        .navigationTitle(settings.localized("feature_photo_search"))
        .toolbar {
            if !downloadedModels.isEmpty {
                ToolbarItem(placement: .primaryAction) {
                    Button { showSettings.toggle() } label: {
                        Label(settings.localized("feature_settings_title"), systemImage: "slider.horizontal.3")
                    }
                }
            }
        }
        .inspector(isPresented: $showSettings) {
            MacMediaSearchInspector(
                selectedModelId: Binding(get: { selectedModel?.id ?? "" }, set: { selectedModelId = $0 }),
                availableModels: downloadedModels,
                countText: String(format: settings.localized("photo_search_count"), model.progress.processed, model.items.count),
                onClearAll: { model.clearAll() },
                onDismiss: { showSettings = false }
            ) {
                PhotosPicker(settings.localized("photo_search_select_photos"), selection: $pickerItems, matching: .any(of: [.images, .videos]))
                    .buttonStyle(.borderedProminent)
                if model.source != .allPhotos {
                    Button { Task { await model.useAllPhotos() } } label: {
                        Text(settings.localized("photo_search_all_photos"))
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                }
            }
            .inspectorColumnWidth(min: 260, ideal: 300, max: 400)
        }
        .onChange(of: pickerItems) { _, items in
            guard !items.isEmpty else { return }
            pickerItems = []
            Task { await model.importPhotos(items) }
        }
        .task(id: selectedModel?.id) { await model.start(model: selectedModel) }
        // Same guard as iOS: keep the engine while the viewer is presented.
        .onDisappear { if viewing == nil { model.stop() } }
        .sheet(item: $viewing) { item in
            MacPhotoSearchViewer(item: item, model: model) {
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
        return VStack(alignment: .leading, spacing: 12) {
            MacMediaSearchStatusView(
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
                    MacPhotoSearchThumbnail(item: seed, model: model).frame(width: 48, height: 48)
                    Text(settings.localized("photo_search_similar_results")).font(.headline)
                    Spacer()
                    Button { model.clearSearch() } label: { Image(systemName: "xmark.circle.fill") }
                        .buttonStyle(.borderless)
                }
            }
            if model.results != nil && !model.progress.isComplete {
                Text(settings.localized("media_search_incomplete_warning"))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            if model.source == .allPhotos && model.libraryAccessDenied {
                HStack(spacing: 12) {
                    Text(settings.localized("photo_search_permission_denied"))
                    Button(settings.localized("media_search_grant_access")) {
                        if let url = URL(string: UIApplication.openSettingsURLString) { UIApplication.shared.open(url) }
                    }
                    .buttonStyle(.borderedProminent)
                }
            }
            if model.results?.isEmpty == true && !model.isSearching {
                Text(settings.localized("media_search_no_results"))
                    .foregroundStyle(.secondary)
            }
            ScrollView {
                LazyVGrid(columns: [GridItem(.adaptive(minimum: 140), spacing: 6)], spacing: 6) {
                    ForEach(shown) { item in
                        Button { viewing = item } label: {
                            MacPhotoSearchThumbnail(item: item, model: model)
                        }
                        .buttonStyle(.plain)
                    }
                }
            }
        }
        .padding()
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
        .searchable(
            text: Binding(get: { model.query }, set: { model.updateQuery($0) }),
            placement: .toolbar,
            prompt: Text(settings.localized("photo_search_hint"))
        )
        .toolbar {
            if model.isSearching {
                ToolbarItem(placement: .primaryAction) {
                    ProgressView().controlSize(.small)
                }
            }
        }
    }
}

private struct MacPhotoSearchThumbnail: View {
    let item: PhotoSearchModel.Item
    @ObservedObject var model: PhotoSearchModel
    @State private var image: UIImage?

    var body: some View {
        Color(nsColor: .quaternaryLabelColor)
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
            .clipShape(RoundedRectangle(cornerRadius: 6))
            .contentShape(RoundedRectangle(cornerRadius: 6))
            .task(id: item.id) {
                image = await model.loadImage(item, targetSize: CGSize(width: 300, height: 300))
            }
    }
}

/// Mac counterpart of the iOS full-screen viewer, presented as a sheet.
private struct MacPhotoSearchViewer: View {
    let item: PhotoSearchModel.Item
    @ObservedObject var model: PhotoSearchModel
    let onFindSimilar: () -> Void
    let onClose: () -> Void
    @EnvironmentObject var settings: AppSettings
    @State private var image: UIImage?
    @State private var player: AVPlayer?

    var body: some View {
        VStack(spacing: 0) {
            ZStack {
                Color.black
                if item.isVideo {
                    if let player {
                        VideoPlayer(player: player)
                    } else {
                        ProgressView()
                    }
                } else if let image {
                    Image(uiImage: image).resizable().scaledToFit()
                } else {
                    ProgressView()
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)

            Divider()

            HStack {
                Button(action: onClose) {
                    Image(systemName: "xmark")
                }
                .keyboardShortcut(.cancelAction)
                Spacer()
                Button(action: onFindSimilar) {
                    Label(settings.localized("photo_search_find_similar"), systemImage: "sparkle.magnifyingglass")
                }
                .buttonStyle(.borderedProminent)
                .keyboardShortcut(.defaultAction)
            }
            .controlSize(.large)
            .padding()
        }
        .frame(minWidth: 640, idealWidth: 960, minHeight: 480, idealHeight: 720)
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

// MARK: - Shared Mac media search components

/// Mac counterpart of `MediaSearchGateView` (same strings and action).
struct MacMediaSearchGate: View {
    let icon: String
    let onNavigateToModels: () -> Void
    @EnvironmentObject var settings: AppSettings

    var body: some View {
        ContentUnavailableView {
            Label(settings.localized("media_search_download_model"), systemImage: icon)
        } description: {
            Text(settings.localized("media_search_download_model_desc"))
        } actions: {
            Button(settings.localized("download_models"), action: onNavigateToModels)
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
        }
    }
}

/// Mac counterpart of `MediaSearchOnboardingView`.
struct MacMediaSearchOnboarding<Actions: View>: View {
    let icon: String
    let title: String
    let description: String
    @ViewBuilder let actions: () -> Actions

    var body: some View {
        ContentUnavailableView {
            Label(title, systemImage: icon)
        } description: {
            Text(description)
        } actions: {
            VStack(spacing: 10, content: actions)
                .controlSize(.large)
        }
    }
}

/// Mac counterpart of `MediaSearchStatusCard` (same states, strings and actions).
struct MacMediaSearchStatusView: View {
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
            GroupBox {
                VStack(alignment: .leading, spacing: 8) {
                    if isLoadingModel {
                        Text(settings.localized("media_search_loading_model"))
                        ProgressView().progressViewStyle(.linear)
                    } else if modelError {
                        HStack {
                            Text(settings.localized("media_search_model_failed")).foregroundStyle(.red)
                            Spacer()
                            Button(settings.localized("retry"), action: onRetry)
                        }
                    } else {
                        HStack {
                            VStack(alignment: .leading, spacing: 2) {
                                Text(settings.localized(isPaused ? "media_search_paused" : "media_search_analyzing"))
                                    .font(.headline)
                                Text(String(format: settings.localized("media_search_progress"), progress.processed, progress.total, progress.percent))
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                                    .monospacedDigit()
                            }
                            Spacer()
                            Button(action: isPaused ? onResume : onPause) {
                                Label(settings.localized(isPaused ? "media_search_resume" : "media_search_pause"),
                                      systemImage: isPaused ? "play.fill" : "pause.fill")
                            }
                        }
                        ProgressView(value: Double(progress.percent), total: 100)
                    }
                }
                .padding(4)
            }
        }
    }
}

/// Mac counterpart of `MediaSearchSettingsSheet`, shown in an inspector column.
struct MacMediaSearchInspector<LibraryActions: View>: View {
    @Binding var selectedModelId: String
    let availableModels: [AIModel]
    let countText: String
    let onClearAll: () -> Void
    let onDismiss: () -> Void
    @ViewBuilder let libraryActions: () -> LibraryActions

    @EnvironmentObject var settings: AppSettings
    @State private var confirmClear = false

    var body: some View {
        Form {
            Section(settings.localized("select_model")) {
                Picker(settings.localized("select_model"), selection: $selectedModelId) {
                    ForEach(availableModels) { model in
                        Text(model.name).tag(model.id)
                    }
                }
                .labelsHidden()
            }
            Section(settings.localized("media_search_library")) {
                Text(countText)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                libraryActions()
                Button(role: .destructive) { confirmClear = true } label: {
                    Text(settings.localized("media_search_clear"))
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
            }
        }
        .formStyle(.grouped)
        .confirmationDialog(settings.localized("media_search_clear"), isPresented: $confirmClear, titleVisibility: .visible) {
            Button(settings.localized("media_search_clear"), role: .destructive) {
                onClearAll()
                onDismiss()
            }
            Button(settings.localized("cancel"), role: .cancel) {}
        }
    }
}
#endif
