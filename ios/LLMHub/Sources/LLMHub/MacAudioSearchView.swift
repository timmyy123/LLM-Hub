//
//  MacAudioSearchView.swift
//  LLMHub
//
//  Native macOS Audio Search. Same model (`AudioSearchModel`), player
//  (`AudioMomentPlayer`), persisted keys and localized strings as
//  `AudioSearchScreen` on iOS.
//

#if os(macOS)
import SwiftUI
import UniformTypeIdentifiers

struct MacAudioSearchView: View {
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
                MacMediaSearchGate(icon: "waveform.badge.magnifyingglass", onNavigateToModels: onNavigateToModels)
            } else if model.items.isEmpty {
                MacMediaSearchOnboarding(
                    icon: "waveform.badge.magnifyingglass",
                    title: settings.localized("audio_search_onboarding_title"),
                    description: settings.localized("audio_search_onboarding_desc")
                ) {
                    Button(settings.localized("audio_search_import")) { showImporter = true }
                        .buttonStyle(.borderedProminent)
                    Text(settings.localized("audio_search_voice_memos"))
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)
                }
            } else {
                mainView
            }
        }
        .navigationTitle(settings.localized("feature_audio_search"))
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
                countText: String(format: settings.localized("audio_search_count"), model.progress.processed, model.items.count),
                onClearAll: {
                    player.stop()
                    model.clearAll()
                },
                onDismiss: { showSettings = false }
            ) {
                Button {
                    showSettings = false
                    showImporter = true
                } label: {
                    Text(settings.localized("audio_search_import"))
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
            }
            .inspectorColumnWidth(min: 260, ideal: 300, max: 400)
        }
        .fileImporter(isPresented: $showImporter, allowedContentTypes: AudioSearchModel.importTypes, allowsMultipleSelection: true) { result in
            if case .success(let urls) = result, !urls.isEmpty {
                model.importFiles(urls)
            }
        }
        .onReceive(NotificationCenter.default.publisher(for: .audioSearchLibraryChanged)) { _ in
            model.reloadFromDisk()
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
                MacMediaSearchStatusView(
                    isLoadingModel: model.isLoadingModel,
                    modelError: model.modelError,
                    progress: model.progress,
                    isPaused: model.isPaused,
                    onPause: model.pause,
                    onResume: model.resume,
                    onRetry: model.retryModel
                )
                if model.results != nil && !model.progress.isComplete {
                    Text(settings.localized("media_search_incomplete_warning"))
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                if let results = model.results {
                    if results.isEmpty && !model.isSearching {
                        Text(settings.localized("media_search_no_results"))
                            .foregroundStyle(.secondary)
                    }
                    ForEach(results) { match in
                        MacAudioSearchCard(item: match.item, match: match, url: model.fileURL(match.item), player: player)
                    }
                } else {
                    ForEach(model.items) { item in
                        MacAudioSearchCard(item: item, match: nil, url: model.fileURL(item), player: player)
                    }
                }
            }
            .padding()
        }
        .searchable(
            text: Binding(get: { model.query }, set: { model.updateQuery($0) }),
            placement: .toolbar,
            prompt: Text(settings.localized("audio_search_hint"))
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

private func macAudioFormatMs(_ ms: Int) -> String {
    let s = max(0, ms / 1000)
    return String(format: "%d:%02d", s / 60, s % 60)
}

private struct MacAudioSearchCard: View {
    let item: AudioSearchModel.Item
    let match: AudioSearchModel.Match?
    let url: URL
    @ObservedObject var player: AudioMomentPlayer
    @EnvironmentObject var settings: AppSettings

    var body: some View {
        let isCurrent = player.playingId == item.id
        let total = isCurrent && player.durationMs > 0 ? player.durationMs : item.durationMs
        GroupBox {
            VStack(alignment: .leading, spacing: 10) {
                HStack(spacing: 10) {
                    Button { player.toggle(item, url: url) } label: {
                        Image(systemName: isCurrent && player.isPlaying ? "pause.fill" : "play.fill")
                            .frame(width: 18, height: 18)
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.large)
                    VStack(alignment: .leading, spacing: 2) {
                        Text(item.name).font(.headline).lineLimit(1)
                        Text(isCurrent ? "\(macAudioFormatMs(player.positionMs)) / \(macAudioFormatMs(total))" : macAudioFormatMs(total))
                            .font(.caption.monospacedDigit())
                            .foregroundStyle(.secondary)
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
                                    Label(String(format: settings.localized("audio_search_moment"), macAudioFormatMs(moment.startMs)), systemImage: "play.circle")
                                }
                                .buttonStyle(.bordered)
                            }
                        }
                    }
                }
            }
            .padding(4)
        }
    }
}
#endif
