//
//  MacVideoMomentView.swift
//  LLMHub
//
//  Native macOS Video Moment Finder. Same model (`VideoMomentModel`), player
//  (`MomentPlayer` / `MomentPlayerView`), persisted keys and localized strings
//  as `VideoMomentScreen` on iOS.
//

#if os(macOS)
import AVKit
import PhotosUI
import SwiftUI

struct MacVideoMomentView: View {
    let onNavigateToModels: () -> Void

    @EnvironmentObject var settings: AppSettings
    @StateObject private var model = VideoMomentModel()
    @StateObject private var playback = MomentPlayer()
    @AppStorage("video_moment_model_id") private var selectedModelId = ""
    @State private var showSettings = false
    @State private var pickedVideo: PhotosPickerItem?
    @State private var selectedMoment: VideoMomentModel.Moment?
    @State private var editing: VideoMomentModel.Moment?

    private var downloadedModels: [AIModel] { MediaSearchConfig.downloadedModels() }
    private var selectedModel: AIModel? {
        downloadedModels.first { $0.id == selectedModelId } ?? downloadedModels.first
    }

    /// iOS back button behavior while a video is open: close it and return to the grid.
    private func closeOpenVideo() {
        playback.stop()
        model.closeVideo()
    }

    var body: some View {
        Group {
            if downloadedModels.isEmpty {
                MacMediaSearchGate(icon: "film.stack", onNavigateToModels: onNavigateToModels)
            } else if model.videos.isEmpty && !model.isImporting && model.progress.total == 0 {
                MacMediaSearchOnboarding(
                    icon: "film.stack",
                    title: settings.localized("video_moment_onboarding_title"),
                    description: settings.localized("video_moment_onboarding_desc")
                ) {
                    PhotosPicker(settings.localized("video_moment_pick"), selection: $pickedVideo, matching: .videos)
                        .buttonStyle(.borderedProminent)
                }
            } else {
                main
            }
        }
        .navigationTitle(model.openVideo?.name ?? settings.localized("feature_video_moment"))
        .toolbar {
            if model.openVideo != nil {
                ToolbarItem(placement: .navigation) {
                    Button { closeOpenVideo() } label: { Image(systemName: "arrow.left") }
                }
            }
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
                countText: String(format: settings.localized("media_search_progress"), model.progress.processed, model.videos.count, model.progress.percent),
                onClearAll: { model.clearAll() },
                onDismiss: { showSettings = false }
            ) {
                PhotosPicker(settings.localized("video_moment_pick"), selection: $pickedVideo, matching: .videos)
                    .buttonStyle(.borderedProminent)
            }
            .inspectorColumnWidth(min: 260, ideal: 300, max: 400)
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
            MacClipEditSheet(moment: moment) { editing = nil }
                .environmentObject(settings)
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
        let pickTitle = settings.localized("video_moment_pick")
        return ScrollView {
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 160), spacing: 16)], spacing: 16) {
                PhotosPicker(selection: $pickedVideo, matching: .videos) {
                    VStack(spacing: 8) {
                        ZStack {
                            RoundedRectangle(cornerRadius: 12).fill(Color(nsColor: .quaternaryLabelColor))
                            Image(systemName: "plus").font(.largeTitle)
                        }
                        .aspectRatio(1, contentMode: .fit)
                        Text(pickTitle).font(.caption)
                    }
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                if model.isImporting {
                    VStack(spacing: 8) {
                        ZStack {
                            RoundedRectangle(cornerRadius: 12).fill(Color(nsColor: .quaternaryLabelColor))
                            ProgressView()
                        }
                        .aspectRatio(1, contentMode: .fit)
                        Text(settings.localized("video_moment_pick")).font(.caption)
                    }
                }
                ForEach(model.videos) { video in
                    Button { model.open(video) } label: {
                        VStack(spacing: 8) {
                            MacVideoPoster(video: video, clock: macMomentDurationClock(video.durationMs))
                            Text(video.name).font(.caption).lineLimit(1)
                        }
                        .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding()
        }
    }

    private func videoSearch(_ video: VideoMomentModel.Video) -> some View {
        let results = model.results ?? []
        let intervals = mergedIntervals(results)
        return VStack(spacing: 0) {
            VStack(alignment: .leading, spacing: 8) {
                MacMediaSearchStatusView(
                    isLoadingModel: model.isLoadingModel,
                    modelError: model.modelError,
                    progress: model.progress,
                    isPaused: model.isPaused,
                    onPause: model.pause,
                    onResume: model.resume,
                    onRetry: model.retryModel
                )
                if model.results?.isEmpty == true && !model.isSearching {
                    Text(settings.localized("media_search_no_results")).foregroundStyle(.secondary)
                }
                if !results.isEmpty {
                    Text(settings.localized("video_moment_top")).font(.caption).foregroundStyle(.secondary)
                    HStack(alignment: .center, spacing: 16) {
                        ScrollView(.horizontal, showsIndicators: false) {
                            HStack(spacing: 12) {
                                ForEach(results) { moment in
                                    MacMomentThumb(moment: moment, selected: selectedMoment?.id == moment.id) {
                                        selectedMoment = moment
                                        let end = intervals.first { $0.ids.contains(moment.id) }?.endMs ?? moment.endMs
                                        playback.playClip(startMs: moment.startMs, endMs: end)
                                    }
                                }
                            }
                            .padding(4)
                        }
                        if selectedMoment != nil {
                            Button { editing = selectedMoment } label: {
                                Label(settings.localized("video_moment_save_edit"), systemImage: "pencil")
                            }
                            .buttonStyle(.bordered)
                            .controlSize(.large)
                        }
                    }
                }
            }
            .padding(.horizontal)
            .padding(.vertical, 8)
            .frame(maxWidth: .infinity, alignment: .leading)

            MomentPlayerView(player: playback.player)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .background(Color.black)

            Divider()

            MacMomentTransport(
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
        .searchable(
            text: Binding(get: { model.query }, set: { model.updateQuery($0) }),
            placement: .toolbar,
            prompt: Text(settings.localized("video_moment_hint"))
        )
        .onSubmit(of: .search) { model.submitSearch() }
        .toolbar {
            if model.isSearching {
                ToolbarItem(placement: .primaryAction) {
                    ProgressView().controlSize(.small)
                }
            }
        }
    }
}

/// Same format as the iOS grid's `clock(_:)`.
private func macMomentDurationClock(_ ms: Int) -> String {
    let s = max(0, ms / 1000)
    return String(format: "%d:%02d", s / 60, s % 60)
}

/// Same format as the iOS `thumbClock(_:)`.
private func macMomentThumbClock(_ ms: Int) -> String {
    let s = max(0, ms / 1000)
    return String(format: "%02d:%02d", s / 60, s % 60)
}

private struct MacVideoPoster: View {
    let video: VideoMomentModel.Video
    let clock: String
    @State private var image: UIImage?

    var body: some View {
        Color(nsColor: .quaternaryLabelColor)
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
                    .shadow(radius: 2)
                    .padding(8)
            }
            .clipped()
            .clipShape(RoundedRectangle(cornerRadius: 12))
            .task(id: video.id) {
                let url = VideoMomentModel.playbackURL(video)
                if let data = await videoFrameJPEG(url: url, timeMs: 500) {
                    image = UIImage(data: data)
                }
            }
    }
}

private struct MacMomentThumb: View {
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
                        Color(nsColor: .quaternaryLabelColor)
                    }
                }
                VStack {
                    Text(String(format: "%.2f", moment.score)).font(.caption2).foregroundStyle(.white).padding(.top, 4)
                    Spacer()
                    Text("\(macMomentThumbClock(moment.startMs)) - \(macMomentThumbClock(moment.endMs))")
                        .font(.caption2.bold())
                        .foregroundStyle(.white)
                        .padding(.bottom, 4)
                }
                .shadow(radius: 2)
            }
            .frame(width: 84, height: 120)
            .clipShape(RoundedRectangle(cornerRadius: 8))
            .overlay(RoundedRectangle(cornerRadius: 8).stroke(selected ? Color.accentColor : Color.white, lineWidth: selected ? 4 : 2))
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .task(id: moment.id) {
            let url = VideoMomentModel.playbackURL(moment.video)
            let start = moment.startMs
            if let data = await videoFrameJPEG(url: url, timeMs: start) {
                image = UIImage(data: data)
            }
        }
    }
}

/// Mac counterpart of the iOS `ClipEditSheet` (same controls; Save closes it).
private struct MacClipEditSheet: View {
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
            Text(macMomentThumbClock(Int(start))).monospacedDigit()
            Slider(value: $start, in: 0...Double(max(moment.video.durationMs, 1)))
            Text(macMomentThumbClock(Int(end))).monospacedDigit()
            Slider(value: $end, in: 0...Double(max(moment.video.durationMs, 1)))
            HStack {
                Spacer()
                Button(settings.localized("video_moment_save"), action: onDone)
                    .buttonStyle(.borderedProminent)
                    .controlSize(.large)
                    .keyboardShortcut(.defaultAction)
            }
        }
        .padding(24)
        .frame(width: 460)
        // iOS dismisses this sheet by swiping down; Escape does the same here.
        .onExitCommand(perform: onDone)
    }
}

/// Mac counterpart of the iOS `MomentTransport`: play/pause, time, timeline with
/// result markers (scrub by click or drag) and mute.
private struct MacMomentTransport: View {
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
        HStack(spacing: 14) {
            Button(action: onToggle) {
                Image(systemName: isPlaying ? "pause.fill" : "play.fill")
                    .frame(width: 20, height: 20)
            }
            .buttonStyle(.borderless)
            .keyboardShortcut(.space, modifiers: [])

            Text("\(macMomentThumbClock(Int(playhead * Double(durationMs)))) / \(macMomentThumbClock(durationMs))")
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)

            GeometryReader { geo in
                let width = max(geo.size.width, 1)
                ZStack(alignment: .leading) {
                    Capsule().fill(Color.primary.opacity(0.25)).frame(height: 6)
                    Capsule().fill(Color.primary).frame(width: width * playhead, height: 6)
                    ForEach(intervals) { interval in
                        let start = CGFloat(interval.startMs) / CGFloat(max(durationMs, 1))
                        let end = CGFloat(interval.endMs) / CGFloat(max(durationMs, 1))
                        let selected = selectedId.map { interval.ids.contains($0) } ?? false
                        let markerWidth = max(6, width * (end - start))
                        let x = min(width - markerWidth, width * start)
                        RoundedRectangle(cornerRadius: 3)
                            .fill(selected ? Color.accentColor : Color.primary.opacity(0.85))
                            .frame(width: markerWidth, height: selected ? 18 : 14)
                            .offset(x: x)
                    }
                    RoundedRectangle(cornerRadius: 2)
                        .fill(Color.primary)
                        .frame(width: 4, height: 22)
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
            .frame(height: 32)

            Button(action: onMute) {
                Image(systemName: isMuted ? "speaker.slash.fill" : "speaker.wave.2.fill")
                    .frame(width: 20, height: 20)
            }
            .buttonStyle(.borderless)
        }
        .padding(.horizontal)
        .padding(.vertical, 8)
    }
}
#endif
