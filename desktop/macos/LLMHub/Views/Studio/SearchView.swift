import AppKit
import SwiftUI
import UniformTypeIdentifiers

/// Studio › Search: photos and audio found by description or by example with EmbeddingGemma 2 on this Mac, as in the
/// Android app's Photo Search and Audio Search. The folder and index on the left, the query and results on the right.
struct SearchView: View {
    var body: some View {
        HStack(spacing: 0) {
            SearchSidebar()
                .frame(width: 320)
            Divider()
            SearchResults()
        }
    }
}

private struct SearchSidebar: View {
    @Environment(AppModel.self) private var app

    var body: some View {
        @Bindable var search = app.studio.search
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                GlassEffectContainer(spacing: 4) {
                    HStack(spacing: 4) {
                        modeTab("photos", L.s("SearchPhotos"), "photo")
                        modeTab("audio", L.s("SearchAudio"), "waveform")
                    }
                }
                .accessibilityIdentifier("SearchModes")

                if search.hasModel {
                    VStack(alignment: .leading, spacing: 6) {
                        FieldLabel(L.s("SearchModel"))
                        Picker("", selection: Binding(get: { search.selectedModel?.id ?? "" }, set: { search.selectedModelId = $0 })) {
                            ForEach(search.models) { Text($0.name).tag($0.id) }
                        }
                        .labelsHidden()
                        .controlSize(.large)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .disabled(search.isIndexing)
                        .accessibilityIdentifier("SearchModelPicker")
                    }
                } else {
                    Text(L.s("SearchGetModel")).font(.system(size: 13)).foregroundStyle(Theme.textMuted)
                        .fixedSize(horizontal: false, vertical: true)
                        .accessibilityIdentifier("SearchGetModel")
                }

                VStack(alignment: .leading, spacing: 6) {
                    FieldLabel(L.s("SearchFolder"))
                    HStack(spacing: 8) {
                        Text(search.folder.replacingOccurrences(of: FileManager.default.homeDirectoryForCurrentUser.path, with: "~"))
                            .font(.system(size: 12))
                            .lineLimit(1)
                            .truncationMode(.middle)
                            .help(search.folder)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .accessibilityIdentifier("SearchFolder")
                        Button(L.s("SearchChange")) { pickFolder() }
                            .buttonStyle(.glass)
                            .disabled(search.isIndexing)
                            .accessibilityIdentifier("SearchChangeFolder")
                    }
                }

                HStack(spacing: 8) {
                    Button {
                        search.startIndexing()
                    } label: {
                        HStack(spacing: 6) {
                            if search.isIndexing { ProgressView().controlSize(.mini) }
                            Text(L.s("SearchIndex")).fontWeight(.semibold)
                        }
                        .frame(maxWidth: .infinity, minHeight: 30)
                    }
                    .buttonStyle(.glassProminent)
                    .disabled(search.isIndexing || !search.hasModel)
                    .accessibilityIdentifier("SearchIndexButton")
                    if search.isIndexing {
                        Button {
                            search.stopIndexing()
                        } label: {
                            Text(L.s("Stop")).frame(minHeight: 30)
                        }
                        .buttonStyle(.glass)
                        .accessibilityIdentifier("SearchStopButton")
                    }
                }
                .controlSize(.large)

                if search.isIndexing {
                    ProgressView(value: search.progress).accessibilityIdentifier("SearchProgress")
                }
                Text(search.indexedText).font(.system(size: 12)).foregroundStyle(Theme.textMuted)
                    .accessibilityIdentifier("SearchIndexed")
                if !search.status.isEmpty {
                    Text(search.status).font(.system(size: 12)).foregroundStyle(Theme.textMuted)
                        .fixedSize(horizontal: false, vertical: true)
                        .textSelection(.enabled)
                        .accessibilityIdentifier("SearchStatus")
                }
                Text(L.s("SearchHelp")).font(.system(size: 12)).foregroundStyle(Theme.textFaint)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .padding(20)
        }
    }

    private func modeTab(_ mode: String, _ title: String, _ icon: String) -> some View {
        let search = app.studio.search
        let selected = search.mode == mode
        return Button {
            search.mode = mode
        } label: {
            Label(title, systemImage: icon)
                .font(.system(size: 13))
                .padding(.horizontal, 14)
                .frame(height: 30)
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .foregroundStyle(selected ? .primary : Theme.textMuted)
        .glassEffect(selected ? .regular.interactive() : .identity, in: .capsule)
        .disabled(search.isIndexing)
        .accessibilityAddTraits(selected ? .isSelected : [])
        .accessibilityIdentifier("SearchMode_\(mode)")
    }

    private func pickFolder() {
        let search = app.studio.search
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.directoryURL = URL(fileURLWithPath: search.folder, isDirectory: true)
        if panel.runModal() == .OK, let url = panel.url { search.setFolder(url.path) }
    }
}

private struct SearchResults: View {
    @Environment(AppModel.self) private var app

    var body: some View {
        @Bindable var search = app.studio.search
        VStack(alignment: .leading, spacing: 14) {
            HStack(spacing: 8) {
                HStack(spacing: 8) {
                    Image(systemName: "magnifyingglass").foregroundStyle(Theme.textMuted)
                    TextField(search.placeholder, text: $search.query)
                        .textFieldStyle(.plain)
                        .font(.system(size: 14))
                        .onSubmit { Task { await search.search() } }
                        .accessibilityIdentifier("SearchQuery")
                }
                .padding(.horizontal, 14)
                .frame(height: 38)
                .glassEffect(.regular, in: .capsule)
                Button {
                    Task { await search.search() }
                } label: {
                    Group {
                        if search.isSearching { ProgressView().controlSize(.small) } else { Image(systemName: "arrow.right") }
                    }
                    .frame(width: 18, height: 18)
                }
                .buttonStyle(.glassProminent)
                .buttonBorderShape(.circle)
                .controlSize(.large)
                .disabled(search.query.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || search.isSearching)
                .help(L.s("StudioSearch"))
                .accessibilityLabel(L.s("StudioSearch"))
                .accessibilityIdentifier("SearchButton")
                if search.isPhotos {
                    Button {
                        pickPhoto()
                    } label: {
                        Label(L.s("SearchByPhoto"), systemImage: "photo.on.rectangle")
                    }
                    .buttonStyle(.glass)
                    .controlSize(.large)
                    .disabled(search.isSearching)
                    .accessibilityIdentifier("SearchByPhotoButton")
                }
            }

            ScrollView {
                if search.isPhotos {
                    LazyVGrid(columns: [GridItem(.adaptive(minimum: 200), spacing: 12)], spacing: 12) {
                        ForEach(search.photos) { PhotoMatchTile(match: $0) }
                    }
                    .padding(2)
                    .accessibilityIdentifier("SearchPhotoResults")
                } else {
                    LazyVStack(spacing: 8) {
                        ForEach(search.audio) { AudioMatchRow(match: $0) }
                    }
                    .accessibilityIdentifier("SearchAudioResults")
                }
            }
        }
        .padding(20)
    }

    private func pickPhoto() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = MediaDecoders.imageExtensions.compactMap { UTType(filenameExtension: $0) }
        panel.allowsMultipleSelection = false
        panel.directoryURL = URL(fileURLWithPath: app.studio.search.photoFolder, isDirectory: true)
        if panel.runModal() == .OK, let url = panel.url { Task { await app.studio.search.searchByPhoto(url) } }
    }
}

private struct PhotoMatchTile: View {
    let match: PhotoMatch

    var body: some View {
        Button {
            NSWorkspace.shared.open(match.url)
        } label: {
            Color.clear
                .aspectRatio(4 / 3, contentMode: .fit)
                .overlay { Thumbnail(url: match.url, maxPixel: 480) }
                .clipShape(.rect(cornerRadius: 12))
                .overlay(alignment: .bottomTrailing) {
                    Text(match.scoreText)
                        .font(.system(size: 11, weight: .medium))
                        .monospacedDigit()
                        .padding(.horizontal, 8)
                        .padding(.vertical, 3)
                        .glassEffect(.regular, in: .capsule)
                        .padding(8)
                        .accessibilityIdentifier("SearchPhotoScore")
                }
                .contentShape(.rect(cornerRadius: 12))
        }
        .buttonStyle(.plain)
        .help(match.name)
        .contextMenu {
            Button(L.s("StudioOpen")) { NSWorkspace.shared.open(match.url) }
            Button(L.s("StudioShowInFinder")) { NSWorkspace.shared.activateFileViewerSelecting([match.url]) }
        }
        .accessibilityLabel("\(match.name) \(match.scoreText)")
        .accessibilityIdentifier("SearchPhotoMatch")
    }
}

private struct AudioMatchRow: View {
    @Environment(AppModel.self) private var app
    let match: AudioMatch

    var body: some View {
        let search = app.studio.search
        HStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 6) {
                Text(match.name).fontWeight(.semibold).lineLimit(1).truncationMode(.middle)
                    .accessibilityIdentifier("SearchAudioName")
                GlassEffectContainer(spacing: 6) {
                    HStack(spacing: 6) {
                        ForEach(match.moments) { moment in
                            let playing = search.playing == "\(match.path)#\(moment.startMs)"
                            Button {
                                Task { await search.play(match, moment) }
                            } label: {
                                Label(moment.label, systemImage: playing ? "stop.fill" : "play.fill")
                                    .font(.system(size: 12))
                                    .monospacedDigit()
                            }
                            .buttonStyle(.glass)
                            .help(L.s("SearchPlay"))
                            .accessibilityIdentifier("SearchMoment")
                        }
                    }
                }
            }
            Spacer()
            Text(match.scoreText).font(.system(size: 12)).foregroundStyle(Theme.textMuted).monospacedDigit()
            Button {
                NSWorkspace.shared.activateFileViewerSelecting([match.url])
            } label: {
                Image(systemName: "folder")
            }
            .buttonStyle(.plain)
            .foregroundStyle(Theme.textMuted)
            .help(L.s("StudioShowInFinder"))
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
        .background(Theme.surfaceAlt.opacity(0.6), in: .rect(cornerRadius: 12))
        .accessibilityIdentifier("SearchAudioMatch")
    }
}
