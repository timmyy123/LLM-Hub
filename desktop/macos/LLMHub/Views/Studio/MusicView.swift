import AppKit
import SwiftUI

/// Studio › Music: a live jam with Magenta RealTime (Music.dc.html): transport and level bars, style sliders that the
/// music follows while it plays, sampling settings, and the saved recordings.
struct MusicView: View {
    @Environment(AppModel.self) private var app

    var body: some View {
        if MusicService.supported {
            ScrollView {
                VStack(alignment: .leading, spacing: 24) {
                    NowPlaying()
                    StylesSection()
                    MusicSettingsSection()
                    RecordingsSection()
                }
                .padding(28)
                .frame(maxWidth: 1100)
                .frame(maxWidth: .infinity)
            }
        } else {
            VStack(spacing: 12) {
                Image(systemName: "music.note").font(.system(size: 32)).foregroundStyle(Theme.textMuted)
                Text(L.s("MusicNeedsAppleSilicon")).multilineTextAlignment(.center).foregroundStyle(.secondary)
                    .accessibilityIdentifier("MusicUnavailable")
            }
            .frame(maxWidth: 560)
        }
    }
}

private struct NowPlaying: View {
    @Environment(AppModel.self) private var app

    var body: some View {
        let music = app.studio.music
        VStack(alignment: .leading, spacing: 18) {
            HStack(spacing: 16) {
                Button {
                    music.toggle()
                } label: {
                    Group {
                        if music.isBusy { ProgressView().controlSize(.small) } else {
                            Image(systemName: music.isPlaying ? "stop.fill" : "play.fill").font(.system(size: 20))
                        }
                    }
                    .frame(width: 40, height: 40)
                }
                .buttonStyle(.glassProminent)
                .buttonBorderShape(.circle)
                .controlSize(.large)
                .disabled(music.isBusy)
                .help(L.s(music.isPlaying ? "MusicStop" : "MusicPlay"))
                .accessibilityLabel(L.s(music.isPlaying ? "MusicStop" : "MusicPlay"))
                .accessibilityIdentifier("MusicPlayButton")

                VStack(alignment: .leading, spacing: 2) {
                    Text(L.s("MusicLiveJam")).font(.system(size: 18, weight: .semibold))
                    Text(subtitle).font(.system(size: 13)).foregroundStyle(Theme.textMuted)
                        .accessibilityIdentifier("MusicSubtitle")
                    if !music.status.isEmpty {
                        Text(music.status).font(.system(size: 12)).foregroundStyle(Theme.textMuted).lineLimit(2).textSelection(.enabled)
                            .accessibilityIdentifier("MusicStatusText")
                    } else if music.isPlaying && music.elapsed > 4 && music.speed > 0 && music.speed < 0.95 {
                        Text(L.s("MusicSlow")).font(.system(size: 12)).foregroundStyle(Theme.warning).lineLimit(2)
                    }
                }
                Spacer()
                Button {
                    music.setRecording(!music.isRecording)
                } label: {
                    HStack(spacing: 8) {
                        Circle().fill(music.isRecording ? Color.red : Theme.textMuted).frame(width: 8, height: 8)
                        Text(L.s(music.isRecording ? "MusicRecording" : "MusicRecord"))
                    }
                }
                .buttonStyle(.glass)
                .tint(music.isRecording ? .red : nil)
                .controlSize(.large)
                .accessibilityIdentifier("MusicRecordButton")
                Button(L.s("MusicExport")) { music.export() }
                    .buttonStyle(.glass)
                    .controlSize(.large)
                    .disabled(music.elapsed <= 0)
                    .accessibilityIdentifier("MusicExportButton")
            }
            LevelBars(levels: music.levels, active: music.isPlaying)
                .frame(height: 96)
        }
        .padding(20)
        .glassEffect(.regular, in: .rect(cornerRadius: 16))
    }

    private var subtitle: String {
        let music = app.studio.music
        let seconds = Int(music.elapsed)
        let time = String(format: "%02d:%02d", seconds / 60, seconds % 60)
        let count = music.activeStyleCount
        return count == 1 ? L.f("MusicOneStyle", time) : L.f("MusicBlending", "\(count)", time)
    }
}

private struct LevelBars: View {
    let levels: [Float]
    let active: Bool

    var body: some View {
        GeometryReader { geometry in
            HStack(alignment: .center, spacing: 3) {
                ForEach(levels.indices, id: \.self) { index in
                    let idle = 0.12 + 0.1 * abs(sin(Double(index) * 0.37) * cos(Double(index) * 0.11))
                    let value = active ? max(0.06, Double(levels[index])) : idle
                    Capsule()
                        .fill(active ? Color.accentColor : Theme.controlBorder)
                        .frame(height: max(3, geometry.size.height * value))
                        .frame(maxWidth: .infinity)
                }
            }
            .frame(maxHeight: .infinity)
            .animation(.linear(duration: 0.1), value: levels)
        }
        .accessibilityHidden(true)
    }
}

private struct StylesSection: View {
    @Environment(AppModel.self) private var app

    var body: some View {
        @Bindable var music = app.studio.music
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .firstTextBaseline, spacing: 10) {
                Text(L.s("MusicStyles")).font(.system(size: 15, weight: .semibold))
                Text(L.s("MusicStylesHelp")).font(.system(size: 12)).foregroundStyle(Theme.textFaint)
            }
            GlassEffectContainer(spacing: 12) {
                LazyVGrid(columns: [GridItem(.adaptive(minimum: 240), spacing: 12)], spacing: 12) {
                    ForEach($music.styles) { $style in
                        VStack(alignment: .leading, spacing: 10) {
                            HStack(spacing: 8) {
                                TextField(L.s("MusicStylePlaceholder"), text: $style.text)
                                    .textFieldStyle(.plain)
                                    .font(.system(size: 14, weight: .medium))
                                    .accessibilityIdentifier("MusicStyleText")
                                Text("\(Int((style.weight * 100).rounded()))%").font(.system(size: 13)).foregroundStyle(Theme.textMuted)
                                    .monospacedDigit()
                                Button {
                                    music.removeStyle(style.id)
                                } label: {
                                    Image(systemName: "xmark").font(.system(size: 10, weight: .semibold))
                                }
                                .buttonStyle(.plain)
                                .foregroundStyle(Theme.textFaint)
                                .help(L.s("MusicRemoveStyle"))
                                .accessibilityLabel(L.s("MusicRemoveStyle"))
                            }
                            Slider(value: $style.weight, in: 0...1)
                                .accessibilityIdentifier("MusicStyleWeight")
                        }
                        .padding(14)
                        .glassEffect(.regular, in: .rect(cornerRadius: 12))
                    }
                    Button {
                        music.addStyle()
                    } label: {
                        HStack(spacing: 8) {
                            Image(systemName: "plus").font(.system(size: 12, weight: .medium))
                            Text(L.s("MusicAddStyle"))
                        }
                        .frame(maxWidth: .infinity, minHeight: 84)
                        .contentShape(Rectangle())
                        .overlay(RoundedRectangle(cornerRadius: 12).strokeBorder(Theme.controlBorder, style: StrokeStyle(lineWidth: 1, dash: [4, 3])))
                    }
                    .buttonStyle(.plain)
                    .foregroundStyle(.secondary)
                    .accessibilityIdentifier("MusicAddStyle")
                }
            }
        }
    }
}

private struct MusicSettingsSection: View {
    @Environment(AppModel.self) private var app

    var body: some View {
        @Bindable var music = app.studio.music
        GlassEffectContainer(spacing: 12) {
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 220), spacing: 12)], spacing: 12) {
                slider(L.s("MusicTemperature"), $music.temperature, "MusicTemperature")
                slider(L.s("MusicGuidance"), $music.guidance, "MusicGuidance")
                slider(L.s("MusicVariety"), $music.variety, "MusicVariety")
                VStack(alignment: .leading, spacing: 4) {
                    FieldLabel(L.s("MusicRunsOn"))
                    Text(L.f("MusicMetal", app.hardware.gpuName.isEmpty ? "Apple GPU" : app.hardware.gpuName)).lineLimit(1)
                }
                .frame(maxWidth: .infinity, minHeight: 50, alignment: .leading)
                .padding(14)
                .glassEffect(.regular, in: .rect(cornerRadius: 12))
            }
        }
    }

    private func slider(_ title: String, _ value: Binding<Double>, _ id: String) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            FieldLabel(title)
            Slider(value: value, in: 0...1).accessibilityIdentifier(id)
        }
        .frame(maxWidth: .infinity, minHeight: 50, alignment: .leading)
        .padding(14)
        .glassEffect(.regular, in: .rect(cornerRadius: 12))
    }
}

private struct RecordingsSection: View {
    @Environment(AppModel.self) private var app

    var body: some View {
        let music = app.studio.music
        VStack(alignment: .leading, spacing: 8) {
            Text(L.s("MusicRecordings")).font(.system(size: 15, weight: .semibold))
            if music.recordings.isEmpty {
                Text(L.f("MusicNoRecordings", MusicService.outputFolder.path.replacingOccurrences(of: FileManager.default.homeDirectoryForCurrentUser.path, with: "~")))
                    .font(.system(size: 12)).foregroundStyle(Theme.textFaint)
            }
            ForEach(music.recordings) { recording in
                HStack(spacing: 12) {
                    Button {
                        music.playRecording(recording)
                    } label: {
                        Image(systemName: music.playingRecording == recording.url ? "stop.fill" : "play.fill").font(.system(size: 11)).frame(width: 18, height: 18)
                    }
                    .buttonStyle(.glass)
                    .buttonBorderShape(.circle)
                    .accessibilityLabel(L.s("MusicPlay"))
                    Text(recording.url.deletingPathExtension().lastPathComponent).lineLimit(1).truncationMode(.middle)
                    Spacer()
                    Text("\(recording.date.formatted(date: .abbreviated, time: .shortened)) · \(String(format: "%d:%02d", Int(recording.seconds) / 60, Int(recording.seconds) % 60))")
                        .font(.system(size: 12)).foregroundStyle(Theme.textMuted).monospacedDigit()
                    Button {
                        NSWorkspace.shared.activateFileViewerSelecting([recording.url])
                    } label: {
                        Image(systemName: "folder")
                    }
                    .buttonStyle(.plain)
                    .foregroundStyle(Theme.textMuted)
                    .help(L.s("StudioShowInFinder"))
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .background(Theme.surfaceAlt.opacity(0.6), in: .rect(cornerRadius: 10))
                .accessibilityIdentifier("MusicRecording")
            }
        }
        .onAppear { music.loadRecordings() }
    }
}
