import SwiftUI

/// Studio: mode tabs (Image, Video, Music, Voice, Search) and the engine status, then the mode's page.
struct StudioView: View {
    @Environment(AppModel.self) private var app

    var body: some View {
        let studio = app.studio
        VStack(spacing: 0) {
            StudioHeader()
            Divider()
            Group {
                switch studio.mode {
                case "music": MusicView()
                case "voice": VoiceView()
                case "search": SearchView()
                default: ImageStudioView()
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
    }
}

private struct StudioHeader: View {
    @Environment(AppModel.self) private var app

    var body: some View {
        let studio = app.studio
        HStack(spacing: 8) {
            GlassEffectContainer(spacing: 4) {
                HStack(spacing: 4) {
                    tab("image", L.s("StudioImage"))
                    tab("video", L.s("StudioVideo"))
                    tab("music", L.s("StudioMusic"))
                    tab("voice", L.s("StudioVoice"))
                    tab("search", L.s("StudioSearch"))
                }
            }
            .accessibilityIdentifier("StudioTabs")
            Spacer()
            if studio.isImageOrVideo {
                status(studio.comfy.isRunning ? Theme.success : Theme.textMuted, studio.comfy.statusLine, id: "ComfyStatus")
            } else if studio.mode == "music" {
                status(studio.music.isPlaying ? Theme.success : Theme.textMuted, musicLine, id: "MusicStatus")
            }
        }
        .padding(.horizontal, 20)
        .frame(minHeight: 52)
    }

    private var musicLine: String {
        let music = app.studio.music
        if !MusicService.supported { return L.s("MusicNeedsAppleSilicon") }
        if music.isPlaying { return L.s("MusicStatusLive") }
        if music.isDownloaded { return L.s("MusicStatusReady") }
        return L.f("MusicStatusDownload", Fmt.bytes(MusicService.downloadSize))
    }

    private func tab(_ mode: String, _ title: String) -> some View {
        let selected = app.studio.mode == mode
        return Button {
            app.studio.mode = mode
        } label: {
            Text(title)
                .font(.system(size: 13))
                .padding(.horizontal, 14)
                .frame(height: 32)
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .foregroundStyle(selected ? .primary : Theme.textMuted)
        .glassEffect(selected ? .regular.interactive() : .identity, in: .rect(cornerRadius: 8))
        .accessibilityAddTraits(selected ? .isSelected : [])
        .accessibilityIdentifier("StudioTab_\(mode)")
    }

    private func status(_ color: Color, _ text: String, id: String) -> some View {
        HStack(spacing: 8) {
            Circle().fill(color).frame(width: 8, height: 8)
            Text(text).font(.system(size: 12)).foregroundStyle(Theme.textMuted).lineLimit(1)
                .accessibilityIdentifier(id)
        }
    }
}

/// Small caption above a field, as in the design (12pt, medium, muted).
struct FieldLabel: View {
    let text: String
    init(_ text: String) { self.text = text }
    var body: some View {
        Text(text).font(.system(size: 12, weight: .medium)).foregroundStyle(Theme.textMuted)
    }
}
