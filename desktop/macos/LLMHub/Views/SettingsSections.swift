import AppKit
import SwiftUI
import StableAudioMac
import UniformTypeIdentifiers

/// What Settings › Engines, Studio backends, Agent and Storage show: engines on disk, Claude Code, folder sizes.
/// Read off the main thread (sizes walk ComfyUI's Python, tens of thousands of files).
@MainActor @Observable
final class SettingsInfo {
    struct Engine: Identifiable {
        var id: String { name }
        let name: String
        let version: String
        let detail: String
        let installed: Bool
        let folder: URL?
    }

    struct Folder: Identifiable {
        var id: String { key }
        let key: String
        let name: String
        let url: URL
        let bytes: Int64
        var size: String { Fmt.bytes(bytes) }
    }

    static let shared = SettingsInfo()

    private(set) var engines: [Engine] = []
    private(set) var folders: [Folder] = []
    private(set) var studioModels: [Folder] = []
    private(set) var claudePath = ""
    private(set) var claudeVersion = ""
    var total: String { Fmt.bytes(folders.filter { $0.key != "gallery" }.reduce(0) { $0 + $1.bytes }) }

    nonisolated static var comfyRoot: URL { AppPaths.engines.appendingPathComponent("comfyui", isDirectory: true) }
    nonisolated static var gallery: URL {
        FileManager.default.urls(for: .picturesDirectory, in: .userDomainMask)[0].appendingPathComponent("LLM Hub", isDirectory: true)
    }

    func refresh(running: String?) async {
        engines = await Task.detached { Self.readEngines(running: running) }.value
        let claude = await Task.detached { Self.findClaude() }.value
        claudePath = claude ?? L.s("SettingsNotInstalled")
        claudeVersion = await Task.detached { claude.flatMap(Self.version) ?? "" }.value
        let (folders, models) = await Task.detached { (Self.readFolders(), Self.readStudioModels()) }.value
        self.folders = folders
        studioModels = models
    }

    nonisolated private static func readEngines(running: String?) -> [Engine] {
        let llama = AppPaths.engines.appendingPathComponent("llama.cpp", isDirectory: true)
        let builds = ((try? FileManager.default.contentsOfDirectory(atPath: llama.path)) ?? []).filter { $0.hasPrefix("b") }.sorted()
        var list = [Engine(name: "llama.cpp", version: builds.last.map { String($0.prefix { $0 != "-" }) } ?? "",
                           detail: builds.isEmpty ? L.s("SettingsEngineOnDemand") : running.map { L.f("SettingsEngineRunning", $0) } ?? "Metal",
                           installed: !builds.isEmpty, folder: llama)]
        list.append(Engine(name: "LiteRT-LM", version: "", detail: L.s("SettingsLiteRtDetail"), installed: true, folder: nil))
        list.append(Engine(name: "Apple Foundation Models", version: "", detail: L.s("SettingsFoundationDetail"), installed: true, folder: nil))
        return list
    }

    nonisolated static func findClaude() -> String? {
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        return ["\(home)/.local/bin/claude", "/opt/homebrew/bin/claude", "/usr/local/bin/claude", "\(home)/.claude/local/claude"]
            .first { FileManager.default.isExecutableFile(atPath: $0) }
    }

    nonisolated private static func version(_ claude: String) -> String? {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: claude)
        process.arguments = ["--version"]
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = Pipe()
        guard (try? process.run()) != nil else { return nil }
        process.waitUntilExit()
        return String(decoding: pipe.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self).split(separator: "\n").first.map(String.init)
    }

    nonisolated private static func readFolders() -> [Folder] {
        let engines = AppPaths.engines
        let speech = ["supertonic-3", "whisper.cpp"].map { engines.appendingPathComponent($0) }
        let speechBytes = speech.reduce(0) { $0 + size($1) }
        return [
            Folder(key: "models", name: L.s("SettingsStorageModels"), url: AppPaths.models, bytes: size(AppPaths.models)),
            Folder(key: "engines", name: L.s("SettingsStorageEngines"), url: engines, bytes: size(engines) - size(comfyRoot) - speechBytes),
            Folder(key: "speech", name: L.s("SettingsSpeech"), url: engines, bytes: speechBytes),
            Folder(key: "studio", name: L.s("SettingsStudio"), url: comfyRoot, bytes: size(comfyRoot)),
            Folder(key: "chats", name: L.s("SettingsStorageChats"), url: AppPaths.root, bytes: fileSize(AppPaths.database)),
            Folder(key: "attachments", name: L.s("SettingsStorageAttachments"), url: AppPaths.attachments, bytes: size(AppPaths.attachments)),
            Folder(key: "design", name: L.s("NavDesign"), url: AppPaths.design, bytes: size(AppPaths.design)),
            Folder(key: "agent", name: L.s("NavAgent"), url: AppPaths.agent, bytes: size(AppPaths.agent)),
            Folder(key: "logs", name: L.s("SettingsStorageLogs"), url: AppPaths.logs, bytes: size(AppPaths.logs)),
            Folder(key: "gallery", name: L.s("StudioGallery"), url: gallery, bytes: size(gallery)),
        ]
    }

    nonisolated private static func readStudioModels() -> [Folder] {
        let models = comfyRoot.appendingPathComponent("data/models", isDirectory: true)
        guard let files = FileManager.default.enumerator(at: models, includingPropertiesForKeys: [.fileSizeKey]) else { return [] }
        return files.compactMap { $0 as? URL }.filter { $0.pathExtension == "safetensors" }
            .map { Folder(key: $0.path, name: $0.lastPathComponent, url: $0, bytes: fileSize($0)) }
            .sorted { $0.bytes > $1.bytes }
    }

    nonisolated private static func fileSize(_ url: URL) -> Int64 {
        Int64((try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0)
    }

    nonisolated private static func size(_ folder: URL) -> Int64 {
        guard let files = FileManager.default.enumerator(at: folder, includingPropertiesForKeys: [.totalFileAllocatedSizeKey, .isSymbolicLinkKey],
                                                         options: [], errorHandler: nil) else { return 0 }
        var total: Int64 = 0
        for case let url as URL in files {
            let values = try? url.resourceValues(forKeys: [.totalFileAllocatedSizeKey, .isSymbolicLinkKey])
            if values?.isSymbolicLink == true {
                files.skipDescendants()
                continue
            }
            total += Int64(values?.totalFileAllocatedSize ?? 0)
        }
        return total
    }
}

/// A settings page: title, help text, then content, in the same frame as General and Local API.
private struct SectionPage<Content: View>: View {
    let title: String
    let help: String
    @ViewBuilder let content: Content

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                VStack(alignment: .leading, spacing: 4) {
                    Text(title).font(.system(size: 20, weight: .semibold))
                    if !help.isEmpty { Text(help).font(.system(size: 13)).foregroundStyle(Theme.textMuted) }
                }
                content
            }
            .frame(maxWidth: 760, alignment: .leading)
            .padding(EdgeInsets(top: 24, leading: 28, bottom: 24, trailing: 28))
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }
}

/// A row card: Liquid Glass, as the design's settings cards float over the window.
private struct Card<Content: View>: View {
    @ViewBuilder let content: Content
    var body: some View {
        VStack(alignment: .leading, spacing: 10) { content }
            .padding(16)
            .frame(maxWidth: .infinity, alignment: .leading)
            .glassEffect(.regular, in: .rect(cornerRadius: 14))
    }
}

private func openFolder(_ url: URL) {
    try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    NSWorkspace.shared.activateFileViewerSelecting([url])
}

struct EnginesSettings: View {
    @Environment(AppModel.self) private var app
    private let info = SettingsInfo.shared

    var body: some View {
        SectionPage(title: L.s("SettingsEngines"), help: L.s("SettingsEnginesHelp")) {
            GlassEffectContainer {
                VStack(spacing: 10) {
                    ForEach(info.engines) { engine in
                        Card {
                            HStack(spacing: 12) {
                                Circle().fill(engine.installed ? Theme.success : Theme.textMuted).frame(width: 8, height: 8)
                                VStack(alignment: .leading, spacing: 2) {
                                    HStack(spacing: 8) {
                                        Text(engine.name).fontWeight(.semibold)
                                        Text(engine.version).font(.system(size: 12, design: .monospaced)).foregroundStyle(Theme.textFaint)
                                    }
                                    Text(engine.detail).font(.system(size: 12)).foregroundStyle(Theme.textMuted)
                                }
                                Spacer()
                                if let folder = engine.folder {
                                    Button(L.s("SettingsOpenFolder")) { openFolder(folder) }.buttonStyle(.glass)
                                }
                            }
                        }
                        .accessibilityIdentifier("EngineCard")
                    }
                }
            }
        }
        .task { await info.refresh(running: app.engines.loadedModel.map { _ in app.engines.backend }) }
    }
}

struct SpeechSettings: View {
    @Environment(AppModel.self) private var app
    @State private var voice = AppSettings.current.voice
    @State private var build = AppSettings.current.ttsBuild
    @State private var language = AppSettings.current.ttsLanguage
    @State private var speed = AppSettings.current.ttsSpeed
    @State private var testing = false

    var body: some View {
        SectionPage(title: L.s("SettingsSpeech"), help: L.s("SettingsSpeechHelp")) {
            Card {
                LabeledContent(L.s("SettingsTts"), value: "Supertonic 3")
                LabeledContent(L.s("SettingsAsr"), value: "Whisper large-v3 turbo · q5_0")
            }
            Card {
                Picker(L.s("SpeechBuild"), selection: $build) {
                    ForEach(SupertonicVoices.Build.allCases, id: \.rawValue) { b in
                        Text("\(b.label) · \(Fmt.bytes(b.size))").tag(b.rawValue)
                    }
                }
                .accessibilityIdentifier("SpeechBuildPicker")
                .onChange(of: build) {
                    AppSettings.current.ttsBuild = build
                    AppSettings.save()
                    app.speech.unloadTts()
                }
                HStack(spacing: 10) {
                    VoicePicker(voice: $voice, title: L.s("StudioVoice"))
                    Button(L.s("SettingsTestVoice")) { test() }
                        .buttonStyle(.glass)
                        .disabled(testing)
                        .accessibilityIdentifier("TestVoiceButton")
                    Button(L.s("SpeechImportVoice")) { importVoice() }
                        .buttonStyle(.glass)
                        .accessibilityIdentifier("ImportVoiceButton")
                    if testing { ProgressView().controlSize(.small) }
                }
                Picker(L.s("SpeechLanguage"), selection: $language) {
                    Text(L.s("SpeechLanguageAuto")).tag("")
                    ForEach(Supertonic.spoken.sorted { SupertonicVoices.languageName($0) < SupertonicVoices.languageName($1) }, id: \.self) {
                        Text(SupertonicVoices.languageName($0)).tag($0)
                    }
                }
                .accessibilityIdentifier("SpeechLanguagePicker")
                .onChange(of: language) {
                    AppSettings.current.ttsLanguage = language
                    AppSettings.save()
                }
                LabeledContent(L.s("SpeechSpeed")) {
                    HStack(spacing: 10) {
                        Slider(value: Binding(get: { speed }, set: { speed = ($0 * 20).rounded() / 20 }), in: 0.5 ... 2)
                            .frame(maxWidth: 260)
                            .accessibilityIdentifier("SpeechSpeedSlider")
                            .onChange(of: speed) {
                                AppSettings.current.ttsSpeed = speed
                                AppSettings.save()
                            }
                        Text(String(format: "%.2f×", speed)).monospacedDigit().foregroundStyle(Theme.textMuted).frame(width: 48, alignment: .trailing)
                    }
                }
                if !app.speech.status.isEmpty {
                    Text(app.speech.status).font(.system(size: 12)).foregroundStyle(Theme.textMuted).accessibilityIdentifier("SpeechStatus")
                }
            }
            let custom = customVoices
            if !custom.isEmpty {
                Text(L.s("SpeechCustomVoices")).font(.system(size: 12, weight: .medium)).foregroundStyle(Theme.textFaint)
                VStack(spacing: 6) {
                    ForEach(custom) { item in
                        HStack {
                            Image(systemName: "waveform").foregroundStyle(Theme.textMuted)
                            Text(item.label).lineLimit(1)
                            Spacer()
                            Button {
                                app.speech.deleteVoice(item.key)
                                voice = AppSettings.current.voice
                            } label: { Image(systemName: "trash") }
                            .buttonStyle(.borderless)
                            .help(L.s("Delete"))
                            .accessibilityLabel(L.s("Delete"))
                        }
                        .padding(.horizontal, 12)
                        .padding(.vertical, 8)
                        .glassEffect(.regular, in: .rect(cornerRadius: 9))
                        .accessibilityIdentifier("CustomVoiceRow")
                    }
                }
            }
        }
        .onAppear { voice = AppSettings.current.voice }
    }

    private var customVoices: [SupertonicVoices.Voice] {
        _ = app.speech.voicesRevision
        return SupertonicVoices.custom()
    }

    private func test() {
        testing = true
        Task {
            do {
                let (url, _) = try await app.speech.synthesize(L.s("SettingsVoiceSample"), voice: voice)
                await app.speech.play(url)
            } catch {
                app.speech.status = L.f("ErrorPrefix", error.localizedDescription)
            }
            testing = false
        }
    }

    private func importVoice() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.json]
        panel.allowsMultipleSelection = false
        panel.message = L.s("SpeechImportVoice")
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do {
            voice = try app.speech.importVoice(from: url)
            app.speech.status = ""
        } catch {
            app.speech.status = L.f("SpeechImportFailed", error.localizedDescription)
        }
    }
}

/// The Supertonic voice picker (presets, then imported voices), shared by Settings › Speech and voice and Studio › Voice.
/// Changing it saves the choice for read aloud too.
struct VoicePicker: View {
    @Environment(AppModel.self) private var app
    @Binding var voice: String
    var title = ""

    var body: some View {
        let _ = app.speech.voicesRevision
        let voices = SupertonicVoices.all()
        Picker(title, selection: $voice) {
            ForEach(voices.filter { !$0.isCustom }) { Text($0.label).tag($0.key) }
            if voices.contains(where: \.isCustom) {
                Divider()
                ForEach(voices.filter(\.isCustom)) { Text($0.label).tag($0.key) }
            }
        }
        .fixedSize()
        .labelsHidden(title.isEmpty)
        .accessibilityIdentifier("VoicePicker")
        .onChange(of: voice) {
            AppSettings.current.voice = voice
            AppSettings.save()
        }
    }
}

private extension View {
    @ViewBuilder func labelsHidden(_ hidden: Bool) -> some View {
        if hidden { labelsHidden() } else { self }
    }
}

struct StudioSettings: View {
    @Environment(AppModel.self) private var app
    private let info = SettingsInfo.shared

    var body: some View {
        SectionPage(title: L.s("SettingsStudio"), help: L.s("SettingsStudioHelp")) {
            Card {
                Text("ComfyUI").fontWeight(.semibold)
                Text(SettingsInfo.comfyRoot.path).font(.system(size: 12, design: .monospaced)).foregroundStyle(Theme.textFaint).textSelection(.enabled)
                Button(L.s("SettingsOpenFolder")) { openFolder(SettingsInfo.comfyRoot) }.buttonStyle(.glass)
            }
            if !info.studioModels.isEmpty {
                Text(L.s("SettingsStudioModels")).font(.system(size: 12, weight: .medium)).foregroundStyle(Theme.textFaint)
                VStack(spacing: 6) {
                    ForEach(info.studioModels) { model in
                        HStack {
                            Text(model.name).font(.system(size: 12, design: .monospaced)).lineLimit(1)
                            Spacer()
                            Text(model.size).font(.system(size: 12)).foregroundStyle(Theme.textMuted)
                            Button {
                                try? FileManager.default.removeItem(at: model.url)
                                Task { await info.refresh(running: nil) }
                            } label: { Image(systemName: "trash") }
                            .buttonStyle(.borderless)
                            .help(L.s("Delete"))
                        }
                        .padding(.horizontal, 12)
                        .padding(.vertical, 8)
                        .glassEffect(.regular, in: .rect(cornerRadius: 9))
                    }
                }
            }
            Card {
                Text(L.s("SettingsMusic")).fontWeight(.semibold)
                Text(L.s(MusicService.supported ? "SettingsMusicMac" : "MusicNeedsAppleSilicon")).font(.system(size: 12)).foregroundStyle(Theme.textMuted)
            }
            StableAudioCard()
        }
        .task { await info.refresh(running: nil) }
    }
}

/// Settings › Studio backends: Stable Audio 3 for music clips, with the attribution and license its terms ask for.
private struct StableAudioCard: View {
    @Environment(AppModel.self) private var app
    @State private var revision = 0

    var body: some View {
        let _ = revision
        let clips = app.studio.clips
        Card {
            HStack(alignment: .firstTextBaseline) {
                Text("Stable Audio 3").fontWeight(.semibold)
                Spacer()
                Text(StabilityAttribution.poweredBy).font(.system(size: 12, weight: .medium)).foregroundStyle(Theme.textMuted)
                    .accessibilityIdentifier("SettingsPoweredByStability")
            }
            Text(L.s("SettingsStableAudioHelp")).font(.system(size: 12)).foregroundStyle(Theme.textMuted)
            ForEach(StableAudioService.models, id: \.self) { model in
                HStack {
                    Text(StableAudioService.name(model)).font(.system(size: 12))
                    Spacer()
                    Text(StableAudio3.isInstalled(model, in: StableAudioService.folder) ? L.s("Installed") : L.f("StudioDownloadNote", Fmt.bytes(model.downloadSize)))
                        .font(.system(size: 12)).foregroundStyle(Theme.textMuted)
                }
            }
            HStack(spacing: 8) {
                Button(L.s("SettingsLicense")) { StabilityAttribution.open(StabilityAttribution.license) }.buttonStyle(.glass)
                    .accessibilityIdentifier("StabilityLicenseButton")
                Button("NOTICE") { StabilityAttribution.open(StabilityAttribution.notice) }.buttonStyle(.glass)
                Button(L.s("SettingsOpenFolder")) { openFolder(StableAudioService.folder) }.buttonStyle(.glass)
                if StableAudioService.models.contains(where: { StableAudio3.isInstalled($0, in: StableAudioService.folder) }) {
                    Button(L.s("SettingsRemove")) {
                        clips.remove()
                        revision += 1
                    }
                    .buttonStyle(.glass)
                    .disabled(clips.isGenerating)
                }
            }
        }
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("StableAudioCard")
    }
}

struct AgentSettings: View {
    private let info = SettingsInfo.shared
    @State private var context = AppSettings.current.apiContextLength

    var body: some View {
        SectionPage(title: L.s("SettingsAgent"), help: L.s("SettingsAgentHelp")) {
            Card {
                HStack(spacing: 8) {
                    Text(L.s("SettingsClaudeCode")).fontWeight(.semibold)
                    Text(info.claudeVersion).font(.system(size: 12, design: .monospaced)).foregroundStyle(Theme.textFaint)
                        .accessibilityIdentifier("ClaudeVersion")
                }
                Text(info.claudePath).font(.system(size: 12, design: .monospaced)).foregroundStyle(Theme.textMuted).textSelection(.enabled)
            }
            Picker(L.s("SettingsAgentContext"), selection: $context) {
                ForEach([32768, 65536, 131072], id: \.self) { Text("\($0 / 1024)K").tag($0) }
            }
            .fixedSize()
            .onChange(of: context) {
                AppSettings.current.apiContextLength = context
                AppSettings.save()
            }
            Text(L.s("SettingsAgentContextHelp")).font(.system(size: 12)).foregroundStyle(Theme.textMuted)
            Text(L.s("SettingsAgentNote")).font(.system(size: 12)).foregroundStyle(Theme.textFaint)
        }
        .task { await info.refresh(running: nil) }
    }
}

struct WebSearchSettings: View {
    @State private var on = AppSettings.current.webSearch

    var body: some View {
        SectionPage(title: L.s("SettingsWebSearch"), help: "") {
            Toggle(L.s("SettingsWebSearchDefault"), isOn: $on)
                .toggleStyle(.switch)
                .accessibilityIdentifier("WebSearchToggle")
                .onChange(of: on) {
                    AppSettings.current.webSearch = on
                    AppSettings.save()
                }
            Text(L.s("SettingsWebSearchHelp")).font(.system(size: 12)).foregroundStyle(Theme.textMuted)
        }
    }
}

struct StorageSettings: View {
    private let info = SettingsInfo.shared

    var body: some View {
        SectionPage(title: L.s("SettingsStorage"), help: L.s("SettingsStorageHelp")) {
            if !info.folders.isEmpty {
                Text(L.f("SettingsTotal", info.total)).fontWeight(.semibold).accessibilityIdentifier("StorageTotal")
            }
            VStack(spacing: 6) {
                ForEach(info.folders) { folder in
                    HStack(spacing: 12) {
                        VStack(alignment: .leading, spacing: 2) {
                            Text(folder.name).fontWeight(.medium)
                            Text(folder.url.path).font(.system(size: 11, design: .monospaced)).foregroundStyle(Theme.textFaint).lineLimit(1).truncationMode(.middle)
                        }
                        Spacer()
                        Text(folder.size).foregroundStyle(Theme.textMuted)
                        if folder.key == "logs" {
                            Button(L.s("SettingsClear")) {
                                for file in (try? FileManager.default.contentsOfDirectory(at: folder.url, includingPropertiesForKeys: nil)) ?? [] {
                                    try? FileManager.default.removeItem(at: file)
                                }
                                Task { await info.refresh(running: nil) }
                            }
                            .buttonStyle(.glass)
                        }
                        Button { openFolder(folder.url) } label: { Image(systemName: "folder") }
                            .buttonStyle(.borderless)
                            .help(L.s("SettingsOpenFolder"))
                    }
                    .padding(.horizontal, 14)
                    .padding(.vertical, 10)
                    .glassEffect(.regular, in: .rect(cornerRadius: 10))
                    .accessibilityIdentifier("StorageRow")
                }
            }
        }
        .task { await info.refresh(running: nil) }
    }
}
