import AppKit
import Foundation
import Observation

/// A gallery entry: an image or video file in ~/Pictures/LLM Hub.
struct GalleryItem: Identifiable, Hashable {
    let url: URL
    var id: URL { url }
    var name: String { url.lastPathComponent }
    var isVideo: Bool {
        let ext = url.pathExtension.lowercased()
        return (ext == "webp" && name.lowercased().contains("ltxv")) || ext == "webm" || ext == "mp4"
    }
}

/// Studio: image and video (ComfyUI), music (Magenta RealTime on MLX) and voice (Supertonic).
/// The view model behind StudioView; a port of core/LLMHub.Core/ViewModels/StudioViewModel.cs.
@MainActor @Observable
final class StudioService {
    private weak var app: AppModel?
    let comfy = ComfyService()
    let music = MusicService()
    /// Fixed-length clips with Stable Audio 3, next to the live jam.
    let clips = StableAudioService()

    static let aspects = ["3:2", "1:1", "16:9", "9:16"]
    static let counts = [1, 2, 4]

    /// image, video, music or voice.
    var mode = "image" {
        didSet {
            if mode == "video" && workflow.kind != "video" { workflow = .ltxVideo }
            else if mode == "image" && workflow.kind != "image" { workflow = .fluxSchnell }
        }
    }
    var workflow = ComfyWorkflow.fluxSchnell {
        didSet { if workflow != oldValue { steps = workflow.defaultSteps } }
    }
    var prompt = ""
    var aspect = "3:2"
    var count = 1
    var steps = ComfyWorkflow.fluxSchnell.defaultSteps
    /// A number, or empty for a random seed.
    var seed = ""
    var referenceImage = ""
    var isGenerating = false
    var isImproving = false
    var progress = 0.0
    var status = ""
    var gallery: [GalleryItem] = []
    var selected: GalleryItem?
    /// Number of the image or clip being made and how many there are, for "Generating 1 of 4".
    var runningCount = 0
    var voiceText = ""
    var voice = AppSettings.current.voice {
        didSet {
            AppSettings.current.voice = voice
            AppSettings.save()
        }
    }
    var lastSpeech: URL?
    var voiceStatus = ""
    var isSpeaking = false
    /// Bumped when models land on disk so the download note refreshes.
    private var modelsRevision = 0

    @ObservationIgnored private var run: Task<Void, Never>?

    func attach(_ app: AppModel) {
        self.app = app
        loadGallery()
    }

    var workflows: [ComfyWorkflow] { ComfyWorkflow.all }
    var needsReference: Bool { workflow.needsImage }
    var isImageOrVideo: Bool { mode == "image" || mode == "video" }
    /// "6.9 GB to download on first use" under the workflow picker, or "".
    var downloadNote: String {
        _ = modelsRevision
        return ComfyService.hasModels(workflow) ? "" : L.f("StudioDownloadNote", Fmt.bytes(workflow.downloadSize))
    }
    var galleryPathLabel: String {
        ComfyService.gallery.path.replacingOccurrences(of: FileManager.default.homeDirectoryForCurrentUser.path, with: "~")
    }

    // MARK: - Generate

    func generate() {
        guard !isGenerating else { return }
        run = Task { await generateNow() }
    }

    /// Runs one generation to the end; the test hook awaits this directly.
    func generateNow() async {
        if workflow.needsImage && !FileManager.default.fileExists(atPath: referenceImage) {
            status = L.s("StudioNeedsImage")
            return
        }
        let text = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        if !workflow.needsImage && text.isEmpty { return }
        isGenerating = true
        progress = 0
        let workflow = workflow
        let images = workflow.kind == "video" || workflow.needsImage ? 1 : count
        runningCount = images
        // Keeps App Nap from throttling multi-GB downloads and long runs while LLM Hub is in the background.
        let activity = ProcessInfo.processInfo.beginActivity(options: [.userInitiated, .idleSystemSleepDisabled], reason: "Studio generation")
        defer {
            ProcessInfo.processInfo.endActivity(activity)
            isGenerating = false
            progress = 0
            runningCount = 0
        }
        do {
            status = L.s("ComfyStarting")
            try await comfy.ensureRunning()
            try await comfy.ensureModels(workflow)
            modelsRevision += 1
            let image = workflow.needsImage ? try await comfy.upload(URL(fileURLWithPath: referenceImage)) : nil
            let (width, height) = ComfyWorkflow.size(workflow, aspect: aspect)
            let seedValue = Int64(seed.trimmingCharacters(in: .whitespaces)) ?? Int64.random(in: 1..<(1 << 48))
            let graph = workflow.build(prompt: text, width: width, height: height, count: images, steps: max(1, steps), seed: seedValue, image: image)
            let clock = Date()
            status = L.f("ComfyStep", "0", "\(max(1, steps))")
            let outputs = try await comfy.run(graph) { [weak self] value, line in
                Task { @MainActor in
                    self?.progress = value
                    self?.status = line
                }
            }
            let took = Date().timeIntervalSince(clock)
            Log.info("Studio \(workflow.id) \(workflow.id == "upscale" ? "" : "\(width)x\(height) seed \(seedValue)"): \(outputs.count) file(s) in \(String(format: "%.1f", took))s")
            status = L.f("StudioDone", "\(outputs.count)", String(format: "%.0f", took))
            gallery.insert(contentsOf: outputs.map(GalleryItem.init(url:)), at: 0)
            selected = gallery.first
        } catch is CancellationError {
            status = ""
        } catch let error as URLError where error.code == .cancelled {
            status = ""
        } catch {
            Log.error("Studio generation failed", error)
            status = L.f("ErrorPrefix", error.localizedDescription)
        }
    }

    func cancel() { run?.cancel() }

    /// Asks the local chat model to expand a short idea into a detailed image or video prompt.
    func improvePrompt() {
        let text = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty, !isGenerating, !isImproving, let app, let model = app.chat.selectedModel else {
            if app?.chat.selectedModel == nil { status = L.s("EmptyNoModel") }
            return
        }
        isImproving = true
        status = L.s("StudioImproving")
        let instruction = workflow.kind == "video"
            ? "Rewrite this as one vivid prompt for an image-to-video model: describe the motion, camera movement and mood in under 60 words. Reply with the prompt only."
            : "Rewrite this as one detailed prompt for a text-to-image model: subject, setting, lighting, style and composition, under 70 words. Reply with the prompt only."
        Task {
            defer { isImproving = false }
            do {
                try await app.engines.load(model)
                var reply = ""
                for try await delta in app.engines.stream(model, turns: [ChatTurn(role: "system", content: instruction), ChatTurn(role: "user", content: text)]) {
                    reply += delta.content ?? ""
                }
                var splitter = ThinkSplitter()
                let answer = splitter.push(reply).answer.trimmingCharacters(in: .whitespacesAndNewlines).trimmingCharacters(in: CharacterSet(charactersIn: "\""))
                if !answer.isEmpty { prompt = answer }
                status = ""
            } catch {
                Log.error("Improve prompt failed", error)
                status = L.f("ErrorPrefix", error.localizedDescription)
            }
        }
    }

    /// Studio › Music clips: the user's own local chat model (never a separate prompt model) turns a short idea into a
    /// Stable Audio 3 style prompt (genre, instruments, mood, tempo).
    func improveClipPrompt() {
        let text = clips.prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty, !clips.isGenerating, !isImproving, let app, let model = app.chat.selectedModel else {
            if app?.chat.selectedModel == nil { clips.status = L.s("EmptyNoModel") }
            return
        }
        isImproving = true
        clips.status = L.s("StudioImproving")
        let instruction = "Rewrite this as one prompt for a text-to-music model: genre, instruments, mood, tempo in BPM and production style, "
            + "as a comma-separated description under 40 words. No lyrics, artist or song names. Reply with the prompt only."
        Task {
            defer { isImproving = false }
            do {
                try await app.engines.load(model)
                var reply = ""
                for try await delta in app.engines.stream(model, turns: [ChatTurn(role: "system", content: instruction), ChatTurn(role: "user", content: text)]) {
                    reply += delta.content ?? ""
                }
                var splitter = ThinkSplitter()
                let answer = splitter.push(reply).answer.trimmingCharacters(in: .whitespacesAndNewlines).trimmingCharacters(in: CharacterSet(charactersIn: "\""))
                if !answer.isEmpty { clips.prompt = answer }
                clips.status = ""
            } catch {
                Log.error("Improve music prompt failed", error)
                clips.status = L.f("ErrorPrefix", error.localizedDescription)
            }
        }
    }

    // MARK: - Gallery actions

    func animate(_ item: GalleryItem) {
        mode = "video"
        workflow = .ltxVideo
        referenceImage = item.url.path
    }

    func upscale(_ item: GalleryItem) {
        mode = "image"
        workflow = .upscale
        referenceImage = item.url.path
    }

    func openInComfy() {
        Task {
            do {
                try await comfy.ensureRunning()
                if let url = comfy.url, let target = URL(string: url) { NSWorkspace.shared.open(target) }
            } catch {
                status = L.f("ErrorPrefix", error.localizedDescription)
            }
        }
    }

    func loadGallery() {
        let keys: [URLResourceKey] = [.contentModificationDateKey]
        let files = (try? FileManager.default.contentsOfDirectory(at: ComfyService.gallery, includingPropertiesForKeys: keys)) ?? []
        gallery = files.filter { ["png", "webp", "jpg", "jpeg", "webm", "mp4"].contains($0.pathExtension.lowercased()) }
            .map { ($0, (try? $0.resourceValues(forKeys: Set(keys)).contentModificationDate) ?? .distantPast) }
            .sorted { $0.1 > $1.1 }.prefix(200).map { GalleryItem(url: $0.0) }
        if selected == nil { selected = gallery.first }
    }

    // MARK: - Voice

    func speak() {
        let text = voiceText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty, !isSpeaking, let app else { return }
        isSpeaking = true
        voiceStatus = ""
        let voice = voice
        Task {
            defer { isSpeaking = false }
            do {
                let (url, seconds) = try await app.speech.synthesize(text, voice: voice, language: nil)
                let folder = ComfyService.gallery.appendingPathComponent("Voice", isDirectory: true)
                try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
                let stamp = DateFormatter()
                stamp.dateFormat = "yyyyMMdd-HHmmss"
                let target = folder.appendingPathComponent("\(stamp.string(from: Date()))-\(voice).wav")
                try? FileManager.default.removeItem(at: target)
                try FileManager.default.copyItem(at: url, to: target)
                lastSpeech = target
                voiceStatus = L.f("StudioSpeechDone", String(format: "%.1f", seconds))
                Log.info("Studio voice \(voice): \(String(format: "%.1f", seconds))s -> \(target.path)")
                await app.speech.play(target)
            } catch {
                Log.error("Voice generation failed", error)
                voiceStatus = L.f("ErrorPrefix", error.localizedDescription)
            }
        }
    }
}
