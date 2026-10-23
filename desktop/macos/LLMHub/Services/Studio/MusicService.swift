import AVFoundation
import Foundation
import MagentaMac
import Observation

/// Studio › Music: live music from Magenta RealTime 2 Small on the Apple GPU (MLX), the same model and runtime as the
/// iOS app. Text styles are blended with MusicCoCa and can change while it plays; everything played can be recorded or
/// exported as WAV to ~/Pictures/LLM Hub/Music.
@MainActor @Observable
final class MusicService {
    struct Style: Identifiable, Hashable {
        let id = UUID()
        var text: String
        var weight: Double
    }

    struct Recording: Identifiable, Hashable {
        var id: URL { url }
        let url: URL
        let date: Date
        let seconds: Double
    }

    /// MRT2 Small and MusicCoCa, pinned to the same Hugging Face commit as the iOS app.
    private static let commit = "https://huggingface.co/google/magenta-realtime-2/resolve/010aa0dcb0dfd27b24f0ad07b4dad63e8f9521cc/"
    private static let files: [(path: String, folder: String, size: Int64)] = [
        ("models/mrt2_small/mrt2_small.mlxfn", "mrt2_small", 0),
        ("models/mrt2_small/mrt2_small_state.safetensors", "mrt2_small", 0),
        ("resources/musiccoca/text_encoder.tflite", "musiccoca", 0),
        ("resources/musiccoca/mapper.tflite", "musiccoca", 0),
        ("resources/musiccoca/pretrained_vector_quantizer.tflite", "musiccoca", 0),
        ("resources/musiccoca/spm.model", "musiccoca", 0),
    ]
    /// About 1.04 GB in all (the iOS catalog's size for MRT2 Small with MusicCoCa).
    static let downloadSize: Int64 = 1_042_112_092

    var styles: [Style] = [Style(text: "Lo-fi hip hop", weight: 0.7), Style(text: "Warm Rhodes piano", weight: 0.45),
                           Style(text: "Rain ambience", weight: 0.2)] {
        didSet { if styles != oldValue { stylesChanged() } }
    }
    /// Sliders 0...1, mapped to Magenta's temperature (0.5...2, 1.3 at 0.55), guidance (0...5, 3 at 0.6) and top-k (1...100).
    var temperature = 0.55 { didSet { controlsChanged() } }
    var guidance = 0.6 { didSet { controlsChanged() } }
    var variety = 0.4 { didSet { controlsChanged() } }

    var isPlaying = false
    var isBusy = false
    var isRecording = false
    var status = ""
    var elapsed = 0.0
    /// The last 80 frames' loudness (0...1) for the level bars.
    var levels = [Float](repeating: 0, count: 80)
    var recordings: [Recording] = []
    /// Frames made per second of wall time divided by 25; below 1 the Mac is slower than real time.
    var speed = 0.0
    /// Plays silently (the --musiclive test).
    @ObservationIgnored var muted = false

    @ObservationIgnored private var engine: MagentaLive?
    @ObservationIgnored private var sink: MusicSink?
    @ObservationIgnored private var player: AVAudioPlayer?
    @ObservationIgnored private var styleTask: Task<Void, Never>?
    @ObservationIgnored private var meter: Timer?
    @ObservationIgnored private var playActivity: NSObjectProtocol?

    static var supported: Bool {
        #if arch(arm64)
        true
        #else
        false
        #endif
    }

    nonisolated static var folder: URL { AppPaths.engines.appendingPathComponent("magenta", isDirectory: true) }
    nonisolated static var outputFolder: URL {
        let url = ComfyService.gallery.appendingPathComponent("Music", isDirectory: true)
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }
    nonisolated private static var resources: URL { folder.appendingPathComponent("musiccoca", isDirectory: true) }

    var isDownloaded: Bool {
        Self.files.allSatisfy { FileManager.default.fileExists(atPath: Self.target(for: $0.path, $0.folder).path) }
    }

    init() { loadRecordings() }

    private static func target(for path: String, _ folder: String) -> URL {
        Self.folder.appendingPathComponent(folder, isDirectory: true).appendingPathComponent((path as NSString).lastPathComponent)
    }

    var controls: MagentaLive.Controls {
        var c = MagentaLive.Controls()
        c.temperature = Float(0.5 + temperature * 1.5)
        c.guidance = Float(guidance * 5)
        c.topK = Int32(max(1, (variety * 100).rounded()))
        return c
    }

    private var weightedStyles: [(text: String, weight: Float)] {
        styles.map { ($0.text, Float($0.weight)) }
    }

    var activeStyleCount: Int { styles.filter { $0.weight > 0 && !$0.text.trimmingCharacters(in: .whitespaces).isEmpty }.count }

    // MARK: - Setup

    /// Downloads the model on first use and loads it on the GPU.
    func prepare() async throws -> MagentaLive {
        if let engine { return engine }
        guard Self.supported else { throw ComfyService.error(L.s("MusicNeedsAppleSilicon")) }
        let activity = ProcessInfo.processInfo.beginActivity(options: [.userInitiated, .idleSystemSleepDisabled], reason: "Magenta RealTime setup")
        defer { ProcessInfo.processInfo.endActivity(activity) }
        if !isDownloaded {
            var done: Int64 = 0
            for file in Self.files {
                let target = Self.target(for: file.path, file.folder)
                if FileManager.default.fileExists(atPath: target.path) { continue }
                try FileManager.default.createDirectory(at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
                let before = done
                try await ComfyService.resumableDownload(request: URLRequest(url: URL(string: Self.commit + file.path)!), to: target,
                                                           expectedSize: 0) { bytes in
                    Task { @MainActor in
                        self.status = L.f("MusicDownloading", String(format: "%.0f", min(99, Double(before + bytes) * 100 / Double(Self.downloadSize))))
                    }
                }
                done += (try? FileManager.default.attributesOfItem(atPath: target.path)[.size] as? NSNumber)?.int64Value ?? 0
            }
            Log.info("Magenta RealTime downloaded (\(Fmt.bytes(done)))")
        }
        status = L.s("MusicLoading")
        let model = Self.folder.appendingPathComponent("mrt2_small", isDirectory: true)
        let clock = Date()
        let loaded = try await Task.detached(priority: .userInitiated) {
            try MagentaLive(function: model.appendingPathComponent("mrt2_small.mlxfn"),
                            state: model.appendingPathComponent("mrt2_small_state.safetensors"))
        }.value
        Log.info("Magenta RealTime loaded in \(String(format: "%.1f", Date().timeIntervalSince(clock)))s")
        engine = loaded
        status = ""
        return loaded
    }

    // MARK: - Live

    func toggle() {
        if isPlaying { stop() } else { play() }
    }

    func play() {
        guard !isPlaying, !isBusy else { return }
        isBusy = true
        status = ""
        Task {
            do {
                let engine = try await prepare()
                let styles = weightedStyles
                let resources = Self.resources
                let tokens = try await Task.detached { try MusicCoCa.tokens(styles: styles, resources: resources) }.value
                try start(engine, tokens: tokens)
            } catch {
                Log.error("Music failed to start", error)
                status = L.f("ErrorPrefix", error.localizedDescription)
            }
            isBusy = false
        }
    }

    private func start(_ engine: MagentaLive, tokens: [Int32]) throws {
        let sink = try MusicSink(session: FileManager.default.temporaryDirectory.appendingPathComponent("llmhub-music-session.wav"))
        self.sink = sink
        try sink.startPlayback(muted: muted)
        if isRecording { try sink.startRecording(to: newRecordingURL()) }
        isPlaying = true
        elapsed = 0
        // Live audio must not be throttled by App Nap when the window is in the background.
        playActivity = ProcessInfo.processInfo.beginActivity(options: [.userInitiated, .latencyCritical, .idleSystemSleepDisabled], reason: "Live music")
        let controls = controls
        let thread = Thread {
            let clock = Date()
            do {
                let frames = try engine.run(tokens: tokens, controls: controls, shouldStop: { sink.isStopped }) { pcm in
                    try sink.write(pcm)
                }
                Log.info("Music stopped after \(frames) frames (\(String(format: "%.1f", Double(frames) / 25))s audio in \(String(format: "%.1f", Date().timeIntervalSince(clock)))s)")
            } catch {
                Log.error("Music generation failed", error)
                Task { @MainActor in self.status = L.f("ErrorPrefix", error.localizedDescription) }
            }
            Task { @MainActor in self.finished(sink) }
        }
        thread.qualityOfService = .userInteractive
        thread.stackSize = 8 << 20
        thread.start()
        meter = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { [weak self] _ in
            MainActor.assumeIsolated { self?.updateMeter() }
        }
    }

    func stop() {
        sink?.stop()
    }

    private func finished(_ sink: MusicSink) {
        guard sink === self.sink else { return }
        meter?.invalidate()
        meter = nil
        if let url = sink.stopRecording() { added(url) }
        Log.info(String(format: "Music played %.1fs, %.2fs of underrun", sink.snapshot().played, Double(sink.underrunFrames) / Double(MagentaLive.sampleRate)))
        sink.stopPlayback()
        if let playActivity { ProcessInfo.processInfo.endActivity(playActivity) }
        playActivity = nil
        isPlaying = false
        levels = [Float](repeating: 0, count: 80)
    }

    private func updateMeter() {
        guard let sink else { return }
        let snapshot = sink.snapshot()
        elapsed = snapshot.played
        levels = snapshot.levels
        speed = snapshot.speed
    }

    func setRecording(_ on: Bool) {
        isRecording = on
        guard let sink, isPlaying else { return }
        if on {
            do { try sink.startRecording(to: newRecordingURL()) } catch {
                status = L.f("ErrorPrefix", error.localizedDescription)
                isRecording = false
            }
        } else if let url = sink.stopRecording() {
            added(url)
        }
    }

    /// Saves everything made since Play as a WAV in the gallery.
    func export() {
        guard let sink else { return }
        do {
            let url = newRecordingURL()
            try sink.exportSession(to: url)
            added(url)
        } catch {
            status = L.f("ErrorPrefix", error.localizedDescription)
        }
    }

    private func added(_ url: URL) {
        status = L.f("MusicSaved", url.lastPathComponent)
        Log.info("Music saved \(url.path)")
        loadRecordings()
    }

    private func newRecordingURL(named title: String? = nil) -> URL {
        let stamp = DateFormatter()
        stamp.dateFormat = "yyyyMMdd-HHmmss"
        let name = (title ?? styles.max { $0.weight < $1.weight }?.text ?? "music")
            .components(separatedBy: CharacterSet.alphanumerics.inverted).filter { !$0.isEmpty }.prefix(4).joined(separator: "-").lowercased()
        return Self.outputFolder.appendingPathComponent("\(stamp.string(from: Date()))-\(name.isEmpty ? "music" : name).wav")
    }

    // MARK: - Styles and settings while playing

    func addStyle() { styles.append(Style(text: "", weight: 0.5)) }
    func removeStyle(_ id: UUID) { styles.removeAll { $0.id == id } }

    private func stylesChanged() {
        guard isPlaying, let engine else { return }
        styleTask?.cancel()
        let styles = weightedStyles
        let resources = Self.resources
        styleTask = Task {
            try? await Task.sleep(for: .milliseconds(350))
            guard !Task.isCancelled else { return }
            do {
                let tokens = try await Task.detached { try MusicCoCa.tokens(styles: styles, resources: resources) }.value
                engine.update(tokens: tokens)
            } catch MusicCoCa.CoCaError.noStyle {
            } catch {
                Log.error("Music style update failed", error)
            }
        }
    }

    private func controlsChanged() { engine?.update(controls: controls) }

    // MARK: - Recordings

    func loadRecordings() {
        let files = (try? FileManager.default.contentsOfDirectory(at: Self.outputFolder, includingPropertiesForKeys: [.contentModificationDateKey, .fileSizeKey])) ?? []
        recordings = files.filter { $0.pathExtension.lowercased() == "wav" }.compactMap { url in
            let values = try? url.resourceValues(forKeys: [.contentModificationDateKey, .fileSizeKey])
            let bytes = Double(values?.fileSize ?? 0)
            return Recording(url: url, date: values?.contentModificationDate ?? .distantPast, seconds: max(0, bytes - 44) / Double(MagentaLive.sampleRate * 4))
        }.sorted { $0.date > $1.date }.prefix(30).map { $0 }
    }

    var playingRecording: URL?

    func playRecording(_ recording: Recording) {
        if playingRecording == recording.url {
            player?.stop()
            playingRecording = nil
            return
        }
        player?.stop()
        player = try? AVAudioPlayer(contentsOf: recording.url)
        player?.play()
        playingRecording = player == nil ? nil : recording.url
        let seconds = recording.seconds
        Task { [weak self] in
            try? await Task.sleep(for: .seconds(seconds + 0.2))
            if self?.playingRecording == recording.url { self?.playingRecording = nil }
        }
    }

    // MARK: - Fixed-length clip (used by --musictest)

    /// Makes `seconds` of music for `prompt` without playing it and saves it as WAV. Returns the file, its length and
    /// how many seconds it took.
    func render(prompt: String, seconds: Double) async throws -> (url: URL, seconds: Double, took: Double) {
        let engine = try await prepare()
        let resources = Self.resources
        let controls = controls
        let url = newRecordingURL(named: prompt)
        let clock = Date()
        let activity = ProcessInfo.processInfo.beginActivity(options: [.userInitiated, .idleSystemSleepDisabled], reason: "Music clip")
        defer { ProcessInfo.processInfo.endActivity(activity) }
        let frames = try await Task.detached(priority: .userInitiated) { () -> Int in
            let tokens = try MusicCoCa.tokens(styles: [(prompt, 1)], resources: resources)
            let writer = try WavWriter(url: url)
            defer { writer.close() }
            return try engine.run(tokens: tokens, controls: controls, frames: max(1, Int((seconds * 25).rounded(.up))), shouldStop: { false }) { pcm in
                writer.append(pcm)
            }
        }.value
        loadRecordings()
        return (url, Double(frames) / 25, Date().timeIntervalSince(clock))
    }
}

/// Where generated frames go, on the generation thread: the speaker (through the iOS app's lock-free ring), the session
/// WAV for Export, an optional recording, and the level meter.
final class MusicSink: @unchecked Sendable {
    private let lock = NSLock()
    private let ring: MagentaAudioBuffer
    private let audio = AVAudioEngine()
    private var source: AVAudioSourceNode?
    private let session: WavWriter
    private var recording: WavWriter?
    private var levels = [Float](repeating: 0, count: 80)
    private var frames = 0
    private var stopped = false
    private let started = Date()

    init(session url: URL) throws {
        // Two seconds of audio before playback starts absorb timing spikes; four bound the look-ahead.
        ring = try MagentaAudioBuffer(capacityFrames: 4 * MagentaLive.sampleRate, startupFrames: 2 * MagentaLive.sampleRate)
        session = try WavWriter(url: url)
    }

    var isStopped: Bool {
        lock.lock()
        defer { lock.unlock() }
        return stopped
    }

    func startPlayback(muted: Bool) throws {
        let format = AVAudioFormat(standardFormatWithSampleRate: Double(MagentaLive.sampleRate), channels: 2)!
        let ring = ring
        let node = AVAudioSourceNode(format: format) { _, _, count, list in
            let channels = UnsafeMutableAudioBufferListPointer(list)
            guard channels.count == 2, let left = channels[0].mData?.assumingMemoryBound(to: Float.self),
                  let right = channels[1].mData?.assumingMemoryBound(to: Float.self) else { return noErr }
            ring.read(left: left, right: right, frames: Int(count))
            return noErr
        }
        source = node
        audio.attach(node)
        audio.connect(node, to: audio.mainMixerNode, format: format)
        if muted { audio.mainMixerNode.outputVolume = 0 }
        audio.prepare()
        try audio.start()
    }

    func write(_ pcm: Data) throws {
        lock.lock()
        session.append(pcm)
        recording?.append(pcm)
        frames += 1
        levels.removeFirst()
        levels.append(Self.level(pcm))
        lock.unlock()
        // Only this producer waits when it is ahead; the render thread never waits on it.
        var offset = 0
        while offset < pcm.count / 4 && !isStopped {
            let written = ring.writePCM(pcm, frameOffset: offset)
            offset += written
            if written == 0 { Thread.sleep(forTimeInterval: 0.005) }
        }
    }

    var underrunFrames: UInt64 { ring.underrunFrames }

    func stop() {
        lock.lock()
        stopped = true
        lock.unlock()
        ring.stop()
    }

    func stopPlayback() {
        audio.stop()
        session.close()
    }

    func startRecording(to url: URL) throws {
        let writer = try WavWriter(url: url)
        lock.lock()
        recording = writer
        lock.unlock()
    }

    /// Closes the recording and returns its file, if one was running.
    func stopRecording() -> URL? {
        lock.lock()
        let writer = recording
        recording = nil
        lock.unlock()
        writer?.close()
        return writer?.url
    }

    func exportSession(to url: URL) throws {
        lock.lock()
        defer { lock.unlock() }
        try session.copy(to: url)
    }

    func snapshot() -> (played: Double, levels: [Float], speed: Double) {
        lock.lock()
        defer { lock.unlock() }
        let made = Double(frames) / 25
        let played = max(0, made - Double(ring.bufferedFrames) / Double(MagentaLive.sampleRate))
        return (played, levels, made / max(0.1, Date().timeIntervalSince(started)))
    }

    private static func level(_ pcm: Data) -> Float {
        pcm.withUnsafeBytes { bytes in
            let samples = bytes.bindMemory(to: Int16.self)
            var sum: Float = 0
            for i in stride(from: 0, to: samples.count, by: 4) {
                let v = Float(Int16(littleEndian: samples[i])) / 32_768
                sum += v * v
            }
            let rms = sqrt(sum / Float(max(1, samples.count / 4)))
            return min(1, rms * 3.5)
        }
    }
}

/// 48 kHz 16-bit stereo WAV written as frames arrive; the header sizes are filled in on close (or on copy).
final class WavWriter: @unchecked Sendable {
    let url: URL
    private let handle: FileHandle
    private var bytes: UInt32 = 0
    private var closed = false

    init(url: URL) throws {
        self.url = url
        try? FileManager.default.removeItem(at: url)
        FileManager.default.createFile(atPath: url.path, contents: Self.header(dataBytes: 0))
        handle = try FileHandle(forWritingTo: url)
        handle.seekToEndOfFile()
    }

    func append(_ pcm: Data) {
        guard !closed else { return }
        handle.write(pcm)
        bytes &+= UInt32(pcm.count)
    }

    func close() {
        guard !closed else { return }
        closed = true
        handle.seek(toFileOffset: 0)
        handle.write(Self.header(dataBytes: bytes))
        try? handle.close()
    }

    /// A finished copy of what is written so far, while writing goes on.
    func copy(to target: URL) throws {
        if !closed { handle.synchronizeFile() }
        try? FileManager.default.removeItem(at: target)
        try FileManager.default.copyItem(at: url, to: target)
        let out = try FileHandle(forWritingTo: target)
        out.write(Self.header(dataBytes: bytes))
        try out.close()
    }

    static func header(dataBytes: UInt32) -> Data {
        var data = Data()
        func u32(_ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { data.append(contentsOf: $0) } }
        func u16(_ v: UInt16) { withUnsafeBytes(of: v.littleEndian) { data.append(contentsOf: $0) } }
        let rate = UInt32(MagentaLive.sampleRate)
        data.append(contentsOf: Array("RIFF".utf8)); u32(36 &+ dataBytes)
        data.append(contentsOf: Array("WAVE".utf8))
        data.append(contentsOf: Array("fmt ".utf8)); u32(16); u16(1); u16(2); u32(rate); u32(rate * 4); u16(4); u16(16)
        data.append(contentsOf: Array("data".utf8)); u32(dataBytes)
        return data
    }
}
