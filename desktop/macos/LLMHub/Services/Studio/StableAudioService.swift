import Foundation
import Observation
import StableAudioMac

/// Studio › Music clips: fixed-length text to music with Stable Audio 3 (Small-Music, and Medium on Macs with 16 GB or
/// more) on the Apple GPU, next to the live Magenta RealTime jam. Weights come from Stability's ungated
/// stable-audio-3-optimized repo on first use; clips are saved as 16-bit WAV to ~/Pictures/LLM Hub/Music.
/// Powered by Stability AI: the models are under the Stability AI Community License (see Licenses/ in the app bundle).
@MainActor @Observable
final class StableAudioService {
    var prompt = "Warm lo-fi hip hop with dusty drums, Rhodes piano and vinyl crackle, 85 BPM"
    var seconds = 30.0
    /// A number, or empty for a random seed.
    var seed = ""
    var model = StableAudio3.Model.smallMusic {
        didSet { seconds = min(seconds, model.maxSeconds) }
    }
    var isGenerating = false
    var progress = 0.0
    var status = ""
    /// The clip just made, for the "Show in Finder" and play buttons.
    var lastClip: URL?
    /// Bumped when files land on disk so the download note refreshes.
    private var revision = 0

    @ObservationIgnored private var run: Task<Void, Never>?
    @ObservationIgnored private let cancelFlag = CancelFlag()
    @ObservationIgnored private lazy var engine = StableAudio3(folder: Self.folder)

    nonisolated static var folder: URL { AppPaths.engines.appendingPathComponent("stable-audio-3", isDirectory: true) }

    /// Medium needs about 5 GB of unified memory while it runs; offer it from 16 GB.
    static var models: [StableAudio3.Model] {
        ProcessInfo.processInfo.physicalMemory >= 16 << 30 ? [.smallMusic, .medium] : [.smallMusic]
    }

    static func name(_ model: StableAudio3.Model) -> String {
        model == .smallMusic ? "Stable Audio 3 Small-Music" : "Stable Audio 3 Medium"
    }

    var isInstalled: Bool {
        _ = revision
        return StableAudio3.isInstalled(model, in: Self.folder)
    }

    /// Bytes still to download for the chosen model.
    var downloadNote: String {
        _ = revision
        let missing = model.files.filter { !Self.hasFile($0) }.reduce(Int64(0)) { $0 + $1.size }
        return missing > 0 ? L.f("StudioDownloadNote", Fmt.bytes(missing)) : ""
    }

    nonisolated private static func hasFile(_ file: (name: String, size: Int64)) -> Bool {
        let size = (try? FileManager.default.attributesOfItem(atPath: folder.appendingPathComponent(file.name).path)[.size] as? NSNumber)?.int64Value
        return size == file.size
    }

    // MARK: Generate

    func generate(_ music: MusicService) {
        guard !isGenerating else { return }
        run = Task {
            do {
                _ = try await render(prompt: prompt, seconds: seconds, seed: UInt64(seed.trimmingCharacters(in: .whitespaces)), model: model)
                music.loadRecordings()
            } catch {}
        }
    }

    func cancel() {
        cancelFlag.set(true)
        run?.cancel()
    }

    /// Makes one clip and saves it. Also the --cliptest hook's entry point.
    func render(prompt: String, seconds: Double, seed: UInt64?, model: StableAudio3.Model) async throws
        -> (url: URL, seconds: Double, took: Double, seed: UInt64, timings: StableAudio3.Timings, peak: Float, rms: Float) {
        let text = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { throw ComfyService.error(L.s("ClipNeedsPrompt")) }
        isGenerating = true
        progress = 0
        cancelFlag.set(false)
        idleUnload?.cancel()
        let activity = ProcessInfo.processInfo.beginActivity(options: [.userInitiated, .idleSystemSleepDisabled], reason: "Stable Audio 3 clip")
        defer {
            ProcessInfo.processInfo.endActivity(activity)
            isGenerating = false
            progress = 0
        }
        do {
            try await download(model)
            let seed = seed ?? UInt64.random(in: 0 ..< (1 << 31))
            let length = min(max(1, seconds), model.maxSeconds)
            let clock = Date()
            let engine = engine
            let flag = cancelFlag
            status = L.s("ClipLoading")
            let (samples, timings) = try await Task.detached(priority: .userInitiated) {
                try engine.generate(model, prompt: text, seconds: length, seed: seed, stage: { stage in
                    Task { @MainActor [weak self] in self?.show(stage) }
                }, isCancelled: { flag.isSet })
            }.value
            let took = Date().timeIntervalSince(clock)
            let url = Self.clipURL(prompt: text, seed: seed)
            let (peak, rms) = try await Task.detached { try Self.writeWav(samples, to: url) }.value
            lastClip = url
            status = L.f("ClipDone", String(format: "%.0f", length), String(format: "%.1f", took))
            scheduleUnload()
            Log.info(String(format: "Stable Audio 3 %@ %.0fs seed %llu in %.1fs (load %.1f, encode %.2f, sample %.1f, decode %.1f): %@",
                            model.rawValue, length, seed, took, timings.load, timings.encode, timings.sample, timings.decode, url.path))
            return (url, length, took, seed, timings, peak, rms)
        } catch is CancellationError {
            status = ""
            throw CancellationError()
        } catch {
            Log.error("Stable Audio 3 failed", error)
            status = L.f("ErrorPrefix", error.localizedDescription)
            throw error
        }
    }

    private func show(_ stage: StableAudio3.Stage) {
        guard isGenerating else { return }
        switch stage {
        case .loading: status = L.s("ClipLoading")
        case .encoding: status = L.s("ClipEncoding"); progress = 0.05
        case .sampling(let step, let total):
            status = L.f("ComfyStep", "\(step)", "\(total)")
            progress = 0.05 + 0.8 * Double(step - 1) / Double(total)
        case .decoding: status = L.s("ClipDecoding"); progress = 0.9
        }
    }

    /// Frees the models' memory (Settings › Studio backends, or when Studio is left alone).
    func unload() { engine.unload() }

    @ObservationIgnored private var idleUnload: Task<Void, Never>?

    /// The models stay in memory for the next clip (about 2 GB for Small, 5 GB for Medium) and are freed after
    /// five quiet minutes.
    private func scheduleUnload() {
        idleUnload?.cancel()
        idleUnload = Task { [weak self] in
            try? await Task.sleep(for: .seconds(300))
            guard let self, !Task.isCancelled, !self.isGenerating else { return }
            let engine = self.engine
            await Task.detached { engine.unload() }.value
            Log.info("Stable Audio 3 unloaded after 5 idle minutes")
        }
    }

    /// Deletes the downloaded weights.
    func remove() {
        engine.unload()
        try? FileManager.default.removeItem(at: Self.folder)
        revision += 1
    }

    private func download(_ model: StableAudio3.Model) async throws {
        guard !StableAudio3.isInstalled(model, in: Self.folder) else { return }
        try FileManager.default.createDirectory(at: Self.folder, withIntermediateDirectories: true)
        var done: Int64 = 0
        let total = model.downloadSize
        for file in model.files {
            if Self.hasFile(file) {
                done += file.size
                continue
            }
            let before = done
            let target = Self.folder.appendingPathComponent(file.name)
            Log.info("Downloading \(file.name) (\(Fmt.bytes(file.size)))")
            try await ComfyService.resumableDownload(request: URLRequest(url: URL(string: StableAudio3.repoBase + file.name)!), to: target,
                                                       expectedSize: file.size) { bytes in
                Task { @MainActor in
                    self.status = L.f("ClipDownloading", String(format: "%.0f", min(99, Double(before + bytes) * 100 / Double(total))))
                    self.progress = 0
                }
            }
            if Task.isCancelled || cancelFlag.isSet { throw CancellationError() }
            done += file.size
            revision += 1
        }
        Log.info("\(Self.name(model)) downloaded (\(Fmt.bytes(total)))")
    }

    nonisolated private static func clipURL(prompt: String, seed: UInt64) -> URL {
        let stamp = DateFormatter()
        stamp.dateFormat = "yyyyMMdd-HHmmss"
        let words = prompt.components(separatedBy: CharacterSet.alphanumerics.inverted).filter { !$0.isEmpty }.prefix(5).joined(separator: "-")
        return MusicService.outputFolder.appendingPathComponent("\(stamp.string(from: Date()))-sa3-\(words.isEmpty ? "clip" : words)-\(seed).wav")
    }

    /// 16-bit stereo WAV at 44.1 kHz from interleaved samples (clipped to -1...1, as sa3_mlx.py's save_wav). Returns the
    /// peak and RMS, so tests can tell real audio from silence.
    nonisolated static func writeWav(_ samples: [Float], to url: URL) throws -> (peak: Float, rms: Float) {
        var pcm = Data(count: samples.count * 2)
        var peak: Float = 0
        var sum: Double = 0
        pcm.withUnsafeMutableBytes { raw in
            let out = raw.bindMemory(to: Int16.self)
            for (i, s) in samples.enumerated() {
                guard s.isFinite else { out[i] = 0; continue }
                peak = max(peak, abs(s))
                sum += Double(s * s)
                out[i] = Int16(max(-1, min(1, s)) * 32767).littleEndian
            }
        }
        var data = Data()
        func u32(_ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { data.append(contentsOf: $0) } }
        func u16(_ v: UInt16) { withUnsafeBytes(of: v.littleEndian) { data.append(contentsOf: $0) } }
        let rate = UInt32(StableAudio3.sampleRate)
        data.append(contentsOf: Array("RIFF".utf8)); u32(36 + UInt32(pcm.count))
        data.append(contentsOf: Array("WAVE".utf8))
        data.append(contentsOf: Array("fmt ".utf8)); u32(16); u16(1); u16(2); u32(rate); u32(rate * 4); u16(4); u16(16)
        data.append(contentsOf: Array("data".utf8)); u32(UInt32(pcm.count))
        try (data + pcm).write(to: url, options: .atomic)
        return (peak, Float((sum / Double(max(1, samples.count))).squareRoot()))
    }
}

/// A cancel switch the generation thread can read.
final class CancelFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var value = false
    var isSet: Bool { lock.withLock { value } }
    func set(_ on: Bool) { lock.withLock { value = on } }
}
