import AVFoundation
import MagentaRuntime
import SwiftUI

@MainActor
final class AudioRecorder: NSObject, ObservableObject, AVAudioRecorderDelegate {
    @Published var isRecording = false
    @Published var isPreparing = false
    @Published var lastRecordedURL: URL?

    private var recorder: AVAudioRecorder?
    private var meterTimer: Timer?
    private var silenceStart: Date?
    private var finishHandler: ((URL) -> Void)?

    var silenceThresholdDb: Float = -45.0
    var silenceDuration: TimeInterval = 1.2

    func startRecording(outputURL: URL, autoStopAfterSilence: Bool, isFloat32Wav: Bool = false, onFinish: ((URL) -> Void)? = nil) async -> Bool {
        guard !isRecording, !isPreparing else { return false }
        isPreparing = true
        finishHandler = onFinish

        let micOK = await AVAudioApplication.requestRecordPermission()
        guard micOK else {
            isPreparing = false
            return false
        }

        #if os(iOS)
        let sessionConfigured = await MainActor.run {
            let session = AVAudioSession.sharedInstance()
            do {
                try session.setCategory(.playAndRecord, mode: .spokenAudio, options: [.defaultToSpeaker, .allowBluetoothHFP])
                try session.setActive(true)
                return true
            } catch {
                return false
            }
        }
        #else
        let sessionConfigured = true
        #endif

        guard sessionConfigured else {
            isPreparing = false
            return false
        }

        let settings: [String: Any]
        if isFloat32Wav {
            settings = [
                AVFormatIDKey: Int(kAudioFormatLinearPCM),
                AVSampleRateKey: 16000.0,
                AVNumberOfChannelsKey: 1,
                AVLinearPCMBitDepthKey: 32,
                AVLinearPCMIsFloatKey: true,
                AVLinearPCMIsBigEndianKey: false,
                AVLinearPCMIsNonInterleaved: false
            ]
        } else {
            settings = [
                AVFormatIDKey: Int(kAudioFormatMPEG4AAC),
                AVSampleRateKey: 16000,
                AVNumberOfChannelsKey: 1,
                AVEncoderAudioQualityKey: AVAudioQuality.high.rawValue
            ]
        }

        do {
            if FileManager.default.fileExists(atPath: outputURL.path) {
                try FileManager.default.removeItem(at: outputURL)
            }
            let recorder = try AVAudioRecorder(url: outputURL, settings: settings)
            recorder.isMeteringEnabled = true
            recorder.delegate = self
            recorder.prepareToRecord()
            recorder.record()

            self.recorder = recorder
            self.lastRecordedURL = outputURL
            self.isRecording = true
            self.isPreparing = false
            self.silenceStart = nil

            if autoStopAfterSilence {
                startMeteringTimer()
            }
            return true
        } catch {
            isPreparing = false
            return false
        }
    }

    func stopRecording() -> URL? {
        guard let recorder = recorder else { return nil }
        stopMeteringTimer()
        recorder.stop()
        self.recorder = nil
        self.isRecording = false
        self.isPreparing = false
        let url = recorder.url
        lastRecordedURL = url
        finishHandler?(url)
        finishHandler = nil
        return url
    }

    func cancelRecording() {
        stopMeteringTimer()
        recorder?.stop()
        recorder = nil
        isRecording = false
        isPreparing = false
        finishHandler = nil
    }

    private func startMeteringTimer() {
        stopMeteringTimer()
        meterTimer = Timer.scheduledTimer(withTimeInterval: 0.2, repeats: true) { [weak self] _ in
            DispatchQueue.main.async {
                guard let self = self, let recorder = self.recorder else { return }
                guard recorder.currentTime >= 1.5 else {
                    self.silenceStart = nil
                    return
                }
                recorder.updateMeters()
                let power = recorder.averagePower(forChannel: 0)
                if power < self.silenceThresholdDb {
                    if self.silenceStart == nil {
                        self.silenceStart = Date()
                    } else if let start = self.silenceStart,
                              Date().timeIntervalSince(start) >= self.silenceDuration {
                        _ = self.stopRecording()
                    }
                } else {
                    self.silenceStart = nil
                }
            }
        }
    }

    private func stopMeteringTimer() {
        meterTimer?.invalidate()
        meterTimer = nil
        silenceStart = nil
    }
}

func prepareGemmaAudioInput(from sourceURL: URL, destinationDirectory: URL, filePrefix: String) -> URL? {
    let destinationURL = destinationDirectory
        .appendingPathComponent("\(filePrefix)_\(UUID().uuidString)")
        .appendingPathExtension("wav")

    do {
        return try convertAudioFileToWav(sourceURL: sourceURL, destinationURL: destinationURL)
    } catch {
        NSLog("[LLMHub][Audio] Failed to prepare Gemma audio input: \(error.localizedDescription)")
        return nil
    }
}

func convertAudioFileToWav(sourceURL: URL, destinationURL: URL) throws -> URL {
    let accessing = sourceURL.startAccessingSecurityScopedResource()
    defer {
        if accessing {
            sourceURL.stopAccessingSecurityScopedResource()
        }
    }

    let inputFile = try AVAudioFile(forReading: sourceURL)

    guard let outputFormat = AVAudioFormat(
        commonFormat: .pcmFormatFloat32,
        sampleRate: 16_000,
        channels: 1,
        interleaved: false
    ) else {
        throw NSError(
            domain: "LLMHubAudioConversion",
            code: -1,
            userInfo: [NSLocalizedDescriptionKey: "Unable to create output audio format"]
        )
    }

    guard let converter = AVAudioConverter(from: inputFile.processingFormat, to: outputFormat) else {
        throw NSError(
            domain: "LLMHubAudioConversion",
            code: -2,
            userInfo: [NSLocalizedDescriptionKey: "Unable to create audio converter"]
        )
    }

    if FileManager.default.fileExists(atPath: destinationURL.path) {
        try FileManager.default.removeItem(at: destinationURL)
    }

    let outputFile = try AVAudioFile(forWriting: destinationURL, settings: outputFormat.settings)
    let inputFrameCapacity: AVAudioFrameCount = 4096
    let inputBuffer = AVAudioPCMBuffer(pcmFormat: inputFile.processingFormat, frameCapacity: inputFrameCapacity)!

    var reachedEndOfStream = false

    while true {
        let outputFrameCapacity = max(
            inputFrameCapacity,
            AVAudioFrameCount((Double(inputFrameCapacity) * outputFormat.sampleRate / inputFile.processingFormat.sampleRate).rounded(.up)) + 16
        )
        guard let outputBuffer = AVAudioPCMBuffer(pcmFormat: outputFormat, frameCapacity: outputFrameCapacity) else {
            throw NSError(
                domain: "LLMHubAudioConversion",
                code: -3,
                userInfo: [NSLocalizedDescriptionKey: "Unable to create output buffer"]
            )
        }

        var conversionError: NSError?
        let status = converter.convert(to: outputBuffer, error: &conversionError) { _, outputStatus in
            if reachedEndOfStream {
                outputStatus.pointee = .endOfStream
                return nil
            }

            do {
                try inputFile.read(into: inputBuffer)
            } catch {
                reachedEndOfStream = true
                outputStatus.pointee = .endOfStream
                return nil
            }

            if inputBuffer.frameLength == 0 {
                reachedEndOfStream = true
                outputStatus.pointee = .endOfStream
                return nil
            }

            outputStatus.pointee = .haveData
            return inputBuffer
        }

        if let conversionError {
            throw conversionError
        }

        if outputBuffer.frameLength > 0 {
            try outputFile.write(from: outputBuffer)
        }

        if status == .endOfStream {
            break
        }
    }

    return destinationURL
}

@MainActor
final class AudioPlaybackController: NSObject, ObservableObject, @preconcurrency AVAudioPlayerDelegate {
    @Published var isPlaying = false

    private var player: AVAudioPlayer?

    func toggle(url: URL) {
        if isPlaying {
            stop()
            return
        }

        do {
            player = try AVAudioPlayer(contentsOf: url)
            player?.delegate = self
            player?.prepareToPlay()
            player?.play()
            isPlaying = true
        } catch {
            stop()
        }
    }

    func stop() {
        player?.stop()
        player = nil
        isPlaying = false
    }

    func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        stop()
    }
}

struct AudioPlaybackButton: View {
    let url: URL
    @StateObject private var controller = AudioPlaybackController()

    var body: some View {
        Button {
            controller.toggle(url: url)
        } label: {
            Image(systemName: controller.isPlaying ? "stop.fill" : "play.fill")
                .font(.system(size: 14, weight: .bold))
                .frame(width: 36, height: 36)
        }
        .audioToolsIconButtonStyle(cornerRadius: 12)
    }
}

private extension View {
    func audioToolsIconButtonStyle(cornerRadius: CGFloat = 10) -> some View {
        self
            .foregroundStyle(.white)
            .background(
                RoundedRectangle(cornerRadius: cornerRadius)
                    .fill(Color.white.opacity(0.08))
            )
            .overlay(
                RoundedRectangle(cornerRadius: cornerRadius)
                    .stroke(Color.white.opacity(0.16), lineWidth: 1)
            )
    }
}

/// Stop is graceful (save the recording); task cancellation aborts the run.
private final class MusicGenerationStop: @unchecked Sendable {
    private let lock = NSLock()
    private var requested = false

    var isRequested: Bool {
        lock.lock()
        defer { lock.unlock() }
        return requested
    }

    func request() {
        lock.lock()
        requested = true
        lock.unlock()
    }
}

/// Accessed only by the generation worker. Stream to disk rather than retaining
/// an ever-growing PCM Data buffer during an unlimited session.
private final class MusicPCMWriter: @unchecked Sendable {
    private var file: AVAudioFile?
    private let format = AVAudioFormat(commonFormat: .pcmFormatInt16, sampleRate: 48_000,
                                      channels: 2, interleaved: true)!
    private(set) var sampleCount: Int64 = 0

    init(url: URL) throws {
        file = try AVAudioFile(forWriting: url, settings: format.settings,
                               commonFormat: .pcmFormatInt16, interleaved: true)
    }

    func append(_ pcm: Data) throws {
        let count = AVAudioFrameCount(pcm.count / 4)
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: count),
              let destination = buffer.mutableAudioBufferList.pointee.mBuffers.mData else {
            throw POSIXError(.ENOMEM)
        }
        buffer.frameLength = count
        pcm.withUnsafeBytes { bytes in
            if let source = bytes.baseAddress { destination.copyMemory(from: source, byteCount: pcm.count) }
        }
        try file?.write(from: buffer)
        sampleCount += Int64(count)
    }

    func finish() { file = nil }
}

/// A continuous render source, independent of inference and UI callbacks.
/// Two seconds of startup audio absorb timing spikes; the four-second ring
/// bounds lookahead without locking or allocating on the audio render thread.
private final class MusicLivePlayback: @unchecked Sendable {
    private let engine = AVAudioEngine()
    private let controlQueue = DispatchQueue(label: "LLMHub.MusicPlayback.Control")
    private let format = AVAudioFormat(standardFormatWithSampleRate: 48_000, channels: 2)!
    private let buffer: MagentaAudioBuffer
    private let source: AVAudioSourceNode

    init() throws {
        let buffer = try MagentaAudioBuffer(capacityFrames: 4 * 48_000, startupFrames: 2 * 48_000)
        self.buffer = buffer
        source = AVAudioSourceNode(format: format) { _, _, frames, audio in
            let channels = UnsafeMutableAudioBufferListPointer(audio)
            guard channels.count == 2,
                  let left = channels[0].mData?.assumingMemoryBound(to: Float.self),
                  let right = channels[1].mData?.assumingMemoryBound(to: Float.self) else {
                for channel in channels {
                    channel.mData?.initializeMemory(as: UInt8.self, repeating: 0, count: Int(channel.mDataByteSize))
                }
                return noErr
            }
            buffer.read(left: left, right: right, frames: Int(frames))
            return noErr
        }
        #if os(iOS)
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playback, mode: .default)
        try session.setPreferredIOBufferDuration(0.02)
        try session.setActive(true)
        #endif
        engine.attach(source)
        engine.connect(source, to: engine.mainMixerNode, format: format)
        engine.prepare()
        try engine.start()
    }

    func enqueue(_ pcm: Data) throws {
        var offset = 0
        while offset < pcm.count / 4 && !buffer.isStopped {
            try Task.checkCancellation()
            guard engine.isRunning else { throw CancellationError() }
            let written = buffer.writePCM(pcm, frameOffset: offset)
            offset += written
            // Only the producer waits when ahead. Rendering never waits on it.
            if written == 0 { Thread.sleep(forTimeInterval: 0.005) }
        }
    }

    func drain() throws {
        buffer.finish() // Also releases short clips below the startup threshold.
        while buffer.bufferedFrames > 0 && !buffer.isStopped {
            try Task.checkCancellation()
            guard engine.isRunning else { throw CancellationError() }
            Thread.sleep(forTimeInterval: 0.005)
        }
        // Empty storage means samples entered the render pipeline. Let the last
        // hardware buffer reach the speaker before normal completion tears it down.
        let deadline = ProcessInfo.processInfo.systemUptime + source.outputPresentationLatency + 0.04
        while !buffer.isStopped && ProcessInfo.processInfo.systemUptime < deadline {
            try Task.checkCancellation()
            Thread.sleep(forTimeInterval: 0.005)
        }
        try Task.checkCancellation()
    }

    func stop() {
        buffer.stop()
        controlQueue.async { [self] in
            engine.stop()
        }
    }

    func logTiming(generatedSeconds: Double, elapsedSeconds: Double) {
        NSLog("[LLMHub][MusicGen] generated=%.2fs elapsed=%.2fs buffered=%.2fs underrun=%.3fs",
              generatedSeconds, elapsedSeconds, Double(buffer.bufferedFrames) / 48_000,
              Double(buffer.underrunFrames) / 48_000)
    }
}

@MainActor
public final class MusicGeneratorBackend: ObservableObject {
    public static let shared = MusicGeneratorBackend()

    @Published public var isGenerating: Bool = false
    @Published public var progress: Double = 0.0
    @Published public private(set) var generatedDurationSeconds: Double = 0.0
    @Published public var generatedAudioURL: URL? = nil
    @Published public var errorMessage: String? = nil
    @Published public private(set) var isLoaded: Bool = false
    @Published public private(set) var loadedModelName: String? = nil

    private var loadedSession: MagentaRealtimeEngine.Session?
    private var loadedResourceDirectory: URL?
    private var livePlayback: MusicLivePlayback?
    private var generationTask: Task<(URL, Double), Error>?
    private var generationStop: MusicGenerationStop?
    private var loadTask: Task<MagentaRealtimeEngine.Session, Error>?

    private struct ModelArtifacts: Sendable {
        let functionURL: URL
        let stateURL: URL
        let resourceDirectory: URL
    }

    private init() {}

    public func loadModel(modelName: String) async -> Bool {
        guard !isGenerating, loadTask == nil else { return false }
        defer { loadTask = nil }
        if isLoaded, loadedModelName == modelName, loadedSession != nil {
            return true
        }
        errorMessage = nil
        do {
            let artifacts = try Self.resolveArtifacts(modelName: modelName)
            let task = Task.detached(priority: .userInitiated) {
                try Task.checkCancellation()
                let session = try MagentaRealtimeEngine.load(
                    functionURL: artifacts.functionURL,
                    stateURL: artifacts.stateURL
                )
                try Task.checkCancellation()
                return session
            }
            loadTask = task
            let session = try await task.value
            loadTask = nil
            loadedSession = session
            loadedResourceDirectory = artifacts.resourceDirectory
            loadedModelName = modelName
            isLoaded = true
            NSLog("[LLMHub][MusicGen] Loaded model: \(modelName)")
            return true
        } catch {
            unloadModel()
            errorMessage = error.localizedDescription
            NSLog("[LLMHub][MusicGen] Load error: \(error.localizedDescription)")
            return false
        }
    }

    public func unloadModel() {
        loadTask?.cancel()
        generationStop?.request()
        generationTask?.cancel()
        livePlayback?.stop()
        loadedSession = nil
        loadedResourceDirectory = nil
        loadedModelName = nil
        isLoaded = false
        progress = 0
        NSLog("[LLMHub][MusicGen] Unloaded model")
    }

    public func stopLiveGeneration() {
        generationStop?.request()
        livePlayback?.stop()
    }

    public func generateMusic(
        modelName: String,
        prompt: String,
        durationSeconds: Double,
        live: Bool = false,
        unlimited: Bool = false,
        seed: UInt64? = nil
    ) async -> URL? {
        guard !isGenerating, generationTask == nil else { return nil }
        guard await loadModel(modelName: modelName),
              let session = loadedSession,
              let resourceDirectory = loadedResourceDirectory else {
            return nil
        }
        isGenerating = true
        progress = 0.05
        generatedDurationSeconds = 0
        errorMessage = nil
        generatedAudioURL = nil
        defer {
            livePlayback = nil
            generationTask = nil
            generationStop = nil
            isGenerating = false
        }

        let documentsDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let runsUntilStopped = live && unlimited
        // CAF supports long recordings without WAV's 32-bit file size limit.
        let fileExtension = runsUntilStopped ? "caf" : "wav"
        let outputURL = documentsDir.appendingPathComponent("generated_music_\(UUID().uuidString).\(fileExtension)")
        do {
            let stop = MusicGenerationStop()
            generationStop = stop
            let task = Task.detached(priority: .userInitiated) { () throws -> (URL, Double) in
                try Task.checkCancellation()
                // Audio-session activation and graph setup can block. They belong
                // on this worker, before the first inference, rather than MainActor.
                let playback = live ? try MusicLivePlayback() : nil
                defer { playback?.stop() }
                await MainActor.run {
                    let backend = MusicGeneratorBackend.shared
                    if backend.generationStop === stop && !stop.isRequested {
                        backend.livePlayback = playback
                    } else {
                        playback?.stop()
                    }
                }
                try Task.checkCancellation()
                let writer = try MusicPCMWriter(url: outputURL)
                let generationStart = ProcessInfo.processInfo.systemUptime
                var completed = false
                defer {
                    writer.finish()
                    if !completed { try? FileManager.default.removeItem(at: outputURL) }
                }
                _ = try MagentaRealtimeEngine.generate(
                    session: session,
                    prompt: prompt,
                    resourceDirectory: resourceDirectory,
                    durationSeconds: runsUntilStopped ? nil : durationSeconds,
                    seed: seed,
                    collectAudio: false,
                    shouldStop: { stop.isRequested },
                    onAudioFrame: { pcm in
                        try writer.append(pcm)
                        try playback?.enqueue(pcm)
                        // Refresh elapsed time twice per second, away from the audio callback.
                        if writer.sampleCount % 240_000 == 0 {
                            playback?.logTiming(generatedSeconds: Double(writer.sampleCount) / 48_000,
                                elapsedSeconds: ProcessInfo.processInfo.systemUptime - generationStart)
                        }
                        if writer.sampleCount % 24_960 == 0 {
                            let seconds = Double(writer.sampleCount) / 48_000
                            Task { @MainActor in
                                let backend = MusicGeneratorBackend.shared
                                if backend.generationStop === stop {
                                    backend.generatedDurationSeconds = seconds
                                }
                            }
                        }
                    },
                    progress: { fraction in
                        Task { @MainActor in
                            let backend = MusicGeneratorBackend.shared
                            if backend.generationStop === stop {
                                backend.progress = 0.05 + fraction * 0.90
                            }
                        }
                    }
                )
                try playback?.drain()
                playback?.logTiming(generatedSeconds: Double(writer.sampleCount) / 48_000,
                    elapsedSeconds: ProcessInfo.processInfo.systemUptime - generationStart)
                try Task.checkCancellation()
                guard writer.sampleCount > 0 else { throw CancellationError() }
                completed = true
                return (outputURL, Double(writer.sampleCount) / 48_000)
            }
            generationTask = task
            let (result, duration) = try await task.value
            generatedDurationSeconds = duration
            progress = 1.0
            generatedAudioURL = result
            return result
        } catch is CancellationError {
            progress = 0
            return nil
        } catch {
            errorMessage = error.localizedDescription
            NSLog("[LLMHub][MusicGen] Generation error: \(error.localizedDescription)")
            return nil
        }
    }

    nonisolated private static func resolveArtifacts(modelName: String) throws -> ModelArtifacts {
        let documentsDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let modelDirName = modelName.replacingOccurrences(of: " ", with: "_")
            .replacingOccurrences(of: "[^a-zA-Z0-9_.-]", with: "", options: .regularExpression)
        let selectedModelID = modelName.lowercased().contains("base")
            ? "magenta_realtime_2_base"
            : "magenta_realtime_2_small"
        let searchDirs = [
            documentsDir.appendingPathComponent("RunAnywhere/Models/FoundationModels/\(selectedModelID)"),
            documentsDir.appendingPathComponent("RunAnywhere/Models/FoundationModels/\(modelDirName)"),
            documentsDir.appendingPathComponent("RunAnywhere/Models/FoundationModels"),
            documentsDir.appendingPathComponent("RunAnywhere/Models/\(modelDirName)"),
            documentsDir.appendingPathComponent("RunAnywhere/Models"),
            documentsDir
        ]
        let artifactPrefix = selectedModelID.hasSuffix("base") ? "mrt2_base" : "mrt2_small"
        var functionURL: URL?
        var stateURL: URL?
        var resourceDirectory: URL?
        for directory in searchDirs where FileManager.default.fileExists(atPath: directory.path) {
            guard let enumerator = FileManager.default.enumerator(
                at: directory,
                includingPropertiesForKeys: nil
            ) else { continue }
            while let fileURL = enumerator.nextObject() as? URL {
                if fileURL.lastPathComponent == "\(artifactPrefix).mlxfn", functionURL == nil { functionURL = fileURL }
                if fileURL.lastPathComponent == "\(artifactPrefix)_state.safetensors", stateURL == nil { stateURL = fileURL }
                if fileURL.lastPathComponent == "spm.model", resourceDirectory == nil {
                    resourceDirectory = fileURL.deletingLastPathComponent()
                }
            }
            if functionURL != nil, stateURL != nil, resourceDirectory != nil { break }
        }
        guard let functionURL, let stateURL, let resourceDirectory else {
            throw NSError(
                domain: "LLMHubMusic",
                code: -404,
                userInfo: [NSLocalizedDescriptionKey: "Magenta model, state, or MusicCoCa prompt resources are missing"]
            )
        }
        return ModelArtifacts(
            functionURL: functionURL,
            stateURL: stateURL,
            resourceDirectory: resourceDirectory
        )
    }
}
