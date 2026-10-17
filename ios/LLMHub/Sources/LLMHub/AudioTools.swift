import AVFoundation
import BackgroundTasks
import MediaPlayer
import UIKit
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
    private let allowsBackgroundPlayback: Bool
    private var preparationTask: Task<Void, Never>?

    init(allowsBackgroundPlayback: Bool = false) {
        self.allowsBackgroundPlayback = allowsBackgroundPlayback
        super.init()
    }

    func toggle(url: URL) {
        if isPlaying || preparationTask != nil {
            stop()
            return
        }

        guard allowsBackgroundPlayback else { play(url: url); return }
        preparationTask = Task { [weak self] in
            do {
                try await Task.detached(priority: .userInitiated) {
                    let session = AVAudioSession.sharedInstance()
                    try session.setCategory(.playback, mode: .default)
                    try session.setActive(true)
                }.value
                guard !Task.isCancelled, let self else { return }
                self.preparationTask = nil
                self.play(url: url)
            } catch {
                guard !Task.isCancelled else { return }
                self?.stop()
            }
        }
    }

    private func play(url: URL) {
        do {
            player = try AVAudioPlayer(contentsOf: url)
            player?.delegate = self
            player?.prepareToPlay()
            isPlaying = player?.play() == true
        } catch {
            stop()
        }
    }

    func stop() {
        preparationTask?.cancel()
        preparationTask = nil
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
    @StateObject private var controller: AudioPlaybackController

    init(url: URL, allowsBackgroundPlayback: Bool = false) {
        self.url = url
        _controller = StateObject(wrappedValue: AudioPlaybackController(allowsBackgroundPlayback: allowsBackgroundPlayback))
    }

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

// TEMPORARY: remove MusicDebug/MusicDiagnosticProbe and their call sites once
// the locked-screen generation failure has been identified and fixed.
private enum MusicDebug {
    static let enabled = false
    static let heartbeatEnabled = false

    static func log(_ message: @autoclosure () -> String) {
        guard enabled else { return }
        NSLog("[LLMHub][MusicDebug] t=%.3f %@", ProcessInfo.processInfo.systemUptime, message())
    }

    static func error(_ error: Error) -> String {
        let value = error as NSError
        let underlying = (value.userInfo[NSUnderlyingErrorKey] as? NSError)
            .map { " underlying=\($0.domain):\($0.code) \($0.localizedDescription)" } ?? ""
        return "domain=\(value.domain) code=\(value.code) \(value.localizedDescription)\(underlying)"
    }
}

/// A low-frequency diagnostic heartbeat, independent of MainActor and the audio
/// render callback. It never schedules playback or inference.
private final class MusicDiagnosticProbe: @unchecked Sendable {
    let id = String(UUID().uuidString.prefix(8))
    private let lock = NSLock()
    private var stage = "starting"
    private var stageSince = ProcessInfo.processInfo.systemUptime
    private var lastFrame: TimeInterval?
    private var seconds = 0.0
    private var playback: (@Sendable () -> String)?
    private var timer: DispatchSourceTimer?

    func event(_ message: String) { MusicDebug.log("run=\(id) \(message)") }

    func setStage(_ value: String) {
        lock.lock()
        if stage != value { stage = value; stageSince = ProcessInfo.processInfo.systemUptime }
        lock.unlock()
    }

    func frame(seconds: Double) {
        lock.lock()
        self.seconds = seconds
        lastFrame = ProcessInfo.processInfo.systemUptime
        lock.unlock()
    }

    func attach(_ playback: MusicLivePlayback) {
        lock.lock()
        self.playback = { [weak playback] in playback?.debugSnapshot ?? "playback=released" }
        lock.unlock()
        event("audio-created \(playback.debugSnapshot)")
    }

    func start() {
        guard MusicDebug.enabled && MusicDebug.heartbeatEnabled else { return }
        let timer = DispatchSource.makeTimerSource(queue: DispatchQueue(label: "LLMHub.MusicDebug", qos: .utility))
        timer.schedule(deadline: .now() + 2, repeating: 2, leeway: .milliseconds(200))
        timer.setEventHandler { [weak self] in self?.snapshot("heartbeat") }
        self.timer = timer
        timer.resume()
    }

    func snapshot(_ label: String) {
        lock.lock()
        let stage = self.stage
        let age = ProcessInfo.processInfo.systemUptime - stageSince
        let frameAge = lastFrame.map { ProcessInfo.processInfo.systemUptime - $0 } ?? -1
        let seconds = self.seconds
        let playback = self.playback
        lock.unlock()
        event(String(format: "%@ stage=%@ stageAge=%.2fs generated=%.2fs frameAge=%.2fs %@",
            label, stage, age, seconds, frameAge, playback?() ?? "playback=not-created"))
    }

    func finish(_ outcome: String) {
        timer?.cancel()
        timer = nil
        snapshot("end outcome=\(outcome)")
    }
}

/// Stop is graceful (save the recording); task cancellation aborts the run.
private final class MusicGenerationStop: @unchecked Sendable {
    private let condition = NSCondition()
    private var requested = false
    private var suspended = false
    private let diagnostic: MusicDiagnosticProbe?
    init(diagnostic: MusicDiagnosticProbe? = nil) { self.diagnostic = diagnostic }

    var isRequested: Bool {
        condition.lock()
        defer { condition.unlock() }
        return requested
    }

    func request(reason: String = "unspecified") {
        condition.lock()
        let first = !requested
        requested = true
        condition.broadcast()
        condition.unlock()
        if first { diagnostic?.event("stop-request reason=\(reason)") }
    }

    func setSuspended(_ value: Bool) {
        condition.lock()
        let changed = suspended != value
        suspended = value
        condition.broadcast()
        condition.unlock()
        if changed { diagnostic?.event("inference-gate suspended=\(value)") }
    }

    // Inference waits at frame boundaries, never on the audio render thread.
    func waitUntilRunnable() -> Bool {
        condition.lock()
        defer { condition.unlock() }
        let waited = suspended && !requested
        if waited {
            diagnostic?.setStage("gate-wait")
            diagnostic?.event("gate-wait-enter")
        }
        while suspended && !requested { condition.wait() }
        if waited { diagnostic?.event("gate-wait-exit stopped=\(requested)") }
        return !requested
    }
}

/// Use the operating system's capability instead of chip names or model IDs.
/// This includes supported M3-and-newer iPads and future supported generations.
/// The iOS app on Apple-silicon Macs uses macOS GPU execution rules.
@MainActor
enum MusicBackgroundSupport {
    static var runsOnMac: Bool {
        #if targetEnvironment(macCatalyst)
        true
        #else
        ProcessInfo.processInfo.isiOSAppOnMac
        #endif
    }

    static var isSupported: Bool {
        if runsOnMac { return true }
        guard UIDevice.current.userInterfaceIdiom == .pad else { return false }
        if #available(iOS 26.0, *) {
            return BGTaskScheduler.supportedResources.contains(.gpu)
        }
        return false
    }
}

/// On iPad a user-initiated GPU task must be granted before background Metal use.
/// A separate instance/identifier per run prevents late system callbacks from
/// attaching to a newer generation.
@MainActor
private final class MusicBackgroundSession {
    private var task: BGTask?
    private var pending: CheckedContinuation<Bool, Never>?
    private var identifier: String?
    private var ended = false
    private var expired = false
    private let onExpiration: @MainActor () -> Void
    private let diagnostic: MusicDiagnosticProbe?
    private(set) var allowsGPU = false
    private(set) var diagnosticStatus = "not-requested"

    init(diagnostic: MusicDiagnosticProbe? = nil, onExpiration: @escaping @MainActor () -> Void) {
        self.diagnostic = diagnostic
        self.onExpiration = onExpiration
    }

    func begin(title: String, subtitle: String) async -> Bool {
        diagnosticStatus = "requesting"
        diagnostic?.setStage("background-request")
        if MusicBackgroundSupport.runsOnMac {
            allowsGPU = true
            diagnosticStatus = "macos"
            return true
        }
        guard MusicBackgroundSupport.isSupported else {
            diagnosticStatus = "unsupported-device-or-os"
            return false
        }
        diagnostic?.event("scheduler-begin appState=\(UIApplication.shared.applicationState.rawValue) backgroundModes=\(Bundle.main.object(forInfoDictionaryKey: "UIBackgroundModes") ?? "missing") permittedIDs=\(Bundle.main.object(forInfoDictionaryKey: "BGTaskSchedulerPermittedIdentifiers") ?? "missing")")
        guard #available(iOS 26.0, *) else {
            diagnosticStatus = "unsupported-os"
            diagnostic?.event("scheduler-unavailable reason=os-version")
            NSLog("[LLMHub][MusicGen] Background GPU unavailable: requires iOS 26 or later; generation will pause while locked")
            return false
        }
        NSLog("[LLMHub][MusicGen] Background GPU resources=%lu iOS=%@ device=%@",
              BGTaskScheduler.supportedResources.rawValue, UIDevice.current.systemVersion, UIDevice.current.model)
        diagnostic?.event("scheduler-resources raw=\(BGTaskScheduler.supportedResources.rawValue) gpu=\(BGTaskScheduler.supportedResources.contains(.gpu))")
        guard BGTaskScheduler.supportedResources.contains(.gpu) else {
            diagnosticStatus = "unsupported-resources raw=\(BGTaskScheduler.supportedResources.rawValue)"
            NSLog("[LLMHub][MusicGen] iOS does not support background GPU on this device; generation will pause while locked")
            return false
        }
        guard UIApplication.shared.applicationState == .active else {
            diagnosticStatus = "app-not-active"
            diagnostic?.event("scheduler-unavailable reason=app-not-active")
            NSLog("[LLMHub][MusicGen] Background GPU request deferred: app is not active")
            return false
        }
        let id = "\(Bundle.main.bundleIdentifier!).music-generation.\(UUID().uuidString)"
        identifier = id
        diagnostic?.event("scheduler-register id=\(id)")
        let registered = BGTaskScheduler.shared.register(forTaskWithIdentifier: id, using: .main) { [weak self] task in
            MusicDebug.log("scheduler-launch-callback id=\(task.identifier) type=\(String(describing: type(of: task)))")
            Task { @MainActor in
                guard let self, !self.ended, self.pending != nil,
                      let continued = task as? BGContinuedProcessingTask else {
                    MusicDebug.log("scheduler-launch-discard id=\(task.identifier)")
                    task.setTaskCompleted(success: false)
                    return
                }
                self.task = continued
                continued.progress.totalUnitCount = 60_000
                continued.expirationHandler = { [weak self] in
                    Task { @MainActor in
                        guard let self, !self.ended else { return }
                        self.diagnostic?.event("scheduler-expired id=\(id)")
                        self.allowsGPU = false
                        self.expired = true
                        self.diagnosticStatus = "expired"
                        NSLog("[LLMHub][MusicGen] Background GPU task expired; saving recording")
                        self.onExpiration()
                    }
                }
                self.allowsGPU = true
                self.diagnosticStatus = "granted"
                self.diagnostic?.event("scheduler-granted id=\(id)")
                NSLog("[LLMHub][MusicGen] Background GPU granted: %@", id)
                self.pending?.resume(returning: true)
                self.pending = nil
            }
        }
        diagnostic?.event("scheduler-registered result=\(registered)")
        guard registered else {
            diagnosticStatus = "registration-failed"
            NSLog("[LLMHub][MusicGen] Background GPU registration failed: %@", id)
            return false
        }
        return await withCheckedContinuation { continuation in
            pending = continuation
            // Submit once. On iOS 27, the async API reports failures that the
            // deprecated synchronous submit API may not deliver to the caller.
            Task { @MainActor [weak self] in
                guard let self else { return }
                do {
                    self.diagnostic?.event("scheduler-submit id=\(id) requiredResources=gpu")
                    try await Self.submitOnce(identifier: id, title: title, subtitle: subtitle)
                    self.diagnostic?.event("scheduler-submit-accepted id=\(id)")
                    if self.ended { BGTaskScheduler.shared.cancel(taskRequestWithIdentifier: id) }
                } catch {
                    self.diagnosticStatus = "submission-error \(MusicDebug.error(error))"
                    self.diagnostic?.event("scheduler-submit-error \(MusicDebug.error(error))")
                    NSLog("[LLMHub][MusicGen] Background GPU request unavailable: %@", error.localizedDescription)
                    if self.pending != nil { self.finish(success: false) }
                }
            }
            Task { @MainActor [weak self] in
                try? await Task.sleep(for: .seconds(10))
                guard let self, self.pending != nil else { return }
                self.diagnosticStatus = "launch-timeout"
                self.diagnostic?.event("scheduler-launch-timeout id=\(id)")
                self.finish(success: false)
            }
        }
    }

    // A nonisolated async method keeps scheduler IPC off MainActor.
    @available(iOS 26.0, *)
    private nonisolated static func submitOnce(identifier: String, title: String, subtitle: String) async throws {
        let request = BGContinuedProcessingTaskRequest(identifier: identifier, title: title, subtitle: subtitle)
        // Live playback cannot wait in a system queue for a future launch.
        request.strategy = .fail
        request.requiredResources = .gpu
        if #available(iOS 27.0, *) {
            try await BGTaskScheduler.shared.submitTaskRequest(request)
        } else {
            try BGTaskScheduler.shared.submit(request)
        }
    }

    func update(seconds: Double, duration: Double?, title: String, subtitle: String) {
        guard #available(iOS 26.0, *), let task = task as? BGContinuedProcessingTask else { return }
        let completed = Int64(seconds * 1000)
        // Unlimited generation has no known endpoint. Report processed audio
        // with an expanding remaining work estimate instead of stalled progress.
        task.progress.totalUnitCount = duration.map { max(1, Int64($0 * 1000)) }
            ?? max(60_000, completed + 60_000)
        task.progress.completedUnitCount = min(completed, task.progress.totalUnitCount - 1)
        task.updateTitle(title, subtitle: subtitle)
    }

    func finish(success: Bool) {
        guard !ended else { return }
        diagnostic?.event("scheduler-finish success=\(success) expired=\(expired) granted=\(allowsGPU) pending=\(pending != nil)")
        ended = true
        allowsGPU = false
        pending?.resume(returning: false)
        pending = nil
        if let task {
            if #available(iOS 26.0, *), success && !expired,
               let continued = task as? BGContinuedProcessingTask {
                continued.progress.completedUnitCount = continued.progress.totalUnitCount
            }
            task.setTaskCompleted(success: success && !expired)
        } else if let identifier {
            BGTaskScheduler.shared.cancel(taskRequestWithIdentifier: identifier)
        }
        task = nil
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
        // Recording continues while locked, after the device's first unlock.
        try FileManager.default.setAttributes([.protectionKey: FileProtectionType.completeUntilFirstUserAuthentication],
                                              ofItemAtPath: url.path)
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
    private let stateLock = NSLock()
    private var paused = false
    private var enqueuedFrames: Int64 = 0
    private let diagnostic: MusicDiagnosticProbe?
    private var configurationObserver: NSObjectProtocol?

    var debugSnapshot: String {
        stateLock.lock()
        let paused = self.paused
        let produced = enqueuedFrames
        stateLock.unlock()
        return "engineRunning=\(engine.isRunning) audioPaused=\(paused) stopped=\(buffer.isStopped) enqueuedFrames=\(produced) bufferedFrames=\(buffer.bufferedFrames) underrunFrames=\(buffer.underrunFrames)"
    }

    var playedSeconds: Double {
        stateLock.lock()
        let produced = enqueuedFrames
        stateLock.unlock()
        return Double(max(0, produced - Int64(buffer.bufferedFrames))) / 48_000
    }

    private var isPaused: Bool {
        stateLock.lock()
        defer { stateLock.unlock() }
        return paused
    }

    init(diagnostic: MusicDiagnosticProbe? = nil) throws {
        self.diagnostic = diagnostic
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
        if MusicDebug.enabled {
            configurationObserver = NotificationCenter.default.addObserver(
                forName: .AVAudioEngineConfigurationChange, object: engine, queue: nil
            ) { [weak self] _ in
                guard let self else { return }
                self.diagnostic?.event("audio-engine-configuration-change \(self.debugSnapshot)")
            }
        }
    }

    deinit {
        if let configurationObserver { NotificationCenter.default.removeObserver(configurationObserver) }
    }

    func enqueue(_ pcm: Data) throws {
        var offset = 0
        while offset < pcm.count / 4 && !buffer.isStopped {
            try Task.checkCancellation()
            if isPaused { Thread.sleep(forTimeInterval: 0.01); continue }
            guard engine.isRunning else {
                diagnostic?.event("audio-engine-not-running \(debugSnapshot)")
                throw CancellationError()
            }
            let written = buffer.writePCM(pcm, frameOffset: offset)
            offset += written
            stateLock.lock()
            enqueuedFrames += Int64(written)
            stateLock.unlock()
            // Only the producer waits when ahead. Rendering never waits on it.
            if written == 0 { Thread.sleep(forTimeInterval: 0.005) }
        }
    }

    func drain() throws {
        buffer.finish() // Also releases short clips below the startup threshold.
        while buffer.bufferedFrames > 0 && !buffer.isStopped {
            try Task.checkCancellation()
            if isPaused { Thread.sleep(forTimeInterval: 0.01); continue }
            guard engine.isRunning else {
                diagnostic?.event("audio-engine-not-running \(debugSnapshot)")
                throw CancellationError()
            }
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

    func pause() {
        stateLock.lock()
        guard !paused else { stateLock.unlock(); return }
        paused = true
        stateLock.unlock()
        diagnostic?.event("audio-pause-request \(debugSnapshot)")
        controlQueue.async { [self] in
            engine.pause()
            diagnostic?.event("audio-paused \(debugSnapshot)")
        }
    }

    func resume(onError: @escaping @Sendable () -> Void) {
        controlQueue.async { [self] in
            guard !buffer.isStopped, isPaused else { return }
            diagnostic?.event("audio-resume-request \(debugSnapshot)")
            do {
                try AVAudioSession.sharedInstance().setActive(true)
                try engine.start()
                stateLock.lock()
                paused = false
                stateLock.unlock()
                diagnostic?.event("audio-resumed \(debugSnapshot)")
            } catch {
                diagnostic?.event("audio-resume-error \(MusicDebug.error(error))")
                onError()
            }
        }
    }

    func stop() {
        diagnostic?.event("audio-stop \(debugSnapshot)")
        buffer.stop()
        controlQueue.async { [self] in
            engine.stop()
        }
    }

    func logTiming(generatedSeconds: Double, elapsedSeconds: Double) {
        guard MusicDebug.enabled else { return }
        NSLog("[LLMHub][MusicGen] generated=%.2fs elapsed=%.2fs buffered=%.2fs underrun=%.3fs",
              generatedSeconds, elapsedSeconds, Double(buffer.bufferedFrames) / 48_000,
              Double(buffer.underrunFrames) / 48_000)
    }
}

/// GPU eligibility governs inference, not the transport controls. Inactive
/// scenes (including lock-screen transitions) are not yet background execution.
struct MusicPlaybackPolicy {
    let playbackPaused: Bool
    let inferenceSuspended: Bool

    init(userPaused: Bool, interrupted: Bool, background: Bool, backgroundGPU: Bool) {
        playbackPaused = userPaused || interrupted
        inferenceSuspended = playbackPaused || (background && !backgroundGPU)
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

    public var supportsLockedScreenGeneration: Bool { MusicBackgroundSupport.isSupported }

    private var loadedSession: MagentaRealtimeEngine.Session?
    private var loadedResourceDirectory: URL?
    private var livePlayback: MusicLivePlayback?
    private var generationTask: Task<(URL, Double), Error>?
    private var generationStop: MusicGenerationStop?
    private var loadTask: Task<MagentaRealtimeEngine.Session, Error>?
    private var backgroundSession: MusicBackgroundSession?
    private var backgroundRequestID: UUID?
    private var diagnostic: MusicDiagnosticProbe?
    private var diagnosticObservers: [NSObjectProtocol] = []
    private var remoteTargets: [(MPRemoteCommand, Any)] = []
    private var isLiveGeneration = false
    private var userPaused = false
    private var interruptionPaused = false
    private var applicationActive = true
    private var applicationBackground = false
    private var livePrompt = ""
    private var liveDuration: Double?
    private var interruptionObserver: NSObjectProtocol?

    private struct ModelArtifacts: Sendable {
        let functionURL: URL
        let stateURL: URL
        let resourceDirectory: URL
    }

    private init() {
        applicationActive = UIApplication.shared.applicationState == .active
        applicationBackground = UIApplication.shared.applicationState == .background
        let notifications: [Notification.Name] = [
            UIApplication.willResignActiveNotification, UIApplication.didEnterBackgroundNotification,
            UIApplication.willEnterForegroundNotification, UIApplication.didBecomeActiveNotification,
            UIApplication.protectedDataWillBecomeUnavailableNotification, UIApplication.protectedDataDidBecomeAvailableNotification,
            AVAudioSession.routeChangeNotification, AVAudioSession.mediaServicesWereLostNotification,
            AVAudioSession.mediaServicesWereResetNotification
        ]
        if MusicDebug.enabled {
            diagnosticObservers = notifications.map { name in
                NotificationCenter.default.addObserver(forName: name, object: nil, queue: .main) { [weak self] notification in
                    let reason = notification.userInfo?[AVAudioSessionRouteChangeReasonKey] as? UInt
                    Task { @MainActor in
                        guard let self, self.isLiveGeneration else { return }
                        self.logDiagnosticState("notification=\(name.rawValue) routeReason=\(reason.map(String.init) ?? "none")")
                    }
                }
            }
        }
        interruptionObserver = NotificationCenter.default.addObserver(
            forName: AVAudioSession.interruptionNotification, object: nil, queue: .main
        ) { [weak self] notification in
            let type = notification.userInfo?[AVAudioSessionInterruptionTypeKey] as? UInt
            let options = notification.userInfo?[AVAudioSessionInterruptionOptionKey] as? UInt ?? 0
            Task { @MainActor in
                guard let self, self.isLiveGeneration else { return }
                self.diagnostic?.event("audio-interruption type=\(type.map(String.init) ?? "unknown") options=\(options)")
                if type == AVAudioSession.InterruptionType.began.rawValue {
                    self.interruptionPaused = true
                } else if options & AVAudioSession.InterruptionOptions.shouldResume.rawValue != 0 {
                    self.interruptionPaused = false
                } else {
                    self.interruptionPaused = false
                    self.userPaused = true
                }
                self.updateSuspension()
            }
        }
    }

    public func applicationWillResignActive() {
        applicationActive = false
        applicationBackground = UIApplication.shared.applicationState == .background
        logDiagnosticState("scene-inactive")
        updateSuspension()
    }

    public func applicationDidEnterBackground() {
        applicationActive = false
        applicationBackground = true
        logDiagnosticState("scene-background")
        updateSuspension()
    }

    public func applicationDidBecomeActive() {
        applicationActive = true
        applicationBackground = false
        logDiagnosticState("scene-active")
        updateSuspension()
    }

    private func logDiagnosticState(_ event: String) {
        guard let diagnostic else { return }
        let session = AVAudioSession.sharedInstance()
        let ports = session.currentRoute.outputs.map { $0.portType.rawValue }.joined(separator: ",")
        diagnostic.event("\(event) appState=\(UIApplication.shared.applicationState.rawValue) active=\(applicationActive) protectedData=\(UIApplication.shared.isProtectedDataAvailable) backgroundGPU=\(backgroundSession?.allowsGPU == true) backgroundAccess=\(backgroundSession?.diagnosticStatus ?? "not-requested") userPaused=\(userPaused) interrupted=\(interruptionPaused) stop=\(generationStop?.isRequested == true) audioCategory=\(session.category.rawValue) audioMode=\(session.mode.rawValue) sampleRate=\(session.sampleRate) ioBuffer=\(session.ioBufferDuration) routes=\(ports)")
        diagnostic.snapshot("state")
    }

    private func requestBackgroundAccess() async {
        guard isGenerating, applicationActive, backgroundRequestID == nil,
              backgroundSession?.allowsGPU != true, let stop = generationStop, !stop.isRequested else {
            diagnostic?.event("background-request-skipped live=\(isLiveGeneration) active=\(applicationActive) inFlight=\(backgroundRequestID != nil) granted=\(backgroundSession?.allowsGPU == true) stopped=\(generationStop?.isRequested == true)")
            return
        }
        let requestID = UUID()
        backgroundRequestID = requestID
        defer { if backgroundRequestID == requestID { backgroundRequestID = nil } }
        backgroundSession?.finish(success: false)
        diagnostic?.event("background-request-begin request=\(requestID)")
        let background = MusicBackgroundSession(diagnostic: diagnostic) { [weak self] in
            guard let self, self.generationStop === stop else { return }
            self.stopLiveGeneration(reason: "background-expired")
        }
        backgroundSession = background
        let granted = await background.begin(title: AppSettings.shared.localized("feature_music_generator"),
                                  subtitle: AppSettings.shared.localized("generating_music"))
        diagnostic?.event("background-request-result granted=\(granted)")
        guard generationStop === stop, !stop.isRequested else { background.finish(success: false); return }
        updateSuspension()
    }

    private var playbackPolicy: MusicPlaybackPolicy {
        MusicPlaybackPolicy(userPaused: userPaused, interrupted: interruptionPaused,
            background: applicationBackground || UIApplication.shared.applicationState == .background,
            backgroundGPU: backgroundSession?.allowsGPU == true)
    }

    private func updateSuspension() {
        guard generationStop != nil else { return }
        let policy = playbackPolicy
        diagnostic?.event("transport-decision audioPaused=\(policy.playbackPaused) inferenceSuspended=\(policy.inferenceSuspended) userPaused=\(userPaused) interrupted=\(interruptionPaused) background=\(applicationBackground) backgroundGPU=\(backgroundSession?.allowsGPU == true)")
        generationStop?.setSuspended(policy.inferenceSuspended)
        guard isLiveGeneration else { return }
        if policy.playbackPaused {
            livePlayback?.pause()
        } else {
            let stop = generationStop
            livePlayback?.resume { [weak self] in
                Task { @MainActor in
                    guard let self, self.generationStop === stop else { return }
                    self.stopLiveGeneration(reason: "audio-resume-error")
                }
            }
        }
        updateNowPlaying()
    }

    private func configureRemoteControls(prompt: String, duration: Double?) {
        livePrompt = prompt
        liveDuration = duration
        let commands = MPRemoteCommandCenter.shared()
        let actions: [(String, MPRemoteCommand, @MainActor () -> Void)] = [
            ("stop", commands.stopCommand, { [weak self] in self?.stopLiveGeneration(reason: "remote-stop") }),
            ("pause", commands.pauseCommand, { [weak self] in self?.userPaused = true; self?.updateSuspension() }),
            ("play", commands.playCommand, { [weak self] in self?.userPaused = false; self?.updateSuspension() }),
            ("toggle-play-pause", commands.togglePlayPauseCommand, { [weak self] in
                guard let self else { return }
                self.userPaused.toggle()
                self.updateSuspension()
            })
        ]
        let stop = generationStop
        for (name, command, action) in actions {
            command.isEnabled = true
            let target = command.addTarget { [weak self] _ in
                Task { @MainActor in
                    guard let self, self.generationStop === stop else { return }
                    self.diagnostic?.event("remote-command name=\(name)")
                    action()
                }
                return .success
            }
            remoteTargets.append((command, target))
        }
        updateNowPlaying()
    }

    private func updateNowPlaying() {
        guard isLiveGeneration else { return }
        let paused = playbackPolicy.playbackPaused
        MPNowPlayingInfoCenter.default().nowPlayingInfo = [
            MPMediaItemPropertyTitle: livePrompt,
            MPMediaItemPropertyArtist: AppSettings.shared.localized("feature_music_generator"),
            MPNowPlayingInfoPropertyIsLiveStream: true,
            MPNowPlayingInfoPropertyElapsedPlaybackTime: livePlayback?.playedSeconds ?? 0,
            MPNowPlayingInfoPropertyPlaybackRate: paused ? 0.0 : 1.0
        ]
    }

    private func clearRemoteControls() {
        for (command, target) in remoteTargets {
            command.removeTarget(target)
            command.isEnabled = false
        }
        if !remoteTargets.isEmpty { MPNowPlayingInfoCenter.default().nowPlayingInfo = nil }
        remoteTargets.removeAll()
    }

    private func updateBackgroundProgress() {
        guard isGenerating else { return }
        let elapsed = Int(generatedDurationSeconds)
        backgroundSession?.update(seconds: generatedDurationSeconds, duration: liveDuration,
            title: AppSettings.shared.localized("feature_music_generator"),
            subtitle: "\(AppSettings.shared.localized("generating_music")) · \(String(format: "%02d:%02d", elapsed / 60, elapsed % 60))")
        updateNowPlaying()
    }

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

    public func unloadModel(reason: String = "unload-model") {
        diagnostic?.event("unload reason=\(reason)")
        loadTask?.cancel()
        generationStop?.request(reason: reason)
        if generationTask == nil { backgroundSession?.finish(success: false) }
        generationTask?.cancel()
        livePlayback?.stop()
        clearRemoteControls()
        loadedSession = nil
        loadedResourceDirectory = nil
        loadedModelName = nil
        isLoaded = false
        progress = 0
        NSLog("[LLMHub][MusicGen] Unloaded model")
    }

    public func stopLiveGeneration(reason: String = "user-stop") {
        diagnostic?.event("stop-called reason=\(reason)")
        generationStop?.request(reason: reason)
        if generationTask == nil { backgroundSession?.finish(success: false) }
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
        let probe = MusicDebug.enabled ? MusicDiagnosticProbe() : nil
        diagnostic = probe
        var outcome = "cancelled"
        probe?.event("start iOS=\(UIDevice.current.systemVersion) live=\(live) unlimited=\(unlimited) duration=\(durationSeconds) model=\(modelName)")
        probe?.start()
        var generationSucceeded = false
        defer {
            probe?.finish(outcome)
            diagnostic = nil
            backgroundSession?.finish(success: generationSucceeded)
            backgroundSession = nil
            backgroundRequestID = nil
            clearRemoteControls()
            isLiveGeneration = false
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
            let stop = MusicGenerationStop(diagnostic: probe)
            generationStop = stop
            isLiveGeneration = live
            liveDuration = runsUntilStopped ? nil : durationSeconds
            userPaused = false
            interruptionPaused = false
            // Both streaming and regular generation need the same GPU grant.
            await requestBackgroundAccess()
            if live {
                configureRemoteControls(prompt: prompt, duration: runsUntilStopped ? nil : durationSeconds)
            }
            guard !stop.isRequested else { throw CancellationError() }
            updateSuspension()
            let task = Task.detached(priority: .userInitiated) { () throws -> (URL, Double) in
                try Task.checkCancellation()
                // Audio-session activation and graph setup can block. They belong
                // on this worker, before the first inference, rather than MainActor.
                probe?.setStage("audio-setup")
                let playback = live ? try MusicLivePlayback(diagnostic: probe) : nil
                if let playback { probe?.attach(playback) }
                defer { playback?.stop() }
                await MainActor.run {
                    let backend = MusicGeneratorBackend.shared
                    if backend.generationStop === stop && !stop.isRequested {
                        backend.livePlayback = playback
                        backend.updateSuspension()
                    } else {
                        playback?.stop()
                    }
                }
                try Task.checkCancellation()
                probe?.setStage("recording-setup")
                let writer = try MusicPCMWriter(url: outputURL)
                let generationStart = ProcessInfo.processInfo.systemUptime
                var completed = false
                defer {
                    writer.finish()
                    if !completed { try? FileManager.default.removeItem(at: outputURL) }
                }
                probe?.setStage("engine-start")
                _ = try MagentaRealtimeEngine.generate(
                    session: session,
                    prompt: prompt,
                    resourceDirectory: resourceDirectory,
                    durationSeconds: runsUntilStopped ? nil : durationSeconds,
                    seed: seed,
                    collectAudio: false,
                    shouldStop: { stop.isRequested },
                    waitUntilReady: {
                        let ready = stop.waitUntilRunnable()
                        if ready { probe?.setStage("inference-or-prompt") }
                        return ready
                    },
                    onAudioFrame: { pcm in
                        probe?.setStage("recording")
                        try writer.append(pcm)
                        probe?.setStage("enqueue")
                        try playback?.enqueue(pcm)
                        probe?.frame(seconds: Double(writer.sampleCount) / 48_000)
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
                                    backend.updateBackgroundProgress()
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
                probe?.event("generation-loop-return stopped=\(stop.isRequested) cancelled=\(Task.isCancelled) samples=\(writer.sampleCount)")
                probe?.setStage("draining")
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
            generationSucceeded = true
            outcome = "saved"
            return result
        } catch is CancellationError {
            probe?.event("generation-cancelled taskCancelled=\(Task.isCancelled) stopRequested=\(generationStop?.isRequested == true)")
            progress = 0
            return nil
        } catch {
            outcome = "error"
            probe?.event("generation-error \(MusicDebug.error(error))")
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
