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

// MARK: - Waveform player (play/pause, real waveform, drag-to-seek, time)

@MainActor
final class WaveformPlaybackController: NSObject, ObservableObject, @preconcurrency AVAudioPlayerDelegate {
    @Published private(set) var isPlaying = false
    @Published private(set) var currentTime: TimeInterval = 0
    @Published private(set) var duration: TimeInterval = 0

    private var player: AVAudioPlayer?
    private var timer: Timer?
    private var url: URL?

    func load(url: URL) {
        guard self.url != url else { return }
        stop()
        self.url = url
        player = try? AVAudioPlayer(contentsOf: url)
        player?.delegate = self
        player?.prepareToPlay()
        duration = player?.duration ?? 0
        currentTime = 0
    }

    func togglePlayPause() {
        guard let player else { return }
        if player.isPlaying {
            player.pause()
            isPlaying = false
            stopTimer()
        } else {
            try? AVAudioSession.sharedInstance().setCategory(.playback, mode: .default)
            try? AVAudioSession.sharedInstance().setActive(true)
            player.play()
            isPlaying = true
            startTimer()
        }
    }

    func seek(toFraction fraction: Double) {
        guard let player, duration > 0 else { return }
        player.currentTime = min(max(fraction, 0), 1) * duration
        currentTime = player.currentTime
    }

    func stop() {
        player?.stop()
        player = nil
        url = nil
        isPlaying = false
        currentTime = 0
        stopTimer()
    }

    func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        isPlaying = false
        currentTime = 0
        player.currentTime = 0
        stopTimer()
    }

    private func startTimer() {
        stopTimer()
        timer = Timer.scheduledTimer(withTimeInterval: 0.05, repeats: true) { [weak self] _ in
            Task { @MainActor in
                guard let self, let player = self.player else { return }
                self.currentTime = player.currentTime
            }
        }
    }

    private func stopTimer() {
        timer?.invalidate()
        timer = nil
    }
}

/// Peak amplitude per bar, normalised to 0.12...1, read from any AVAudioFile-readable file.
func extractWaveformBars(url: URL, barCount: Int = 48) -> [Float] {
    let flat = [Float](repeating: 0.25, count: barCount)
    guard let file = try? AVAudioFile(forReading: url), file.length > 0 else { return flat }
    let totalFrames = AVAudioFrameCount(min(file.length, Int64(Int32.max)))
    guard let buffer = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: totalFrames),
          (try? file.read(into: buffer)) != nil,
          let channel = buffer.floatChannelData?[0] else { return flat }
    let frames = Int(buffer.frameLength)
    guard frames > 0 else { return flat }
    let perBar = max(1, frames / barCount)
    var bars = [Float](repeating: 0, count: barCount)
    for i in 0..<frames {
        let bar = min(i / perBar, barCount - 1)
        bars[bar] = max(bars[bar], abs(channel[i]))
    }
    let peak = bars.max() ?? 0
    guard peak > 0 else { return flat }
    return bars.map { min(1, max(0.12, $0 / peak * 0.88 + 0.12)) }
}

struct WaveformAudioPlayerView: View {
    let url: URL
    var title: String? = nil

    @StateObject private var controller = WaveformPlaybackController()
    @State private var bars: [Float] = Array(repeating: 0.25, count: 48)

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            if let title {
                Label(title, systemImage: "waveform")
                    .font(.caption)
                    .foregroundColor(.white.opacity(0.75))
                    .lineLimit(1)
            }
            HStack(spacing: 10) {
                Button { controller.togglePlayPause() } label: {
                    Image(systemName: controller.isPlaying ? "pause.fill" : "play.fill")
                        .font(.system(size: 14, weight: .bold))
                        .frame(width: 36, height: 36)
                }
                .audioToolsIconButtonStyle(cornerRadius: 12)

                GeometryReader { geo in
                    let progress = controller.duration > 0 ? controller.currentTime / controller.duration : 0
                    Canvas { context, size in
                        let count = bars.count
                        let spacing: CGFloat = 2
                        let barWidth = max(1, (size.width - spacing * CGFloat(count - 1)) / CGFloat(count))
                        let played = progress * Double(count)
                        for (i, value) in bars.enumerated() {
                            let height = max(4, CGFloat(value) * size.height)
                            let rect = CGRect(x: CGFloat(i) * (barWidth + spacing), y: (size.height - height) / 2, width: barWidth, height: height)
                            let color = Double(i) < played ? Color.white : Color.white.opacity(0.35)
                            context.fill(Path(roundedRect: rect, cornerRadius: barWidth / 2), with: .color(color))
                        }
                        if progress > 0 {
                            let x = CGFloat(progress) * size.width
                            context.fill(Path(CGRect(x: x - 1, y: 0, width: 2, height: size.height)), with: .color(.white))
                        }
                    }
                    .contentShape(Rectangle())
                    .gesture(
                        DragGesture(minimumDistance: 0).onChanged { value in
                            controller.seek(toFraction: Double(value.location.x / max(geo.size.width, 1)))
                        }
                    )
                }
                .frame(height: 36)

                Text("\(Self.format(controller.currentTime)) / \(Self.format(controller.duration))")
                    .font(.caption2.monospacedDigit())
                    .foregroundColor(.white.opacity(0.7))
            }
        }
        .padding(10)
        .background(RoundedRectangle(cornerRadius: 14).fill(Color.white.opacity(0.08)))
        .overlay(RoundedRectangle(cornerRadius: 14).stroke(Color.white.opacity(0.16), lineWidth: 1))
        .task(id: url) {
            controller.load(url: url)
            let target = url
            bars = await Task.detached(priority: .utility) { extractWaveformBars(url: target) }.value
        }
        .onDisappear { controller.stop() }
    }

    private static func format(_ t: TimeInterval) -> String {
        let s = Int(t.rounded(.down))
        return String(format: "%d:%02d", s / 60, s % 60)
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

@MainActor
public final class MusicGeneratorBackend: ObservableObject {
    public static let shared = MusicGeneratorBackend()

    @Published public var isGenerating: Bool = false
    @Published public var progress: Double = 0.0
    @Published public var generatedAudioURL: URL? = nil
    @Published public var errorMessage: String? = nil
    @Published public private(set) var isLoaded: Bool = false
    @Published public private(set) var loadedModelName: String? = nil

    private var loadedSession: MagentaRealtimeEngine.Session?
    private var loadedResourceDirectory: URL?

    private struct ModelArtifacts: Sendable {
        let functionURL: URL
        let stateURL: URL
        let resourceDirectory: URL
    }

    private init() {}

    public func loadModel(modelName: String) async -> Bool {
        if isLoaded, loadedModelName == modelName, loadedSession != nil {
            return true
        }
        errorMessage = nil
        do {
            let artifacts = try Self.resolveArtifacts(modelName: modelName)
            let session = try await Task.detached(priority: .userInitiated) {
                try MagentaRealtimeEngine.load(
                    functionURL: artifacts.functionURL,
                    stateURL: artifacts.stateURL
                )
            }.value
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
        loadedSession = nil
        loadedResourceDirectory = nil
        loadedModelName = nil
        isLoaded = false
        progress = 0
        NSLog("[LLMHub][MusicGen] Unloaded model")
    }

    public func generateMusic(
        modelName: String,
        prompt: String,
        durationSeconds: Double
    ) async -> URL? {
        guard await loadModel(modelName: modelName),
              let session = loadedSession,
              let resourceDirectory = loadedResourceDirectory else {
            return nil
        }
        isGenerating = true
        progress = 0.05
        errorMessage = nil
        generatedAudioURL = nil

        let documentsDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let outputURL = documentsDir.appendingPathComponent("generated_music_\(Int(Date().timeIntervalSince1970)).wav")

        let result = await Task.detached(priority: .userInitiated) { () -> URL? in
            do {
                let sampleRate = MagentaRealtimeEngine.sampleRate
                let numChannels: Int = 2

                NSLog("[LLMHub][MusicGen] Running loaded stateful MRT2 MLX model: \(modelName)")
                NSLog("[LLMHub][MusicGen] requestedPrompt=%@ duration=%.2fs", prompt, durationSeconds)
                let pcmData = try MagentaRealtimeEngine.generate(
                    session: session,
                    prompt: prompt,
                    resourceDirectory: resourceDirectory,
                    durationSeconds: durationSeconds,
                    progress: { fraction in
                        Task { @MainActor in
                            MusicGeneratorBackend.shared.progress = 0.05 + fraction * 0.90
                        }
                    }
                )

                let wavData = MusicGeneratorBackend.createWAVHeader(dataSize: pcmData.count, sampleRate: sampleRate, channels: numChannels) + pcmData
                try wavData.write(to: outputURL)
                return outputURL

            } catch {
                NSLog("[LLMHub][MusicGen] Generation error: \(error.localizedDescription)")
                return nil
            }
        }.value

        isGenerating = false
        if let result {
            progress = 1.0
            generatedAudioURL = result
            return result
        } else {
            errorMessage = "Failed to generate music audio"
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
        var functionURL: URL?
        var stateURL: URL?
        var resourceDirectory: URL?
        for directory in searchDirs where FileManager.default.fileExists(atPath: directory.path) {
            guard let enumerator = FileManager.default.enumerator(
                at: directory,
                includingPropertiesForKeys: nil
            ) else { continue }
            while let fileURL = enumerator.nextObject() as? URL {
                if fileURL.pathExtension == "mlxfn", functionURL == nil { functionURL = fileURL }
                if fileURL.lastPathComponent.hasSuffix("_state.safetensors"), stateURL == nil { stateURL = fileURL }
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


    nonisolated private static func createWAVHeader(dataSize: Int, sampleRate: Int, channels: Int) -> Data {
        var data = Data()
        let bitsPerSample = 16
        let byteRate = sampleRate * channels * (bitsPerSample / 8)
        let blockAlign = channels * (bitsPerSample / 8)
        let chunkSize = 36 + dataSize

        data.append(contentsOf: "RIFF".utf8)
        var chunkSize32 = UInt32(chunkSize).littleEndian
        withUnsafeBytes(of: &chunkSize32) { data.append(contentsOf: $0) }

        data.append(contentsOf: "WAVE".utf8)
        data.append(contentsOf: "fmt ".utf8)

        var subchunk1Size = UInt32(16).littleEndian
        withUnsafeBytes(of: &subchunk1Size) { data.append(contentsOf: $0) }

        var audioFormat = UInt16(1).littleEndian
        withUnsafeBytes(of: &audioFormat) { data.append(contentsOf: $0) }

        var numChannels16 = UInt16(channels).littleEndian
        withUnsafeBytes(of: &numChannels16) { data.append(contentsOf: $0) }

        var sampleRate32 = UInt32(sampleRate).littleEndian
        withUnsafeBytes(of: &sampleRate32) { data.append(contentsOf: $0) }

        var byteRate32 = UInt32(byteRate).littleEndian
        withUnsafeBytes(of: &byteRate32) { data.append(contentsOf: $0) }

        var blockAlign16 = UInt16(blockAlign).littleEndian
        withUnsafeBytes(of: &blockAlign16) { data.append(contentsOf: $0) }

        var bitsPerSample16 = UInt16(bitsPerSample).littleEndian
        withUnsafeBytes(of: &bitsPerSample16) { data.append(contentsOf: $0) }

        data.append(contentsOf: "data".utf8)
        var dataSize32 = UInt32(dataSize).littleEndian
        withUnsafeBytes(of: &dataSize32) { data.append(contentsOf: $0) }

        return data
    }
}
