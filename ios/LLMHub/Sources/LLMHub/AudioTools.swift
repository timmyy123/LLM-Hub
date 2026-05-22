import AVFoundation
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
