import Foundation
import Speech
import AVFoundation

/// `SpeechToText` backed by Apple's Speech framework.
///
/// On iOS 13+ and modern Apple Silicon we request on-device recognition so
/// transcription stays local. The recognizer has its own audio buffering —
/// we feed it via AVAudioEngine's input tap and stop when it reports a final
/// result or hits a VAD tail.
///
/// Host app Info.plist must declare:
///   - NSSpeechRecognitionUsageDescription
///   - NSMicrophoneUsageDescription
final class SFSpeechRecognizerSTT: SpeechToText {

    private let recognizer: SFSpeechRecognizer?
    private let engine = AVAudioEngine()
    private var request: SFSpeechAudioBufferRecognitionRequest?
    private var task: SFSpeechRecognitionTask?

    init(locale: Locale = .init(identifier: "en-US")) {
        self.recognizer = SFSpeechRecognizer(locale: locale)
        if #available(iOS 13.0, *), self.recognizer?.supportsOnDeviceRecognition == false {
            // Still usable — just goes over the network if entitlements allow it.
        }
    }

    func recognizeTurn(languageHint: String) async -> String {
        // Ensure authorization.
        let status = await withCheckedContinuation { (cont: CheckedContinuation<SFSpeechRecognizerAuthorizationStatus, Never>) in
            SFSpeechRecognizer.requestAuthorization { cont.resume(returning: $0) }
        }
        guard status == .authorized, let recognizer = recognizer, recognizer.isAvailable else {
            return ""
        }

        let req = SFSpeechAudioBufferRecognitionRequest()
        req.shouldReportPartialResults = false
        if #available(iOS 13.0, *) {
            req.requiresOnDeviceRecognition = recognizer.supportsOnDeviceRecognition
        }
        self.request = req

        do {
            try AVAudioSession.sharedInstance().setCategory(.playAndRecord, mode: .measurement, options: [.duckOthers, .defaultToSpeaker, .allowBluetooth])
            try AVAudioSession.sharedInstance().setActive(true, options: .notifyOthersOnDeactivation)
        } catch {
            return ""
        }

        let input = engine.inputNode
        let format = input.outputFormat(forBus: 0)
        input.installTap(onBus: 0, bufferSize: 1024, format: format) { [weak req] buffer, _ in
            req?.append(buffer)
        }
        engine.prepare()
        do { try engine.start() } catch {
            input.removeTap(onBus: 0)
            return ""
        }

        return await withCheckedContinuation { [weak self] (cont: CheckedContinuation<String, Never>) in
            guard let self else { cont.resume(returning: ""); return }
            self.task = recognizer.recognitionTask(with: req) { [weak self] result, error in
                guard let self else { return }
                if let result, result.isFinal {
                    self.teardown()
                    cont.resume(returning: result.bestTranscription.formattedString)
                    return
                }
                if error != nil {
                    self.teardown()
                    cont.resume(returning: result?.bestTranscription.formattedString ?? "")
                    return
                }
            }
        }
    }

    func cancel() {
        task?.cancel()
        teardown()
    }

    func close() { teardown() }

    private func teardown() {
        if engine.isRunning {
            engine.inputNode.removeTap(onBus: 0)
            engine.stop()
        }
        request?.endAudio()
        request = nil
        task = nil
    }
}
