import Foundation

/// Abstraction over a speech-to-text engine.
///
/// v0 ships `SFSpeechRecognizerSTT` which owns its own mic via the Speech
/// framework. `WhisperSTT` (not yet implemented) will consume PCM from
/// `MicSource` instead.
protocol SpeechToText: AnyObject {
    func recognizeTurn(languageHint: String) async -> String
    func cancel()
    func close()
}
