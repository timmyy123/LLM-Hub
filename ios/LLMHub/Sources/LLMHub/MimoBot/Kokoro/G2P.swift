import Foundation

/// Pluggable grapheme-to-phoneme converter. Implementations turn an utterance
/// of text into a space-joined IPA phoneme string using only symbols present
/// in `KokoroVocab.symbols`.
///
/// v0 ships two implementations:
///   - `DictionaryG2P` — bundled ~150-word English dictionary. Always available.
///   - `EspeakG2P`     — espeak-ng via C bridge. Multilingual, ~95%+ coverage.
///                       Requires libespeak-ng.xcframework (see
///                       scripts/build_espeak_ios.sh and docs/espeak-ng-setup.md).
///
/// Pipeline picks the best one available at startup via `G2PFactory.best()`.
protocol G2P {
    /// Human-readable name shown in the test screen ("dictionary", "espeak-ng").
    var displayName: String { get }

    /// Convert text → space-joined IPA. May return "" for empty / unsupported input.
    func phonemize(_ text: String, language: String) -> String
}

extension G2P {
    func phonemize(_ text: String) -> String { phonemize(text, language: "en-us") }
}

enum G2PFactory {
    /// Best available G2P — espeak-ng if its native lib is present, dictionary otherwise.
    static func best() -> G2P {
        EspeakG2P.tryLoad() ?? DictionaryG2P()
    }
}
