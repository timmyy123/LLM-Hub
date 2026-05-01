import Foundation

/// espeak-ng-backed `G2P` for iOS.
///
/// Stub for now. iOS doesn't permit loading arbitrary dylibs at runtime, and
/// adding espeak-ng as a SwiftPM `binaryTarget` would force every build to
/// have the XCFramework present — too steep a default. The path forward is:
///
///   1. Run `scripts/build_espeak_ios.sh` once to cross-compile espeak-ng for
///      iOS arm64 (device + simulator) and produce
///      `vendor/EspeakNG.xcframework`.
///   2. Add a thin secondary SPM target (or link directly into the host app)
///      that depends on the XCFramework + a clang module map.
///   3. Replace this stub with the real implementation that imports the C
///      headers and calls `espeak_Initialize` / `espeak_TextToPhonemes`.
///
/// Until step 3 lands, `tryLoad()` returns nil and the pipeline falls back to
/// `DictionaryG2P` — same shape as the Android side.
struct EspeakG2P: G2P {
    let displayName: String = "espeak-ng"

    func phonemize(_ text: String, language: String) -> String {
        // Real impl will call espeak's C function here.
        return ""
    }

    /// Returns nil until espeak-ng is wired in via the XCFramework path
    /// described above. Callers that want to know whether espeak is available
    /// should treat a nil return as "fall back to dictionary".
    static func tryLoad() -> EspeakG2P? {
        // TODO(espeak-ios): import EspeakNG, call espeak_Initialize, return Self()
        return nil
    }
}
