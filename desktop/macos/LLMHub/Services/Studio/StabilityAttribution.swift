import AppKit

/// What Stability AI's Community License asks of an app that offers Stable Audio 3: "Powered by Stability AI" shown
/// wherever the models are offered (Studio › Music clips, Settings › Studio backends, the About panel's credits), and
/// the license with a NOTICE file shipped in the app (Contents/Resources/Licenses).
enum StabilityAttribution {
    /// Required wording, kept in English in every language.
    static let poweredBy = "Powered by Stability AI"

    static var licensesFolder: URL? { Bundle.main.url(forResource: "Licenses", withExtension: nil) }
    static var license: URL? { licensesFolder?.appendingPathComponent("Stability-AI-Community-License.md") }
    static var notice: URL? { licensesFolder?.appendingPathComponent("NOTICE.txt") }

    static func open(_ url: URL?) {
        guard let url else { return }
        NSWorkspace.shared.open(url)
    }
}
