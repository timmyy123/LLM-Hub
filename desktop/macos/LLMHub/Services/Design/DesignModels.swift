import Foundation

/// A named visual direction the model designs with: colors, type, and short guidance (our own, not brand copies).
/// Same six systems as the Windows and Linux apps (LLMHub.Core/Services/Design/DesignSystems.cs).
struct DesignSystem: Identifiable, Hashable {
    let id: String
    let name: String
    let colors: [String]
    let headingFont: String
    let bodyFont: String
    let guidance: String

    /// The DESIGN.md-style brief given to the model.
    var brief: String {
        """
        Design system: \(name)
        Palette (use these hex values; first is the page background, last is the darkest ink): \(colors.joined(separator: ", "))
        Headings: \(headingFont). Body text: \(bodyFont). Load fonts from Google Fonts with <link> tags.
        \(guidance)
        """
    }

    static let all: [DesignSystem] = [
        DesignSystem(id: "editorial-warm", name: "Editorial Warm", colors: ["#f6f1e7", "#efe6d6", "#d9c6ae", "#b4532a", "#2a211b"],
                     headingFont: "Fraunces", bodyFont: "Inter",
                     guidance: "Warm cream backgrounds, serif headings, generous whitespace, one terracotta accent used sparingly for calls to action."),
        DesignSystem(id: "clean-modern", name: "Clean Modern", colors: ["#ffffff", "#f3f5f8", "#d7dde6", "#2f6fe4", "#121826"],
                     headingFont: "Inter", bodyFont: "Inter",
                     guidance: "Crisp white surfaces, strong grid, 8 px spacing scale, rounded 12 px cards, one blue accent, high-contrast text."),
        DesignSystem(id: "midnight", name: "Midnight", colors: ["#0f1117", "#171a23", "#2a2f3d", "#8ab4ff", "#eef1f7"],
                     headingFont: "Space Grotesk", bodyFont: "Inter",
                     guidance: "Dark interface, subtle borders instead of shadows, luminous blue accent, tight technical typography."),
        DesignSystem(id: "bold-contrast", name: "Bold Contrast", colors: ["#fffdf5", "#ffe45c", "#ff5a36", "#1d1d1d", "#000000"],
                     headingFont: "Archivo Black", bodyFont: "Archivo",
                     guidance: "Poster-like: oversized headlines, thick black rules, flat saturated color blocks, no gradients."),
        DesignSystem(id: "soft-pastel", name: "Soft Pastel", colors: ["#fbf8ff", "#efe7ff", "#ffd8e4", "#7c5cff", "#2b2440"],
                     headingFont: "DM Serif Display", bodyFont: "DM Sans",
                     guidance: "Gentle pastel fields, rounded 20 px shapes, friendly illustrations described with CSS shapes, airy layouts."),
        DesignSystem(id: "minimal-mono", name: "Minimal Mono", colors: ["#fafafa", "#eeeeee", "#cccccc", "#333333", "#111111"],
                     headingFont: "IBM Plex Sans", bodyFont: "IBM Plex Sans",
                     guidance: "Black and white only, typography-led, thin hairlines, lots of space, no color accents."),
    ]

    static func find(_ id: String) -> DesignSystem { all.first { $0.id == id } ?? all[0] }
}

/// What the Design tab makes; each kind gets its own instructions.
enum DesignKind: String, Codable, CaseIterable, Identifiable {
    case prototype = "Prototype", slides = "Slides", image = "Image", motion = "Motion"

    var id: String { rawValue }
    var label: String { L.s("Design\(rawValue)") }
}

/// A turn in a design project; the assistant's carries the steps it took ("Wrote index.html", "Checked layout at phone width").
struct DesignMessage: Codable, Hashable, Identifiable {
    var id = UUID()
    let role: String
    let text: String
    var steps: [String]

    init(role: String, text: String, steps: [String] = []) {
        self.role = role
        self.text = text
        self.steps = steps
    }

    private enum CodingKeys: String, CodingKey { case role, text, steps }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        role = try c.decode(String.self, forKey: .role)
        text = try c.decode(String.self, forKey: .text)
        steps = try c.decodeIfPresent([String].self, forKey: .steps) ?? []
    }

    func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(role, forKey: .role)
        try c.encode(text, forKey: .text)
        if !steps.isEmpty { try c.encode(steps, forKey: .steps) }
    }

    /// The steps as "✓ Wrote index.html" lines, or "".
    var stepsText: String { steps.map { "✓ " + $0 }.joined(separator: "\n") }
}

/// A design project on disk: design/<id>/v1.html, v2.html, … plus project.json.
struct DesignProject: Codable, Identifiable, Hashable {
    var id = String(UUID().uuidString.replacingOccurrences(of: "-", with: "").lowercased().prefix(12))
    var name = ""
    var kind = DesignKind.prototype
    var systemId = DesignSystem.all[0].id
    var versions = 0
    var updatedAt = Date()
    var history: [DesignMessage] = []

    var folder: URL {
        let url = AppPaths.design.appendingPathComponent(id, isDirectory: true)
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    func versionFile(_ n: Int) -> URL { folder.appendingPathComponent("v\(n).html") }
    var currentFile: URL? { versions > 0 ? versionFile(versions) : nil }
    var system: DesignSystem { DesignSystem.find(systemId) }
    var displayName: String { name.isEmpty ? L.s("DesignUntitled") : name }
}
