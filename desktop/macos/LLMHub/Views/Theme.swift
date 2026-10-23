import AppKit
import SwiftUI

/// Design tokens from the LLM Hub Desktop canvas, in light and dark.
enum Theme {
    static let background = dynamic(light: 0xFAFAF8, dark: 0x1B1B1A)
    static let surface = dynamic(light: 0xFFFFFF, dark: 0x252523)
    static let surfaceAlt = dynamic(light: 0xF5F4F0, dark: 0x1F1F1D)
    static let divider = dynamic(light: 0xE2E0D9, dark: 0x2C2C29)
    static let controlBorder = dynamic(light: 0xD6D4CC, dark: 0x3A3A36)
    static let textMuted = dynamic(light: 0x5F5E59, dark: 0xA5A49E)
    static let userBubble = dynamic(light: 0xECEBE6, dark: 0x2B2B29)
    static let tag = dynamic(light: 0xE3ECFC, dark: 0x262C38)
    static let tagText = dynamic(light: 0x1D4FAE, dark: 0xB8CDF5)
    static let success = dynamic(light: 0x1E8A4C, dark: 0x5FD38D)
    static let warning = dynamic(light: 0x9A6400, dark: 0xE8B04A)
    static let danger = dynamic(light: 0xB3261E, dark: 0xFF9A9A)

    private static func dynamic(light: UInt32, dark: UInt32) -> Color {
        Color(nsColor: NSColor(name: nil) { appearance in
            let hex = appearance.bestMatch(from: [.darkAqua, .aqua]) == .darkAqua ? dark : light
            return NSColor(srgbRed: CGFloat((hex >> 16) & 0xFF) / 255, green: CGFloat((hex >> 8) & 0xFF) / 255,
                           blue: CGFloat(hex & 0xFF) / 255, alpha: 1)
        })
    }
}

struct Tag: View {
    let text: String
    var body: some View {
        Text(text)
            .font(.system(size: 11))
            .foregroundStyle(Theme.tagText)
            .padding(.horizontal, 7)
            .padding(.vertical, 1)
            .background(Theme.tag, in: RoundedRectangle(cornerRadius: 6))
    }
}
