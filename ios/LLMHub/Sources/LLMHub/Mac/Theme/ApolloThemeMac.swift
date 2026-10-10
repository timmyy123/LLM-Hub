import SwiftUI

extension Color {
    init(hex: String) {
        let hex = hex.trimmingCharacters(in: CharacterSet.alphanumerics.inverted)
        var int: UInt64 = 0
        Scanner(string: hex).scanHexInt64(&int)
        let a, r, g, b: UInt64
        switch hex.count {
        case 3:
            (a, r, g, b) = (255, (int >> 8) * 17, (int >> 4 & 0xF) * 17, (int & 0xF) * 17)
        case 6:
            (a, r, g, b) = (255, int >> 16, int >> 8 & 0xFF, int & 0xFF)
        case 8:
            (a, r, g, b) = (int >> 24, int >> 16 & 0xFF, int >> 8 & 0xFF, int & 0xFF)
        default:
            (a, r, g, b) = (255, 0, 0, 0)
        }
        self.init(
            .sRGB,
            red: Double(r) / 255,
            green: Double(g) / 255,
            blue:  Double(b) / 255,
            opacity: Double(a) / 255
        )
    }
}

public enum ApolloPalette {
    public static let accent = Color(hex: "8AAE9F")
    public static let accentStrong = Color(hex: "A7D4C3")
    public static let accentMuted = Color(hex: "5C7682")
    public static let accentSoft = Color(hex: "6F8E85")
    public static let warning = Color(hex: "E0A96D")
    public static let destructive = Color(hex: "E96A63")
    public static let bgDeep = Color(hex: "090d16")
    public static let bgSurface = Color(hex: "111827")
    public static let bgSurfaceHover = Color(hex: "1f293d")
    public static let borderGlass = Color.white.opacity(0.08)
}

public struct ApolloLiquidBackground: View {
    public init() {}

    public var body: some View {
        ZStack {
            LinearGradient(
                colors: [Color.black, Color(hex: "0b0f19"), Color(hex: "131a2a")],
                startPoint: .top,
                endPoint: .bottom
            )
            .ignoresSafeArea()

            Circle()
                .fill(ApolloPalette.accentSoft.opacity(0.06))
                .frame(width: 450, height: 450)
                .blur(radius: 120)
                .offset(x: 240, y: -200)

            Circle()
                .fill(ApolloPalette.accent.opacity(0.12))
                .frame(width: 380, height: 380)
                .blur(radius: 110)
                .offset(x: -260, y: -160)

            Circle()
                .fill(ApolloPalette.accentMuted.opacity(0.14))
                .frame(width: 480, height: 480)
                .blur(radius: 130)
                .offset(x: 220, y: 300)
        }
    }
}

public struct ApolloCardModifier: ViewModifier {
    var padding: CGFloat = 16
    var cornerRadius: CGFloat = 14

    public func body(content: Content) -> some View {
        content
            .padding(padding)
            .background(
                RoundedRectangle(cornerRadius: cornerRadius)
                    .fill(ApolloPalette.bgSurface.opacity(0.72))
            )
            .overlay(
                RoundedRectangle(cornerRadius: cornerRadius)
                    .stroke(ApolloPalette.borderGlass, lineWidth: 1)
            )
            .shadow(color: Color.black.opacity(0.25), radius: 8, x: 0, y: 4)
    }
}

public struct ApolloIconButtonStyle: ButtonStyle {
    public init() {}

    public func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .foregroundColor(.white.opacity(configuration.isPressed ? 0.75 : 0.95))
            .scaleEffect(configuration.isPressed ? 0.96 : 1)
            .animation(.easeOut(duration: 0.15), value: configuration.isPressed)
    }
}

public struct ApolloPrimaryButtonStyle: ButtonStyle {
    public var isDestructive: Bool = false
    public init(isDestructive: Bool = false) {
        self.isDestructive = isDestructive
    }

    public func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.system(size: 13, weight: .semibold))
            .padding(.horizontal, 14)
            .padding(.vertical, 8)
            .background(
                RoundedRectangle(cornerRadius: 8)
                    .fill(isDestructive ? ApolloPalette.destructive : ApolloPalette.accent)
                    .opacity(configuration.isPressed ? 0.75 : 1.0)
            )
            .foregroundColor(.black)
            .scaleEffect(configuration.isPressed ? 0.98 : 1.0)
            .animation(.easeOut(duration: 0.12), value: configuration.isPressed)
    }
}

public struct ApolloSecondaryButtonStyle: ButtonStyle {
    public init() {}

    public func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.system(size: 13, weight: .medium))
            .padding(.horizontal, 12)
            .padding(.vertical, 7)
            .background(
                RoundedRectangle(cornerRadius: 8)
                    .fill(Color.white.opacity(configuration.isPressed ? 0.12 : 0.06))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 8)
                    .stroke(ApolloPalette.borderGlass, lineWidth: 1)
            )
            .foregroundColor(.white.opacity(0.9))
            .scaleEffect(configuration.isPressed ? 0.98 : 1.0)
            .animation(.easeOut(duration: 0.12), value: configuration.isPressed)
    }
}

extension View {
    public func apolloCard(padding: CGFloat = 16, cornerRadius: CGFloat = 14) -> some View {
        modifier(ApolloCardModifier(padding: padding, cornerRadius: cornerRadius))
    }

    public func apolloScreenBackground() -> some View {
        ZStack {
            ApolloLiquidBackground()
            self
        }
    }
}
