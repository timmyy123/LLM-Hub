import SwiftUI
import UIKit

enum ApolloPalette {
    static let accent = Color(hex: "8AAE9F")
    static let accentStrong = Color(hex: "A7D4C3")
    static let accentMuted = Color(hex: "5C7682")
    static let accentSoft = Color(hex: "6F8E85")
    static let warning = Color(hex: "E0A96D")
    static let destructive = Color(hex: "E96A63")
}

struct ApolloLiquidBackground: View {
    var body: some View {
        ZStack {
            LinearGradient(
                colors: [Color.black, Color(hex: "0b0f19"), Color(hex: "131a2a")],
                startPoint: .top,
                endPoint: .bottom
            )
            .ignoresSafeArea()

            Circle()
                .fill(Color.white.opacity(0.18))
                .frame(width: 260, height: 260)
                .blur(radius: 80)
                .offset(x: 130, y: -260)

            Circle()
                .fill(ApolloPalette.accent.opacity(0.18))
                .frame(width: 220, height: 220)
                .blur(radius: 90)
                .offset(x: -140, y: -220)

            Circle()
                .fill(ApolloPalette.accentMuted.opacity(0.16))
                .frame(width: 300, height: 300)
                .blur(radius: 110)
                .offset(x: 160, y: 260)
        }
    }
}

struct ApolloIconButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .foregroundColor(.white.opacity(configuration.isPressed ? 0.75 : 0.95))
            .scaleEffect(configuration.isPressed ? 0.96 : 1)
    }
}

private struct ApolloScreenBackgroundModifier: ViewModifier {
    func body(content: Content) -> some View {
        ZStack {
            ApolloLiquidBackground()
            content
        }
    }
}

extension View {
    func apolloScreenBackground() -> some View {
        modifier(ApolloScreenBackgroundModifier())
    }

    func enableSwipeBack() -> some View {
        background(ApolloSwipeBackEnabler())
    }

    /// Applies `transform` only when `value` is non-nil.
    /// Used to conditionally set `.environment(\.layoutDirection, ...)` so we
    /// never override the system layout direction when the user has chosen
    /// "System Default" — preventing SwiftUI from mirroring text glyphs.
    @ViewBuilder
    func ifLet<T>(_ value: T?, transform: (Self, T) -> some View) -> some View {
        if let value {
            transform(self, value)
        } else {
            self
        }
    }
}

private struct ApolloSwipeBackEnabler: UIViewControllerRepresentable {
    func makeUIViewController(context: Context) -> UIViewController {
        let controller = UIViewController()
        controller.view.backgroundColor = .clear
        return controller
    }

    func updateUIViewController(_ uiViewController: UIViewController, context: Context) {
        DispatchQueue.main.async {
            guard let navigationController = uiViewController.navigationController else { return }
            navigationController.interactivePopGestureRecognizer?.isEnabled = true
            navigationController.interactivePopGestureRecognizer?.delegate = nil
        }
    }
}
