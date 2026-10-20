import SwiftUI

/// Launch animation. Its first frame matches the static launch screen (`UILaunchScreen`:
/// `LaunchBackground` colour + `SplashLogo` at its 180 pt size, centred on the full screen),
/// so the hand-off from the system launch screen is invisible. The app's own background
/// fades in under the logo while its glow swells once, then the logo eases forward and fades
/// while the home screen (same background) crossfades in.
struct SplashScreenView: View {
    var onFinished: () -> Void

    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var glowing = false
    @State private var leaving = false

    private static let background = Color("LaunchBackground")
    private static let logoSize: CGFloat = 180

    var body: some View {
        ZStack {
            Self.background
            ApolloLiquidBackground()
                .opacity(glowing ? 1 : 0)

            // Soft bloom in the logo's own colours, behind the logo.
            Circle()
                .fill(
                    AngularGradient(
                        colors: [
                            Color(hex: "1EE3FF"),
                            Color(hex: "2F4BFF"),
                            Color(hex: "A44BFF"),
                            Color(hex: "FF5CF0"),
                            Color(hex: "1EE3FF"),
                        ],
                        center: .center,
                        angle: .degrees(glowing && !reduceMotion ? 40 : 0)
                    )
                )
                .frame(width: Self.logoSize * 1.15, height: Self.logoSize * 1.15)
                .blur(radius: 70)
                .scaleEffect(glowing && !reduceMotion ? 1.2 : 0.9)
                .opacity(leaving ? 0 : (glowing ? 0.35 : 0))

            Image("SplashLogo")
                .resizable()
                .scaledToFit()
                .frame(width: Self.logoSize, height: Self.logoSize)
                .scaleEffect(leaving && !reduceMotion ? 1.12 : 1)
                .opacity(leaving ? 0 : 1)
        }
        .ignoresSafeArea()
        .accessibilityHidden(true)
        .onAppear {
            withAnimation(.easeInOut(duration: 0.7)) {
                glowing = true
            }
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.75) {
                // Home starts crossfading in at the same moment, so there is never an empty frame.
                withAnimation(.easeInOut(duration: 0.45)) {
                    leaving = true
                    onFinished()
                }
            }
        }
    }
}
