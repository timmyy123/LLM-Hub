import SwiftUI

struct SplashScreenView: View {
    @ObservedObject private var settings = AppSettings.shared
    
    @State private var logoScale: CGFloat = 0.2
    @State private var logoOpacity: Double = 0.0
    @State private var textOpacity: Double = 0.0
    @State private var textKerning: CGFloat = 12.0
    @State private var glowRotation: Double = 0.0
    @State private var glowScale: CGFloat = 0.8
    @State private var subtitleOpacity: Double = 0.0
    
    var onFinished: () -> Void
    
    var body: some View {
        ZStack {
            // High-end dark ambient liquid background
            ApolloLiquidBackground()
            
            // Cosmic/Aura Glow behind the logo
            ZStack {
                Circle()
                    .fill(
                        AngularGradient(
                            colors: [
                                Color(hex: "FF5E7E"), // Vibrant coral/pink
                                Color(hex: "A7D4C3"), // Brand mint green
                                Color(hex: "8AAE9F"), // Brand sage
                                Color(hex: "3F5EFB"), // Premium blue
                                Color(hex: "FF5E7E")
                            ],
                            center: .center
                        )
                    )
                    .frame(width: 180, height: 180)
                    .blur(radius: 40)
                    .scaleEffect(glowScale)
                    .rotationEffect(.degrees(glowRotation))
                    .opacity(logoOpacity * 0.45)
            }
            
            VStack(spacing: 24) {
                Spacer()
                
                // Centered App Icon with Spring Scaling
                Image("SplashLogo")
                    .resizable()
                    .scaledToFit()
                    .frame(width: 130, height: 130)
                    .clipShape(RoundedRectangle(cornerRadius: 30, style: .continuous))
                    .shadow(color: Color.black.opacity(0.4), radius: 15, x: 0, y: 10)
                    .scaleEffect(logoScale)
                    .opacity(logoOpacity)
                
                // Cinematic Title Fade-in & Kerning Animation
                VStack(spacing: 8) {
                    Text(settings.localized("app_name").uppercased())
                        .font(.system(size: 28, weight: .bold, design: .rounded))
                        .foregroundColor(.white)
                        .kerning(textKerning)
                        .opacity(textOpacity)
                        .offset(x: textKerning / 2) // Centering correction for kerning offset
                    
                    Text(settings.localized("on_device_ai_engine"))
                        .font(.system(size: 10, weight: .semibold, design: .default))
                        .foregroundColor(ApolloPalette.accentStrong.opacity(0.7))
                        .kerning(4)
                        .opacity(subtitleOpacity)
                        .offset(x: 2)
                }
                
                Spacer()
                
                // Bottom secure badge
                Text(settings.localized("secure_private"))
                    .font(.system(size: 9, weight: .bold))
                    .foregroundColor(.white.opacity(0.35))
                    .kerning(2)
                    .padding(.bottom, 24)
                    .opacity(textOpacity)
            }
        }
        .onAppear {
            // 1. Spring scale-in and fade-in logo (faster)
            withAnimation(.spring(response: 0.55, dampingFraction: 0.65, blendDuration: 0)) {
                logoScale = 1.0
                logoOpacity = 1.0
            }
            
            // 2. Slow rotation of the background glow
            withAnimation(.linear(duration: 8.0).repeatForever(autoreverses: false)) {
                glowRotation = 360.0
            }
            
            // 3. Ambient pulsing of the background glow
            withAnimation(.easeInOut(duration: 2.2).repeatForever(autoreverses: true)) {
                glowScale = 1.15
            }
            
            // 4. Fade-in and animate title spacing (faster)
            withAnimation(.easeOut(duration: 0.6).delay(0.1)) {
                textOpacity = 1.0
                textKerning = 5.0
            }
            
            // 5. Fade-in subtitle (faster)
            withAnimation(.easeOut(duration: 0.5).delay(0.3)) {
                subtitleOpacity = 1.0
            }
            
            // 6. Transition to home screen (1.2 seconds instead of 2.3 seconds)
            DispatchQueue.main.asyncAfter(deadline: .now() + 1.2) {
                withAnimation(.easeOut(duration: 0.35)) {
                    // Trigger screen transition
                    onFinished()
                }
            }
        }
    }
}
