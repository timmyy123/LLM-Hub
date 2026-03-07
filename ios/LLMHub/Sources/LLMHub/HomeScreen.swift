import SwiftUI

struct FeatureCard {
    let title: String
    let description: String
    let iconSystemName: String
    let gradient: [Color]
    let route: String
}

struct HomeScreen: View {
    let features = [
        FeatureCard(title: "AI Chat", description: "Chat with AI models directly on your device", iconSystemName: "bubble.left.and.bubble.right.fill", gradient: [Color(hex: "667eea"), Color(hex: "764ba2")], route: "chat"),
        FeatureCard(title: "Writing Aid", description: "Improve or rewrite your text anywhere", iconSystemName: "pencil.line", gradient: [Color(hex: "f093fb"), Color(hex: "f5576c")], route: "writing_aid"),
        FeatureCard(title: "Translator", description: "Offline translation across languages", iconSystemName: "network", gradient: [Color(hex: "4facfe"), Color(hex: "00f2fe")], route: "translator"),
        FeatureCard(title: "Transcriber", description: "Accurate voice-to-text conversion", iconSystemName: "mic.fill", gradient: [Color(hex: "43e97b"), Color(hex: "38f9d7")], route: "transcriber"),
        FeatureCard(title: "Scam Detector", description: "Analyze suspicious messages and links", iconSystemName: "shield.fill", gradient: [Color(hex: "fa709a"), Color(hex: "fee140")], route: "scam_detector"),
        FeatureCard(title: "Image Generator", description: "Create images from text descriptions", iconSystemName: "paintpalette.fill", gradient: [Color(hex: "6a11cb"), Color(hex: "2575fc")], route: "image_generator"),
        FeatureCard(title: "Vibe Coder", description: "Describe what you want it to code", iconSystemName: "chevron.left.slash.chevron.right", gradient: [Color(hex: "f794a4"), Color(hex: "fdd6bd")], route: "vibe_coder"),
        FeatureCard(title: "Creator Generation", description: "Generate creator-focused content", iconSystemName: "sparkles", gradient: [Color(hex: "8EC5FC"), Color(hex: "E0C3FC")], route: "creator_generation")
    ]
    
    let columns = [
        GridItem(.flexible(), spacing: 16),
        GridItem(.flexible(), spacing: 16)
    ]
    
    var body: some View {
        ScrollView {
            LazyVGrid(columns: columns, spacing: 16) {
                ForEach(features, id: \.title) { feature in
                    NavigationLink(destination: destination(for: feature.route)) {
                        FeatureCardView(feature: feature)
                    }
                    .buttonStyle(PlainButtonStyle())
                }
            }
            .padding(16)
        }
        .navigationTitle("LLM Hub")
        .toolbar {
            ToolbarItem(placement: .navigationBarTrailing) {
                NavigationLink(destination: ModelDownloadScreen()) {
                    Image(systemName: "arrow.down.circle")
                }
            }
            ToolbarItem(placement: .navigationBarTrailing) {
                NavigationLink(destination: SettingsScreen()) {
                    Image(systemName: "gearshape")
                }
            }
        }
    }
    
    @ViewBuilder
    private func destination(for route: String) -> some View {
        if route == "chat" {
            ChatScreen()
        } else {
            Text("Coming Soon: \(route)")
        }
    }
}

struct FeatureCardView: View {
    let feature: FeatureCard
    
    var body: some View {
        VStack(spacing: 12) {
            ZStack {
                Circle()
                    .fill(Color.white.opacity(0.3))
                    .frame(width: 56, height: 56)
                
                Image(systemName: feature.iconSystemName)
                    .font(.system(size: 28))
                    .foregroundColor(.white)
            }
            
            VStack(spacing: 4) {
                Text(feature.title)
                    .font(.headline)
                    .fontWeight(.bold)
                    .foregroundColor(.white)
                    .multilineTextAlignment(.center)
                
                Text(feature.description)
                    .font(.caption)
                    .foregroundColor(.white.opacity(0.8))
                    .multilineTextAlignment(.center)
                    .lineLimit(2)
            }
        }
        .padding()
        .frame(maxWidth: .infinity, alignment: .center)
        .frame(height: 160)
        .background(
            LinearGradient(gradient: Gradient(colors: feature.gradient.map { $0.opacity(0.9) }),
                           startPoint: .topLeading,
                           endPoint: .bottomTrailing)
        )
        .cornerRadius(24)
        .shadow(radius: 4)
    }
}

extension Color {
    init(hex: String) {
        let hex = hex.trimmingCharacters(in: CharacterSet.alphanumerics.inverted)
        var int: UInt64 = 0
        Scanner(string: hex).scanHexInt64(&int)
        let a, r, g, b: UInt64
        switch hex.count {
        case 3: // RGB (12-bit)
            (a, r, g, b) = (255, (int >> 8) * 17, (int >> 4 & 0xF) * 17, (int & 0xF) * 17)
        case 6: // RGB (24-bit)
            (a, r, g, b) = (255, int >> 16, int >> 8 & 0xFF, int & 0xFF)
        case 8: // ARGB (32-bit)
            (a, r, g, b) = (int >> 24, int >> 16 & 0xFF, int >> 8 & 0xFF, int & 0xFF)
        default:
            (a, r, g, b) = (1, 1, 1, 0)
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
