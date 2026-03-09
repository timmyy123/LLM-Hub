import SwiftUI

struct FeatureCard {
    let titleKey: String
    let descriptionKey: String
    let iconSystemName: String
    let gradient: [Color]
    let route: String
}

struct HomeScreen: View {
    @EnvironmentObject var settings: AppSettings
    var onNavigateToChat: () -> Void
    var onNavigateToModels: () -> Void
    var onNavigateToSettings: () -> Void

    var features: [FeatureCard] {
        [
            FeatureCard(titleKey: "feature_ai_chat", descriptionKey: "feature_ai_chat_desc", iconSystemName: "bubble.left.and.bubble.right.fill", gradient: [Color(hex: "667eea"), Color(hex: "764ba2")], route: "chat"),
            FeatureCard(titleKey: "feature_writing_aid", descriptionKey: "feature_writing_aid_desc", iconSystemName: "pencil.line", gradient: [Color(hex: "f093fb"), Color(hex: "f5576c")], route: "writing_aid"),
            FeatureCard(titleKey: "feature_translator", descriptionKey: "feature_translator_desc", iconSystemName: "network", gradient: [Color(hex: "4facfe"), Color(hex: "00f2fe")], route: "translator"),
            FeatureCard(titleKey: "feature_transcriber", descriptionKey: "feature_transcriber_desc", iconSystemName: "mic.fill", gradient: [Color(hex: "43e97b"), Color(hex: "38f9d7")], route: "transcriber"),
            FeatureCard(titleKey: "feature_scam_detector", descriptionKey: "feature_scam_detector_desc", iconSystemName: "shield.fill", gradient: [Color(hex: "fa709a"), Color(hex: "fee140")], route: "scam_detector"),
            FeatureCard(titleKey: "feature_image_generator", descriptionKey: "feature_image_generator_desc", iconSystemName: "paintpalette.fill", gradient: [Color(hex: "6a11cb"), Color(hex: "2575fc")], route: "image_generator"),
            FeatureCard(titleKey: "feature_vibe_coder", descriptionKey: "feature_vibe_coder_desc", iconSystemName: "chevron.left.slash.chevron.right", gradient: [Color(hex: "f794a4"), Color(hex: "fdd6bd")], route: "vibe_coder"),
            FeatureCard(titleKey: "feature_creator_generation", descriptionKey: "feature_creator_generation_desc", iconSystemName: "sparkles", gradient: [Color(hex: "8EC5FC"), Color(hex: "E0C3FC")], route: "creator_generation")
        ]
    }

    let columns = [
        GridItem(.flexible(), spacing: 14),
        GridItem(.flexible(), spacing: 14)
    ]

    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                // Subtitle removed

                LazyVGrid(columns: columns, spacing: 14) {
                    ForEach(features, id: \.route) { feature in
                        Button {
                            if feature.route == "chat" {
                                onNavigateToChat()
                            }
                            // Other routes: coming soon
                        } label: {
                            FeatureCardView(feature: feature)
                        }
                        .buttonStyle(.plain)
                    }
                }
                .padding(.horizontal, 16)
                .padding(.bottom, 20)
            }
        }
        .navigationTitle("")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .navigationBarLeading) {
                Text(settings.localized("app_name"))
                    .font(.title2.bold())
                    .padding(.leading, 8)
            }
            ToolbarItem(placement: .navigationBarTrailing) {
                HStack(spacing: 14) {
                    Button {
                        onNavigateToModels()
                    } label: {
                        Image(systemName: "arrow.down.circle")
                            .font(.system(size: 18))
                    }
                    Button {
                        onNavigateToSettings()
                    } label: {
                        Image(systemName: "gearshape")
                            .font(.system(size: 18))
                    }
                }
            }
        }
    }
}

struct FeatureCardView: View {
    @EnvironmentObject var settings: AppSettings
    let feature: FeatureCard

    var body: some View {
        VStack(spacing: 12) {
            ZStack {
                Circle()
                    .fill(Color.white.opacity(0.25))
                    .frame(width: 56, height: 56)

                Image(systemName: feature.iconSystemName)
                    .font(.system(size: 26))
                    .foregroundColor(.white)
            }

            VStack(spacing: 4) {
                Text(settings.localized(feature.titleKey))
                    .font(.headline)
                    .fontWeight(.bold)
                    .foregroundColor(.white)
                    .multilineTextAlignment(.center)

                Text(settings.localized(feature.descriptionKey))
                    .font(.caption)
                    .foregroundColor(.white.opacity(0.85))
                    .multilineTextAlignment(.center)
                    .lineLimit(2)
            }
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .center)
        .frame(height: 155)
        .background(
            LinearGradient(
                gradient: Gradient(colors: feature.gradient),
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
        )
        .clipShape(RoundedRectangle(cornerRadius: 20))
        .shadow(color: feature.gradient.first?.opacity(0.4) ?? .clear, radius: 8, x: 0, y: 4)
    }
}

extension Color {
    init(hex: String) {
        let hex = hex.trimmingCharacters(in: CharacterSet.alphanumerics.inverted)
        var int: UInt64 = 0
        Scanner(string: hex).scanHexInt64(&int)
        let a, r, g, b: UInt64
        switch hex.count {
        case 3: (a, r, g, b) = (255, (int >> 8) * 17, (int >> 4 & 0xF) * 17, (int & 0xF) * 17)
        case 6: (a, r, g, b) = (255, int >> 16, int >> 8 & 0xFF, int & 0xFF)
        case 8: (a, r, g, b) = (int >> 24, int >> 16 & 0xFF, int >> 8 & 0xFF, int & 0xFF)
        default: (a, r, g, b) = (1, 1, 1, 0)
        }
        self.init(.sRGB, red: Double(r) / 255, green: Double(g) / 255, blue: Double(b) / 255, opacity: Double(a) / 255)
    }
}
