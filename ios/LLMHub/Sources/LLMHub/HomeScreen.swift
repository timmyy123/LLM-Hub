import Foundation
import SwiftUI

struct FeatureCard {
    let titleKey: String
    let descriptionKey: String
    let iconSystemName: String
    let gradient: [Color]
    let route: String
}

struct HomeScreen: View {
    @Environment(\.openURL) private var openURL
    @EnvironmentObject var settings: AppSettings
    var onNavigateToChat: () -> Void
    var onNavigateToModels: () -> Void
    var onNavigateToSettings: () -> Void
    var onNavigateToRoute: (String) -> Void
    @State private var githubStars: Int? = nil
    @ObservedObject private var purchases = PurchaseManager.shared
    @State private var showPremium = false

    var heroFeature: FeatureCard {
        FeatureCard(titleKey: "feature_ai_chat", descriptionKey: "feature_ai_chat_desc", iconSystemName: "bubble.left.and.bubble.right.fill", gradient: [Color(hex: "7ea3ff"), Color(hex: "5e79da")], route: "chat")
    }

    var toolsFeatures: [FeatureCard] {
        [
            FeatureCard(titleKey: "feature_writing_aid", descriptionKey: "feature_writing_aid_desc", iconSystemName: "pencil.line", gradient: [Color(hex: "91d4ff"), Color(hex: "4e86d5")], route: "writing_aid"),
            FeatureCard(titleKey: "feature_translator", descriptionKey: "feature_translator_desc", iconSystemName: "network", gradient: [Color(hex: "84f1cf"), Color(hex: "4aa897")], route: "translator"),
            FeatureCard(titleKey: "feature_transcriber", descriptionKey: "feature_transcriber_desc", iconSystemName: "mic.fill", gradient: [Color(hex: "b4b2ff"), Color(hex: "6f77cf")], route: "transcriber"),
            FeatureCard(titleKey: "feature_image_generator", descriptionKey: "feature_image_generator_desc", iconSystemName: "paintpalette.fill", gradient: [Color(hex: "9cc3ff"), Color(hex: "5b86d2")], route: "image_generator"),
            FeatureCard(titleKey: "feature_image_upscaler", descriptionKey: "feature_image_upscaler_desc", iconSystemName: "wand.and.stars", gradient: [Color(hex: "8fd3ff"), Color(hex: "3f7dd8")], route: "image_upscaler"),
            FeatureCard(titleKey: "feature_video_generator", descriptionKey: "feature_video_generator_desc", iconSystemName: "video.fill", gradient: [Color(hex: "ff99c8"), Color(hex: "fc4b93")], route: "video_generator")
        ]
    }

    var utilityFeatures: [FeatureCard] {
        [
            FeatureCard(titleKey: "feature_scam_detector", descriptionKey: "feature_scam_detector_desc", iconSystemName: "shield.fill", gradient: [Color(hex: "ffb08a"), Color(hex: "d77c59")], route: "scam_detector"),
            FeatureCard(titleKey: "feature_vibe_coder", descriptionKey: "feature_vibe_coder_desc", iconSystemName: "chevron.left.slash.chevron.right", gradient: [Color(hex: "a8bcff"), Color(hex: "5f76be")], route: "vibe_coder"),
            FeatureCard(titleKey: "feature_vibevoice", descriptionKey: "feature_vibevoice_desc", iconSystemName: "waveform.circle.fill", gradient: [Color(hex: "89d3f7"), Color(hex: "3a68cc")], route: "vibe_voice")
        ]
    }

    var body: some View {
        GeometryReader { geo in
            let horizontalPadding: CGFloat = 16
            let rawUsableWidth = geo.size.width - (horizontalPadding * 2)
            let usableWidth = max(1, rawUsableWidth)
            let topBarHeight: CGFloat = 48
            let topPadding: CGFloat = 10
            let isLandscape = geo.size.width > geo.size.height
            let spacing: CGFloat = {
                if isLandscape {
                    return min(max(usableWidth * 0.020, 12), 16)
                }
                return min(max(usableWidth * 0.014, 8), 12)
            }()

            let toolsColumnsCount = isLandscape ? 4 : 2
            let toolsColumns = Array(repeating: GridItem(.flexible(), spacing: spacing), count: toolsColumnsCount)
            let cardHeight: CGFloat = isLandscape ? 85 : 100
            let gridTopPadding: CGFloat = isLandscape ? 20 : 24
            let gridBottomPadding: CGFloat = 12

            VStack(spacing: 0) {
                HStack(spacing: 12) {
                    Text(settings.localized("app_name"))
                        .font(.title.bold())
                        .foregroundColor(.white)

                    Spacer()

                    if !purchases.isPremium {
                        Button {
                            showPremium = true
                        } label: {
                            Image(systemName: "crown.fill")
                                .font(.system(size: 18, weight: .semibold))
                                .foregroundStyle(Color(hex: "FFD700"))
                                .padding(8)
                                .background(.ultraThinMaterial, in: Circle())
                                .overlay(
                                    Circle()
                                        .stroke(
                                            LinearGradient(
                                                colors: [Color.white.opacity(0.35), Color.white.opacity(0.08)],
                                                startPoint: .top,
                                                endPoint: .bottom
                                            ),
                                            lineWidth: 1
                                        )
                                )
                                .shadow(color: .black.opacity(0.2), radius: 6, x: 0, y: 3)
                        }
                        .buttonStyle(.plain)
                    }

                    HStack(spacing: 10) {
                        if let githubStars, githubStars > 0 {
                            Button {
                                openGithubRepository()
                            } label: {
                                HStack(spacing: 4) {
                                    Image(systemName: "star.fill")
                                        .font(.caption)
                                    Text("\(githubStars)")
                                        .font(.subheadline.bold())
                                        .lineLimit(1)
                                        .fixedSize(horizontal: true, vertical: false)
                                }
                                .padding(.horizontal, 8)
                                .padding(.vertical, 5)
                            }
                            .buttonStyle(.plain)
                        }

                        Button {
                            onNavigateToModels()
                        } label: {
                            Image(systemName: "arrow.down.circle")
                                .font(.system(size: 22))
                        }
                        .buttonStyle(ApolloIconButtonStyle())

                        Button {
                            onNavigateToSettings()
                        } label: {
                            Image(systemName: "gearshape")
                                .font(.system(size: 22))
                        }
                        .buttonStyle(ApolloIconButtonStyle())
                    }
                    .padding(.horizontal, 10)
                    .padding(.vertical, 7)
                    .background(.ultraThinMaterial, in: Capsule())
                    .overlay(
                        Capsule()
                            .stroke(
                                LinearGradient(
                                    colors: [Color.white.opacity(0.35), Color.white.opacity(0.08)],
                                    startPoint: .top,
                                    endPoint: .bottom
                                ),
                                lineWidth: 1
                            )
                    )
                    .shadow(color: .black.opacity(0.32), radius: 10, x: 0, y: 6)
                    .clipShape(Capsule())
                    .fixedSize(horizontal: true, vertical: false)
                }
                .padding(.horizontal, horizontalPadding)
                .padding(.top, topPadding)
                .frame(height: topBarHeight)

                ScrollView {
                    VStack(alignment: .leading, spacing: 20) {
                        // Hero Card for Chat
                        Button {
                            onNavigateToChat()
                        } label: {
                            HomeHeroCardView(feature: heroFeature)
                        }
                        .buttonStyle(.plain)
                        
                        // Tools section
                        VStack(alignment: .leading, spacing: 10) {
                            Text(settings.localized("home_section_tools"))
                                .font(.title2.bold())
                                .foregroundColor(.white)
                                .padding(.horizontal, 4)
                            
                            LazyVGrid(columns: toolsColumns, spacing: spacing) {
                                ForEach(toolsFeatures + utilityFeatures, id: \.route) { feature in
                                    Button {
                                        onNavigateToRoute(feature.route)
                                    } label: {
                                        SmallFeatureCardView(feature: feature)
                                            .frame(height: cardHeight)
                                    }
                                    .buttonStyle(.plain)
                                }
                            }
                        }
                    }
                    .padding(.horizontal, horizontalPadding)
                    .padding(.top, gridTopPadding)
                    .padding(.bottom, gridBottomPadding)
                }
                .ignoresSafeArea(.container, edges: .bottom)
            }
            .onAppear {
                if githubStars == nil {
                    Task {
                        await loadGithubStars()
                    }
                }
            }
        }
        .apolloScreenBackground()
        .safeAreaInset(edge: .bottom, spacing: 0) {
            BannerAdContainer()
        }
        .toolbar(.hidden, for: .navigationBar)
        .sheet(isPresented: $showPremium) {
            PremiumScreen()
                .environmentObject(settings)
        }
    }

    private func loadGithubStars() async {
        guard let url = URL(string: "https://api.github.com/repos/timmyy123/LLM-Hub") else { return }
        var request = URLRequest(url: url)
        request.setValue("application/vnd.github+json", forHTTPHeaderField: "Accept")

        do {
            let (data, response) = try await URLSession.shared.data(for: request)
            guard let http = response as? HTTPURLResponse, (200...299).contains(http.statusCode) else { return }
            if let obj = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                let stars = obj["stargazers_count"] as? Int
            {
                await MainActor.run {
                    githubStars = stars
                }
            }
        } catch {
            // Keep UI clean if network call fails.
        }
    }

    private func openGithubRepository() {
        guard let url = URL(string: "https://github.com/timmyy123/LLM-Hub") else { return }
        openURL(url)
    }
}

struct SmallFeatureCardView: View {
    @EnvironmentObject var settings: AppSettings
    let feature: FeatureCard

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            ZStack {
                Circle()
                    .fill(.ultraThinMaterial)
                    .frame(width: 36, height: 36)
                    .overlay(
                        Circle()
                            .stroke(Color.white.opacity(0.24), lineWidth: 1)
                    )

                Image(systemName: feature.iconSystemName)
                    .font(.system(size: 16, weight: .semibold))
                    .foregroundColor(.white)
            }

            Text(settings.localized(feature.titleKey))
                .font(.subheadline)
                .fontWeight(.bold)
                .foregroundColor(.white)
                .lineLimit(1)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(12)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .background(
            LinearGradient(
                gradient: Gradient(colors: feature.gradient),
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
            .opacity(0.18)
        )
        .background(.ultraThinMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 16))
        .overlay(
            RoundedRectangle(cornerRadius: 16)
                .stroke(
                    LinearGradient(
                        colors: [Color.white.opacity(0.35), Color.white.opacity(0.06)],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    ),
                    lineWidth: 1
                )
        )
        .shadow(color: .black.opacity(0.25), radius: 6, x: 0, y: 4)
    }
}

struct HomeHeroCardView: View {
    @EnvironmentObject var settings: AppSettings
    let feature: FeatureCard
    
    var body: some View {
        ZStack(alignment: .trailing) {
            // Background Image/Icon decoration
            Image(systemName: "bubble.left.and.bubble.right.fill")
                .font(.system(size: 140))
                .foregroundColor(.white.opacity(0.08))
                .offset(x: 20, y: 10)
                .allowsHitTesting(false)
            
            VStack(alignment: .leading, spacing: 12) {
                HStack(spacing: 12) {
                    ZStack {
                        RoundedRectangle(cornerRadius: 14)
                            .fill(.ultraThinMaterial)
                            .frame(width: 50, height: 50)
                            .overlay(
                                RoundedRectangle(cornerRadius: 14)
                                    .stroke(Color.white.opacity(0.24), lineWidth: 1)
                            )
                        Image(systemName: feature.iconSystemName)
                            .font(.system(size: 24, weight: .bold))
                            .foregroundColor(.white)
                    }
                    Text(settings.localized(feature.titleKey))
                        .font(.title2.bold())
                        .foregroundColor(.white)
                }
                
                Text(settings.localized(feature.descriptionKey))
                    .font(.subheadline)
                    .foregroundColor(.white.opacity(0.9))
                    .lineLimit(2)
                    .multilineTextAlignment(.leading)
                
                HStack {
                    Text(settings.localized("chat_now"))
                        .font(.caption.bold())
                        .foregroundColor(Color(hex: "233E88"))
                        .padding(.horizontal, 18)
                        .padding(.vertical, 8)
                        .background(Color.white.opacity(0.75))
                        .cornerRadius(20)
                    Spacer()
                }
            }
            .padding(16)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .background(
            LinearGradient(
                gradient: Gradient(colors: feature.gradient),
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
            .opacity(0.3)
        )
        .background(.ultraThinMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 24))
        .overlay(
            RoundedRectangle(cornerRadius: 24)
                .stroke(
                    LinearGradient(
                        colors: [Color.white.opacity(0.35), Color.white.opacity(0.06)],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    ),
                    lineWidth: 1
                )
        )
        .shadow(color: .black.opacity(0.3), radius: 8, x: 0, y: 6)
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
