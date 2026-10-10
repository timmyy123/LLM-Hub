import Foundation
import SwiftUI

public struct FeatureCardItem: Identifiable {
    public let id = UUID()
    public let titleKey: String
    public let descriptionKey: String
    public let iconSystemName: String
    public let gradient: [Color]
    public let route: String

    public init(titleKey: String, descriptionKey: String, iconSystemName: String, gradient: [Color], route: String) {
        self.titleKey = titleKey
        self.descriptionKey = descriptionKey
        self.iconSystemName = iconSystemName
        self.gradient = gradient
        self.route = route
    }
}

public struct HomeScreen: View {
    @Environment(\.openURL) private var openURL
    @EnvironmentObject var settings: AppSettings
    @ObservedObject private var backend = LLMBackend.shared
    @ObservedObject private var purchases = PurchaseManager.shared

    var onNavigateToRoute: (String) -> Void
    var onShowPremium: () -> Void

    @State private var githubStars: Int? = nil
    @State private var hoveredCard: String? = nil

    public init(onNavigateToRoute: @escaping (String) -> Void, onShowPremium: @escaping () -> Void) {
        self.onNavigateToRoute = onNavigateToRoute
        self.onShowPremium = onShowPremium
    }

    var toolsFeatures: [FeatureCardItem] {
        [
            FeatureCardItem(titleKey: "feature_writing_aid", descriptionKey: "feature_writing_aid_desc", iconSystemName: "pencil.line", gradient: [Color(hex: "91d4ff"), Color(hex: "4e86d5")], route: "writing_aid"),
            FeatureCardItem(titleKey: "feature_translator", descriptionKey: "feature_translator_desc", iconSystemName: "network", gradient: [Color(hex: "84f1cf"), Color(hex: "4aa897")], route: "translator"),
            FeatureCardItem(titleKey: "feature_transcriber", descriptionKey: "feature_transcriber_desc", iconSystemName: "mic.fill", gradient: [Color(hex: "b4b2ff"), Color(hex: "6f77cf")], route: "transcriber"),
            FeatureCardItem(titleKey: "feature_image_generator", descriptionKey: "feature_image_generator_desc", iconSystemName: "paintpalette.fill", gradient: [Color(hex: "9cc3ff"), Color(hex: "5b86d2")], route: "image_generator"),
            FeatureCardItem(titleKey: "feature_image_upscaler", descriptionKey: "feature_image_upscaler_desc", iconSystemName: "wand.and.stars", gradient: [Color(hex: "8fd3ff"), Color(hex: "3f7dd8")], route: "image_upscaler"),
            FeatureCardItem(titleKey: "feature_video_generator", descriptionKey: "feature_video_generator_desc", iconSystemName: "video.fill", gradient: [Color(hex: "ff99c8"), Color(hex: "fc4b93")], route: "video_generator")
        ]
    }

    var utilityFeatures: [FeatureCardItem] {
        [
            FeatureCardItem(titleKey: "feature_vibe_coder", descriptionKey: "feature_vibe_coder_desc", iconSystemName: "chevron.left.slash.chevron.right", gradient: [Color(hex: "a8bcff"), Color(hex: "5f76be")], route: "vibe_coder"),
            FeatureCardItem(titleKey: "feature_agent", descriptionKey: "feature_agent_desc", iconSystemName: "cpu.fill", gradient: [Color(hex: "a78bfa"), Color(hex: "ec4899")], route: "agent"),
            FeatureCardItem(titleKey: "feature_creaitor", descriptionKey: "feature_creaitor_desc", iconSystemName: "person.crop.circle.badge.plus", gradient: [Color(hex: "fbc2eb"), Color(hex: "a6c1ee")], route: "creaitor"),
            FeatureCardItem(titleKey: "feature_scam_detector", descriptionKey: "feature_scam_detector_desc", iconSystemName: "shield.fill", gradient: [Color(hex: "ffb08a"), Color(hex: "d77c59")], route: "scam_detector"),
            FeatureCardItem(titleKey: "feature_vibevoice", descriptionKey: "feature_vibevoice_desc", iconSystemName: "waveform.circle.fill", gradient: [Color(hex: "89d3f7"), Color(hex: "3a68cc")], route: "vibe_voice"),
            FeatureCardItem(titleKey: "feature_music_generator", descriptionKey: "feature_music_generator_desc", iconSystemName: "music.note", gradient: [Color(hex: "ff9a9e"), Color(hex: "fecfef")], route: "music_generator"),
            FeatureCardItem(titleKey: "feature_photo_search", descriptionKey: "feature_photo_search_desc", iconSystemName: "photo.on.rectangle.angled", gradient: [Color(hex: "7fb6ff"), Color(hex: "8e7cff")], route: "photo_search"),
            FeatureCardItem(titleKey: "feature_video_moment", descriptionKey: "feature_video_moment_desc", iconSystemName: "film.stack", gradient: [Color(hex: "6fe3c1"), Color(hex: "3a7bd5")], route: "video_moment")
        ]
    }

    public var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 28) {
                // Top Header Bar
                HStack(alignment: .center) {
                    VStack(alignment: .leading, spacing: 4) {
                        HStack(spacing: 10) {
                            Text(settings.localized("app_name"))
                                .font(.system(size: 28, weight: .bold))
                                .foregroundColor(.white)

                            Text("macOS")
                                .font(.system(size: 11, weight: .semibold))
                                .padding(.horizontal, 7)
                                .padding(.vertical, 3)
                                .background(ApolloPalette.accent.opacity(0.18))
                                .foregroundColor(ApolloPalette.accentStrong)
                                .clipShape(Capsule())
                        }

                        Text("On-device private AI studio powered by Apple Silicon Metal")
                            .font(.system(size: 13))
                            .foregroundColor(.white.opacity(0.6))
                    }

                    Spacer()

                    // Status Bar Actions
                    HStack(spacing: 12) {
                        // Model status pill
                        HStack(spacing: 6) {
                            Circle()
                                .fill(backend.isLoaded ? Color.green : Color.orange)
                                .frame(width: 8, height: 8)
                            Text(backend.currentlyLoadedModel ?? "No Model Active")
                                .font(.system(size: 12, weight: .medium))
                                .foregroundColor(.white.opacity(0.85))
                                .lineLimit(1)
                        }
                        .padding(.horizontal, 10)
                        .padding(.vertical, 6)
                        .background(Color.white.opacity(0.06))
                        .clipShape(Capsule())

                        if !purchases.isPremium {
                            Button {
                                onShowPremium()
                            } label: {
                                HStack(spacing: 5) {
                                    Image(systemName: "crown.fill")
                                        .foregroundStyle(Color(hex: "FFD700"))
                                    Text("Premium")
                                        .font(.system(size: 12, weight: .semibold))
                                }
                                .padding(.horizontal, 10)
                                .padding(.vertical, 6)
                                .background(Color.white.opacity(0.08))
                                .clipShape(Capsule())
                            }
                            .buttonStyle(.plain)
                        }

                        if let stars = githubStars, stars > 0 {
                            Button {
                                if let url = URL(string: "https://github.com/timmyy123/LLM-Hub") {
                                    openURL(url)
                                }
                            } label: {
                                HStack(spacing: 4) {
                                    Image(systemName: "star.fill")
                                        .foregroundStyle(Color.yellow)
                                    Text("\(stars)")
                                        .font(.system(size: 12, weight: .semibold))
                                }
                                .padding(.horizontal, 10)
                                .padding(.vertical, 6)
                                .background(Color.white.opacity(0.08))
                                .clipShape(Capsule())
                            }
                            .buttonStyle(.plain)
                        }
                    }
                }

                // Hero Card: AI Chat
                heroChatCard

                // AI Tools Suite Section
                VStack(alignment: .leading, spacing: 14) {
                    HStack {
                        Image(systemName: "wand.and.stars")
                            .foregroundColor(ApolloPalette.accentStrong)
                        Text(settings.localized("tools_title"))
                            .font(.system(size: 18, weight: .bold))
                            .foregroundColor(.white)
                    }

                    LazyVGrid(columns: [GridItem(.adaptive(minimum: 260, maximum: 380), spacing: 16)], spacing: 16) {
                        ForEach(toolsFeatures) { item in
                            featureCard(item)
                        }
                    }
                }

                // Utility & AI Intelligence Suite Section
                VStack(alignment: .leading, spacing: 14) {
                    HStack {
                        Image(systemName: "square.grid.2x2")
                            .foregroundColor(ApolloPalette.accent)
                        Text(settings.localized("utility_title"))
                            .font(.system(size: 18, weight: .bold))
                            .foregroundColor(.white)
                    }

                    LazyVGrid(columns: [GridItem(.adaptive(minimum: 260, maximum: 380), spacing: 16)], spacing: 16) {
                        ForEach(utilityFeatures) { item in
                            featureCard(item)
                        }
                    }
                }
            }
            .padding(32)
        }
        .apolloScreenBackground()
        .onAppear {
            fetchGithubStars()
        }
    }

    private var heroChatCard: some View {
        Button {
            onNavigateToRoute("chat")
        } label: {
            HStack(spacing: 24) {
                VStack(alignment: .leading, spacing: 10) {
                    HStack(spacing: 8) {
                        Image(systemName: "sparkles")
                            .foregroundColor(ApolloPalette.accentStrong)
                        Text("FEATURED")
                            .font(.system(size: 11, weight: .bold))
                            .foregroundColor(ApolloPalette.accentStrong)
                    }

                    Text(settings.localized("feature_ai_chat"))
                        .font(.system(size: 24, weight: .bold))
                        .foregroundColor(.white)

                    Text(settings.localized("feature_ai_chat_desc"))
                        .font(.system(size: 13))
                        .foregroundColor(.white.opacity(0.75))
                        .lineLimit(2)

                    HStack(spacing: 8) {
                        Text("Start Chatting")
                            .font(.system(size: 13, weight: .semibold))
                            .foregroundColor(.white)
                        Image(systemName: "arrow.right")
                            .font(.system(size: 12, weight: .semibold))
                            .foregroundColor(.white)
                    }
                    .padding(.top, 4)
                }

                Spacer()

                ZStack {
                    Circle()
                        .fill(
                            LinearGradient(
                                colors: [Color(hex: "7ea3ff"), Color(hex: "5e79da")],
                                startPoint: .topLeading,
                                endPoint: .bottomTrailing
                            )
                        )
                        .frame(width: 80, height: 80)
                        .shadow(color: Color(hex: "7ea3ff").opacity(0.4), radius: 14, x: 0, y: 6)

                    Image(systemName: "bubble.left.and.bubble.right.fill")
                        .font(.system(size: 34))
                        .foregroundColor(.white)
                }
            }
            .padding(24)
            .background(
                RoundedRectangle(cornerRadius: 18)
                    .fill(
                        LinearGradient(
                            colors: [Color(hex: "1f2b48").opacity(0.85), Color(hex: "131b2e").opacity(0.85)],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        )
                    )
            )
            .overlay(
                RoundedRectangle(cornerRadius: 18)
                    .stroke(LinearGradient(colors: [ApolloPalette.accentStrong.opacity(0.4), Color.clear], startPoint: .topLeading, endPoint: .bottomTrailing), lineWidth: 1.5)
            )
            .shadow(color: Color.black.opacity(0.3), radius: 12, x: 0, y: 6)
        }
        .buttonStyle(.plain)
    }

    private func featureCard(_ item: FeatureCardItem) -> some View {
        Button {
            onNavigateToRoute(item.route)
        } label: {
            HStack(spacing: 16) {
                ZStack {
                    RoundedRectangle(cornerRadius: 12)
                        .fill(
                            LinearGradient(
                                colors: item.gradient,
                                startPoint: .topLeading,
                                endPoint: .bottomTrailing
                            )
                        )
                        .frame(width: 48, height: 48)
                        .shadow(color: item.gradient.first?.opacity(0.3) ?? .clear, radius: 6, x: 0, y: 3)

                    Image(systemName: item.iconSystemName)
                        .font(.system(size: 22))
                        .foregroundColor(.white)
                }

                VStack(alignment: .leading, spacing: 4) {
                    Text(settings.localized(item.titleKey))
                        .font(.system(size: 15, weight: .semibold))
                        .foregroundColor(.white)

                    Text(settings.localized(item.descriptionKey))
                        .font(.system(size: 12))
                        .foregroundColor(.white.opacity(0.65))
                        .lineLimit(2)
                }

                Spacer(minLength: 0)

                Image(systemName: "chevron.right")
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundColor(.white.opacity(hoveredCard == item.route ? 0.9 : 0.3))
            }
            .padding(16)
            .background(
                RoundedRectangle(cornerRadius: 14)
                    .fill(ApolloPalette.bgSurface.opacity(hoveredCard == item.route ? 0.9 : 0.65))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 14)
                    .stroke(hoveredCard == item.route ? ApolloPalette.accent.opacity(0.5) : ApolloPalette.borderGlass, lineWidth: 1)
            )
            .scaleEffect(hoveredCard == item.route ? 1.015 : 1.0)
            .animation(.easeOut(duration: 0.15), value: hoveredCard)
        }
        .buttonStyle(.plain)
        .onHover { isHovered in
            hoveredCard = isHovered ? item.route : nil
        }
    }

    private func fetchGithubStars() {
        guard let url = URL(string: "https://api.github.com/repos/timmyy123/LLM-Hub") else { return }
        Task {
            if let (data, _) = try? await URLSession.shared.data(from: url),
               let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let stars = json["stargazers_count"] as? Int {
                await MainActor.run {
                    self.githubStars = stars
                }
            }
        }
    }
}
