import SwiftUI

public enum MacScreen: String, CaseIterable, Identifiable {
    case home = "home"
    case chat = "chat"
    case models = "models"
    case vibeCoder = "vibe_coder"
    case writingAid = "writing_aid"
    case translator = "translator"
    case transcriber = "transcriber"
    case scamDetector = "scam_detector"
    case vibeVoice = "vibe_voice"
    case imageGenerator = "image_generator"
    case imageUpscaler = "image_upscaler"
    case videoGenerator = "video_generator"
    case musicGenerator = "music_generator"
    case photoSearch = "photo_search"
    case audioSearch = "audio_search"
    case videoMoment = "video_moment"
    case agent = "agent"
    case creaitor = "creaitor"
    case settings = "settings"

    public var id: String { rawValue }

    public var title: String {
        switch self {
        case .home: return "Dashboard"
        case .chat: return "AI Chat"
        case .models: return "Model Manager"
        case .vibeCoder: return "Vibe Coder"
        case .writingAid: return "Writing Aid"
        case .translator: return "Translator"
        case .transcriber: return "Transcriber"
        case .scamDetector: return "Scam Detector"
        case .vibeVoice: return "Vibe Voice"
        case .imageGenerator: return "Image Generator"
        case .imageUpscaler: return "Image Upscaler"
        case .videoGenerator: return "Video Generator"
        case .musicGenerator: return "Music Generator"
        case .photoSearch: return "Photo Search"
        case .audioSearch: return "Audio Search"
        case .videoMoment: return "Video Moment"
        case .agent: return "AI Agent"
        case .creaitor: return "creAItor Persona"
        case .settings: return "Settings"
        }
    }

    public var icon: String {
        switch self {
        case .home: return "square.grid.2x2"
        case .chat: return "bubble.left.and.bubble.right.fill"
        case .models: return "arrow.down.circle"
        case .vibeCoder: return "chevron.left.slash.chevron.right"
        case .writingAid: return "pencil.line"
        case .translator: return "network"
        case .transcriber: return "mic.fill"
        case .scamDetector: return "shield.fill"
        case .vibeVoice: return "waveform.circle.fill"
        case .imageGenerator: return "paintpalette.fill"
        case .imageUpscaler: return "wand.and.stars"
        case .videoGenerator: return "video.fill"
        case .musicGenerator: return "music.note"
        case .photoSearch: return "photo.on.rectangle.angled"
        case .audioSearch: return "speaker.wave.3.fill"
        case .videoMoment: return "film.stack"
        case .agent: return "cpu.fill"
        case .creaitor: return "person.crop.circle.badge.plus"
        case .settings: return "gearshape"
        }
    }
}

public struct ContentView: View {
    @EnvironmentObject var settings: AppSettings
    @ObservedObject private var backend = LLMBackend.shared
    @ObservedObject private var purchases = PurchaseManager.shared

    @State private var selectedScreen: MacScreen? = .home
    @State private var showPremium: Bool = false

    public init() {}

    public var body: some View {
        NavigationSplitView {
            sidebarContent
                .frame(minWidth: 230, idealWidth: 260, maxWidth: 320)
        } detail: {
            detailContent
        }
        .sheet(isPresented: $showPremium) {
            PremiumScreen()
                .environmentObject(settings)
        }
    }

    // MARK: - Sidebar
    private var sidebarContent: some View {
        VStack(spacing: 0) {
            // App Branding Header
            HStack(spacing: 10) {
                ZStack {
                    RoundedRectangle(cornerRadius: 8)
                        .fill(ApolloPalette.accent)
                        .frame(width: 28, height: 28)
                    Image(systemName: "cpu")
                        .font(.system(size: 14, weight: .bold))
                        .foregroundColor(.black)
                }

                VStack(alignment: .leading, spacing: 1) {
                    Text("LLM Hub")
                        .font(.system(size: 15, weight: .bold))
                        .foregroundColor(.white)
                    Text("macOS Studio")
                        .font(.system(size: 10))
                        .foregroundColor(.white.opacity(0.5))
                }

                Spacer()

                if !purchases.isPremium {
                    Button {
                        showPremium = true
                    } label: {
                        Image(systemName: "crown.fill")
                            .font(.system(size: 13))
                            .foregroundStyle(Color(hex: "FFD700"))
                            .padding(6)
                            .background(Color.white.opacity(0.08))
                            .clipShape(Circle())
                    }
                    .buttonStyle(.plain)
                    .help("Upgrade to Premium")
                }
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 14)
            .background(Color(hex: "080c14"))

            Divider().background(Color.white.opacity(0.08))

            // Navigation Sections
            List(selection: $selectedScreen) {
                Section("Workspace") {
                    sidebarItem(.home)
                    sidebarItem(.chat)
                    sidebarItem(.models)
                }

                Section("Creation & Coding") {
                    sidebarItem(.vibeCoder)
                    sidebarItem(.writingAid)
                    sidebarItem(.imageGenerator)
                    sidebarItem(.imageUpscaler)
                    sidebarItem(.videoGenerator)
                    sidebarItem(.musicGenerator)
                }

                Section("Intelligence & Tools") {
                    sidebarItem(.agent)
                    sidebarItem(.creaitor)
                    sidebarItem(.translator)
                    sidebarItem(.transcriber)
                    sidebarItem(.scamDetector)
                    sidebarItem(.vibeVoice)
                }

                Section("Media Search") {
                    sidebarItem(.photoSearch)
                    sidebarItem(.audioSearch)
                    sidebarItem(.videoMoment)
                }

                Section("System") {
                    sidebarItem(.settings)
                }
            }
            .listStyle(.sidebar)

            Divider().background(Color.white.opacity(0.08))

            // Bottom Status Bar
            HStack(spacing: 8) {
                Circle()
                    .fill(backend.isLoaded ? Color.green : Color.orange)
                    .frame(width: 8, height: 8)

                Text(backend.currentlyLoadedModel ?? "No Model Loaded")
                    .font(.system(size: 11))
                    .foregroundColor(.white.opacity(0.75))
                    .lineLimit(1)

                Spacer()

                Button {
                    selectedScreen = .settings
                } label: {
                    Image(systemName: "gearshape")
                        .font(.system(size: 12))
                        .foregroundColor(.white.opacity(0.5))
                }
                .buttonStyle(.plain)
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 10)
            .background(Color(hex: "080c14"))
        }
        .background(Color(hex: "0a0e18"))
    }

    private func sidebarItem(_ screen: MacScreen) -> some View {
        NavigationLink(value: screen) {
            Label {
                Text(screen.title)
                    .font(.system(size: 13))
            } icon: {
                Image(systemName: screen.icon)
                    .font(.system(size: 13))
                    .foregroundColor(selectedScreen == screen ? ApolloPalette.accentStrong : .white.opacity(0.7))
            }
        }
    }

    // MARK: - Detail Content
    @ViewBuilder
    private var detailContent: some View {
        switch selectedScreen ?? .home {
        case .home:
            HomeScreen(
                onNavigateToRoute: { route in
                    if let found = MacScreen.allCases.first(where: { $0.rawValue == route }) {
                        selectedScreen = found
                    }
                },
                onShowPremium: { showPremium = true }
            )

        case .chat:
            ChatScreen(onNavigateToModels: { selectedScreen = .models })

        case .models:
            ModelDownloadScreen(onShowPremium: { showPremium = true })

        case .settings:
            SettingsScreen(onShowPremium: { showPremium = true })

        case .writingAid:
            WritingAidScreen()

        case .translator:
            TranslatorScreen()

        case .transcriber:
            TranscriberScreen()

        case .scamDetector:
            ScamDetectorScreen()

        case .vibeCoder:
            VibeCoderScreen()

        case .vibeVoice:
            VibeVoiceScreen()

        case .imageGenerator:
            ImageGeneratorScreen()

        case .imageUpscaler:
            ImageUpscalerScreen()

        case .videoGenerator:
            VideoGeneratorScreen()

        case .musicGenerator:
            MusicGeneratorScreen()

        case .photoSearch:
            PhotoSearchScreen()

        case .audioSearch:
            AudioSearchScreen()

        case .videoMoment:
            VideoMomentScreen()

        case .agent:
            AgentScreen()

        case .creaitor:
            CreAItorScreen()
        }
    }
}
