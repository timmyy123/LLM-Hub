import SwiftUI

struct ContentView: View {
    @State private var path = NavigationPath()
    @State private var showPremium = false

    var body: some View {
        NavigationStack(path: $path) {
            HomeScreen(
                onNavigateToChat: { path.append(Screen.chat) },
                onNavigateToModels: { path.append(Screen.models) },
                onNavigateToSettings: { path.append(Screen.settings) },
                onNavigateToRoute: { route in
                    switch route {
                    case "writing_aid":
                        path.append(Screen.writingAid)
                    case "translator":
                        path.append(Screen.translator)
                    case "transcriber":
                        path.append(Screen.transcriber)
                    case "scam_detector":
                        path.append(Screen.scamDetector)
                    case "image_generator":
                        path.append(Screen.imageGenerator)
                    case "vibe_coder":
                        path.append(Screen.vibeCoder)
                    case "vibe_voice":
                        path.append(Screen.vibeVoice)
                    default:
                        break
                    }
                }
            )
            .navigationDestination(for: Screen.self) { screen in
                switch screen {
                case .chat:
                    ChatScreen(
                        onNavigateToSettings: { path.append(Screen.settings) },
                        onNavigateToModels: { path.append(Screen.models) },
                        onNavigateBack: { path.removeLast() }
                    )
                    .navigationBarBackButtonHidden(true)
                    .enableSwipeBack()
                case .models:
                    ModelDownloadScreen(
                        onNavigateBack: { path.removeLast() },
                        onShowPremium: { showPremium = true }
                    )
                    .navigationBarBackButtonHidden(true)
                    .enableSwipeBack()
                case .settings:
                    SettingsScreen(
                        onNavigateBack: { path.removeLast() },
                        onNavigateToModels: { path.append(Screen.models) },
                        onShowPremium: { showPremium = true }
                    )
                    .navigationBarBackButtonHidden(true)
                    .enableSwipeBack()
                case .writingAid:
                    WritingAidScreen(onNavigateBack: { path.removeLast() })
                        .navigationBarBackButtonHidden(true)
                        .enableSwipeBack()
                case .translator:
                    TranslatorScreen(
                        onNavigateBack: { path.removeLast() },
                        onNavigateToModels: { path.append(Screen.models) }
                    )
                    .navigationBarBackButtonHidden(true)
                    .enableSwipeBack()
                case .transcriber:
                    TranscriberScreen(onNavigateBack: { path.removeLast() })
                        .navigationBarBackButtonHidden(true)
                        .enableSwipeBack()
                case .scamDetector:
                    ScamDetectorScreen(onNavigateBack: { path.removeLast() })
                        .navigationBarBackButtonHidden(true)
                        .enableSwipeBack()
                case .vibeCoder:
                    VibeCoderScreen(onNavigateBack: { path.removeLast() })
                        .navigationBarBackButtonHidden(true)
                        .enableSwipeBack()
                case .vibeVoice:
                    VibeVoiceScreen(onNavigateBack: { path.removeLast() })
                        .navigationBarBackButtonHidden(true)
                        .enableSwipeBack()
                case .imageGenerator:
                    ImageGeneratorScreen(
                        onNavigateBack: { path.removeLast() },
                        onNavigateToModels: { path.append(Screen.models) }
                    )
                    .navigationBarBackButtonHidden(true)
                    .enableSwipeBack()
                }
            }
        }
        .sheet(isPresented: $showPremium) {
            PremiumScreen()
                .environmentObject(AppSettings.shared)
        }
    }
}

enum Screen: Hashable {
    case chat
    case models
    case settings
    case writingAid
    case translator
    case transcriber
    case scamDetector
    case vibeCoder
    case vibeVoice
    case imageGenerator
}
