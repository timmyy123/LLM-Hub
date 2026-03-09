import SwiftUI

struct ContentView: View {
    @State private var path = NavigationPath()

    var body: some View {
        NavigationStack(path: $path) {
            HomeScreen(
                onNavigateToChat: { path.append(Screen.chat) },
                onNavigateToModels: { path.append(Screen.models) },
                onNavigateToSettings: { path.append(Screen.settings) }
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
                case .models:
                    ModelDownloadScreen(onNavigateBack: { path.removeLast() })
                        .navigationBarBackButtonHidden(true)
                case .settings:
                    SettingsScreen(
                        onNavigateBack: { path.removeLast() },
                        onNavigateToModels: { path.append(Screen.models) }
                    )
                    .navigationBarBackButtonHidden(true)
                }
            }
        }
    }
}

enum Screen: Hashable {
    case chat
    case models
    case settings
}
