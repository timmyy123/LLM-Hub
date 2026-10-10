import Foundation
import SwiftUI
import AppKit

@main
struct LLMHubMacApp: App {
    @StateObject private var settings = AppSettings.shared

    init() {
        NSLog("[LLMHub macOS] App launched")

        // Register model SHA256 hashes if needed
        Task {
            await PurchaseManager.shared.loadProduct()
        }
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(settings)
                .preferredColorScheme(.dark)
                .environment(\.locale, settings.selectedLanguage.locale)
                .frame(minWidth: 980, minHeight: 650)
        }
        .windowStyle(.titleBar)
        .windowToolbarStyle(.unified)
        .commands {
            CommandGroup(replacing: .newItem) {
                Button("New Chat") {
                    ChatStore.shared.createNewConversation()
                }
                .keyboardShortcut("n", modifiers: .command)
            }

            CommandMenu("Inference") {
                Button("Stop Generation") {
                    LLMBackend.shared.stopGeneration()
                }
                .keyboardShortcut(".", modifiers: .command)
            }
        }

        #if os(macOS)
        Settings {
            SettingsScreen(onShowPremium: {})
                .environmentObject(settings)
                .preferredColorScheme(.dark)
                .frame(width: 600, height: 500)
        }
        #endif
    }
}
