import SwiftUI

@main
struct LLMHubApp: App {
    @StateObject private var settings = AppSettings.shared

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(settings)
                .preferredColorScheme(settings.theme.colorScheme)
                .environment(\.locale, settings.selectedLanguage.locale)
                .environment(\.layoutDirection, settings.selectedLanguage.isRTL ? .rightToLeft : .leftToRight)
        }
    }
}
