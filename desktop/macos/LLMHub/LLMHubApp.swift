import AppKit
import SwiftUI

@main
struct LLMHubApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var delegate
    @State private var app = AppModel.shared

    var body: some Scene {
        Window("LLM Hub", id: "main") {
            RootView()
                .environment(app)
                .frame(minWidth: 1000, minHeight: 640)
        }
        .defaultSize(width: 1440, height: 920)
        .commands {
            CommandGroup(replacing: .newItem) {
                Button(L.s("NewChat")) {
                    app.chat.newChat()
                    app.section = .chat
                }
                .keyboardShortcut("n")
            }
            CommandGroup(replacing: .appSettings) {
                Button(L.s("NavSettings")) { app.section = .settings(.general) }
                    .keyboardShortcut(",")
            }
        }
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        MainActor.assumeIsolated { Self.applyAppearance() }
        #if DEBUG
        DevTools.runIfRequested()
        #endif
    }

    /// Dark is the designed look; light and system are in Settings › General.
    @MainActor static func applyAppearance() {
        switch AppSettings.current.appearance {
        case "light": NSApp.appearance = NSAppearance(named: .aqua)
        case "system": NSApp.appearance = nil
        default: NSApp.appearance = NSAppearance(named: .darkAqua)
        }
    }

    func applicationWillTerminate(_ notification: Notification) {
        MainActor.assumeIsolated {
            AppModel.shared.engines.stop()
            AppModel.shared.embeddings.stop()
        }
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }
}
