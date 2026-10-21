import Foundation
import SwiftUI
import UIKit
import ModelZoo

final class LLMHubAppDelegate: NSObject, UIApplicationDelegate {
    func application(_ application: UIApplication, handleEventsForBackgroundURLSession identifier: String,
                     completionHandler: @escaping () -> Void) {
        guard identifier == BackgroundModelTransfers.identifier else { completionHandler(); return }
        var finishing: UIBackgroundTaskIdentifier = .invalid
        finishing = application.beginBackgroundTask(withName: "Finish model downloads") {
            if finishing != .invalid { application.endBackgroundTask(finishing); finishing = .invalid }
        }
        BackgroundModelTransfers.shared.handleEvents {
            completionHandler()
            // The native task queue is already submitted. Allow the model actor
            // a short finishing window for validation/extraction and auth retries.
            DispatchQueue.main.asyncAfter(deadline: .now() + 20) {
                if finishing != .invalid { application.endBackgroundTask(finishing); finishing = .invalid }
            }
        }
        // Reattach pending model jobs after a system relaunch, including the
        // final validation/extraction phase once all native transfers finish.
        ModelDownloadViewModel.shared.resumePendingDownloads()
    }

    func application(_ application: UIApplication, supportedInterfaceOrientationsFor window: UIWindow?) -> UIInterfaceOrientationMask {
        // iPhone Duo's folded screen reports one of its landscape poses as upside-down
        // portrait, so allow every orientation there. Elsewhere keep the Info.plist
        // sets: all four on iPad, all but upside-down on iPhone.
        if ApolloDevice.hasHinge || window?.traitCollection.userInterfaceIdiom == .pad {
            return .all
        }
        return .allButUpsideDown
    }
}

@main
struct LLMHubApp: App {
    @UIApplicationDelegateAdaptor(LLMHubAppDelegate.self) private var appDelegate
    @StateObject private var settings = AppSettings.shared

    init() {
        let line = "[LLMHub] App launched\n"
        if let data = line.data(using: .utf8) {
            FileHandle.standardError.write(data)
        }
        NSLog("[LLMHub] App launched")
        #if DEBUG
        Task { @MainActor in DeviceSelfTests.runFromLaunchArguments() }
        #endif
        UISwitch.appearance().onTintColor = UIColor(ApolloPalette.accentStrong)

        // Register upscaler hashes so EnvWrapper.env.ensure can verify them
        ModelZoo.mergeFileSHA256([
            "realesrgan_x2plus_f16.ckpt": "98ce77870b5ca059ec004fe8572182dc67ac8d6a2bba8a938df0ba44fbaccc66",
            "realesrgan_x4plus_f16.ckpt": "3db00086d999e590e313dbf45f0701cdf0e3bca3a66a201a3078423501cb58fd",
            "realesrgan_x4plus_anime_6b_f16.ckpt": "3ad598b21e888590d1bd239dc55675de11b245c691728b56859aa05038c69099",
            "esrgan_4x_universal_upscaler_v2_sharp_f16.ckpt": "05a94d4b3c165f58915f5fafba31512ef5f393011450a40e4437c01d2e33c080",
            "remacri_4x_f16.ckpt": "88d7ae8ecce57de2ad3cb67bdee9937ea320fbaa6319b0d7eb78ea1730b70671",
            "4x_ultrasharp_f16.ckpt": "c8e9a1ee8bf5bc71cef7204bf1cf8cb120dc8b578189d33fd94025a6cfa9f0ec"
        ])

        // Warm up StoreKit 2 / restore premium state
        Task {
            await PurchaseManager.shared.loadProduct()
        }

        #if DEBUG
        // App Store screenshot automation (LLMHubScreenshotUITests); inert unless driven.
        ScreenshotHooks.installLanguageObservers()
        ScreenshotHooks.startOutputServerIfRequested()
        #endif
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(settings)
                .preferredColorScheme(.dark)
                .environment(\.locale, settings.selectedLanguage.locale)
                // App notifications, not `scenePhase`: reading scenePhase here re-evaluated the
                // whole view tree on every inactive/active switch, mid warm-open animation.
                .onReceive(NotificationCenter.default.publisher(for: UIApplication.didBecomeActiveNotification)) { _ in
                    MusicGeneratorBackend.shared.applicationDidBecomeActive()
                }
                .onReceive(NotificationCenter.default.publisher(for: UIApplication.willResignActiveNotification)) { _ in
                    MusicGeneratorBackend.shared.applicationWillResignActive()
                }
                .onReceive(NotificationCenter.default.publisher(for: UIApplication.didEnterBackgroundNotification)) { _ in
                    MusicGeneratorBackend.shared.applicationDidEnterBackground()
                }
                .ifLet(layoutDirectionOverride) { view, dir in
                    view.environment(\.layoutDirection, dir)
                }
        }
    }

    /// Resolves the layout direction override for the currently selected language.
    /// Returns `nil` for `.systemDefault` so iOS handles RTL natively without
    /// SwiftUI mirroring text glyphs — only non-nil when we need to *override*
    /// the system direction (e.g. user picks Arabic on an LTR device, or picks
    /// English on an Arabic-locale device).
    private var layoutDirectionOverride: LayoutDirection? {
        switch settings.selectedLanguage {
        case .systemDefault:
            // Let iOS/SwiftUI handle it naturally — do NOT set layoutDirection.
            // Explicitly setting .rightToLeft here causes SwiftUI to mirror the
            // entire coordinate space, which flips text glyphs on screen.
            return nil
        default:
            return settings.selectedLanguage.isRTL ? .rightToLeft : .leftToRight
        }
    }

}
