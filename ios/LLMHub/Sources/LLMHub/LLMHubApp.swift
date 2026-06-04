import Foundation
import LlamaCPPRuntime
import ONNXRuntime
import RunAnywhere
import SwiftUI
import UIKit

@main
struct LLMHubApp: App {
    @StateObject private var settings = AppSettings.shared
    @StateObject private var consent = ConsentManager.shared

    init() {
        let line = "[LLMHub] App launched\n"
        if let data = line.data(using: .utf8) {
            FileHandle.standardError.write(data)
        }
        NSLog("[LLMHub] App launched")
        UISwitch.appearance().onTintColor = UIColor(ApolloPalette.accentStrong)

        // Initialise AdMob SDK before any ad is shown
        AdMobSDK.initialize()

        // Warm up StoreKit 2 / restore premium state
        Task {
            await PurchaseManager.shared.loadProduct()
        }

        // Request EU/GDPR consent info update — form shown automatically if required
        Task { @MainActor in
            ConsentManager.shared.requestConsentUpdate()
        }

        // Initialize RunAnywhere SDK first — sets up C++ module registry
        // and service infrastructure. Backends MUST be registered after this.
        do {
            try RunAnywhere.initialize(environment: .development)
        } catch {
            // Ignore repeated-initialization errors.
        }

        // Register backends AFTER initialize(), matching the RunAnywhere
        // sample app startup order so the module registry is ready.
        // ONNX.register() is @MainActor — use assumeIsolated so it runs
        // synchronously here on the main thread instead of being deferred.
        LlamaCPP.register(priority: 100)
        MainActor.assumeIsolated {
            ONNX.register()
        }

        registerRunAnywhereModelCatalog()

        Task {
            await RunAnywhere.flushPendingRegistrations()
            _ = await RunAnywhere.discoverDownloadedModels()
        }
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(settings)
                .environmentObject(consent)
                .preferredColorScheme(.dark)
                .environment(\.locale, settings.selectedLanguage.locale)
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

    private func registerRunAnywhereModelCatalog() {
        for model in ModelData.allModels() {
            register(model)
        }
    }

    private func register(_ model: AIModel) {
        guard let primaryURL = URL(string: model.url) else { return }

        if model.additionalFiles.isEmpty {
            RunAnywhere.registerModel(
                id: model.id,
                name: model.name,
                url: primaryURL,
                framework: model.inferenceFramework,
                modality: {
                    switch model.category {
                    case .text:
                        return model.supportsVision ? .multimodal : .language
                    case .multimodal:
                        return .multimodal
                    case .embedding:
                        return .embedding
                    case .imageGeneration:
                        return .imageGeneration
                    }
                }(),
                memoryRequirement: model.sizeBytes,
                contextLength: model.contextWindowSize,
                supportsThinking: model.supportsThinking
            )
            return
        }

        let descriptors = model.allDownloadURLs.map {
            ModelFileDescriptor(url: $0, filename: filename(from: $0), isRequired: true)
        }

        RunAnywhere.registerMultiFileModel(
            id: model.id,
            name: model.name,
            files: descriptors,
            framework: model.inferenceFramework,
            modality: {
                switch model.category {
                case .text:
                    return model.supportsVision ? .multimodal : .language
                case .multimodal:
                    return .multimodal
                case .embedding:
                    return .embedding
                case .imageGeneration:
                    return .imageGeneration
                }
            }(),
            memoryRequirement: model.sizeBytes,
            contextLength: model.contextWindowSize
        )
    }

    private func filename(from url: URL) -> String {
        URLComponents(url: url, resolvingAgainstBaseURL: false)?
            .path
            .split(separator: "/")
            .last
            .map(String.init) ?? url.lastPathComponent
    }
}
