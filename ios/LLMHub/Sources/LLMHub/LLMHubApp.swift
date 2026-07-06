import Foundation
import LlamaCPPRuntime
import ONNXRuntime
import RunAnywhere
import SwiftUI
import UIKit
import ModelZoo

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

        // Register upscaler hashes so EnvWrapper.env.ensure can verify them
        ModelZoo.mergeFileSHA256([
            "realesrgan_x2plus_f16.ckpt": "98ce77870b5ca059ec004fe8572182dc67ac8d6a2bba8a938df0ba44fbaccc66",
            "realesrgan_x4plus_f16.ckpt": "3db00086d999e590e313dbf45f0701cdf0e3bca3a66a201a3078423501cb58fd",
            "realesrgan_x4plus_anime_6b_f16.ckpt": "3ad598b21e888590d1bd239dc55675de11b245c691728b56859aa05038c69099",
            "esrgan_4x_universal_upscaler_v2_sharp_f16.ckpt": "05a94d4b3c165f58915f5fafba31512ef5f393011450a40e4437c01d2e33c080",
            "remacri_4x_f16.ckpt": "88d7ae8ecce57de2ad3cb67bdee9937ea320fbaa6319b0d7eb78ea1730b70671",
            "4x_ultrasharp_f16.ckpt": "c8e9a1ee8bf5bc71cef7204bf1cf8cb120dc8b578189d33fd94025a6cfa9f0ec"
        ])

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
        guard model.modelFormat != .drawthings else { return }
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
                    case .videoGeneration:
                        return .imageGeneration // or .videoGeneration if RunAnywhere has it
                    case .imageUpscale:
                        return .imageGeneration
                    case .asr:
                        return .language
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
                case .videoGeneration:
                    return .imageGeneration
                case .imageUpscale:
                    return .imageGeneration
                case .asr:
                    return .language
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
