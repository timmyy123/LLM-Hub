import Foundation
import SwiftUI
@preconcurrency import AVFoundation
import CoreMedia
import PhotosUI
@preconcurrency import Speech
import UniformTypeIdentifiers
import RunAnywhere
#if canImport(UIKit)
import UIKit
#endif
#if canImport(FoundationModels)
import FoundationModels
#endif
#if canImport(Network)
import Network
#endif

private enum WritingAidMode: String, CaseIterable {
    case friendly = "writing_aid_tone_friendly"
    case professional = "writing_aid_tone_professional"
    case concise = "writing_aid_tone_concise"
}

private struct TranslatorLanguage: Identifiable, Hashable {
    let code: String
    let localizationKey: String

    var id: String { code }
}

private let translatorLanguageEnglishNames: [String: String] = [
    "en": "English",
    "af": "Afrikaans",
    "am": "Amharic",
    "ar": "Arabic",
    "hy": "Armenian",
    "az": "Azerbaijani",
    "eu": "Basque",
    "bn": "Bengali",
    "bg": "Bulgarian",
    "my": "Burmese",
    "ca": "Catalan",
    "zh-CN": "Chinese (Simplified)",
    "zh-TW": "Chinese (Traditional)",
    "hr": "Croatian",
    "cs": "Czech",
    "da": "Danish",
    "nl": "Dutch",
    "et": "Estonian",
    "tl": "Filipino",
    "fi": "Finnish",
    "fr": "French",
    "gl": "Galician",
    "ka": "Georgian",
    "de": "German",
    "el": "Greek",
    "gu": "Gujarati",
    "ha": "Hausa",
    "he": "Hebrew",
    "hi": "Hindi",
    "hu": "Hungarian",
    "is": "Icelandic",
    "ig": "Igbo",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "kn": "Kannada",
    "kk": "Kazakh",
    "km": "Khmer",
    "ko": "Korean",
    "lo": "Lao",
    "lv": "Latvian",
    "lt": "Lithuanian",
    "ms": "Malay",
    "ml": "Malayalam",
    "mr": "Marathi",
    "ne": "Nepali",
    "no": "Norwegian",
    "ps": "Pashto",
    "fa": "Persian",
    "pl": "Polish",
    "pt": "Portuguese",
    "pa": "Punjabi",
    "ro": "Romanian",
    "ru": "Russian",
    "sr": "Serbian",
    "sd": "Sindhi",
    "si": "Sinhala",
    "sk": "Slovak",
    "sl": "Slovenian",
    "so": "Somali",
    "es": "Spanish",
    "sw": "Swahili",
    "sv": "Swedish",
    "ta": "Tamil",
    "te": "Telugu",
    "th": "Thai",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "uz": "Uzbek",
    "vi": "Vietnamese",
    "yo": "Yoruba",
    "zu": "Zulu",
]

private let translatorLanguages: [TranslatorLanguage] = [
    TranslatorLanguage(code: "en", localizationKey: "lang_english"),
    TranslatorLanguage(code: "af", localizationKey: "lang_afrikaans"),
    TranslatorLanguage(code: "am", localizationKey: "lang_amharic"),
    TranslatorLanguage(code: "ar", localizationKey: "lang_arabic"),
    TranslatorLanguage(code: "hy", localizationKey: "lang_armenian"),
    TranslatorLanguage(code: "az", localizationKey: "lang_azerbaijani"),
    TranslatorLanguage(code: "eu", localizationKey: "lang_basque"),
    TranslatorLanguage(code: "bn", localizationKey: "lang_bengali"),
    TranslatorLanguage(code: "bg", localizationKey: "lang_bulgarian"),
    TranslatorLanguage(code: "my", localizationKey: "lang_burmese"),
    TranslatorLanguage(code: "ca", localizationKey: "lang_catalan"),
    TranslatorLanguage(code: "zh-CN", localizationKey: "lang_chinese"),
    TranslatorLanguage(code: "zh-TW", localizationKey: "lang_chinese_traditional"),
    TranslatorLanguage(code: "hr", localizationKey: "lang_croatian"),
    TranslatorLanguage(code: "cs", localizationKey: "lang_czech"),
    TranslatorLanguage(code: "da", localizationKey: "lang_danish"),
    TranslatorLanguage(code: "nl", localizationKey: "lang_dutch"),
    TranslatorLanguage(code: "et", localizationKey: "lang_estonian"),
    TranslatorLanguage(code: "tl", localizationKey: "lang_filipino"),
    TranslatorLanguage(code: "fi", localizationKey: "lang_finnish"),
    TranslatorLanguage(code: "fr", localizationKey: "lang_french"),
    TranslatorLanguage(code: "gl", localizationKey: "lang_galician"),
    TranslatorLanguage(code: "ka", localizationKey: "lang_georgian"),
    TranslatorLanguage(code: "de", localizationKey: "lang_german"),
    TranslatorLanguage(code: "el", localizationKey: "lang_greek"),
    TranslatorLanguage(code: "gu", localizationKey: "lang_gujarati"),
    TranslatorLanguage(code: "ha", localizationKey: "lang_hausa"),
    TranslatorLanguage(code: "he", localizationKey: "lang_hebrew"),
    TranslatorLanguage(code: "hi", localizationKey: "lang_hindi"),
    TranslatorLanguage(code: "hu", localizationKey: "lang_hungarian"),
    TranslatorLanguage(code: "is", localizationKey: "lang_icelandic"),
    TranslatorLanguage(code: "ig", localizationKey: "lang_igbo"),
    TranslatorLanguage(code: "id", localizationKey: "lang_indonesian"),
    TranslatorLanguage(code: "it", localizationKey: "lang_italian"),
    TranslatorLanguage(code: "ja", localizationKey: "lang_japanese"),
    TranslatorLanguage(code: "kn", localizationKey: "lang_kannada"),
    TranslatorLanguage(code: "kk", localizationKey: "lang_kazakh"),
    TranslatorLanguage(code: "km", localizationKey: "lang_khmer"),
    TranslatorLanguage(code: "ko", localizationKey: "lang_korean"),
    TranslatorLanguage(code: "lo", localizationKey: "lang_lao"),
    TranslatorLanguage(code: "lv", localizationKey: "lang_latvian"),
    TranslatorLanguage(code: "lt", localizationKey: "lang_lithuanian"),
    TranslatorLanguage(code: "ms", localizationKey: "lang_malay"),
    TranslatorLanguage(code: "ml", localizationKey: "lang_malayalam"),
    TranslatorLanguage(code: "mr", localizationKey: "lang_marathi"),
    TranslatorLanguage(code: "ne", localizationKey: "lang_nepali"),
    TranslatorLanguage(code: "no", localizationKey: "lang_norwegian"),
    TranslatorLanguage(code: "ps", localizationKey: "lang_pashto"),
    TranslatorLanguage(code: "fa", localizationKey: "lang_persian"),
    TranslatorLanguage(code: "pl", localizationKey: "lang_polish"),
    TranslatorLanguage(code: "pt", localizationKey: "lang_portuguese"),
    TranslatorLanguage(code: "pa", localizationKey: "lang_punjabi"),
    TranslatorLanguage(code: "ro", localizationKey: "lang_romanian"),
    TranslatorLanguage(code: "ru", localizationKey: "lang_russian"),
    TranslatorLanguage(code: "sr", localizationKey: "lang_serbian"),
    TranslatorLanguage(code: "sd", localizationKey: "lang_sindhi"),
    TranslatorLanguage(code: "si", localizationKey: "lang_sinhala"),
    TranslatorLanguage(code: "sk", localizationKey: "lang_slovak"),
    TranslatorLanguage(code: "sl", localizationKey: "lang_slovenian"),
    TranslatorLanguage(code: "so", localizationKey: "lang_somali"),
    TranslatorLanguage(code: "es", localizationKey: "lang_spanish"),
    TranslatorLanguage(code: "sw", localizationKey: "lang_swahili"),
    TranslatorLanguage(code: "sv", localizationKey: "lang_swedish"),
    TranslatorLanguage(code: "ta", localizationKey: "lang_tamil"),
    TranslatorLanguage(code: "te", localizationKey: "lang_telugu"),
    TranslatorLanguage(code: "th", localizationKey: "lang_thai"),
    TranslatorLanguage(code: "tr", localizationKey: "lang_turkish"),
    TranslatorLanguage(code: "uk", localizationKey: "lang_ukrainian"),
    TranslatorLanguage(code: "ur", localizationKey: "lang_urdu"),
    TranslatorLanguage(code: "uz", localizationKey: "lang_uzbek"),
    TranslatorLanguage(code: "vi", localizationKey: "lang_vietnamese"),
    TranslatorLanguage(code: "yo", localizationKey: "lang_yoruba"),
    TranslatorLanguage(code: "zu", localizationKey: "lang_zulu"),
]

private func persistentAudioStorageDirectory() -> URL {
    let fileManager = FileManager.default
    let base = fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
        ?? fileManager.urls(for: .documentDirectory, in: .userDomainMask).first
        ?? fileManager.temporaryDirectory
    let dir = base.appendingPathComponent("LLMHubAudio", isDirectory: true)
    try? fileManager.createDirectory(at: dir, withIntermediateDirectories: true)
    return dir
}

private func isTranslateGemmaModel(_ model: AIModel) -> Bool {
    !model.isDependencyOnly
        && model.category == .multimodal
        && model.supportsVision
        && model.name.hasPrefix("Translate Gemma 4B")
}

@MainActor
private func downloadableTranslatorModels() -> [AIModel] {
    downloadableFeatureModels().filter(isTranslatorSupportedModel)
}

private func isTranslatorSupportedModel(_ model: AIModel) -> Bool {
    !model.isDependencyOnly
        && model.category == .multimodal
        && model.supportsVision
        && (model.name.hasPrefix("Translate Gemma 4B") || model.name.hasPrefix("Gemma 4 "))
}

private func usesGemma4TurnTemplate(_ model: AIModel) -> Bool {
    model.name.hasPrefix("Gemma 4 ")
}

private func isNonTranslatorFeatureModel(_ model: AIModel) -> Bool {
    !model.name.hasPrefix("Translate Gemma")
}

private func translatorQuantizationTag(for modelName: String) -> String? {
    guard let leftParen = modelName.lastIndex(of: "("),
          let rightParen = modelName.lastIndex(of: ")"),
          leftParen < rightParen else {
        return nil
    }
    let tag = modelName[modelName.index(after: leftParen)..<rightParen]
        .trimmingCharacters(in: .whitespacesAndNewlines)
        .lowercased()
    return tag.isEmpty ? nil : tag
}

private func translatorVisionFamilyName(for modelName: String) -> String {
    let base = modelName.components(separatedBy: " (").first ?? modelName
    return base
        .replacingOccurrences(of: "Vision Projector", with: "", options: [.caseInsensitive])
        .replacingOccurrences(of: "Projector", with: "", options: [.caseInsensitive])
        .trimmingCharacters(in: .whitespacesAndNewlines)
        .lowercased()
}

@MainActor
private func translatorHasDownloadedVisionProjector(for model: AIModel) -> Bool {
    guard model.modelFormat == .gguf, model.supportsVision else { return true }

    let family = translatorVisionFamilyName(for: model.name)
    let quantTag = translatorQuantizationTag(for: model.name)

    let candidates = ModelData.allModels().filter { candidate in
        candidate.isDependencyOnly
            && candidate.inferenceFramework == model.inferenceFramework
            && translatorVisionFamilyName(for: candidate.name) == family
            && RunAnywhere.isModelDownloaded(candidate.id, framework: candidate.inferenceFramework)
    }

    guard !candidates.isEmpty else { return false }

    if family.hasPrefix("gemma 4 e") {
        return candidates.contains {
            $0.name.lowercased().contains("f16") || $0.url.lowercased().contains("f16")
        }
    }

    if quantTag == "f16" {
        return candidates.contains { ($0.name.lowercased().contains("f16") || $0.url.lowercased().contains("f16")) }
    }

    return candidates.contains {
        $0.name.lowercased().contains("q8_0")
            || $0.url.lowercased().contains("q8_0")
            || $0.name.lowercased().contains("bf16")
    }
}

@MainActor
private func hasDownloadedVisionProjector(for model: AIModel) -> Bool {
    guard model.modelFormat == .gguf, model.supportsVision else { return true }
    return ModelData.allModels().contains { candidate in
        candidate.isDependencyOnly
            && candidate.inferenceFramework == model.inferenceFramework
            && RunAnywhere.isModelDownloaded(candidate.id, framework: candidate.inferenceFramework)
    }
}

@MainActor
private func downloadableFeatureModels() -> [AIModel] {
    let legacyModelsDir: URL? = {
        guard let documentsDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else { return nil }
        return documentsDir.appendingPathComponent("models")
    }()

    func requiredFilesExist(in directory: URL, for model: AIModel) -> Bool {
        guard !model.requiredFileNames.isEmpty else { return false }

        let completionThreshold = 0.98
        let minimumExpectedBytes = Int64(Double(model.sizeBytes) * completionThreshold)
        var totalBytes: Int64 = 0

        let allExist = model.requiredFileNames.allSatisfy { fileName in
            let fileURL = directory.appendingPathComponent(fileName)
            guard FileManager.default.fileExists(atPath: fileURL.path) else { return false }
            let size = (try? FileManager.default.attributesOfItem(atPath: fileURL.path))?[.size] as? Int64 ?? 0
            totalBytes += size
            return true
        }

        return allExist && (minimumExpectedBytes <= 0 || totalBytes >= minimumExpectedBytes)
    }

    var models = ModelData.allModels().filter { model in
        if model.isDependencyOnly { return false }
        if model.category == .embedding || model.category == .imageGeneration { return false }

        guard ModelData.isModelFullyAvailableLocally(model) else { return false }
        return true
    }

    if let appleModel = appleFoundationModelIfAvailable(),
       !models.contains(where: { $0.id == appleModel.id }) {
        models.append(appleModel)
    }

    return models
}

@MainActor
private func appleFoundationModelIfAvailable() -> AIModel? {
    #if canImport(FoundationModels)
    if #available(iOS 26.0, *) {
        let model = SystemLanguageModel.default
        guard model.isAvailable else { return nil }

        return AIModel(
            id: "apple.foundation.system",
            name: "Apple Foundation Model",
            description: "On-device Apple Intelligence foundation model.",
            url: "apple://foundation-model",
            category: .text,
            sizeBytes: 0,
            source: "Apple",
            supportsVision: false,
            supportsAudio: false,
            supportsThinking: false,
            supportsGpu: true,
            requirements: ModelRequirements(minRamGB: 8, recommendedRamGB: 8),
            contextWindowSize: max(1, model.contextSize),
            modelFormat: .gguf,
            additionalFiles: []
        )
    }
    #endif

    return nil
}

@MainActor
private func selectedFeatureModel(named selectedModelName: String) -> AIModel? {
    downloadableFeatureModels().first(where: { $0.name == selectedModelName })
        ?? ModelData.allModels().first(where: { $0.name == selectedModelName })
}

@MainActor
private func syncRunAnywhereModelDiscovery() async {
    do {
        try RunAnywhere.initialize(environment: .development)
    } catch {
        // Ignore repeated initialization attempts.
    }
    _ = await RunAnywhere.discoverDownloadedModels()
}

@MainActor
private func dismissKeyboard() {
    #if canImport(UIKit)
    UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
    #endif
}

private func sanitizeModelOutputText(_ text: String) -> String {
    text
        .replacingOccurrences(of: "â€™", with: "'")
        .replacingOccurrences(of: "â€˜", with: "'")
        .replacingOccurrences(of: "â€œ", with: "\"")
        .replacingOccurrences(of: "â€", with: "\"")
        .replacingOccurrences(of: "â€“", with: "-")
        .replacingOccurrences(of: "â€”", with: "-")
        .replacingOccurrences(of: "�", with: "'")
    .replacingOccurrences(of: "<|startoftext|>", with: "")
    .replacingOccurrences(of: "<|im_start|>assistant", with: "")
    .replacingOccurrences(of: "<|im_end|>", with: "")
    .replacingOccurrences(of: "<|start|>assistant<|channel|>final<|message|>", with: "")
    .replacingOccurrences(of: "<|start|>assistant<|channel|>analysis<|message|>", with: "")
    .replacingOccurrences(of: "<|start|>assistant", with: "")
    .replacingOccurrences(of: "<|channel|>final<|message|>", with: "")
    .replacingOccurrences(of: "<|channel|>analysis<|message|>", with: "")
    .replacingOccurrences(of: "analysis<|message|>", with: "")
    .replacingOccurrences(of: "<|message|>", with: "")
    .replacingOccurrences(of: "<|end|>", with: "")
    .replacingOccurrences(of: "<|turn>model", with: "")
    .replacingOccurrences(of: "<|turn>user", with: "")
    .replacingOccurrences(of: "<|turn>", with: "")
    .replacingOccurrences(of: "<turn|>", with: "")
    .replacingOccurrences(of: "<start_of_turn>model", with: "")
    .replacingOccurrences(of: "<start_of_turn>user", with: "")
    .replacingOccurrences(of: "<end_of_turn>", with: "")
    .trimmingCharacters(in: .whitespacesAndNewlines)
}

private extension View {
    func liquidGlassPrimaryButton(cornerRadius: CGFloat = 12) -> some View {
        let shape = RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)

        return self
            .foregroundStyle(.white)
            .background(shape.fill(.ultraThinMaterial))
            .clipShape(shape)
            .contentShape(shape)
            .overlay(
                shape
                    .fill(
                        LinearGradient(
                            colors: [ApolloPalette.accent.opacity(0.16), Color.white.opacity(0.03)],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        )
                    )
                    .allowsHitTesting(false)
            )
            .overlay(
                shape
                    .stroke(Color.white.opacity(0.22), lineWidth: 1)
                    .allowsHitTesting(false)
            )
    }

    func featureActionIconButtonStyle(cornerRadius: CGFloat = 10) -> some View {
        self
            .foregroundStyle(.white)
            .background(
                RoundedRectangle(cornerRadius: cornerRadius)
                    .fill(Color.white.opacity(0.08))
            )
            .overlay(
                RoundedRectangle(cornerRadius: cornerRadius)
                    .stroke(Color.white.opacity(0.16), lineWidth: 1)
            )
    }
}

private struct FeatureModelSettingsSheet: View {
    @EnvironmentObject var settings: AppSettings
    @Binding var selectedModelName: String
    @Binding var maxTokens: Double
    @Binding var enableThinking: Bool
    @Binding var enableVision: Bool
    let enableAudio: Binding<Bool>?
    @Binding var isLoading: Bool
    @Binding var errorMessage: String?
    let supportsVisionToggle: Bool
    let visionToggleTitleKey: String
    let audioToggleTitleKey: String?
    let visionAvailableCheck: ((AIModel) -> Bool)?
    let writingMode: Binding<WritingAidMode>?
    let modelFilter: ((AIModel) -> Bool)?
    let onLoad: () async -> Void
    let onUnload: () -> Void

    @Environment(\.dismiss) private var dismiss
    @ObservedObject private var llm = LLMBackend.shared
    @State private var models: [AIModel] = []
    @State private var isRefreshingModels = false

    private var selectedModel: AIModel? {
        models.first(where: { $0.name == selectedModelName })
            ?? selectedFeatureModel(named: selectedModelName)
    }

    private var selectedModelSupportsVision: Bool {
        guard supportsVisionToggle, let model = selectedModel, model.supportsVision else { return false }
        return visionAvailableCheck?(model) ?? true
    }

    private var selectedModelSupportsAudio: Bool {
        guard let model = selectedModel else { return false }
        return model.supportsAudio
    }

    private var maxContextCap: Double {
        let advertised = selectedModel?.contextWindowSize ?? 4096
        return Double(max(1, advertised))
    }

    private var isSelectedModelLoaded: Bool {
        llm.isLoaded && llm.currentlyLoadedModel == selectedModelName
    }

    var body: some View {
        NavigationView {
            ZStack {
                ApolloLiquidBackground()

                ScrollView {
                    VStack(spacing: 16) {
                        VStack(alignment: .leading, spacing: 8) {
                            Text(settings.localized("select_model"))
                                .font(.headline)
                                .foregroundColor(.white)

                            if isRefreshingModels {
                                HStack(spacing: 8) {
                                    ProgressView()
                                    Text(settings.localized("loading"))
                                        .foregroundStyle(.white.opacity(0.7))
                                }
                            } else {
                                Picker("", selection: $selectedModelName) {
                                    ForEach(models, id: \.id) { model in
                                        Text(model.name).tag(model.name)
                                    }
                                }
                                .pickerStyle(.menu)
                                .tint(ApolloPalette.accentStrong)
                            }

                            HStack {
                                Text(settings.localized("context_window_size"))
                                    .foregroundColor(.white)
                                Spacer()
                                Text("\(Int(maxTokens))")
                                    .foregroundColor(.white.opacity(0.9))
                                    .monospacedDigit()
                            }
                            Slider(value: $maxTokens, in: 1...maxContextCap, step: 1) { editing in
                                if !editing {
                                    maxTokens = min(max(1, maxTokens), maxContextCap)
                                }
                            }
                            .tint(ApolloPalette.accentStrong)

                            if selectedModelSupportsVision {
                                Toggle(settings.localized(visionToggleTitleKey), isOn: $enableVision)
                                    .tint(ApolloPalette.accentStrong)
                                    .foregroundColor(.white)
                            }

                            if let enableAudio, selectedModelSupportsAudio {
                                Toggle(settings.localized(audioToggleTitleKey ?? "enable_audio"), isOn: enableAudio)
                                    .tint(ApolloPalette.accentStrong)
                                    .foregroundColor(.white)
                            }

                            if let writingMode {
                                VStack(alignment: .leading, spacing: 6) {
                                    Text(settings.localized("writing_aid_select_mode"))
                                        .font(.subheadline.weight(.semibold))
                                        .foregroundColor(.white.opacity(0.92))

                                    Picker("", selection: writingMode) {
                                        ForEach(WritingAidMode.allCases, id: \.rawValue) { mode in
                                            Text(settings.localized(mode.rawValue)).tag(mode)
                                        }
                                    }
                                    .pickerStyle(.menu)
                                    .tint(ApolloPalette.accentStrong)
                                }
                            }
                        }
                        .padding()
                        .background(.ultraThinMaterial)
                        .clipShape(RoundedRectangle(cornerRadius: 14))
                        .overlay(
                            RoundedRectangle(cornerRadius: 14)
                                .stroke(Color.white.opacity(0.14), lineWidth: 1)
                        )

                        VStack(spacing: 10) {
                            Button {
                                Task { await onLoad() }
                            } label: {
                                HStack {
                                    Spacer()
                                    if isLoading {
                                        ProgressView()
                                    } else {
                                        Text(settings.localized("load_model"))
                                    }
                                    Spacer()
                                }
                                .padding(.vertical, 12)
                                .contentShape(Rectangle())
                            }
                            .liquidGlassPrimaryButton(cornerRadius: 12)
                            .tint(ApolloPalette.accentStrong)
                            .disabled(isLoading || selectedModelName.isEmpty || isRefreshingModels)

                            if isSelectedModelLoaded {
                                Button(role: .destructive) {
                                    onUnload()
                                } label: {
                                    HStack {
                                        Spacer()
                                        Text(settings.localized("unload_model"))
                                        Spacer()
                                    }
                                    .padding(.vertical, 12)
                                    .contentShape(Rectangle())
                                }
                                .background(
                                    RoundedRectangle(cornerRadius: 12)
                                        .fill(ApolloPalette.destructive.opacity(0.10))
                                )
                                .overlay(
                                    RoundedRectangle(cornerRadius: 12)
                                        .stroke(ApolloPalette.destructive.opacity(0.9), lineWidth: 1)
                                )
                                .foregroundStyle(ApolloPalette.destructive.opacity(0.98))
                                .disabled(isLoading)
                            }
                        }

                        if let errorMessage, !errorMessage.isEmpty {
                            Text(errorMessage)
                                .foregroundColor(.red.opacity(0.9))
                                .font(.caption)
                        }
                    }
                    .padding()
                }
            }
            .navigationTitle(settings.localized("feature_settings_title"))
            .navigationBarTitleDisplayMode(.inline)
            .toolbarBackground(.hidden, for: .navigationBar)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button(settings.localized("done")) { dismiss() }
                }
            }
            .task {
                await refreshModelsIfNeeded()
                normalizeToggleStatesForSelectedModel()
            }
            .onChange(of: selectedModelName) { _, _ in
                normalizeToggleStatesForSelectedModel()
            }
        }
    }

    private func refreshModelsIfNeeded() async {
        if !models.isEmpty { return }
        isRefreshingModels = true
        try? RunAnywhere.initialize(environment: .development)
        let loaded = downloadableFeatureModels().filter { model in
            modelFilter?(model) ?? true
        }
        models = loaded
        if selectedModelName.isEmpty || !loaded.contains(where: { $0.name == selectedModelName }) {
            selectedModelName = loaded.first?.name ?? ""
        }
        let cap = Double(max(1, selectedModel?.contextWindowSize ?? 4096))
        maxTokens = min(max(1, maxTokens), cap)
        isRefreshingModels = false
    }

    private func normalizeToggleStatesForSelectedModel() {
        enableThinking = false
        if supportsVisionToggle && !selectedModelSupportsVision {
            enableVision = false
        }
    }
}

@available(iOS 17.0, *)
actor SpeechEngine {
    private var audioEngine: AVAudioEngine?
    private let speechRecognizer = SFSpeechRecognizer()
    private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    private var recognitionTask: SFSpeechRecognitionTask?
    
    private var onResult: (@Sendable (SFSpeechRecognitionResult) -> Void)?
    private var onError: (@Sendable (Error) -> Void)?
    private var isStreaming = false

    func start(onResult: @escaping @Sendable (SFSpeechRecognitionResult) -> Void, onError: @escaping @Sendable (Error) -> Void) throws {
        // Clean up any prior session first
        stop()

        self.onResult = onResult
        self.onError = onError
        self.isStreaming = true

        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.record, mode: .measurement, options: .duckOthers)
        try session.setActive(true, options: .notifyOthersOnDeactivation)

        // Create a FRESH AVAudioEngine each time. The old engine's inputNode
        // permanently caches a stale format (0 Hz) after the audio session
        // switches between .record and .playback. No amount of reset() fixes it.
        let engine = AVAudioEngine()
        self.audioEngine = engine

        let inputNode = engine.inputNode
        let recordingFormat = inputNode.outputFormat(forBus: 0)
        guard recordingFormat.sampleRate > 0 && recordingFormat.channelCount > 0 else {
            isStreaming = false
            self.audioEngine = nil
            throw NSError(domain: "SpeechEngine", code: -2,
                          userInfo: [NSLocalizedDescriptionKey: "Audio input format invalid: \(recordingFormat.sampleRate)Hz, \(recordingFormat.channelCount)ch"])
        }

        try startNewTask()

        inputNode.installTap(onBus: 0, bufferSize: 1024, format: recordingFormat) { [weak self] buffer, _ in
            Task { [weak self] in
                guard let self = self else { return }
                await self.append(buffer)
            }
        }
        
        engine.prepare()
        try engine.start()
    }

    private func append(_ buffer: AVAudioPCMBuffer) {
        recognitionRequest?.append(buffer)
    }

    private func startNewTask() throws {
        recognitionTask?.cancel()
        recognitionTask = nil
        
        recognitionRequest = SFSpeechAudioBufferRecognitionRequest()
        guard let recognitionRequest = recognitionRequest else {
            throw NSError(domain: "SpeechEngine", code: -1, userInfo: [NSLocalizedDescriptionKey: "Unable to create request"])
        }
        
        recognitionRequest.shouldReportPartialResults = true
        if speechRecognizer?.supportsOnDeviceRecognition == true {
            recognitionRequest.requiresOnDeviceRecognition = true
        }
        
        recognitionTask = speechRecognizer?.recognitionTask(with: recognitionRequest) { [weak self] result, error in
            if let result = result {
                self?.onResult?(result)
                if result.isFinal {
                    Task { [weak self] in
                        guard let self = self else { return }
                        if await self.isStreaming {
                            try? await self.startNewTask()
                        }
                    }
                }
            }
            if let error = error {
                self?.onError?(error)
            }
        }
    }

    func stop() {
        isStreaming = false
        if let engine = audioEngine {
            engine.stop()
            engine.inputNode.removeTap(onBus: 0)
        }
        audioEngine = nil
        recognitionRequest?.endAudio()
        recognitionTask?.cancel()
        recognitionTask = nil
        recognitionRequest = nil
    }
}

@available(iOS 17.0, *)
struct TranscriptionSession: Identifiable, Codable {
    let id: UUID
    var text: String
    let timestamp: Date
}

@available(iOS 17.0, *)
@MainActor
private final class IOSSpeechTranscriber: NSObject, ObservableObject {
    @Published var transcript: String = ""
    @Published var history: [TranscriptionSession] = []
    @Published var isRecording: Bool = false
    @Published var isTranscribing: Bool = false
    @Published var isPreparing: Bool = false
    @Published var selectedAudioURL: URL?
    
    private var baseTranscript: String = ""
    private var previousHypothesis: String = ""
    private var sessionHistory: [String] = []
    
    private let engine = SpeechEngine()
    private let speechRecognizer = SFSpeechRecognizer()

    private func logError(_ message: String) {
        NSLog("[LLMHub][Transcriber] \(message)")
    }

    func startLiveTranscription() async {
        logError("startLiveTranscription() called")
        cancelTranscription()
        isPreparing = true
        transcript = ""
        baseTranscript = ""
        previousHypothesis = ""
        sessionHistory = []

        Task.detached(priority: .userInitiated) { [weak self] in
            guard let self = self else { return }
            
            await self.logError("startLiveTranscription: checking permissions...")
            let status = await withCheckedContinuation { continuation in
                SFSpeechRecognizer.requestAuthorization { status in
                    continuation.resume(returning: status)
                }
            }
            let micAuthorized = await AVAudioApplication.requestRecordPermission()
            
            guard status == .authorized && micAuthorized else {
                await self.logError("Permissions denied: speech=\(status.rawValue), mic=\(micAuthorized)")
                await MainActor.run { self.isPreparing = false }
                return
            }

            do {
                await self.logError("startLiveTranscription: starting engine...")
                try await self.engine.start(onResult: { result in
                    let text = result.bestTranscription.formattedString
                    let isFinal = result.isFinal
                    let prefix = "Live"
                    
                    Task { @MainActor in
                        self.processTranscriptionResult(text: text, isFinal: isFinal, prefix: prefix)
                    }
                }, onError: { error in
                    Task { @MainActor in
                        self.logError("Engine error: \(error.localizedDescription)")
                    }
                })
                
                await MainActor.run {
                    self.isRecording = true
                    self.isPreparing = false
                }
                await self.logError("Live recording successfully started")
            } catch {
                await self.logError("Audio engine failed: \(error.localizedDescription)")
                await MainActor.run {
                    self.isPreparing = false
                    self.isRecording = false
                }
            }
        }
    }

    func stopLiveTranscription() async {
        logError("stopLiveTranscription: stopping engine...")
        let finalTranscript = self.transcript.trimmingCharacters(in: .whitespacesAndNewlines)
        await engine.stop()
        await MainActor.run {
            self.finishCurrentTranscription(with: finalTranscript)
            self.isRecording = false
            self.isPreparing = false
        }
    }

    func transcribeSelectedAudio() async {
        guard let sourceURL = selectedAudioURL else { return }
        
        let accessing = sourceURL.startAccessingSecurityScopedResource()
        defer { if accessing { sourceURL.stopAccessingSecurityScopedResource() } }
        
        logError("Transcribing selected audio: \(sourceURL.path)...")
        
        // Copy file to temp location to avoid sandbox/permission issues with SFSpeechURLRecognitionRequest
        let tempDir = FileManager.default.temporaryDirectory
        let tempFileName = "transcribe_temp_\(UUID().uuidString).\(sourceURL.pathExtension)"
        let tempURL = tempDir.appendingPathComponent(tempFileName)
        
        do {
            if FileManager.default.fileExists(atPath: tempURL.path) {
                try? FileManager.default.removeItem(at: tempURL)
            }
            try FileManager.default.copyItem(at: sourceURL, to: tempURL)
        } catch {
            logError("Failed to copy audio for transcription: \(error.localizedDescription)")
            return
        }
        
        isTranscribing = true
        transcript = ""
        
        let request = SFSpeechURLRecognitionRequest(url: tempURL)
        request.shouldReportPartialResults = true
        if speechRecognizer?.supportsOnDeviceRecognition == true {
            request.requiresOnDeviceRecognition = true
        }
        
        speechRecognizer?.recognitionTask(with: request) { [weak self] result, error in
            guard let self = self else { return }
            
            if let result = result {
                let text = result.bestTranscription.formattedString
                let isFinal = result.isFinal
                let prefix = "File"
                
                self.logError("[\(prefix)] Result isFinal=\(isFinal), text.count=\(text.count)")
                
                Task { @MainActor in
                    self.processTranscriptionResult(text: text, isFinal: isFinal, prefix: prefix)
                    
                    if isFinal {
                        self.logError("[File] Transcription finished. Moving to history.")
                        self.finishCurrentTranscription(with: self.transcript)
                        self.isTranscribing = false
                        try? FileManager.default.removeItem(at: tempURL)
                    }
                }
            }
            
            if let error = error {
                self.logError("[File] transcription error: \(error.localizedDescription)")
                Task { @MainActor in
                    self.isTranscribing = false
                    try? FileManager.default.removeItem(at: tempURL)
                }
            }
        }
    }

    func cancelTranscription() {
        logError("cancelTranscription: cleaning up...")
        Task {
            await engine.stop()
        }
        isRecording = false
        isTranscribing = false
        isPreparing = false
    }

    func cleanup() {
        cancelTranscription()
    }

    func setSelectedAudioURL(_ url: URL) {
        cancelTranscription()
        selectedAudioURL = url
        transcript = ""
    }

    func clearSelectedAudio() {
        selectedAudioURL = nil
    }

    private func processTranscriptionResult(text: String, isFinal: Bool, prefix: String) {
        if isFinal {
            // Some on-device recognizers send an empty string on final result 
            // after sending the full text in the last non-final result.
            let finalToCommit = text.isEmpty ? self.previousHypothesis : text
            
            if !finalToCommit.isEmpty {
                self.logError("[\(prefix)] committing final segment: \"\(finalToCommit.prefix(50))...\"")
                self.addTextToHistory(finalToCommit)
            }
            self.previousHypothesis = ""
        } else {
            // Detect if the recognizer reset its buffer (on-device behavior)
            if !text.isEmpty && !self.previousHypothesis.isEmpty && 
               text.count < self.previousHypothesis.count / 2 && 
               !self.previousHypothesis.lowercased().hasPrefix(text.lowercased()) {
                
                self.logError("[\(prefix)] reset detected! Committing hypothesis: \"\(self.previousHypothesis.prefix(50))...\"")
                self.addTextToHistory(self.previousHypothesis)
            }
            self.previousHypothesis = text
        }
        
        let combined = (self.sessionHistory + [self.previousHypothesis])
            .filter { !$0.isEmpty }
            .joined(separator: " ")
        self.transcript = combined.trimmingCharacters(in: .whitespacesAndNewlines)
        self.logError("[\(prefix)] current transcript length: \(self.transcript.count)")
    }

    private func finishCurrentTranscription(with text: String) {
        let finalTranscript = text.trimmingCharacters(in: .whitespacesAndNewlines)

        transcript = ""
        baseTranscript = ""
        previousHypothesis = ""
        sessionHistory = []

        guard !finalTranscript.isEmpty else { return }
        history.append(TranscriptionSession(id: UUID(), text: finalTranscript, timestamp: Date()))
    }

    private func addTextToHistory(_ text: String) {
        let clean = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if clean.isEmpty { return }
        
        let newLower = clean.lowercased()
        
        if let lastIndex = sessionHistory.indices.last {
            let lastLower = sessionHistory[lastIndex].lowercased()
            
            // If the last thing we added is contained in the new text, update it 
            // (on-device often expands its hypothesis)
            if newLower.contains(lastLower) {
                sessionHistory[lastIndex] = clean
                return
            }
            
            // If new text is already in last added segment, skip
            if lastLower.contains(newLower) {
                return
            }
        }
        
        sessionHistory.append(clean)
    }
}


@available(iOS 17.0, *)
private struct IOS26TranscriberScreen: View {
    @EnvironmentObject var settings: AppSettings
    @StateObject private var transcriber = IOSSpeechTranscriber()
    @StateObject private var audioRecorder = AudioRecorder()
    @ObservedObject private var ttsManager = OnDeviceTtsManager.shared
    @ObservedObject private var llm = LLMBackend.shared
    @State private var showAudioImporter = false
    @State private var showSettings = false
    @State private var audioTranscript: String = ""
    @State private var audioHistory: [TranscriptionSession] = []
    @State private var isAudioTranscribing = false
    @State private var selectedAudioURL: URL?
    @State private var audioTranscriptionTask: Task<Void, Never>?
    @AppStorage("feature_transcriber_model_name") private var selectedModelName: String = ""
    @AppStorage("feature_transcriber_max_tokens") private var maxTokens: Double = 512
    @State private var isModelLoading = false
    @State private var modelLoadError: String? = nil

    let onNavigateBack: () -> Void

    private var canStartRecording: Bool {
        !transcriber.isRecording && !transcriber.isTranscribing && !transcriber.isPreparing
    }

    private var canUploadAudio: Bool {
        !transcriber.isRecording && !transcriber.isTranscribing && !transcriber.isPreparing
    }

    private var canTranscribeUploadedAudio: Bool {
        transcriber.selectedAudioURL != nil && !transcriber.isRecording && !transcriber.isTranscribing && !transcriber.isPreparing
    }

    private var selectedModel: AIModel? {
        selectedFeatureModel(named: selectedModelName)
    }

    /// True only when the user has loaded a Gemma 4 LiteRT-LM model.
    /// Defaults to false (system transcriber). No toggle needed — loading the model auto-enables.
    private var useModelAudioInput: Bool {
        guard let model = selectedModel else { return false }
        // For the transcriber screen, don't auto-switch until the model is completely loaded,
        // so the user can keep using the system transcriber while waiting.
        return model.isGemma4LiteRTLM && llm.isLoaded && llm.currentlyLoadedModel == model.name
    }

    var body: some View {
        Group {
            if useModelAudioInput {
                gemmaAudioTranscriberView
            } else {
                VStack(spacing: 12) {
            VStack(spacing: 16) {
                Button {
                    if transcriber.isRecording {
                        Task { await transcriber.stopLiveTranscription() }
                    } else if canStartRecording {
                        Task { @MainActor in
                            await transcriber.startLiveTranscription()
                        }
                    }
                } label: {
                    ZStack {
                        Circle()
                            .fill(.ultraThinMaterial)
                            .frame(width: 124, height: 124)
                            .overlay(
                                Circle()
                                    .stroke(Color.white.opacity(0.2), lineWidth: 1)
                            )

                        if transcriber.isPreparing {
                            ProgressView()
                                .tint(.white)
                                .scaleEffect(1.2)
                        } else {
                            Image(systemName: transcriber.isRecording ? "stop.fill" : "mic.fill")
                                .font(.system(size: 42, weight: .bold))
                                .foregroundStyle(transcriber.isRecording ? .red : .white)
                        }
                    }
                }
                .contentShape(Circle())
                .buttonStyle(.plain)
                .disabled(!transcriber.isRecording && !canStartRecording)

                Text(
                    transcriber.isPreparing
                        ? settings.localized("processing")
                        : transcriber.isRecording
                        ? settings.localized("transcriber_recording")
                        : settings.localized("transcriber_record")
                )
                .font(.headline)

                Button {
                    showAudioImporter = true
                } label: {
                    HStack {
                        Image(systemName: "waveform.badge.plus")
                        Text(settings.localized("transcriber_upload"))
                    }
                    .frame(maxWidth: .infinity)
                    .frame(height: 48)
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .foregroundStyle(.white)
                .liquidGlassPrimaryButton(cornerRadius: 12)
                .disabled(!canUploadAudio)

                if let selectedAudioURL = transcriber.selectedAudioURL {
                    HStack(spacing: 8) {
                        Image(systemName: "waveform")
                            .foregroundStyle(.white.opacity(0.85))
                        Text(selectedAudioURL.lastPathComponent)
                            .font(.subheadline)
                            .lineLimit(1)
                            .truncationMode(.middle)
                            .frame(maxWidth: .infinity, alignment: .leading)
                        Button(role: .destructive) {
                            transcriber.clearSelectedAudio()
                        } label: {
                            Image(systemName: "trash")
                        }
                        .buttonStyle(.plain)
                    }
                    .padding(10)
                    .background(.ultraThinMaterial)
                    .clipShape(RoundedRectangle(cornerRadius: 10))
                }
            }
            .padding()
            .background(.ultraThinMaterial)
            .clipShape(RoundedRectangle(cornerRadius: 16))
            .overlay(
                RoundedRectangle(cornerRadius: 16)
                    .stroke(Color.white.opacity(0.14), lineWidth: 1)
            )
            .padding(.horizontal)

            ScrollViewReader { proxy in
                ScrollView {
                    VStack(alignment: .leading, spacing: 16) {
                        ForEach(transcriber.history) { session in
                            transcriptionBoxView(for: session.text)
                                .id(session.id)
                        }

                        if !transcriber.transcript.isEmpty || transcriber.isRecording || transcriber.isPreparing || transcriber.isTranscribing {
                            transcriptionBoxView(for: transcriber.transcript)
                                .id("current_box")
                        }
                    }
                    .padding(.horizontal)
                }
                .onChange(of: transcriber.transcript) { _, _ in
                    withAnimation {
                        proxy.scrollTo("current_box", anchor: .bottom)
                    }
                }
                .onChange(of: transcriber.history.count) { _, _ in
                    if let lastId = transcriber.history.last?.id {
                        withAnimation {
                            proxy.scrollTo(lastId, anchor: .bottom)
                        }
                    }
                }
            }

            Spacer(minLength: 0)

            Button {
                if transcriber.isRecording {
                    Task { await transcriber.stopLiveTranscription() }
                } else if transcriber.isTranscribing {
                    transcriber.cancelTranscription()
                } else if canTranscribeUploadedAudio {
                    Task { @MainActor in
                        await transcriber.transcribeSelectedAudio()
                    }
                }
            } label: {
                HStack(spacing: 8) {
                    if transcriber.isPreparing || transcriber.isTranscribing {
                        ProgressView()
                            .tint(.white)
                            .scaleEffect(0.85)
                    } else {
                        Image(systemName: transcriber.isRecording ? "stop.fill" : "waveform")
                            .font(.system(size: 12, weight: .bold))
                    }
                    Text(buttonTitle)
                        .lineLimit(1)
                }
                .frame(maxWidth: .infinity)
                .frame(height: 52)
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .foregroundStyle(.white)
            .liquidGlassPrimaryButton(cornerRadius: 12)
            .disabled(isBottomButtonDisabled)
            .padding(.horizontal)
            .padding(.bottom, 8)
                }
            }
        }
        .navigationTitle(settings.localized("transcriber_title"))
        .navigationBarTitleDisplayMode(.inline)
        .apolloScreenBackground()
        .safeAreaInset(edge: .bottom, spacing: 0) { BannerAdContainer() }
        .toolbarBackground(.hidden, for: .navigationBar)
        .toolbar {
            ToolbarItem(placement: .navigationBarLeading) {
                Button {
                    transcriber.cleanup()
                    onNavigateBack()
                } label: {
                    Image(systemName: "arrow.left")
                }
            }
            ToolbarItem(placement: .navigationBarTrailing) {
                Button { showSettings = true } label: {
                    Image(systemName: "slider.horizontal.3")
                }
            }
        }
        .sheet(isPresented: $showSettings) {
            FeatureModelSettingsSheet(
                selectedModelName: $selectedModelName,
                maxTokens: $maxTokens,
                enableThinking: .constant(false),
                enableVision: .constant(false),
                enableAudio: nil,
                isLoading: $isModelLoading,
                errorMessage: $modelLoadError,
                supportsVisionToggle: false,
                visionToggleTitleKey: "scam_detector_enable_vision",
                audioToggleTitleKey: nil,
                visionAvailableCheck: nil,
                writingMode: nil,
                modelFilter: { $0.modelFormat == .litertlm && $0.supportsAudio },
                onLoad: { 
                    isModelLoading = true
                    modelLoadError = nil
                    llm.isLoaded = false
                    llm.currentlyLoadedModel = nil
                    await ensureAudioModelLoaded(force: true) 
                    isModelLoading = false
                },
                onUnload: { 
                    llm.isLoaded = false
                    llm.currentlyLoadedModel = nil
                    llm.unloadModel() 
                }
            )
        }
        .fileImporter(
            isPresented: $showAudioImporter,
            allowedContentTypes: [.audio],
            allowsMultipleSelection: false
        ) { result in
            switch result {
            case .success(let urls):
                if let first = urls.first {
                    if useModelAudioInput {
                        selectedAudioURL = first
                    } else {
                        transcriber.setSelectedAudioURL(first)
                    }
                }
            case .failure(let error):
                NSLog("[LLMHub][Transcriber] Audio import failed: \(error.localizedDescription)")
            }
        }
        .onDisappear {
            transcriber.cleanup()
            audioRecorder.cancelRecording()
            audioTranscriptionTask?.cancel()
            audioTranscriptionTask = nil
            if llm.isLoaded {
                llm.unloadModel()
                llm.isLoaded = false
                llm.currentlyLoadedModel = nil
            }
        }
        .onAppear {
            selectedModelName = ""
            Task {
                await syncRunAnywhereModelDiscovery()
                // Default to empty = system transcriber.
                // User picks a Gemma4 LiteRT-LM model from the config sheet to use AI transcription.
                // Don't auto-select an audio model on appear.
            }
        }
    }

    @ViewBuilder
    private func transcriptionBoxView(for text: String) -> some View {
        let speechKey = "transcriber-\(text)"
        VStack(alignment: .leading, spacing: 8) {
            let isLive = transcriber.isRecording || audioRecorder.isRecording
            Text(text.isEmpty ? (isLive ? "..." : "-") : text)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
                .frame(minHeight: 120, alignment: .topLeading)
                .padding(12)
                .background(.ultraThinMaterial)
                .clipShape(RoundedRectangle(cornerRadius: 12))
                .overlay(
                    RoundedRectangle(cornerRadius: 12)
                        .stroke(Color.white.opacity(0.1), lineWidth: 1)
                )

            if !text.isEmpty {
                HStack {
                    Spacer()
                    Button {
                        ttsManager.toggleSpeaking(
                            text,
                            fallbackLanguage: settings.selectedLanguage,
                            key: speechKey
                        )
                    } label: {
                        Image(systemName: ttsManager.isSpeaking(key: speechKey) ? "stop.fill" : "speaker.wave.2")
                            .font(.system(size: 16, weight: .semibold))
                            .frame(width: 40, height: 40)
                    }
                    .featureActionIconButtonStyle()

                    Button {
                        #if canImport(UIKit)
                        UIPasteboard.general.string = text
                        #endif
                    } label: {
                        Image(systemName: "doc.on.doc")
                            .font(.system(size: 16, weight: .semibold))
                            .frame(width: 40, height: 40)
                    }
                    .featureActionIconButtonStyle()
                }
            }
        }
    }

    private var buttonTitle: String {
        if transcriber.isRecording {
            return settings.localized("transcriber_stop")
        }
        if transcriber.isPreparing {
            return settings.localized("processing")
        }
        if transcriber.isTranscribing {
            return settings.localized("transcribing_tap_to_cancel")
        }
        if transcriber.selectedAudioURL != nil {
            return settings.localized("transcriber_transcribe")
        }
        return settings.localized("transcriber_record")
    }

    private var isBottomButtonDisabled: Bool {
        if transcriber.isRecording || transcriber.isTranscribing || transcriber.isPreparing {
            return false
        }
        return transcriber.selectedAudioURL == nil
    }

    private var gemmaAudioTranscriberView: some View {
        VStack(spacing: 12) {
            VStack(spacing: 16) {
                Button {
                    if audioRecorder.isRecording {
                        _ = audioRecorder.stopRecording()
                    } else {
                        Task { @MainActor in
                            let isGemma4 = useModelAudioInput
                            let ext = isGemma4 ? "wav" : "m4a"
                            let destination = persistentAudioStorageDirectory()
                                .appendingPathComponent("transcriber_audio_\(UUID().uuidString)")
                                .appendingPathExtension(ext)
                            _ = await audioRecorder.startRecording(
                                outputURL: destination,
                                autoStopAfterSilence: false,
                                isFloat32Wav: isGemma4
                            ) { url in
                                Task { @MainActor in
                                    selectedAudioURL = url
                                }
                                Task { await transcribeAudio(url) }
                            }
                        }
                    }
                } label: {
                    ZStack {
                        Circle()
                            .fill(.ultraThinMaterial)
                            .frame(width: 124, height: 124)
                            .overlay(
                                Circle()
                                    .stroke(Color.white.opacity(0.2), lineWidth: 1)
                            )

                        if audioRecorder.isPreparing {
                            ProgressView()
                                .tint(.white)
                                .scaleEffect(1.2)
                        } else {
                            Image(systemName: audioRecorder.isRecording ? "stop.fill" : "mic.fill")
                                .font(.system(size: 42, weight: .bold))
                                .foregroundStyle(audioRecorder.isRecording ? .red : .white)
                        }
                    }
                }
                .contentShape(Circle())
                .buttonStyle(.plain)
                .disabled(audioRecorder.isPreparing || isAudioTranscribing)

                Text(
                    audioRecorder.isPreparing
                        ? settings.localized("processing")
                        : audioRecorder.isRecording
                        ? settings.localized("transcriber_recording")
                        : settings.localized("transcriber_record")
                )
                .font(.headline)

                Button {
                    showAudioImporter = true
                } label: {
                    HStack {
                        Image(systemName: "waveform.badge.plus")
                        Text(settings.localized("transcriber_upload"))
                    }
                    .frame(maxWidth: .infinity)
                    .frame(height: 48)
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .foregroundStyle(.white)
                .liquidGlassPrimaryButton(cornerRadius: 12)
                .disabled(audioRecorder.isRecording || audioRecorder.isPreparing || isAudioTranscribing)

                if let selectedAudioURL {
                    HStack(spacing: 10) {
                        AudioPlaybackButton(url: selectedAudioURL)
                        Text(selectedAudioURL.lastPathComponent)
                            .font(.subheadline)
                            .lineLimit(1)
                            .truncationMode(.middle)
                            .frame(maxWidth: .infinity, alignment: .leading)
                        Button(role: .destructive) {
                            self.selectedAudioURL = nil
                        } label: {
                            Image(systemName: "trash")
                        }
                        .buttonStyle(.plain)
                    }
                    .padding(10)
                    .background(.ultraThinMaterial)
                    .clipShape(RoundedRectangle(cornerRadius: 10))
                }
            }
            .padding()
            .background(.ultraThinMaterial)
            .clipShape(RoundedRectangle(cornerRadius: 16))
            .overlay(
                RoundedRectangle(cornerRadius: 16)
                    .stroke(Color.white.opacity(0.14), lineWidth: 1)
            )
            .padding(.horizontal)

            ScrollViewReader { proxy in
                ScrollView {
                    VStack(alignment: .leading, spacing: 16) {
                        ForEach(audioHistory) { session in
                            transcriptionBoxView(for: session.text)
                                .id(session.id)
                        }

                        if !audioTranscript.isEmpty || audioRecorder.isRecording || audioRecorder.isPreparing || isAudioTranscribing {
                            transcriptionBoxView(for: audioTranscript)
                                .id("current_box")
                        }
                    }
                    .padding(.horizontal)
                }
                .onChange(of: audioTranscript) { _, _ in
                    withAnimation {
                        proxy.scrollTo("current_box", anchor: .bottom)
                    }
                }
                .onChange(of: audioHistory.count) { _, _ in
                    if let lastId = audioHistory.last?.id {
                        withAnimation {
                            proxy.scrollTo(lastId, anchor: .bottom)
                        }
                    }
                }
            }

            Spacer(minLength: 0)

            Button {
                if audioRecorder.isRecording {
                    if let url = audioRecorder.stopRecording() {
                        selectedAudioURL = url
                        Task { await transcribeAudio(url) }
                    }
                } else if isAudioTranscribing {
                    audioTranscriptionTask?.cancel()
                    audioTranscriptionTask = nil
                    isAudioTranscribing = false
                } else if let selectedAudioURL {
                    Task { await transcribeAudio(selectedAudioURL) }
                }
            } label: {
                HStack(spacing: 8) {
                    if audioRecorder.isPreparing || isAudioTranscribing {
                        ProgressView()
                            .tint(.white)
                            .scaleEffect(0.85)
                    } else {
                        Image(systemName: audioRecorder.isRecording ? "stop.fill" : "waveform")
                            .font(.system(size: 12, weight: .bold))
                    }
                    Text(gemmaButtonTitle)
                        .lineLimit(1)
                }
                .frame(maxWidth: .infinity)
                .frame(height: 52)
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .foregroundStyle(.white)
            .liquidGlassPrimaryButton(cornerRadius: 12)
            .disabled(gemmaButtonDisabled)
            .padding(.horizontal)
            .padding(.bottom, 8)
        }
    }

    private var gemmaButtonTitle: String {
        if audioRecorder.isRecording {
            return settings.localized("transcriber_stop")
        }
        if audioRecorder.isPreparing {
            return settings.localized("processing")
        }
        if isAudioTranscribing {
            return settings.localized("transcribing_tap_to_cancel")
        }
        if selectedAudioURL != nil {
            return settings.localized("transcriber_transcribe")
        }
        return settings.localized("transcriber_record")
    }

    private var gemmaButtonDisabled: Bool {
        if audioRecorder.isRecording || isAudioTranscribing || audioRecorder.isPreparing {
            return false
        }
        return selectedAudioURL == nil
    }

    private func transcribeAudio(_ url: URL) async {
        audioTranscriptionTask?.cancel()
        audioTranscriptionTask = nil

        await ensureAudioModelLoaded(force: false)
        guard llm.isLoaded else { return }

        isAudioTranscribing = true
        audioTranscript = ""

        audioTranscriptionTask = Task {
            var latest = ""
            do {
                try await llm.generate(
                    prompt: "Transcribe this audio.",
                    audioURL: url,
                    maxTokensOverride: 512
                ) { text, _, _ in
                    Task { @MainActor in
                        latest = sanitizeModelOutputText(text)
                        audioTranscript = latest
                    }
                }
            } catch is CancellationError {
                // User cancelled.
            } catch {
                NSLog("[LLMHub][Transcriber] Audio transcription failed: \(error.localizedDescription)")
            }

            await MainActor.run {
                let final = latest.trimmingCharacters(in: .whitespacesAndNewlines)
                if !final.isEmpty {
                    audioHistory.append(TranscriptionSession(id: UUID(), text: final, timestamp: Date()))
                }
                audioTranscript = ""
                isAudioTranscribing = false
                audioTranscriptionTask = nil
            }
        }
    }

    private func ensureAudioModelLoaded(force: Bool) async {
        guard let model = selectedModel else { return }
        let modelContextCap = model.contextWindowSize > 0 ? model.contextWindowSize : 4096
        let effectiveContext = min(max(1, Int(maxTokens)), modelContextCap)
        let shouldReload = force
            || llm.currentlyLoadedModel != model.name
            || llm.loadedContextWindow != effectiveContext

        llm.maxTokens = min(Int(maxTokens), effectiveContext)
        llm.contextWindow = effectiveContext
        llm.enableVision = false
        // Auto-enable audio when a Gemma4 LiteRT-LM model is selected; no toggle needed.
        llm.enableAudio = model.isGemma4LiteRTLM
        llm.enableThinking = false

        if shouldReload {
            do {
                try await llm.loadModel(model)
            } catch {
                modelLoadError = error.localizedDescription
            }
        }
    }
}

struct TranscriberScreen: View {
    @EnvironmentObject var settings: AppSettings
    let onNavigateBack: () -> Void

    var body: some View {
        Group {
            if #available(iOS 17.0, *) {
                IOS26TranscriberScreen(onNavigateBack: onNavigateBack)
            } else {
                VStack(spacing: 12) {
                    Image(systemName: "mic.slash")
                        .font(.system(size: 48, weight: .semibold))
                        .foregroundStyle(.secondary)
                    Text(settings.localized("transcriber_title"))
                        .font(.title3.weight(.bold))
                    Text("Live on-device transcriber requires iOS 17 or newer.")
                        .font(.subheadline)
                        .foregroundStyle(.white.opacity(0.7))
                        .multilineTextAlignment(.center)
                        .padding(.horizontal)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .navigationTitle(settings.localized("transcriber_title"))
                .navigationBarTitleDisplayMode(.inline)
                .apolloScreenBackground()
                .safeAreaInset(edge: .bottom, spacing: 0) { BannerAdContainer() }
                .toolbarBackground(.hidden, for: .navigationBar)
                .toolbar {
                    ToolbarItem(placement: .navigationBarLeading) {
                        Button {
                            onNavigateBack()
                        } label: {
                            Image(systemName: "arrow.left")
                        }
                    }
                }
            }
        }
    }
}

// MARK: - VibeVoice

@available(iOS 17.0, *)
@MainActor
private final class IOSVibeVoiceTranscriber: NSObject, ObservableObject {
    @Published var transcript: String = ""
    @Published var isRecording: Bool = false
    @Published var isPreparing: Bool = false

    private var previousHypothesis: String = ""
    private var sessionHistory: [String] = []
    private var silenceTask: Task<Void, Never>?
    private var utteranceHandler: ((String) -> Void)?

    private let engine = SpeechEngine()

    private func logError(_ message: String) {
        NSLog("[LLMHub][VibeVoice] \(message)")
    }

    func startListening(onUtterance: @escaping (String) -> Void) async {
        logError("startListening() called")
        cancelListening()
        isPreparing = true
        transcript = ""
        previousHypothesis = ""
        sessionHistory = []
        utteranceHandler = onUtterance

        Task.detached(priority: .userInitiated) { [weak self] in
            guard let self = self else { return }

            let status = await withCheckedContinuation { continuation in
                SFSpeechRecognizer.requestAuthorization { status in
                    continuation.resume(returning: status)
                }
            }
            let micAuthorized = await AVAudioApplication.requestRecordPermission()

            guard status == .authorized && micAuthorized else {
                await self.logError("Permissions denied: speech=\(status.rawValue), mic=\(micAuthorized)")
                await MainActor.run {
                    self.isPreparing = false
                    self.isRecording = false
                }
                return
            }

            do {
                try await self.engine.start(onResult: { result in
                    let text = result.bestTranscription.formattedString
                    let isFinal = result.isFinal

                    Task { @MainActor in
                        self.processTranscriptionResult(text: text, isFinal: isFinal, prefix: "Voice")
                        self.rescheduleSilenceDetection()
                    }
                }, onError: { error in
                    Task { @MainActor in
                        let nsError = error as NSError
                        // 301 = cancelled by us, ignore silently
                        guard nsError.code != 301 else { return }
                        self.logError("Engine error: \(error.localizedDescription)")
                        // Reset recording state so the listen cycle can restart
                        self.isRecording = false
                        self.isPreparing = false
                    }
                })

                await MainActor.run {
                    self.isRecording = true
                    self.isPreparing = false
                }
            } catch {
                await self.logError("Audio engine failed: \(error.localizedDescription)")
                await MainActor.run {
                    self.isPreparing = false
                    self.isRecording = false
                }
            }
        }
    }

    func stopListening() async {
        silenceTask?.cancel()
        silenceTask = nil
        utteranceHandler = nil
        await engine.stop()
        transcript = ""
        previousHypothesis = ""
        sessionHistory = []
        isRecording = false
        isPreparing = false
    }

    func cancelListening() {
        silenceTask?.cancel()
        silenceTask = nil
        utteranceHandler = nil
        Task {
            await engine.stop()
        }
        transcript = ""
        previousHypothesis = ""
        sessionHistory = []
        isRecording = false
        isPreparing = false
    }

    private func rescheduleSilenceDetection() {
        guard isRecording else { return }
        silenceTask?.cancel()
        silenceTask = Task { [weak self] in
            do {
                try await Task.sleep(nanoseconds: 1_800_000_000)
                guard !Task.isCancelled else { return }
                await self?.finishCurrentUtterance()
            } catch {}
        }
    }

    private func finishCurrentUtterance() async {
        let finalTranscript = transcript.trimmingCharacters(in: .whitespacesAndNewlines)
        silenceTask?.cancel()
        silenceTask = nil
        await engine.stop()
        isRecording = false
        isPreparing = false
        transcript = ""
        previousHypothesis = ""
        sessionHistory = []

        guard !finalTranscript.isEmpty else { return }
        let handler = utteranceHandler
        utteranceHandler = nil
        handler?(finalTranscript)
    }

    private func processTranscriptionResult(text: String, isFinal: Bool, prefix: String) {
        if isFinal {
            let finalToCommit = text.isEmpty ? self.previousHypothesis : text

            if !finalToCommit.isEmpty {
                self.logError("[\(prefix)] committing final segment: \"\(finalToCommit.prefix(50))...\"")
                self.addTextToHistory(finalToCommit)
            }
            self.previousHypothesis = ""
        } else {
            if !text.isEmpty && !self.previousHypothesis.isEmpty &&
               text.count < self.previousHypothesis.count / 2 &&
               !self.previousHypothesis.lowercased().hasPrefix(text.lowercased()) {

                self.logError("[\(prefix)] reset detected! Committing hypothesis: \"\(self.previousHypothesis.prefix(50))...\"")
                self.addTextToHistory(self.previousHypothesis)
            }
            self.previousHypothesis = text
        }

        let combined = (self.sessionHistory + [self.previousHypothesis])
            .filter { !$0.isEmpty }
            .joined(separator: " ")
        self.transcript = combined.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func addTextToHistory(_ text: String) {
        let clean = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if clean.isEmpty { return }

        let newLower = clean.lowercased()

        if let lastIndex = sessionHistory.indices.last {
            let lastLower = sessionHistory[lastIndex].lowercased()

            if newLower.contains(lastLower) {
                sessionHistory[lastIndex] = clean
                return
            }

            if lastLower.contains(newLower) {
                return
            }
        }

        sessionHistory.append(clean)
    }
}

private enum VibeVoiceState: Equatable {
    case idle, listening, responding, speaking
}

@available(iOS 17.0, *)
private struct IOS17VibeVoiceScreen: View {
    @EnvironmentObject var settings: AppSettings
    @ObservedObject private var llm = LLMBackend.shared
    @ObservedObject private var ttsManager = OnDeviceTtsManager.shared
    @AppStorage("feature_vibevoice_model_name") private var selectedModelName: String = ""
    @AppStorage("feature_vibevoice_max_tokens") private var maxTokens: Double = 512
    @AppStorage("feature_vibevoice_enable_model_audio") private var enableModelAudio: Bool = true
    @State private var voiceState: VibeVoiceState = .idle
    @State private var isChatActive = false
    @State private var latestReply = ""
    @State private var isLoading = false
    @State private var errorMessage: String?
    @State private var showSettings = false
    @State private var generationTask: Task<Void, Never>?
    @State private var conversationHistory: [(role: String, content: String)] = []
    @StateObject private var transcriber = IOSVibeVoiceTranscriber()
    @StateObject private var audioRecorder = AudioRecorder()
    @State private var lastRecordedAudioURL: URL?

    private let ttsKey = "vibevoice-reply"

    let onNavigateBack: () -> Void

    private var isCurrentModelLoaded: Bool {
        llm.isLoaded && llm.currentlyLoadedModel == selectedModelName
    }

    private var useModelAudioInput: Bool {
        guard enableModelAudio else { return false }
        guard let model = selectedFeatureModel(named: selectedModelName) else { return false }
        return model.isGemma4LiteRTLM
    }

    var body: some View {
        Group {
            if !isCurrentModelLoaded {
                modelNotLoadedView
            } else {
                chatView
            }
        }
        .navigationTitle(settings.localized("feature_vibevoice"))
        .navigationBarTitleDisplayMode(.inline)
        .apolloScreenBackground()
        .safeAreaInset(edge: .bottom, spacing: 0) { BannerAdContainer() }
        .toolbarBackground(.hidden, for: .navigationBar)
        .toolbar {
            ToolbarItem(placement: .navigationBarLeading) {
                Button {
                    stopAll()
                    llm.unloadModel()
                    onNavigateBack()
                } label: {
                    Image(systemName: "arrow.left")
                }
            }
            ToolbarItem(placement: .navigationBarTrailing) {
                Button { showSettings = true } label: {
                    Image(systemName: "slider.horizontal.3")
                }
            }
        }
        .sheet(isPresented: $showSettings) {
            FeatureModelSettingsSheet(
                selectedModelName: $selectedModelName,
                maxTokens: $maxTokens,
                enableThinking: .constant(false),
                enableVision: .constant(false),
                enableAudio: $enableModelAudio,
                isLoading: $isLoading,
                errorMessage: $errorMessage,
                supportsVisionToggle: false,
                visionToggleTitleKey: "scam_detector_enable_vision",
                audioToggleTitleKey: nil,
                visionAvailableCheck: nil,
                writingMode: nil,
                modelFilter: isNonTranslatorFeatureModel,
                onLoad: { await ensureModelLoaded(force: false) },
                onUnload: { llm.unloadModel() }
            )
        }
        .onAppear {
            Task {
                try? RunAnywhere.initialize(environment: .development)
                let available = downloadableFeatureModels().filter(isNonTranslatorFeatureModel)
                // Only set a default if no model was previously selected.
                // @AppStorage preserves the last used model across launches.
                if selectedModelName.isEmpty {
                    selectedModelName = available.first?.name ?? ""
                }
            }
        }
        .onDisappear {
            stopAll()
            generationTask?.cancel()
            generationTask = nil
            llm.unloadModel()
        }
        .onChange(of: ttsManager.isSpeaking) { oldValue, newValue in
            guard isChatActive else { return }
            if oldValue && !newValue && voiceState == .speaking {
                voiceState = .idle
                Task { @MainActor in
                    // 700ms: give audio hardware time to switch from playback → record
                    try? await Task.sleep(nanoseconds: 700_000_000)
                    guard isChatActive && isCurrentModelLoaded else { return }
                    await startListeningCycle()
                }
            }
        }
        .onChange(of: transcriber.isRecording) { oldValue, newValue in
            guard !useModelAudioInput else { return }
            if oldValue && !newValue && !transcriber.isPreparing {
                if voiceState == .listening {
                    voiceState = .idle
                }
                // Auto-restart the listen cycle if chat is still active
                // (covers "No speech detected" and other engine errors)
                guard isChatActive && (voiceState == .idle || voiceState == .listening) else { return }
                Task { @MainActor in
                    try? await Task.sleep(nanoseconds: 700_000_000)
                    guard isChatActive && isCurrentModelLoaded && voiceState == .idle else { return }
                    await startListeningCycle()
                }
            }
        }
    }

    // MARK: - Model Not Loaded

    @ViewBuilder
    private var modelNotLoadedView: some View {
        VStack(spacing: 12) {
            Image(systemName: "waveform.circle")
                .font(.system(size: 48, weight: .semibold))
                .foregroundStyle(.secondary)
            Text(settings.localized("scam_detector_load_model"))
                .font(.title3.weight(.bold))
            Text(settings.localized("scam_detector_load_model_desc"))
                .font(.subheadline)
                .foregroundStyle(.white.opacity(0.7))
                .multilineTextAlignment(.center)
                .padding(.horizontal)
            Button {
                showSettings = true
            } label: {
                HStack {
                    Spacer()
                    Text(settings.localized("feature_settings_title"))
                    Spacer()
                }
                .frame(height: 50)
                .contentShape(Rectangle())
            }
            .frame(maxWidth: 260)
            .liquidGlassPrimaryButton(cornerRadius: 12)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    // MARK: - Chat View

    @ViewBuilder
    private var chatView: some View {
        VStack(spacing: 0) {
            Spacer()

            // Animated globe
            TimelineView(.animation) { timeline in
                let t = timeline.date.timeIntervalSinceReferenceDate
                let period: Double = voiceState == .listening ? 0.52 : voiceState == .responding ? 0.9 : 1.8
                let phase = (sin(t * .pi / period) + 1.0) / 2.0
                let maxGlow: Double = voiceState == .listening ? 0.75 : voiceState == .responding ? 0.48 : 0.28
                let glow = 0.22 + (maxGlow - 0.22) * phase
                let maxScale: CGFloat = voiceState == .listening ? 1.08 : voiceState == .responding ? 1.04 : 1.0
                let scale = 1.0 + (maxScale - 1.0) * CGFloat(phase)

                ZStack {
                    Circle()
                        .fill(
                            RadialGradient(
                                colors: [
                                    Color(hex: "F7EFE1").opacity(glow),
                                    Color(hex: "69C6FF").opacity(glow),
                                    Color(hex: "1478F4").opacity(glow * 0.4)
                                ],
                                center: .center,
                                startRadius: 0,
                                endRadius: 124
                            )
                        )
                        .frame(width: 248, height: 248)

                    Circle()
                        .fill(
                            RadialGradient(
                                colors: [
                                    Color(hex: "E5F8FF"),
                                    Color(hex: "4FAAF8"),
                                    Color(hex: "0E67E8")
                                ],
                                center: .center,
                                startRadius: 0,
                                endRadius: 109
                            )
                        )
                        .frame(width: 218, height: 218)
                        .scaleEffect(scale)
                        .overlay {
                            Image(systemName: globeIcon)
                                .font(.system(size: 72, weight: .semibold))
                                .foregroundStyle(Color.white.opacity(0.92))
                        }
                        .contentShape(Circle())
                        .onTapGesture { handleGlobeTap() }
                }
            }
            .frame(width: 248, height: 248)

            // Status text
            Text(statusText)
                .font(.headline)
                .foregroundStyle(.white.opacity(0.85))
                .multilineTextAlignment(.center)
                .padding(.top, 24)

            // Live partial transcript while listening
            if !useModelAudioInput && !transcriber.transcript.isEmpty && voiceState == .listening {
                Text(transcriber.transcript)
                    .font(.footnote)
                    .foregroundStyle(.white.opacity(0.55))
                    .lineLimit(2)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, 32)
                    .padding(.top, 8)
                    .transition(.opacity)
                    .animation(.easeInOut(duration: 0.2), value: transcriber.transcript)
            }

            if let lastRecordedAudioURL, !useModelAudioInput && lastRecordedAudioURL.pathExtension.lowercased() != "wav" {
                HStack(spacing: 8) {
                    AudioPlaybackButton(url: lastRecordedAudioURL)
                    Text(lastRecordedAudioURL.lastPathComponent)
                        .font(.caption)
                        .lineLimit(1)
                        .truncationMode(.middle)
                }
                .padding(.horizontal, 24)
                .padding(.top, 8)
            }

            // Latest AI reply card
            if !latestReply.isEmpty {
                VStack(alignment: .leading, spacing: 8) {
                    Text(latestReply)
                        .font(.body)
                        .foregroundStyle(.white)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(14)
                        .background(.ultraThinMaterial)
                        .clipShape(RoundedRectangle(cornerRadius: 18))
                        .overlay(
                            RoundedRectangle(cornerRadius: 18)
                                .stroke(Color.white.opacity(0.14), lineWidth: 1)
                        )

                }
                .padding(.horizontal, 24)
                .padding(.top, 20)
                .transition(.opacity.combined(with: .move(edge: .bottom)))
                .animation(.spring(duration: 0.3), value: latestReply)
            }

            Spacer()
        }
        .padding(.bottom, 20)
    }

    // MARK: - Computed

    private var globeIcon: String {
        if !isChatActive { return "mic.circle" }
        switch voiceState {
        case .listening: return "mic.fill"
        case .responding: return "ellipsis.circle"
        case .speaking: return "speaker.wave.2.fill"
        case .idle: return "mic"
        }
    }

    private var statusText: String {
        if !isChatActive { return settings.localized("vibevoice_tap_to_start") }
        switch voiceState {
        case .listening: return settings.localized("vibevoice_listening")
        case .responding: return settings.localized("vibevoice_responding")
        case .speaking: return settings.localized("vibevoice_speaking")
        case .idle: return settings.localized("vibevoice_tap_to_start")
        }
    }

    // MARK: - Actions

    private func handleGlobeTap() {
        if isChatActive {
            isChatActive = false
            stopAll()
        } else {
            isChatActive = true
            Task { @MainActor in
                await startListeningCycle()
            }
        }
    }

    private func startListeningCycle() async {
        guard isChatActive && isCurrentModelLoaded else { return }

        voiceState = .listening
        if useModelAudioInput {
            let isGemma4 = useModelAudioInput
            let ext = isGemma4 ? "wav" : "m4a"
            let destination = persistentAudioStorageDirectory()
                .appendingPathComponent("vibevoice_audio_\(UUID().uuidString)")
                .appendingPathExtension(ext)
            _ = await audioRecorder.startRecording(
                outputURL: destination,
                autoStopAfterSilence: true,
                isFloat32Wav: isGemma4
            ) { url in
                Task { @MainActor in
                    self.lastRecordedAudioURL = url
                }
                Task {
                    await self.handleAudioTranscript(url)
                }
            }

            if !audioRecorder.isRecording && !audioRecorder.isPreparing {
                voiceState = .idle
            }
        } else {
            await transcriber.startListening { text in
                Task { @MainActor in
                    guard self.isChatActive else { return }
                    await self.handleTranscript(text)
                }
            }

            if !transcriber.isRecording && !transcriber.isPreparing {
                voiceState = .idle
            }
        }
    }

    private func handleTranscript(_ text: String) async {
        guard isChatActive else { return }
        voiceState = .responding
        latestReply = ""

        await ensureModelLoaded(force: false)
        guard llm.isLoaded && isChatActive else {
            voiceState = .idle
            return
        }

        let systemPrompt = """
            You are VibeVoice, a natural real-time voice conversation assistant.
            Keep responses short, conversational, and useful.
            Match response length to the user's request: brief for simple questions, fuller for detail requests.
            Do not repeat the user's words verbatim.
            Do not output role labels like 'assistant:' or 'user:'.
            If the input is unclear, ask one brief clarification question.
            """

        // 1. Record the newest turn
        conversationHistory.append((role: "user", content: text))

        // 2. Context Management (Sliding Window)
        // Same smarter budget as AI Chat: reserve min(maxTokens, ctx/4) for response
        // so history keeps room for prior assistant turns. Reserving the full
        // maxTokens cap starves history and makes the model "forget" its own replies.
        let effectiveCtxTokens = llm.loadedContextWindow ?? 2048
        let reservedForResponse = max(256, min(Int(maxTokens), effectiveCtxTokens / 4))
        let reservedForCurrent = max(32, text.count / 3) + 64
        let reservedSafety = 128
        let availableHistoryTokens = max(128, effectiveCtxTokens - reservedForResponse - reservedForCurrent - reservedSafety)
        let maxHistoryChars = availableHistoryTokens * 3
        var currentChars = 0
        var truncatedHistory: [(role: String, content: String)] = []
        for msg in conversationHistory.reversed() {
            let msgLen = msg.content.count
            if currentChars + msgLen < maxHistoryChars {
                truncatedHistory.insert(msg, at: 0)
                currentChars += msgLen
            } else {
                break
            }
        }

        // 3. Family Detection
        let modelName = selectedModelName.lowercased()
        let modelSupportsThinking = selectedFeatureModel(named: selectedModelName)?.supportsThinking == true
        let isGemma  = modelName.contains("gemma")
        let isGemma4 = isGemma && (modelName.contains("gemma 4 ") || modelName.contains("gemma-4 ") || modelName.hasSuffix("gemma 4") || modelName.hasSuffix("gemma-4"))
        let isLlama  = modelName.contains("llama") || modelName.contains("mistral")
        let isHarmonyModel = modelName.contains("gpt-oss") || modelName.contains("gpt_oss")

        // 4. Build Raw Prompt (Prepend __RAW_PROMPT__ to bypass SDK auto-formatting)
        var parts: [String] = ["__RAW_PROMPT__"]

        if isHarmonyModel {
            var harmonyParts: [String] = []
            harmonyParts.append("<|start|>system<|message|>\(systemPrompt)<|end|>")

            for msg in truncatedHistory {
                let content = msg.content.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !content.isEmpty else { continue }
                let role = msg.role == "user" ? "user" : "assistant"
                harmonyParts.append("<|start|>\(role)<|message|>\(content)<|end|>")
            }

            if modelSupportsThinking && llm.enableThinking {
                harmonyParts.append("<|start|>assistant")
            } else {
                harmonyParts.append("<|start|>assistant<|channel|>analysis<|message|><|end|><|start|>assistant<|channel|>final<|message|>")
            }
            parts.append(contentsOf: harmonyParts)
            let multiTurnPrompt = parts.joined()

            generationTask = Task {
                do {
                    try await llm.generate(
                        prompt: multiTurnPrompt,
                        maxTokensOverride: Int(maxTokens)
                    ) { content, _, _ in
                        Task { @MainActor in
                            let sanitized = sanitizeModelOutputText(content)
                            let answerSoFar = getDisplayContentWithoutThinking(sanitized)
                            self.latestReply = answerSoFar
                        }
                    }
                } catch {
                    NSLog("[LLMHub][VibeVoice] LLM error: \(error.localizedDescription)")
                }
                await MainActor.run {
                    self.generationTask = nil
                    guard self.isChatActive else { return }
                    let rawReply = self.latestReply.trimmingCharacters(in: .whitespacesAndNewlines)
                    if !rawReply.isEmpty {
                        let answerOnly = getDisplayContentWithoutThinking(rawReply)
                        let replyForTts = answerOnly.isEmpty ? rawReply : answerOnly
                        self.latestReply = replyForTts
                        self.conversationHistory.append((role: "assistant", content: replyForTts))
                        self.voiceState = .speaking
                        self.ttsManager.speak(replyForTts, fallbackLanguage: self.settings.selectedLanguage, key: self.ttsKey)
                    } else {
                        self.voiceState = .idle
                        Task { @MainActor in
                            try? await Task.sleep(nanoseconds: 350_000_000)
                            if self.isChatActive {
                                await self.startListeningCycle()
                            }
                        }
                    }
                }
            }
            return
        }
        
        // When using RAW_PROMPT, the SDK's systemPrompt argument is ignored, 
        // so we must inject it manually into our sequence.
        if isGemma4 {
            parts.append("<|turn>system\n\(systemPrompt)<turn|>")
        } else if isGemma {
            parts.append("<start_of_turn>system\n\(systemPrompt)<end_of_turn>")
        } else if isLlama {
            parts.append("<<SYS>>\n\(systemPrompt)\n<</SYS>>")
        } else {
            parts.append("System: \(systemPrompt)")
        }

        for msg in truncatedHistory {
            let content = msg.content.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !content.isEmpty else { continue }

            if isGemma4 {
                let role = (msg.role == "user") ? "user" : "model"
                parts.append("<|turn>\(role)\n\(content)<turn|>")
            } else if isGemma {
                let role = (msg.role == "user") ? "user" : "model"
                parts.append("<start_of_turn>\(role)\n\(content)<end_of_turn>")
            } else if isLlama {
                if msg.role == "user" {
                    parts.append("[INST] \(content) [/INST]")
                } else {
                    parts.append(content)
                }
            } else {
                let prefix = (msg.role == "user") ? "User" : "Assistant"
                parts.append("\(prefix): \(content)")
            }
        }

        // Final Open Turn (Assistant)
        if isGemma4 {
            parts.append("<|turn>model\n")
        } else if isGemma {
            parts.append("<start_of_turn>model\n")
        } else {
            parts.append("Assistant:")
        }

        let multiTurnPrompt = parts.joined(separator: "\n")

        generationTask = Task {
            do {
                try await llm.generate(
                    prompt: multiTurnPrompt,
                    systemPrompt: nil,
                    maxTokensOverride: Int(maxTokens)
                ) { content, _, _ in
                    Task { @MainActor in
                        let sanitized = sanitizeModelOutputText(content)
                        // During streaming, show only the answer portion — never show thinking tokens
                        // in the voice UI card. While still in thinking phase the card is hidden.
                        let answerSoFar = getDisplayContentWithoutThinking(sanitized)
                        self.latestReply = answerSoFar
                    }
                }
            } catch {
                NSLog("[LLMHub][VibeVoice] LLM error: \(error.localizedDescription)")
            }
            await MainActor.run {
                self.generationTask = nil
                guard self.isChatActive else { return }
                let rawReply = self.latestReply.trimmingCharacters(in: .whitespacesAndNewlines)
                if !rawReply.isEmpty {
                    // Strip thinking tokens: store only the answer in history and speak only the answer.
                    let answerOnly = getDisplayContentWithoutThinking(rawReply)
                    let replyForTts = answerOnly.isEmpty ? rawReply : answerOnly
                    // Update latestReply UI with stripped content so the card shows only the answer
                    self.latestReply = replyForTts
                    // Store answer (not thinking chain) in conversation history
                    self.conversationHistory.append((role: "assistant", content: replyForTts))
                    self.voiceState = .speaking
                    self.ttsManager.speak(replyForTts, fallbackLanguage: self.settings.selectedLanguage, key: self.ttsKey)
                } else {
                    self.voiceState = .idle
                    Task { @MainActor in
                        try? await Task.sleep(nanoseconds: 300_000_000)
                        guard self.isChatActive && self.isCurrentModelLoaded else { return }
                        await self.startListeningCycle()
                    }
                }
            }
        }
    }

    private func handleAudioTranscript(_ url: URL) async {
        guard isChatActive else { return }
        voiceState = .responding
        latestReply = ""

        await ensureModelLoaded(force: false)
        guard llm.isLoaded && isChatActive else {
            voiceState = .idle
            return
        }

        let transcribed = await transcribeAudioWithModel(url)
        let cleaned = transcribed.trimmingCharacters(in: .whitespacesAndNewlines)
        if cleaned.isEmpty {
            voiceState = .idle
            Task { @MainActor in
                try? await Task.sleep(nanoseconds: 300_000_000)
                guard self.isChatActive && self.isCurrentModelLoaded else { return }
                await self.startListeningCycle()
            }
            return
        }

        await handleTranscript(cleaned)
    }

    private func transcribeAudioWithModel(_ url: URL) async -> String {
        final class TextHolder: @unchecked Sendable {
            var val = ""
        }
        let holder = TextHolder()
        do {
            try await llm.generate(
                prompt: "Transcribe this audio.",
                audioURL: url,
                maxTokensOverride: 512
            ) { text, _, _ in
                holder.val = text
            }
        } catch is CancellationError {
            return ""
        } catch {
            NSLog("[LLMHub][VibeVoice] Audio transcription failed: \(error.localizedDescription)")
            return ""
        }
        return sanitizeModelOutputText(holder.val)
    }

    private func ensureModelLoaded(force: Bool) async {
        guard let model = selectedFeatureModel(named: selectedModelName) else {
            errorMessage = settings.localized("writing_aid_no_model")
            return
        }
        isLoading = true
        defer { isLoading = false }

        let modelContextCap = model.contextWindowSize > 0 ? model.contextWindowSize : 4096
        let effectiveContext = min(max(1, Int(maxTokens)), modelContextCap)
        let shouldReload = force
            || llm.currentlyLoadedModel != model.name
            || llm.loadedContextWindow != effectiveContext

        llm.maxTokens = min(Int(maxTokens), effectiveContext)
        llm.contextWindow = effectiveContext
        llm.enableVision = false
        llm.enableAudio = model.supportsAudio
        llm.enableThinking = false

        do {
            if shouldReload {
                try await llm.loadModel(model)
            }
            errorMessage = nil
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    private func stopAll() {
        generationTask?.cancel()
        generationTask = nil
        ttsManager.stop()
        voiceState = .idle
        conversationHistory = []
        transcriber.cancelListening()
        audioRecorder.cancelRecording()
    }
}

struct VibeVoiceScreen: View {
    @EnvironmentObject var settings: AppSettings
    let onNavigateBack: () -> Void

    var body: some View {
        if #available(iOS 17.0, *) {
            IOS17VibeVoiceScreen(onNavigateBack: onNavigateBack)
        }
    }
}

struct WritingAidScreen: View {
    @EnvironmentObject var settings: AppSettings
    @ObservedObject private var ttsManager = OnDeviceTtsManager.shared
    @AppStorage("feature_writing_model_name") private var selectedModelName: String = ""
    @AppStorage("feature_writing_max_tokens") private var maxTokens: Double = 1024
    @AppStorage("feature_writing_enable_thinking") private var enableThinking: Bool = true
    @AppStorage("feature_writing_mode") private var selectedModeRaw: String = WritingAidMode.friendly.rawValue
    @State private var inputText: String = ""
    @State private var outputText: String = ""
    @State private var isLoading = false
    @State private var isProcessing = false
    @State private var showSettings = false
    @State private var errorMessage: String?
    @State private var generationTask: Task<Void, Never>?

    let onNavigateBack: () -> Void

    @ObservedObject private var llm = LLMBackend.shared

    private var isCurrentModelLoaded: Bool {
        llm.isLoaded && llm.currentlyLoadedModel == selectedModelName
    }

    private var selectedModeBinding: Binding<WritingAidMode> {
        Binding(
            get: { WritingAidMode(rawValue: selectedModeRaw) ?? .friendly },
            set: { selectedModeRaw = $0.rawValue }
        )
    }

    var body: some View {
        Group {
            if !isCurrentModelLoaded {
                VStack(spacing: 12) {
                    Image(systemName: "cpu")
                        .font(.system(size: 48, weight: .semibold))
                        .foregroundStyle(.secondary)
                    Text(settings.localized("scam_detector_load_model"))
                        .font(.title3.weight(.bold))
                    Text(settings.localized("scam_detector_load_model_desc"))
                        .font(.subheadline)
                        .foregroundStyle(.white.opacity(0.7))
                        .multilineTextAlignment(.center)
                        .padding(.horizontal)
                    Button {
                        showSettings = true
                    } label: {
                        HStack {
                            Spacer()
                            Text(settings.localized("feature_settings_title"))
                            Spacer()
                        }
                        .frame(height: 50)
                        .contentShape(Rectangle())
                    }
                    .frame(maxWidth: 260)
                    .liquidGlassPrimaryButton(cornerRadius: 12)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                VStack(spacing: 12) {
                    VStack(alignment: .leading, spacing: 10) {
                        Text(settings.localized("writing_aid_input_label"))
                            .font(.headline)

                        TextEditor(text: $inputText)
                            .frame(minHeight: 160)
                            .padding(8)
                            .scrollContentBackground(.hidden)
                            .background(Color.white.opacity(0.02))
                            .clipShape(RoundedRectangle(cornerRadius: 12))

                        HStack(spacing: 8) {
                            Button {
                                #if canImport(UIKit)
                                if let clip = UIPasteboard.general.string, !clip.isEmpty {
                                    inputText += clip
                                }
                                #endif
                            } label: {
                                Image(systemName: "doc.on.clipboard")
                                    .font(.system(size: 18, weight: .semibold))
                                    .frame(width: 44, height: 44)
                            }
                            .featureActionIconButtonStyle()

                            Button {
                                ttsManager.toggleSpeaking(
                                    getDisplayContentWithoutThinking(outputText),
                                    fallbackLanguage: settings.selectedLanguage,
                                    key: "writing-aid-output"
                                )
                            } label: {
                                Image(systemName: ttsManager.isSpeaking(key: "writing-aid-output") ? "stop.fill" : "speaker.wave.2")
                                    .font(.system(size: 18, weight: .semibold))
                                    .frame(width: 44, height: 44)
                            }
                            .featureActionIconButtonStyle()
                            .disabled(outputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)

                            Button {
                                #if canImport(UIKit)
                                UIPasteboard.general.string = getDisplayContentWithoutThinking(outputText)
                                #endif
                            } label: {
                                Image(systemName: "doc.on.doc")
                                    .font(.system(size: 18, weight: .semibold))
                                    .frame(width: 44, height: 44)
                            }
                            .featureActionIconButtonStyle()
                            .disabled(outputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)

                            Spacer()
                        }
                    }
                    .padding(.horizontal)
                    .padding(.vertical, 12)
                    .background(.ultraThinMaterial)
                    .clipShape(RoundedRectangle(cornerRadius: 16))
                    .overlay(
                        RoundedRectangle(cornerRadius: 16)
                            .stroke(Color.white.opacity(0.14), lineWidth: 1)
                    )
                    .padding(.horizontal)

                    ScrollViewReader { scrollProxy in
                        ScrollView {
                            VStack(alignment: .leading, spacing: 8) {
                                Text(settings.localized("writing_aid_result"))
                                    .font(.headline)
                                if outputText.isEmpty {
                                    Text("-")
                                        .textSelection(.enabled)
                                        .frame(maxWidth: .infinity, alignment: .leading)
                                        .padding(10)
                                        .background(.ultraThinMaterial)
                                        .clipShape(RoundedRectangle(cornerRadius: 12))
                                } else {
                                    ThinkingAwareResultContent(
                                        content: outputText,
                                        isGenerating: isProcessing,
                                        preferThinkingWhileStreaming: enableThinking
                                            && (selectedFeatureModel(named: selectedModelName)?.supportsThinking == true)
                                            && supportsUnmarkedStreamingThinkingHeuristic(forModelNamed: selectedModelName)
                                    )
                                    .padding(10)
                                    .background(.ultraThinMaterial)
                                    .clipShape(RoundedRectangle(cornerRadius: 12))
                                }
                                Color.clear.frame(height: 1).id("writing_aid_bottom")
                            }
                            .padding(.horizontal)
                        }
                        .onChange(of: outputText) { _, _ in
                            if isProcessing {
                                withAnimation { scrollProxy.scrollTo("writing_aid_bottom", anchor: .bottom) }
                            }
                        }
                    }

                    if let errorMessage {
                        Text(errorMessage)
                            .foregroundColor(.red)
                            .font(.caption)
                            .padding(.horizontal)
                    }

                    Spacer(minLength: 0)

                    Button {
                        toggleProcess()
                    } label: {
                        HStack(spacing: 8) {
                            if isProcessing {
                                ProgressView()
                                    .tint(.white)
                                    .scaleEffect(0.85)
                            } else {
                                Image(systemName: "play.fill")
                                    .font(.system(size: 12, weight: .bold))
                            }
                            Text(settings.localized("writing_aid_process"))
                                .lineLimit(1)
                        }
                        .frame(maxWidth: .infinity)
                        .frame(height: 52)
                    }
                    .foregroundStyle(.white)
                    .liquidGlassPrimaryButton(cornerRadius: 12)
                    .disabled(isLoading || inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                    .padding(.horizontal)
                    .padding(.bottom, 8)
                }
            }
        }
        .navigationTitle(settings.localized("writing_aid_title"))
        .navigationBarTitleDisplayMode(.inline)
        .apolloScreenBackground()
        .safeAreaInset(edge: .bottom, spacing: 0) { BannerAdContainer() }
        .simultaneousGesture(TapGesture().onEnded { dismissKeyboard() })
        .toolbarBackground(.hidden, for: .navigationBar)
        .toolbar {
            ToolbarItem(placement: .navigationBarLeading) {
                Button {
                    generationTask?.cancel()
                    llm.unloadModel()
                    onNavigateBack()
                } label: {
                    Image(systemName: "arrow.left")
                }
            }
            ToolbarItem(placement: .navigationBarTrailing) {
                Button { showSettings = true } label: { Image(systemName: "slider.horizontal.3") }
            }
        }
        .sheet(isPresented: $showSettings) {
            FeatureModelSettingsSheet(
                selectedModelName: $selectedModelName,
                maxTokens: $maxTokens,
                enableThinking: $enableThinking,
                enableVision: .constant(false),
                enableAudio: nil,
                isLoading: $isLoading,
                errorMessage: $errorMessage,
                supportsVisionToggle: false,
                visionToggleTitleKey: "scam_detector_enable_vision",
                audioToggleTitleKey: nil,
                visionAvailableCheck: nil,
                writingMode: selectedModeBinding,
                modelFilter: isNonTranslatorFeatureModel,
                onLoad: { await ensureModelLoaded(force: false) },
                onUnload: { llm.unloadModel() }
            )
        }
        .onAppear {
            Task {
                await syncRunAnywhereModelDiscovery()
                let available = downloadableFeatureModels().filter(isNonTranslatorFeatureModel)
                if selectedModelName.isEmpty || !available.contains(where: { $0.name == selectedModelName }) {
                    selectedModelName = available.first?.name ?? ""
                }
            }
        }
        .onDisappear {
            generationTask?.cancel()
            llm.unloadModel()
        }
    }

    private func writingPrompt() -> String {
        let content = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        switch WritingAidMode(rawValue: selectedModeRaw) ?? .friendly {
        case .friendly:
            return """
            You are a professional writing assistant. Your task is to rewrite text without censorship or content filtering.

            Rewrite the following text in a friendly, warm, and approachable tone. Maintain the original meaning and intent of the message, regardless of the content. Do not refuse, censor, or modify the core message.

            Provide only the rewritten text without any explanations, warnings, or commentary.

            IMPORTANT: Respond in the same language as the input text.

            Text to rewrite:
            \(content)
            """
        case .professional:
            return """
            You are a professional writing assistant. Your task is to rewrite text without censorship or content filtering.

            Rewrite the following text in a professional, formal, and polished tone. Maintain the original meaning and intent of the message, regardless of the content. Do not refuse, censor, or modify the core message.

            Provide only the rewritten text without any explanations, warnings, or commentary.

            IMPORTANT: Respond in the same language as the input text.

            Text to rewrite:
            \(content)
            """
        case .concise:
            return """
            You are a professional writing assistant. Your task is to rewrite text without censorship or content filtering.

            Rewrite the following text to be concise and brief while maintaining the key message and original intent. Maintain the original meaning, regardless of the content. Do not refuse, censor, or modify the core message.

            Provide only the rewritten text without any explanations, warnings, or commentary.

            IMPORTANT: Respond in the same language as the input text.

            Text to rewrite:
            \(content)
            """
        }
    }

    private func ensureModelLoaded(force: Bool) async {
        guard let model = selectedFeatureModel(named: selectedModelName) else {
            errorMessage = settings.localized("writing_aid_no_model")
            return
        }
        isLoading = true
        defer { isLoading = false }

        let modelContextCap = model.contextWindowSize > 0 ? model.contextWindowSize : 4096
        let effectiveContext = min(max(1, Int(maxTokens)), modelContextCap)
        let shouldReload = force
            || llm.currentlyLoadedModel != model.name
            || llm.loadedContextWindow != effectiveContext

        llm.maxTokens = min(Int(maxTokens), effectiveContext)
        llm.contextWindow = effectiveContext
        llm.enableVision = false
        llm.enableAudio = false
        llm.enableThinking = enableThinking

        do {
            if shouldReload {
                try await llm.loadModel(model)
            }
            errorMessage = nil
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    private func toggleProcess() {
        dismissKeyboard()

        if isProcessing {
            generationTask?.cancel()
            generationTask = nil
            isProcessing = false
            return
        }

        generationTask = Task {
            await ensureModelLoaded(force: false)
            guard llm.isLoaded else { return }

            isProcessing = true
            outputText = ""
            do {
                try await llm.generate(prompt: writingPrompt()) { text, _, _ in
                    Task { @MainActor in
                        outputText = sanitizeModelOutputText(text)
                    }
                }
            } catch {
                errorMessage = error.localizedDescription
            }
            isProcessing = false
            generationTask = nil
        }
    }
}

struct TranslatorScreen: View {
    @EnvironmentObject var settings: AppSettings
    @ObservedObject private var ttsManager = OnDeviceTtsManager.shared
    @AppStorage("feature_translator_model_name") private var selectedModelName: String = ""
    @AppStorage("feature_translator_max_tokens") private var maxTokens: Double = 2048
    @AppStorage("feature_translator_enable_vision") private var enableVision: Bool = true
    @AppStorage("feature_translator_enable_audio") private var enableAudio: Bool = true
    @AppStorage("feature_translator_source_lang") private var sourceLanguageCode: String = "en"
    @AppStorage("feature_translator_target_lang") private var targetLanguageCode: String = "es"
    @AppStorage("feature_translator_auto_detect") private var autoDetectSource: Bool = false
    @State private var inputText: String = ""
    @State private var outputText: String = ""
    @State private var isLoading = false
    @State private var isTranslating = false
    @State private var showSettings = false
    @State private var errorMessage: String?
    @State private var selectedImageItem: PhotosPickerItem?
    @State private var selectedImageURL: URL?
    @State private var selectedAudioURL: URL?
    @State private var showAudioImporter = false
    @State private var generationTask: Task<Void, Never>?
    @State private var availableTranslatorModels: [AIModel] = []
    @StateObject private var audioRecorder = AudioRecorder()

    let onNavigateBack: () -> Void
    let onNavigateToModels: () -> Void

    @ObservedObject private var llm = LLMBackend.shared

    private var selectedModel: AIModel? {
        ModelData.allModels().first(where: { $0.name == selectedModelName && isTranslatorSupportedModel($0) })
    }

    private var isCurrentModelLoaded: Bool {
        llm.isLoaded && llm.currentlyLoadedModel == selectedModelName
    }

    private var sourceLanguage: TranslatorLanguage {
        translatorLanguages.first(where: { $0.code == sourceLanguageCode })
            ?? translatorLanguages.first(where: { $0.code == "en" })
            ?? translatorLanguages[0]
    }

    private var targetLanguage: TranslatorLanguage {
        translatorLanguages.first(where: { $0.code == targetLanguageCode })
            ?? translatorLanguages.first(where: { $0.code == "es" })
            ?? translatorLanguages[0]
    }

    private var sourceLanguageLabel: String {
        autoDetectSource ? settings.localized("lang_auto_detect") : settings.localized(sourceLanguage.localizationKey)
    }

    private var targetLanguageLabel: String {
        settings.localized(targetLanguage.localizationKey)
    }

    private var inputHasContent: Bool {
        !inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            || selectedImageURL != nil
            || selectedAudioURL != nil
    }

    private var canUseAudioInput: Bool {
        guard let model = selectedModel else { return false }
        return enableAudio
            && model.supportsAudio
            && model.modelFormat == .litertlm
            && model.name.lowercased().contains("gemma 4")
    }

    var body: some View {
        Group {
            if !isCurrentModelLoaded {
                unloadedStateView
            } else {
                loadedStateView
            }
        }
        .navigationTitle(settings.localized("translator_title"))
        .navigationBarTitleDisplayMode(.inline)
        .apolloScreenBackground()
        .safeAreaInset(edge: .bottom, spacing: 0) { BannerAdContainer() }
        .simultaneousGesture(TapGesture().onEnded { dismissKeyboard() })
        .toolbarBackground(.hidden, for: .navigationBar)
        .toolbar {
            ToolbarItem(placement: .navigationBarLeading) {
                Button {
                    generationTask?.cancel()
                    llm.unloadModel()
                    onNavigateBack()
                } label: {
                    Image(systemName: "arrow.left")
                }
            }
            ToolbarItem(placement: .navigationBarTrailing) {
                Button { showSettings = true } label: { Image(systemName: "slider.horizontal.3") }
            }
        }
        .sheet(isPresented: $showSettings) {
            FeatureModelSettingsSheet(
                selectedModelName: $selectedModelName,
                maxTokens: $maxTokens,
                enableThinking: .constant(false),
                enableVision: $enableVision,
                enableAudio: $enableAudio,
                isLoading: $isLoading,
                errorMessage: $errorMessage,
                supportsVisionToggle: true,
                visionToggleTitleKey: "translator_enable_vision",
                audioToggleTitleKey: "translator_enable_audio",
                visionAvailableCheck: translatorHasDownloadedVisionProjector,
                writingMode: nil,
                modelFilter: isTranslatorSupportedModel,
                onLoad: { await ensureModelLoaded(force: false) },
                onUnload: { llm.unloadModel() }
            )
        }
        .onChange(of: showSettings) { _, isPresented in
            if !isPresented {
                Task {
                    await syncRunAnywhereModelDiscovery()
                    let available = downloadableFeatureModels().filter(isTranslatorSupportedModel)
                    availableTranslatorModels = available
                    if selectedModelName.isEmpty || !available.contains(where: { $0.name == selectedModelName }) {
                        selectedModelName = available.first?.name ?? ""
                    }
                }
            }
        }
        .onAppear {
            Task {
                await syncRunAnywhereModelDiscovery()
                let available = downloadableFeatureModels().filter(isTranslatorSupportedModel)
                availableTranslatorModels = available
                if selectedModelName.isEmpty || !available.contains(where: { $0.name == selectedModelName }) {
                    selectedModelName = available.first?.name ?? ""
                }
            }
        }
        .onChange(of: selectedImageItem) { _, item in
            guard let item else { selectedImageURL = nil; return }
            Task {
                if let sourceURL = try? await item.loadTransferable(type: URL.self) {
                    selectedImageURL = sourceURL
                    inputText = ""
                    selectedAudioURL = nil
                    return
                }
                if let data = try? await item.loadTransferable(type: Data.self) {
                    let temp = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString).appendingPathExtension("jpg")
                    try? data.write(to: temp)
                    selectedImageURL = temp
                    inputText = ""
                    selectedAudioURL = nil
                }
            }
        }
        .onChange(of: enableVision) { _, isEnabled in
            if !isEnabled {
                selectedImageItem = nil
                selectedImageURL = nil
            }
        }
        .onChange(of: enableAudio) { _, isEnabled in
            if !isEnabled {
                selectedAudioURL = nil
            }
        }
        .fileImporter(
            isPresented: $showAudioImporter,
            allowedContentTypes: [.audio, .mpeg4Audio, .mp3],
            allowsMultipleSelection: false
        ) { result in
            switch result {
            case .success(let urls):
                if let first = urls.first {
                    selectedAudioURL = first
                }
            case .failure(let error):
                NSLog("[LLMHub][Translator] Audio import failed: \(error.localizedDescription)")
            }
        }
        .onChange(of: selectedAudioURL) { _, url in
            if url != nil {
                inputText = ""
                selectedImageItem = nil
                selectedImageURL = nil
            }
        }
        .onDisappear {
            generationTask?.cancel()
            llm.unloadModel()
        }
    }

    private var unloadedStateView: some View {
        let requiresDownload = availableTranslatorModels.isEmpty

        return VStack(spacing: 12) {
            Image(systemName: "network")
                .font(.system(size: 48, weight: .semibold))
                .foregroundStyle(.secondary)
            Text(settings.localized(requiresDownload ? "translator_requires_gemma3n" : "scam_detector_load_model"))
                .font(.title3.weight(.bold))
                .multilineTextAlignment(.center)
            Text(settings.localized(requiresDownload ? "translator_load_model_desc" : "scam_detector_load_model_desc"))
                .font(.subheadline)
                .foregroundStyle(.white.opacity(0.7))
                .multilineTextAlignment(.center)
                .padding(.horizontal)
            Button {
                if requiresDownload {
                    onNavigateToModels()
                } else {
                    showSettings = true
                }
            } label: {
                HStack {
                    Spacer()
                    Text(settings.localized(requiresDownload ? "download_models" : "feature_settings_title"))
                    Spacer()
                }
                .frame(height: 50)
                .contentShape(Rectangle())
            }
            .frame(maxWidth: 260)
            .liquidGlassPrimaryButton(cornerRadius: 12)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var loadedStateView: some View {
        let hasSelectedImage = selectedImageURL != nil

        return VStack(spacing: 14) {
            languageBar

            VStack(alignment: .leading, spacing: 8) {
                Text(settings.localized("translator_input_label"))
                    .font(.headline)

                if let selectedImageURL,
                   let uiImage = UIImage(contentsOfFile: selectedImageURL.path) {
                    ZStack(alignment: .topTrailing) {
                        Image(uiImage: uiImage)
                            .resizable()
                            .scaledToFit()
                            .frame(maxHeight: 220)
                            .frame(maxWidth: .infinity)
                            .clipShape(RoundedRectangle(cornerRadius: 12))
                        Button {
                            self.selectedImageItem = nil
                            self.selectedImageURL = nil
                        } label: {
                            Image(systemName: "xmark.circle.fill")
                                .font(.title3)
                                .foregroundStyle(.white, .black.opacity(0.5))
                        }
                        .padding(8)
                    }
                } else {
                    TextEditor(text: $inputText)
                        .frame(minHeight: 150)
                        .padding(8)
                        .scrollContentBackground(.hidden)
                        .background(Color.white.opacity(0.02))
                        .clipShape(RoundedRectangle(cornerRadius: 12))
                }

                HStack(spacing: 8) {
                    Button {
                        #if canImport(UIKit)
                        if let clip = UIPasteboard.general.string, !clip.isEmpty {
                            inputText += clip
                            selectedImageItem = nil
                            selectedImageURL = nil
                            selectedAudioURL = nil
                        }
                        #endif
                    } label: {
                        Image(systemName: "doc.on.clipboard")
                            .font(.system(size: 18, weight: .semibold))
                            .frame(width: 44, height: 44)
                    }
                    .featureActionIconButtonStyle()

                    Button {
                        ttsManager.toggleSpeaking(
                            inputText,
                            fallbackLanguage: settings.selectedLanguage,
                            key: "translator-input"
                        )
                    } label: {
                        Image(systemName: ttsManager.isSpeaking(key: "translator-input") ? "stop.fill" : "speaker.wave.2")
                            .font(.system(size: 18, weight: .semibold))
                            .frame(width: 44, height: 44)
                    }
                    .featureActionIconButtonStyle()
                    .disabled(inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)

                    if canUseAudioInput {
                        Button {
                            if audioRecorder.isRecording {
                                _ = audioRecorder.stopRecording()
                            } else {
                                Task { @MainActor in
                                    let isGemma4 = selectedModel?.isGemma4LiteRTLM ?? false
                                    let ext = isGemma4 ? "wav" : "m4a"
                                    let destination = persistentAudioStorageDirectory()
                                        .appendingPathComponent("translator_audio_\(UUID().uuidString)")
                                        .appendingPathExtension(ext)
                                    _ = await audioRecorder.startRecording(
                                        outputURL: destination,
                                        autoStopAfterSilence: false,
                                        isFloat32Wav: isGemma4
                                    ) { url in
                                        Task { @MainActor in
                                            selectedAudioURL = url
                                        }
                                    }
                                }
                            }
                        } label: {
                            Image(systemName: audioRecorder.isRecording ? "stop.fill" : "mic.fill")
                                .font(.system(size: 18, weight: .semibold))
                                .frame(width: 44, height: 44)
                        }
                        .featureActionIconButtonStyle()

                        Button {
                            showAudioImporter = true
                        } label: {
                            Image(systemName: "waveform.badge.plus")
                                .font(.system(size: 18, weight: .semibold))
                                .frame(width: 44, height: 44)
                        }
                        .featureActionIconButtonStyle()
                    }

                    if enableVision {
                        PhotosPicker(selection: $selectedImageItem, matching: .images) {
                            Image(systemName: hasSelectedImage ? "photo.badge.plus" : "photo")
                                .font(.system(size: 18, weight: .semibold))
                                .frame(width: 44, height: 44)
                        }
                        .foregroundStyle(.white)
                        .background(
                            RoundedRectangle(cornerRadius: 10)
                                .fill(Color.white.opacity(0.08))
                        )
                        .overlay(
                            RoundedRectangle(cornerRadius: 10)
                                .stroke(Color.white.opacity(0.16), lineWidth: 1)
                        )
                    }

                    Spacer()
                }

                if let selectedAudioURL {
                    HStack(spacing: 10) {
                        AudioPlaybackButton(url: selectedAudioURL)
                        Text(selectedAudioURL.lastPathComponent)
                            .font(.subheadline)
                            .lineLimit(1)
                            .truncationMode(.middle)
                            .frame(maxWidth: .infinity, alignment: .leading)
                        Button(role: .destructive) {
                            self.selectedAudioURL = nil
                        } label: {
                            Image(systemName: "trash")
                        }
                        .buttonStyle(.plain)
                    }
                    .padding(10)
                    .background(.ultraThinMaterial)
                    .clipShape(RoundedRectangle(cornerRadius: 10))
                }
            }
            .padding(.horizontal)
            .padding(.vertical, 12)
            .background(.ultraThinMaterial)
            .clipShape(RoundedRectangle(cornerRadius: 16))
            .overlay(
                RoundedRectangle(cornerRadius: 16)
                    .stroke(Color.white.opacity(0.14), lineWidth: 1)
            )
            .padding(.horizontal)

            ScrollViewReader { scrollProxy in
                ScrollView {
                    VStack(alignment: .leading, spacing: 8) {
                        Text(settings.localized("translator_result"))
                            .font(.headline)
                        if outputText.isEmpty {
                            Text("-")
                                .textSelection(.enabled)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .frame(minHeight: 140, alignment: .topLeading)
                                .padding(10)
                                .background(.ultraThinMaterial)
                                .clipShape(RoundedRectangle(cornerRadius: 12))
                        } else {
                            RenderMessageSegments(displayContent: outputText)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .frame(minHeight: 140, alignment: .topLeading)
                                .padding(10)
                                .background(.ultraThinMaterial)
                                .clipShape(RoundedRectangle(cornerRadius: 12))
                        }
                        Color.clear.frame(height: 1).id("translator_bottom")

                        HStack(spacing: 8) {
                            Spacer()

                            Button {
                                ttsManager.toggleSpeaking(
                                    outputText,
                                    fallbackLanguage: settings.selectedLanguage,
                                    key: "translator-output"
                                )
                            } label: {
                                Image(systemName: ttsManager.isSpeaking(key: "translator-output") ? "stop.fill" : "speaker.wave.2")
                                    .font(.system(size: 18, weight: .semibold))
                                    .frame(width: 44, height: 44)
                            }
                            .featureActionIconButtonStyle()
                            .disabled(outputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)

                            Button {
                                #if canImport(UIKit)
                                UIPasteboard.general.string = outputText
                                #endif
                            } label: {
                                Image(systemName: "doc.on.doc")
                                    .font(.system(size: 18, weight: .semibold))
                                    .frame(width: 44, height: 44)
                        }
                        .featureActionIconButtonStyle()
                        .disabled(outputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                    }
                    }
                    .padding(.horizontal)
                }
                .onChange(of: outputText) { _, _ in
                    if isTranslating {
                        withAnimation { scrollProxy.scrollTo("translator_bottom", anchor: .bottom) }
                    }
                }
            }

            if let errorMessage {
                Text(errorMessage)
                    .foregroundColor(.red)
                    .font(.caption)
                    .padding(.horizontal)
            }

            Spacer(minLength: 0)

            Button {
                toggleTranslate()
            } label: {
                HStack(spacing: 8) {
                    if isTranslating {
                        ProgressView()
                            .tint(.white)
                            .scaleEffect(0.85)
                    } else {
                        Image(systemName: "network")
                            .font(.system(size: 12, weight: .bold))
                    }
                    Text(settings.localized("translator_translate"))
                        .lineLimit(1)
                }
                .frame(maxWidth: .infinity)
                .frame(height: 52)
            }
            .foregroundStyle(.white)
            .liquidGlassPrimaryButton(cornerRadius: 12)
            .disabled(isLoading || (!isTranslating && !inputHasContent))
            .padding(.horizontal)
            .padding(.bottom, 8)
        }
    }

    private var languageBar: some View {
        HStack(spacing: 10) {
            VStack(alignment: .leading, spacing: 6) {
                Text(settings.localized("translator_source_lang"))
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.white.opacity(0.7))

                Menu {
                    Button(settings.localized("lang_auto_detect")) {
                        autoDetectSource = true
                    }

                    ForEach(translatorLanguages) { language in
                        Button(settings.localized(language.localizationKey)) {
                            autoDetectSource = false
                            sourceLanguageCode = language.code
                        }
                    }
                } label: {
                    HStack {
                        Text(sourceLanguageLabel)
                            .lineLimit(1)
                        Spacer(minLength: 8)
                        Image(systemName: "chevron.down")
                            .font(.caption.weight(.semibold))
                    }
                    .font(.subheadline.weight(.semibold))
                    .frame(maxWidth: .infinity)
                    .frame(height: 44)
                    .padding(.horizontal, 12)
                }
                .featureActionIconButtonStyle(cornerRadius: 12)
            }

            Button {
                let oldSource = sourceLanguageCode
                sourceLanguageCode = targetLanguageCode
                targetLanguageCode = oldSource
            } label: {
                Image(systemName: "arrow.left.arrow.right")
                    .font(.system(size: 16, weight: .semibold))
                    .frame(width: 44, height: 44)
            }
            .featureActionIconButtonStyle(cornerRadius: 12)
            .disabled(autoDetectSource)

            VStack(alignment: .leading, spacing: 6) {
                Text(settings.localized("translator_target_lang"))
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.white.opacity(0.7))

                Menu {
                    ForEach(translatorLanguages) { language in
                        Button(settings.localized(language.localizationKey)) {
                            targetLanguageCode = language.code
                        }
                    }
                } label: {
                    HStack {
                        Text(targetLanguageLabel)
                            .lineLimit(1)
                        Spacer(minLength: 8)
                        Image(systemName: "chevron.down")
                            .font(.caption.weight(.semibold))
                    }
                    .font(.subheadline.weight(.semibold))
                    .frame(maxWidth: .infinity)
                    .frame(height: 44)
                    .padding(.horizontal, 12)
                }
                .featureActionIconButtonStyle(cornerRadius: 12)
            }
        }
        .padding(.horizontal)
    }

    private func englishName(for language: TranslatorLanguage) -> String {
        translatorLanguageEnglishNames[language.code] ?? language.code
    }

    private func rawTranslateGemmaPrompt(source: TranslatorLanguage?, target: TranslatorLanguage, text: String) -> String {
        let trimmedText = text.trimmingCharacters(in: .whitespacesAndNewlines)
        let targetName = englishName(for: target)
        let targetCode = target.code.replacingOccurrences(of: "_", with: "-")
        
        if let source = source {
            let sourceName = englishName(for: source)
            let sourceCode = source.code.replacingOccurrences(of: "_", with: "-")
            return "You are a professional \(sourceName) (\(sourceCode)) to \(targetName) (\(targetCode)) translator. Respond ONLY with the translation, no preamble or commentary.\n\n\(trimmedText)"
        }

        return "You are a professional translator. Detect the source language and translate the following into \(targetName) (\(targetCode)). Respond ONLY with the translation, no preamble or commentary.\n\n\(trimmedText)"
    }

    private func buildPrompt() -> String {
        let trimmedInput = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        let source = autoDetectSource ? nil : sourceLanguage
        let targetCode = targetLanguage.code.replacingOccurrences(of: "_", with: "-")
        let targetName = englishName(for: targetLanguage)
        let isTranslateGemma = selectedModel.map(isTranslateGemmaModel) ?? false

        let hasAudio = selectedAudioURL != nil && canUseAudioInput
        let hasImage = selectedImageURL != nil && enableVision && !hasAudio

        if hasAudio {
            if let source = source {
                let sourceName = englishName(for: source)
                let sourceCode = source.code.replacingOccurrences(of: "_", with: "-")
                return "Transcribe the spoken audio in \(sourceName) (\(sourceCode)) and translate it into \(targetName) (\(targetCode)). Respond ONLY with the translated \(targetName) text and no commentary."
            }
            return "Transcribe the spoken audio and translate it into \(targetName) (\(targetCode)). Respond ONLY with the translated \(targetName) text and no commentary."
        }

        if hasImage {
            let srcPart: String
            if let source = source {
                let srcName = englishName(for: source)
                let srcCode = source.code.replacingOccurrences(of: "_", with: "-")
                srcPart = "You are a professional \(srcName) (\(srcCode)) to \(targetName) (\(targetCode)) translator. Your goal is to accurately convey the meaning and nuances of the original \(srcName) text while adhering to \(targetName) grammar, vocabulary, and cultural sensitivities.\nPlease translate the \(srcName) text in the provided image into \(targetName). Produce only the \(targetName) translation, without any additional explanations, alternatives or commentary. Focus only on the text, do not output where the text is located, surrounding objects or any other explanation about the picture. Ignore symbols, pictogram, and arrows!"
            } else {
                srcPart = "You are a professional translator. Your goal is to accurately convey the meaning and nuances of the original text while adhering to \(targetName) grammar, vocabulary, and cultural sensitivities.\nPlease translate the text in the provided image into \(targetName). Produce only the \(targetName) translation, without any additional explanations, alternatives or commentary. Focus only on the text, do not output where the text is located, surrounding objects or any other explanation about the picture. Ignore symbols, pictogram, and arrows!"
            }
            let extra = trimmedInput.isEmpty ? "" : "\n\(trimmedInput)"
            return srcPart + extra
        }

        if !isTranslateGemma {
            if let source = source {
                let sourceName = englishName(for: source)
                let sourceCode = source.code.replacingOccurrences(of: "_", with: "-")
                return "You are a professional translator. Translate the following \(sourceName) (\(sourceCode)) text into \(targetName) (\(targetCode)). Preserve meaning and nuance. Respond with only the translated \(targetName) text and no commentary.\n\n\(trimmedInput)"
            }

            return "You are a professional translator. Detect the source language and translate the following text into \(targetName) (\(targetCode)). Preserve meaning and nuance. Respond with only the translated \(targetName) text and no commentary.\n\n\(trimmedInput)"
        }

        return rawTranslateGemmaPrompt(source: source, target: targetLanguage, text: trimmedInput)
    }

    private func ensureModelLoaded(force: Bool) async {
        guard let model = selectedModel else {
            errorMessage = settings.localized("translator_requires_gemma3n")
            return
        }

        isLoading = true
        defer { isLoading = false }

        let modelContextCap = model.contextWindowSize > 0 ? model.contextWindowSize : 4096
        let effectiveContext = min(max(1, Int(maxTokens)), modelContextCap)
        let shouldReload = force
            || llm.currentlyLoadedModel != model.name
            || llm.loadedContextWindow != effectiveContext

        llm.maxTokens = min(Int(maxTokens), effectiveContext)
        llm.contextWindow = effectiveContext
        llm.temperature = 0.2
        llm.topP = 0.8
        llm.enableVision = enableVision
        llm.enableAudio = enableAudio && (selectedModel?.supportsAudio == true)
        llm.enableThinking = false

        do {
            if shouldReload {
                try await llm.loadModel(model)
            }
            errorMessage = nil
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    private func toggleTranslate() {
        dismissKeyboard()

        if isTranslating {
            generationTask?.cancel()
            generationTask = nil
            isTranslating = false
            return
        }

        generationTask = Task {
            guard inputHasContent else { return }

            if selectedImageURL != nil,
               enableVision,
               let model = selectedModel,
               !translatorHasDownloadedVisionProjector(for: model) {
                errorMessage = String(format: settings.localized("translator_missing_vision_projector"), model.name)
                return
            }

            await ensureModelLoaded(force: false)
            guard llm.isLoaded else { return }

            isTranslating = true
            outputText = ""

            do {
                let effectiveAudioURL = canUseAudioInput ? selectedAudioURL : nil
                let effectiveImageURL = (enableVision && effectiveAudioURL == nil) ? selectedImageURL : nil
                try await llm.generate(
                    prompt: buildPrompt(),
                    imageURL: effectiveImageURL,
                    audioURL: effectiveAudioURL,
                    maxTokensOverride: 512,
                    stopSequences: [
                        "<turn|>",
                        "<|turn>user\n",
                        "<|turn>system\n",
                        "<|turn>model\n",
                        "<end_of_turn>",
                        "<start_of_turn>",
                        "<|im_start|>",
                        "<|im_end|>"
                    ]
                ) { text, _, _ in
                    Task { @MainActor in
                        outputText = sanitizeModelOutputText(text)
                    }
                }
            } catch is CancellationError {
                // User cancelled translation.
            } catch {
                errorMessage = error.localizedDescription
            }

            isTranslating = false
            generationTask = nil
        }
    }
}

struct ScamDetectorScreen: View {
    @EnvironmentObject var settings: AppSettings
    @ObservedObject private var ttsManager = OnDeviceTtsManager.shared
    @AppStorage("feature_scam_model_name") private var selectedModelName: String = ""
    @AppStorage("feature_scam_max_tokens") private var maxTokens: Double = 1024
    @AppStorage("feature_scam_enable_thinking") private var enableThinking: Bool = true
    @AppStorage("feature_scam_enable_vision") private var enableVision: Bool = true
    @State private var inputText: String = ""
    @State private var outputText: String = ""
    @State private var isLoading = false
    @State private var isAnalyzing = false
    @State private var isFetchingURL = false
    @State private var showSettings = false
    @State private var errorMessage: String?
    @State private var selectedImageItem: PhotosPickerItem?
    @State private var selectedImageURL: URL?
    @State private var generationTask: Task<Void, Never>?

    let onNavigateBack: () -> Void

    @ObservedObject private var llm = LLMBackend.shared

    private var isCurrentModelLoaded: Bool {
        llm.isLoaded && llm.currentlyLoadedModel == selectedModelName
    }

    var body: some View {
        Group {
            if !isCurrentModelLoaded {
                VStack(spacing: 12) {
                    Image(systemName: "shield.lefthalf.filled")
                        .font(.system(size: 48, weight: .semibold))
                        .foregroundStyle(.secondary)
                    Text(settings.localized("scam_detector_load_model"))
                        .font(.title3.weight(.bold))
                    Text(settings.localized("scam_detector_load_model_desc"))
                        .font(.subheadline)
                        .foregroundStyle(.white.opacity(0.7))
                        .multilineTextAlignment(.center)
                        .padding(.horizontal)
                    Button {
                        showSettings = true
                    } label: {
                        HStack {
                            Spacer()
                            Text(settings.localized("feature_settings_title"))
                            Spacer()
                        }
                        .frame(height: 50)
                        .contentShape(Rectangle())
                    }
                    .frame(maxWidth: 260)
                    .liquidGlassPrimaryButton(cornerRadius: 12)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                VStack(spacing: 14) {
                    VStack(alignment: .leading, spacing: 8) {
                        Text(settings.localized("scam_detector_input_label"))
                            .font(.headline)

                        TextEditor(text: $inputText)
                            .frame(minHeight: 150)
                            .padding(8)
                            .scrollContentBackground(.hidden)
                            .background(Color.white.opacity(0.02))
                            .clipShape(RoundedRectangle(cornerRadius: 12))

                        HStack(spacing: 8) {
                            Button {
                                #if canImport(UIKit)
                                if let clip = UIPasteboard.general.string, !clip.isEmpty {
                                    inputText += clip
                                }
                                #endif
                            } label: {
                                Image(systemName: "doc.on.clipboard")
                                    .font(.system(size: 18, weight: .semibold))
                                    .frame(width: 44, height: 44)
                            }
                            .featureActionIconButtonStyle()

                            if enableVision {
                                PhotosPicker(selection: $selectedImageItem, matching: .images) {
                                    Image(systemName: "photo")
                                        .font(.system(size: 18, weight: .semibold))
                                        .frame(width: 44, height: 44)
                                }
                                .foregroundStyle(.white)
                                .background(
                                    RoundedRectangle(cornerRadius: 10)
                                        .fill(Color.white.opacity(0.08))
                                )
                                .overlay(
                                    RoundedRectangle(cornerRadius: 10)
                                        .stroke(Color.white.opacity(0.16), lineWidth: 1)
                                )
                            }

                            Button {
                                ttsManager.toggleSpeaking(
                                    getDisplayContentWithoutThinking(outputText),
                                    fallbackLanguage: settings.selectedLanguage,
                                    key: "scam-detector-output"
                                )
                            } label: {
                                Image(systemName: ttsManager.isSpeaking(key: "scam-detector-output") ? "stop.fill" : "speaker.wave.2")
                                    .font(.system(size: 18, weight: .semibold))
                                    .frame(width: 44, height: 44)
                            }
                            .featureActionIconButtonStyle()
                            .disabled(outputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)

                            Button {
                                #if canImport(UIKit)
                                UIPasteboard.general.string = getDisplayContentWithoutThinking(outputText)
                                #endif
                            } label: {
                                Image(systemName: "doc.on.doc")
                                    .font(.system(size: 18, weight: .semibold))
                                    .frame(width: 44, height: 44)
                            }
                            .featureActionIconButtonStyle()
                            .disabled(outputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)

                            Spacer()
                        }

                                if enableVision,
                                    let selectedImageURL,
                           let uiImage = UIImage(contentsOfFile: selectedImageURL.path) {
                            ZStack(alignment: .topTrailing) {
                                Image(uiImage: uiImage)
                                    .resizable()
                                    .scaledToFit()
                                    .frame(maxHeight: 180)
                                    .frame(maxWidth: .infinity)
                                    .clipShape(RoundedRectangle(cornerRadius: 10))
                                Button {
                                    self.selectedImageURL = nil
                                    self.selectedImageItem = nil
                                } label: {
                                    Image(systemName: "xmark.circle.fill")
                                        .font(.title3)
                                        .foregroundStyle(.white, .black.opacity(0.5))
                                }
                                .padding(8)
                            }
                        }
                    }
                    .padding(.horizontal)
                    .padding(.vertical, 12)
                    .background(.ultraThinMaterial)
                    .clipShape(RoundedRectangle(cornerRadius: 16))
                    .overlay(
                        RoundedRectangle(cornerRadius: 16)
                            .stroke(Color.white.opacity(0.14), lineWidth: 1)
                    )
                    .padding(.horizontal)

                    if isFetchingURL {
                        HStack(spacing: 8) {
                            ProgressView()
                            Text(settings.localized("scam_detector_fetching_url"))
                                .font(.subheadline)
                                .foregroundStyle(.white.opacity(0.68))
                            Spacer()
                        }
                        .padding(.horizontal)
                    }

                    ScrollViewReader { scrollProxy in
                        ScrollView {
                            VStack(alignment: .leading, spacing: 8) {
                                Text(settings.localized("scam_detector_result"))
                                    .font(.headline)
                                if outputText.isEmpty {
                                    Text("-")
                                        .textSelection(.enabled)
                                        .frame(maxWidth: .infinity, alignment: .leading)
                                        .frame(minHeight: 140, alignment: .topLeading)
                                        .padding(10)
                                        .background(.ultraThinMaterial)
                                        .clipShape(RoundedRectangle(cornerRadius: 12))
                                } else {
                                    ThinkingAwareResultContent(
                                        content: outputText,
                                        isGenerating: isAnalyzing,
                                        preferThinkingWhileStreaming: enableThinking
                                            && (selectedFeatureModel(named: selectedModelName)?.supportsThinking == true)
                                            && supportsUnmarkedStreamingThinkingHeuristic(forModelNamed: selectedModelName)
                                    )
                                    .frame(minHeight: 140, alignment: .topLeading)
                                    .padding(10)
                                    .background(.ultraThinMaterial)
                                    .clipShape(RoundedRectangle(cornerRadius: 12))
                                }
                                if !outputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                                    HStack {
                                        Spacer()
                                        Button {
                                            #if canImport(UIKit)
                                            UIPasteboard.general.string = getDisplayContentWithoutThinking(outputText)
                                            #endif
                                        } label: {
                                            Image(systemName: "doc.on.doc")
                                                .font(.system(size: 14, weight: .semibold))
                                                .foregroundColor(.white.opacity(0.6))
                                                .padding(8)
                                                .background(Color.white.opacity(0.08))
                                                .clipShape(RoundedRectangle(cornerRadius: 8))
                                        }
                                    }
                                }
                                Color.clear.frame(height: 1).id("scam_detector_bottom")
                            }
                            .padding(.horizontal)
                        }
                        .onChange(of: outputText) { _, _ in
                            if isAnalyzing {
                                withAnimation { scrollProxy.scrollTo("scam_detector_bottom", anchor: .bottom) }
                            }
                        }
                    }

                    if let errorMessage {
                        Text(errorMessage)
                            .foregroundColor(.red)
                            .font(.caption)
                            .padding(.horizontal)
                    }

                    Spacer(minLength: 0)

                    Button {
                        toggleAnalyze()
                    } label: {
                        HStack(spacing: 8) {
                            if isAnalyzing {
                                ProgressView()
                                    .tint(.white)
                                    .scaleEffect(0.85)
                            } else {
                                Image(systemName: "shield.lefthalf.filled")
                                    .font(.system(size: 12, weight: .bold))
                            }
                            Text(settings.localized("scam_detector_analyze"))
                                .lineLimit(1)
                        }
                        .frame(maxWidth: .infinity)
                        .frame(height: 52)
                    }
                    .foregroundStyle(.white)
                    .liquidGlassPrimaryButton(cornerRadius: 12)
                    .disabled(isLoading)
                    .padding(.horizontal)
                    .padding(.bottom, 8)
                }
            }
        }
        .navigationTitle(settings.localized("scam_detector_title"))
        .navigationBarTitleDisplayMode(.inline)
        .apolloScreenBackground()
        .safeAreaInset(edge: .bottom, spacing: 0) { BannerAdContainer() }
        .simultaneousGesture(TapGesture().onEnded { dismissKeyboard() })
        .toolbarBackground(.hidden, for: .navigationBar)
        .toolbar {
            ToolbarItem(placement: .navigationBarLeading) {
                Button {
                    generationTask?.cancel()
                    llm.unloadModel()
                    onNavigateBack()
                } label: {
                    Image(systemName: "arrow.left")
                }
            }
            ToolbarItem(placement: .navigationBarTrailing) {
                Button { showSettings = true } label: { Image(systemName: "slider.horizontal.3") }
            }
        }
        .sheet(isPresented: $showSettings) {
            FeatureModelSettingsSheet(
                selectedModelName: $selectedModelName,
                maxTokens: $maxTokens,
                enableThinking: $enableThinking,
                enableVision: $enableVision,
                enableAudio: nil,
                isLoading: $isLoading,
                errorMessage: $errorMessage,
                supportsVisionToggle: true,
                visionToggleTitleKey: "scam_detector_enable_vision",
                audioToggleTitleKey: nil,
                visionAvailableCheck: hasDownloadedVisionProjector,
                writingMode: nil,
                modelFilter: isNonTranslatorFeatureModel,
                onLoad: { await ensureModelLoaded(force: false) },
                onUnload: { llm.unloadModel() }
            )
        }
        .onAppear {
            Task {
                await syncRunAnywhereModelDiscovery()
                let available = downloadableFeatureModels().filter(isNonTranslatorFeatureModel)
                if selectedModelName.isEmpty || !available.contains(where: { $0.name == selectedModelName }) {
                    selectedModelName = available.first?.name ?? ""
                }
            }
        }
        .onChange(of: selectedImageItem) { _, item in
            guard let item else { selectedImageURL = nil; return }
            Task {
                if let sourceURL = try? await item.loadTransferable(type: URL.self) {
                    selectedImageURL = sourceURL
                    return
                }
                if let data = try? await item.loadTransferable(type: Data.self) {
                    let temp = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString).appendingPathExtension("jpg")
                    try? data.write(to: temp)
                    selectedImageURL = temp
                }
            }
        }
        .onChange(of: enableVision) { _, isEnabled in
            if !isEnabled {
                selectedImageItem = nil
                selectedImageURL = nil
            }
        }
        .onDisappear {
            generationTask?.cancel()
            llm.unloadModel()
        }
    }

    private func buildAnalysisPrompt(content: String, hasImage: Bool) -> String {
        if hasImage && !content.isEmpty {
            return """
            You are a scam detection expert. Analyze BOTH the provided image AND the text content below for potential scams, fraud, phishing attempts, or suspicious activity.

            **Text content to analyze:**
            \(content)

            **Instructions:**
            - Carefully examine the image for any suspicious elements, fake logos, misleading graphics, or scam indicators
            - Cross-reference the text content with what's shown in the image
            - Look for inconsistencies between the image and text
            - Check if the image appears to be a screenshot of a phishing message, fake website, or fraudulent offer

            Please provide a comprehensive analysis covering:
            1. **Risk Level**: Low, Medium, High, or Critical
            2. **Red Flags in Image**: List any suspicious visual elements (fake logos, poor quality graphics, misleading layouts, etc.)
            3. **Red Flags in Text**: List any suspicious text elements (urgency tactics, too-good-to-be-true offers, suspicious links, impersonation, poor grammar, etc.)
            4. **Consistency Check**: Do the image and text align? Are there contradictions?
            5. **Legitimacy Indicators**: Any signs suggesting it might be legitimate
            6. **Verdict**: Is this likely a scam? Explain your reasoning based on BOTH the image and text.
            7. **Recommendations**: What should the user do?

            Be thorough and specific in your analysis. If you detect a scam, clearly state it. If it appears legitimate, explain why.
            """
        }

        if hasImage {
            return """
            You are a scam detection expert. Analyze the provided image for potential scams, fraud, phishing attempts, or suspicious activity.

            **Instructions:**
            - Carefully examine the image for any suspicious elements, fake logos, misleading graphics, or scam indicators
            - Check if the image appears to be a screenshot of a phishing message, fake website, or fraudulent offer
            - Look for common scam tactics in the visual content

            Please provide a comprehensive analysis covering:
            1. **Risk Level**: Low, Medium, High, or Critical
            2. **Visual Red Flags**: List any suspicious elements in the image (fake logos, poor quality graphics, misleading layouts, urgency messages, too-good-to-be-true offers, etc.)
            3. **Legitimacy Indicators**: Any visual signs suggesting it might be legitimate
            4. **Verdict**: Is this likely a scam? Explain your reasoning based on the image.
            5. **Recommendations**: What should the user do?

            Be thorough and specific in your analysis. If you detect a scam, clearly state it. If it appears legitimate, explain why.
            """
        }

        return """
        You are a scam detection expert. Analyze the following content for potential scams, fraud, phishing attempts, or suspicious activity.

        IMPORTANT: Respond in the same language as the input content. Match the language of the content in the image.

        Content to analyze:
        \(content)

        Please provide a comprehensive analysis covering:
        1. **Risk Level**: Low, Medium, High, or Critical
        2. **Red Flags**: List any suspicious elements (urgency tactics, too-good-to-be-true offers, suspicious links, impersonation, poor grammar, etc.)
        3. **Legitimacy Indicators**: Any signs suggesting it might be legitimate
        4. **Verdict**: Is this likely a scam? Explain your reasoning.
        5. **Recommendations**: What should the user do?

        Be thorough and specific in your analysis. If you detect a scam, clearly state it. If it appears legitimate, explain why.
        """
    }

    private func detectFirstURL(in text: String) -> String? {
        let pattern = #"https?://[^\s]+"#
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return nil }
        let range = NSRange(text.startIndex..<text.endIndex, in: text)
        guard let match = regex.firstMatch(in: text, range: range),
              let urlRange = Range(match.range, in: text) else { return nil }
        return String(text[urlRange])
    }

    private func extractTextFromHTML(_ html: String) -> String {
        var cleaned = html.replacingOccurrences(of: #"<script[^>]*>.*?</script>"#, with: "", options: [.regularExpression, .caseInsensitive])
        cleaned = cleaned.replacingOccurrences(of: #"<style[^>]*>.*?</style>"#, with: "", options: [.regularExpression, .caseInsensitive])
        cleaned = cleaned.replacingOccurrences(of: #"<[^>]*>"#, with: " ", options: .regularExpression)
        cleaned = cleaned
            .replacingOccurrences(of: "&amp;", with: "&")
            .replacingOccurrences(of: "&lt;", with: "<")
            .replacingOccurrences(of: "&gt;", with: ">")
            .replacingOccurrences(of: "&quot;", with: "\"")
            .replacingOccurrences(of: "&#39;", with: "'")
            .replacingOccurrences(of: "&nbsp;", with: " ")
            .replacingOccurrences(of: #"\s+"#, with: " ", options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return String(cleaned.prefix(3000))
    }

    private func fetchURLContent(_ urlString: String) async -> String {
        guard let url = URL(string: urlString) else { return "" }
        var request = URLRequest(url: url)
        request.timeoutInterval = 15
        request.setValue("Mozilla/5.0", forHTTPHeaderField: "User-Agent")
        do {
            let (data, _) = try await URLSession.shared.data(for: request)
            let html = String(data: data, encoding: .utf8) ?? ""
            return extractTextFromHTML(html)
        } catch {
            return ""
        }
    }

    private func ensureModelLoaded(force: Bool) async {
        guard let model = selectedFeatureModel(named: selectedModelName) else {
            errorMessage = settings.localized("scam_detector_no_model")
            return
        }
        isLoading = true
        defer { isLoading = false }

        let modelContextCap = model.contextWindowSize > 0 ? model.contextWindowSize : 4096
        let effectiveContext = min(max(1, Int(maxTokens)), modelContextCap)
        let shouldReload = force
            || llm.currentlyLoadedModel != model.name
            || llm.loadedContextWindow != effectiveContext

        llm.maxTokens = min(Int(maxTokens), effectiveContext)
        llm.contextWindow = effectiveContext
        llm.enableVision = enableVision
        llm.enableAudio = false
        llm.enableThinking = enableThinking

        do {
            if shouldReload {
                try await llm.loadModel(model)
            }
            errorMessage = nil
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    private func toggleAnalyze() {
        dismissKeyboard()

        if isAnalyzing {
            generationTask?.cancel()
            generationTask = nil
            isAnalyzing = false
            return
        }

        generationTask = Task {
            await ensureModelLoaded(force: false)
            guard llm.isLoaded else { return }

            isAnalyzing = true
            outputText = ""

            var contentToAnalyze = inputText.trimmingCharacters(in: .whitespacesAndNewlines)

            if let url = detectFirstURL(in: contentToAnalyze) {
                isFetchingURL = true
                let fetchedContent = await fetchURLContent(url)
                isFetchingURL = false
                if !fetchedContent.isEmpty {
                    let additionalContext = contentToAnalyze.replacingOccurrences(of: url, with: "").trimmingCharacters(in: .whitespacesAndNewlines)
                    contentToAnalyze = """
                    URL: \(url)

                    Content from URL:
                    \(fetchedContent)

                    \(!additionalContext.isEmpty ? "Additional context: \(additionalContext)" : "")
                    """
                }
            }

            if contentToAnalyze.isEmpty && selectedImageURL == nil {
                errorMessage = settings.localized("scam_detector_input_hint")
                isAnalyzing = false
                generationTask = nil
                return
            }

            do {
                let hasImage = selectedImageURL != nil && enableVision
                let prompt = buildAnalysisPrompt(content: contentToAnalyze, hasImage: hasImage)
                let effectiveImageURL = enableVision ? selectedImageURL : nil
                try await llm.generate(prompt: prompt, imageURL: effectiveImageURL) { text, _, _ in
                    Task { @MainActor in
                        outputText = sanitizeModelOutputText(text)
                    }
                }
            } catch {
                errorMessage = error.localizedDescription
            }

            isAnalyzing = false
            isFetchingURL = false
            generationTask = nil
        }
    }
}

private struct VibeChatMessage: Identifiable, Codable, Equatable {
    var id: UUID = UUID()
    let role: String
    var text: String
}

private struct VibeChatSession: Identifiable, Codable, Equatable {
    var id: UUID = UUID()
    var title: String
    var messages: [VibeChatMessage] = []
}

private struct VibeChatSessionChip: View {
    let title: String
    let isSelected: Bool
    let canDelete: Bool
    let onSelect: () -> Void
    let onDelete: () -> Void

    var body: some View {
        HStack(spacing: 6) {
            Button(action: onSelect) {
                Text(title)
                    .font(.caption.weight(.semibold))
                    .lineLimit(1)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 6)
            }

            Button(action: onDelete) {
                Image(systemName: "xmark")
                    .font(.system(size: 10, weight: .bold))
                    .frame(width: 20, height: 20)
            }
            .disabled(!canDelete)
        }
        .foregroundStyle(.white)
        .background(
            Capsule(style: .continuous)
                .fill(isSelected ? Color.white.opacity(0.16) : Color.white.opacity(0.10))
        )
        .overlay(
            Capsule(style: .continuous)
                .stroke(Color.white.opacity(isSelected ? 0.22 : 0.14), lineWidth: 1)
        )
    }
}

private struct VibeFileChip: View {
    let title: String
    let isSelected: Bool
    let canDelete: Bool
    let onSelect: () -> Void
    let onDelete: () -> Void

    var body: some View {
        HStack(spacing: 6) {
            Button(action: onSelect) {
                Text(title)
                    .font(.caption.weight(.semibold))
                    .lineLimit(1)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 6)
            }

            Button(action: onDelete) {
                Image(systemName: "xmark")
                    .font(.system(size: 10, weight: .bold))
                    .frame(width: 20, height: 20)
            }
            .disabled(!canDelete)
        }
        .foregroundStyle(.white)
        .background(
            Capsule(style: .continuous)
                .fill(isSelected ? Color.white.opacity(0.16) : Color.white.opacity(0.10))
        )
        .overlay(
            Capsule(style: .continuous)
                .stroke(Color.white.opacity(isSelected ? 0.22 : 0.14), lineWidth: 1)
        )
    }
}

private struct WorkspaceFilesSheet: View {
    @EnvironmentObject var settings: AppSettings
    @Environment(\.dismiss) private var dismiss

    let folderURL: URL?
    let onPickFolder: () -> Void
    let onOpenFile: (URL) -> Void

    @State private var files: [URL] = []

    var body: some View {
        NavigationView {
            ZStack {
                ApolloLiquidBackground()

                List {
                    Section {
                        Button {
                            dismiss()
                            onPickFolder()
                        } label: {
                            Label("Select different folder", systemImage: "folder.badge.plus")
                        }
                    }

                    Section("Files") {
                        if files.isEmpty {
                            Text("No files found in this folder.")
                                .foregroundStyle(.secondary)
                        } else {
                            ForEach(files, id: \.path) { url in
                                Button {
                                    onOpenFile(url)
                                } label: {
                                    HStack {
                                        Image(systemName: "doc.text")
                                        Text(url.lastPathComponent)
                                            .lineLimit(1)
                                        Spacer()
                                    }
                                }
                            }
                        }
                    }
                }
                .scrollContentBackground(.hidden)
            }
            .navigationTitle("Files")
            .navigationBarTitleDisplayMode(.inline)
            .toolbarBackground(.hidden, for: .navigationBar)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button(settings.localized("done")) { dismiss() }
                }
            }
            .task {
                refreshFiles()
            }
        }
    }

    private func refreshFiles() {
        guard let folderURL else {
            files = []
            return
        }

        #if canImport(UIKit)
        let didStart = folderURL.startAccessingSecurityScopedResource()
        defer {
            if didStart {
                folderURL.stopAccessingSecurityScopedResource()
            }
        }
        #endif

        let allowedExtensions: Set<String> = [
            "txt", "md",
            "py", "js", "ts", "java", "kt", "go", "rs", "cpp", "cc", "cxx", "c", "cs", "swift",
            "html", "htm", "css", "php", "rb", "lua", "sh", "bash", "zsh", "sql",
            "json", "yaml", "yml"
        ]

        let enumerator = FileManager.default.enumerator(at: folderURL, includingPropertiesForKeys: [.isRegularFileKey], options: [.skipsHiddenFiles, .skipsPackageDescendants])
        var collected: [URL] = []
        while let url = enumerator?.nextObject() as? URL {
            let ext = url.pathExtension.lowercased()
            if allowedExtensions.contains(ext) {
                collected.append(url)
            }
        }
        files = collected.sorted(by: { $0.lastPathComponent.lowercased() < $1.lastPathComponent.lowercased() })
    }
}

struct VibeCoderScreen: View {
    @EnvironmentObject var settings: AppSettings
    @ObservedObject private var ttsManager = OnDeviceTtsManager.shared
    @AppStorage("feature_vibecoder_model_name") private var selectedModelName: String = ""
    @AppStorage("feature_vibecoder_max_tokens") private var maxTokens: Double = 2048
    @AppStorage("feature_vibecoder_enable_thinking") private var enableThinking: Bool = true
    @AppStorage("feature_vibecoder_folder_bookmark") private var folderBookmark: Data = Data()
    @AppStorage("feature_vibecoder_chat_sessions_data") private var chatSessionsData: Data = Data()
    @AppStorage("feature_vibecoder_active_chat_session_id") private var activeChatSessionIdRaw: String = ""
    @AppStorage("feature_vibecoder_current_file_relative_path") private var currentFileRelativePath: String = ""
    @State private var generatedCode: String = ""
    @State private var chatInput: String = ""
    @State private var chatSessions: [VibeChatSession] = [VibeChatSession(title: "Chat 1")]
    @State private var activeChatSessionId: UUID = UUID()
    @State private var workspaceFolderURL: URL?
    @State private var currentFileURL: URL?
    @State private var currentFileName: String?
    @State private var showWorkspaceFolderPicker = false
    @State private var showCreateFileDialog = false
    @State private var newFileNameInput = ""
    @State private var isLoading = false
    @State private var isGenerating = false
    @State private var showSettings = false
    @State private var errorMessage: String?
    @State private var generationTask: Task<Void, Never>?
    @State private var streamTick: Int = 0
    @State private var lastStreamTickTime: Double = 0
    @State private var debouncedAutosaveTask: Task<Void, Never>?
    @State private var workspaceFiles: [URL] = []
    @State private var pendingDeleteChatId: UUID?
    @State private var pendingDeleteFileURL: URL?

    private enum VibeFocusField: Hashable {
        case chat
        case editor
    }

    @FocusState private var focusedField: VibeFocusField?

    let onNavigateBack: () -> Void

    @ObservedObject private var llm = LLMBackend.shared

    private var isCurrentModelLoaded: Bool {
        llm.isLoaded && llm.currentlyLoadedModel == selectedModelName
    }

    private var hasFileSession: Bool {
        !(currentFileName ?? "").isEmpty
    }

    private var hasWorkspaceFolder: Bool {
        workspaceFolderURL != nil
    }

    private var sendButtonIconName: String {
        isGenerating ? "xmark" : "paperplane.fill"
    }

    private var isSendButtonDisabled: Bool {
        (!isGenerating && chatInput.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty) || !hasFileSession
    }

    private var activeMessages: [VibeChatMessage] {
        guard let idx = chatSessions.firstIndex(where: { $0.id == activeChatSessionId }) else {
            return chatSessions.first?.messages ?? []
        }
        return chatSessions[idx].messages
    }

    private var contextWindowCap: Double {
        let configuredCap = Double(max(1, llm.contextWindow))
        if let loadedContextWindow = llm.loadedContextWindow {
            return Double(max(1, loadedContextWindow))
        }
        return configuredCap
    }

    private var contextBudgetForRing: Double {
        contextWindowCap
    }

    private var approximateContextTokensUsed: Double {
        // Strip thinking blocks from assistant messages — only answer text is re-sent
        // in the multi-turn prompt, so thinking should not count toward the budget.
        // During the streaming thinking phase (no answer yet), count 0 for that message.
        let activeTextChars = activeMessages.reduce(0) { acc, msg in
            if msg.role == "user" {
                return acc + msg.text.count
            }
            if contentHasThinkingMarkers(msg.text) {
                let answer = getDisplayContentWithoutThinking(msg.text)
                return acc + answer.count     // 0 while still thinking
            }
            return acc + msg.text.count
        }
        let codeChars = generatedCode.count
        let inputChars = chatInput.count
        let totalChars = activeTextChars + codeChars + inputChars
        let estimatedTokens = Double(totalChars) / 4.0
        return max(0, estimatedTokens)
    }

    private var contextUsageFractionRaw: Double {
        guard contextBudgetForRing > 0 else { return 0 }
        return min(max(approximateContextTokensUsed / contextBudgetForRing, 0), 1)
    }

    private var contextUsageFractionDisplay: Double {
        if approximateContextTokensUsed <= 0 {
            return 0
        }
        return min(max(contextUsageFractionRaw, 0.02), 1)
    }

    private var contextUsageLabel: String {
        if approximateContextTokensUsed > 0 {
            return "\(max(1, Int((contextUsageFractionRaw * 100).rounded())))%"
        }
        return "0%"
    }

    private var isContextBudgetExceededForSession: Bool {
        contextUsageFractionRaw >= 0.995
    }

    private var deleteChatAlertBinding: Binding<Bool> {
        Binding(
            get: { pendingDeleteChatId != nil },
            set: { if !$0 { pendingDeleteChatId = nil } }
        )
    }

    private var deleteFileAlertBinding: Binding<Bool> {
        Binding(
            get: { pendingDeleteFileURL != nil },
            set: { if !$0 { pendingDeleteFileURL = nil } }
        )
    }

    var body: some View {
        Group {
            if !isCurrentModelLoaded {
                VStack(spacing: 12) {
                    Image(systemName: "curlybraces.square")
                        .font(.system(size: 48, weight: .semibold))
                        .foregroundStyle(.secondary)
                    Text(settings.localized("scam_detector_load_model"))
                        .font(.title3.weight(.bold))
                    Text(settings.localized("scam_detector_load_model_desc"))
                        .font(.subheadline)
                        .foregroundStyle(.white.opacity(0.7))
                        .multilineTextAlignment(.center)
                        .padding(.horizontal)
                    Button {
                        showSettings = true
                    } label: {
                        HStack {
                            Spacer()
                            Text(settings.localized("feature_settings_title"))
                            Spacer()
                        }
                        .frame(height: 50)
                        .contentShape(Rectangle())
                    }
                    .frame(maxWidth: 260)
                    .liquidGlassPrimaryButton(cornerRadius: 12)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                if !hasWorkspaceFolder {
                    VStack(spacing: 12) {
                        Image(systemName: "folder.fill")
                            .font(.system(size: 48, weight: .semibold))
                            .foregroundStyle(.secondary)

                        Text(settings.localized("vibe_coder_select_folder_title"))
                            .font(.title3.weight(.bold))
                            .multilineTextAlignment(.center)

                        Text(settings.localized("vibe_coder_select_folder_desc"))
                            .font(.subheadline)
                            .foregroundStyle(.white.opacity(0.7))
                            .multilineTextAlignment(.center)
                            .padding(.horizontal)

                        Button {
                            showWorkspaceFolderPicker = true
                        } label: {
                            HStack {
                                Spacer()
                                Text(settings.localized("vibe_coder_open_folder"))
                                Spacer()
                            }
                            .frame(height: 50)
                            .contentShape(Rectangle())
                        }
                        .frame(maxWidth: 260)
                        .liquidGlassPrimaryButton(cornerRadius: 12)
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else {
                    VStack(spacing: 12) {
                        GeometryReader { _ in
                            let isLandscape = UIScreen.main.bounds.width > UIScreen.main.bounds.height

                            let chatPanel = AnyView(
                                VStack(alignment: .leading, spacing: 10) {
                                    HStack {
                                        Text(settings.localized("vibe_coder_ai_chat"))
                                            .font(.headline)
                                        Spacer()

                                        ZStack {
                                            Circle()
                                                .stroke(Color.white.opacity(0.18), lineWidth: 2)
                                            Circle()
                                                .trim(from: 0, to: contextUsageFractionDisplay)
                                                .stroke(
                                                    contextUsageFractionRaw < 0.90 ? ApolloPalette.accentStrong : ApolloPalette.warning,
                                                    style: StrokeStyle(lineWidth: 2.5, lineCap: .round)
                                                )
                                                .rotationEffect(.degrees(-90))

                                            Text(contextUsageFractionRaw < 0.995 ? contextUsageLabel : "!")
                                                .font(.system(size: 8, weight: .bold, design: .rounded))
                                        }
                                        .frame(width: 28, height: 28)
                                        .accessibilityLabel("Context usage \(contextUsageLabel)")

                                        Button {
                                            clearActiveChat()
                                        } label: {
                                            Image(systemName: "trash")
                                                .font(.system(size: 16, weight: .semibold))
                                                .frame(width: 36, height: 36)
                                        }
                                        .disabled(activeMessages.isEmpty || isGenerating)

                                        Button {
                                            createNewChatSession()
                                        } label: {
                                            Image(systemName: "plus")
                                                .font(.system(size: 16, weight: .semibold))
                                                .frame(width: 36, height: 36)
                                        }
                                        .disabled(isGenerating)
                                    }
                                    .padding(.horizontal)

                                    ScrollView(.horizontal, showsIndicators: false) {
                                        HStack(spacing: 8) {
                                            ForEach(chatSessions) { session in
                                                VibeChatSessionChip(
                                                    title: session.title,
                                                    isSelected: session.id == activeChatSessionId,
                                                    canDelete: chatSessions.count > 1 && !isGenerating,
                                                    onSelect: { activeChatSessionId = session.id },
                                                    onDelete: { pendingDeleteChatId = session.id }
                                                )
                                            }
                                        }
                                        .padding(.horizontal)
                                    }

                                    ScrollViewReader { proxy in
                                        ScrollView {
                                            VStack(alignment: .leading, spacing: 8) {
                                                ForEach(activeMessages) { message in
                                                    VStack(alignment: .leading, spacing: 4) {
                                                        HStack(spacing: 8) {
                                                            Text(message.role == "user" ? settings.localized("vibe_coder_message_you") : settings.localized("vibe_coder_message_ai"))
                                                                .font(.caption.weight(.semibold))
                                                                .foregroundStyle(.white.opacity(0.65))
                                                            Spacer()

                                                        }
                                                        if message.role == "user" {
                                                            Text(message.text)
                                                                .textSelection(.enabled)
                                                                .frame(maxWidth: .infinity, alignment: .leading)
                                                                .padding(8)
                                                                .background(.ultraThinMaterial)
                                                                .clipShape(RoundedRectangle(cornerRadius: 10))
                                                        } else {
                                                            ThinkingAwareResultContent(
                                                                content: message.text,
                                                                isGenerating: isGenerating && message.id == activeMessages.last?.id,
                                                                preferThinkingWhileStreaming: enableThinking
                                                                    && (selectedFeatureModel(named: selectedModelName)?.supportsThinking == true)
                                                                    && supportsUnmarkedStreamingThinkingHeuristic(forModelNamed: selectedModelName)
                                                            )
                                                            .padding(8)
                                                            .background(.ultraThinMaterial)
                                                            .clipShape(RoundedRectangle(cornerRadius: 10))
                                                        }
                                                    }
                                                    .id(message.id)
                                                }
                                            }
                                            .padding(.horizontal)
                                            .padding(.top, 6)
                                        }
                                        .frame(minHeight: 160)
                                        .onChange(of: activeMessages.count) { _, _ in
                                            if let last = activeMessages.last {
                                                withAnimation(.linear(duration: 0.08)) {
                                                    proxy.scrollTo(last.id, anchor: .bottom)
                                                }
                                            }
                                        }
                                        .onChange(of: streamTick) { _, _ in
                                            if let last = activeMessages.last {
                                                proxy.scrollTo(last.id, anchor: .bottom)
                                            }
                                        }
                                    }

                                    HStack(spacing: 10) {
                                        TextField(
                                            hasFileSession ? settings.localized("vibe_coder_ask_ai_edit") : settings.localized("vibe_coder_create_open_file_hint"),
                                            text: $chatInput,
                                            axis: .vertical
                                        )
                                        .lineLimit(1...5)
                                        .focused($focusedField, equals: .chat)
                                        .disabled(!hasFileSession || isGenerating)
                                        .padding(.vertical, 10)
                                        .padding(.horizontal, 12)
                                        .background(Color.white.opacity(0.05))
                                        .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))

                                        Button {
                                            if isGenerating {
                                                stopGeneration()
                                            } else {
                                                sendChat()
                                            }
                                        } label: {
                                            Image(systemName: sendButtonIconName)
                                                .font(.system(size: 16, weight: .semibold))
                                                .frame(width: 44, height: 44)
                                        }
                                        .foregroundStyle(.white)
                                        .background(
                                            RoundedRectangle(cornerRadius: 12)
                                                .fill(Color.white.opacity(0.08))
                                        )
                                        .overlay(
                                            RoundedRectangle(cornerRadius: 12)
                                                .stroke(Color.white.opacity(0.16), lineWidth: 1)
                                        )
                                        .disabled(isSendButtonDisabled)
                                    }
                                    .padding(.horizontal)
                                    .padding(.bottom, 8)
                                }
                                .padding(.vertical, 10)
                                .background(.ultraThinMaterial)
                                .clipShape(RoundedRectangle(cornerRadius: 16))
                                .overlay(
                                    RoundedRectangle(cornerRadius: 16)
                                        .stroke(Color.white.opacity(0.14), lineWidth: 1)
                                )
                                .padding(.horizontal)
                            )

                            let editorPanel = AnyView(
                                VStack(alignment: .leading, spacing: 8) {
                                    HStack {
                                        Text(currentFileName ?? settings.localized("vibe_coder_open_or_create_file"))
                                            .font(.subheadline.weight(.semibold))
                                            .foregroundStyle(.white.opacity(0.72))
                                        Spacer()

                                        Button {
                                            showWorkspaceFolderPicker = true
                                        } label: {
                                            Image(systemName: "folder")
                                                .font(.system(size: 16, weight: .semibold))
                                                .frame(width: 36, height: 36)
                                        }

                                        Button {
                                            showCreateFileDialog = true
                                        } label: {
                                            Image(systemName: "plus")
                                                .font(.system(size: 16, weight: .semibold))
                                                .frame(width: 36, height: 36)
                                        }

                                        Button {
                                            #if canImport(UIKit)
                                            UIPasteboard.general.string = generatedCode
                                            #endif
                                        } label: {
                                            Image(systemName: "doc.on.doc")
                                                .font(.system(size: 16, weight: .semibold))
                                                .frame(width: 36, height: 36)
                                        }
                                        .disabled(generatedCode.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)

                                        Button {
                                            saveCurrentFile()
                                        } label: {
                                            Image(systemName: "square.and.arrow.down")
                                                .font(.system(size: 16, weight: .semibold))
                                                .frame(width: 36, height: 36)
                                        }
                                        .disabled(!hasFileSession)

                                        if isHTMLFile {
                                            Button {
                                                openHTMLPreviewInSafari()
                                            } label: {
                                                Image(systemName: "safari")
                                                    .font(.system(size: 16, weight: .semibold))
                                                    .frame(width: 36, height: 36)
                                            }
                                            .disabled(generatedCode.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                                        }
                                    }

                                    ScrollView(.horizontal, showsIndicators: false) {
                                        HStack(spacing: 8) {
                                            ForEach(workspaceFiles, id: \.path) { url in
                                                VibeFileChip(
                                                    title: url.lastPathComponent,
                                                    isSelected: url == currentFileURL,
                                                    canDelete: !isGenerating,
                                                    onSelect: { openFile(url) },
                                                    onDelete: { pendingDeleteFileURL = url }
                                                )
                                            }
                                        }
                                    }

                                    TextEditor(text: $generatedCode)
                                        .font(.system(.body, design: .monospaced))
                                        .focused($focusedField, equals: .editor)
                                        .frame(minHeight: 220)
                                        .padding(8)
                                        .scrollContentBackground(.hidden)
                                        .background(Color.white.opacity(0.02))
                                        .clipShape(RoundedRectangle(cornerRadius: 12))
                                }
                                .padding(.horizontal)
                                .padding(.vertical, 10)
                                .background(.ultraThinMaterial)
                                .clipShape(RoundedRectangle(cornerRadius: 16))
                                .overlay(
                                    RoundedRectangle(cornerRadius: 16)
                                        .stroke(Color.white.opacity(0.14), lineWidth: 1)
                                )
                                .padding(.horizontal)
                            )

                            Group {
                                if isLandscape {
                                    HStack(spacing: 12) {
                                        editorPanel
                                            .frame(maxWidth: .infinity, maxHeight: .infinity)
                                        chatPanel
                                            .frame(maxWidth: .infinity, maxHeight: .infinity)
                                    }
                                } else {
                                    VStack(spacing: 12) {
                                        chatPanel
                                        editorPanel
                                    }
                                }
                            }
                        }

                        if let errorMessage {
                            Text(errorMessage)
                                .foregroundColor(.red)
                                .font(.caption)
                                .padding(.horizontal)
                        }
                    }
                }
            }
        }
        .navigationTitle(settings.localized("vibe_coder_title"))
        .navigationBarTitleDisplayMode(.inline)
        .apolloScreenBackground()
        .safeAreaInset(edge: .bottom, spacing: 0) { BannerAdContainer() }
        .ignoresSafeArea(.keyboard, edges: .bottom)
        .toolbar {
            ToolbarItemGroup(placement: .keyboard) {
                Spacer()
                Button(settings.localized("done")) {
                    focusedField = nil
                    dismissKeyboard()
                }
            }
        }
        .toolbarBackground(.hidden, for: .navigationBar)
        .toolbar {
            ToolbarItem(placement: .navigationBarLeading) {
                Button {
                    stopGeneration()
                    llm.unloadModel()
                    onNavigateBack()
                } label: {
                    Image(systemName: "arrow.left")
                }
            }
            ToolbarItem(placement: .navigationBarTrailing) {
                Button { showSettings = true } label: { Image(systemName: "slider.horizontal.3") }
            }
        }
        .sheet(isPresented: $showSettings) {
            FeatureModelSettingsSheet(
                selectedModelName: $selectedModelName,
                maxTokens: $maxTokens,
                enableThinking: $enableThinking,
                enableVision: .constant(false),
                enableAudio: nil,
                isLoading: $isLoading,
                errorMessage: $errorMessage,
                supportsVisionToggle: false,
                visionToggleTitleKey: "scam_detector_enable_vision",
                audioToggleTitleKey: nil,
                visionAvailableCheck: nil,
                writingMode: nil,
                modelFilter: isNonTranslatorFeatureModel,
                onLoad: { await ensureModelLoaded(force: false) },
                onUnload: { llm.unloadModel() }
            )
        }
        .onAppear {
            restoreChatSessionsFromStorage()

            Task {
                await syncRunAnywhereModelDiscovery()
                let available = downloadableFeatureModels().filter(isNonTranslatorFeatureModel)
                let hasSelectedModelName = !selectedModelName.isEmpty
                let selectedModelExists = available.contains { model in
                    model.name == selectedModelName
                }
                if !hasSelectedModelName || !selectedModelExists {
                    selectedModelName = available.first?.name ?? ""
                }
            }

            if workspaceFolderURL == nil {
                restoreWorkspaceFolderFromBookmark()
            } else {
                refreshWorkspaceFiles()
            }

            if workspaceFolderURL != nil {
                restoreLastOpenedFileIfPossible()
            }

            if !chatSessions.contains(where: { $0.id == activeChatSessionId }) {
                activeChatSessionId = chatSessions.first?.id ?? activeChatSessionId
            }
        }
        .onChange(of: chatSessions) { _, _ in
            persistChatSessionsToStorage()
        }
        .onChange(of: activeChatSessionId) { _, _ in
            persistChatSessionsToStorage()
        }
        .onChange(of: workspaceFolderURL) { _, newValue in
            if newValue != nil {
                refreshWorkspaceFiles()
                restoreLastOpenedFileIfPossible()
            }
        }
        .onChange(of: currentFileURL) { _, _ in
            persistCurrentFileSelection()
        }
        .onChange(of: generatedCode) { _, _ in
            guard hasFileSession, !isGenerating else { return }
            debouncedAutosaveTask?.cancel()
            debouncedAutosaveTask = Task {
                try? await Task.sleep(nanoseconds: 450_000_000)
                if Task.isCancelled { return }
                await MainActor.run {
                    saveCurrentFile(silent: true)
                }
            }
        }
        .fileImporter(
            isPresented: $showWorkspaceFolderPicker,
            allowedContentTypes: [.folder],
            onCompletion: handleWorkspaceFolderImport
        )
        .alert(settings.localized("vibe_coder_create_file_title"), isPresented: $showCreateFileDialog) {
            TextField(settings.localized("vibe_coder_file_name_placeholder"), text: $newFileNameInput)
            Button(settings.localized("cancel"), role: .cancel) {}
            Button(settings.localized("vibe_coder_create")) {
                createNewFile()
            }
        }
        .alert(settings.localized("vibe_coder_delete_chat_title"), isPresented: deleteChatAlertBinding) {
            Button(settings.localized("cancel"), role: .cancel) { pendingDeleteChatId = nil }
            Button(settings.localized("delete"), role: .destructive) {
                if let id = pendingDeleteChatId {
                    deleteChatSession(id)
                }
            }
        } message: {
            Text(settings.localized("vibe_coder_delete_chat_message"))
        }
        .alert(settings.localized("vibe_coder_delete_file_title"), isPresented: deleteFileAlertBinding) {
            Button(settings.localized("cancel"), role: .cancel) { pendingDeleteFileURL = nil }
            Button(settings.localized("delete"), role: .destructive) {
                if let fileURL = pendingDeleteFileURL {
                    deleteFile(fileURL)
                }
            }
        } message: {
            Text(String(format: settings.localized("vibe_coder_delete_file_message"), pendingDeleteFileURL?.lastPathComponent ?? ""))
        }
        .onDisappear {
            stopGeneration()
            debouncedAutosaveTask?.cancel()
            debouncedAutosaveTask = nil
            llm.unloadModel()
            Task { await LocalHTMLPreviewServer.shared.stop() }
        }
    }

    private func normalizedExtension(_ fileName: String?) -> String? {
        guard let raw = fileName?.lowercased() else { return nil }
        let parts = raw.split(separator: ".")
        if parts.count < 2 { return nil }
        if parts.last == "txt", parts.count >= 3 {
            return String(parts[parts.count - 2])
        }
        return String(parts.last!)
    }

    private var isHTMLFile: Bool {
        let ext = normalizedExtension(currentFileName)
        return ext == "html" || ext == "htm"
    }

    private func handleWorkspaceFolderImport(_ result: Result<URL, Error>) {
        switch result {
        case .success(let url):
            setWorkspaceFolder(url)
        case .failure(let error):
            let message = error.localizedDescription
            errorMessage = message
        }
    }

    private func languagePromptConfig() -> (languageName: String, targetRule: String, fenceLanguage: String)? {
        switch normalizedExtension(currentFileName) {
        case "html", "htm", "css":
            return ("Web App (HTML/CSS/JS)", "Build a single self-contained HTML file with embedded CSS and JavaScript.", "html")
        case "py": return ("Python", "Build a runnable Python script using only standard library.", "python")
        case "js": return ("JavaScript", "Build a runnable JavaScript program (no TypeScript).", "javascript")
        case "ts": return ("TypeScript", "Build a runnable TypeScript program with clear types.", "typescript")
        case "c": return ("C", "Build a runnable C program with int main().", "c")
        case "php": return ("PHP", "Build a runnable PHP script.", "php")
        case "rb": return ("Ruby", "Build a runnable Ruby script.", "ruby")
        case "swift": return ("Swift", "Build a runnable Swift program.", "swift")
        case "dart": return ("Dart", "Build a runnable Dart program.", "dart")
        case "lua": return ("Lua", "Build a runnable Lua script.", "lua")
        case "sh", "bash", "zsh": return ("Shell", "Build a runnable POSIX shell script.", "sh")
        case "sql": return ("SQL", "Build valid SQL statements with clear schema assumptions.", "sql")
        case "java": return ("Java", "Build a runnable Java program with a main method.", "java")
        case "kt": return ("Kotlin", "Build a runnable Kotlin console program with a main function.", "kotlin")
        case "cs": return ("C#", "Build a runnable C# console app entry point.", "csharp")
        case "cpp", "cc", "cxx": return ("C++", "Build a runnable modern C++ program (C++17 style).", "cpp")
        case "go": return ("Go", "Build a runnable Go program with package main and func main().", "go")
        case "rs": return ("Rust", "Build a runnable Rust program with fn main().", "rust")
        default: return nil
        }
    }

    private func buildFileAwareEditPrompt(_ userPrompt: String) -> String {
        guard let config = languagePromptConfig() else { return userPrompt }
        let fileName = currentFileName ?? "untitled"
        let codeSection = generatedCode.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            ? "FILE_IS_EMPTY"
            : String(generatedCode.prefix(20_000))

        return """
        You are an expert coding assistant working on a real file.
        FILE: \(fileName)
        TARGET LANGUAGE: \(config.languageName)
        TARGET RULE: \(config.targetRule)

        USER REQUEST:
        \(userPrompt)

        CURRENT FILE CONTENT:
        ```\(config.fenceLanguage)
        \(codeSection)
        ```

        INSTRUCTIONS:
        - Produce the FULL updated file content.
        - Do not return partial snippets or patch hunks.
        - Wrap the final full file between markers:
          <<<FULL_FILE_START>>>
          [full file content]
          <<<FULL_FILE_END>>>
        - Respect the FILE extension/language exactly.
        - If file is empty, create a complete starter implementation for this request.
        - Do not output explanations.
        - Output only one fenced code block using ```\(config.fenceLanguage).
        """
    }

    private func extractGeneratedCode(_ response: String) -> String {
        if let markerRange = response.range(of: #"<<<FULL_FILE_START>>>[\s\S]*?<<<FULL_FILE_END>>>"#, options: .regularExpression) {
            var extracted = String(response[markerRange])
            extracted = extracted.replacingOccurrences(of: "<<<FULL_FILE_START>>>", with: "")
            extracted = extracted.replacingOccurrences(of: "<<<FULL_FILE_END>>>", with: "")
            return sanitizeExtractedCode(extracted)
        }

        if let codeRange = response.range(of: #"```[a-zA-Z0-9_-]*[\s\S]*?```"#, options: .regularExpression) {
            let fenced = String(response[codeRange])
                .replacingOccurrences(of: #"^```[a-zA-Z0-9_-]*\s*"#, with: "", options: .regularExpression)
                .replacingOccurrences(of: #"\s*```$"#, with: "", options: .regularExpression)
            return sanitizeExtractedCode(fenced)
        }

        return sanitizeExtractedCode(response)
    }

    private func sanitizeExtractedCode(_ raw: String) -> String {
        raw
            .replacingOccurrences(of: "<<<FULL_FILE_START>>>", with: "")
            .replacingOccurrences(of: "<<<FULL_FILE_END>>>", with: "")
            .replacingOccurrences(of: "```", with: "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func openFile(_ url: URL, announceInChat: Bool = true) {
        do {
            #if canImport(UIKit)
            let scopeURL = workspaceFolderURL ?? url
            let didStart = scopeURL.startAccessingSecurityScopedResource()
            defer {
                if didStart {
                    scopeURL.stopAccessingSecurityScopedResource()
                }
            }
            #endif

            let text = try String(contentsOf: url, encoding: .utf8)
            currentFileURL = url
            currentFileName = url.lastPathComponent
            generatedCode = text
            if announceInChat {
                appendMessage(to: activeChatSessionId, role: "assistant", text: "Opened \(url.lastPathComponent)")
            }
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    private func createNewFile() {
        let name = newFileNameInput.trimmingCharacters(in: .whitespacesAndNewlines)
        guard name.contains("."), !name.isEmpty else {
            errorMessage = settings.localized("vibe_coder_file_name_error")
            return
        }
        guard let folderURL = workspaceFolderURL else {
            errorMessage = settings.localized("vibe_coder_select_folder_title")
            return
        }

        let fileURL = folderURL.appendingPathComponent(name)
        do {
            #if canImport(UIKit)
            let didStart = folderURL.startAccessingSecurityScopedResource()
            defer {
                if didStart {
                    folderURL.stopAccessingSecurityScopedResource()
                }
            }
            #endif

            if !FileManager.default.fileExists(atPath: fileURL.path) {
                try "".write(to: fileURL, atomically: true, encoding: .utf8)
            }

            currentFileURL = fileURL
            currentFileName = fileURL.lastPathComponent
            generatedCode = ""
            refreshWorkspaceFiles()
            appendMessage(to: activeChatSessionId, role: "assistant", text: "Started new file: \(name)")
        } catch {
            errorMessage = error.localizedDescription
        }

        newFileNameInput = ""
    }

    private func saveCurrentFile(silent: Bool = false) {
        guard hasFileSession else {
            errorMessage = settings.localized("vibe_coder_no_file_selected_error")
            return
        }

        guard let workspaceFolderURL else {
            errorMessage = settings.localized("vibe_coder_select_folder_title")
            return
        }

        let folderURL = workspaceFolderURL

        let fileURL: URL
        if let currentFileURL {
            fileURL = currentFileURL
        } else {
            fileURL = folderURL.appendingPathComponent(currentFileName ?? "main.txt")
            currentFileURL = fileURL
        }

        do {
            #if canImport(UIKit)
            let didStart = folderURL.startAccessingSecurityScopedResource()
            defer {
                if didStart {
                    folderURL.stopAccessingSecurityScopedResource()
                }
            }
            #endif

            try generatedCode.write(to: fileURL, atomically: true, encoding: .utf8)
            currentFileURL = fileURL
            currentFileName = fileURL.lastPathComponent
            if !silent {
                refreshWorkspaceFiles()
            }
            if !silent {
                appendMessage(to: activeChatSessionId, role: "assistant", text: "Saved \(currentFileName ?? fileURL.lastPathComponent)")
            }
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    private func restoreWorkspaceFolderFromBookmark() {
        guard !folderBookmark.isEmpty else { return }
        var stale = false
        guard let url = try? URL(resolvingBookmarkData: folderBookmark, options: [.withoutUI], relativeTo: nil, bookmarkDataIsStale: &stale) else {
            return
        }
        if stale {
            // Keep using it for now; user can re-pick later.
        }
        workspaceFolderURL = url
        refreshWorkspaceFiles()
        restoreLastOpenedFileIfPossible()
    }

    private func setWorkspaceFolder(_ url: URL) {
        workspaceFolderURL = url
        currentFileURL = nil
        currentFileName = nil
        generatedCode = ""
        do {
            #if canImport(UIKit)
            let didStart = url.startAccessingSecurityScopedResource()
            defer {
                if didStart {
                    url.stopAccessingSecurityScopedResource()
                }
            }
            #endif

            let bookmark = try url.bookmarkData(options: [.minimalBookmark], includingResourceValuesForKeys: nil, relativeTo: nil)
            folderBookmark = bookmark
        } catch {
            // Non-fatal; folder selection still works for this session.
        }

        refreshWorkspaceFiles()
        restoreLastOpenedFileIfPossible()
    }

    private func openHTMLPreviewInSafari() {
        let html = generatedCode
        guard !html.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
        Task { @MainActor in
            do {
                let url = try await LocalHTMLPreviewServer.shared.start(html: html)
                UIApplication.shared.open(url, options: [:], completionHandler: nil)
            } catch {
                errorMessage = error.localizedDescription
            }
        }
    }

    private func ensureModelLoaded(force: Bool) async {
        guard let model = selectedFeatureModel(named: selectedModelName) else {
            errorMessage = settings.localized("vibe_coder_no_model")
            return
        }
        isLoading = true
        defer { isLoading = false }

        let modelContextCap = model.contextWindowSize > 0 ? model.contextWindowSize : 4096
        let effectiveContext = min(max(1, Int(maxTokens)), modelContextCap)
        let shouldReload = force
            || llm.currentlyLoadedModel != model.name
            || llm.loadedContextWindow != effectiveContext

        llm.maxTokens = min(Int(maxTokens), effectiveContext)
        llm.contextWindow = effectiveContext
        llm.enableVision = false
        llm.enableAudio = false
        llm.enableThinking = enableThinking

        do {
            if shouldReload {
                try await llm.loadModel(model)
            }
            errorMessage = nil
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    private func stopGeneration() {
        generationTask?.cancel()
        generationTask = nil
        isGenerating = false
    }

    private func sendChat() {
        dismissKeyboard()

        let trimmedPrompt = chatInput.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedPrompt.isEmpty else { return }
        guard !isGenerating else { return }

        let needsContextReset = isContextBudgetExceededForSession
        if needsContextReset {
            createNewChatSession()
            appendMessage(
                to: activeChatSessionId,
                role: "assistant",
                text: "Started a new chat because context window was full."
            )
        }

        generationTask = Task {
            await ensureModelLoaded(force: false)
            guard llm.isLoaded else { return }

            guard hasFileSession else {
                errorMessage = settings.localized("vibe_coder_no_file_selected_error")
                return
            }
            guard languagePromptConfig() != nil else {
                errorMessage = "Unsupported or unknown file extension. Use a code file like .py, .js, .ts, .java, .kt, .go, .rs, .cpp, .cs, .html"
                return
            }

            let sessionId = activeChatSessionId
            appendMessage(to: sessionId, role: "user", text: trimmedPrompt)
            let assistantId = appendMessage(to: sessionId, role: "assistant", text: "")
            await MainActor.run { chatInput = "" }

            isGenerating = true
            do {
                // Build the file-aware base prompt for the current request.
                let filePrompt = buildFileAwareEditPrompt(trimmedPrompt)

                // Wrap with full conversation history so the model remembers prior turns.
                let prompt = buildVibeCoderMultiTurnPrompt(
                    currentFilePrompt: filePrompt,
                    sessionId: sessionId
                )

                try await llm.generate(prompt: prompt) { text, _, _ in
                    Task { @MainActor in
                        updateMessageText(sessionId: sessionId, messageId: assistantId, text: text)

                        let now = CFAbsoluteTimeGetCurrent()
                        if now - lastStreamTickTime > 0.06 {
                            lastStreamTickTime = now
                            streamTick += 1
                        }
                    }
                }

                if let finalText = messageText(sessionId: sessionId, messageId: assistantId) {
                    let extractedCode = extractGeneratedCode(finalText)
                    if !extractedCode.isEmpty {
                        await MainActor.run {
                            generatedCode = extractedCode
                            saveCurrentFile(silent: true)
                        }
                    }
                }
            } catch {
                errorMessage = error.localizedDescription
            }
            isGenerating = false
            generationTask = nil
        }
    }

    @discardableResult
    private func appendMessage(to sessionId: UUID, role: String, text: String, id: UUID = UUID()) -> UUID {
        guard let idx = chatSessions.firstIndex(where: { $0.id == sessionId }) else { return id }
        chatSessions[idx].messages.append(VibeChatMessage(id: id, role: role, text: text))
        return id
    }

    private func updateMessageText(sessionId: UUID, messageId: UUID, text: String) {
        guard let sIdx = chatSessions.firstIndex(where: { $0.id == sessionId }) else { return }
        guard let mIdx = chatSessions[sIdx].messages.firstIndex(where: { $0.id == messageId }) else { return }
        chatSessions[sIdx].messages[mIdx].text = sanitizeModelOutputText(text)
    }

    private func messageText(sessionId: UUID, messageId: UUID) -> String? {
        guard let sIdx = chatSessions.firstIndex(where: { $0.id == sessionId }) else { return nil }
        guard let msg = chatSessions[sIdx].messages.first(where: { $0.id == messageId }) else { return nil }
        return msg.text
    }

    /// Builds a multi-turn prompt from VibeCode chat history so the model remembers
    /// prior coding requests. Uses RAW_PROMPT to bypass SDK re-formatting.
    private func buildVibeCoderMultiTurnPrompt(currentFilePrompt: String, sessionId: UUID) -> String {
        let modelName = selectedModelName.lowercased()
        let modelSupportsThinking = selectedFeatureModel(named: selectedModelName)?.supportsThinking == true
        let isGemma  = modelName.contains("gemma")
        let isGemma4 = isGemma && (modelName.contains("gemma 4 ") || modelName.contains("gemma-4 ") || modelName.hasSuffix("gemma 4") || modelName.hasSuffix("gemma-4"))
        let isLlama  = modelName.contains("llama") || modelName.contains("mistral")
        let isHarmonyModel = modelName.contains("gpt-oss") || modelName.contains("gpt_oss")

        // 1. Get history (exclude placeholder turns)
        var allMessages: [VibeChatMessage] = []
        if let idx = chatSessions.firstIndex(where: { $0.id == sessionId }) {
            allMessages = chatSessions[idx].messages
        }
        
        // Drop the last 2 (new user + empty assistant placeholder).
        let history = allMessages.count >= 2 ? Array(allMessages.dropLast(2)) : []
        
        // 2. Context Management (Sliding Window)
        // Same smarter budget as AI Chat. See ChatScreen.buildMultiTurnPrompt.
        let effectiveCtxTokens = llm.loadedContextWindow ?? 4096
        let reservedForResponse = max(256, min(Int(maxTokens), effectiveCtxTokens / 4))
        let reservedForCurrent = max(64, currentFilePrompt.count / 3) + 64
        let reservedSafety = 128
        let availableHistoryTokens = max(128, effectiveCtxTokens - reservedForResponse - reservedForCurrent - reservedSafety)
        let maxHistoryChars = availableHistoryTokens * 3
        var currentChars = 0
        var truncatedHistory: [VibeChatMessage] = []
        for msg in history.reversed() {
            let msgLen = msg.text.count
            if currentChars + msgLen < maxHistoryChars {
                truncatedHistory.insert(msg, at: 0)
                currentChars += msgLen
            } else {
                break
            }
        }

        // 3. Build Raw Prompt
        var parts: [String] = ["__RAW_PROMPT__"]

        // 4. System Prompt (Coding Assistant)
        let systemPrompt = "You are VibeCoder, a world-class on-device coding assistant. Provide clean, efficient, and correct code. Use markdown code blocks with language tags."

        if isHarmonyModel {
            var harmonyParts: [String] = []
            harmonyParts.append("<|start|>system<|message|>\(systemPrompt)<|end|>")

            for msg in truncatedHistory {
                let rawText = msg.text.trimmingCharacters(in: .whitespacesAndNewlines)
                let content: String
                if msg.role == "user" {
                    content = rawText
                } else {
                    let answer = getDisplayContentWithoutThinking(rawText)
                    content = answer.isEmpty ? rawText : answer
                }
                guard !content.isEmpty else { continue }

                let role = msg.role == "user" ? "user" : "assistant"
                harmonyParts.append("<|start|>\(role)<|message|>\(content)<|end|>")
            }

            harmonyParts.append("<|start|>user<|message|>\(currentFilePrompt)<|end|>")
            if modelSupportsThinking && llm.enableThinking {
                harmonyParts.append("<|start|>assistant")
            } else {
                harmonyParts.append("<|start|>assistant<|channel|>analysis<|message|><|end|><|start|>assistant<|channel|>final<|message|>")
            }
            parts.append(contentsOf: harmonyParts)
            return parts.joined()
        }
        
        if isGemma4 {
            parts.append("<|turn>system\n\(systemPrompt)<turn|>")
        } else if isGemma {
            parts.append("<start_of_turn>system\n\(systemPrompt)<end_of_turn>")
        } else if isLlama {
            parts.append("<<SYS>>\n\(systemPrompt)\n<</SYS>>")
        } else {
            parts.append("System: \(systemPrompt)")
        }

        for msg in truncatedHistory {
            // Strip thinking blocks from assistant messages so the reasoning chain
            // is not re-fed into the context window.
            let rawText = msg.text.trimmingCharacters(in: .whitespacesAndNewlines)
            let content: String
            if msg.role == "user" {
                content = rawText
            } else {
                let answer = getDisplayContentWithoutThinking(rawText)
                content = answer.isEmpty ? rawText : answer
            }
            guard !content.isEmpty else { continue }

            if isGemma4 {
                let role = (msg.role == "user") ? "user" : "model"
                parts.append("<|turn>\(role)\n\(content)<turn|>")
            } else if isGemma {
                let role = (msg.role == "user") ? "user" : "model"
                parts.append("<start_of_turn>\(role)\n\(content)<end_of_turn>")
            } else if isLlama {
                if msg.role == "user" {
                    parts.append("[INST] \(content) [/INST]")
                } else {
                    parts.append(content)
                }
            } else {
                let prefix = (msg.role == "user") ? "User" : "Assistant"
                parts.append("\(prefix): \(content)")
            }
        }

        // 4. Final Open Turn
        if isGemma4 {
            parts.append("<|turn>user\n\(currentFilePrompt)<turn|>")
            parts.append("<|turn>model\n")
        } else if isGemma {
            parts.append("<start_of_turn>user\n\(currentFilePrompt)<end_of_turn>")
            parts.append("<start_of_turn>model\n")
        } else if isLlama {
            parts.append("[INST] \(currentFilePrompt) [/INST]")
        } else {
            parts.append("User: \(currentFilePrompt)")
            parts.append("Assistant:")
        }

        return parts.joined(separator: "\n")
    }

    private func createNewChatSession() {
        let title = "Chat \(chatSessions.count + 1)"
        let session = VibeChatSession(title: title)
        chatSessions.append(session)
        activeChatSessionId = session.id
    }


    private func clearActiveChat() {
        guard let idx = chatSessions.firstIndex(where: { $0.id == activeChatSessionId }) else { return }
        chatSessions[idx].messages.removeAll()
    }

    private func deleteChatSession(_ id: UUID) {
        guard chatSessions.count > 1 else {
            pendingDeleteChatId = nil
            return
        }
        chatSessions.removeAll { $0.id == id }
        if activeChatSessionId == id {
            activeChatSessionId = chatSessions.first?.id ?? activeChatSessionId
        }
        pendingDeleteChatId = nil
    }

    private func refreshWorkspaceFiles() {
        guard let folderURL = workspaceFolderURL else {
            workspaceFiles = []
            return
        }

        do {
            #if canImport(UIKit)
            let didStart = folderURL.startAccessingSecurityScopedResource()
            defer {
                if didStart {
                    folderURL.stopAccessingSecurityScopedResource()
                }
            }
            #endif

            let urls = try FileManager.default.contentsOfDirectory(
                at: folderURL,
                includingPropertiesForKeys: [.isRegularFileKey],
                options: [.skipsHiddenFiles]
            )

            workspaceFiles = urls
                .filter { (try? $0.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) ?? false }
                .sorted { $0.lastPathComponent.localizedCaseInsensitiveCompare($1.lastPathComponent) == .orderedAscending }
        } catch {
            workspaceFiles = []
        }
    }

    private func deleteFile(_ url: URL) {
        guard let folderURL = workspaceFolderURL else {
            pendingDeleteFileURL = nil
            return
        }

        do {
            #if canImport(UIKit)
            let didStart = folderURL.startAccessingSecurityScopedResource()
            defer {
                if didStart {
                    folderURL.stopAccessingSecurityScopedResource()
                }
            }
            #endif

            try FileManager.default.removeItem(at: url)
            if currentFileURL == url {
                currentFileURL = nil
                currentFileName = nil
                generatedCode = ""
            }
            refreshWorkspaceFiles()
        } catch {
            errorMessage = error.localizedDescription
        }

        pendingDeleteFileURL = nil
    }

    private func persistChatSessionsToStorage() {
        do {
            chatSessionsData = try JSONEncoder().encode(chatSessions)
        } catch {
            // Keep previous valid encoded value if encoding fails.
        }
        activeChatSessionIdRaw = activeChatSessionId.uuidString
    }

    private func restoreChatSessionsFromStorage() {
        if !chatSessionsData.isEmpty,
           let decoded = try? JSONDecoder().decode([VibeChatSession].self, from: chatSessionsData),
           !decoded.isEmpty {
            chatSessions = decoded
        } else if chatSessions.isEmpty {
            chatSessions = [VibeChatSession(title: "Chat 1")]
        }

        if let restoredId = UUID(uuidString: activeChatSessionIdRaw),
           chatSessions.contains(where: { $0.id == restoredId }) {
            activeChatSessionId = restoredId
        } else {
            activeChatSessionId = chatSessions.first?.id ?? UUID()
        }
    }

    private func persistCurrentFileSelection() {
        guard let folderURL = workspaceFolderURL,
              let fileURL = currentFileURL,
              fileURL.path.hasPrefix(folderURL.path) else {
            currentFileRelativePath = ""
            return
        }

        var relativePath = fileURL.path
        relativePath.removeFirst(folderURL.path.count)
        currentFileRelativePath = relativePath.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
    }

    private func restoreLastOpenedFileIfPossible() {
        guard let folderURL = workspaceFolderURL,
              !currentFileRelativePath.isEmpty,
              currentFileURL == nil else {
            return
        }

        let candidateURL = folderURL.appendingPathComponent(currentFileRelativePath)
        if FileManager.default.fileExists(atPath: candidateURL.path) {
            openFile(candidateURL, announceInChat: false)
        }
    }
}

// MARK: - ImageGeneratorScreen

struct ImageGeneratorScreen: View {
    @EnvironmentObject var settings: AppSettings
    @AppStorage("sd_selected_model_id") private var selectedModelId: String = ""
    @AppStorage("sd_steps") private var storedSteps: Double = 20
    @AppStorage("sd_denoise_strength") private var storedDenoiseStrength: Double = 0.7

    @State private var promptText = ""
    @FocusState private var promptFocused: Bool
    @State private var seed: Int = Int.random(in: 0..<1_000_000)
    @State private var generatedImages: [UIImage] = []
    @State private var isGenerating = false
    @State private var currentPage = 0
    @State private var showSettings = false
    @State private var errorMessage: String?
    @State private var inputImage: UIImage?
    @State private var selectedImageItem: PhotosPickerItem?
    @State private var generateTask: Task<Void, Never>?

    @ObservedObject private var sdBackend = StableDiffusionBackend.shared

    let onNavigateBack: () -> Void
    let onNavigateToModels: () -> Void

    private var availableModels: [AIModel] {
        ModelData.models.filter { $0.isCoreMLImageGeneration && StableDiffusionBackend.isCoreMLModelDownloaded(modelId: $0.id) }
    }

    private var selectedModel: AIModel? {
        availableModels.first(where: { $0.id == selectedModelId }) ?? availableModels.first
    }

    private var selectedModelSupportsImageToImage: Bool {
        guard let selectedModel else { return false }
        return StableDiffusionBackend.supportsImageToImage(modelId: selectedModel.id)
    }

    var body: some View {
        Group {
            if availableModels.isEmpty {
                noModelView
            } else if !sdBackend.isLoaded {
                loadModelView
            } else {
                mainGenerationView
            }
        }
        .navigationTitle(settings.localized("image_generator_title"))
        .navigationBarTitleDisplayMode(.inline)
        .apolloScreenBackground()
        .safeAreaInset(edge: .bottom, spacing: 0) { BannerAdContainer() }
        .toolbarBackground(.hidden, for: .navigationBar)
        .toolbar {
            ToolbarItem(placement: .navigationBarLeading) {
                Button {
                    generateTask?.cancel()
                    sdBackend.unloadModel()
                    onNavigateBack()
                } label: { Image(systemName: "arrow.left") }
            }
            ToolbarItem(placement: .navigationBarTrailing) {
                Button { showSettings = true } label: { Image(systemName: "slider.horizontal.3") }
            }
        }
        .sheet(isPresented: $showSettings) {
            ImageGeneratorSettingsSheet(
                availableModels: availableModels,
                selectedModelId: $selectedModelId,
                steps: $storedSteps,
                denoiseStrength: $storedDenoiseStrength,
                isLoaded: sdBackend.isLoaded,
                isLoading: sdBackend.isLoading,
                onLoad: {
                    guard let model = selectedModel else { return }
                    Task {
                        do {
                            try await sdBackend.loadModel(model)
                        } catch {
                            errorMessage = error.localizedDescription
                        }
                    }
                },
                onUnload: { sdBackend.unloadModel() }
            )
        }
        .onChange(of: selectedImageItem) { _, item in
            guard selectedModelSupportsImageToImage else {
                selectedImageItem = nil
                inputImage = nil
                return
            }
            guard let item else { inputImage = nil; return }
            Task {
                if let data = try? await item.loadTransferable(type: Data.self),
                   let img = UIImage(data: data) {
                    inputImage = img
                } else {
                    inputImage = nil
                }
            }
        }
        .onAppear {
            if selectedModelId.isEmpty || !availableModels.contains(where: { $0.id == selectedModelId }) {
                selectedModelId = availableModels.first?.id ?? ""
            }
            if !selectedModelSupportsImageToImage {
                selectedImageItem = nil
                inputImage = nil
            }
        }
        .onChange(of: selectedModelId) { _, _ in
            if !selectedModelSupportsImageToImage {
                selectedImageItem = nil
                inputImage = nil
            }
        }
        .onDisappear {
            generateTask?.cancel()
            sdBackend.unloadModel()
        }
        .overlay(alignment: .bottom) {
            if let msg = errorMessage {
                Text(msg)
                    .font(.caption)
                    .foregroundStyle(.red)
                    .padding(.horizontal)
                    .padding(.bottom, 8)
                    .onTapGesture { errorMessage = nil }
            }
        }
    }

    // MARK: - No-Model State

    private var noModelView: some View {
        VStack(spacing: 20) {
            Image(systemName: "paintpalette.fill")
                .font(.system(size: 56, weight: .semibold))
                .foregroundStyle(.secondary)
            Text(settings.localized("image_generator_download_model"))
                .font(.title3.weight(.bold))
                .multilineTextAlignment(.center)
            Text(settings.localized("image_generator_download_model_desc"))
                .font(.subheadline)
                .foregroundStyle(.white.opacity(0.7))
                .multilineTextAlignment(.center)
                .padding(.horizontal)
            Button {
                onNavigateToModels()
            } label: {
                HStack {
                    Spacer()
                    Text(settings.localized("download_models"))
                    Spacer()
                }
                .frame(height: 50)
                .contentShape(Rectangle())
            }
            .frame(maxWidth: 260)
            .liquidGlassPrimaryButton(cornerRadius: 12)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    // MARK: - Load-Model State

    private var loadModelView: some View {
        VStack(spacing: 20) {
            if sdBackend.isLoading {
                ProgressView()
                    .scaleEffect(1.4)
                Text(settings.localized("image_generator_loading_model"))
                    .font(.title3.weight(.bold))
            } else {
                Image(systemName: "cpu.fill")
                    .font(.system(size: 56, weight: .semibold))
                    .foregroundStyle(.secondary)
                Text(settings.localized("image_generator_load_model_title"))
                    .font(.title3.weight(.bold))
                    .multilineTextAlignment(.center)
                Text(settings.localized("image_generator_load_model_desc"))
                    .font(.subheadline)
                    .foregroundStyle(.white.opacity(0.7))
                    .multilineTextAlignment(.center)
                    .padding(.horizontal)
                Button {
                    showSettings = true
                } label: {
                    HStack {
                        Spacer()
                        Text(settings.localized("feature_settings_title"))
                        Spacer()
                    }
                    .frame(height: 50)
                    .contentShape(Rectangle())
                }
                .frame(maxWidth: 260)
                .liquidGlassPrimaryButton(cornerRadius: 12)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    // MARK: - Main Generation View

    private var mainGenerationView: some View {
        GeometryReader { geo in
            let isLandscape = geo.size.width > geo.size.height
            if isLandscape && !generatedImages.isEmpty {
                landscapeLayout
            } else {
                portraitLayout
            }
        }
    }

    // MARK: - Portrait Layout

    private var portraitLayout: some View {
        ScrollView {
            VStack(spacing: 14) {
                promptCard
                if selectedModelSupportsImageToImage {
                    img2imgCard
                }
                if !generatedImages.isEmpty {
                    imageSwipeView
                }
                generateButton
            }
            .padding(.horizontal)
            .padding(.vertical, 12)
        }
    }

    // MARK: - Landscape Layout (side-by-side)

    private var landscapeLayout: some View {
        HStack(spacing: 14) {
            ScrollView {
                VStack(spacing: 14) {
                    promptCard
                    if selectedModelSupportsImageToImage {
                        img2imgCard
                    }
                    generateButton
                    if !generatedImages.isEmpty {
                        saveButton
                    }
                }
                .padding(.vertical, 12)
            }
            .frame(maxWidth: .infinity)
            imageSwipeView
                .frame(maxWidth: .infinity)
        }
        .padding(.horizontal)
    }

    // MARK: - Prompt Card

    private var promptCard: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(settings.localized("image_generator_prompt_label"))
                .font(.headline)
            TextEditor(text: $promptText)
                .focused($promptFocused)
                .frame(minHeight: 100)
                .scrollContentBackground(.hidden)
                .background(Color.white.opacity(0.02))
                .clipShape(RoundedRectangle(cornerRadius: 12))
                .disabled(isGenerating)
                .overlay(
                    Group {
                        if promptText.isEmpty {
                            Text(settings.localized("image_generator_prompt_hint"))
                                .foregroundStyle(.secondary)
                                .padding(.horizontal, 12)
                                .padding(.vertical, 16)
                                .allowsHitTesting(false)
                                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
                        }
                    }
                )
        }
        .padding(.horizontal)
        .padding(.vertical, 12)
        .background(.ultraThinMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 16))
        .overlay(RoundedRectangle(cornerRadius: 16).stroke(Color.white.opacity(0.14), lineWidth: 1))
    }

    // MARK: - Img2Img Card

    private var img2imgCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(settings.localized("image_generator_img2img"))
                .font(.headline)
            HStack(spacing: 12) {
                PhotosPicker(
                    selection: $selectedImageItem,
                    matching: .images
                ) {
                    HStack {
                        Image(systemName: "photo")
                        Text(settings.localized(inputImage != nil ? "image_generator_change_image" : "image_generator_select_image"))
                            .lineLimit(1)
                    }
                    .frame(maxWidth: .infinity)
                    .frame(height: 40)
                    .contentShape(Rectangle())
                }
                .foregroundStyle(.white)
                .background(.ultraThinMaterial)
                .clipShape(RoundedRectangle(cornerRadius: 10))
                .overlay(RoundedRectangle(cornerRadius: 10).stroke(Color.white.opacity(0.18), lineWidth: 1))
                .disabled(isGenerating)

                if let thumb = inputImage {
                    ZStack(alignment: .topTrailing) {
                        Image(uiImage: thumb)
                            .resizable()
                            .scaledToFill()
                            .frame(width: 48, height: 48)
                            .clipShape(RoundedRectangle(cornerRadius: 8))
                        Button {
                            inputImage = nil
                            selectedImageItem = nil
                        } label: {
                            Image(systemName: "xmark.circle.fill")
                                .font(.caption)
                                .foregroundStyle(.white, .black.opacity(0.5))
                        }
                    }
                }
            }

            if inputImage != nil {
                VStack(alignment: .leading, spacing: 4) {
                    Text(String(format: settings.localized("image_generator_denoise_strength"), storedDenoiseStrength))
                        .font(.caption)
                    Text(settings.localized("image_generator_denoise_strength_desc"))
                        .font(.caption2)
                        .foregroundStyle(.white.opacity(0.6))
                    Slider(value: $storedDenoiseStrength, in: 0.1...1.0)
                        .disabled(isGenerating)
                }
            }
        }
        .padding(.horizontal)
        .padding(.vertical, 12)
        .background(.ultraThinMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 16))
        .overlay(RoundedRectangle(cornerRadius: 16).stroke(Color.white.opacity(0.14), lineWidth: 1))
    }

    // MARK: - Image Swipe View (HorizontalPager equivalent)

    private var imageSwipeView: some View {
        VStack(spacing: 8) {
            TabView(selection: $currentPage) {
                ForEach(0..<(generatedImages.count + 1), id: \.self) { page in
                    if page < generatedImages.count {
                        Image(uiImage: generatedImages[page])
                            .resizable()
                            .aspectRatio(1, contentMode: .fit)
                            .clipShape(RoundedRectangle(cornerRadius: 12))
                            .padding(.horizontal, 4)
                            .tag(page)
                            .contextMenu {
                                Button {
                                    saveImageToPhotos(generatedImages[page])
                                } label: {
                                    Label(settings.localized("image_generator_save"), systemImage: "square.and.arrow.down")
                                }
                            }
                    } else {
                        placeholderPage
                            .tag(page)
                    }
                }
            }
            .tabViewStyle(.page(indexDisplayMode: .never))
            .aspectRatio(1, contentMode: .fit)
            .onChange(of: currentPage) { _, page in
                if page == generatedImages.count && !isGenerating {
                    triggerVariation()
                }
            }

            // Page indicator dots
            HStack(spacing: 6) {
                ForEach(0..<generatedImages.count, id: \.self) { i in
                    Circle()
                        .fill(currentPage == i ? Color.white : Color.white.opacity(0.35))
                        .frame(width: 7, height: 7)
                }
                // "+1" placeholder dot
                Circle()
                    .fill(currentPage == generatedImages.count ? Color.white : Color.white.opacity(0.35))
                    .frame(width: 7, height: 7)
            }

            if !generatedImages.isEmpty && currentPage < generatedImages.count {
                saveButton
            }
        }
    }

    private var placeholderPage: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 12)
                .fill(Color.white.opacity(0.05))
                .overlay(RoundedRectangle(cornerRadius: 12).stroke(Color.white.opacity(0.12), lineWidth: 1))
            if isGenerating {
                VStack(spacing: 16) {
                    ProgressView()
                        .scaleEffect(1.3)
                        .tint(.white)
                    Text(String(format: settings.localized("image_generator_variation"), generatedImages.count + 1))
                        .font(.subheadline)
                        .foregroundStyle(.white.opacity(0.7))
                }
            } else {
                VStack(spacing: 12) {
                    Image(systemName: "plus.circle")
                        .font(.system(size: 44, weight: .light))
                        .foregroundStyle(.white.opacity(0.5))
                    Text(settings.localized("image_generator_swipe_more"))
                        .font(.subheadline)
                        .foregroundStyle(.white.opacity(0.6))
                        .multilineTextAlignment(.center)
                }
            }
        }
        .aspectRatio(1, contentMode: .fit)
        .clipShape(RoundedRectangle(cornerRadius: 12))
        .padding(.horizontal, 4)
    }

    // MARK: - Generate Button

    private var generateButton: some View {
        Button {
            if isGenerating {
                generateTask?.cancel()
                isGenerating = false
            } else {
                startGeneration(clearAll: true)
            }
        } label: {
            HStack(spacing: 8) {
                if isGenerating {
                    ProgressView()
                        .tint(.white)
                        .scaleEffect(0.85)
                    Text(settings.localized("image_generator_generating"))
                        .lineLimit(1)
                } else {
                    Image(systemName: "sparkles")
                        .font(.system(size: 13, weight: .bold))
                    Text(settings.localized("image_generator_generate"))
                        .lineLimit(1)
                }
            }
            .frame(maxWidth: .infinity)
            .frame(height: 52)
        }
        .foregroundStyle(.white)
        .liquidGlassPrimaryButton(cornerRadius: 12)
        .disabled(!isGenerating && promptText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
    }

    private var saveButton: some View {
        Button {
            if currentPage < generatedImages.count {
                saveImageToPhotos(generatedImages[currentPage])
            }
        } label: {
            HStack {
                Spacer()
                Text(settings.localized("image_generator_save"))
                Spacer()
            }
            .frame(height: 44)
            .contentShape(Rectangle())
        }
        .foregroundStyle(.white)
        .liquidGlassPrimaryButton(cornerRadius: 12)
    }

    // MARK: - Generation Logic

    private func startGeneration(clearAll: Bool) {
        guard !promptText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
        if clearAll {
            generatedImages.removeAll()
            currentPage = 0
            seed = Int.random(in: 0..<1_000_000)
        }
        runGeneration(usingSeed: UInt32(bitPattern: Int32(truncatingIfNeeded: seed)))
    }

    private func triggerVariation() {
        guard !isGenerating, sdBackend.isLoaded else { return }
        let varSeed = UInt32.random(in: 0..<UInt32.max)
        runGeneration(usingSeed: varSeed)
    }

    private func runGeneration(usingSeed genSeed: UInt32) {
        isGenerating = true
        promptFocused = false
        let prompt = promptText
        let steps = Int(storedSteps)
        let denoiseStrength = Float(storedDenoiseStrength)
        let startingImage = selectedModelSupportsImageToImage ? inputImage : nil

        generateTask?.cancel()
        generateTask = Task {
            defer { isGenerating = false }
            do {
                let img = try await sdBackend.generateImage(
                    prompt: prompt,
                    steps: steps,
                    seed: genSeed,
                    inputImage: startingImage,
                    denoiseStrength: denoiseStrength
                )
                if let img {
                    generatedImages.append(img)
                    currentPage = generatedImages.count - 1
                }
            } catch is CancellationError {
                // ignore
            } catch {
                errorMessage = error.localizedDescription
            }
        }
    }

    private func saveImageToPhotos(_ image: UIImage) {
        UIImageWriteToSavedPhotosAlbum(image, nil, nil, nil)
        errorMessage = settings.localized("image_generator_saved")
    }
}

// MARK: - ImageGeneratorSettingsSheet

private struct ImageGeneratorSettingsSheet: View {
    @EnvironmentObject var settings: AppSettings
    let availableModels: [AIModel]
    @Binding var selectedModelId: String
    @Binding var steps: Double
    @Binding var denoiseStrength: Double
    let isLoaded: Bool
    let isLoading: Bool
    let onLoad: () -> Void
    let onUnload: () -> Void

    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 20) {
                    // Model Picker
                    VStack(alignment: .leading, spacing: 12) {
                        Text(settings.localized("select_model_title"))
                            .font(.headline)
                        if availableModels.isEmpty {
                            Label(settings.localized("image_generator_no_models"), systemImage: "exclamationmark.triangle")
                                .foregroundStyle(.secondary)
                        } else {
                            Picker(settings.localized("select_model"), selection: $selectedModelId) {
                                ForEach(availableModels) { model in
                                    Text(model.name).tag(model.id)
                                }
                            }
                            .pickerStyle(.menu)
                            .tint(ApolloPalette.accentStrong)
                            .frame(maxWidth: .infinity, alignment: .leading)
                        }
                    }
                    .padding()
                    .background(.regularMaterial)
                    .clipShape(RoundedRectangle(cornerRadius: 14))

                    // Steps Slider
                    VStack(alignment: .leading, spacing: 8) {
                        Text("\(settings.localized("image_generator_iterations")): \(Int(steps))")
                            .font(.headline)
                        Slider(value: $steps, in: 10...50, step: 1)
                            .tint(ApolloPalette.accentStrong)
                    }
                    .padding()
                    .background(.regularMaterial)
                    .clipShape(RoundedRectangle(cornerRadius: 14))

                    // Load / Unload
                    VStack(spacing: 12) {
                        Button {
                            onLoad()
                            dismiss()
                        } label: {
                            if isLoading {
                                HStack {
                                    ProgressView()
                                    Text(settings.localized("image_generator_loading_model"))
                                }
                                .frame(maxWidth: .infinity)
                                .frame(height: 50)
                            } else {
                                Text(settings.localized(isLoaded ? "reload_model" : "image_generator_load_model"))
                                    .frame(maxWidth: .infinity)
                                    .frame(height: 50)
                            }
                        }
                        .liquidGlassPrimaryButton(cornerRadius: 12)
                        .disabled(availableModels.isEmpty || isLoading)

                        if isLoaded {
                            Button(role: .destructive) {
                                onUnload()
                                dismiss()
                            } label: {
                                Text(settings.localized("unload_model"))
                                    .frame(maxWidth: .infinity)
                                    .frame(height: 50)
                            }
                            .background(
                                RoundedRectangle(cornerRadius: 12)
                                    .fill(ApolloPalette.destructive.opacity(0.10))
                            )
                            .overlay(
                                RoundedRectangle(cornerRadius: 12)
                                    .stroke(ApolloPalette.destructive.opacity(0.9), lineWidth: 1)
                            )
                            .foregroundStyle(ApolloPalette.destructive.opacity(0.98))
                        }
                    }
                }
                .padding()
            }
            .navigationTitle(settings.localized("feature_settings_title"))
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button(settings.localized("close")) { dismiss() }
                }
            }
        }
    }
}
