import Foundation
import SwiftUI
import UniformTypeIdentifiers
@preconcurrency import MediaGenerationKit
@preconcurrency import ModelZoo
import ModelOp

struct ImageExtra: Codable, Identifiable, Sendable {
    enum Kind: String, CaseIterable, Codable, Identifiable {
        case lora, clipL, clipG, textEncoder, vae, embedding, tags, definitions, translations
        var id: String { rawValue }
        var label: String {
            switch self {
            case .lora: return "LoRA"
            case .clipL: return "CLIP-L"
            case .clipG: return "CLIP-G"
            case .textEncoder: return "image_extras_text_encoder"
            case .vae: return "VAE"
            case .embedding: return "image_extras_embedding"
            case .tags: return "image_extras_tags"
            case .definitions: return "image_extras_definitions"
            case .translations: return "image_extras_translations"
            }
        }
        var isPromptResource: Bool { self == .tags || self == .definitions || self == .translations }
        var isEncoder: Bool { self == .clipL || self == .clipG || self == .textEncoder || self == .vae }
    }
    var id = UUID().uuidString
    var name: String
    var file: String
    var kind: Kind
    var weight: Float = 1
    var enabled = true
    var tags: [String] = []
    var mappings: [String: String] = [:]
    var keyword: String?
}

struct ImageGenerationOptions: Codable, Sendable {
    var negativePrompt = "ugly, blurry, bad anatomy, bad quality"
    var guidance: Float = 7
    var clipSkip = 1
    var assets: [ImageExtra] = []

    var encoderFingerprint: String {
        assets.filter { $0.enabled && ($0.kind.isEncoder || $0.kind == .embedding) }
            .map { $0.kind.rawValue + ":" + $0.file }.sorted().joined(separator: "|")
    }
    func translatedPrompt(_ prompt: String) -> String {
        let mappings = assets.filter { $0.enabled && $0.kind == .translations }
            .reduce(into: [String: String]()) { $0.merge($1.mappings) { _, new in new } }
        let keys = mappings.keys.filter { !$0.isEmpty }.sorted { $0.count > $1.count }
        guard !keys.isEmpty else { return prompt }
        let pattern = "(?<![\\p{L}\\p{N}_])(?:" + keys.map(NSRegularExpression.escapedPattern(for:)).joined(separator: "|") + ")(?![\\p{L}\\p{N}_])"
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return prompt }
        let result = NSMutableString(string: prompt)
        for match in regex.matches(in: prompt, range: NSRange(prompt.startIndex..., in: prompt)).reversed() {
            if let replacement = mappings[(prompt as NSString).substring(with: match.range)] {
                result.replaceCharacters(in: match.range, with: replacement)
            }
        }
        return result as String
    }
}

@MainActor
final class ImageExtrasStore: ObservableObject {
    static let shared = ImageExtrasStore()
    @Published var options: ImageGenerationOptions { didSet { save() } }
    private let key = "image_generation_extras_v1"
    private init() {
        options = UserDefaults.standard.data(forKey: key)
            .flatMap { try? JSONDecoder().decode(ImageGenerationOptions.self, from: $0) } ?? ImageGenerationOptions()
    }
    private func save() {
        if let data = try? JSONEncoder().encode(options) { UserDefaults.standard.set(data, forKey: key) }
    }
    func remove(_ id: String) {
        let asset = options.assets.first { $0.id == id }
        options.assets.removeAll { $0.id == id }
        if let file = asset?.file, file.hasPrefix("extra_"), !file.contains("/") {
            try? FileManager.default.removeItem(at: ModelZoo.persistentModelsDirectory().appendingPathComponent(file))
        }
    }

    func importFile(_ source: URL, kind: ImageExtra.Kind, modelID: String?, name displayName: String? = nil) async throws {
        let accessing = source.startAccessingSecurityScopedResource()
        defer { if accessing { source.stopAccessingSecurityScopedResource() } }
        let trimmed = displayName?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        let name = trimmed.isEmpty ? source.deletingPathExtension().lastPathComponent : trimmed
        var asset = ImageExtra(name: name, file: "", kind: kind)
        if kind.isPromptResource {
            let values = try source.resourceValues(forKeys: [.fileSizeKey])
            guard (values.fileSize ?? 0) <= 2_000_000 else { throw CocoaError(.fileReadTooLarge) }
            let text = try String(contentsOf: source, encoding: .utf8)
            if kind == .tags {
                if let list = try? JSONDecoder().decode([String].self, from: Data(text.utf8)) { asset.tags = list }
                else { asset.tags = text.components(separatedBy: .newlines).map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty } }
                guard !asset.tags.isEmpty else { throw CocoaError(.fileReadCorruptFile) }
            } else {
                if let mappings = try? JSONDecoder().decode([String: String].self, from: Data(text.utf8)) { asset.mappings = mappings }
                else {
                    for line in text.components(separatedBy: .newlines) where !line.trimmingCharacters(in: .whitespaces).isEmpty {
                        guard let separator = line.firstIndex(of: "=") else { throw CocoaError(.fileReadCorruptFile) }
                        let key = line[..<separator].trimmingCharacters(in: .whitespaces)
                        let value = line[line.index(after: separator)...].trimmingCharacters(in: .whitespaces)
                        guard !key.isEmpty else { throw CocoaError(.fileReadCorruptFile) }
                        asset.mappings[key] = value
                    }
                }
                guard !asset.mappings.isEmpty, !asset.mappings.keys.contains("") else { throw CocoaError(.fileReadCorruptFile) }
            }
        } else {
            let suffix = source.pathExtension.lowercased()
            guard suffix == "ckpt" || suffix == "safetensors" else { throw CocoaError(.fileReadUnsupportedScheme) }
            guard kind == .lora || kind == .embedding || suffix == "ckpt" else { throw CocoaError(.fileReadCorruptFile) }
            let file = "extra_" + asset.id + "_f16.ckpt"
            let destination = ModelZoo.persistentModelsDirectory().appendingPathComponent(file)
            do {
                if kind == .lora && suffix == "safetensors" {
                    let converted = try await Task.detached(priority: .userInitiated) {
                        var importer = LoRAImporter(file: source)
                        try importer.import(to: destination)
                        return importer.version?.rawValue
                    }.value
                    guard let converted, let version = ModelZoo.specificationForModel(modelID ?? "")?.version,
                          converted == version.rawValue else { throw CocoaError(.fileReadCorruptFile) }
                    LoRAZoo.appendCustomSpecification(.init(name: name, file: file, prefix: "", version: version))
                } else if kind == .embedding && suffix == "safetensors" {
                    let keyword = name.components(separatedBy: .whitespacesAndNewlines).joined(separator: "_").lowercased()
                    let (length, versionName) = try await Task.detached(priority: .userInitiated) {
                        let result = try EmbeddingImporter.import(downloadedFile: source.path, name: name, filename: file, keyword: keyword)
                        return (result.multivector, result.version.rawValue)
                    }.value
                    guard let version = ModelZoo.specificationForModel(modelID ?? "")?.version,
                          version.rawValue == versionName else { throw CocoaError(.fileReadCorruptFile) }
                    TextualInversionZoo.appendCustomSpecification(.init(name: name, file: file, keyword: keyword, length: length, version: version))
                    asset.keyword = keyword
                } else {
                    try await Task.detached(priority: .userInitiated) {
                        try FileManager.default.copyItem(at: source, to: destination)
                    }.value
                    if kind == .lora {
                        guard let version = ModelZoo.specificationForModel(modelID ?? "")?.version else { throw CocoaError(.fileReadCorruptFile) }
                        LoRAZoo.appendCustomSpecification(.init(name: name, file: file, prefix: "", version: version))
                    }
                    if kind == .embedding { throw CocoaError(.fileReadCorruptFile) }
                }
                asset.file = file
            } catch {
                try? FileManager.default.removeItem(at: destination)
                throw error
            }
        }
        if kind.isEncoder {
            for index in options.assets.indices where options.assets[index].kind == kind { options.assets[index].enabled = false }
        }
        options.assets.append(asset)
    }
}

// Shared with the existing steps control so modifier sliders use the same appearance.
struct ImageGenerationSettingsSlider: View {
    let title: String
    @Binding var value: Double
    let range: ClosedRange<Double>
    let step: Double
    let identifier: String
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title).font(.headline)
            Slider(value: $value, in: range, step: step)
                .tint(ApolloPalette.accentStrong)
                .accessibilityIdentifier(identifier)
        }
    }
}

struct ImageExtrasEditor: View {
    @EnvironmentObject private var settings: AppSettings
    @ObservedObject private var store = ImageExtrasStore.shared
    @Binding var prompt: String
    /// The image model add-ons attach to (the one selected in Image Generator).
    var modelID: String
    @State private var pendingKind: ImageExtra.Kind?
    @State private var showImporter = false
    @State private var importFailed = false

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text(settings.localized("image_extras_title")).font(.headline)
            VStack(alignment: .leading, spacing: 8) {
                Text(settings.localized("image_extras_negative")).font(.headline)
                TextEditor(text: $store.options.negativePrompt)
                    .frame(minHeight: 80)
                    .scrollContentBackground(.hidden)
                    .background(Color.white.opacity(0.02))
                    .clipShape(RoundedRectangle(cornerRadius: 12))
                    .accessibilityIdentifier("imageModifiers.negativePrompt")
            }
            ImageGenerationSettingsSlider(
                title: settings.localized("image_extras_guidance") + ": " + String(format: "%.1f", store.options.guidance),
                value: Binding(get: { Double(store.options.guidance) }, set: { store.options.guidance = Float($0) }),
                range: 0...20, step: 0.5, identifier: "imageModifiers.guidance"
            )
            ImageGenerationSettingsSlider(
                title: "\(settings.localized("image_extras_clip_skip")): \(store.options.clipSkip)",
                value: Binding(get: { Double(store.options.clipSkip) }, set: { store.options.clipSkip = Int($0) }),
                range: 1...12, step: 1, identifier: "imageModifiers.clipSkip"
            )
            // Add-ons are imported here, attached to the selected model: they do nothing on their
            // own. Prompt lists work with every model.
            Menu {
                ForEach(ImageExtra.Kind.allCases) { kind in
                    Button(settings.localized(kind.label)) {
                        pendingKind = kind
                        showImporter = true
                    }
                    .disabled(!kind.isPromptResource && modelID.isEmpty)
                }
            } label: {
                Label(settings.localized("image_extras_add"), systemImage: "plus")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.bordered)
            .accessibilityIdentifier("imageModifiers.add")
            .fileImporter(isPresented: $showImporter, allowedContentTypes: [.item]) { result in
                guard let kind = pendingKind, case .success(let url) = result else { return }
                pendingKind = nil
                Task {
                    do {
                        try await store.importFile(url, kind: kind, modelID: kind.isPromptResource ? nil : modelID)
                    } catch {
                        NSLog("[ImageExtras] Import failed: %@", error.localizedDescription)
                        importFailed = true
                    }
                }
            }
            .alert(settings.localized("image_extras_invalid"), isPresented: $importFailed) {
                Button(settings.localized("ok"), role: .cancel) {}
            }
            Text(settings.localized("image_extras_ios_help"))
                .font(.caption).foregroundStyle(.secondary)
            ImagePromptResources(prompt: $prompt)
            ForEach($store.options.assets) { $asset in
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        if asset.kind != .embedding {
                            Toggle(asset.name, isOn: $asset.enabled)
                                .onChange(of: asset.enabled) { _, enabled in
                                    if enabled && asset.kind.isEncoder {
                                        for index in store.options.assets.indices where store.options.assets[index].kind == asset.kind && store.options.assets[index].id != asset.id {
                                            store.options.assets[index].enabled = false
                                        }
                                    }
                                }
                        } else { Text(asset.name) }
                        Button(role: .destructive) { store.remove(asset.id) } label: { Image(systemName: "trash") }
                            .accessibilityLabel(settings.localized("delete"))
                    }
                    Text(settings.localized(asset.kind.label)).font(.caption).foregroundStyle(.secondary)
                    if asset.kind == .lora {
                        ImageGenerationSettingsSlider(
                            title: settings.localized("image_extras_weight") + ": " + String(format: "%.2f", asset.weight),
                            value: Binding(get: { Double(asset.weight) }, set: { asset.weight = Float($0) }),
                            range: -2...2, step: 0.05, identifier: "imageModifiers.weight." + asset.id
                        )
                    }
                }
            }
        }
        .tint(ApolloPalette.accentStrong)
    }
}

struct ImagePromptResources: View {
    @ObservedObject private var store = ImageExtrasStore.shared
    @Binding var prompt: String
    var body: some View {
        let assets = store.options.assets.filter { $0.enabled }
        let tags = assets.flatMap { asset -> [(String, String)] in
            if asset.kind == .tags { return asset.tags.map { ($0, $0) } }
            if asset.kind == .definitions { return asset.mappings.sorted { $0.key < $1.key }.map { ($0.key, $0.value) } }
            if let keyword = asset.keyword { return [(keyword, "<" + keyword + ">")] }
            return []
        }
        if !tags.isEmpty {
            ScrollView(.horizontal, showsIndicators: false) {
                LazyHStack {
                    ForEach(Array(tags.enumerated()), id: \.offset) { _, tag in
                        Button { prompt += (prompt.isEmpty ? "" : ", ") + tag.1 } label: {
                            Text(tag.0).padding(.horizontal, 12).padding(.vertical, 8)
                        }
                        .liquidGlassPrimaryButton(cornerRadius: 12)
                    }
                }
            }
        }
    }
}
