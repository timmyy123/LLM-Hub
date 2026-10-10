import SwiftUI
import AppKit

public struct ModelDownloadScreen: View {
    @EnvironmentObject var settings: AppSettings
    @ObservedObject private var modelManager = ModelManager.shared
    @ObservedObject private var downloader = ModelDownloader.shared
    @ObservedObject private var backend = LLMBackend.shared

    @State private var searchText: String = ""
    @State private var selectedCategory: ModelCategoryFilter = .all
    @State private var hoveredModelId: String? = nil

    var onShowPremium: () -> Void

    public init(onShowPremium: @escaping () -> Void) {
        self.onShowPremium = onShowPremium
    }

    public enum ModelCategoryFilter: String, CaseIterable, Identifiable {
        case all = "All"
        case llm = "LLM"
        case vision = "Vision"
        case code = "Code"
        case audio = "Audio"
        case agent = "Agent"

        public var id: String { rawValue }
    }

    public var filteredModels: [AIModel] {
        ModelData.models.filter { model in
            let matchesSearch = searchText.isEmpty ||
                model.name.localizedCaseInsensitiveContains(searchText) ||
                model.description.localizedCaseInsensitiveContains(searchText) ||
                model.tags.contains { $0.localizedCaseInsensitiveContains(searchText) }

            let matchesCat: Bool = {
                switch selectedCategory {
                case .all: return true
                case .llm: return model.category == .chat || model.category == .general
                case .vision: return model.category == .multimodal || model.category == .imageGen || model.category == .videoGen
                case .code: return model.category == .coding || model.tags.contains("code")
                case .audio: return model.category == .audio || model.category == .speechToText
                case .agent: return model.category == .agent || model.tags.contains("agent")
                }
            }()

            return matchesSearch && matchesCat
        }
    }

    public var body: some View {
        VStack(spacing: 0) {
            // Header Bar
            headerBar

            Divider().background(Color.white.opacity(0.08))

            // Storage & Quick Actions
            storageBanner

            Divider().background(Color.white.opacity(0.08))

            // Models Grid
            ScrollView {
                LazyVGrid(columns: [GridItem(.adaptive(minimum: 320, maximum: 440), spacing: 18)], spacing: 18) {
                    ForEach(filteredModels, id: \.id) { model in
                        modelCard(model)
                    }
                }
                .padding(24)
            }
        }
        .apolloScreenBackground()
    }

    // MARK: - Header
    private var headerBar: some View {
        VStack(spacing: 14) {
            HStack(spacing: 16) {
                VStack(alignment: .leading, spacing: 2) {
                    Text(settings.localized("models_title"))
                        .font(.system(size: 24, weight: .bold))
                        .foregroundColor(.white)
                    Text("Download and manage local models with zero internet inference")
                        .font(.system(size: 13))
                        .foregroundColor(.white.opacity(0.6))
                }

                Spacer()

                // Import Custom Model Button
                Button {
                    importCustomModel()
                } label: {
                    HStack(spacing: 6) {
                        Image(systemName: "plus.circle.fill")
                        Text("Import Local GGUF...")
                            .font(.system(size: 13, weight: .semibold))
                    }
                    .padding(.horizontal, 14)
                    .padding(.vertical, 8)
                    .background(Color.white.opacity(0.08))
                    .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color.white.opacity(0.15), lineWidth: 1))
                    .cornerRadius(8)
                }
                .buttonStyle(.plain)
            }

            // Search Bar & Filter Chips
            HStack(spacing: 12) {
                HStack(spacing: 8) {
                    Image(systemName: "magnifyingglass")
                        .foregroundColor(.white.opacity(0.5))
                    TextField("Search models, quants, architectures...", text: $searchText)
                        .textFieldStyle(.plain)
                        .font(.system(size: 13))
                        .foregroundColor(.white)
                    if !searchText.isEmpty {
                        Button {
                            searchText = ""
                        } label: {
                            Image(systemName: "xmark.circle.fill")
                                .foregroundColor(.white.opacity(0.5))
                        }
                        .buttonStyle(.plain)
                    }
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .background(Color(hex: "090d16"))
                .cornerRadius(8)
                .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color.white.opacity(0.12), lineWidth: 1))

                // Categories
                HStack(spacing: 6) {
                    ForEach(ModelCategoryFilter.allCases) { cat in
                        Button {
                            selectedCategory = cat
                        } label: {
                            Text(cat.rawValue)
                                .font(.system(size: 12, weight: selectedCategory == cat ? .semibold : .regular))
                                .padding(.horizontal, 12)
                                .padding(.vertical, 6)
                                .background(selectedCategory == cat ? ApolloPalette.accent : Color.white.opacity(0.06))
                                .foregroundColor(selectedCategory == cat ? .black : .white.opacity(0.8))
                                .cornerRadius(6)
                        }
                        .buttonStyle(.plain)
                    }
                }
            }
        }
        .padding(20)
        .background(Color(hex: "0d121c"))
    }

    // MARK: - Storage Banner
    private var storageBanner: some View {
        HStack(spacing: 24) {
            let downloaded = modelManager.downloadedModels
            let totalBytes = downloaded.reduce(0) { $0 + (modelManager.fileSize(for: $1) ?? 0) }

            HStack(spacing: 8) {
                Image(systemName: "internaldrive")
                    .foregroundColor(ApolloPalette.accentStrong)
                Text("\(downloaded.count) Models Downloaded")
                    .font(.system(size: 13, weight: .medium))
                    .foregroundColor(.white)
            }

            HStack(spacing: 8) {
                Image(systemName: "chart.bar.fill")
                    .foregroundColor(ApolloPalette.accent)
                Text("Total Disk Space: \(ByteCountFormatter.string(fromByteCount: totalBytes, countStyle: .file))")
                    .font(.system(size: 13))
                    .foregroundColor(.white.opacity(0.7))
            }

            Spacer()

            Button("Reveal Models Folder") {
                if let dir = modelManager.modelsDirectoryURL {
                    NSWorkspace.shared.activateFileViewerSelecting([dir])
                }
            }
            .buttonStyle(ApolloSecondaryButtonStyle())
        }
        .padding(.horizontal, 24)
        .padding(.vertical, 10)
        .background(Color(hex: "101624"))
    }

    // MARK: - Model Card
    private func modelCard(_ model: AIModel) -> some View {
        let isDownloaded = modelManager.isDownloaded(model)
        let isDownloading = downloader.isDownloading(model)
        let isActive = backend.currentlyLoadedModel == model.name

        return VStack(alignment: .leading, spacing: 12) {
            // Title & Badges
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 4) {
                    HStack(spacing: 8) {
                        Text(model.name)
                            .font(.system(size: 16, weight: .bold))
                            .foregroundColor(.white)

                        if isActive {
                            Text("ACTIVE")
                                .font(.system(size: 9, weight: .bold))
                                .padding(.horizontal, 6)
                                .padding(.vertical, 2)
                                .background(Color.green)
                                .foregroundColor(.black)
                                .cornerRadius(4)
                        }
                    }

                    if let quant = model.quantization {
                        Text(quant.rawValue)
                            .font(.system(size: 11, weight: .semibold, design: .monospaced))
                            .foregroundColor(ApolloPalette.accentStrong)
                    }
                }

                Spacer()

                // RAM & File Size
                VStack(alignment: .trailing, spacing: 2) {
                    Text(model.formattedSize)
                        .font(.system(size: 13, weight: .semibold))
                        .foregroundColor(.white.opacity(0.9))

                    if let ram = model.requiredRAM {
                        Text("\(ram) RAM")
                            .font(.system(size: 10))
                            .foregroundColor(.white.opacity(0.5))
                    }
                }
            }

            Text(model.description)
                .font(.system(size: 12))
                .foregroundColor(.white.opacity(0.7))
                .lineLimit(3)
                .fixedSize(horizontal: false, vertical: true)

            // Tags
            HStack(spacing: 6) {
                ForEach(model.tags.prefix(3), id: \.self) { tag in
                    Text("#\(tag)")
                        .font(.system(size: 10))
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(Color.white.opacity(0.06))
                        .foregroundColor(.white.opacity(0.5))
                        .cornerRadius(4)
                }
            }

            Divider().background(Color.white.opacity(0.06))

            // Action Row
            if isDownloading {
                // Download progress
                VStack(alignment: .leading, spacing: 6) {
                    let progress = downloader.progress(for: model)
                    HStack {
                        Text(downloader.statusText(for: model))
                            .font(.system(size: 11))
                            .foregroundColor(ApolloPalette.accentStrong)
                        Spacer()
                        Text("\(Int(progress * 100))%")
                            .font(.system(size: 11, weight: .semibold))
                            .foregroundColor(.white)
                    }

                    ProgressView(value: progress)
                        .tint(ApolloPalette.accent)

                    HStack {
                        Spacer()
                        Button("Cancel") {
                            downloader.cancel(model)
                        }
                        .font(.system(size: 11))
                        .foregroundColor(ApolloPalette.destructive)
                        .buttonStyle(.plain)
                    }
                }
            } else if isDownloaded {
                // Already downloaded
                HStack(spacing: 10) {
                    Button {
                        selectAsActiveModel(model)
                    } label: {
                        Text(isActive ? "Loaded in Memory" : "Set as Active")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(isActive ? ApolloSecondaryButtonStyle() : ApolloPrimaryButtonStyle())
                    .disabled(isActive)

                    // Reveal in Finder
                    Button {
                        if let fileURL = modelManager.localURL(for: model) {
                            NSWorkspace.shared.activateFileViewerSelecting([fileURL])
                        }
                    } label: {
                        Image(systemName: "folder")
                    }
                    .buttonStyle(ApolloSecondaryButtonStyle())
                    .help("Reveal in Finder")

                    // Delete model
                    Button {
                        deleteModel(model)
                    } label: {
                        Image(systemName: "trash")
                            .foregroundColor(ApolloPalette.destructive)
                    }
                    .buttonStyle(ApolloSecondaryButtonStyle())
                    .help("Delete Model File")
                }
            } else {
                // Not downloaded
                Button {
                    startDownload(model)
                } label: {
                    HStack(spacing: 6) {
                        Image(systemName: "arrow.down.circle.fill")
                        Text("Download Model")
                            .font(.system(size: 13, weight: .semibold))
                    }
                    .frame(maxWidth: .infinity)
                }
                .buttonStyle(ApolloPrimaryButtonStyle())
            }
        }
        .padding(18)
        .background(
            RoundedRectangle(cornerRadius: 14)
                .fill(ApolloPalette.bgSurface.opacity(0.8))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 14)
                .stroke(isActive ? ApolloPalette.accent.opacity(0.6) : ApolloPalette.borderGlass, lineWidth: 1)
        )
        .shadow(color: Color.black.opacity(0.2), radius: 6, x: 0, y: 3)
    }

    // MARK: - Actions
    private func selectAsActiveModel(_ model: AIModel) {
        Task {
            _ = try? await backend.loadModel(model, contextWindow: 4096)
        }
    }

    private func startDownload(_ model: AIModel) {
        downloader.startDownload(model)
    }

    private func deleteModel(_ model: AIModel) {
        modelManager.delete(model)
    }

    private func importCustomModel() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        panel.allowedContentTypes = [] // allow all model files

        if panel.runModal() == .OK, let url = panel.url {
            modelManager.importCustomModel(from: url)
        }
    }
}
