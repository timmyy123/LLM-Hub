import SwiftUI

// MARK: - Download ViewModel
@MainActor
class ModelDownloadViewModel: ObservableObject {
    @Published var models: [AIModel] = ModelData.models
    @Published var selectedCategory: ModelCategory = .multimodal
    @Published var searchText: String = ""
    @Published var downloadStates: [String: DownloadState] = [:]
    @Published var expandedModelId: String? = nil

    init() {
        // Initialize with default states
        for model in ModelData.models {
            downloadStates[model.id] = .notDownloaded
        }
        refreshStatuses()
    }

    func refreshStatuses() {
        // Check filesystem for existing models
        let modelsDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!.appendingPathComponent("models")
        
        for model in ModelData.models {
            let modelDir = modelsDir.appendingPathComponent(model.id)
            let weightsFile = modelDir.appendingPathComponent("model.safetensors")
            if FileManager.default.fileExists(atPath: weightsFile.path) {
                downloadStates[model.id] = .downloaded
            } else {
                // If it's currently downloading in this session, don't overwrite with notDownloaded
                if case .downloading = downloadStates[model.id] {
                    continue
                }
                downloadStates[model.id] = .notDownloaded
            }
        }
    }

    var filteredModels: [AIModel] {
        let categoryFiltered = models.filter { $0.category == selectedCategory }
        if searchText.isEmpty { return categoryFiltered }
        return categoryFiltered.filter { $0.name.localizedCaseInsensitiveContains(searchText) || $0.description.localizedCaseInsensitiveContains(searchText) }
    }

    func toggleExpand(_ id: String) {
        if expandedModelId == id {
            expandedModelId = nil
        } else {
            expandedModelId = id
        }
    }

    func startDownload(_ model: AIModel) {
        downloadStates[model.id] = .downloading(progress: 0, downloaded: "0 MB", speed: "0 KB/s")
        
        Task {
            let modelsDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!.appendingPathComponent("models")
            let destinationDir = modelsDir.appendingPathComponent(model.id)
            
            do {
                try await ModelDownloader.shared.downloadModel(
                    model,
                    hfToken: nil,
                    destinationDir: destinationDir,
                    onProgress: { update in
                        Task { @MainActor in
                            let downloadedLabel = ByteCountFormatter.string(fromByteCount: update.bytesDownloaded, countStyle: .file)
                            let speedLabel = ByteCountFormatter.string(fromByteCount: Int64(update.speedBytesPerSecond), countStyle: .file) + "/s"
                            let progress = Double(update.bytesDownloaded) / Double(max(1, update.totalBytes))
                            self.downloadStates[model.id] = .downloading(progress: progress, downloaded: downloadedLabel, speed: speedLabel)
                        }
                    }
                )
                
                await MainActor.run {
                    self.downloadStates[model.id] = .downloaded
                }
            } catch {
                await MainActor.run {
                    self.downloadStates[model.id] = .error(message: error.localizedDescription)
                }
            }
        }
    }

    func pauseDownload(_ id: String) {
        // Future: implement pause in ModelDownloader
    }

    func resumeDownload(_ id: String) {
        if let model = models.first(where: { $0.id == id }) {
            startDownload(model)
        }
    }

    func deleteModel(_ id: String) {
        let model = models.first(where: { $0.id == id })
        if let model = model {
            let modelsDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!.appendingPathComponent("models")
            let destinationDir = modelsDir.appendingPathComponent(model.id)
            try? FileManager.default.removeItem(at: destinationDir)
            downloadStates[id] = .notDownloaded
        }
    }
}

// MARK: - Model Row View
struct ModelRowView: View {
    let model: AIModel
    let state: DownloadState
    let isExpanded: Bool
    let onDownload: () -> Void
    let onPause: () -> Void
    let onResume: () -> Void
    let onDelete: () -> Void
    let onExpand: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Header
            Button(action: onExpand) {
                HStack(spacing: 12) {
                    // Model icon
                    ZStack {
                        RoundedRectangle(cornerRadius: 10)
                            .fill(categoryGradient)
                            .frame(width: 44, height: 44)
                        Image(systemName: model.category.icon)
                            .font(.system(size: 20))
                            .foregroundColor(.white)
                    }

                    VStack(alignment: .leading, spacing: 3) {
                        Text(model.name)
                            .font(.subheadline.bold())
                            .foregroundColor(.primary)
                            .lineLimit(2)

                        HStack(spacing: 6) {
                            StatusBadge(state: state)
                            Text("•")
                                .foregroundColor(.secondary)
                            Text(model.sizeLabel)
                                .font(.caption)
                                .foregroundColor(.secondary)
                            Text("•")
                                .foregroundColor(.secondary)
                            Text(model.ramLabel)
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }

                        // Capability badges
                        HStack(spacing: 4) {
                            if model.supportsVision {
                                capabilityBadge(String(localized: "vision", bundle: .module), color: .purple)
                            }
                            if model.supportsAudio {
                                capabilityBadge(String(localized: "audio", bundle: .module), color: .orange)
                            }
                            capabilityBadge(String(localized: "text", bundle: .module), color: .indigo)
                        }
                    }

                    Spacer()

                    Image(systemName: isExpanded ? "chevron.up" : "chevron.down")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                .padding(.vertical, 12)
                .padding(.horizontal, 16)
                .contentShape(Rectangle()) // Makes the whole area clickable
            }
            .buttonStyle(.plain)

            // Download progress
            if case .downloading(let progress, let downloaded, let speed) = state {
                VStack(spacing: 4) {
                    ProgressView(value: progress)
                        .tint(.indigo)
                        .padding(.horizontal, 16)
                    HStack {
                        Text("Downloading... \(downloaded) / \(model.sizeLabel) (\(speed))")
                            .font(.caption)
                            .foregroundColor(.secondary)
                        Spacer()
                        Text("\(Int(progress * 100))%")
                            .font(.caption.bold())
                            .foregroundColor(.indigo)
                    }
                    .padding(.horizontal, 16)
                }
                .padding(.bottom, 8)
            }

            // Expanded details
            if isExpanded {
                VStack(alignment: .leading, spacing: 12) {
                    Divider()
                        .padding(.horizontal, 16)

                    Text(model.description)
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                        .padding(.horizontal, 16)

                    HStack(spacing: 6) {
                        Image(systemName: "link")
                            .font(.caption)
                            .foregroundColor(.indigo)
                        Text(model.repoId)
                            .font(.caption)
                            .foregroundColor(.indigo)
                            .lineLimit(1)
                    }
                    .padding(.horizontal, 16)

                    // Action buttons
                    HStack(spacing: 10) {
                        switch state {
                        case .notDownloaded, .error:
                            Button(action: onDownload) {
                                Label(String(localized: "download", bundle: .module), systemImage: "square.and.arrow.down")
                                    .frame(maxWidth: .infinity)
                                    .padding(.vertical, 10)
                                    .background(.indigo.gradient)
                                    .foregroundColor(.white)
                                    .clipShape(RoundedRectangle(cornerRadius: 10))
                            }
                        case .downloading:
                            Button(action: onPause) {
                                Label(String(localized: "cancel", bundle: .module), systemImage: "xmark.circle")
                                    .frame(maxWidth: .infinity)
                                    .padding(.vertical, 10)
                                    .background(.orange.gradient)
                                    .foregroundColor(.white)
                                    .clipShape(RoundedRectangle(cornerRadius: 10))
                            }
                        case .paused:
                            Button(action: onResume) {
                                Label(String(localized: "retry", bundle: .module), systemImage: "play.fill")
                                    .frame(maxWidth: .infinity)
                                    .padding(.vertical, 10)
                                    .background(.green.gradient)
                                    .foregroundColor(.white)
                                    .clipShape(RoundedRectangle(cornerRadius: 10))
                            }
                            Button(action: onDelete) {
                                Label(String(localized: "delete", bundle: .module), systemImage: "trash")
                                    .padding(.vertical, 10)
                                    .padding(.horizontal, 14)
                                    .background(.red.opacity(0.1))
                                    .foregroundColor(.red)
                                    .clipShape(RoundedRectangle(cornerRadius: 10))
                            }
                        case .downloaded:
                            Button(action: {}) {
                                Label(String(localized: "reload_model", bundle: .module), systemImage: "play.circle.fill")
                                    .frame(maxWidth: .infinity)
                                    .padding(.vertical, 10)
                                    .background(.green.gradient)
                                    .foregroundColor(.white)
                                    .clipShape(RoundedRectangle(cornerRadius: 10))
                            }
                            Button(action: onDelete) {
                                Image(systemName: "trash")
                                    .padding(.vertical, 10)
                                    .padding(.horizontal, 14)
                                    .background(.red.opacity(0.1))
                                    .foregroundColor(.red)
                                    .clipShape(RoundedRectangle(cornerRadius: 10))
                            }
                        }
                    }
                    .font(.subheadline.bold())
                    .padding(.horizontal, 16)
                    .padding(.bottom, 12)
                }
            }
        }
        .background(Color(.secondarySystemBackground))
        .clipShape(RoundedRectangle(cornerRadius: 14))
    }

    private var categoryGradient: LinearGradient {
        switch model.category {
        case .text:       return LinearGradient(colors: [.indigo, .blue], startPoint: .topLeading, endPoint: .bottomTrailing)
        case .multimodal: return LinearGradient(colors: [.purple, .pink], startPoint: .topLeading, endPoint: .bottomTrailing)
        case .embedding:  return LinearGradient(colors: [.teal, .cyan], startPoint: .topLeading, endPoint: .bottomTrailing)
        case .imageGeneration: return LinearGradient(colors: [.orange, .red], startPoint: .topLeading, endPoint: .bottomTrailing)
        }
    }

    private func capabilityBadge(_ text: String, color: Color) -> some View {
        Text(text)
            .font(.caption2)
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(color.opacity(0.15))
            .foregroundColor(color)
            .clipShape(Capsule())
    }
}

// MARK: - Status Badge
struct StatusBadge: View {
    let state: DownloadState

    var body: some View {
        HStack(spacing: 4) {
            Circle()
                .fill(dotColor)
                .frame(width: 6, height: 6)
            Text(label)
                .font(.caption)
                .foregroundColor(dotColor)
        }
    }

    private var label: String {
        switch state {
        case .notDownloaded:   return "Not downloaded"
        case .downloading:     return "Downloading..."
        case .paused:          return "Paused"
        case .downloaded:      return "Downloaded"
        case .error:           return "Error"
        }
    }

    private var dotColor: Color {
        switch state {
        case .notDownloaded:  return .secondary
        case .downloading:    return .indigo
        case .paused:         return .orange
        case .downloaded:     return .green
        case .error:          return .red
        }
    }
}

// MARK: - ModelDownloadScreen
struct ModelDownloadScreen: View {
    @StateObject private var vm = ModelDownloadViewModel()
    var onNavigateBack: () -> Void

    var body: some View {
        VStack(spacing: 0) {
            // Category Tabs
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 10) {
                    ForEach(ModelCategory.allCases, id: \.self) { cat in
                        CategoryTab(
                            category: cat,
                            isSelected: vm.selectedCategory == cat,
                            count: vm.models.filter { $0.category == cat }.count
                        ) {
                            withAnimation(.spring(response: 0.3)) {
                                vm.selectedCategory = cat
                            }
                        }
                    }
                }
                .padding(.horizontal, 16)
                .padding(.vertical, 12)
            }
            .background(.thinMaterial)

            // Search
            HStack {
                Image(systemName: "magnifyingglass")
                    .foregroundColor(.secondary)
                TextField("Search models...", text: $vm.searchText)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background(Color(.secondarySystemBackground))
            .clipShape(RoundedRectangle(cornerRadius: 10))
            .padding(.horizontal, 16)
            .padding(.vertical, 8)

            // Category description
            HStack {
                Image(systemName: vm.selectedCategory.icon)
                    .foregroundColor(.indigo)
                Text(vm.selectedCategory.description)
                    .font(.subheadline)
                    .foregroundColor(.secondary)
                Spacer()
            }
            .padding(.horizontal, 16)
            .padding(.bottom, 8)

            // Model list
            ScrollView {
                LazyVStack(spacing: 10) {
                    if vm.filteredModels.isEmpty {
                        VStack(spacing: 16) {
                            Image(systemName: "magnifyingglass")
                                .font(.system(size: 48))
                                .foregroundColor(.secondary.opacity(0.5))
                            Text("No models found")
                                .foregroundColor(.secondary)
                        }
                        .padding(.top, 60)
                    } else {
                        ForEach(vm.filteredModels) { model in
                            ModelRowView(
                                model: model,
                                state: vm.downloadStates[model.id] ?? .notDownloaded,
                                isExpanded: vm.expandedModelId == model.id,
                                onDownload: { vm.startDownload(model) },
                                onPause:    { vm.pauseDownload(model.id) },
                                onResume:   { vm.resumeDownload(model.id) },
                                onDelete:   { vm.deleteModel(model.id) },
                                onExpand:   { vm.toggleExpand(model.id) }
                            )
                        }
                    }
                }
                .padding(.horizontal, 16)
                .padding(.bottom, 24)
            }
        }
        .background(Color(.systemGroupedBackground))
        .navigationTitle("AI Models")
        .navigationBarTitleDisplayMode(.large)
        .toolbar {
            ToolbarItem(placement: .navigationBarLeading) {
                Button {
                    onNavigateBack()
                } label: {
                    HStack(spacing: 4) {
                        Image(systemName: "chevron.left")
                        Text("Back")
                    }
                }
            }
        }
        .onAppear {
            vm.refreshStatuses()
        }
    }
}

// MARK: - Category Tab
struct CategoryTab: View {
    let category: ModelCategory
    let isSelected: Bool
    let count: Int
    let onTap: () -> Void

    var body: some View {
        Button(action: onTap) {
            HStack(spacing: 6) {
                Image(systemName: category.icon)
                    .font(.subheadline)
                Text(category.rawValue)
                    .font(.subheadline.bold())
                Text("\(count)")
                    .font(.caption)
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(isSelected ? Color.white.opacity(0.25) : Color.secondary.opacity(0.15))
                    .foregroundColor(isSelected ? .white : .secondary)
                    .clipShape(Capsule())
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 8)
            .background(isSelected ? AnyShapeStyle(.indigo.gradient) : AnyShapeStyle(Color(.secondarySystemBackground)))
            .foregroundColor(isSelected ? .white : .primary)
            .clipShape(RoundedRectangle(cornerRadius: 10))
        }
        .buttonStyle(.plain)
    }
}
