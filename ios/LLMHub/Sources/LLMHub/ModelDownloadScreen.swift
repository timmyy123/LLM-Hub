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
        guard let documentsDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else { return }
        let modelsDir = documentsDir.appendingPathComponent("models")
        
        for model in models {
            let modelDir = modelsDir.appendingPathComponent(model.id)
            var allExist = true
            
            if model.files.isEmpty {
                allExist = false
            } else {
                for fileName in model.files {
                    let filePath = modelDir.appendingPathComponent(fileName)
                    if !FileManager.default.fileExists(atPath: filePath.path) {
                        allExist = false
                        break
                    }
                }
            }
            
            if allExist {
                self.downloadStates[model.id] = .downloaded
            } else {
                // If it's currently downloading in this session, don't overwrite
                if case .downloading = downloadStates[model.id] {
                    continue
                }
                self.downloadStates[model.id] = .notDownloaded
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

    private var downloadTasks: [String: Task<Void, Never>] = [:]

    func startDownload(_ model: AIModel) {
        // Cancel existing task if any
        downloadTasks[model.id]?.cancel()
        
        let task = Task {
            guard let documentsDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else { return }
            let modelsDir = documentsDir.appendingPathComponent("models")
            let destinationDir = modelsDir.appendingPathComponent(model.id)
            
            await MainActor.run {
                downloadStates[model.id] = .downloading(progress: 0, downloaded: "0 MB", speed: "0 KB/s")
            }
            
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
                    self.downloadTasks.removeValue(forKey: model.id)
                }
            } catch is CancellationError {
                await MainActor.run {
                    self.downloadStates[model.id] = .paused
                }
            } catch let error as URLError where error.code == .cancelled {
                await MainActor.run {
                    self.downloadStates[model.id] = .paused
                }
            } catch {
                await MainActor.run {
                    self.downloadStates[model.id] = .error(message: error.localizedDescription)
                }
            }
        }
        downloadTasks[model.id] = task
    }

    func pauseDownload(_ id: String) {
        downloadTasks[id]?.cancel()
        downloadTasks.removeValue(forKey: id)
        downloadStates[id] = .paused
    }

    func resumeDownload(_ id: String) {
        if let model = models.first(where: { $0.id == id }) {
            startDownload(model)
        }
    }

    func deleteModel(_ id: String) {
        downloadTasks[id]?.cancel()
        downloadTasks.removeValue(forKey: id)
        
        let model = models.first(where: { $0.id == id })
        if let model = model {
            guard let documentsDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else { return }
            let modelsDir = documentsDir.appendingPathComponent("models")
            let destinationDir = modelsDir.appendingPathComponent(model.id)
            try? FileManager.default.removeItem(at: destinationDir)
            downloadStates[id] = .notDownloaded
        }
    }
}

// MARK: - Model Row View
struct ModelRowView: View {
    @EnvironmentObject var settings: AppSettings
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
                            Text(String(format: settings.localized("ram_requirement_format"), Int(model.requirements.minRamGB)))
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }

                        // Capability badges
                        HStack(spacing: 4) {
                            if model.supportsVision {
                                capabilityBadge(settings.localized("vision"), color: .purple)
                            }
                            if model.supportsAudio {
                                capabilityBadge(settings.localized("audio"), color: .orange)
                            }
                            capabilityBadge(settings.localized("text_only"), color: .indigo)
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
                        Text("\(settings.localized("downloading")) \(downloaded) / \(model.sizeLabel) (\(speed))")
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


                    // Model description removed per user request


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
                        case .notDownloaded:
                            Button(action: onDownload) {
                                Label(settings.localized("download"), systemImage: "arrow.down.circle.fill")
                                    .frame(maxWidth: .infinity)
                                    .padding(.vertical, 10)
                                    .background(.indigo.gradient)
                                    .foregroundColor(.white)
                                    .clipShape(RoundedRectangle(cornerRadius: 10))
                            }
                        case .error:
                            Button(action: onDownload) {
                                Label(settings.localized("retry"), systemImage: "arrow.clockwise")
                                    .frame(maxWidth: .infinity)
                                    .padding(.vertical, 10)
                                    .background(.red.gradient)
                                    .foregroundColor(.white)
                                    .clipShape(RoundedRectangle(cornerRadius: 10))
                            }
                        case .downloading:
                            Button(action: onPause) {
                                Label(settings.localized("pause_download"), systemImage: "pause.circle.fill")
                                    .frame(maxWidth: .infinity)
                                    .padding(.vertical, 10)
                                    .background(.orange.gradient)
                                    .foregroundColor(.white)
                                    .clipShape(RoundedRectangle(cornerRadius: 10))
                            }
                            Button(action: onDelete) {
                                Image(systemName: "xmark")
                                    .padding(.vertical, 10)
                                    .padding(.horizontal, 14)
                                    .background(Color.red.opacity(0.1))
                                    .foregroundColor(.red)
                                    .clipShape(RoundedRectangle(cornerRadius: 10))
                            }
                        case .paused:
                            Button(action: onResume) {
                                Label(settings.localized("resume_download"), systemImage: "play.circle.fill")
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
                                    .background(Color.red.opacity(0.1))
                                    .foregroundColor(.red)
                                    .clipShape(RoundedRectangle(cornerRadius: 10))
                            }
                        case .downloaded:
                            Button(action: {}) {
                                Label(settings.localized("reload_model"), systemImage: "play.circle.fill")
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
    @EnvironmentObject var settings: AppSettings
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
        case .notDownloaded: return settings.localized("not_downloaded")
        case .downloading: return settings.localized("downloading")
        case .paused: return settings.localized("paused")
        case .downloaded: return settings.localized("downloaded")
        case .error(let msg): return "\(settings.localized("error")): \(msg)"
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
    @EnvironmentObject var settings: AppSettings
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



            // Model list
            ScrollView {
                LazyVStack(spacing: 12) {
                    if vm.filteredModels.isEmpty {
                        VStack(spacing: 16) {
                            Image(systemName: "magnifyingglass")
                                .font(.system(size: 48))
                                .foregroundColor(.secondary.opacity(0.5))
                            Text(settings.localized("no_models_available"))
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
                .padding(.top, 24)
                .padding(.bottom, 24)
            }
        }
        .background(Color(.systemGroupedBackground))
        .navigationTitle(settings.localized("ai_models"))
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .navigationBarLeading) {
                Button {
                    onNavigateBack()
                } label: {
                    Image(systemName: "arrow.left")
                        .font(.headline)
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
    @EnvironmentObject var settings: AppSettings
    let category: ModelCategory
    let isSelected: Bool
    let count: Int
    let onTap: () -> Void

    var body: some View {
        Button(action: onTap) {
            HStack(spacing: 6) {
                Image(systemName: category.icon)
                    .font(.subheadline)
                Text(settings.localized(category.titleKey))
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

