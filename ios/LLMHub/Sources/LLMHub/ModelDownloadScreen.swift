import SwiftUI
import RunAnywhere
import UniformTypeIdentifiers

struct ModelFamilyGroup: Identifiable {
    let title: String
    let models: [AIModel]
    var id: String { title }
}

private extension URLError.Code {
    var isTransientDownloadFailure: Bool {
        switch self {
        case .networkConnectionLost,
            .notConnectedToInternet,
            .timedOut,
            .cannotConnectToHost,
            .cannotFindHost,
            .dnsLookupFailed,
            .resourceUnavailable,
            .internationalRoamingOff,
            .callIsActive,
            .dataNotAllowed:
            return true
        default:
            return false
        }
    }
}

// MARK: - Download ViewModel
@MainActor
class ModelDownloadViewModel: ObservableObject {
    @Published var models: [AIModel] = ModelData.models
    @Published var selectedCategory: ModelCategory = .multimodal
    @Published var searchText: String = ""
    @Published var downloadStates: [String: DownloadState] = [:]
    @Published var expandedModelId: String? = nil
    @Published var expandedFamilyTitle: String? = nil
    private let pendingDownloadsKey = "ios_pending_model_download_ids"

    private func legacyModelDirectory(for model: AIModel) -> URL? {
        guard let documentsDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else { return nil }
        return documentsDir.appendingPathComponent("models").appendingPathComponent(model.id)
    }

    private func destinationDirectory(for model: AIModel) throws -> URL {
        if model.isCoreMLImageGeneration {
            guard let dir = StableDiffusionBackend.sdModelDirectory(for: model.id) else {
                throw NSError(domain: "ModelDownload", code: -3, userInfo: [NSLocalizedDescriptionKey: "Cannot resolve documents directory"])
            }
            return dir
        }
        return try SimplifiedFileManager.shared.getModelFolderURL(modelId: model.id, framework: model.inferenceFramework)
    }

    private func requiredFilesExist(in directory: URL, for model: AIModel) -> (allExist: Bool, totalBytes: Int64) {
        // CoreML models: check for sentinel file written after ZIP extraction
        if model.isCoreMLImageGeneration {
            let sentinel = directory.appendingPathComponent("_downloaded")
            let exists = FileManager.default.fileExists(atPath: sentinel.path)
            return (exists, exists ? model.sizeBytes : 0)
        }

        var allExist = true
        var totalLocalBytes: Int64 = 0

        if model.requiredFileNames.isEmpty {
            return (false, 0)
        }

        for fileName in model.requiredFileNames {
            let filePath = directory.appendingPathComponent(fileName)
            if !FileManager.default.fileExists(atPath: filePath.path) {
                allExist = false
                break
            }
            if let attrs = try? FileManager.default.attributesOfItem(atPath: filePath.path),
               let fileSize = attrs[.size] as? Int64 {
                totalLocalBytes += fileSize
            }
        }

        return (allExist, totalLocalBytes)
    }

    private func verifiedInstallMarkerExists(in directory: URL, for model: AIModel) -> Bool {
        let markerURL = ModelDownloader.installMarkerURL(for: directory)
        guard let data = try? Data(contentsOf: markerURL) else { return false }

        struct ModelInstallMarker: Codable {
            let version: Int
            let modelId: String
            let totalBytes: Int64
            let fileNames: [String]
        }

        guard let marker = try? JSONDecoder().decode(ModelInstallMarker.self, from: data) else {
            return false
        }

        return marker.version == 1
            && marker.modelId == model.id
            && marker.fileNames == model.requiredFileNames
    }

    private func backfillVerifiedInstallMarker(in directory: URL, for model: AIModel, totalBytes: Int64) {
        struct ModelInstallMarker: Codable {
            let version: Int
            let modelId: String
            let totalBytes: Int64
            let fileNames: [String]
        }

        let marker = ModelInstallMarker(
            version: 1,
            modelId: model.id,
            totalBytes: totalBytes,
            fileNames: model.requiredFileNames
        )

        guard let data = try? JSONEncoder().encode(marker) else { return }
        let markerURL = ModelDownloader.installMarkerURL(for: directory)
        FileManager.default.createFile(atPath: markerURL.path, contents: data)
    }

    // MARK: - Imported models persistence
    private let importedModelsKey = "imported_models_ios"

    static var importedModelsDirectory: URL? {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first?
            .appendingPathComponent("ImportedModels", isDirectory: true)
    }

    static func customModelDirectory(for modelId: String) -> URL {
        (try? SimplifiedFileManager.shared.getModelFolderURL(modelId: modelId, framework: .llamaCpp))
            ?? FileManager.default.temporaryDirectory.appendingPathComponent(modelId, isDirectory: true)
    }

    private static func migrateCustomModelIntoRunAnywhere(_ model: AIModel) -> AIModel {
        guard model.source == "Custom" else { return model }

        let destinationDir = customModelDirectory(for: model.id)
        try? FileManager.default.createDirectory(at: destinationDir, withIntermediateDirectories: true)

        let destinationModelURL = destinationDir.appendingPathComponent(URL(fileURLWithPath: model.url).lastPathComponent)
        if model.url != destinationModelURL.path,
           FileManager.default.fileExists(atPath: model.url),
           !FileManager.default.fileExists(atPath: destinationModelURL.path) {
            try? FileManager.default.copyItem(at: URL(fileURLWithPath: model.url), to: destinationModelURL)
        }

        let migratedAdditional = model.additionalFiles.map { path -> String in
            let sourceURL = URL(fileURLWithPath: path)
            let destinationURL = destinationDir.appendingPathComponent(sourceURL.lastPathComponent)
            if path != destinationURL.path,
               FileManager.default.fileExists(atPath: path),
               !FileManager.default.fileExists(atPath: destinationURL.path) {
                try? FileManager.default.copyItem(at: sourceURL, to: destinationURL)
            }
            return destinationURL.path
        }

        return AIModel(
            id: model.id, name: model.name, description: model.description,
            url: destinationModelURL.path, category: model.category, sizeBytes: model.sizeBytes,
            source: model.source, supportsVision: model.supportsVision,
            supportsAudio: model.supportsAudio, supportsThinking: model.supportsThinking,
            supportsGpu: model.supportsGpu, requirements: model.requirements,
            contextWindowSize: model.contextWindowSize, modelFormat: model.modelFormat,
            additionalFiles: migratedAdditional
        )
    }

    init() {
        do {
            try RunAnywhere.initialize(environment: .development)
        } catch {
            // Ignore repeated initialization attempts.
        }

        Task {
            _ = await RunAnywhere.discoverDownloadedModels()
        }

        // Initialize with default states for built-in models
        for model in ModelData.models {
            downloadStates[model.id] = .notDownloaded
        }

        // Load custom imported models and mark them as downloaded
        loadImportedModels()

        refreshStatuses()
        resumePendingDownloads()
    }

    // MARK: - Import External Model

    /// Adds an externally imported GGUF model (source == "Custom") to the model list.
    /// Returns false if a model with the same name already exists.
    @discardableResult
    func addExternalModel(_ model: AIModel) -> Bool {
        guard !models.contains(where: { $0.name == model.name }) else { return false }
        models.append(model)
        downloadStates[model.id] = .downloaded
        saveImportedModels()
        return true
    }

    /// Imports a vision projector file for an already-added custom model.
    /// Copies the projector to the model's directory and updates additionalFiles.
    func importVisionProjector(for modelId: String, fileName: String, from sourceURL: URL) {
        guard let idx = models.firstIndex(where: { $0.id == modelId }) else { return }
        let model = models[idx]

        let destDir = Self.customModelDirectory(for: model.id)
        try? FileManager.default.createDirectory(at: destDir, withIntermediateDirectories: true)

        let destFile = destDir.appendingPathComponent(fileName)
        try? FileManager.default.removeItem(at: destFile)
        try? FileManager.default.copyItem(at: sourceURL, to: destFile)

        // Rebuild the model with updated additionalFiles
        var files = model.additionalFiles
        if !files.contains(destFile.path) { files.append(destFile.path) }
        let updated = AIModel(
            id: model.id, name: model.name, description: model.description,
            url: model.url, category: model.category, sizeBytes: model.sizeBytes,
            source: model.source, supportsVision: model.supportsVision,
            supportsAudio: model.supportsAudio, supportsThinking: model.supportsThinking,
            supportsGpu: model.supportsGpu, requirements: model.requirements,
            contextWindowSize: model.contextWindowSize, modelFormat: model.modelFormat,
            additionalFiles: files
        )
        models[idx] = updated
        saveImportedModels()
    }

    private func loadImportedModels() {
        guard let data = UserDefaults.standard.data(forKey: importedModelsKey),
              let imported = try? JSONDecoder().decode([AIModel].self, from: data)
        else { return }

        var needsResave = false
        for raw in imported {
            guard !models.contains(where: { $0.id == raw.id }) else { continue }
            let model = Self.migrateCustomModelIntoRunAnywhere(ModelData.normalizeCustomModel(raw))
            if model.url != raw.url || model.additionalFiles != raw.additionalFiles { needsResave = true }
            models.append(model)
            downloadStates[model.id] = .downloaded
        }
        if needsResave { saveImportedModels() }
    }

    /// Re-root absolute paths after iOS container relocation.
    private static func rerootCustomPaths(_ model: AIModel) -> AIModel {
        guard model.source == "Custom" else { return model }
        let fixedURL = rerootPath(model.url)
        let fixedAdditional = model.additionalFiles.map { rerootPath($0) }
        guard fixedURL != model.url || fixedAdditional != model.additionalFiles else { return model }
        return AIModel(
            id: model.id, name: model.name, description: model.description,
            url: fixedURL, category: model.category, sizeBytes: model.sizeBytes,
            source: model.source, supportsVision: model.supportsVision,
            supportsAudio: model.supportsAudio, supportsThinking: model.supportsThinking,
            supportsGpu: model.supportsGpu, requirements: model.requirements,
            contextWindowSize: model.contextWindowSize, modelFormat: model.modelFormat,
            additionalFiles: fixedAdditional
        )
    }

    private static func rerootPath(_ storedPath: String) -> String {
        guard let docsDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else { return storedPath }
        if let marker = storedPath.range(of: "/Documents/ImportedModels/") {
            let relativeSuffix = String(storedPath[marker.upperBound...])
            return docsDir.appendingPathComponent("ImportedModels").appendingPathComponent(relativeSuffix).path
        }
        if let marker = storedPath.range(of: "/Documents/RunAnywhere/") {
            let relativeSuffix = String(storedPath[marker.upperBound...])
            return docsDir.appendingPathComponent("RunAnywhere").appendingPathComponent(relativeSuffix).path
        }
        return storedPath
    }

    private func saveImportedModels() {
        let imported = models.filter { $0.source == "Custom" }
        if let data = try? JSONEncoder().encode(imported) {
            UserDefaults.standard.set(data, forKey: importedModelsKey)
        }
    }

    private func loadPendingDownloadIDs() -> Set<String> {
        let ids = UserDefaults.standard.stringArray(forKey: pendingDownloadsKey) ?? []
        return Set(ids)
    }

    private func savePendingDownloadIDs(_ ids: Set<String>) {
        UserDefaults.standard.set(Array(ids), forKey: pendingDownloadsKey)
    }

    private func markPending(_ id: String) {
        var ids = loadPendingDownloadIDs()
        ids.insert(id)
        savePendingDownloadIDs(ids)
    }

    private func clearPending(_ id: String) {
        var ids = loadPendingDownloadIDs()
        ids.remove(id)
        savePendingDownloadIDs(ids)
    }

    func refreshStatuses() {
        let pendingIDs = loadPendingDownloadIDs()

        for model in models {
            // Custom imported models manage their own state via loadImportedModels/addExternalModel
            if model.source == "Custom" { continue }

            var allExist = false
            var totalLocalBytes: Int64 = 0
            var hasVerifiedInstallMarker = false
            var markerDirectory: URL?

            if let runAnywhereDir = try? destinationDirectory(for: model),
               FileManager.default.fileExists(atPath: runAnywhereDir.path) {
                let status = requiredFilesExist(in: runAnywhereDir, for: model)
                allExist = status.allExist
                totalLocalBytes = status.totalBytes
                if allExist {
                    hasVerifiedInstallMarker = verifiedInstallMarkerExists(in: runAnywhereDir, for: model)
                    markerDirectory = runAnywhereDir
                }
            }

            if !allExist, let legacyDir = legacyModelDirectory(for: model), FileManager.default.fileExists(atPath: legacyDir.path) {
                let status = requiredFilesExist(in: legacyDir, for: model)
                allExist = status.allExist
                totalLocalBytes = max(totalLocalBytes, status.totalBytes)
                if allExist {
                    hasVerifiedInstallMarker = verifiedInstallMarkerExists(in: legacyDir, for: model)
                    markerDirectory = legacyDir
                }
            }
            
            // Be conservative after interruption: if a model is still marked pending, force it
            // back through the downloader's exact per-file verification instead of trusting a
            // near-complete local byte count.
            if pendingIDs.contains(model.id) {
                // If it's currently downloading in this session, don't overwrite.
                if case .downloading = downloadStates[model.id] {
                    continue
                }
                self.downloadStates[model.id] = totalLocalBytes > 0 ? .paused : .notDownloaded
                continue
            }

            // Use the explicit verified install marker written only after full per-file validation.
            // For legacy installs created before this marker existed, backfill it only on exact
            // byte matches so near-complete interrupted downloads are never promoted to downloaded.
            if allExist && !hasVerifiedInstallMarker && totalLocalBytes == model.sizeBytes, let markerDirectory {
                backfillVerifiedInstallMarker(in: markerDirectory, for: model, totalBytes: totalLocalBytes)
                hasVerifiedInstallMarker = true
            }

            if allExist && hasVerifiedInstallMarker {
                self.downloadStates[model.id] = .downloaded
            } else {
                // If it's currently downloading in this session, don't overwrite
                if case .downloading = downloadStates[model.id] {
                    continue
                }
                self.downloadStates[model.id] = totalLocalBytes > 0 ? .paused : .notDownloaded
            }
        }
    }

    var filteredModels: [AIModel] {
        let categoryFiltered = models.filter { $0.category == selectedCategory }
        if searchText.isEmpty { return categoryFiltered }
        return categoryFiltered.filter { $0.name.localizedCaseInsensitiveContains(searchText) || $0.description.localizedCaseInsensitiveContains(searchText) }
    }

    var groupedFilteredModels: [ModelFamilyGroup] {
        let grouped = Dictionary(grouping: filteredModels, by: familyName(for:))
        return grouped
            .map { key, value in
                let sortedModels = value.sorted { lhs, rhs in
                    quantizationLabel(for: lhs).localizedCaseInsensitiveCompare(quantizationLabel(for: rhs)) == .orderedAscending
                }
                return ModelFamilyGroup(title: key, models: sortedModels)
            }
            .sorted { $0.title.localizedCaseInsensitiveCompare($1.title) == .orderedAscending }
    }

    func toggleFamily(_ title: String) {
        if expandedFamilyTitle == title {
            expandedFamilyTitle = nil
        } else {
            expandedFamilyTitle = title
        }
    }

    func quantizationLabel(for model: AIModel) -> String {
        guard let openIndex = model.name.lastIndex(of: "("),
              let closeIndex = model.name.lastIndex(of: ")"),
              openIndex < closeIndex else {
            return model.name
        }
        return String(model.name[model.name.index(after: openIndex)..<closeIndex]).trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func familyName(for model: AIModel) -> String {
        guard let openIndex = model.name.lastIndex(of: "("),
              let closeIndex = model.name.lastIndex(of: ")"),
              openIndex < closeIndex else {
            return model.name
        }

        let suffix = model.name[model.name.index(after: openIndex)..<closeIndex]
        let suffixUpper = suffix.uppercased().replacingOccurrences(of: "-", with: "_")
        let likelyQuant = suffixUpper.contains("Q") || suffixUpper.contains("IQ") || suffixUpper.contains("F16") || suffixUpper.contains("BF16") || suffixUpper.contains("INT") || suffixUpper.contains("FP16") || suffixUpper.contains("FP8")
        if likelyQuant {
            return model.name[..<openIndex].trimmingCharacters(in: .whitespacesAndNewlines)
        }
        return model.name
    }

    func toggleExpand(_ id: String) {
        if expandedModelId == id {
            expandedModelId = nil
        } else {
            expandedModelId = id
        }
    }

    private var downloadTasks: [String: Task<Void, Never>] = [:]
    private var autoResumableDownloads: Set<String> = []

    func startDownload(_ model: AIModel) {
        // Cancel existing task if any
        downloadTasks[model.id]?.cancel()
        markPending(model.id)
        
        let task = Task {
            do {
                try RunAnywhere.initialize(environment: .development)
            } catch {
                // Initialization may already be in progress/complete in other flows.
            }

            let destinationDir: URL
            do {
                destinationDir = try destinationDirectory(for: model)
            } catch {
                await MainActor.run {
                    self.downloadStates[model.id] = .error(message: error.localizedDescription)
                    self.clearPending(model.id)
                }
                return
            }
            
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
                    self.clearPending(model.id)
                    self.refreshStatuses()
                }

                _ = await RunAnywhere.discoverDownloadedModels()
            } catch is CancellationError {
                await MainActor.run {
                    self.downloadStates[model.id] = .paused
                }
            } catch let error as URLError where error.code == .cancelled {
                await MainActor.run {
                    self.downloadStates[model.id] = .paused
                }
            } catch let error as URLError where error.code.isTransientDownloadFailure {
                await MainActor.run {
                    // Treat temporary connection drops as recoverable; avoid surfacing noisy errors.
                    self.downloadStates[model.id] = .paused
                    self.autoResumableDownloads.insert(model.id)
                }
            } catch let error as NSError where error.domain == "ModelDownloader" && error.code == -2 {
                await MainActor.run {
                    // Incomplete file set after interruption is recoverable by resuming.
                    self.downloadStates[model.id] = .paused
                    self.autoResumableDownloads.insert(model.id)
                }
            } catch {
                await MainActor.run {
                    self.downloadStates[model.id] = .error(message: error.localizedDescription)
                    self.clearPending(model.id)
                }
            }
        }
        downloadTasks[model.id] = task
    }

    func pauseDownload(_ id: String) {
        downloadTasks[id]?.cancel()
        downloadTasks.removeValue(forKey: id)
        downloadStates[id] = .paused
        clearPending(id)
    }

    func resumeDownload(_ id: String) {
        if let model = models.first(where: { $0.id == id }) {
            autoResumableDownloads.remove(id)
            startDownload(model)
        }
    }

    func resumeAutoResumableDownloads() {
        let ids = autoResumableDownloads
        autoResumableDownloads.removeAll()
        for id in ids {
            if case .paused = downloadStates[id], let model = models.first(where: { $0.id == id }) {
                startDownload(model)
            }
        }
    }

    func resumePendingDownloads() {
        let ids = loadPendingDownloadIDs()
        for id in ids {
            guard downloadTasks[id] == nil,
                  let model = models.first(where: { $0.id == id }) else {
                continue
            }

            switch downloadStates[id] {
            case .downloaded:
                clearPending(id)
            case .downloading:
                continue
            case .paused, .notDownloaded, .error, .none:
                startDownload(model)
            }
        }
    }

    func deleteModel(_ id: String) {
        downloadTasks[id]?.cancel()
        downloadTasks.removeValue(forKey: id)
        clearPending(id)

        let model = models.first(where: { $0.id == id })
        if let model = model {
            if model.source == "Custom" {
                let dir = Self.customModelDirectory(for: model.id)
                try? FileManager.default.removeItem(at: dir)
                models.removeAll { $0.id == id }
                saveImportedModels()
            } else {
                if let destinationDir = try? destinationDirectory(for: model) {
                    try? FileManager.default.removeItem(at: destinationDir)
                }
                if let legacyDir = legacyModelDirectory(for: model) {
                    try? FileManager.default.removeItem(at: legacyDir)
                }
            }
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
                    VStack(alignment: .leading, spacing: 3) {
                        Text(model.name)
                            .font(.subheadline.bold())
                            .foregroundColor(.white)
                            .lineLimit(2)

                        HStack(spacing: 6) {
                            StatusBadge(state: state)
                            Text("•")
                                .foregroundColor(.secondary)
                            Text(model.sizeLabel)
                                .font(.caption)
                                .foregroundColor(.white.opacity(0.62))
                            Text("•")
                                .foregroundColor(.white.opacity(0.52))
                            Text(String(format: settings.localized("ram_requirement_format"), Int(model.requirements.minRamGB)))
                                .font(.caption)
                                .foregroundColor(.white.opacity(0.62))
                        }

                        // Capability badges
                        HStack(spacing: 4) {
                            if model.supportsVision {
                                capabilityBadge(settings.localized("vision"), color: ApolloPalette.accentStrong)
                            }
                            if model.supportsAudio {
                                capabilityBadge(settings.localized("audio"), color: ApolloPalette.warning)
                            }
                            if !model.supportsVision && !model.supportsAudio {
                                capabilityBadge(settings.localized("text_only"), color: ApolloPalette.accent)
                            }
                        }
                    }

                    Spacer()

                    Image(systemName: isExpanded ? "chevron.up" : "chevron.down")
                        .font(.caption)
                        .foregroundColor(.white.opacity(0.56))
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
                        .tint(ApolloPalette.accentStrong)
                        .padding(.horizontal, 16)
                    HStack {
                        Text("\(settings.localized("downloading")) \(downloaded) / \(model.sizeLabel) (\(speed))")
                            .font(.caption)
                            .foregroundColor(.secondary)
                        Spacer()
                        Text("\(Int(progress * 100))%")
                            .font(.caption.bold())
                            .foregroundColor(ApolloPalette.accentStrong)
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
                        Text(model.url)
                            .font(.caption)
                            .foregroundColor(ApolloPalette.accentStrong)
                            .lineLimit(1)
                    }
                    .padding(.horizontal, 16)

                    Text(settings.localized("model_download_interrupted_warning"))
                        .font(.caption)
                        .foregroundColor(Color.orange.opacity(0.95))
                        .lineLimit(nil)
                        .fixedSize(horizontal: false, vertical: true)
                        .padding(.horizontal, 16)

                    // Action buttons
                    HStack(spacing: 10) {
                        switch state {
                        case .notDownloaded:
                            Button(action: onDownload) {
                                Label(settings.localized("download"), systemImage: "arrow.down.circle.fill")
                                    .frame(maxWidth: .infinity)
                                    .padding(.vertical, 10)
                                    .background(
                                        LinearGradient(
                                            colors: [ApolloPalette.accentSoft, ApolloPalette.accentMuted],
                                            startPoint: .topLeading,
                                            endPoint: .bottomTrailing
                                        )
                                    )
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
                                    .background(ApolloPalette.warning.gradient)
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
                                    .background(ApolloPalette.accentSoft.gradient)
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
                            Button(action: onDelete) {
                                Label(settings.localized("action_delete"), systemImage: "trash")
                                    .frame(maxWidth: .infinity)
                                    .padding(.vertical, 10)
                                    .background(.red.gradient)
                                    .foregroundColor(.white)
                                    .clipShape(RoundedRectangle(cornerRadius: 10))
                            }
                        }
                    }
                    .font(.subheadline.bold())
                    .padding(.horizontal, 16)
                }
                .padding(.bottom, 12)
            }
        }
        .background(.ultraThinMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 14))
        .overlay(
            RoundedRectangle(cornerRadius: 14)
                .stroke(Color.white.opacity(0.16), lineWidth: 1)
        )
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
        case .error: return settings.localized("error")
        }
    }

    private var dotColor: Color {
        switch state {
        case .notDownloaded:  return .secondary
        case .downloading:    return ApolloPalette.accentStrong
        case .paused:         return ApolloPalette.warning
        case .downloaded:     return ApolloPalette.accentSoft
        case .error:          return ApolloPalette.destructive
        }
    }
}

// MARK: - ModelDownloadScreen
struct ModelDownloadScreen: View {
    @EnvironmentObject var settings: AppSettings
    @Environment(\.scenePhase) private var scenePhase
    @StateObject private var vm = ModelDownloadViewModel()
    @StateObject private var purchases = PurchaseManager.shared
    @State private var showImportSheet = false
    @State private var showPremiumForImport = false
    var onNavigateBack: () -> Void
    var onShowPremium: (() -> Void)? = nil

    var body: some View {
        ZStack {
            ApolloLiquidBackground()

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
                .background(.ultraThinMaterial)
                .overlay(alignment: .bottom) {
                    Rectangle()
                        .fill(Color.white.opacity(0.08))
                        .frame(height: 1)
                }


                // Model list
                ScrollView {
                    LazyVStack(spacing: 12) {
                        if vm.filteredModels.isEmpty {
                            VStack(spacing: 16) {
                                Image(systemName: "magnifyingglass")
                                    .font(.system(size: 48))
                                    .foregroundColor(.white.opacity(0.45))
                                Text(settings.localized("no_models_available"))
                                    .foregroundColor(.white.opacity(0.72))
                            }
                            .padding(.top, 60)
                        } else {
                            ForEach(vm.groupedFilteredModels) { family in
                                VStack(alignment: .leading, spacing: 10) {
                                    Button {
                                        withAnimation(.spring(response: 0.28, dampingFraction: 0.85)) {
                                            vm.toggleFamily(family.title)
                                        }
                                    } label: {
                                        HStack(spacing: 10) {
                                            Text(family.title)
                                                .font(.subheadline.bold())
                                                .foregroundColor(.white)
                                            Text("(\(family.models.count))")
                                                .font(.caption)
                                                .foregroundColor(.white.opacity(0.6))
                                            Spacer()
                                            Image(systemName: vm.expandedFamilyTitle == family.title ? "chevron.up" : "chevron.down")
                                                .font(.caption)
                                                .foregroundColor(.white.opacity(0.55))
                                        }
                                        .padding(.horizontal, 14)
                                        .padding(.vertical, 12)
                                        .background(.ultraThinMaterial)
                                        .clipShape(RoundedRectangle(cornerRadius: 12))
                                        .overlay(
                                            RoundedRectangle(cornerRadius: 12)
                                                .stroke(Color.white.opacity(0.14), lineWidth: 1)
                                        )
                                    }
                                    .buttonStyle(.plain)

                                    if vm.expandedFamilyTitle == family.title {
                                        ForEach(family.models) { model in
                                            VStack(alignment: .leading, spacing: 6) {
                                                Text(vm.quantizationLabel(for: model))
                                                    .font(.caption.bold())
                                                    .foregroundColor(.white.opacity(0.72))
                                                    .padding(.horizontal, 4)

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
                                }
                            }
                        }
                    }
                    .padding(.horizontal, 16)
                    .padding(.top, 24)
                    .padding(.bottom, 24)
                }
            }
        }
        .navigationTitle(settings.localized("ai_models"))
        .navigationBarTitleDisplayMode(.inline)
        .safeAreaInset(edge: .bottom, spacing: 0) { BannerAdContainer() }
        .toolbarBackground(.hidden, for: .navigationBar)
        .toolbar {
            ToolbarItem(placement: .navigationBarLeading) {
                Button {
                    onNavigateBack()
                } label: {
                    Image(systemName: "arrow.left")
                        .font(.headline)
                }
            }
            ToolbarItem(placement: .navigationBarTrailing) {
                Button {
                    if purchases.isPremium {
                        showImportSheet = true
                    } else {
                        onShowPremium?()
                    }
                } label: {
                    HStack(spacing: 4) {
                        if !purchases.isPremium {
                            Image(systemName: "crown.fill")
                                .font(.system(size: 11))
                                .foregroundStyle(Color(hex: "FFD700"))
                        }
                        Image(systemName: "plus")
                            .font(.system(size: 14, weight: .semibold))
                    }
                }
                .accessibilityLabel(settings.localized("import_external_model"))
            }
        }
        .sheet(isPresented: $showImportSheet) {
            ImportExternalModelSheet(vm: vm)
                .environmentObject(settings)
        }
        .onAppear {
            Task {
                try? await RunAnywhere.completeServicesInitialization()
                vm.refreshStatuses()
                vm.resumePendingDownloads()
            }
        }
        .onChange(of: scenePhase) { _, newPhase in
            if newPhase == .active {
                vm.resumeAutoResumableDownloads()
                vm.resumePendingDownloads()
            }
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
            .background(
                isSelected
                    ? AnyShapeStyle(
                        LinearGradient(
                            colors: [ApolloPalette.accentSoft, ApolloPalette.accentMuted],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        )
                    )
                    : AnyShapeStyle(.ultraThinMaterial)
            )
            .foregroundColor(isSelected ? .white : .white.opacity(0.78))
            .clipShape(RoundedRectangle(cornerRadius: 10))
            .overlay(
                RoundedRectangle(cornerRadius: 10)
                    .stroke(Color.white.opacity(isSelected ? 0.24 : 0.12), lineWidth: 1)
            )
        }
        .buttonStyle(.plain)
    }
}

// MARK: - Import External Model Sheet

struct ImportExternalModelSheet: View {
    @EnvironmentObject var settings: AppSettings
    @Environment(\.dismiss) private var dismiss
    @ObservedObject var vm: ModelDownloadViewModel

    @State private var modelName = ""
    @State private var selectedFileName = ""
    @State private var selectedFileURL: URL? = nil
    @State private var supportsVision = false
    @State private var projectorFileName = ""
    @State private var projectorFileURL: URL? = nil
    @State private var contextWindowSize = "4096"
    @State private var showFilePicker = false
    @State private var showProjectorPicker = false
    @State private var showError = false
    @State private var errorMessage = ""
    @State private var isImporting = false

    var body: some View {
        NavigationStack {
            ZStack {
                ApolloLiquidBackground()

                ScrollView {
                    VStack(spacing: 14) {

                        // Model name
                        importField(label: settings.localized("model_name")) {
                            TextField(settings.localized("model_name"), text: $modelName)
                                .foregroundColor(.white)
                        }

                        // GGUF file picker
                        importField(label: "GGUF File") {
                            Button { showFilePicker = true } label: {
                                HStack(spacing: 10) {
                                    Image(systemName: "doc.badge.plus")
                                        .font(.system(size: 15, weight: .medium))
                                        .foregroundColor(.white.opacity(0.7))
                                    Text(selectedFileName.isEmpty ? settings.localized("select_model_file") : selectedFileName)
                                        .lineLimit(1)
                                        .truncationMode(.middle)
                                        .foregroundColor(selectedFileName.isEmpty ? .white.opacity(0.5) : .white)
                                    Spacer()
                                    if !selectedFileName.isEmpty {
                                        Image(systemName: "checkmark")
                                            .font(.system(size: 12, weight: .semibold))
                                            .foregroundColor(.white.opacity(0.7))
                                    }
                                }
                            }
                            .buttonStyle(.plain)
                        }
                        .fileImporter(isPresented: $showFilePicker, allowedContentTypes: [UTType.data], allowsMultipleSelection: false) { result in
                            handleFileSelected(result: result)
                        }

                        // Context window size
                        importField(label: settings.localized("context_window_size")) {
                            TextField("4096", text: $contextWindowSize)
                                .keyboardType(.numberPad)
                                .foregroundColor(.white)
                        }

                        // Vision toggle
                        glassRow {
                            Text(settings.localized("supports_vision"))
                                .font(.subheadline)
                                .foregroundColor(.white)
                            Spacer()
                            Toggle("", isOn: $supportsVision)
                                .labelsHidden()
                                .tint(ApolloPalette.accentStrong)
                        }

                        // Vision projector picker
                        if supportsVision {
                            importField(label: "Vision Projector") {
                                Button { showProjectorPicker = true } label: {
                                    HStack(spacing: 10) {
                                        Image(systemName: "camera.badge.plus")
                                            .font(.system(size: 15, weight: .medium))
                                            .foregroundColor(.white.opacity(0.7))
                                        Text(projectorFileName.isEmpty ? (settings.localized("select") + " Vision Projector") : projectorFileName)
                                            .lineLimit(1)
                                            .truncationMode(.middle)
                                            .foregroundColor(projectorFileName.isEmpty ? .white.opacity(0.5) : .white)
                                        Spacer()
                                        if !projectorFileName.isEmpty {
                                            Image(systemName: "checkmark")
                                                .font(.system(size: 12, weight: .semibold))
                                                .foregroundColor(.white.opacity(0.7))
                                        }
                                    }
                                }
                                .buttonStyle(.plain)
                            }
                            .fileImporter(isPresented: $showProjectorPicker, allowedContentTypes: [UTType.data], allowsMultipleSelection: false) { result in
                                handleProjectorSelected(result: result)
                            }
                            .transition(.opacity.combined(with: .move(edge: .top)))
                        }

                        // Error message
                        if showError {
                            Text(errorMessage)
                                .font(.caption)
                                .foregroundColor(.white.opacity(0.8))
                                .padding(12)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .background(.ultraThinMaterial)
                                .clipShape(RoundedRectangle(cornerRadius: 10))
                                .overlay(RoundedRectangle(cornerRadius: 10).stroke(Color.white.opacity(0.12), lineWidth: 1))
                        }

                        Spacer(minLength: 20)
                    }
                    .padding(20)
                    .animation(.spring(response: 0.3), value: supportsVision)
                }
            }
            .navigationTitle(settings.localized("import_external_model"))
            .navigationBarTitleDisplayMode(.inline)
            .toolbarBackground(.hidden, for: .navigationBar)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button(settings.localized("cancel")) { dismiss() }
                        .foregroundColor(.white)
                }
                ToolbarItem(placement: .navigationBarTrailing) {
                    if isImporting {
                        ProgressView().tint(.white)
                    } else {
                        Button(settings.localized("import_model")) { performImport() }
                            .bold()
                            .foregroundColor(canImport ? .white : .white.opacity(0.35))
                            .disabled(!canImport)
                    }
                }
            }
        }
    }

    @ViewBuilder
    private func importField<Content: View>(label: String, @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(label)
                .font(.caption.bold())
                .foregroundColor(.white.opacity(0.55))
            glassRow { content() }
        }
    }

    @ViewBuilder
    private func glassRow<Content: View>(@ViewBuilder content: () -> Content) -> some View {
        HStack { content() }
            .padding(12)
            .background(.ultraThinMaterial)
            .clipShape(RoundedRectangle(cornerRadius: 10))
            .overlay(RoundedRectangle(cornerRadius: 10).stroke(Color.white.opacity(0.1), lineWidth: 1))
    }

    private var canImport: Bool {
        !modelName.trimmingCharacters(in: .whitespaces).isEmpty
            && selectedFileURL != nil
            && (!supportsVision || projectorFileURL != nil)
    }

    private func handleFileSelected(result: Result<[URL], Error>) {
        switch result {
        case .success(let urls):
            guard let url = urls.first else { return }
            let ext = url.pathExtension.lowercased()
            guard ext == "gguf" else {
                showError = true
                errorMessage = settings.localized("unsupported_file_format")
                return
            }
            if url.lastPathComponent.lowercased().contains("mmproj") {
                showError = true
                errorMessage = "Select the main GGUF model file here, not the mmproj vision projector."
                selectedFileURL = nil
                selectedFileName = ""
                return
            }
            selectedFileURL = url
            let name = url.lastPathComponent
            selectedFileName = name.count > 40 ? String(name.prefix(37)) + "..." : name
            showError = false
        case .failure:
            break
        }
    }

    private func handleProjectorSelected(result: Result<[URL], Error>) {
        switch result {
        case .success(let urls):
            guard let url = urls.first else { return }
            guard url.pathExtension.lowercased() == "gguf" else {
                showError = true
                errorMessage = settings.localized("unsupported_file_format")
                return
            }
            guard url.lastPathComponent.lowercased().contains("mmproj") else {
                showError = true
                errorMessage = "Select the mmproj GGUF file as the vision projector."
                projectorFileURL = nil
                projectorFileName = ""
                return
            }
            projectorFileURL = url
            let name = url.lastPathComponent
            projectorFileName = name.count > 40 ? String(name.prefix(37)) + "..." : name
            showError = false
        case .failure:
            break
        }
    }

    private func performImport() {
        let name = modelName.trimmingCharacters(in: .whitespaces)
        guard !name.isEmpty, let sourceURL = selectedFileURL else { return }

        // Duplicate name check
        if vm.models.contains(where: { $0.name == name }) {
            showError = true
            errorMessage = String(format: settings.localized("model_name_already_exists"), name)
            return
        }

        let contextSize = Int(contextWindowSize) ?? 4096

        isImporting = true

        Task {
            let modelId = name.lowercased()
                .replacingOccurrences(of: " ", with: "_")
                .filter { $0.isLetter || $0.isNumber || $0 == "_" }
                + "_custom"

            // Custom GGUFs must live in the SDK-managed folder so RunAnywhere.loadModel(id)
            // resolves the same file the model registry points to.
            let importDir = ModelDownloadViewModel.customModelDirectory(for: modelId)
            try? FileManager.default.createDirectory(at: importDir, withIntermediateDirectories: true)

            // Copy the GGUF file — security-scoped access required
            let accessing = sourceURL.startAccessingSecurityScopedResource()
            defer { if accessing { sourceURL.stopAccessingSecurityScopedResource() } }

            let destFile = importDir.appendingPathComponent(sourceURL.lastPathComponent)
            try? FileManager.default.removeItem(at: destFile)
            do {
                try FileManager.default.copyItem(at: sourceURL, to: destFile)
            } catch {
                await MainActor.run {
                    isImporting = false
                    showError = true
                    errorMessage = "Failed to copy model file: \(error.localizedDescription)"
                }
                return
            }

            // Get file size
            let fileSize = (try? FileManager.default.attributesOfItem(atPath: destFile.path)[.size] as? Int64) ?? 0

            let model = AIModel(
                id: modelId,
                name: name,
                description: "Imported GGUF model",
                url: destFile.path,
                category: supportsVision ? .multimodal : .text,
                sizeBytes: fileSize,
                source: "Custom",
                supportsVision: supportsVision,
                supportsAudio: false,
                supportsThinking: false,
                supportsGpu: true,
                requirements: ModelRequirements(minRamGB: max(2, Int(fileSize / 1_073_741_824) + 1), recommendedRamGB: max(4, Int(fileSize / 1_073_741_824) + 2)),
                contextWindowSize: contextSize,
                modelFormat: .gguf,
                additionalFiles: []
            )

            await MainActor.run {
                let success = vm.addExternalModel(model)
                if success {
                    // Import vision projector if selected
                    if supportsVision, let projURL = projectorFileURL {
                        let pAccessing = projURL.startAccessingSecurityScopedResource()
                        defer { if pAccessing { projURL.stopAccessingSecurityScopedResource() } }
                        vm.importVisionProjector(for: modelId, fileName: projURL.lastPathComponent, from: projURL)
                    }
                    isImporting = false
                    dismiss()
                } else {
                    isImporting = false
                    showError = true
                    errorMessage = String(format: settings.localized("model_name_already_exists"), name)
                }
            }
        }
    }
}
