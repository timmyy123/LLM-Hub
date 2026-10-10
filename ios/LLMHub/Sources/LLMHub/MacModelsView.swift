//
//  MacModelsView.swift
//  LLMHub
//
//  Native macOS model catalog. Drives the same `ModelDownloadViewModel`,
//  import clients, persisted state and localized strings as
//  `ModelDownloadScreen` on iOS.
//

#if os(macOS)
import SwiftUI
import UniformTypeIdentifiers

struct MacModelsView: View {
    @EnvironmentObject var settings: AppSettings
    @Environment(\.scenePhase) private var scenePhase
    @StateObject private var vm = ModelDownloadViewModel.shared
    @StateObject private var purchases = PurchaseManager.shared
    @State private var showImportSheet = false
    @State private var editingTemplateModel: AIModel? = nil
    var onShowPremium: (() -> Void)? = nil

    init(onShowPremium: (() -> Void)? = nil) {
        self.onShowPremium = onShowPremium
    }

    private func familyExpansion(_ title: String) -> Binding<Bool> {
        Binding(
            get: { vm.expandedFamilyTitle == title },
            set: { expanded in
                if expanded != (vm.expandedFamilyTitle == title) {
                    withAnimation(.spring(response: 0.28, dampingFraction: 0.85)) {
                        vm.toggleFamily(title)
                    }
                }
            }
        )
    }

    var body: some View {
        Group {
            if vm.filteredModels.isEmpty {
                ContentUnavailableView {
                    Label(settings.localized("no_models_available"), systemImage: "magnifyingglass")
                }
            } else {
                List {
                    ForEach(vm.groupedFilteredModels) { family in
                        DisclosureGroup(isExpanded: familyExpansion(family.title)) {
                            ForEach(family.models) { model in
                                MacModelRow(
                                    model: model,
                                    quantizationLabel: vm.quantizationLabel(for: model),
                                    state: vm.downloadStates[model.id] ?? .notDownloaded,
                                    isExpanded: vm.expandedModelId == model.id,
                                    onDownload: { vm.startDownload(model) },
                                    onPause: { vm.pauseDownload(model.id) },
                                    onResume: { vm.resumeDownload(model.id) },
                                    onDelete: { vm.deleteModel(model.id) },
                                    onExpand: { vm.toggleExpand(model.id) },
                                    onEditPromptTemplate: { editingTemplateModel = model }
                                )
                            }
                        } label: {
                            HStack(spacing: 8) {
                                Text(family.title)
                                    .font(.headline)
                                Text("(\(family.models.count))")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                                Spacer()
                            }
                            .contentShape(Rectangle())
                            .onTapGesture {
                                withAnimation(.spring(response: 0.28, dampingFraction: 0.85)) {
                                    vm.toggleFamily(family.title)
                                }
                            }
                        }
                        .padding(.vertical, 2)
                    }
                }
                .listStyle(.inset)
            }
        }
        .navigationTitle(settings.localized("ai_models"))
        .toolbar {
            ToolbarItem(placement: .principal) {
                Picker(settings.localized("ai_models"), selection: Binding(
                    get: { vm.selectedCategory },
                    set: { cat in
                        withAnimation(.spring(response: 0.3)) { vm.selectedCategory = cat }
                    }
                )) {
                    ForEach(ModelCategory.allCases, id: \.self) { cat in
                        Label(
                            "\(settings.localized(cat.titleKey))  \(vm.models.filter { $0.category == cat }.count)",
                            systemImage: cat.icon
                        )
                        .tag(cat)
                    }
                }
                .pickerStyle(.menu)
                .labelStyle(.titleAndIcon)
                .fixedSize()
            }
            ToolbarItem(placement: .primaryAction) {
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
                                .foregroundStyle(Color(hex: "FFD700"))
                        }
                        Image(systemName: "plus")
                    }
                }
                .help(settings.localized("import_external_model"))
                .accessibilityLabel(settings.localized("import_external_model"))
            }
        }
        .sheet(isPresented: $showImportSheet) {
            MacImportExternalModelSheet(vm: vm)
                .environmentObject(settings)
                .frame(minWidth: 560, idealWidth: 640, minHeight: 560, idealHeight: 720)
        }
        .sheet(item: $editingTemplateModel) { model in
            MacEditPromptTemplateSheet(model: model, vm: vm)
                .environmentObject(settings)
                .frame(minWidth: 520, idealWidth: 600, minHeight: 440, idealHeight: 540)
        }
        .onAppear {
            Task {
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

// MARK: - Model Row

private struct MacModelRow: View {
    @EnvironmentObject var settings: AppSettings
    let model: AIModel
    let quantizationLabel: String
    let state: DownloadState
    let isExpanded: Bool
    let onDownload: () -> Void
    let onPause: () -> Void
    let onResume: () -> Void
    let onDelete: () -> Void
    let onExpand: () -> Void
    let onEditPromptTemplate: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Button(action: onExpand) {
                HStack(alignment: .top, spacing: 10) {
                    VStack(alignment: .leading, spacing: 4) {
                        Text(quantizationLabel)
                            .font(.caption.bold())
                            .foregroundStyle(.secondary)
                        Text(model.name)
                            .font(.body.weight(.semibold))
                            .lineLimit(2)

                        HStack(spacing: 6) {
                            StatusBadge(state: state)
                            Text("•").foregroundStyle(.tertiary)
                            Text(model.sizeLabel)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            Text("•").foregroundStyle(.tertiary)
                            Text(model.requirements.minRamGB > 0
                                ? String(format: settings.localized("ram_requirement_format"), model.requirements.minRamGB)
                                : settings.localized("drawthings_ram_unknown"))
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }

                        HStack(spacing: 4) {
                            if model.supportsVision {
                                capabilityBadge(settings.localized("vision"), color: ApolloPalette.accentStrong)
                            }
                            if model.supportsAudio {
                                capabilityBadge(settings.localized("audio"), color: ApolloPalette.warning)
                            }
                            if !model.supportsVision && !model.supportsAudio {
                                if model.category != .imageGeneration && model.category != .videoGeneration && model.category != .imageUpscale {
                                    capabilityBadge(settings.localized("text_only"), color: ApolloPalette.accent)
                                }
                            }
                        }
                    }
                    Spacer()
                    Image(systemName: isExpanded ? "chevron.up" : "chevron.down")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)

            if case .downloading(let progress, let downloaded, let speed) = state {
                VStack(spacing: 4) {
                    ProgressView(value: progress)
                        .tint(ApolloPalette.accentStrong)
                    HStack {
                        Text("\(settings.localized("downloading")) \(downloaded) / \(model.sizeLabel) (\(speed))")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        Spacer()
                        Text("\(Int(progress * 100))%")
                            .font(.caption.bold())
                            .monospacedDigit()
                            .foregroundStyle(ApolloPalette.accentStrong)
                    }
                }
            }

            if isExpanded {
                Divider()

                Text(model.url)
                    .font(.caption)
                    .foregroundStyle(ApolloPalette.accentStrong)
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .textSelection(.enabled)

                if model.source == "Custom" {
                    VStack(alignment: .leading, spacing: 6) {
                        HStack {
                            Label(settings.localized("prompt_template"), systemImage: "text.quote")
                                .font(.caption.bold())
                            Spacer()
                            Button(action: onEditPromptTemplate) {
                                Label(settings.localized("edit"), systemImage: "pencil")
                            }
                            .buttonStyle(.bordered)
                            .controlSize(.small)
                        }

                        if let template = model.promptTemplate, !template.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                            Text(template)
                                .font(.system(.caption2, design: .monospaced))
                                .foregroundStyle(.secondary)
                                .lineLimit(3)
                                .padding(8)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .background(Color(nsColor: .textBackgroundColor).opacity(0.35), in: RoundedRectangle(cornerRadius: 6))
                        } else {
                            Text(settings.localized("prompt_template_none"))
                                .font(.caption2)
                                .italic()
                                .foregroundStyle(.tertiary)
                        }
                    }
                }

                Text(settings.localized("model_download_interrupted_warning"))
                    .font(.caption)
                    .foregroundStyle(Color.orange.opacity(0.95))
                    .fixedSize(horizontal: false, vertical: true)

                HStack(spacing: 8) {
                    switch state {
                    case .notDownloaded:
                        Button(action: onDownload) {
                            Label(settings.localized("download"), systemImage: "arrow.down.circle.fill")
                        }
                        .buttonStyle(.borderedProminent)
                    case .error:
                        Button(action: onDownload) {
                            Label(settings.localized("retry"), systemImage: "arrow.clockwise")
                        }
                        .buttonStyle(.borderedProminent)
                        .tint(.red)
                    case .downloading:
                        Button(action: onPause) {
                            Label(settings.localized("pause_download"), systemImage: "pause.circle.fill")
                        }
                        .buttonStyle(.borderedProminent)
                        .tint(ApolloPalette.warning)
                        Button(role: .destructive, action: onDelete) {
                            Image(systemName: "xmark")
                        }
                        .buttonStyle(.bordered)
                    case .paused:
                        Button(action: onResume) {
                            Label(settings.localized("resume_download"), systemImage: "play.circle.fill")
                        }
                        .buttonStyle(.borderedProminent)
                        Button(role: .destructive, action: onDelete) {
                            Image(systemName: "trash")
                        }
                        .buttonStyle(.bordered)
                    case .downloaded:
                        Button(role: .destructive, action: onDelete) {
                            Label(settings.localized("action_delete"), systemImage: "trash")
                        }
                        .buttonStyle(.bordered)
                    }
                    Spacer()
                }
                .controlSize(.regular)
            }
        }
        .padding(.vertical, 6)
    }

    private func capabilityBadge(_ text: String, color: Color) -> some View {
        Text(text)
            .font(.caption2)
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(color.opacity(0.15), in: Capsule())
            .foregroundStyle(color)
    }
}

// MARK: - Edit Prompt Template Sheet

struct MacEditPromptTemplateSheet: View {
    @EnvironmentObject var settings: AppSettings
    @Environment(\.dismiss) private var dismiss
    let model: AIModel
    @ObservedObject var vm: ModelDownloadViewModel

    @State private var promptTemplate: String = ""

    init(model: AIModel, vm: ModelDownloadViewModel) {
        self.model = model
        self.vm = vm
        _promptTemplate = State(initialValue: model.promptTemplate ?? "")
    }

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    LabeledContent(model.name) {
                        Text(model.modelFormat.rawValue.uppercased())
                            .font(.caption2.bold())
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(ApolloPalette.accentStrong.opacity(0.2), in: Capsule())
                            .foregroundStyle(ApolloPalette.accentStrong)
                    }
                    Text(model.url)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                }

                Section {
                    TextField(
                        "",
                        text: $promptTemplate,
                        prompt: Text(settings.localized("prompt_template_placeholder")),
                        axis: .vertical
                    )
                    .labelsHidden()
                    .lineLimit(5...15)
                    .font(.system(.body, design: .monospaced))
                } header: {
                    HStack {
                        Text(settings.localized("prompt_template"))
                        Spacer()
                        if !promptTemplate.isEmpty {
                            Button(settings.localized("clear"), role: .destructive) {
                                promptTemplate = ""
                            }
                            .buttonStyle(.borderless)
                            .controlSize(.small)
                        }
                    }
                } footer: {
                    Text(settings.localized("prompt_template_hint"))
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            .formStyle(.grouped)
            .navigationTitle(settings.localized("edit_prompt_template"))
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button(settings.localized("cancel")) { dismiss() }
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button(settings.localized("save")) {
                        vm.updatePromptTemplate(for: model.id, promptTemplate: promptTemplate)
                        dismiss()
                    }
                }
            }
        }
    }
}

// MARK: - Import External Model Sheet

struct MacImportExternalModelSheet: View {
    @EnvironmentObject var settings: AppSettings
    @Environment(\.dismiss) private var dismiss
    @ObservedObject var vm: ModelDownloadViewModel

    @State private var modelName = ""
    @State private var selectedFileName = ""
    @State private var selectedFileURL: URL? = nil
    @State private var supportsVision = false
    @State private var supportsGpu = true
    @State private var supportsMtp = false
    @State private var projectorFileName = ""
    @State private var projectorFileURL: URL? = nil
    @State private var contextWindowSize = "4096"
    @State private var showFilePicker = false
    @State private var showProjectorPicker = false
    @State private var showError = false
    @State private var errorMessage = ""
    @State private var isImporting = false
    @State private var promptTemplate = ""
    @State private var modelFormat: ModelFormat = .gguf
    @State private var importKind: ModelImportKind = .gguf
    @State private var drawThingsQuery = ""
    @State private var drawThingsModels: [DrawThingsCatalogEntry] = []
    @State private var selectedDrawThingsModel: DrawThingsCatalogEntry?
    @State private var isLoadingDrawThings = false
    @State private var hasLoadedDrawThingsCatalog = false
    @State private var drawThingsLoadID = UUID()
    @State private var drawThingsSizeModelID: String?
    @State private var drawThingsSize: Int64?
    @State private var isCheckingDrawThingsSize = false
    @State private var hfQuery = ""
    @State private var hfFiles: [HuggingFaceImportFile] = []
    @State private var isSearchingHuggingFace = false
    @State private var selectedHuggingFaceModel: HuggingFaceImportFile?
    @State private var selectedHuggingFaceProjector: HuggingFaceImportFile?

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    TextField(settings.localized("model_name"), text: $modelName)

                    Picker(settings.localized("model_format"), selection: $importKind) {
                        ForEach(ModelImportKind.allCases) { kind in
                            Text(settings.localized(kind.title)).tag(kind)
                        }
                    }
                    .pickerStyle(.segmented)
                    .disabled(isImporting)
                }

                if importKind.isMedia {
                    MacDrawThingsSearchSection(
                        query: $drawThingsQuery,
                        selectedModel: $selectedDrawThingsModel,
                        modelName: $modelName,
                        models: drawThingsModels,
                        kind: importKind,
                        importedIDs: Set(vm.models.map(\.id)),
                        isLoading: isLoadingDrawThings,
                        downloadSize: drawThingsSize,
                        isCheckingSize: isCheckingDrawThingsSize,
                        onRefresh: { await loadDrawThingsCatalog() }
                    )
                    .id(importKind)
                } else {
                    MacHuggingFaceSearchSection(
                        query: $hfQuery,
                        files: $hfFiles,
                        isSearching: $isSearchingHuggingFace,
                        selectedModel: $selectedHuggingFaceModel,
                        modelName: $modelName,
                        modelFormat: modelFormat,
                        onSearch: { await searchHuggingFace() }
                    )

                    Section {
                        // File import remains available alongside Hugging Face search.
                        LabeledContent(modelFormat == .gguf ? "GGUF File" : "LiteRT-LM File") {
                            HStack(spacing: 6) {
                                if !selectedFileName.isEmpty {
                                    Image(systemName: "checkmark")
                                        .foregroundStyle(.secondary)
                                }
                                Button {
                                    showFilePicker = true
                                } label: {
                                    Label(
                                        selectedFileName.isEmpty ? settings.localized("select_model_file") : selectedFileName,
                                        systemImage: "doc.badge.plus"
                                    )
                                    .lineLimit(1)
                                    .truncationMode(.middle)
                                }
                                .buttonStyle(.bordered)
                            }
                        }
                        .fileImporter(isPresented: $showFilePicker, allowedContentTypes: [UTType.data], allowsMultipleSelection: false) { result in
                            handleFileSelected(result: result)
                        }

                        // GGUF declares its own context length in the file header.
                        if modelFormat != .gguf {
                            TextField(settings.localized("context_window_size"), text: $contextWindowSize, prompt: Text("4096"))
                        }
                    }

                    Section {
                        TextField(
                            "",
                            text: $promptTemplate,
                            prompt: Text(settings.localized("prompt_template_placeholder")),
                            axis: .vertical
                        )
                        .labelsHidden()
                        .lineLimit(3...6)
                        .font(.system(.body, design: .monospaced))
                    } header: {
                        Text(settings.localized("prompt_template_optional"))
                    } footer: {
                        Text(settings.localized("prompt_template_hint"))
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }

                    visionSection
                }

                if showError {
                    Section {
                        Text(errorMessage)
                            .font(.callout)
                            .foregroundStyle(.secondary)
                            .textSelection(.enabled)
                    }
                }
            }
            .formStyle(.grouped)
            .animation(.spring(response: 0.3), value: supportsVision)
            .onChange(of: importKind) { _, kind in
                if kind == .gguf { modelFormat = .gguf }
                if kind == .liteRT { modelFormat = .litertlm }
                selectedDrawThingsModel = nil
                drawThingsSize = nil
                selectedHuggingFaceModel = nil
                hfFiles = []
                selectedFileURL = nil
                selectedFileName = ""
                modelName = ""
                showError = false
                supportsVision = false
                projectorFileURL = nil
                projectorFileName = ""
                selectedHuggingFaceProjector = nil
            }
            .task(id: importKind) {
                if importKind.isMedia && !hasLoadedDrawThingsCatalog {
                    await loadDrawThingsCatalog()
                }
            }
            .task(id: selectedDrawThingsModel?.id) {
                drawThingsSize = nil
                drawThingsSizeModelID = nil
                isCheckingDrawThingsSize = false
                guard let selected = selectedDrawThingsModel else { return }
                isCheckingDrawThingsSize = true
                let size = await DrawThingsCatalogClient.downloadSize(for: selected)
                guard !Task.isCancelled, selectedDrawThingsModel?.id == selected.id else { return }
                drawThingsSize = size
                drawThingsSizeModelID = selected.id
                isCheckingDrawThingsSize = false
            }
            .navigationTitle(settings.localized("import_external_model"))
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button(settings.localized("cancel")) { dismiss() }
                }
                ToolbarItem(placement: .confirmationAction) {
                    if isImporting {
                        ProgressView().controlSize(.small)
                    } else {
                        Button(settings.localized("import_model")) { performImport() }
                            .disabled(!canImport)
                    }
                }
            }
        }
    }

    @ViewBuilder
    private var visionSection: some View {
        Section {
            Toggle(settings.localized("supports_vision"), isOn: $supportsVision)

            if modelFormat == .litertlm {
                Toggle(settings.localized("supports_gpu"), isOn: $supportsGpu)
                Toggle(settings.localized("supports_mtp"), isOn: $supportsMtp)
            }

            // Vision projector file picker (only GGUF needs a separate mmproj file)
            if modelFormat == .gguf && supportsVision {
                LabeledContent("Vision Projector") {
                    HStack(spacing: 6) {
                        if !projectorFileName.isEmpty {
                            Image(systemName: "checkmark")
                                .foregroundStyle(.secondary)
                        }
                        Button {
                            showProjectorPicker = true
                        } label: {
                            Label(
                                projectorFileName.isEmpty ? (settings.localized("select") + " Vision Projector") : projectorFileName,
                                systemImage: "camera.badge.plus"
                            )
                            .lineLimit(1)
                            .truncationMode(.middle)
                        }
                        .buttonStyle(.bordered)
                    }
                }
                .fileImporter(isPresented: $showProjectorPicker, allowedContentTypes: [UTType.data], allowsMultipleSelection: false) { result in
                    handleProjectorSelected(result: result)
                }
            }

            // Show HF projectors when a GGUF model is selected (auto-enables vision on pick)
            if modelFormat == .gguf, let selectedRepo = selectedHuggingFaceModel?.repo {
                let projectorFiles = hfFiles.filter { $0.isProjector && $0.repo == selectedRepo }
                ForEach(projectorFiles) { file in
                    Button {
                        selectedHuggingFaceProjector = file
                        projectorFileURL = nil
                        projectorFileName = file.path
                        supportsVision = true
                    } label: {
                        HStack {
                            Text(file.path).lineLimit(1)
                            Spacer()
                            if selectedHuggingFaceProjector?.id == file.id {
                                Image(systemName: "checkmark")
                            }
                        }
                        .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                    .font(.caption)
                    .foregroundStyle(selectedHuggingFaceProjector?.id == file.id ? ApolloPalette.accentStrong : .primary)
                }
            }
        }
    }

    private var canImport: Bool {
        if importKind.isMedia {
            return selectedDrawThingsModel != nil
                && drawThingsSizeModelID == selectedDrawThingsModel?.id
                && !isCheckingDrawThingsSize
                && !modelName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        }
        return !modelName.trimmingCharacters(in: .whitespaces).isEmpty
            && (selectedFileURL != nil || selectedHuggingFaceModel != nil)
            && (!supportsVision || projectorFileURL != nil || selectedHuggingFaceProjector != nil)
    }

    private func handleFileSelected(result: Result<[URL], Error>) {
        switch result {
        case .success(let urls):
            guard let url = urls.first else { return }
            let ext = url.pathExtension.lowercased()
            guard ext == modelFormat.rawValue else {
                showError = true
                errorMessage = settings.localized("unsupported_file_format")
                return
            }
            if modelFormat == .gguf && url.lastPathComponent.lowercased().contains("mmproj") {
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
        if importKind.isMedia {
            importDrawThingsModel()
            return
        }
        let name = modelName.trimmingCharacters(in: .whitespaces)
        guard !name.isEmpty, selectedFileURL != nil || selectedHuggingFaceModel != nil else { return }

        if vm.models.contains(where: { $0.name == name }) {
            showError = true
            errorMessage = String(format: settings.localized("model_name_already_exists"), name)
            return
        }

        // Remote GGUF headers become available after download; the runtime reads them then.
        let contextSize = modelFormat == .gguf ? 4096 : (Int(contextWindowSize) ?? 4096)
        let templateValue = promptTemplate.trimmingCharacters(in: .whitespacesAndNewlines)

        let modelId = name.lowercased()
            .replacingOccurrences(of: " ", with: "_")
            .filter { $0.isLetter || $0.isNumber || $0 == "_" }
            + "_custom"

        if let remoteModel = selectedHuggingFaceModel {
            var additionalURLs: [String] = []
            if supportsVision, let remoteProjector = selectedHuggingFaceProjector {
                additionalURLs.append(remoteProjector.downloadURL.absoluteString)
            }

            let model = AIModel(
                id: modelId,
                name: name,
                description: "Imported \(modelFormat == .gguf ? "GGUF" : "LiteRT-LM") model",
                url: remoteModel.downloadURL.absoluteString,
                category: supportsVision ? .multimodal : .text,
                sizeBytes: remoteModel.size + (selectedHuggingFaceProjector?.size ?? 0),
                source: "Custom",
                supportsVision: supportsVision,
                supportsAudio: false,
                supportsThinking: false,
                supportsGpu: supportsGpu,
                supportsMtp: modelFormat == .litertlm ? supportsMtp : true,
                requirements: ModelRequirements(minRamGB: max(2, Int(remoteModel.size / 1_073_741_824) + 1), recommendedRamGB: max(4, Int(remoteModel.size / 1_073_741_824) + 2)),
                contextWindowSize: contextSize,
                modelFormat: modelFormat,
                additionalFiles: additionalURLs,
                promptTemplate: templateValue.isEmpty ? nil : templateValue
            )

            let success = vm.addExternalModel(model)
            if success {
                vm.downloadStates[model.id] = .notDownloaded
                vm.startDownload(model)
                dismiss()
            } else {
                showError = true
                errorMessage = String(format: settings.localized("model_name_already_exists"), name)
            }
            return
        }

        // Local file import
        isImporting = true
        Task {
            let importDir = ModelDownloadViewModel.customModelDirectory(for: modelId)
            try? FileManager.default.createDirectory(at: importDir, withIntermediateDirectories: true)

            let sourceURL = selectedFileURL!
            let destFile = importDir.appendingPathComponent(sourceURL.lastPathComponent)
            try? FileManager.default.removeItem(at: destFile)
            do {
                let accessing = sourceURL.startAccessingSecurityScopedResource()
                defer { if accessing { sourceURL.stopAccessingSecurityScopedResource() } }
                try FileManager.default.copyItem(at: sourceURL, to: destFile)
            } catch {
                await MainActor.run {
                    isImporting = false
                    showError = true
                    errorMessage = "Failed to copy model file: \(error.localizedDescription)"
                }
                return
            }

            let fileSize = (try? FileManager.default.attributesOfItem(atPath: destFile.path)[.size] as? Int64) ?? 0

            let model = AIModel(
                id: modelId,
                name: name,
                description: "Imported \(modelFormat == .gguf ? "GGUF" : "LiteRT-LM") model",
                url: destFile.path,
                category: supportsVision ? .multimodal : .text,
                sizeBytes: fileSize,
                source: "Custom",
                supportsVision: supportsVision,
                supportsAudio: false,
                supportsThinking: false,
                supportsGpu: supportsGpu,
                supportsMtp: modelFormat == .litertlm ? supportsMtp : true,
                requirements: ModelRequirements(minRamGB: max(2, Int(fileSize / 1_073_741_824) + 1), recommendedRamGB: max(4, Int(fileSize / 1_073_741_824) + 2)),
                contextWindowSize: modelFormat == .gguf
                    ? (GGUFLayerLimits.readContextLength(from: destFile) ?? contextSize)
                    : contextSize,
                modelFormat: modelFormat,
                additionalFiles: [],
                promptTemplate: templateValue.isEmpty ? nil : templateValue
            )

            await MainActor.run {
                let success = vm.addExternalModel(model)
                if success {
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

    private func loadDrawThingsCatalog() async {
        let loadID = UUID()
        drawThingsLoadID = loadID
        isLoadingDrawThings = true
        defer {
            if drawThingsLoadID == loadID { isLoadingDrawThings = false }
        }
        if drawThingsModels.isEmpty { drawThingsModels = DrawThingsCatalogClient.builtinModels() }
        do {
            let models = try await DrawThingsCatalogClient.load()
            guard !Task.isCancelled, drawThingsLoadID == loadID else { return }
            drawThingsModels = models
            hasLoadedDrawThingsCatalog = true
            showError = false
        } catch is CancellationError {
        } catch {
            guard !Task.isCancelled, drawThingsLoadID == loadID else { return }
            showError = true
            errorMessage = settings.localized("drawthings_catalog_load_error") + " " + error.localizedDescription
        }
    }

    private func importDrawThingsModel() {
        guard let entry = selectedDrawThingsModel, canImport else { return }
        let name = modelName.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !vm.models.contains(where: { $0.id == entry.id || $0.name == name }) else {
            showError = true
            errorMessage = String(format: settings.localized("model_name_already_exists"), name)
            return
        }
        do {
            try DrawThingsCatalogClient.register(entry, catalog: drawThingsModels)
            let size = drawThingsSize ?? 0
            let model = AIModel(
                id: entry.id, name: name,
                description: entry.specification.note ?? entry.name,
                url: "https://static.libnnc.org/" + entry.id,
                category: entry.isVideo ? .videoGeneration : .imageGeneration,
                sizeBytes: size, source: "Draw Things",
                supportsVision: false, supportsAudio: false, supportsThinking: false,
                supportsGpu: true,
                requirements: ModelRequirements(minRamGB: 0, recommendedRamGB: 0),
                contextWindowSize: 0, modelFormat: .drawthings
            )
            if vm.addDrawThingsModel(model) { dismiss() }
        } catch {
            showError = true
            errorMessage = error.localizedDescription
        }
    }

    private func searchHuggingFace() async {
        isSearchingHuggingFace = true
        defer { isSearchingHuggingFace = false }
        let kind = importKind
        do {
            let files = try await HuggingFaceImportClient.search(query: hfQuery, format: modelFormat, token: huggingFaceToken)
            guard importKind == kind else { return }
            hfFiles = files
        } catch {
            guard importKind == kind else { return }
            showError = true
            errorMessage = error.localizedDescription
        }
    }

    private var huggingFaceToken: String? {
        settings.effectiveHfToken
    }
}

// MARK: - Hugging Face Search

private struct MacHuggingFaceSearchSection: View {
    @EnvironmentObject private var settings: AppSettings
    @Binding var query: String
    @Binding var files: [HuggingFaceImportFile]
    @Binding var isSearching: Bool
    @Binding var selectedModel: HuggingFaceImportFile?
    @Binding var modelName: String
    let modelFormat: ModelFormat
    let onSearch: () async -> Void

    @State private var currentPage = 0
    private let pageSize = 10

    private var nonProjectorFiles: [HuggingFaceImportFile] { files.filter { !$0.isProjector } }
    private var totalPages: Int { max(1, Int(ceil(Double(nonProjectorFiles.count) / Double(pageSize)))) }
    private var pagedFiles: [HuggingFaceImportFile] {
        let start = currentPage * pageSize
        let end = min(start + pageSize, nonProjectorFiles.count)
        guard start < nonProjectorFiles.count else { return [] }
        return Array(nonProjectorFiles[start..<end])
    }

    private var canSearch: Bool {
        !query.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && !isSearching
    }

    private func runSearch() {
        guard canSearch else { return }
        Task {
            currentPage = 0
            await onSearch()
        }
    }

    var body: some View {
        Section {
            HStack {
                TextField("", text: $query, prompt: Text(settings.localized("search_huggingface_placeholder")))
                    .labelsHidden()
                    .textFieldStyle(.roundedBorder)
                    .onSubmit { runSearch() }
                Button {
                    runSearch()
                } label: {
                    if isSearching {
                        ProgressView().controlSize(.small)
                    } else {
                        Image(systemName: "magnifyingglass")
                    }
                }
                .buttonStyle(.bordered)
                .disabled(!canSearch)
            }

            if !nonProjectorFiles.isEmpty {
                Text(settings.localized("huggingface_results"))
                    .font(.caption)
                    .foregroundStyle(.secondary)
                ForEach(pagedFiles) { file in
                    Button {
                        selectedModel = file
                        modelName = file.path.replacingOccurrences(of: ".\(modelFormat.rawValue)", with: "")
                    } label: {
                        HStack {
                            Text("\(file.repo)/\(file.path)").lineLimit(1).truncationMode(.middle)
                            Spacer()
                            Text(file.sizeLabel).font(.caption).monospacedDigit()
                            if selectedModel?.id == file.id {
                                Image(systemName: "checkmark")
                            }
                        }
                        .contentShape(Rectangle())
                        .foregroundStyle(selectedModel?.id == file.id ? ApolloPalette.accentStrong : .primary)
                    }
                    .buttonStyle(.plain)
                }
                if totalPages > 1 {
                    HStack {
                        Button { currentPage -= 1 } label: { Image(systemName: "chevron.left") }
                            .disabled(currentPage <= 0)
                        Spacer()
                        Text("\(currentPage + 1) / \(totalPages)")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        Spacer()
                        Button { currentPage += 1 } label: { Image(systemName: "chevron.right") }
                            .disabled(currentPage >= totalPages - 1)
                    }
                    .buttonStyle(.borderless)
                }
            }
        } header: {
            Text(settings.localized("search_huggingface"))
        }
    }
}

// MARK: - Draw Things Search

private struct MacDrawThingsSearchSection: View {
    @EnvironmentObject private var settings: AppSettings
    @Binding var query: String
    @Binding var selectedModel: DrawThingsCatalogEntry?
    @Binding var modelName: String
    let models: [DrawThingsCatalogEntry]
    let kind: ModelImportKind
    let importedIDs: Set<String>
    let isLoading: Bool
    let downloadSize: Int64?
    let isCheckingSize: Bool
    let onRefresh: () async -> Void
    @State private var currentPage = 0
    @State private var resultSizes: [String: Int64] = [:]
    @State private var unavailableSizes: Set<String> = []
    private let pageSize = 10

    private var results: [DrawThingsCatalogEntry] {
        models.filter { $0.isVideo == (kind == .video) && $0.matches(query) }
    }
    private var totalPages: Int { max(1, (results.count + pageSize - 1) / pageSize) }
    private var page: [DrawThingsCatalogEntry] {
        Array(results.dropFirst(currentPage * pageSize).prefix(pageSize))
    }

    var body: some View {
        Section {
            HStack {
                TextField("", text: $query, prompt: Text(settings.localized("search_drawthings_placeholder")))
                    .labelsHidden()
                    .textFieldStyle(.roundedBorder)
                    .autocorrectionDisabled()
                Button {
                    Task { await onRefresh() }
                } label: {
                    if isLoading {
                        ProgressView().controlSize(.small)
                    } else {
                        Image(systemName: "arrow.clockwise")
                    }
                }
                .buttonStyle(.bordered)
                .help(settings.localized("refresh_drawthings_catalog"))
                .accessibilityLabel(settings.localized("refresh_drawthings_catalog"))
                .disabled(isLoading)
            }

            if results.isEmpty && !isLoading {
                Text(settings.localized("drawthings_no_results"))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else {
                Text(String(format: settings.localized("drawthings_result_count"), results.count))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            ForEach(page) { entry in
                Button {
                    selectedModel = entry
                    modelName = entry.name
                } label: {
                    HStack(spacing: 10) {
                        VStack(alignment: .leading, spacing: 2) {
                            Text(entry.name).multilineTextAlignment(.leading)
                            Text(entry.id)
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                                .lineLimit(1)
                                .truncationMode(.middle)
                        }
                        Spacer(minLength: 0)
                        if let size = resultSizes[entry.id] {
                            Text(ByteCountFormatter.string(fromByteCount: size, countStyle: .file))
                                .font(.caption)
                                .monospacedDigit()
                                .fixedSize(horizontal: true, vertical: false)
                        } else if unavailableSizes.contains(entry.id) {
                            Text(settings.localized("drawthings_size_unknown"))
                                .font(.caption2)
                                .multilineTextAlignment(.trailing)
                        } else {
                            ProgressView()
                                .controlSize(.small)
                                .accessibilityLabel(settings.localized("drawthings_checking_size"))
                        }
                        if importedIDs.contains(entry.id) {
                            Text(settings.localized("drawthings_already_added")).font(.caption)
                        } else if selectedModel?.id == entry.id {
                            Image(systemName: "checkmark.circle.fill")
                        }
                    }
                    .contentShape(Rectangle())
                    .foregroundStyle(selectedModel?.id == entry.id ? ApolloPalette.accentStrong : .primary)
                }
                .buttonStyle(.plain)
                .disabled(importedIDs.contains(entry.id))
                .task(id: entry.id) {
                    guard resultSizes[entry.id] == nil else { return }
                    let size = await DrawThingsCatalogClient.downloadSize(for: entry)
                    guard !Task.isCancelled else { return }
                    if let size {
                        resultSizes[entry.id] = size
                        unavailableSizes.remove(entry.id)
                    } else {
                        unavailableSizes.insert(entry.id)
                    }
                }
            }

            if totalPages > 1 {
                HStack {
                    Button { currentPage -= 1 } label: { Image(systemName: "chevron.left") }
                        .disabled(currentPage == 0)
                    Spacer()
                    Text("\(currentPage + 1) / \(totalPages)")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Spacer()
                    Button { currentPage += 1 } label: { Image(systemName: "chevron.right") }
                        .disabled(currentPage >= totalPages - 1)
                }
                .buttonStyle(.borderless)
            }

            if let selectedModel {
                VStack(alignment: .leading, spacing: 6) {
                    Text(selectedModel.name).font(.callout.bold())
                    if isCheckingSize {
                        HStack {
                            ProgressView().controlSize(.small)
                            Text(settings.localized("drawthings_checking_size"))
                        }
                    } else if let downloadSize {
                        Text(settings.localized("drawthings_download_size") + ": " + ByteCountFormatter.string(fromByteCount: downloadSize, countStyle: .file))
                    } else {
                        Text(settings.localized("drawthings_size_unknown"))
                    }
                    Text(settings.localized("drawthings_dependencies_hint"))
                        .foregroundStyle(.secondary)
                    if let note = selectedModel.specification.note, !note.isEmpty {
                        Text(.init(note)).foregroundStyle(.secondary)
                    }
                }
                .font(.caption)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
        } header: {
            Text(settings.localized("search_drawthings"))
        }
        .onChange(of: query) { _, _ in currentPage = 0 }
        .onChange(of: models.count) { _, _ in currentPage = min(currentPage, totalPages - 1) }
    }
}
#endif
