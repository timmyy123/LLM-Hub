//
//  MacSettingsView.swift
//  LLMHub
//
//  Native macOS Settings. Same rows, persisted settings, sub-sheets, memory
//  management and localized strings as `SettingsScreen` on iOS.
//

#if os(macOS)
import SwiftUI
import AppKit

struct MacSettingsView: View {
    @EnvironmentObject var settings: AppSettings
    @Environment(\.openURL) private var openURL
    @StateObject private var purchases = PurchaseManager.shared
    @StateObject private var ragManager = RagServiceManager.shared

    var onNavigateToModels: () -> Void
    var onShowPremium: (() -> Void)? = nil

    init(onNavigateToModels: @escaping () -> Void, onShowPremium: (() -> Void)? = nil) {
        self.onNavigateToModels = onNavigateToModels
        self.onShowPremium = onShowPremium
    }

    @State private var showMemoryDialog = false
    @State private var showAbout = false
    @State private var showTerms = false
    @State private var showTTSAlert = false
    @State private var showHfTokenDialog = false

    var body: some View {
        Form {
            // MARK: Models
            Section(settings.localized("models")) {
                MacSettingsNavRow(
                    icon: "square.and.arrow.down.fill",
                    title: settings.localized("download_models"),
                    subtitle: settings.localized("browse_download_models")
                ) {
                    onNavigateToModels()
                }

                MacSettingsNavRow(
                    icon: "key.fill",
                    title: settings.localized("hf_token_title"),
                    subtitle: settings.hasCustomHfToken
                        ? settings.localized("hf_token_custom_active")
                        : settings.localized("hf_token_using_default")
                ) {
                    showHfTokenDialog = true
                }
            }

            // MARK: RAG / Embedding
            Section(settings.localized("embedding_models")) {
                MacEmbeddingModelSelectorRow(onNavigateToModels: onNavigateToModels)

                MacSettingsToggleRow(
                    icon: "brain",
                    title: settings.localized("memory"),
                    subtitle: settings.selectedEmbeddingModelId != nil
                        ? settings.localized("memory_description_enabled")
                        : settings.localized("memory_requires_rag"),
                    isOn: Binding(
                        get: { settings.memoryEnabled && settings.selectedEmbeddingModelId != nil },
                        set: { newValue in
                            guard settings.selectedEmbeddingModelId != nil else { return }
                            settings.memoryEnabled = newValue
                        }
                    )
                )
                .disabled(settings.selectedEmbeddingModelId == nil)

                if settings.memoryEnabled && settings.selectedEmbeddingModelId != nil {
                    MacSettingsNavRow(
                        icon: "tray.full",
                        title: settings.localized("manage_memory"),
                        subtitle: settings.localized("manage_memory_subtitle")
                    ) {
                        Task {
                            await RagServiceManager.shared.initialize(modelId: settings.selectedEmbeddingModelId)
                            await MainActor.run {
                                showMemoryDialog = true
                            }
                        }
                    }
                }
            }

            // MARK: Appearance
            Section(settings.localized("appearance")) {
                MacSettingsToggleRow(
                    icon: "speaker.wave.2.fill",
                    title: settings.localized("auto_readout"),
                    subtitle: settings.localized("auto_readout_description"),
                    isOn: $settings.autoReadoutEnabled
                )

                MacSettingsNavRow(
                    icon: "waveform",
                    title: settings.localized("text_to_speech_voices"),
                    subtitle: settings.localized("text_to_speech_voices_description")
                ) {
                    showTTSAlert = true
                }

                Picker(selection: $settings.selectedLanguage) {
                    ForEach(AppLanguage.allCases) { lang in
                        Text(settings.localized(lang.displayNameKey)).tag(lang)
                    }
                } label: {
                    MacSettingsRowLabel(
                        icon: "globe",
                        title: settings.localized("language"),
                        subtitle: settings.localized(settings.selectedLanguage.displayNameKey)
                    )
                }
                .pickerStyle(.menu)
            }

            // MARK: Information
            Section(settings.localized("information")) {
                MacSettingsNavRow(
                    icon: "info.circle.fill",
                    title: settings.localized("about"),
                    subtitle: settings.localized("app_information_contact")
                ) { showAbout = true }

                MacSettingsNavRow(
                    icon: "doc.text.fill",
                    title: settings.localized("terms_of_service"),
                    subtitle: settings.localized("legal_terms_conditions")
                ) { showTerms = true }
            }

            // MARK: Premium
            Section(settings.localized("premium_title")) {
                MacSettingsNavRow(
                    icon: purchases.isPremium ? "crown.fill" : "crown",
                    iconColor: Color(hex: "FFD700"),
                    title: settings.localized(purchases.isPremium ? "premium_active_title" : "premium_go_premium"),
                    subtitle: settings.localized(purchases.isPremium ? "premium_active_subtitle_short" : "premium_tap_to_unlock")
                ) {
                    onShowPremium?()
                }
            }

            // MARK: Source Code
            Section(settings.localized("source_code_section")) {
                MacSettingsNavRow(
                    icon: "chevron.left.forwardslash.chevron.right",
                    title: settings.localized("github_repository"),
                    subtitle: settings.localized("view_source_contribute"),
                    trailingSymbol: "arrow.up.right.square"
                ) {
                    if let url = URL(string: "https://github.com/timmyy123/LLM-Hub") {
                        openURL(url)
                    }
                }
            }
        }
        .formStyle(.grouped)
        .navigationTitle(settings.localized("feature_settings_title"))
        .sheet(isPresented: $showMemoryDialog) {
            MacMemoryManagerSheet(onDismiss: { showMemoryDialog = false })
                .environmentObject(settings)
                .frame(minWidth: 560, idealWidth: 640, minHeight: 600, idealHeight: 720)
        }
        .sheet(isPresented: $showAbout) {
            MacAboutSheet()
                .environmentObject(settings)
                .frame(minWidth: 520, idealWidth: 600, minHeight: 520, idealHeight: 640)
        }
        .sheet(isPresented: $showTerms) {
            MacTermsOfServiceSheet()
                .environmentObject(settings)
                .frame(minWidth: 560, idealWidth: 640, minHeight: 600, idealHeight: 760)
        }
        .sheet(isPresented: $showHfTokenDialog) {
            MacHuggingFaceTokenSheet(onDismiss: { showHfTokenDialog = false })
                .environmentObject(settings)
                .frame(minWidth: 480, idealWidth: 540, minHeight: 360, idealHeight: 400)
        }
        .onChange(of: settings.selectedEmbeddingModelId) { _, newId in
            // Initialization and re-embedding handled by the embedding selector row.
            // This handles external changes (e.g. model deleted from download screen).
            Task {
                if newId == nil {
                    await RagServiceManager.shared.initialize(modelId: nil)
                }
            }
        }
        .alert(settings.localized("text_to_speech_voices"), isPresented: $showTTSAlert) {
            Button(settings.localized("ok"), role: .cancel) {}
        } message: {
            Text(settings.localized("tts_voices_nav_hint"))
        }
    }
}

// MARK: - Row building blocks

/// Icon tile + title + subtitle, in the style of macOS System Settings.
private struct MacSettingsRowLabel: View {
    let icon: String
    var iconColor: Color = ApolloPalette.accentStrong
    let title: String
    let subtitle: String

    var body: some View {
        HStack(spacing: 10) {
            RoundedRectangle(cornerRadius: 6, style: .continuous)
                .fill(iconColor.gradient)
                .frame(width: 24, height: 24)
                .overlay {
                    Image(systemName: icon)
                        .font(.system(size: 12, weight: .semibold))
                        .foregroundStyle(.white)
                }
            VStack(alignment: .leading, spacing: 1) {
                Text(title)
                Text(subtitle)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
            }
        }
    }
}

private struct MacSettingsNavRow: View {
    let icon: String
    var iconColor: Color = ApolloPalette.accentStrong
    let title: String
    let subtitle: String
    var trailingSymbol: String = "chevron.right"
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack {
                MacSettingsRowLabel(icon: icon, iconColor: iconColor, title: title, subtitle: subtitle)
                Spacer()
                Image(systemName: trailingSymbol)
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.tertiary)
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }
}

private struct MacSettingsToggleRow: View {
    let icon: String
    let title: String
    let subtitle: String
    @Binding var isOn: Bool

    var body: some View {
        Toggle(isOn: $isOn) {
            MacSettingsRowLabel(icon: icon, title: title, subtitle: subtitle)
        }
        .toggleStyle(.switch)
    }
}

// MARK: - Embedding Model Selector

private struct MacEmbeddingModelSelectorRow: View {
    @EnvironmentObject var settings: AppSettings
    let onNavigateToModels: () -> Void

    private var downloadedEmbeddingModels: [AIModel] {
        ModelData.allModels().filter { model in
            model.category == .embedding
                && ModelData.isModelFullyAvailableLocally(model)
        }
    }

    private var selectedModel: AIModel? {
        guard let id = settings.selectedEmbeddingModelId else { return nil }
        return ModelData.allModels().first { $0.id == id }
    }

    var body: some View {
        let models = downloadedEmbeddingModels
        HStack {
            MacSettingsRowLabel(
                icon: "link.circle.fill",
                title: settings.localized("embedding_model"),
                subtitle: selectedModel?.name ?? settings.localized("no_embedding_model_selected")
            )
            Spacer()
            if models.isEmpty {
                Button {
                    onNavigateToModels()
                } label: {
                    Image(systemName: "chevron.right")
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(.tertiary)
                }
                .buttonStyle(.plain)
            } else {
                Menu {
                    Section(settings.localized("select_embedding_model")) {
                        ForEach(models) { model in
                            Button {
                                select(model)
                            } label: {
                                if settings.selectedEmbeddingModelId == model.id {
                                    Label(model.name, systemImage: "checkmark")
                                } else {
                                    Text(model.name)
                                }
                            }
                        }
                    }
                    if settings.selectedEmbeddingModelId != nil {
                        Divider()
                        Button(settings.localized("disable_embeddings"), role: .destructive) {
                            settings.selectedEmbeddingModelId = nil
                            settings.memoryEnabled = false
                        }
                    }
                } label: {
                    Text(selectedModel?.name ?? settings.localized("no_embedding_model_selected"))
                }
                .menuStyle(.button)
                .fixedSize()
            }
        }
        .contentShape(Rectangle())
        .onTapGesture {
            if models.isEmpty { onNavigateToModels() }
        }
    }

    private func select(_ model: AIModel) {
        let previousModelId = settings.selectedEmbeddingModelId
        settings.selectedEmbeddingModelId = model.id
        Task {
            if previousModelId != nil && previousModelId != model.id {
                print("[Memory] Embedding model changed from \(previousModelId ?? "nil") to \(model.id) — re-embedding global memory")
                await RagServiceManager.shared.reembedGlobalMemory(newModelId: model.id)
            } else {
                await RagServiceManager.shared.initialize(modelId: model.id)
            }
        }
    }
}

// MARK: - Memory Manager

struct MacMemoryManagerSheet: View {
    @EnvironmentObject var settings: AppSettings
    let onDismiss: () -> Void

    @State private var pasteText = ""
    @State private var showDocPicker = false
    @State private var statusMessage: String? = nil
    @State private var showClearConfirm = false
    @State private var isSaving = false
    @State private var showChatImport = false
    @State private var editingDocument: MemoryDocument? = nil
    @StateObject private var memoryStore = MemoryStore.shared
    @StateObject private var ragManager = RagServiceManager.shared

    private var canSave: Bool {
        !pasteText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && !isSaving
    }

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    ZStack(alignment: .topLeading) {
                        TextEditor(text: $pasteText)
                            .font(.body)
                            .scrollContentBackground(.hidden)
                            .frame(minHeight: 120)
                        if pasteText.isEmpty {
                            Text(settings.localized("paste_memory_placeholder"))
                                .foregroundStyle(.tertiary)
                                .padding(.leading, 5)
                                .allowsHitTesting(false)
                        }
                    }

                    HStack(spacing: 8) {
                        Button {
                            showDocPicker = true
                        } label: {
                            Label(settings.localized("upload_file"), systemImage: "doc.badge.plus")
                        }
                        .buttonStyle(.bordered)

                        Button {
                            showChatImport = true
                        } label: {
                            Label(settings.localized("import_chat_history"), systemImage: "bubble.left.and.bubble.right")
                        }
                        .buttonStyle(.bordered)

                        Spacer()

                        Button {
                            savePastedText()
                        } label: {
                            if isSaving {
                                ProgressView().controlSize(.small)
                            } else {
                                Label(settings.localized("save_to_memory"), systemImage: "brain")
                            }
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(!canSave)
                    }

                    if let msg = statusMessage {
                        Text(msg)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                } header: {
                    Text(settings.localized("paste_or_upload_to_memory"))
                }

                Section {
                    if memoryStore.documents.isEmpty {
                        Text(settings.localized("no_memories"))
                            .font(.callout)
                            .foregroundStyle(.secondary)
                    } else {
                        ForEach(memoryStore.documents) { doc in
                            MacMemoryDocumentRow(
                                doc: doc,
                                onEdit: { editingDocument = doc },
                                onDelete: {
                                    Task { await RagServiceManager.shared.removeGlobalDocument(docId: doc.id) }
                                }
                            )
                        }
                    }
                } header: {
                    HStack {
                        Text(settings.localized("saved_memories"))
                        Spacer()
                        if !memoryStore.documents.isEmpty {
                            Button(settings.localized("clear_all"), role: .destructive) {
                                showClearConfirm = true
                            }
                            .buttonStyle(.borderless)
                            .controlSize(.small)
                            .foregroundStyle(.red)
                        }
                    }
                }
            }
            .formStyle(.grouped)
            .navigationTitle(settings.localized("manage_memory"))
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button(settings.localized("done")) { onDismiss() }
                }
            }
            .confirmationDialog(settings.localized("confirm_replace_memory_title"), isPresented: $showClearConfirm, titleVisibility: .visible) {
                Button(settings.localized("clear_all"), role: .destructive) {
                    Task { await RagServiceManager.shared.clearGlobalMemory() }
                }
                Button(settings.localized("cancel"), role: .cancel) {}
            } message: {
                Text(settings.localized("confirm_replace_memory_message"))
            }
            .fileImporter(
                isPresented: $showDocPicker,
                allowedContentTypes: DocumentTextExtractor.supportedTypes,
                allowsMultipleSelection: false
            ) { result in
                guard case .success(let urls) = result, let url = urls.first else { return }
                let fileName = url.lastPathComponent
                isSaving = true
                Task {
                    let text: String
                    do {
                        text = try DocumentTextExtractor.extract(from: url)
                    } catch {
                        await MainActor.run {
                            isSaving = false
                            statusMessage = error.localizedDescription
                        }
                        return
                    }
                    let ok = await RagServiceManager.shared.addGlobalMemory(text: text, fileName: fileName, metadata: "uploaded")
                    await MainActor.run {
                        isSaving = false
                        statusMessage = ok
                            ? settings.localized("memory_upload_success")
                            : settings.localized("memory_upload_failed")
                    }
                }
            }
            .sheet(isPresented: $showChatImport) {
                MacChatImportSheet(
                    onDismiss: { showChatImport = false },
                    onImport: { _, success in
                        statusMessage = success
                            ? settings.localized("chat_imported_to_memory")
                            : (RagServiceManager.shared.embeddingNotReadyReason ?? settings.localized("memory_upload_failed"))
                    }
                )
                .environmentObject(settings)
                .frame(minWidth: 480, idealWidth: 560, minHeight: 480, idealHeight: 600)
            }
            .sheet(item: $editingDocument) { doc in
                MacEditMemorySheet(document: doc, onDismiss: { editingDocument = nil })
                    .environmentObject(settings)
                    .frame(minWidth: 480, idealWidth: 560, minHeight: 400, idealHeight: 500)
            }
        }
    }

    private func savePastedText() {
        let trimmed = pasteText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        isSaving = true
        Task {
            let ts = Int(Date().timeIntervalSince1970)
            let ok = await RagServiceManager.shared.addGlobalMemory(
                text: trimmed,
                fileName: "pasted_memory_\(ts).txt",
                metadata: "pasted"
            )
            await MainActor.run {
                isSaving = false
                if ok {
                    statusMessage = settings.localized("memory_save_success")
                    pasteText = ""
                } else {
                    statusMessage = settings.localized("memory_save_failed")
                }
            }
        }
    }
}

private struct MacMemoryDocumentRow: View {
    @EnvironmentObject var settings: AppSettings
    let doc: MemoryDocument
    let onEdit: () -> Void
    let onDelete: () -> Void

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            VStack(alignment: .leading, spacing: 3) {
                Text(displayTitle)
                    .lineLimit(2)
                Text(metaLabel + " • " + doc.createdAt.formatted(.dateTime.month().day().year()))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer(minLength: 4)
            if doc.metadata == "pasted" {
                Button(action: onEdit) {
                    Image(systemName: "pencil")
                }
                .buttonStyle(.borderless)
                .help(settings.localized("edit_memory"))
            }
            Button(role: .destructive, action: onDelete) {
                Image(systemName: "trash")
                    .foregroundStyle(.red)
            }
            .buttonStyle(.borderless)
        }
        .contextMenu {
            if doc.metadata == "pasted" {
                Button(settings.localized("edit_memory"), action: onEdit)
            }
            Button(settings.localized("action_delete"), role: .destructive, action: onDelete)
        }
    }

    private var displayTitle: String {
        if doc.metadata == "pasted" {
            return doc.content.trimmingCharacters(in: .whitespacesAndNewlines)
                .replacingOccurrences(of: "\n", with: " ")
                .prefix(120).description
        }
        return doc.fileName
    }

    private var metaLabel: String {
        switch doc.metadata {
        case "uploaded":    return settings.localized("global_memory_uploaded_by")
        case "pasted":      return settings.localized("global_memory_pasted_by")
        case "chat_import": return settings.localized("chat_imported_to_memory")
        default:            return doc.metadata
        }
    }
}

private struct MacEditMemorySheet: View {
    @EnvironmentObject var settings: AppSettings
    let document: MemoryDocument
    let onDismiss: () -> Void

    @State private var editedContent: String
    @State private var isSaving = false

    init(document: MemoryDocument, onDismiss: @escaping () -> Void) {
        self.document = document
        self.onDismiss = onDismiss
        self._editedContent = State(initialValue: document.content)
    }

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    TextEditor(text: $editedContent)
                        .font(.body)
                        .scrollContentBackground(.hidden)
                        .frame(minHeight: 200)
                }
            }
            .formStyle(.grouped)
            .navigationTitle(settings.localized("edit_memory"))
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button(settings.localized("cancel")) { onDismiss() }
                }
                ToolbarItem(placement: .confirmationAction) {
                    if isSaving {
                        ProgressView().controlSize(.small)
                    } else {
                        Button(settings.localized("save_changes")) {
                            let trimmed = editedContent.trimmingCharacters(in: .whitespacesAndNewlines)
                            guard !trimmed.isEmpty else { return }
                            isSaving = true
                            Task {
                                await RagServiceManager.shared.updateGlobalMemoryDocument(docId: document.id, newContent: trimmed)
                                await MainActor.run {
                                    isSaving = false
                                    onDismiss()
                                }
                            }
                        }
                        .disabled(editedContent.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                    }
                }
            }
        }
    }
}

private struct MacChatImportSheet: View {
    @EnvironmentObject var settings: AppSettings
    let onDismiss: () -> Void
    let onImport: ([ChatSession], Bool) -> Void

    @StateObject private var chatStore = ChatStore.shared
    @State private var selectedIds: Set<UUID> = []
    @State private var isImporting = false

    private func selection(for id: UUID) -> Binding<Bool> {
        Binding(
            get: { selectedIds.contains(id) },
            set: { isOn in
                if isOn { selectedIds.insert(id) } else { selectedIds.remove(id) }
            }
        )
    }

    var body: some View {
        NavigationStack {
            Group {
                if chatStore.chatSessions.isEmpty {
                    Text(settings.localized("no_memories"))
                        .foregroundStyle(.secondary)
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else {
                    Form {
                        Section {
                            ForEach(chatStore.chatSessions) { session in
                                Toggle(isOn: selection(for: session.id)) {
                                    VStack(alignment: .leading, spacing: 2) {
                                        Text(session.title.isEmpty ? settings.localized("drawer_new_chat") : session.title)
                                        Text("\(session.messages.count) messages")
                                            .font(.caption)
                                            .foregroundStyle(.secondary)
                                    }
                                }
                                .toggleStyle(.checkbox)
                            }
                        }
                    }
                    .formStyle(.grouped)
                }
            }
            .navigationTitle(settings.localized("select_chats_to_import"))
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button(settings.localized("cancel")) { onDismiss() }
                }
                ToolbarItem(placement: .confirmationAction) {
                    if isImporting {
                        ProgressView().controlSize(.small)
                    } else {
                        Button(settings.localized("import_chat_history")) {
                            performImport()
                        }
                        .disabled(selectedIds.isEmpty)
                    }
                }
            }
        }
    }

    private func performImport() {
        let toImport = chatStore.chatSessions.filter { selectedIds.contains($0.id) }
        guard !toImport.isEmpty else { return }
        isImporting = true
        Task {
            var allSucceeded = true
            for session in toImport {
                let chatText = session.messages.map { msg in
                    let role = msg.isFromUser ? "User" : "Assistant"
                    return "\(role): \(msg.content)"
                }.joined(separator: "\n\n")
                guard !chatText.isEmpty else { continue }
                let title = session.title.isEmpty ? settings.localized("drawer_new_chat") : session.title
                let ok = await RagServiceManager.shared.addGlobalMemory(
                    text: chatText,
                    fileName: "Chat: \(title)",
                    metadata: "chat_import"
                )
                if !ok { allSucceeded = false }
            }
            await MainActor.run {
                isImporting = false
                onImport(toImport, allSucceeded)
                onDismiss()
            }
        }
    }
}

// MARK: - About

private struct MacAboutSheet: View {
    @EnvironmentObject var settings: AppSettings
    @Environment(\.dismiss) private var dismiss

    private var appVersion: String {
        let version = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "1.0"
        let build = Bundle.main.infoDictionary?["CFBundleVersion"] as? String ?? "1"
        return "\(version) (\(build))"
    }

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    HStack(spacing: 12) {
                        if let icon = NSApp.applicationIconImage {
                            Image(nsImage: icon)
                                .resizable()
                                .frame(width: 48, height: 48)
                        }
                        VStack(alignment: .leading, spacing: 3) {
                            Text(settings.localized("about_llm_hub"))
                                .font(.headline)
                            Text("v\(appVersion)")
                                .font(.subheadline)
                                .foregroundStyle(.secondary)
                                .textSelection(.enabled)
                        }
                    }
                    .padding(.vertical, 4)
                }

                Section {
                    Text(settings.localized("about_description"))
                        .textSelection(.enabled)
                } header: {
                    Label(settings.localized("about"), systemImage: "info.circle")
                }

                Section {
                    Text(settings.localized("about_developer_info"))
                        .textSelection(.enabled)
                } header: {
                    Label("Developer", systemImage: "person.circle")
                }

                Section {
                    Text(settings.localized("about_tech_stack"))
                        .textSelection(.enabled)
                } header: {
                    Label("Technology", systemImage: "cpu")
                }
            }
            .formStyle(.grouped)
            .navigationTitle(settings.localized("about_llm_hub"))
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button(settings.localized("done")) { dismiss() }
                }
            }
        }
    }
}

// MARK: - Terms of Service

private struct MacTermsOfServiceSheet: View {
    @EnvironmentObject var settings: AppSettings
    @Environment(\.dismiss) private var dismiss

    private let tosSections: [(titleKey: String, bodyKey: String)] = [
        ("tos_acceptance_title", "tos_acceptance_text"),
        ("tos_app_description_title", "tos_app_description_text"),
        ("tos_user_responsibilities_title", "tos_user_responsibilities_text"),
        ("tos_privacy_data_title", "tos_privacy_data_text"),
        ("tos_disclaimer_title", "tos_disclaimer_text"),
        ("tos_limitation_liability_title", "tos_limitation_liability_text"),
        ("tos_model_usage_title", "tos_model_usage_text"),
        ("tos_open_source_title", "tos_open_source_text"),
        ("tos_changes_title", "tos_changes_text"),
        ("tos_contact_title", "tos_contact_text"),
    ]

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    VStack(alignment: .leading, spacing: 6) {
                        Text(settings.localized("tos_welcome_text"))
                        Text(settings.localized("tos_last_updated"))
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    .textSelection(.enabled)
                }

                ForEach(Array(tosSections.enumerated()), id: \.offset) { _, section in
                    Section(settings.localized(section.titleKey)) {
                        Text(settings.localized(section.bodyKey))
                            .textSelection(.enabled)
                    }
                }
            }
            .formStyle(.grouped)
            .navigationTitle(settings.localized("terms_of_service"))
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button(settings.localized("done")) { dismiss() }
                }
            }
        }
    }
}

// MARK: - Hugging Face Token

private struct MacHuggingFaceTokenSheet: View {
    @EnvironmentObject var settings: AppSettings
    let onDismiss: () -> Void
    @State private var tokenText: String = ""
    @State private var isSecured: Bool = true

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    Text(settings.localized("hf_token_explanation"))
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                } header: {
                    Text(settings.localized("hf_token_dialog_title"))
                }

                Section {
                    HStack {
                        Group {
                            if isSecured {
                                SecureField("", text: $tokenText, prompt: Text(settings.localized("hf_token_placeholder")))
                            } else {
                                TextField("", text: $tokenText, prompt: Text(settings.localized("hf_token_placeholder")))
                            }
                        }
                        .labelsHidden()
                        .textFieldStyle(.roundedBorder)
                        .autocorrectionDisabled()

                        Button {
                            isSecured.toggle()
                        } label: {
                            Image(systemName: isSecured ? "eye.slash" : "eye")
                        }
                        .buttonStyle(.borderless)
                    }

                    if settings.hasCustomHfToken {
                        HStack {
                            Label(settings.localized("hf_token_custom_active"), systemImage: "checkmark.circle.fill")
                                .font(.caption)
                                .foregroundStyle(ApolloPalette.accentStrong)
                            Spacer()
                            Button(settings.localized("hf_token_clear"), role: .destructive) {
                                settings.customHfToken = ""
                                tokenText = ""
                                onDismiss()
                            }
                            .buttonStyle(.bordered)
                            .tint(.red)
                        }
                    } else {
                        Label(settings.localized("hf_token_using_default"), systemImage: "info.circle")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
            }
            .formStyle(.grouped)
            .navigationTitle(settings.localized("hf_token_title"))
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button(settings.localized("close")) { onDismiss() }
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button(settings.localized("save")) {
                        settings.customHfToken = tokenText.trimmingCharacters(in: .whitespacesAndNewlines)
                        onDismiss()
                    }
                }
            }
            .onAppear {
                tokenText = settings.customHfToken
            }
        }
    }
}
#endif
