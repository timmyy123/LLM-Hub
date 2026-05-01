import SwiftUI
import RunAnywhere

// MARK: - Settings Screen (mirroring Android SettingsScreen.kt)
struct SettingsScreen: View {
    @EnvironmentObject var settings: AppSettings
    @Environment(\.openURL) var openURL

    var onNavigateBack: () -> Void
    var onNavigateToModels: () -> Void
    var onNavigateToMimoBot: (() -> Void)? = nil

    @State private var showLanguageDialog = false
    @State private var showMemoryDialog = false
    @State private var memoryPasteText = ""
    @State private var showMemoryDocPicker = false
    @State private var showMemoryClearConfirm = false
    @State private var memoryStatusMessage: String? = nil
    @StateObject private var ragManager = RagServiceManager.shared
    @State private var showAbout = false
    @State private var showTerms = false

    var body: some View {
        ZStack {
            ApolloLiquidBackground()

            List {
                // MARK: Models Section
                Section {
                    SettingsRow(
                        icon: "square.and.arrow.down.fill",
                        iconColor: ApolloPalette.accentStrong,
                        titleKey: "download_models",
                        subtitleKey: "browse_download_models"
                    ) {
                        onNavigateToModels()
                    }
                    .listRowInsets(EdgeInsets(top: 6, leading: 14, bottom: 6, trailing: 14))
                    .listRowBackground(Color.clear)
                } header: {
                    SectionHeader(titleKey: "models", icon: "cpu")
                }

                // MARK: RAG / Embedding Section
                Section {
                    // Embedding model selector
                    EmbeddingModelSelectorRow(onNavigateToModels: onNavigateToModels)
                        .listRowInsets(EdgeInsets(top: 6, leading: 14, bottom: 6, trailing: 14))
                        .listRowBackground(Color.clear)

                    // Memory toggle (requires embedding model)
                    SettingsToggleRow(
                        icon: "brain",
                        iconColor: ApolloPalette.accentStrong,
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
                    .listRowInsets(EdgeInsets(top: 6, leading: 14, bottom: 6, trailing: 14))
                    .listRowBackground(Color.clear)

                    // Memory manager row (only when memory enabled)
                    if settings.memoryEnabled && settings.selectedEmbeddingModelId != nil {
                        SettingsRow(
                            icon: "tray.full",
                            iconColor: ApolloPalette.accentStrong,
                            titleKey: "manage_memory",
                            subtitleKey: "manage_memory_subtitle"
                        ) {
                            Task {
                                _ = await RunAnywhere.discoverDownloadedModels()
                                await RagServiceManager.shared.initialize(modelId: settings.selectedEmbeddingModelId)
                                await MainActor.run {
                                    showMemoryDialog = true
                                }
                            }
                        }
                        .listRowInsets(EdgeInsets(top: 6, leading: 14, bottom: 6, trailing: 14))
                        .listRowBackground(Color.clear)
                    }
                } header: {
                    SectionHeader(titleKey: "embedding_models", icon: "link.circle")
                }

                // MARK: Appearance Section
                Section {
                    SettingsToggleRow(
                        icon: "speaker.wave.2.fill",
                        iconColor: ApolloPalette.accentStrong,
                        title: settings.localized("auto_readout"),
                        subtitle: settings.localized("auto_readout_description"),
                        isOn: $settings.autoReadoutEnabled
                    )
                    .listRowInsets(EdgeInsets(top: 6, leading: 14, bottom: 6, trailing: 14))
                    .listRowBackground(Color.clear)

                    SettingsRow(
                        icon: "globe",
                        iconColor: ApolloPalette.accentStrong,
                        titleKey: "language",
                        subtitleString: settings.localized(settings.selectedLanguage.displayNameKey)
                    ) {
                        showLanguageDialog = true
                    }
                    .listRowInsets(EdgeInsets(top: 6, leading: 14, bottom: 6, trailing: 14))
                    .listRowBackground(Color.clear)
                } header: {
                    SectionHeader(titleKey: "appearance", icon: "paintbrush")
                }

                // MARK: Experimental Section
                if let onNavigateToMimoBot {
                    Section {
                        SettingsRow(
                            icon: "mic.circle.fill",
                            iconColor: ApolloPalette.accentStrong,
                            titleKey: "Mimo Bot (test)",
                            subtitleString: "Voice loop using phone mic + speaker"
                        ) { onNavigateToMimoBot() }
                        .listRowInsets(EdgeInsets(top: 6, leading: 14, bottom: 6, trailing: 14))
                        .listRowBackground(Color.clear)
                    } header: {
                        SectionHeader(titleKey: "Experimental", icon: "wand.and.stars")
                    }
                }

                // MARK: Information Section
                Section {
                    SettingsRow(
                        icon: "info.circle.fill",
                        iconColor: ApolloPalette.accentStrong,
                        titleKey: "about",
                        subtitleKey: "app_information_contact"
                    ) { showAbout = true }
                    .listRowInsets(EdgeInsets(top: 6, leading: 14, bottom: 6, trailing: 14))
                    .listRowBackground(Color.clear)

                    SettingsRow(
                        icon: "doc.text.fill",
                        iconColor: ApolloPalette.accentStrong,
                        titleKey: "terms_of_service",
                        subtitleKey: "legal_terms_conditions"
                    ) { showTerms = true }
                    .listRowInsets(EdgeInsets(top: 6, leading: 14, bottom: 6, trailing: 14))
                    .listRowBackground(Color.clear)
                } header: {
                    SectionHeader(titleKey: "information", icon: "info.circle")
                }

                // MARK: Source Code Section
                Section {
                    SettingsRow(
                        icon: "chevron.left.forwardslash.chevron.right",
                        iconColor: ApolloPalette.accentStrong,
                        titleKey: "github_repository",
                        subtitleKey: "view_source_contribute"
                    ) {
                        if let url = URL(string: "https://github.com/timmyy123/LLM-Hub") {
                            openURL(url)
                        }
                    }
                    .listRowInsets(EdgeInsets(top: 6, leading: 14, bottom: 6, trailing: 14))
                    .listRowBackground(Color.clear)
                } header: {
                    SectionHeader(titleKey: "source_code_section", icon: "curlybraces")
                }
            }
            .listStyle(.insetGrouped)
            .scrollContentBackground(.hidden)
        }
        .navigationTitle(settings.localized("feature_settings_title"))
        .navigationBarTitleDisplayMode(.large)
        .toolbarBackground(.hidden, for: .navigationBar)
        .toolbar {
            ToolbarItem(placement: .navigationBarLeading) {
                Button {
                    onNavigateBack()
                } label: {
                    HStack(spacing: 4) {
                        Image(systemName: "chevron.left")
                        Text(settings.localized("back"))
                    }
                }
            }
        }
        // Language Dialog
        .sheet(isPresented: $showLanguageDialog) {
            LanguagePickerSheet()
                .environmentObject(settings)
        }
        // Memory Manager Sheet
        .sheet(isPresented: $showMemoryDialog) {
            MemoryManagerSheet(onDismiss: { showMemoryDialog = false })
                .environmentObject(settings)
        }
        .sheet(isPresented: $showAbout) {
            AboutScreen()
                .environmentObject(settings)
        }
        .sheet(isPresented: $showTerms) {
            TermsOfServiceScreen()
                .environmentObject(settings)
        }
        .onChange(of: settings.selectedEmbeddingModelId) { _, newId in
            // Initialization and re-embedding handled by EmbeddingModelSelectorRow.
            // This handles external changes (e.g. model deleted from download screen).
            Task {
                if newId == nil {
                    await RagServiceManager.shared.initialize(modelId: nil)
                }
            }
        }
    }
}

// MARK: - Embedding Model Selector Row

private struct EmbeddingModelSelectorRow: View {
    @EnvironmentObject var settings: AppSettings
    let onNavigateToModels: () -> Void
    @State private var showPicker = false

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
        Button {
            if downloadedEmbeddingModels.isEmpty {
                onNavigateToModels()
            } else {
                showPicker = true
            }
        } label: {
            HStack(spacing: 14) {
                RoundedRectangle(cornerRadius: 8)
                    .fill(LinearGradient(colors: [ApolloPalette.accentSoft, ApolloPalette.accentMuted], startPoint: .topLeading, endPoint: .bottomTrailing))
                    .frame(width: 32, height: 32)
                    .overlay {
                        Image(systemName: "link.circle.fill")
                            .font(.system(size: 16))
                            .foregroundColor(.white)
                    }
                VStack(alignment: .leading, spacing: 2) {
                    Text(settings.localized("embedding_model"))
                        .font(.subheadline).foregroundColor(.white)
                    Text(selectedModel?.name ?? settings.localized("no_embedding_model_selected"))
                        .font(.caption).foregroundColor(.white.opacity(0.65)).lineLimit(1)
                }
                Spacer()
                Image(systemName: "chevron.right")
                    .font(.caption.bold()).foregroundColor(.white.opacity(0.55))
            }
            .padding(.horizontal, 12).padding(.vertical, 12)
            .background(.ultraThinMaterial)
            .clipShape(RoundedRectangle(cornerRadius: 16))
            .overlay(RoundedRectangle(cornerRadius: 16).stroke(Color.white.opacity(0.16), lineWidth: 1))
        }
        .buttonStyle(.plain)
        .confirmationDialog(settings.localized("select_embedding_model"), isPresented: $showPicker, titleVisibility: .visible) {
            ForEach(downloadedEmbeddingModels) { model in
                Button(model.name) {
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
            if settings.selectedEmbeddingModelId != nil {
                Button(settings.localized("disable_embeddings"), role: .destructive) {
                    settings.selectedEmbeddingModelId = nil
                    settings.memoryEnabled = false
                }
            }
            Button(settings.localized("cancel"), role: .cancel) {}
        }
    }
}

// MARK: - Memory Manager Sheet

struct MemoryManagerSheet: View {
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

    var body: some View {
        NavigationView {
            ZStack {
                ApolloLiquidBackground()
                ScrollView {
                    VStack(spacing: 16) {
                        // Paste text field
                        VStack(alignment: .leading, spacing: 8) {
                            Text(settings.localized("paste_or_upload_to_memory"))
                                .font(.subheadline).foregroundColor(.white.opacity(0.8))
                            ZStack(alignment: .topLeading) {
                                TextEditor(text: $pasteText)
                                    .frame(minHeight: 120)
                                    .padding(8)
                                    .background(.ultraThinMaterial)
                                    .clipShape(RoundedRectangle(cornerRadius: 12))
                                    .overlay(RoundedRectangle(cornerRadius: 12).stroke(Color.white.opacity(0.16), lineWidth: 1))
                                    .foregroundColor(.white)
                                if pasteText.isEmpty {
                                    Text(settings.localized("paste_memory_placeholder"))
                                        .foregroundColor(.white.opacity(0.35))
                                        .padding(.horizontal, 12).padding(.top, 16)
                                        .allowsHitTesting(false)
                                }
                            }
                        }
                        .padding(.horizontal)

                        HStack(spacing: 12) {
                            // Upload file button
                            Button {
                                showDocPicker = true
                            } label: {
                                Label(settings.localized("upload_file"), systemImage: "doc.badge.plus")
                                    .foregroundColor(.white)
                                    .padding(.horizontal, 16).padding(.vertical, 10)
                                    .background(.ultraThinMaterial)
                                    .clipShape(Capsule())
                                    .overlay(Capsule().stroke(Color.white.opacity(0.16), lineWidth: 1))
                            }

                            // Save pasted text button
                            Button {
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
                            } label: {
                                if isSaving {
                                    ProgressView().tint(.white)
                                } else {
                                    Label(settings.localized("save_to_memory"), systemImage: "brain")
                                        .foregroundColor(.white)
                                }
                            }
                            .padding(.horizontal, 16).padding(.vertical, 10)
                            .background(ApolloPalette.accentStrong)
                            .clipShape(Capsule())
                            .disabled(pasteText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || isSaving)
                        }
                        .padding(.horizontal)

                        // Import Chat History button
                        Button {
                            showChatImport = true
                        } label: {
                            Label(settings.localized("import_chat_history"), systemImage: "bubble.left.and.bubble.right")
                                .foregroundColor(.white)
                                .frame(maxWidth: .infinity)
                                .padding(.vertical, 12)
                                .background(.ultraThinMaterial)
                                .clipShape(RoundedRectangle(cornerRadius: 12))
                                .overlay(RoundedRectangle(cornerRadius: 12).stroke(Color.white.opacity(0.16), lineWidth: 1))
                        }
                        .padding(.horizontal)

                        // Status
                        if let msg = statusMessage {
                            Text(msg)
                                .font(.caption).foregroundColor(.white.opacity(0.8))
                                .padding(.horizontal)
                        }

                        Divider().background(Color.white.opacity(0.12))

                        // Saved memories list
                        VStack(alignment: .leading, spacing: 8) {
                            HStack {
                                Text(settings.localized("saved_memories"))
                                    .font(.headline).foregroundColor(.white)
                                Spacer()
                                if !memoryStore.documents.isEmpty {
                                    Button {
                                        showClearConfirm = true
                                    } label: {
                                        Text(settings.localized("clear_all"))
                                            .font(.caption)
                                            .foregroundColor(.red.opacity(0.8))
                                    }
                                }
                            }
                            .padding(.horizontal)

                            if memoryStore.documents.isEmpty {
                                Text(settings.localized("no_memories"))
                                    .font(.caption).foregroundColor(.white.opacity(0.55))
                                    .padding(.horizontal)
                            } else {
                                ForEach(memoryStore.documents) { doc in
                                    MemoryDocumentRow(
                                        doc: doc,
                                        onEdit: { editingDocument = doc },
                                        onDelete: {
                                            Task { await RagServiceManager.shared.removeGlobalDocument(docId: doc.id) }
                                        }
                                    )
                                    .padding(.horizontal)
                                }
                            }
                        }
                    }
                    .padding(.vertical)
                }
            }
            .navigationTitle(settings.localized("manage_memory"))
            .navigationBarTitleDisplayMode(.inline)
            .toolbarBackground(.hidden, for: .navigationBar)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button(settings.localized("done")) { onDismiss() }
                        .foregroundColor(.white)
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
                ChatImportSheet(
                    onDismiss: { showChatImport = false },
                    onImport: { _, success in
                        statusMessage = success
                            ? settings.localized("chat_imported_to_memory")
                            : (RagServiceManager.shared.embeddingNotReadyReason ?? settings.localized("memory_upload_failed"))
                    }
                )
                .environmentObject(settings)
            }
            .sheet(item: $editingDocument) { doc in
                EditMemorySheet(document: doc, onDismiss: { editingDocument = nil })
                    .environmentObject(settings)
            }
        }
    }
}

// MARK: - MemoryDocumentRow

private struct MemoryDocumentRow: View {
    @EnvironmentObject var settings: AppSettings
    let doc: MemoryDocument
    let onEdit: () -> Void
    let onDelete: () -> Void

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            VStack(alignment: .leading, spacing: 3) {
                Text(displayTitle)
                    .font(.subheadline).foregroundColor(.white).lineLimit(2)
                Text(metaLabel + " • " + doc.createdAt.formatted(.dateTime.month().day().year()))
                    .font(.caption).foregroundColor(.white.opacity(0.55))
            }
            Spacer(minLength: 4)
            if doc.metadata == "pasted" {
                Button(action: onEdit) {
                    Image(systemName: "pencil")
                        .foregroundColor(.white.opacity(0.7))
                        .frame(width: 32, height: 32)
                }
            }
            Button(action: onDelete) {
                Image(systemName: "trash")
                    .foregroundColor(.red.opacity(0.7))
                    .frame(width: 32, height: 32)
            }
        }
        .padding(10)
        .background(.ultraThinMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 10))
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

// MARK: - EditMemorySheet

private struct EditMemorySheet: View {
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
        NavigationView {
            ZStack {
                ApolloLiquidBackground()
                VStack(spacing: 16) {
                    TextEditor(text: $editedContent)
                        .frame(maxWidth: .infinity, minHeight: 200)
                        .padding(8)
                        .background(.ultraThinMaterial)
                        .clipShape(RoundedRectangle(cornerRadius: 12))
                        .overlay(RoundedRectangle(cornerRadius: 12).stroke(Color.white.opacity(0.16), lineWidth: 1))
                        .foregroundColor(.white)
                        .padding(.horizontal)
                    Spacer()
                }
                .padding(.top, 16)
            }
            .navigationTitle(settings.localized("edit_memory"))
            .navigationBarTitleDisplayMode(.inline)
            .toolbarBackground(.hidden, for: .navigationBar)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button(settings.localized("cancel")) { onDismiss() }
                        .foregroundColor(.white)
                }
                ToolbarItem(placement: .confirmationAction) {
                    if isSaving {
                        ProgressView().tint(.white)
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
                        .foregroundColor(.white)
                        .disabled(editedContent.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                    }
                }
            }
        }
    }
}

// MARK: - ChatImportSheet

private struct ChatImportSheet: View {
    @EnvironmentObject var settings: AppSettings
    let onDismiss: () -> Void
    let onImport: ([ChatSession], Bool) -> Void

    @StateObject private var chatStore = ChatStore.shared
    @State private var selectedIds: Set<UUID> = []
    @State private var isImporting = false

    var body: some View {
        NavigationView {
            ZStack {
                ApolloLiquidBackground()
                if chatStore.chatSessions.isEmpty {
                    Text(settings.localized("no_memories"))
                        .foregroundColor(.white.opacity(0.55))
                } else {
                    List(chatStore.chatSessions) { session in
                        Button {
                            if selectedIds.contains(session.id) {
                                selectedIds.remove(session.id)
                            } else {
                                selectedIds.insert(session.id)
                            }
                        } label: {
                            HStack {
                                Image(systemName: selectedIds.contains(session.id) ? "checkmark.circle.fill" : "circle")
                                    .foregroundColor(selectedIds.contains(session.id) ? ApolloPalette.accentStrong : .white.opacity(0.4))
                                VStack(alignment: .leading, spacing: 2) {
                                    Text(session.title.isEmpty ? settings.localized("drawer_new_chat") : session.title)
                                        .foregroundColor(.white).font(.subheadline)
                                    Text("\(session.messages.count) messages")
                                        .foregroundColor(.white.opacity(0.5)).font(.caption)
                                }
                            }
                        }
                        .listRowBackground(Color.clear)
                        .listRowInsets(EdgeInsets(top: 6, leading: 14, bottom: 6, trailing: 14))
                    }
                    .listStyle(.insetGrouped)
                    .scrollContentBackground(.hidden)
                }
            }
            .navigationTitle(settings.localized("select_chats_to_import"))
            .navigationBarTitleDisplayMode(.inline)
            .toolbarBackground(.hidden, for: .navigationBar)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button(settings.localized("cancel")) { onDismiss() }
                        .foregroundColor(.white)
                }
                ToolbarItem(placement: .confirmationAction) {
                    if isImporting {
                        ProgressView().tint(.white)
                    } else {
                        Button(settings.localized("import_chat_history")) {
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
                        .foregroundColor(selectedIds.isEmpty ? .white.opacity(0.4) : .white)
                        .disabled(selectedIds.isEmpty)
                    }
                }
            }
        }
    }
}

// MARK: - Language Picker Sheet
struct LanguagePickerSheet: View {
    @EnvironmentObject var settings: AppSettings
    @Environment(\.dismiss) var dismiss

    var body: some View {
        ZStack {
            ApolloLiquidBackground()

            List {
                ForEach(AppLanguage.allCases) { lang in
                    Button {
                        settings.selectedLanguage = lang
                        dismiss()
                    } label: {
                        HStack {
                            Text(settings.localized(lang.displayNameKey))
                                .foregroundColor(.white)
                                .environment(\.layoutDirection, lang.isRTL ? .rightToLeft : .leftToRight)
                            Spacer()
                            if settings.selectedLanguage == lang {
                                Image(systemName: "checkmark.circle.fill")
                                    .foregroundColor(ApolloPalette.accentStrong)
                            }
                        }
                    }
                    .listRowBackground(Color.clear)
                    .listRowInsets(EdgeInsets(top: 6, leading: 14, bottom: 6, trailing: 14))
                }
            }
            .listStyle(.insetGrouped)
            .scrollContentBackground(.hidden)
        }
        .navigationTitle(settings.localized("select_language"))
        .navigationBarTitleDisplayMode(.inline)
        .toolbarBackground(.hidden, for: .navigationBar)
        .toolbar {
            ToolbarItem(placement: .confirmationAction) {
                Button(settings.localized("done")) { dismiss() }
            }
        }
    }
}


// MARK: - Reusable Components

struct SectionHeader: View {
    @EnvironmentObject var settings: AppSettings
    let titleKey: String
    let icon: String

    var body: some View {
        HStack(spacing: 6) {
            Image(systemName: icon)
                .foregroundColor(ApolloPalette.accentStrong)
            Text(settings.localized(titleKey))
        }

        .font(.footnote.bold())
        .foregroundColor(.white.opacity(0.74))
        .textCase(nil)
    }
}

struct SettingsRow: View {
    @EnvironmentObject var settings: AppSettings
    let icon: String
    let iconColor: Color
    let titleKey: String
    var subtitleKey: String? = nil
    var subtitleString: String? = nil
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 14) {
                RoundedRectangle(cornerRadius: 8)
                    .fill(
                        LinearGradient(
                            colors: [ApolloPalette.accentSoft, ApolloPalette.accentMuted],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        )
                    )
                    .frame(width: 32, height: 32)
                    .overlay {
                        Image(systemName: icon)
                            .font(.system(size: 16))
                            .foregroundColor(.white)
                    }

                VStack(alignment: .leading, spacing: 2) {
                    Text(settings.localized(titleKey))
                        .font(.subheadline)
                        .foregroundColor(.white)
                    if let sk = subtitleKey {
                        Text(settings.localized(sk))
                            .font(.caption)
                            .foregroundColor(.white.opacity(0.65))
                            .lineLimit(1)
                    } else if let ss = subtitleString {
                        Text(verbatim: ss)
                            .font(.caption)
                            .foregroundColor(.white.opacity(0.65))
                            .lineLimit(1)
                    }
                }

                Spacer()

                Image(systemName: "chevron.right")
                    .font(.caption.bold())
                    .foregroundColor(.white.opacity(0.55))
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 12)
            .background(.ultraThinMaterial)
            .clipShape(RoundedRectangle(cornerRadius: 16))
            .overlay(
                RoundedRectangle(cornerRadius: 16)
                    .stroke(Color.white.opacity(0.16), lineWidth: 1)
            )
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }
}

struct SettingsToggleRow: View {
    let icon: String
    let iconColor: Color
    let title: String
    let subtitle: String
    @Binding var isOn: Bool

    var body: some View {
        HStack(spacing: 14) {
            RoundedRectangle(cornerRadius: 8)
                .fill(
                    LinearGradient(
                        colors: [ApolloPalette.accentSoft, ApolloPalette.accentMuted],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    )
                )
                .frame(width: 32, height: 32)
                .overlay {
                    Image(systemName: icon)
                        .font(.system(size: 16))
                        .foregroundColor(.white)
                }

            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.subheadline)
                    .foregroundColor(.white)
                Text(subtitle)
                    .font(.caption)
                    .foregroundColor(.white.opacity(0.65))
                    .lineLimit(2)
            }

            Spacer()

            Toggle("", isOn: $isOn)
                .labelsHidden()
                .tint(ApolloPalette.accentStrong)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 12)
        .background(.ultraThinMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 16))
        .overlay(
            RoundedRectangle(cornerRadius: 16)
                .stroke(Color.white.opacity(0.16), lineWidth: 1)
        )
    }
}

// MARK: - About Screen

struct AboutScreen: View {
    @EnvironmentObject var settings: AppSettings
    @Environment(\.dismiss) private var dismiss

    private var appVersion: String {
        let version = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "1.0"
        let build = Bundle.main.infoDictionary?["CFBundleVersion"] as? String ?? "1"
        return "\(version) (\(build))"
    }

    var body: some View {
        NavigationView {
            ZStack {
                ApolloLiquidBackground()

                List {
                    // App identity header row
                    Section {
                        VStack(alignment: .leading, spacing: 3) {
                            Text(settings.localized("about_llm_hub"))
                                .font(.headline)
                                .foregroundColor(.white)
                            Text("v\(appVersion)")
                                .font(.subheadline)
                                .foregroundColor(.white.opacity(0.55))
                        }
                        .padding(.vertical, 4)
                        .listRowBackground(Color.clear)
                        .listRowInsets(EdgeInsets(top: 6, leading: 14, bottom: 6, trailing: 14))
                    }

                    Section {
                        Text(settings.localized("about_description"))
                            .font(.body)
                            .foregroundColor(.white.opacity(0.85))
                            .listRowBackground(Color.clear)
                            .listRowInsets(EdgeInsets(top: 6, leading: 14, bottom: 6, trailing: 14))
                    } header: {
                        SectionHeader(titleKey: "about", icon: "info.circle")
                    }

                    Section {
                        Text(settings.localized("about_developer_info"))
                            .font(.body)
                            .foregroundColor(.white.opacity(0.85))
                            .listRowBackground(Color.clear)
                            .listRowInsets(EdgeInsets(top: 6, leading: 14, bottom: 6, trailing: 14))
                    } header: {
                        HStack(spacing: 6) {
                            Image(systemName: "person.circle")
                                .foregroundColor(ApolloPalette.accentStrong)
                            Text("Developer")
                        }
                        .font(.footnote.bold())
                        .foregroundColor(.white.opacity(0.74))
                        .textCase(nil)
                    }

                    Section {
                        Text(settings.localized("about_tech_stack"))
                            .font(.body)
                            .foregroundColor(.white.opacity(0.85))
                            .listRowBackground(Color.clear)
                            .listRowInsets(EdgeInsets(top: 6, leading: 14, bottom: 6, trailing: 14))
                    } header: {
                        HStack(spacing: 6) {
                            Image(systemName: "cpu")
                                .foregroundColor(ApolloPalette.accentStrong)
                            Text("Technology")
                        }
                        .font(.footnote.bold())
                        .foregroundColor(.white.opacity(0.74))
                        .textCase(nil)
                    }
                }
                .listStyle(.insetGrouped)
                .scrollContentBackground(.hidden)
            }
            .navigationTitle(settings.localized("about_llm_hub"))
            .navigationBarTitleDisplayMode(.inline)
            .toolbarBackground(.hidden, for: .navigationBar)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button(settings.localized("done")) { dismiss() }
                        .foregroundColor(.white)
                }
            }
        }
    }
}

// MARK: - Terms of Service Screen

struct TermsOfServiceScreen: View {
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
        NavigationView {
            ZStack {
                ApolloLiquidBackground()

                List {
                    Section {
                        VStack(alignment: .leading, spacing: 6) {
                            Text(settings.localized("tos_welcome_text"))
                                .font(.body)
                                .foregroundColor(.white.opacity(0.85))
                            Text(settings.localized("tos_last_updated"))
                                .font(.caption)
                                .foregroundColor(.white.opacity(0.45))
                        }
                        .listRowBackground(Color.clear)
                        .listRowInsets(EdgeInsets(top: 6, leading: 14, bottom: 6, trailing: 14))
                    }

                    ForEach(Array(tosSections.enumerated()), id: \.offset) { _, section in
                        Section {
                            Text(settings.localized(section.bodyKey))
                                .font(.body)
                                .foregroundColor(.white.opacity(0.85))
                                .listRowBackground(Color.clear)
                                .listRowInsets(EdgeInsets(top: 6, leading: 14, bottom: 6, trailing: 14))
                        } header: {
                            Text(settings.localized(section.titleKey))
                                .font(.footnote.bold())
                                .foregroundColor(.white.opacity(0.74))
                                .textCase(nil)
                        }
                    }
                }
                .listStyle(.insetGrouped)
                .scrollContentBackground(.hidden)
            }
            .navigationTitle(settings.localized("terms_of_service"))
            .navigationBarTitleDisplayMode(.inline)
            .toolbarBackground(.hidden, for: .navigationBar)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button(settings.localized("done")) { dismiss() }
                        .foregroundColor(.white)
                }
            }
        }
    }
}
