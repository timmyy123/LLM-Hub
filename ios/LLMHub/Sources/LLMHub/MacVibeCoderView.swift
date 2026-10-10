//
//  MacVibeCoderView.swift
//  LLMHub
//
//  Native macOS Vibe Coder. Same persisted settings, chat sessions, workspace
//  folder bookmark, prompts, model handling and localized strings as
//  `VibeCoderScreen` on iOS, laid out as file list | editor + live preview | chat.
//

#if os(macOS)
import SwiftUI
import AppKit
import WebKit

struct MacVibeCoderView: View {
    @EnvironmentObject var settings: AppSettings
    @ObservedObject private var ttsManager = OnDeviceTtsManager.shared
    @ObservedObject private var llm = LLMBackend.shared

    @AppStorage("feature_vibecoder_model_name") private var selectedModelName: String = ""
    @AppStorage("feature_vibecoder_enable_thinking") private var enableThinking: Bool = true
    @AppStorage("feature_vibecoder_folder_bookmark") private var folderBookmark: Data = Data()
    @AppStorage("feature_vibecoder_chat_sessions_data") private var chatSessionsData: Data = Data()
    @AppStorage("feature_vibecoder_active_chat_session_id") private var activeChatSessionIdRaw: String = ""
    @AppStorage("feature_vibecoder_current_file_relative_path") private var currentFileRelativePath: String = ""

    @State private var maxTokens: Double = 4096
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
    @State private var previewHTML: String = ""
    @State private var previewUpdateTask: Task<Void, Never>?

    private enum VibeFocusField: Hashable {
        case chat
        case editor
    }

    @FocusState private var focusedField: VibeFocusField?

    // MARK: - Context window persistence (same keys as iOS)

    private func loadVibecoderContextWindow(for modelName: String) -> Double {
        let key = "feature_vibecoder_max_tokens_\(modelName)"
        let val = UserDefaults.standard.double(forKey: key)
        if val > 0 {
            return val
        }
        if let model = selectedFeatureModel(named: modelName) {
            let cap = contextLimitForFeatureModel(model)
            return Double(min(featureDefaultContextWindow, cap))
        }
        return 4096
    }

    private func saveVibecoderContextWindow(value: Double, for modelName: String) {
        let key = "feature_vibecoder_max_tokens_\(modelName)"
        UserDefaults.standard.set(value, forKey: key)
    }

    // MARK: - Derived state

    private var isSelectedModelMuseGlimmer: Bool {
        selectedFeatureModel(named: selectedModelName)?.chatTemplateFamily == .museGlimmer
    }

    private var isSelectedModelGranite42: Bool {
        let name = selectedModelName.lowercased()
        return name.contains("granite-4.2") || name.contains("granite 4.2")
    }

    private var effectiveEnableThinking: Bool {
        enableThinking && !isSelectedModelMuseGlimmer && !isSelectedModelGranite42
    }

    private var preferThinkingWhileStreaming: Bool {
        effectiveEnableThinking
            && (selectedFeatureModel(named: selectedModelName)?.supportsThinking == true)
            && supportsUnmarkedStreamingThinkingHeuristic(forModelNamed: selectedModelName)
    }

    private var hasFileSession: Bool {
        !(currentFileName ?? "").isEmpty
    }

    private var hasWorkspaceFolder: Bool {
        workspaceFolderURL != nil
    }

    private var hasCode: Bool {
        !generatedCode.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    private var sendButtonIconName: String {
        isGenerating ? "xmark" : "paperplane.fill"
    }

    private var isSendButtonDisabled: Bool {
        (!isGenerating && chatInput.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty) || !hasFileSession || isLoading
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
        let activeTextChars = activeMessages.reduce(0) { acc, msg in
            if msg.role == "user" {
                return acc + msg.text.count
            }
            if contentHasThinkingMarkers(msg.text) {
                let answer = getDisplayContentWithoutThinking(msg.text)
                return acc + answer.count
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

    private var fileSelectionBinding: Binding<URL?> {
        Binding(
            get: { currentFileURL },
            set: { newValue in
                if let newValue, newValue != currentFileURL {
                    openFile(newValue)
                }
            }
        )
    }

    // MARK: - Body

    var body: some View {
        Group {
            if selectedModelName.isEmpty {
                MacFeatureLoadModelPrompt { showSettings = true }
            } else if !hasWorkspaceFolder {
                noFolderView
            } else {
                workspaceView
            }
        }
        .navigationTitle(settings.localized("vibe_coder_title"))
        .toolbar {
            if !selectedModelName.isEmpty && hasWorkspaceFolder {
                ToolbarItemGroup(placement: .primaryAction) {
                    Button {
                        showWorkspaceFolderPicker = true
                    } label: {
                        Label(settings.localized("vibe_coder_open_folder"), systemImage: "folder")
                    }

                    Button {
                        showCreateFileDialog = true
                    } label: {
                        Label(settings.localized("vibe_coder_new_file"), systemImage: "plus")
                    }

                    Button {
                        NSPasteboard.general.clearContents()
                        NSPasteboard.general.setString(generatedCode, forType: .string)
                    } label: {
                        Label(settings.localized("vibe_coder_copy_code"), systemImage: "doc.on.doc")
                    }
                    .disabled(!hasCode)

                    Button {
                        saveCurrentFile()
                    } label: {
                        Label(settings.localized("vibe_coder_save_file"), systemImage: "square.and.arrow.down")
                    }
                    .keyboardShortcut("s", modifiers: .command)
                    .disabled(!hasFileSession)

                    if isHTMLFile {
                        Button {
                            openHTMLPreviewInSafari()
                        } label: {
                            Image(systemName: "safari")
                        }
                        .disabled(!hasCode)
                    }
                }
            }
            ToolbarItem(placement: .primaryAction) {
                Button {
                    showSettings.toggle()
                } label: {
                    Label(settings.localized("feature_settings_title"), systemImage: "slider.horizontal.3")
                }
            }
        }
        .inspector(isPresented: $showSettings) {
            MacFeatureModelInspector(
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
                onUnload: { llm.unloadModel() },
                showsThinkingToggle: !isSelectedModelMuseGlimmer
            )
            .inspectorColumnWidth(min: 280, ideal: 320, max: 420)
        }
        .onAppear {
            restoreChatSessionsFromStorage()

            Task {
                await refreshDownloadedModelStatus()
                let available = downloadableFeatureModels().filter(isNonTranslatorFeatureModel)
                let hasSelectedModelName = !selectedModelName.isEmpty
                let selectedModelExists = available.contains { model in
                    model.name == selectedModelName
                }
                if !hasSelectedModelName || !selectedModelExists {
                    selectedModelName = available.first?.name ?? ""
                }
                maxTokens = loadVibecoderContextWindow(for: selectedModelName)
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
        .onChange(of: selectedModelName) { _, newModelName in
            maxTokens = loadVibecoderContextWindow(for: newModelName)
        }
        .onChange(of: maxTokens) { _, newValue in
            saveVibecoderContextWindow(value: newValue, for: selectedModelName)
        }
        .onChange(of: workspaceFolderURL) { _, newValue in
            if newValue != nil {
                refreshWorkspaceFiles()
                restoreLastOpenedFileIfPossible()
            }
        }
        .onChange(of: currentFileURL) { _, _ in
            persistCurrentFileSelection()
            previewHTML = generatedCode
        }
        .onChange(of: generatedCode) { _, _ in
            schedulePreviewUpdate()
            guard hasFileSession, !isGenerating else { return }
            scheduleAutosave()
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
            previewUpdateTask?.cancel()
            previewUpdateTask = nil
            llm.unloadModel()
            stopPreviewServer()
        }
    }

    // MARK: - Empty states

    private var noFolderView: some View {
        ContentUnavailableView {
            Label(settings.localized("vibe_coder_select_folder_title"), systemImage: "folder.fill")
        } description: {
            Text(settings.localized("vibe_coder_select_folder_desc"))
        } actions: {
            Button(settings.localized("vibe_coder_open_folder")) {
                showWorkspaceFolderPicker = true
            }
            .buttonStyle(.borderedProminent)
        }
    }

    // MARK: - Workspace

    private var workspaceView: some View {
        VStack(spacing: 0) {
            HStack(spacing: 0) {
                fileListColumn
                    .frame(width: 210)

                Divider()

                editorColumn
                    .frame(minWidth: 280, maxWidth: .infinity, maxHeight: .infinity)

                if isHTMLFile {
                    Divider()
                    previewColumn
                        .frame(minWidth: 240, maxWidth: .infinity, maxHeight: .infinity)
                }

                Divider()

                chatColumn
                    .frame(minWidth: 300, idealWidth: 340, maxWidth: 380, maxHeight: .infinity)
            }

            if let errorMessage {
                Divider()
                HStack {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundStyle(.red)
                    Text(errorMessage)
                        .font(.caption)
                        .foregroundStyle(.red)
                        .textSelection(.enabled)
                        .lineLimit(3)
                    Spacer()
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 6)
            }
        }
    }

    private var fileListColumn: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: 6) {
                Image(systemName: "folder")
                    .foregroundStyle(.secondary)
                Text(workspaceFolderURL?.lastPathComponent ?? "")
                    .font(.headline)
                    .lineLimit(1)
                    .truncationMode(.middle)
                Spacer()
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 10)

            List(selection: fileSelectionBinding) {
                ForEach(workspaceFiles, id: \.self) { url in
                    HStack(spacing: 6) {
                        Label(url.lastPathComponent, systemImage: "doc.text")
                            .lineLimit(1)
                            .truncationMode(.middle)
                        Spacer(minLength: 4)
                        Button {
                            pendingDeleteFileURL = url
                        } label: {
                            Image(systemName: "xmark")
                                .font(.caption.weight(.bold))
                        }
                        .buttonStyle(.borderless)
                        .help(settings.localized("vibe_coder_delete_file_cd"))
                        .disabled(isGenerating)
                    }
                    .tag(url)
                    .contextMenu {
                        Button(settings.localized("delete"), role: .destructive) {
                            pendingDeleteFileURL = url
                        }
                        .disabled(isGenerating)
                    }
                }
            }
            .listStyle(.sidebar)
        }
    }

    private var editorColumn: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(currentFileName ?? settings.localized("vibe_coder_open_or_create_file"))
                .font(.headline)
                .foregroundStyle(currentFileName == nil ? .secondary : .primary)
                .lineLimit(1)
                .truncationMode(.middle)

            MacCodeTextView(text: $generatedCode)
                .focused($focusedField, equals: .editor)
                .background(Color(nsColor: .textBackgroundColor).opacity(0.35), in: RoundedRectangle(cornerRadius: 8))
                .clipShape(RoundedRectangle(cornerRadius: 8))
                .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color(nsColor: .separatorColor)))
        }
        .padding(12)
    }

    private var previewColumn: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(settings.localized("vibe_coder_preview"))
                .font(.headline)
            MacHTMLPreviewView(html: previewHTML)
                .clipShape(RoundedRectangle(cornerRadius: 8))
                .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color(nsColor: .separatorColor)))
        }
        .padding(12)
    }

    private var chatColumn: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 8) {
                Text(settings.localized("vibe_coder_ai_chat"))
                    .font(.headline)
                Spacer()

                ZStack {
                    Circle()
                        .stroke(Color.secondary.opacity(0.3), lineWidth: 2)
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
                .frame(width: 26, height: 26)
                .accessibilityLabel("Context usage \(contextUsageLabel)")

                Button {
                    clearActiveChat()
                } label: {
                    Image(systemName: "trash")
                }
                .help(settings.localized("vibe_coder_clear_chat"))
                .disabled(activeMessages.isEmpty || isGenerating)

                Button {
                    createNewChatSession()
                } label: {
                    Image(systemName: "plus")
                }
                .help(settings.localized("vibe_coder_new_chat"))
                .disabled(isGenerating)
            }
            .buttonStyle(.borderless)

            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 6) {
                    ForEach(chatSessions) { session in
                        MacVibeChip(
                            title: session.title,
                            isSelected: session.id == activeChatSessionId,
                            canDelete: chatSessions.count > 1 && !isGenerating,
                            deleteHelp: settings.localized("vibe_coder_delete_chat_cd"),
                            onSelect: { activeChatSessionId = session.id },
                            onDelete: { pendingDeleteChatId = session.id }
                        )
                    }
                }
            }

            ScrollViewReader { proxy in
                ScrollView {
                    VStack(alignment: .leading, spacing: 10) {
                        ForEach(activeMessages) { message in
                            VStack(alignment: .leading, spacing: 4) {
                                Text(message.role == "user" ? settings.localized("vibe_coder_message_you") : settings.localized("vibe_coder_message_ai"))
                                    .font(.caption.weight(.semibold))
                                    .foregroundStyle(.secondary)
                                if message.role == "user" {
                                    Text(message.text)
                                        .textSelection(.enabled)
                                        .frame(maxWidth: .infinity, alignment: .leading)
                                        .padding(8)
                                        .background(Color.accentColor.opacity(0.18), in: RoundedRectangle(cornerRadius: 8))
                                } else {
                                    ThinkingAwareResultContent(
                                        content: message.text,
                                        isGenerating: isGenerating && message.id == activeMessages.last?.id,
                                        preferThinkingWhileStreaming: preferThinkingWhileStreaming
                                    )
                                    .frame(maxWidth: .infinity, alignment: .leading)
                                    .padding(8)
                                    .background(Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: 8))
                                }
                            }
                            .id(message.id)
                        }
                    }
                    .padding(.vertical, 4)
                }
                .frame(maxHeight: .infinity)
                .onChange(of: activeMessages.count) { _, _ in
                    scrollToLast(proxy: proxy)
                }
                .onChange(of: streamTick) { _, _ in
                    scrollToLast(proxy: proxy)
                }
            }

            HStack(alignment: .bottom, spacing: 8) {
                TextField(
                    hasFileSession ? settings.localized("vibe_coder_ask_ai_edit") : settings.localized("vibe_coder_create_open_file_hint"),
                    text: $chatInput,
                    axis: .vertical
                )
                .lineLimit(1...5)
                .textFieldStyle(.roundedBorder)
                .focused($focusedField, equals: .chat)
                .disabled(!hasFileSession || isGenerating || isLoading)

                Button {
                    if isGenerating {
                        stopGeneration()
                    } else {
                        sendChat()
                    }
                } label: {
                    if isLoading {
                        ProgressView().controlSize(.small)
                    } else {
                        Image(systemName: sendButtonIconName)
                    }
                }
                .buttonStyle(.borderedProminent)
                .help(isGenerating ? settings.localized("vibe_coder_stop_generation") : "")
                .keyboardShortcut(.return, modifiers: .command)
                .disabled(isSendButtonDisabled)
            }
        }
        .padding(12)
    }

    // MARK: - Helpers

    private func scrollToLast(proxy: ScrollViewProxy) {
        guard let last = activeMessages.last else { return }
        withAnimation(.linear(duration: 0.08)) {
            proxy.scrollTo(last.id, anchor: .bottom)
        }
    }

    private func scheduleAutosave() {
        debouncedAutosaveTask?.cancel()
        debouncedAutosaveTask = Task {
            try? await Task.sleep(nanoseconds: 450_000_000)
            if Task.isCancelled { return }
            await MainActor.run {
                saveCurrentFile(silent: true)
            }
        }
    }

    private func schedulePreviewUpdate() {
        previewUpdateTask?.cancel()
        previewUpdateTask = Task {
            try? await Task.sleep(nanoseconds: 450_000_000)
            if Task.isCancelled { return }
            await MainActor.run {
                previewHTML = generatedCode
            }
        }
    }

    private func stopPreviewServer() {
        Task { await LocalHTMLPreviewServer.shared.stop() }
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

    // MARK: - Files

    private func openFile(_ url: URL, announceInChat: Bool = true) {
        do {
            let scopeURL = workspaceFolderURL ?? url
            let didStart = scopeURL.startAccessingSecurityScopedResource()
            defer {
                if didStart {
                    scopeURL.stopAccessingSecurityScopedResource()
                }
            }

            let text = try String(contentsOf: url, encoding: .utf8)
            currentFileURL = url
            currentFileName = url.lastPathComponent
            generatedCode = text
            previewHTML = text
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
            let didStart = folderURL.startAccessingSecurityScopedResource()
            defer {
                if didStart {
                    folderURL.stopAccessingSecurityScopedResource()
                }
            }

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
            let didStart = folderURL.startAccessingSecurityScopedResource()
            defer {
                if didStart {
                    folderURL.stopAccessingSecurityScopedResource()
                }
            }

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

    /// Same flow as iOS. On macOS (sandboxed) the bookmark must carry a
    /// security scope to survive relaunches; a bookmark saved without one is
    /// still accepted as a fallback.
    private func restoreWorkspaceFolderFromBookmark() {
        guard !folderBookmark.isEmpty else { return }
        var stale = false
        let resolved = (try? URL(resolvingBookmarkData: folderBookmark, options: [.withSecurityScope, .withoutUI], relativeTo: nil, bookmarkDataIsStale: &stale))
            ?? (try? URL(resolvingBookmarkData: folderBookmark, options: [.withoutUI], relativeTo: nil, bookmarkDataIsStale: &stale))
        guard let url = resolved else {
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
            let didStart = url.startAccessingSecurityScopedResource()
            defer {
                if didStart {
                    url.stopAccessingSecurityScopedResource()
                }
            }

            let bookmark = try url.bookmarkData(options: [.withSecurityScope], includingResourceValuesForKeys: nil, relativeTo: nil)
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
                NSWorkspace.shared.open(url)
            } catch {
                errorMessage = error.localizedDescription
            }
        }
    }

    // MARK: - Model

    private func ensureModelLoaded(force: Bool) async {
        guard let model = selectedFeatureModel(named: selectedModelName) else {
            errorMessage = settings.localized("vibe_coder_no_model")
            return
        }
        isLoading = true
        defer { isLoading = false }

        let modelContextCap = contextLimitForFeatureModel(model)
        let effectiveContext = min(max(1, Int(maxTokens)), modelContextCap)
        let shouldReload = force
            || llm.currentlyLoadedModel != model.name
            || llm.loadedContextWindow != effectiveContext

        llm.maxTokens = min(Int(maxTokens), effectiveContext)
        llm.contextWindow = effectiveContext
        llm.enableVision = false
        llm.enableAudio = false
        llm.enableThinking = effectiveEnableThinking

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
        focusedField = nil

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
                let filePrompt = buildFileAwareEditPrompt(trimmedPrompt)

                let savedThinking = llm.enableThinking
                llm.enableThinking = effectiveEnableThinking
                defer { llm.enableThinking = savedThinking }

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
    /// Verbatim copy of the iOS implementation.
    private func buildVibeCoderMultiTurnPrompt(currentFilePrompt: String, sessionId: UUID) -> String {
        let modelName = selectedModelName.lowercased()
        let modelSupportsThinking = selectedFeatureModel(named: selectedModelName)?.supportsThinking == true
        let isGemma      = modelName.contains("gemma")
        let isGemma4     = isGemma && (modelName.contains("gemma 4") || modelName.contains("gemma-4")) && !modelName.contains("translate")
        let isLlama      = modelName.contains("llama") || modelName.contains("mistral")
        let isLlama3     = isLlama && (modelName.contains("llama-3") || modelName.contains("llama 3") || modelName.contains("llama-3."))
        let isHarmonyModel = modelName.contains("gpt-oss") || modelName.contains("gpt_oss")
        let isMuseGlimmer = selectedFeatureModel(named: selectedModelName)?.chatTemplateFamily == .museGlimmer
        let isGranite42  = modelName.contains("granite-4.2") || modelName.contains("granite 4.2")
        let isGranite    = modelName.contains("granite") && !isGranite42
        let isPhi4       = modelName.contains("phi-4") || modelName.contains("phi 4") || modelName.contains("phi4")
        let isChatML     = modelName.contains("lfm") || modelName.contains("liquid")

        var allMessages: [VibeChatMessage] = []
        if let idx = chatSessions.firstIndex(where: { $0.id == sessionId }) {
            allMessages = chatSessions[idx].messages
        }

        // Drop the last 2 (new user + empty assistant placeholder).
        let history = allMessages.count >= 2 ? Array(allMessages.dropLast(2)) : []

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

        var parts: [String] = ["__RAW_PROMPT__"]

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

        if isMuseGlimmer {
            let dateFormatter = DateFormatter()
            dateFormatter.calendar = Calendar(identifier: .gregorian)
            dateFormatter.locale = Locale(identifier: "en_US_POSIX")
            dateFormatter.dateFormat = "yyyy-MM-dd"

            parts.append("<|begin_of_text|>")
            let validRecipients = llm.enableThinking ? "\"self\", \"user\"" : "\"user\""
            parts.append("<|start|>system<|message|>\(systemPrompt)\nKnowledge cutoff: 2026-01-04.\nCurrent date: \(dateFormatter.string(from: Date())).\n\nReasoning strength: high.\n\n# Valid recipients: \(validRecipients).<|eot|>")

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

                if msg.role == "user" {
                    parts.append("<|start|>user<|message|>\(content)<|eot|>")
                } else {
                    parts.append("<|start|>assistant to=user<|message|>\(content)<|eot|>")
                }
            }

            parts.append("<|start|>user<|message|>\(currentFilePrompt)<|eot|>")
            parts.append(llm.enableThinking
                ? "<|start|>assistant"
                : "<|start|>assistant to=user<|message|>")
            return parts.joined()
        }

        if isPhi4 {
            parts.append("<|system|>\n\(systemPrompt)<|end|>")
        } else if isGemma4 {
            parts.append("<|turn>system\n\(systemPrompt)<turn|>")
        } else if isGemma {
            parts.append("<start_of_turn>system\n\(systemPrompt)<end_of_turn>")
        } else if isLlama3 {
            parts.append("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n\(systemPrompt)<|eot_id|>")
        } else if isLlama {
            parts.append("<<SYS>>\n\(systemPrompt)\n<</SYS>>")
        } else if isGranite42 {
            parts.append("<|im_start|>system\n\(systemPrompt)<|im_end|>")
        } else if isGranite {
            parts.append("<|start_of_role|>system<|end_of_role|>\(systemPrompt)<|end_of_text|>")
        } else if isChatML {
            parts.append("<|startoftext|><|im_start|>system\n\(systemPrompt)<|im_end|>")
        } else {
            parts.append("System: \(systemPrompt)")
        }

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

            if isPhi4 {
                let role = (msg.role == "user") ? "user" : "assistant"
                parts.append("<|\(role)|>\n\(content)<|end|>")
            } else if isGemma4 {
                let role = (msg.role == "user") ? "user" : "model"
                parts.append("<|turn>\(role)\n\(content)<turn|>")
            } else if isGemma {
                let role = (msg.role == "user") ? "user" : "model"
                parts.append("<start_of_turn>\(role)\n\(content)<end_of_turn>")
            } else if isLlama3 {
                let role = (msg.role == "user") ? "user" : "assistant"
                parts.append("<|start_header_id|>\(role)<|end_header_id|>\n\n\(content)<|eot_id|>")
            } else if isLlama {
                if msg.role == "user" {
                    parts.append("[INST] \(content) [/INST]")
                } else {
                    parts.append(content)
                }
            } else if isGranite42 {
                let role = (msg.role == "user") ? "user" : "assistant"
                let prefix = (role == "assistant") ? "<think></think>" : ""
                parts.append("<|im_start|>\(role)\n\(prefix)\(content)<|im_end|>")
            } else if isGranite {
                let role = (msg.role == "user") ? "user" : "assistant"
                parts.append("<|start_of_role|>\(role)<|end_of_role|>\(content)<|end_of_text|>")
            } else if isChatML {
                let role = (msg.role == "user") ? "user" : "assistant"
                parts.append("<|im_start|>\(role)\n\(content)<|im_end|>")
            } else {
                let prefix = (msg.role == "user") ? "User" : "Assistant"
                parts.append("\(prefix): \(content)")
            }
        }

        if isPhi4 {
            parts.append("<|user|>\n\(currentFilePrompt)<|end|>")
            parts.append("<|assistant|>\n")
        } else if isGemma4 {
            parts.append("<|turn>user\n\(currentFilePrompt)<turn|>")
            parts.append("<|turn>model\n")
        } else if isGemma {
            parts.append("<start_of_turn>user\n\(currentFilePrompt)<end_of_turn>")
            parts.append("<start_of_turn>model\n")
        } else if isLlama3 {
            parts.append("<|start_header_id|>user<|end_header_id|>\n\n\(currentFilePrompt)<|eot_id|>")
            parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
        } else if isLlama {
            parts.append("[INST] \(currentFilePrompt) [/INST]")
        } else if isGranite42 {
            parts.append("<|im_start|>user\n\(currentFilePrompt)<|im_end|>")
            parts.append("<|im_start|>assistant\n<think></think>")
        } else if isGranite {
            parts.append("<|start_of_role|>user<|end_of_role|>\(currentFilePrompt)<|end_of_text|>")
            parts.append("<|start_of_role|>assistant<|end_of_role|>")
        } else if isChatML {
            parts.append("<|im_start|>user\n\(currentFilePrompt)<|im_end|>")
            parts.append("<|im_start|>assistant\n")
        } else {
            parts.append("User: \(currentFilePrompt)")
            parts.append("Assistant:")
        }

        return parts.joined(separator: "\n")
    }

    // MARK: - Chat sessions

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
            let didStart = folderURL.startAccessingSecurityScopedResource()
            defer {
                if didStart {
                    folderURL.stopAccessingSecurityScopedResource()
                }
            }

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
            let didStart = folderURL.startAccessingSecurityScopedResource()
            defer {
                if didStart {
                    folderURL.stopAccessingSecurityScopedResource()
                }
            }

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

// MARK: - Chat session chip

private struct MacVibeChip: View {
    let title: String
    let isSelected: Bool
    let canDelete: Bool
    let deleteHelp: String
    let onSelect: () -> Void
    let onDelete: () -> Void

    var body: some View {
        HStack(spacing: 2) {
            Button(action: onSelect) {
                Text(title)
                    .lineLimit(1)
            }
            .buttonStyle(.borderless)
            .foregroundStyle(isSelected ? Color.primary : Color.secondary)

            Button(action: onDelete) {
                Image(systemName: "xmark")
                    .font(.system(size: 9, weight: .bold))
            }
            .buttonStyle(.borderless)
            .help(deleteHelp)
            .disabled(!canDelete)
        }
        .font(.callout.weight(.medium))
        .padding(.horizontal, 10)
        .padding(.vertical, 4)
        .background(
            Capsule(style: .continuous)
                .fill(isSelected ? Color.accentColor.opacity(0.28) : Color(nsColor: .controlBackgroundColor))
        )
        .overlay(
            Capsule(style: .continuous)
                .stroke(Color(nsColor: .separatorColor), lineWidth: 1)
        )
    }
}

// MARK: - Code editor (NSTextView with smart substitutions disabled)

private struct MacCodeTextView: NSViewRepresentable {
    @Binding var text: String

    func makeCoordinator() -> Coordinator { Coordinator(text: $text) }

    func makeNSView(context: Context) -> NSScrollView {
        let scrollView = NSTextView.scrollableTextView()
        scrollView.drawsBackground = false
        scrollView.hasVerticalScroller = true
        scrollView.hasHorizontalScroller = true
        scrollView.autohidesScrollers = true

        guard let textView = scrollView.documentView as? NSTextView else { return scrollView }
        textView.delegate = context.coordinator
        textView.isRichText = false
        textView.importsGraphics = false
        textView.allowsUndo = true
        textView.drawsBackground = false
        textView.font = .monospacedSystemFont(ofSize: NSFont.systemFontSize, weight: .regular)
        textView.textColor = .textColor
        textView.insertionPointColor = .textColor
        textView.isAutomaticQuoteSubstitutionEnabled = false
        textView.isAutomaticDashSubstitutionEnabled = false
        textView.isAutomaticTextReplacementEnabled = false
        textView.isAutomaticSpellingCorrectionEnabled = false
        textView.isContinuousSpellCheckingEnabled = false
        textView.isGrammarCheckingEnabled = false
        textView.isAutomaticLinkDetectionEnabled = false
        textView.smartInsertDeleteEnabled = false
        textView.textContainerInset = NSSize(width: 6, height: 8)
        // No line wrapping, like a code editor.
        textView.isHorizontallyResizable = true
        textView.maxSize = NSSize(width: CGFloat.greatestFiniteMagnitude, height: CGFloat.greatestFiniteMagnitude)
        textView.textContainer?.widthTracksTextView = false
        textView.textContainer?.containerSize = NSSize(width: CGFloat.greatestFiniteMagnitude, height: CGFloat.greatestFiniteMagnitude)
        textView.string = text
        return scrollView
    }

    func updateNSView(_ scrollView: NSScrollView, context: Context) {
        context.coordinator.text = $text
        guard let textView = scrollView.documentView as? NSTextView else { return }
        if textView.string != text {
            let selection = textView.selectedRanges
            textView.string = text
            let length = (text as NSString).length
            textView.selectedRanges = selection.map { value in
                let range = value.rangeValue
                let location = min(range.location, length)
                let len = min(range.length, length - location)
                return NSValue(range: NSRange(location: location, length: len))
            }
        }
    }

    final class Coordinator: NSObject, NSTextViewDelegate {
        var text: Binding<String>

        init(text: Binding<String>) {
            self.text = text
        }

        func textDidChange(_ notification: Notification) {
            guard let textView = notification.object as? NSTextView else { return }
            text.wrappedValue = textView.string
        }
    }
}

// MARK: - Live HTML preview (WKWebView)

private struct MacHTMLPreviewView: NSViewRepresentable {
    let html: String

    func makeCoordinator() -> Coordinator { Coordinator() }

    func makeNSView(context: Context) -> WKWebView {
        let webView = WKWebView(frame: .zero, configuration: WKWebViewConfiguration())
        webView.setValue(false, forKey: "drawsBackground")
        context.coordinator.lastHTML = html
        webView.loadHTMLString(html, baseURL: nil)
        return webView
    }

    func updateNSView(_ webView: WKWebView, context: Context) {
        guard context.coordinator.lastHTML != html else { return }
        context.coordinator.lastHTML = html
        webView.loadHTMLString(html, baseURL: nil)
    }

    final class Coordinator {
        var lastHTML: String?
    }
}
#endif
