//
//  MacChatView.swift
//  LLMHub
//
//  Native macOS chat. Drives the same `ChatViewModel` as the iOS ChatScreen
//  (sending, web search, attachments, mic, TTS, edit/regenerate, context
//  usage) with a desktop layout and existing localized strings.
//

#if os(macOS)
import PhotosUI
import SwiftUI
import UniformTypeIdentifiers

struct MacChatView: View {
    @EnvironmentObject var settings: AppSettings
    @ObservedObject var vm: ChatViewModel
    let onNavigateToModels: () -> Void

    @ObservedObject private var ttsManager = OnDeviceTtsManager.shared
    @StateObject private var micTranscriber = ChatMicTranscriber()
    @StateObject private var audioRecorder = AudioRecorder()

    @State private var showInspector = false
    @State private var copiedMessageId: UUID?
    @State private var selectedImageItem: PhotosPickerItem?
    @State private var showPhotosPicker = false
    @State private var attachedImageURL: URL?
    @State private var attachedAudioURL: URL?
    @State private var attachedDocumentURL: URL?
    @State private var attachedDocumentName: String?
    @State private var previewImagePath: String?
    @State private var showDocumentImporter = false
    @State private var showAudioImporter = false
    @State private var userHasScrolledUp = false
    @State private var hasDownloadedModels = true
    @FocusState private var isComposerFocused: Bool

    var body: some View {
        VStack(spacing: 0) {
            messageList
            Divider()
            composer
        }
        .navigationTitle(settings.localized("feature_ai_chat"))
        .navigationSubtitle(vm.selectedModelName)
        .toolbar { toolbarContent }
        .inspector(isPresented: $showInspector) {
            MacChatSettingsInspector(vm: vm)
                .inspectorColumnWidth(min: 300, ideal: 340, max: 440)
        }
        .sheet(isPresented: Binding(
            get: { previewImagePath != nil },
            set: { if !$0 { previewImagePath = nil } }
        )) {
            MacChatImagePreview(path: previewImagePath)
        }
        .fileImporter(
            isPresented: $showAudioImporter,
            allowedContentTypes: [.audio, .mpeg4Audio, .mp3],
            allowsMultipleSelection: false
        ) { result in
            guard case .success(let urls) = result, let sourceURL = urls.first else { return }
            importAudio(from: sourceURL)
        }
        .background(
            EmptyView().fileImporter(
                isPresented: $showDocumentImporter,
                allowedContentTypes: DocumentTextExtractor.supportedTypes,
                allowsMultipleSelection: false
            ) { result in
                guard case .success(let urls) = result, let sourceURL = urls.first else { return }
                attachedDocumentURL = sourceURL
                attachedDocumentName = sourceURL.lastPathComponent
                Task {
                    let didAccess = sourceURL.startAccessingSecurityScopedResource()
                    defer { if didAccess { sourceURL.stopAccessingSecurityScopedResource() } }
                    if let text = try? DocumentTextExtractor.extract(from: sourceURL) {
                        await MainActor.run { vm.pendingAttachedDocumentChars = min(text.count, 8000) }
                    }
                }
            }
        )
        .photosPicker(isPresented: $showPhotosPicker, selection: $selectedImageItem, matching: .images)
        .onChange(of: selectedImageItem) { _, item in
            loadPickedImage(item)
        }
        .onChange(of: vm.enableVision) { _, enabled in
            if !enabled { attachedImageURL = nil; selectedImageItem = nil }
        }
        .onChange(of: vm.enableAudio) { _, enabled in
            if !enabled { attachedAudioURL = nil }
        }
        .onChange(of: vm.selectedModelName) { _, _ in
            let model = ModelData.allModels().first(where: { $0.name == vm.selectedModelName })
            if !((model?.supportsVision == true) && vm.enableVision) {
                attachedImageURL = nil
                selectedImageItem = nil
            }
            if !((model?.supportsAudio == true) && vm.enableAudio) {
                attachedAudioURL = nil
            }
        }
        .onChange(of: micTranscriber.liveText) { _, newText in
            guard !shouldUseModelAudioInput, !newText.isEmpty else { return }
            vm.inputText = newText
        }
        .onChange(of: userHasScrolledUp) { _, scrolledUp in
            vm.viewIsScrolledUp = scrolledUp
            if !scrolledUp { vm.flushThrottledStreamUpdate() }
        }
        .task {
            hasDownloadedModels = !chatDownloadedModels().isEmpty
        }
        .onAppear {
            // Same entry behavior as the iOS chat screen.
            vm.unloadModel()
            Task {
                await RagServiceManager.shared.initialize(modelId: AppSettings.shared.selectedEmbeddingModelId)
            }
        }
        .onDisappear {
            vm.stopAutoReadout()
            vm.unloadModel()
        }
    }

    // MARK: - Toolbar

    @ToolbarContentBuilder
    private var toolbarContent: some ToolbarContent {
        if vm.isBackendLoading {
            ToolbarItem(placement: .navigation) {
                ProgressView().controlSize(.small)
            }
        }
        ToolbarItem(placement: .primaryAction) {
            Button {
                vm.newChat()
            } label: {
                Label(settings.localized("drawer_new_chat"), systemImage: "square.and.pencil")
            }
            .keyboardShortcut("n", modifiers: .command)
        }
        ToolbarItem(placement: .primaryAction) {
            Button {
                showInspector.toggle()
            } label: {
                Label(settings.localized("feature_settings_title"), systemImage: "slider.horizontal.3")
            }
        }
    }

    // MARK: - Messages

    private var messageList: some View {
        ScrollViewReader { proxy in
            ScrollView {
                VStack(spacing: 14) {
                    if vm.messages.isEmpty {
                        emptyState
                    } else {
                        ForEach(vm.messages) { message in
                            messageRow(for: message)
                        }
                        Color.clear.frame(height: 1).id("mac_chat_bottom")
                    }
                }
                .frame(maxWidth: 820)
                .padding(.horizontal, 24)
                .padding(.vertical, 18)
                .frame(maxWidth: .infinity)
            }
            .modifier(MacScrolledUpTracker(isScrolledUp: $userHasScrolledUp))
            .overlay(alignment: .bottomTrailing) {
                if userHasScrolledUp, !vm.messages.isEmpty {
                    Button {
                        userHasScrolledUp = false
                        withAnimation { proxy.scrollTo("mac_chat_bottom", anchor: .bottom) }
                    } label: {
                        Image(systemName: "chevron.down")
                            .font(.system(size: 14, weight: .semibold))
                            .frame(width: 32, height: 32)
                            .background(.regularMaterial, in: Circle())
                    }
                    .buttonStyle(.plain)
                    .padding(16)
                }
            }
            .overlay(alignment: .bottom) {
                if copiedMessageId != nil {
                    Text(settings.localized("message_copied"))
                        .font(.caption)
                        .padding(.horizontal, 12)
                        .padding(.vertical, 6)
                        .background(.regularMaterial, in: Capsule())
                        .padding(.bottom, 12)
                        .transition(.opacity)
                }
            }
            .onChange(of: vm.messages.count) { _, _ in
                guard !userHasScrolledUp else { return }
                withAnimation { proxy.scrollTo("mac_chat_bottom", anchor: .bottom) }
            }
            .onChange(of: vm.messages.last?.content) { _, _ in
                guard vm.isGenerating, !userHasScrolledUp else { return }
                proxy.scrollTo("mac_chat_bottom", anchor: .bottom)
            }
            .onChange(of: vm.currentSessionId) { _, _ in
                userHasScrolledUp = false
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.05) {
                    proxy.scrollTo("mac_chat_bottom", anchor: .bottom)
                }
            }
        }
    }

    private var emptyState: some View {
        VStack(spacing: 14) {
            Spacer(minLength: 80)
            Text(settings.localized("welcome_to_llm_hub"))
                .font(.title2.bold())
            if !hasDownloadedModels {
                Text(settings.localized("no_models_downloaded"))
                    .foregroundStyle(.secondary)
                Button {
                    onNavigateToModels()
                } label: {
                    Label(settings.localized("download_a_model"), systemImage: "arrow.down.circle")
                }
                .buttonStyle(.borderedProminent)
            } else if vm.selectedModelName == settings.localized("no_model_selected") {
                Text(settings.localized("load_model_to_start"))
                    .foregroundStyle(.secondary)
            }
        }
        .frame(maxWidth: .infinity)
    }

    @ViewBuilder
    private func messageRow(for message: ChatMessage) -> some View {
        let isLatestAssistant = message.id == vm.latestAssistantMessageId
        let canRegenerate = isLatestAssistant && !vm.isGenerating && !message.isGenerating
        let canEditUser = message.isFromUser && message.id == vm.latestUserMessageId && !vm.isGenerating
        let canEditAssistant = !message.isFromUser && !vm.isGenerating && !message.isGenerating
        let modelSupportsThinking = chatModel(named: vm.selectedModelName)?.supportsThinking == true
        let useHeuristic = supportsUnmarkedStreamingThinkingHeuristic(forModelNamed: vm.selectedModelName)
        let hasSpeakableContent = !message.isFromUser && !message.content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty

        MacChatMessageRow(
            message: message,
            preferThinkingWhileStreaming: modelSupportsThinking && vm.enableThinking && useHeuristic,
            onCopy: {
                vm.copyMessage(message)
                copiedMessageId = message.id
                DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
                    if copiedMessageId == message.id { copiedMessageId = nil }
                }
            },
            onOpenImage: { previewImagePath = $0 },
            onEditUserMessage: canEditUser ? { vm.editUserPrompt(message.id, newText: $0) } : nil,
            onEditAssistantMessage: canEditAssistant ? { vm.editAssistantMessage(message.id, newText: $0) } : nil,
            onRegenerateResponse: canRegenerate ? {
                userHasScrolledUp = false
                vm.regenerateResponse(for: message.id)
            } : nil,
            onToggleTts: hasSpeakableContent ? {
                let isThinkingActive = modelSupportsThinking && vm.enableThinking && useHeuristic
                let contentToSpeak: String
                if contentHasThinkingMarkers(message.content) {
                    contentToSpeak = getDisplayContentWithoutThinking(message.content)
                } else if message.isGenerating && isThinkingActive {
                    contentToSpeak = ""
                } else {
                    contentToSpeak = message.content
                }
                if !contentToSpeak.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    ttsManager.toggleSpeaking(contentToSpeak, fallbackLanguage: settings.selectedLanguage, key: message.id.uuidString)
                }
            } : nil,
            isTtsSpeaking: ttsManager.isSpeaking(key: message.id.uuidString)
        )
        .id(message.id)
    }

    // MARK: - Composer

    private var hasAttachments: Bool {
        attachedImageURL != nil || attachedAudioURL != nil || attachedDocumentURL != nil
    }

    private var canSend: Bool {
        !vm.inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || hasAttachments
    }

    private var canAttachImages: Bool {
        let model = chatModel(named: vm.selectedModelName)
        return vm.enableVision
            && (model?.supportsVision == true)
            && (model.map { LLMBackend.shared.isVisionProjectorAvailable(for: $0) } ?? false)
    }

    private var composer: some View {
        VStack(alignment: .leading, spacing: 8) {
            if hasAttachments {
                HStack(spacing: 8) {
                    if attachedImageURL != nil {
                        attachmentChip(settings.localized("vision"), icon: "photo") {
                            attachedImageURL = nil
                            selectedImageItem = nil
                        }
                    }
                    if attachedAudioURL != nil {
                        attachmentChip(settings.localized("audio"), icon: "waveform") {
                            attachedAudioURL = nil
                        }
                    }
                    if let docName = attachedDocumentName {
                        attachmentChip(docName, icon: "doc.text") {
                            attachedDocumentURL = nil
                            attachedDocumentName = nil
                        }
                    }
                }
            }

            HStack(alignment: .bottom, spacing: 10) {
                Menu {
                    if canAttachImages {
                        Button(settings.localized("images")) { showPhotosPicker = true }
                    }
                    Button(settings.localized("documents")) { showDocumentImporter = true }
                    Button(settings.localized("audio_file")) { showAudioImporter = true }
                } label: {
                    Image(systemName: "plus")
                }
                .menuStyle(.button)
                .menuIndicator(.hidden)
                .buttonStyle(.bordered)
                .fixedSize()
                .disabled(vm.isGenerating)

                Toggle(isOn: $vm.isWebSearchEnabled) {
                    Label(settings.localized("web_search"), systemImage: vm.isSearching ? "arrow.triangle.2.circlepath" : "globe")
                }
                .toggleStyle(.button)
                .disabled(vm.isGenerating)

                TextField(
                    micTranscriber.isPreparing && !shouldUseModelAudioInput
                        ? settings.localized("preparing_mic")
                        : settings.localized("type_a_message"),
                    text: $vm.inputText,
                    axis: .vertical
                )
                .textFieldStyle(.plain)
                .lineLimit(1...8)
                .focused($isComposerFocused)
                .padding(.horizontal, 10)
                .padding(.vertical, 7)
                .background(Color(nsColor: .textBackgroundColor).opacity(0.35), in: RoundedRectangle(cornerRadius: 8))
                .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color(nsColor: .separatorColor)))
                .onSubmit { submit() }

                contextUsageRing

                Button {
                    toggleMic()
                } label: {
                    Image(systemName: (micTranscriber.isPreparing || audioRecorder.isPreparing) ? "ellipsis"
                          : (micTranscriber.isRecording || audioRecorder.isRecording) ? "stop.fill" : "mic.fill")
                        .foregroundStyle((micTranscriber.isRecording || audioRecorder.isRecording) ? Color.red : Color.primary)
                }
                .buttonStyle(.bordered)
                .disabled(vm.isGenerating)

                Button {
                    if micTranscriber.isRecording {
                        Task { _ = await micTranscriber.stopLive() }
                    }
                    if vm.isGenerating {
                        vm.stopGeneration()
                    } else {
                        submit()
                    }
                } label: {
                    Image(systemName: vm.isGenerating ? "stop.fill" : "arrow.up")
                }
                .buttonStyle(.borderedProminent)
                .tint(vm.isGenerating ? .red : ApolloPalette.accentStrong)
                .disabled(!vm.isGenerating && !canSend)
            }
        }
        .frame(maxWidth: 860)
        .padding(.horizontal, 20)
        .padding(.vertical, 12)
        .frame(maxWidth: .infinity)
    }

    private var contextUsageRing: some View {
        ZStack {
            Circle()
                .stroke(Color.secondary.opacity(0.35), lineWidth: 1.5)
            Circle()
                .trim(from: 0, to: vm.contextUsageFractionDisplay)
                .stroke(
                    vm.contextUsageFractionRaw < 0.90 ? ApolloPalette.accentStrong : ApolloPalette.warning,
                    style: StrokeStyle(lineWidth: 2, lineCap: .round)
                )
                .rotationEffect(.degrees(-90))
            Text(vm.contextUsageFractionRaw < 0.995 ? vm.contextUsageLabel : "!")
                .font(.system(size: 8, weight: .bold, design: .rounded))
        }
        .frame(width: 28, height: 28)
    }

    private func attachmentChip(_ label: String, icon: String, onRemove: @escaping () -> Void) -> some View {
        HStack(spacing: 6) {
            Image(systemName: icon)
            Text(label).lineLimit(1)
            Button(action: onRemove) {
                Image(systemName: "xmark.circle.fill")
            }
            .buttonStyle(.plain)
        }
        .font(.caption)
        .padding(.horizontal, 10)
        .padding(.vertical, 5)
        .background(.quaternary, in: Capsule())
    }

    // MARK: - Actions

    private func submit() {
        guard !vm.isGenerating, canSend else { return }
        if vm.isWebSearchEnabled {
            if vm.sendMessageWithWebSearch(documentURL: attachedDocumentURL, documentName: attachedDocumentName) {
                clearAttachments()
            }
        } else if vm.sendMessage(imageURL: attachedImageURL, audioURL: attachedAudioURL, documentURL: attachedDocumentURL, documentName: attachedDocumentName) {
            clearAttachments()
        }
    }

    private func clearAttachments() {
        attachedImageURL = nil
        attachedAudioURL = nil
        attachedDocumentURL = nil
        attachedDocumentName = nil
        selectedImageItem = nil
        vm.pendingAttachedDocumentChars = 0
    }

    private var shouldUseModelAudioInput: Bool {
        guard let model = chatModel(named: vm.selectedModelName) else { return false }
        guard model.isGemma4LiteRTLM, vm.enableAudio else { return false }
        return vm.loadedModelName == vm.selectedModelName
    }

    private func toggleMic() {
        if shouldUseModelAudioInput {
            if audioRecorder.isRecording {
                if let url = audioRecorder.stopRecording() {
                    attachedAudioURL = url
                }
            } else {
                Task { @MainActor in
                    let destination = persistentAttachmentDirectoryURL()
                        .appendingPathComponent("audio_\(UUID().uuidString)")
                        .appendingPathExtension("wav")
                    _ = await audioRecorder.startRecording(outputURL: destination, autoStopAfterSilence: false, isFloat32Wav: true) { url in
                        Task { @MainActor in attachedAudioURL = url }
                    }
                }
            }
        } else if micTranscriber.isRecording {
            Task { _ = await micTranscriber.stopLive() }
        } else {
            Task { await micTranscriber.startLive() }
        }
    }

    private func importAudio(from sourceURL: URL) {
        Task { @MainActor in
            let didAccess = sourceURL.startAccessingSecurityScopedResource()
            defer { if didAccess { sourceURL.stopAccessingSecurityScopedResource() } }
            if shouldUseModelAudioInput {
                if let convertedURL = prepareGemmaAudioInput(
                    from: sourceURL,
                    destinationDirectory: persistentAttachmentDirectoryURL(),
                    filePrefix: "chat_audio"
                ) {
                    attachedAudioURL = convertedURL
                }
            } else {
                let speechURL = prepareGemmaAudioInput(
                    from: sourceURL,
                    destinationDirectory: FileManager.default.temporaryDirectory,
                    filePrefix: "chat_speech"
                ) ?? sourceURL
                let transcript = await micTranscriber.transcribeFile(speechURL)
                if !transcript.isEmpty {
                    vm.inputText += (vm.inputText.isEmpty ? "" : " ") + transcript
                }
            }
        }
    }

    private func loadPickedImage(_ item: PhotosPickerItem?) {
        guard let item else {
            attachedImageURL = nil
            return
        }
        Task {
            if let sourceURL = try? await item.loadTransferable(type: URL.self),
               let copied = copyAttachment(sourceURL) {
                await MainActor.run { attachedImageURL = copied }
                return
            }
            if let data = try? await item.loadTransferable(type: Data.self) {
                let ext = item.supportedContentTypes.compactMap { $0.preferredFilenameExtension }.first ?? "bin"
                let url = persistentAttachmentDirectoryURL()
                    .appendingPathComponent(UUID().uuidString)
                    .appendingPathExtension(ext)
                if (try? data.write(to: url, options: .atomic)) != nil {
                    await MainActor.run { attachedImageURL = url }
                }
            }
        }
    }

    private func copyAttachment(_ sourceURL: URL) -> URL? {
        let ext = sourceURL.pathExtension
        let destination = persistentAttachmentDirectoryURL()
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension(ext)
        let didAccess = sourceURL.startAccessingSecurityScopedResource()
        defer { if didAccess { sourceURL.stopAccessingSecurityScopedResource() } }
        do {
            try FileManager.default.copyItem(at: sourceURL, to: destination)
            return destination
        } catch {
            return nil
        }
    }

    /// Same model availability rule as the iOS chat empty state; evaluated once
    /// per appearance instead of on every render (it touches the file system).
    private func chatDownloadedModels() -> [AIModel] {
        var models = ModelData.allModels().filter { model in
            if model.isDependencyOnly { return false }
            if model.name.hasPrefix("Translate Gemma") { return false }
            if !model.isLanguageModel { return false }
            return ModelData.isModelFullyAvailableLocally(model)
        }
        if let appleModel = chatAppleFoundationModelIfAvailable(),
           !models.contains(where: { $0.id == appleModel.id }) {
            models.append(appleModel)
        }
        return models
    }
}

// MARK: - Message row

/// Desktop rendering of a chat message with the same actions as iOS
/// `MessageBubble`: copy, read aloud, edit, regenerate, token stats, time.
struct MacChatMessageRow: View {
    @EnvironmentObject var settings: AppSettings
    let message: ChatMessage
    let preferThinkingWhileStreaming: Bool
    let onCopy: () -> Void
    let onOpenImage: (String) -> Void
    let onEditUserMessage: ((String) -> Void)?
    let onEditAssistantMessage: ((String) -> Void)?
    let onRegenerateResponse: (() -> Void)?
    let onToggleTts: (() -> Void)?
    let isTtsSpeaking: Bool

    @State private var isEditing = false
    @State private var editedText = ""
    @State private var isHovering = false

    private var hasContent: Bool {
        !message.content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    var body: some View {
        VStack(alignment: message.isFromUser ? .trailing : .leading, spacing: 6) {
            if isEditing {
                editor
            } else if message.isFromUser {
                userContent
            } else if message.isGenerating && !hasContent {
                TypingIndicator().padding(.vertical, 6)
            } else {
                ThinkingAwareResultContent(
                    content: message.content,
                    isGenerating: message.isGenerating,
                    preferThinkingWhileStreaming: preferThinkingWhileStreaming,
                    useChatRenderer: true
                )
                .frame(maxWidth: .infinity, alignment: .leading)
            }

            if !isEditing && (hasContent || message.attachmentImagePath != nil
                              || message.attachmentAudioPath != nil || message.attachmentDocumentName != nil) {
                actionRow
            }
        }
        .frame(maxWidth: .infinity, alignment: message.isFromUser ? .trailing : .leading)
        .onHover { isHovering = $0 }
        .contextMenu {
            Button(settings.localized("copy_message"), action: onCopy)
        }
    }

    private var userContent: some View {
        VStack(alignment: .trailing, spacing: 8) {
            if let imagePath = message.attachmentImagePath,
               let url = resolveStoredAttachmentURL(imagePath),
               let image = NSImage(contentsOf: url) {
                Image(nsImage: image)
                    .resizable()
                    .scaledToFit()
                    .frame(maxWidth: 260, maxHeight: 260)
                    .clipShape(RoundedRectangle(cornerRadius: 10))
                    .onTapGesture { onOpenImage(imagePath) }
            }
            if let audioPath = message.attachmentAudioPath {
                HStack(spacing: 8) {
                    if let audioURL = resolveStoredAttachmentURL(audioPath) {
                        AudioPlaybackButton(url: audioURL)
                    }
                    Label(settings.localized("audio"), systemImage: "waveform")
                        .font(.caption)
                }
                .padding(.horizontal, 10)
                .padding(.vertical, 6)
                .background(.quaternary, in: RoundedRectangle(cornerRadius: 10))
            }
            if let docName = message.attachmentDocumentName {
                Label(docName, systemImage: "doc.text")
                    .font(.caption)
                    .lineLimit(1)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 6)
                    .background(.quaternary, in: RoundedRectangle(cornerRadius: 10))
            }
            if hasContent {
                Text(message.content)
                    .textSelection(.enabled)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                    .background(ApolloPalette.accentStrong.opacity(0.28), in: RoundedRectangle(cornerRadius: 14))
            }
        }
        .frame(maxWidth: 560, alignment: .trailing)
    }

    private var editor: some View {
        VStack(alignment: .trailing, spacing: 8) {
            TextEditor(text: $editedText)
                .font(.body)
                .scrollContentBackground(.hidden)
                .padding(6)
                .frame(minHeight: 90)
                .background(Color(nsColor: .textBackgroundColor).opacity(0.35), in: RoundedRectangle(cornerRadius: 8))
                .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color(nsColor: .separatorColor)))
            HStack(spacing: 8) {
                Button {
                    isEditing = false
                    editedText = ""
                } label: {
                    Image(systemName: "xmark")
                }
                .keyboardShortcut(.cancelAction)
                Button {
                    let trimmed = editedText.trimmingCharacters(in: .whitespacesAndNewlines)
                    guard !trimmed.isEmpty else { return }
                    if message.isFromUser {
                        onEditUserMessage?(trimmed)
                    } else {
                        onEditAssistantMessage?(trimmed)
                    }
                    isEditing = false
                    editedText = ""
                } label: {
                    Image(systemName: "checkmark")
                }
                .buttonStyle(.borderedProminent)
                .keyboardShortcut(.return, modifiers: .command)
                .disabled(editedText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
        }
        .frame(maxWidth: message.isFromUser ? 560 : .infinity)
    }

    private var actionRow: some View {
        HStack(spacing: 10) {
            Button(action: onCopy) { Image(systemName: "doc.on.doc") }

            if !message.isFromUser, let onToggleTts {
                Button(action: onToggleTts) {
                    Image(systemName: isTtsSpeaking ? "stop.fill" : "speaker.wave.2")
                }
            }

            if message.isFromUser ? (hasContent && onEditUserMessage != nil) : (onEditAssistantMessage != nil) {
                Button {
                    editedText = message.content
                    isEditing = true
                } label: {
                    Image(systemName: "pencil")
                }
            }

            if !message.isFromUser, let onRegenerateResponse {
                Button(action: onRegenerateResponse) { Image(systemName: "arrow.clockwise") }
            }

            if !message.isFromUser, let tokenCount = message.tokenCount, let tps = message.tokensPerSecond, tokenCount > 0 {
                let hasMarkers = contentHasThinkingMarkers(message.content)
                let answer = hasMarkers ? getDisplayContentWithoutThinking(message.content) : message.content
                if !hasMarkers || !answer.isEmpty {
                    let displayed: Int = {
                        if hasMarkers && !answer.isEmpty {
                            return max(1, min(tokenCount, Int(ceil(Double(answer.count) / 4.0))))
                        }
                        return tokenCount
                    }()
                    Label(String(format: settings.localized("tokens_per_second_format"), displayed, tps), systemImage: "bolt.fill")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            }

            Text(message.timestamp, style: .time)
                .font(.caption2)
                .foregroundStyle(.tertiary)
        }
        .buttonStyle(.borderless)
        .foregroundStyle(.secondary)
        .font(.callout)
        .opacity(isHovering || message.isGenerating ? 1 : 0.55)
    }
}

/// Image attachment viewer (sheet on macOS).
private struct MacChatImagePreview: View {
    let path: String?
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        Group {
            if let url = resolveStoredAttachmentURL(path), let image = NSImage(contentsOf: url) {
                Image(nsImage: image)
                    .resizable()
                    .scaledToFit()
                    .padding(20)
            } else {
                Text("Image unavailable")
                    .font(.headline)
            }
        }
        .frame(minWidth: 600, idealWidth: 900, minHeight: 450, idealHeight: 700)
        .background(Color.black)
        .onTapGesture { dismiss() }
        .onExitCommand { dismiss() }
    }
}

/// Tracks whether the user scrolled away from the bottom of the transcript
/// (used to pause auto-follow while streaming, as on iOS).
private struct MacScrolledUpTracker: ViewModifier {
    @Binding var isScrolledUp: Bool

    func body(content: Content) -> some View {
        if #available(macOS 15.0, *) {
            content.onScrollGeometryChange(for: Bool.self) { geometry in
                geometry.contentOffset.y + geometry.containerSize.height < geometry.contentSize.height - 60
            } action: { _, scrolledUp in
                if isScrolledUp != scrolledUp { isScrolledUp = scrolledUp }
            }
        } else {
            content
        }
    }
}
#endif
