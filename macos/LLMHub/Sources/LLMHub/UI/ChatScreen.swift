import SwiftUI
import AppKit

public struct ChatScreen: View {
    @EnvironmentObject var settings: AppSettings
    @ObservedObject private var backend = LLMBackend.shared
    @StateObject private var chatStore = ChatStore.shared

    @State private var inputText: String = ""
    @State private var isGenerating: Bool = false
    @State private var showSettings: Bool = false
    @State private var showModelSelector: Bool = false
    @State private var attachedImageURL: URL? = nil

    // Chat parameters
    @State private var temperature: Double = 0.7
    @State private var topP: Double = 0.9
    @State private var topK: Int = 40
    @State private var maxTokens: Int = 2048
    @State private var systemPrompt: String = "You are a helpful, respectful, and honest assistant."
    @State private var contextWindow: Int = 4096

    // Generation metrics
    @State private var generationSpeed: Double = 0.0
    @State private var generationStartTime: Date? = nil

    var onNavigateToModels: () -> Void

    public init(onNavigateToModels: @escaping () -> Void) {
        self.onNavigateToModels = onNavigateToModels
    }

    public var body: some View {
        HSplitView {
            // Left Pane: Conversation Sidebar
            conversationSidebar
                .frame(minWidth: 220, idealWidth: 260, maxWidth: 340)

            // Right Pane: Active Chat Area
            activeChatArea
                .frame(minWidth: 500)
        }
        .apolloScreenBackground()
        .sheet(isPresented: $showSettings) {
            ChatSettingsSheet(
                temperature: $temperature,
                topP: $topP,
                topK: $topK,
                maxTokens: $maxTokens,
                systemPrompt: $systemPrompt,
                contextWindow: $contextWindow
            )
        }
    }

    // MARK: - Conversation Sidebar
    private var conversationSidebar: some View {
        VStack(spacing: 0) {
            // New Chat button
            HStack {
                Button {
                    startNewChat()
                } label: {
                    HStack(spacing: 6) {
                        Image(systemName: "plus.bubble.fill")
                        Text("New Chat")
                            .font(.system(size: 13, weight: .semibold))
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 8)
                    .background(ApolloPalette.accent)
                    .foregroundColor(.black)
                    .cornerRadius(8)
                }
                .buttonStyle(.plain)
            }
            .padding(12)

            Divider().background(Color.white.opacity(0.08))

            // Conversations List
            List(selection: Binding(
                get: { chatStore.currentConversationId },
                set: { if let id = $0 { chatStore.selectConversation(id) } }
            )) {
                ForEach(chatStore.conversations) { conv in
                    HStack(spacing: 10) {
                        Image(systemName: "bubble.left")
                            .font(.system(size: 12))
                            .foregroundColor(chatStore.currentConversationId == conv.id ? ApolloPalette.accentStrong : .white.opacity(0.5))

                        VStack(alignment: .leading, spacing: 2) {
                            Text(conv.title.isEmpty ? "New Chat" : conv.title)
                                .font(.system(size: 13, weight: .medium))
                                .foregroundColor(.white.opacity(0.9))
                                .lineLimit(1)

                            Text(conv.lastUpdated, style: .time)
                                .font(.system(size: 10))
                                .foregroundColor(.white.opacity(0.4))
                        }

                        Spacer()

                        Button {
                            chatStore.deleteConversation(conv.id)
                        } label: {
                            Image(systemName: "trash")
                                .font(.system(size: 11))
                                .foregroundColor(.white.opacity(0.3))
                        }
                        .buttonStyle(.plain)
                    }
                    .padding(.vertical, 4)
                    .tag(conv.id)
                }
            }
            .listStyle(.sidebar)
        }
        .background(Color(hex: "090d16").opacity(0.95))
    }

    // MARK: - Active Chat Area
    private var activeChatArea: some View {
        VStack(spacing: 0) {
            // Top Toolbar
            chatHeaderToolbar

            Divider().background(Color.white.opacity(0.08))

            // Messages ScrollView
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(spacing: 20) {
                        if chatStore.currentMessages.isEmpty {
                            emptyChatPlaceholder
                                .padding(.top, 80)
                        } else {
                            ForEach(chatStore.currentMessages) { message in
                                chatMessageBubble(message)
                                    .id(message.id)
                            }
                        }
                    }
                    .padding(24)
                }
                .onChange(of: chatStore.currentMessages.count) { _, _ in
                    if let last = chatStore.currentMessages.last {
                        withAnimation {
                            proxy.scrollTo(last.id, anchor: .bottom)
                        }
                    }
                }
            }

            // Bottom Input Bar
            Divider().background(Color.white.opacity(0.08))
            chatInputBar
        }
    }

    // MARK: - Header
    private var chatHeaderToolbar: some View {
        HStack(spacing: 14) {
            // Model Selector Menu
            Menu {
                let models = ModelManager.shared.downloadedModels
                if models.isEmpty {
                    Button("No models downloaded — Go to Models") {
                        onNavigateToModels()
                    }
                } else {
                    ForEach(models, id: \.id) { model in
                        Button {
                            loadModel(model)
                        } label: {
                            HStack {
                                Text(model.name)
                                if backend.currentlyLoadedModel == model.name {
                                    Image(systemName: "checkmark")
                                }
                            }
                        }
                    }
                }
                Divider()
                Button("Download More Models...") {
                    onNavigateToModels()
                }
            } label: {
                HStack(spacing: 6) {
                    Circle()
                        .fill(backend.isLoaded ? Color.green : Color.orange)
                        .frame(width: 8, height: 8)
                    Text(backend.currentlyLoadedModel ?? "Select Model")
                        .font(.system(size: 13, weight: .semibold))
                        .foregroundColor(.white)
                    Image(systemName: "chevron.down")
                        .font(.system(size: 10))
                        .foregroundColor(.white.opacity(0.6))
                }
                .padding(.horizontal, 10)
                .padding(.vertical, 5)
                .background(Color.white.opacity(0.08))
                .cornerRadius(8)
            }
            .menuStyle(.borderlessButton)

            if generationSpeed > 0 {
                HStack(spacing: 4) {
                    Image(systemName: "bolt.fill")
                        .font(.system(size: 10))
                        .foregroundColor(.yellow)
                    Text(String(format: "%.1f tok/s", generationSpeed))
                        .font(.system(size: 11, design: .monospaced))
                        .foregroundColor(.white.opacity(0.8))
                }
                .padding(.horizontal, 8)
                .padding(.vertical, 4)
                .background(Color.white.opacity(0.06))
                .cornerRadius(6)
            }

            Spacer()

            // Chat Settings
            Button {
                showSettings = true
            } label: {
                Image(systemName: "slider.horizontal.3")
                    .font(.system(size: 14))
                    .foregroundColor(.white.opacity(0.8))
            }
            .buttonStyle(.plain)
            .help("Chat Parameters")

            // Clear chat
            Button {
                chatStore.clearCurrentConversation()
            } label: {
                Image(systemName: "arrow.counterclockwise")
                    .font(.system(size: 14))
                    .foregroundColor(.white.opacity(0.8))
            }
            .buttonStyle(.plain)
            .help("Clear Chat")
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 12)
        .background(Color(hex: "0d121c"))
    }

    // MARK: - Message Bubble
    private func chatMessageBubble(_ message: ChatMessage) -> some View {
        HStack(alignment: .top, spacing: 14) {
            if message.role == .assistant {
                // AI Avatar
                ZStack {
                    Circle()
                        .fill(
                            LinearGradient(
                                colors: [ApolloPalette.accent, ApolloPalette.accentMuted],
                                startPoint: .topLeading,
                                endPoint: .bottomTrailing
                            )
                        )
                        .frame(width: 32, height: 32)
                    Image(systemName: "sparkles")
                        .font(.system(size: 14))
                        .foregroundColor(.black)
                }

                VStack(alignment: .leading, spacing: 8) {
                    // Thinking Block (if reasoning content is present)
                    if let thinking = message.thinkingContent, !thinking.isEmpty {
                        ThinkingBlockView(reasoning: thinking)
                    }

                    MarkdownTextView(text: message.content)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            } else {
                Spacer()

                VStack(alignment: .trailing, spacing: 6) {
                    if let imgUrl = message.imageURL, let image = NSImage(contentsOf: imgUrl) {
                        Image(nsImage: image)
                            .resizable()
                            .scaledToFit()
                            .frame(maxWidth: 300, maxHeight: 200)
                            .cornerRadius(10)
                    }

                    Text(message.content)
                        .font(.system(size: 14))
                        .foregroundColor(.white)
                        .padding(.horizontal, 16)
                        .padding(.vertical, 10)
                        .background(ApolloPalette.accent.opacity(0.25))
                        .overlay(
                            RoundedRectangle(cornerRadius: 14)
                                .stroke(ApolloPalette.accent.opacity(0.4), lineWidth: 1)
                        )
                        .cornerRadius(14)
                }
            }
        }
    }

    // MARK: - Input Bar
    private var chatInputBar: some View {
        VStack(spacing: 8) {
            // Attachment Preview
            if let imgUrl = attachedImageURL, let image = NSImage(contentsOf: imgUrl) {
                HStack {
                    Image(nsImage: image)
                        .resizable()
                        .scaledToFill()
                        .frame(width: 44, height: 44)
                        .cornerRadius(6)

                    Text(imgUrl.lastPathComponent)
                        .font(.system(size: 12))
                        .foregroundColor(.white.opacity(0.7))
                        .lineLimit(1)

                    Spacer()

                    Button {
                        attachedImageURL = nil
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                            .font(.system(size: 16))
                            .foregroundColor(.white.opacity(0.6))
                    }
                    .buttonStyle(.plain)
                }
                .padding(8)
                .background(Color.white.opacity(0.06))
                .cornerRadius(8)
            }

            HStack(alignment: .bottom, spacing: 12) {
                // Attach file button
                Button {
                    pickAttachment()
                } label: {
                    Image(systemName: "paperclip")
                        .font(.system(size: 18))
                        .foregroundColor(.white.opacity(0.7))
                        .padding(8)
                }
                .buttonStyle(.plain)
                .help("Attach Image or Document")

                // Text Editor
                TextEditor(text: $inputText)
                    .font(.system(size: 14))
                    .frame(minHeight: 38, maxHeight: 120)
                    .padding(6)
                    .background(Color(hex: "090d16"))
                    .cornerRadius(10)
                    .overlay(
                        RoundedRectangle(cornerRadius: 10)
                            .stroke(Color.white.opacity(0.12), lineWidth: 1)
                    )

                // Send or Stop Button
                if isGenerating {
                    Button {
                        stopGeneration()
                    } label: {
                        Image(systemName: "stop.fill")
                            .font(.system(size: 14, weight: .bold))
                            .foregroundColor(.white)
                            .frame(width: 36, height: 36)
                            .background(ApolloPalette.destructive)
                            .clipShape(Circle())
                    }
                    .buttonStyle(.plain)
                } else {
                    Button {
                        sendMessage()
                    } label: {
                        Image(systemName: "arrow.up")
                            .font(.system(size: 15, weight: .bold))
                            .foregroundColor(.black)
                            .frame(width: 36, height: 36)
                            .background(inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? Color.white.opacity(0.2) : ApolloPalette.accent)
                            .clipShape(Circle())
                    }
                    .buttonStyle(.plain)
                    .disabled(inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                }
            }
        }
        .padding(16)
        .background(Color(hex: "0b0f19"))
    }

    private var emptyChatPlaceholder: some View {
        VStack(spacing: 16) {
            ZStack {
                Circle()
                    .fill(ApolloPalette.accent.opacity(0.12))
                    .frame(width: 72, height: 72)
                Image(systemName: "bubble.left.and.bubble.right.fill")
                    .font(.system(size: 32))
                    .foregroundColor(ApolloPalette.accentStrong)
            }

            Text("How can I help you today?")
                .font(.system(size: 20, weight: .bold))
                .foregroundColor(.white)

            Text("Type a prompt or choose a starter below to begin.")
                .font(.system(size: 13))
                .foregroundColor(.white.opacity(0.6))

            HStack(spacing: 12) {
                starterButton("Explain Quantum Computing simply")
                starterButton("Write a Python script for file indexing")
                starterButton("Draft an email proposing a partnership")
            }
            .padding(.top, 8)
        }
    }

    private func starterButton(_ text: String) -> some View {
        Button {
            inputText = text
            sendMessage()
        } label: {
            Text(text)
                .font(.system(size: 12))
                .foregroundColor(.white.opacity(0.85))
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .background(Color.white.opacity(0.06))
                .cornerRadius(8)
                .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color.white.opacity(0.1), lineWidth: 1))
        }
        .buttonStyle(.plain)
    }

    // MARK: - Actions
    private func startNewChat() {
        chatStore.createNewConversation()
        inputText = ""
        attachedImageURL = nil
    }

    private func pickAttachment() {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        panel.canChooseFiles = true
        panel.allowedContentTypes = [.image, .pdf, .plainText]

        if panel.runModal() == .OK, let url = panel.url {
            attachedImageURL = url
        }
    }

    private func loadModel(_ model: AIModel) {
        Task {
            _ = try? await backend.loadModel(model, contextWindow: contextWindow)
        }
    }

    private func sendMessage() {
        let prompt = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !prompt.isEmpty else { return }

        let userMsg = ChatMessage(role: .user, content: prompt, imageURL: attachedImageURL)
        chatStore.appendMessage(userMsg)

        inputText = ""
        let attached = attachedImageURL
        attachedImageURL = nil

        isGenerating = true
        generationStartTime = Date()

        let assistantMsgId = UUID()
        let assistantMsg = ChatMessage(id: assistantMsgId, role: .assistant, content: "")
        chatStore.appendMessage(assistantMsg)

        Task {
            var tokenCount = 0
            do {
                let stream = try await backend.generateStream(
                    prompt: prompt,
                    systemPrompt: systemPrompt,
                    imageURL: attached,
                    temperature: temperature,
                    topP: topP,
                    maxTokens: maxTokens
                )

                for try await chunk in stream {
                    chatStore.appendChunkToMessage(id: assistantMsgId, chunk: chunk)
                    tokenCount += 1
                    if let start = generationStartTime {
                        let elapsed = Date().timeIntervalSince(start)
                        if elapsed > 0 {
                            generationSpeed = Double(tokenCount) / elapsed
                        }
                    }
                }
            } catch {
                chatStore.appendChunkToMessage(id: assistantMsgId, chunk: "\n\n*[Error: \(error.localizedDescription)]*")
            }

            await MainActor.run {
                isGenerating = false
            }
        }
    }

    private func stopGeneration() {
        backend.stopGeneration()
        isGenerating = false
    }
}
