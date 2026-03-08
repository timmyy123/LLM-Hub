import SwiftUI

// MARK: - Data Models
struct ChatMessage: Identifiable, Equatable, Sendable {
    let id: UUID
    var content: String
    let isFromUser: Bool
    let timestamp: Date
    var isGenerating: Bool

    init(id: UUID = UUID(), content: String, isFromUser: Bool, timestamp: Date = Date(), isGenerating: Bool = false) {
        self.id = id
        self.content = content
        self.isFromUser = isFromUser
        self.timestamp = timestamp
        self.isGenerating = isGenerating
    }
}

struct ChatSession: Identifiable, Sendable {
    let id: UUID
    var title: String
    var messages: [ChatMessage]
    let createdAt: Date

    init(id: UUID = UUID(), title: String = "", messages: [ChatMessage] = [], createdAt: Date = Date()) {
        self.id = id
        self.title = title
        self.messages = messages
        self.createdAt = createdAt
    }
}

// MARK: - Chat ViewModel
@MainActor
class ChatViewModel: ObservableObject {
    @Published var messages: [ChatMessage] = []
    @Published var inputText: String = ""
    @Published var isGenerating: Bool = false
    @Published var tokensPerSecond: Double = 0
    @Published var totalTokens: Int = 0
    @Published var selectedModelName: String = AppSettings.shared.localized("no_model_selected")
    @Published var chatSessions: [ChatSession] = []
    @Published var currentSessionId: UUID = UUID()

    private var streamingTask: Task<Void, Never>?

    init() {
        let session = ChatSession(title: AppSettings.shared.localized("drawer_new_chat"))
        chatSessions = [session]
        currentSessionId = session.id
    }

    func sendMessage() {
        guard !inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
        guard !isGenerating else { return }

        let userMsg = ChatMessage(content: inputText, isFromUser: true)
        messages.append(userMsg)
        inputText = ""

        let aiMsg = ChatMessage(content: "", isFromUser: false, isGenerating: true)
        messages.append(aiMsg)
        isGenerating = true

        streamingTask = Task {
            let fakeResponse = AppSettings.shared.localized("please_download_model")
            var current = ""
            let startTime = Date()
            var tokenCount = 0

            for char in fakeResponse {
                if Task.isCancelled { break }
                try? await Task.sleep(nanoseconds: 20_000_000)
                current += String(char)
                tokenCount += 1

                let capturedCurrent = current
                let capturedTokenCount = tokenCount

                await MainActor.run {
                    if let idx = self.messages.indices.last, !self.messages[idx].isFromUser {
                        self.messages[idx].content = capturedCurrent
                        self.totalTokens = capturedTokenCount
                        let elapsed = Date().timeIntervalSince(startTime)
                        self.tokensPerSecond = elapsed > 0 ? Double(capturedTokenCount) / elapsed : 0
                    }
                }
            }

            await MainActor.run {
                if let idx = self.messages.indices.last, !self.messages[idx].isFromUser {
                    self.messages[idx].isGenerating = false
                }
                self.isGenerating = false
            }
        }
    }

    func stopGeneration() {
        streamingTask?.cancel()
        streamingTask = nil
        if let idx = messages.indices.last, !messages[idx].isFromUser {
            messages[idx].isGenerating = false
        }
        isGenerating = false
    }

    func copyMessage(_ message: ChatMessage) {
        UIPasteboard.general.string = message.content
    }

    func newChat() {
        let session = ChatSession(title: AppSettings.shared.localized("drawer_new_chat"))
        chatSessions.insert(session, at: 0)
        currentSessionId = session.id
        messages = []
        isGenerating = false
        streamingTask?.cancel()
    }

    func deleteSession(_ id: UUID) {
        chatSessions.removeAll { $0.id == id }
        if currentSessionId == id {
            if let first = chatSessions.first {
                currentSessionId = first.id
                messages = first.messages
            } else {
                newChat()
            }
        }
    }
}

// MARK: - Message Bubble
struct MessageBubble: View {
    @EnvironmentObject var settings: AppSettings
    let message: ChatMessage
    let onCopy: () -> Void
    @State private var isCopied = false
    @State private var showActions = false

    var body: some View {
        HStack(alignment: .bottom, spacing: 8) {
            if message.isFromUser { Spacer(minLength: 40) }

            if !message.isFromUser {
                Image(systemName: "cpu")
                    .font(.system(size: 16))
                    .foregroundColor(.white)
                    .frame(width: 28, height: 28)
                    .background(Color.indigo.gradient)
                    .clipShape(Circle())
            }

            VStack(alignment: message.isFromUser ? .trailing : .leading, spacing: 4) {
                ZStack(alignment: .bottomTrailing) {
                    Text(message.content.isEmpty && message.isGenerating ? "●●●" : (message.content.isEmpty ? " " : message.content))
                        .font(.body)
                        .foregroundColor(message.isFromUser ? .white : .primary)
                        .padding(.horizontal, 14)
                        .padding(.vertical, 10)
                        .background(
                            RoundedRectangle(cornerRadius: 18)
                                .fill(message.isFromUser
                                      ? LinearGradient(colors: [Color.indigo, Color.purple], startPoint: .topLeading, endPoint: .bottomTrailing)
                                      : LinearGradient(colors: [Color(.secondarySystemBackground)], startPoint: .top, endPoint: .bottom))
                        )
                        .contentShape(RoundedRectangle(cornerRadius: 18))
                        .onLongPressGesture {
                            showActions = true
                        }

                    if message.isGenerating {
                        TypingIndicator()
                            .padding(10)
                    }
                }

                Text(message.timestamp, style: .time)
                    .font(.caption2)
                    .foregroundColor(.secondary)
                    .padding(.horizontal, 4)
            }

            if !message.isFromUser { Spacer(minLength: 40) }
            if message.isFromUser {
                Image(systemName: "person.crop.circle.fill")
                    .font(.system(size: 28))
                    .foregroundColor(.indigo.opacity(0.7))
            }
        }
        .confirmationDialog(settings.localized("more_options"), isPresented: $showActions) {
            Button(settings.localized("copy_message")) {
                onCopy()
                isCopied = true
                DispatchQueue.main.asyncAfter(deadline: .now() + 2) { isCopied = false }
            }
            Button(settings.localized("cancel"), role: .cancel) {}
        }
    }
}

// MARK: - Typing Indicator
struct TypingIndicator: View {
    @State private var phase = 0.0

    var body: some View {
        HStack(spacing: 4) {
            ForEach(0..<3) { i in
                Circle()
                    .fill(Color.secondary)
                    .frame(width: 6, height: 6)
                    .scaleEffect(1.0 + 0.4 * sin(phase + Double(i) * .pi / 1.5))
            }
        }
        .onAppear {
            withAnimation(.linear(duration: 1).repeatForever(autoreverses: false)) {
                phase = .pi * 2
            }
        }
    }
}

// MARK: - Drawer Panel
struct ChatDrawerPanel: View {
    @EnvironmentObject var settings: AppSettings
    @ObservedObject var vm: ChatViewModel
    let onClose: () -> Void
    let onNavigateToModels: () -> Void
    let onNavigateToSettings: () -> Void
    @State private var showDeleteAllAlert = false

    var body: some View {
        NavigationStack {
            List {
                Section {
                    Button {
                        vm.newChat()
                        onClose()
                    } label: {
                        Label(settings.localized("drawer_new_chat"), systemImage: "plus.bubble.fill")
                            .foregroundColor(.indigo)
                            .fontWeight(.semibold)
                    }
                }

                Section(settings.localized("drawer_recent_chats")) {
                    if vm.chatSessions.isEmpty {
                        Text(settings.localized("drawer_no_chats"))
                            .foregroundColor(.secondary)
                            .font(.subheadline)
                    } else {
                        ForEach(vm.chatSessions) { session in
                            Button {
                                vm.currentSessionId = session.id
                                onClose()
                            } label: {
                                HStack {
                                    Image(systemName: "bubble.left.fill")
                                        .foregroundColor(.indigo.opacity(0.7))
                                    VStack(alignment: .leading, spacing: 2) {
                                        Text(session.title)
                                            .foregroundColor(.primary)
                                            .lineLimit(1)
                                        Text(session.createdAt, style: .relative)
                                            .font(.caption)
                                            .foregroundColor(.secondary)
                                    }
                                    Spacer()
                                    if session.id == vm.currentSessionId {
                                        Image(systemName: "checkmark.circle.fill")
                                            .foregroundColor(.indigo)
                                    }
                                }
                            }
                            .swipeActions(edge: .trailing) {
                                Button(role: .destructive) {
                                    vm.deleteSession(session.id)
                                } label: {
                                    Label(settings.localized("action_delete"), systemImage: "trash")
                                }
                            }
                        }
                    }
                }

                Section {
                    Button {
                        onNavigateToModels()
                        onClose()
                    } label: {
                        Label(settings.localized("drawer_download_models"), systemImage: "square.and.arrow.down")
                    }
                    Button {
                        onNavigateToSettings()
                        onClose()
                    } label: {
                        Label(settings.localized("drawer_settings"), systemImage: "gearshape")
                    }
                    if !vm.chatSessions.isEmpty {
                        Button(role: .destructive) {
                            showDeleteAllAlert = true
                        } label: {
                            Label(settings.localized("drawer_clear_all_chats"), systemImage: "trash")
                        }
                    }
                }
            }
            .navigationTitle(settings.localized("drawer_title"))
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button(settings.localized("done"), action: onClose)
                }
            }
        }
        .alert(settings.localized("dialog_delete_all_chats_title"), isPresented: $showDeleteAllAlert) {
            Button(settings.localized("action_delete_all"), role: .destructive) {
                vm.chatSessions.removeAll()
                vm.newChat()
            }
            Button(settings.localized("action_cancel"), role: .cancel) {}
        } message: {
            Text(settings.localized("dialog_delete_all_chats_message"))
        }
    }
}

// MARK: - Model Selector Sheet
struct ModelSelectorSheet: View {
    @EnvironmentObject var settings: AppSettings
    @ObservedObject var vm: ChatViewModel
    let onNavigateToModels: () -> Void
    @Environment(\.dismiss) var dismiss

    var body: some View {
        NavigationStack {
            VStack(spacing: 24) {
                if downloadedModels.isEmpty {
                    Image(systemName: "cpu.fill")
                        .font(.system(size: 48))
                        .foregroundStyle(.linearGradient(colors: [.indigo, .purple], startPoint: .top, endPoint: .bottom))

                    Text(settings.localized("no_models_downloaded"))
                        .font(.title2.bold())

                    Text(settings.localized("download_models_first"))
                        .multilineTextAlignment(.center)
                        .foregroundColor(.secondary)
                        .padding(.horizontal)

                    Button {
                        onNavigateToModels()
                        dismiss()
                    } label: {
                        Label(settings.localized("feature_download_model"), systemImage: "square.and.arrow.down")
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(.indigo.gradient)
                            .foregroundColor(.white)
                            .clipShape(RoundedRectangle(cornerRadius: 14))
                    }
                    .padding(.horizontal)
                } else {
                    List {
                        Section(settings.localized("downloaded")) {
                            ForEach(downloadedModels) { model in
                                Button {
                                    vm.selectedModelName = model.name
                                    dismiss()
                                } label: {
                                    HStack {
                                        Image(systemName: "cpu")
                                            .foregroundColor(.indigo)
                                        VStack(alignment: .leading) {
                                            Text(model.name)
                                                .font(.headline)
                                            Text(model.sizeLabel)
                                                .font(.caption)
                                                .foregroundColor(.secondary)
                                        }
                                        Spacer()
                                        if vm.selectedModelName == model.name {
                                            Image(systemName: "checkmark.circle.fill")
                                                .foregroundColor(.indigo)
                                        }
                                    }
                                }
                                .foregroundColor(.primary)
                            }
                        }
                    }
                }

                Spacer()
            }
            .padding(.top, downloadedModels.isEmpty ? 32 : 0)
            .navigationTitle(settings.localized("select_model_title"))
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button(settings.localized("cancel")) { dismiss() }
                }
            }
        }
    }

    private var downloadedModels: [AIModel] {
        let modelsDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!.appendingPathComponent("models")
        return ModelData.models.filter { model in
            let weightsFile = modelsDir.appendingPathComponent(model.id).appendingPathComponent("model.safetensors")
            return FileManager.default.fileExists(atPath: weightsFile.path)
        }
    }
}

// MARK: - ChatScreen
struct ChatScreen: View {
    @EnvironmentObject var settings: AppSettings
    @StateObject private var vm = ChatViewModel()
    var onNavigateToSettings: () -> Void
    var onNavigateToModels: () -> Void
    var onNavigateBack: () -> Void

    @State private var showDrawer = false
    @State private var showModelSelector = false
    @State private var copiedMessageId: UUID? = nil

    var body: some View {
        VStack(spacing: 0) {
            if !vm.messages.isEmpty {
                HStack(spacing: 12) {
                    HStack(spacing: 6) {
                        Circle()
                            .fill(vm.isGenerating ? Color.orange : Color.green)
                            .frame(width: 8, height: 8)
                        if vm.isGenerating {
                            Text(settings.localized("thinking_label"))
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    }
                    Spacer()
                    if vm.totalTokens > 0 {
                        Label(String(format: settings.localized("tokens_per_second_format"), vm.totalTokens, vm.tokensPerSecond),
                              systemImage: "bolt.fill")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }
                .padding(.horizontal, 16)
                .padding(.vertical, 8)
                .background(.thinMaterial)
            }

            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(spacing: 12) {
                        if vm.messages.isEmpty {
                            emptyState
                        } else {
                            ForEach(vm.messages) { msg in
                                MessageBubble(message: msg) {
                                    vm.copyMessage(msg)
                                    copiedMessageId = msg.id
                                    DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
                                        copiedMessageId = nil
                                    }
                                }
                                .id(msg.id)
                                .padding(.horizontal, 12)
                            }
                        }
                    }
                    .padding(.vertical, 12)
                }
                .onChange(of: vm.messages.count) { _, _ in
                    if let last = vm.messages.last {
                        withAnimation { proxy.scrollTo(last.id, anchor: .bottom) }
                    }
                }
            }

            if let _ = copiedMessageId {
                Text(settings.localized("message_copied"))
                    .font(.caption)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 6)
                    .background(.ultraThinMaterial)
                    .clipShape(Capsule())
                    .transition(.scale.combined(with: .opacity))
            }

            Divider()

            HStack(spacing: 10) {
                TextField(settings.localized("type_a_message"), text: $vm.inputText, axis: .vertical)
                    .lineLimit(1...5)
                    .padding(.horizontal, 14)
                    .padding(.vertical, 10)
                    .background(Color(.secondarySystemBackground))
                    .clipShape(RoundedRectangle(cornerRadius: 22))
                    .onSubmit { vm.sendMessage() }

                Button {
                    if vm.isGenerating {
                        vm.stopGeneration()
                    } else {
                        vm.sendMessage()
                    }
                } label: {
                    Image(systemName: vm.isGenerating ? "stop.circle.fill" : "arrow.up.circle.fill")
                        .font(.system(size: 34))
                        .foregroundStyle(vm.isGenerating ? .red : .indigo)
                }
                .disabled(!vm.isGenerating && vm.inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 10)
            .background(.background)
        }
        .navigationTitle(vm.chatSessions.first(where: { $0.id == vm.currentSessionId })?.title ?? settings.localized("chat"))
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .navigationBarLeading) {
                Button {
                    showDrawer = true
                } label: {
                    Image(systemName: "line.3.horizontal")
                }
            }
            ToolbarItem(placement: .navigationBarTrailing) {
                Button {
                    showModelSelector = true
                } label: {
                    Image(systemName: "slider.horizontal.3")
                }
            }
        }
        .sheet(isPresented: $showDrawer) {
            ChatDrawerPanel(
                vm: vm,
                onClose: { showDrawer = false },
                onNavigateToModels: onNavigateToModels,
                onNavigateToSettings: onNavigateToSettings
            )
        }
        .sheet(isPresented: $showModelSelector) {
            ModelSelectorSheet(vm: vm, onNavigateToModels: onNavigateToModels)
        }
    }

    var emptyState: some View {
        VStack(spacing: 20) {
            Spacer(minLength: 60)
            if let uiImage = UIImage(named: "Icon") {
                Image(uiImage: uiImage)
                    .resizable()
                    .scaledToFit()
                    .frame(width: 80, height: 80)
                    .cornerRadius(16)
            } else {
                Image(systemName: "cpu")
                    .font(.system(size: 64))
                    .foregroundStyle(.linearGradient(colors: [.indigo, .purple], startPoint: .top, endPoint: .bottom))
            }
            
            Text(settings.localized("welcome_to_llm_hub"))
                .font(.title2.bold())
                
            if downloadedModels.isEmpty {
                Text(settings.localized("no_models_downloaded"))
                    .foregroundColor(.secondary)
                Button {
                    onNavigateToModels()
                } label: {
                    Label(settings.localized("download_a_model"), systemImage: "arrow.down.circle")
                }
                .buttonStyle(.borderedProminent)
            } else if vm.selectedModelName == settings.localized("no_model_selected") {
                Text(settings.localized("load_model_to_start"))
                    .foregroundColor(.secondary)
            } else {
                Text(vm.selectedModelName)
                    .font(.caption)
                    .padding(.horizontal, 12).padding(.vertical, 6)
                    .background(Color.secondary.opacity(0.2))
                    .clipShape(Capsule())
                Text(settings.localized("start_chatting"))
                    .foregroundColor(.secondary)
                    .multilineTextAlignment(.center)
            }
        }
        .padding(.horizontal, 32)
    }
    
    private var downloadedModels: [AIModel] {
        let modelsDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!.appendingPathComponent("models")
        return ModelData.models.filter { model in
            let weightsFile = modelsDir.appendingPathComponent(model.id).appendingPathComponent("model.safetensors")
            return FileManager.default.fileExists(atPath: weightsFile.path)
        }
    }
}
