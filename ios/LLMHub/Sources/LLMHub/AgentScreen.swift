import SwiftUI
import MapKit
import Speech
import AVFoundation

@available(iOS 17.0, *)
public struct AgentScreen: View {
    @Environment(\.dismiss) private var dismiss
    @EnvironmentObject var settings: AppSettings
    @StateObject private var vm = AgentViewModel()
    @StateObject private var chatVm = ChatViewModel()
    @StateObject private var micTranscriber = ChatMicTranscriber()
    @FocusState private var isComposerFocused: Bool

    var onNavigateToModels: (() -> Void)? = nil

    @AppStorage("agent_model_name") private var agentModelName: String = ""
    @AppStorage("agent_max_tokens") private var agentMaxTokens: Double = 4096
    @AppStorage("agent_enable_thinking") private var agentEnableThinking: Bool = true
    @AppStorage("agent_enable_vision") private var agentEnableVision: Bool = false
    @AppStorage("agent_enable_audio") private var agentEnableAudio: Bool = false
    @State private var isLoadingModel = false
    @State private var errorMessage: String? = nil

    @State private var showSettings = false
    @State private var copiedMessageId: UUID? = nil

    public init(onNavigateToModels: (() -> Void)? = nil) {
        self.onNavigateToModels = onNavigateToModels
    }

    private var downloadedModels: [AIModel] {
        ModelData.allModels().filter { model in
            if model.isDependencyOnly { return false }
            if model.name.hasPrefix("Translate Gemma") { return false }
            // Exclude non-LLM model types from agent availability check
            if model.category == .embedding || model.category == .asr { return false }
            if model.category == .imageGeneration || model.category == .videoGeneration || model.category == .imageUpscale { return false }
            if model.name.lowercased().contains("mmproj") || model.name.lowercased().contains("vision projector") || model.name.lowercased().contains("projector") { return false }

            if case .downloaded = ModelManager.shared.modelStatuses[model.id] { return true }
            if ModelData.isModelFullyAvailableLocally(model) { return true }
            return false
        }
    }

    private var isAnyModelDownloaded: Bool {
        !downloadedModels.isEmpty
    }

    private var lastMessageContent: String {
        guard let last = vm.messages.last else { return "" }
        switch last {
        case .text(_, _, let text, _):
            return text
        case .toolCall(_, let toolName, let args, let status, let result):
            return "\(toolName)_\(args)_\(status)_\(result ?? "")"
        case .map(_, let label, _, _):
            return label
        }
    }

    public var body: some View {
        GeometryReader { geo in
            ZStack(alignment: .bottom) {
                ScrollViewReader { proxy in
                    messageList(proxy: proxy)
                }

                composerPanel
            }
        }
        .apolloScreenBackground()
        .navigationTitle(settings.localized("agent_title"))
        .navigationBarTitleDisplayMode(.inline)
        .toolbarBackground(.hidden, for: .navigationBar)
        .toolbar {
            ToolbarItem(placement: .navigationBarLeading) {
                Button {
                    dismiss()
                } label: {
                    Image(systemName: "arrow.left")
                        .foregroundColor(.white)
                }
            }
            ToolbarItem(placement: .navigationBarTrailing) {
                Button {
                    showSettings = true
                } label: {
                    Image(systemName: "slider.horizontal.3")
                        .foregroundColor(.white)
                }
            }
        }
        .sheet(isPresented: $showSettings) {
            agentSettingsSheet
        }
        .onAppear {
            vm.setupWelcomeMessage(settings: settings, isDownloaded: isAnyModelDownloaded)
        }
        .onDisappear {
            LLMBackend.shared.unloadModel()
        }
        .onChange(of: micTranscriber.liveText) { _, newText in
            if !newText.isEmpty {
                vm.inputText = newText
            }
        }
    }

    private func messageList(proxy: ScrollViewProxy) -> some View {
        ScrollView {
            VStack(spacing: 12) {
                if vm.messages.isEmpty {
                    emptyState
                } else {
                    ForEach(vm.messages) { msg in
                        agentMessageBubble(for: msg)
                    }

                    if vm.isGenerating && !hasStreamingAIResponse {
                        HStack(spacing: 10) {
                            ProgressView().tint(Color(hex: "A78BFA"))
                            Text(settings.localized("agent_processing_tool"))
                                .font(.subheadline)
                                .foregroundColor(.white.opacity(0.85))
                            Spacer()
                        }
                        .padding(.horizontal, 14)
                        .padding(.vertical, 10)
                        .background(Color.white.opacity(0.08))
                        .cornerRadius(14)
                        .padding(.horizontal, 16)
                    }
                }

                Color.clear.frame(height: 1).id("bottom_sentinel")
            }
            .padding(.vertical, 12)
            .padding(.bottom, 12)
        }
        .safeAreaPadding(.bottom, 150)
        .scrollDismissesKeyboard(.interactively)
        .onTapGesture { isComposerFocused = false }
        .onChange(of: vm.messages.count) { _, _ in
            withAnimation { proxy.scrollTo("bottom_sentinel", anchor: .bottom) }
        }
        .onChange(of: lastMessageContent) { _, _ in
            proxy.scrollTo("bottom_sentinel", anchor: .bottom)
        }
        .onChange(of: vm.isGenerating) { _, _ in
            withAnimation { proxy.scrollTo("bottom_sentinel", anchor: .bottom) }
        }
        .onChange(of: isComposerFocused) { _, focused in
            guard focused else { return }
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.05) {
                withAnimation { proxy.scrollTo("bottom_sentinel", anchor: .bottom) }
            }
        }
    }

    private var hasStreamingAIResponse: Bool {
        guard let last = vm.messages.last else { return false }
        if case .text(_, .agent, let content, _) = last { return !content.isEmpty }
        return false
    }

    private var agentSettingsSheet: some View {
        FeatureModelSettingsSheet(
            selectedModelName: $agentModelName,
            maxTokens: $agentMaxTokens,
            enableThinking: $agentEnableThinking,
            enableVision: $agentEnableVision,
            enableAudio: nil,
            isLoading: $isLoadingModel,
            errorMessage: $errorMessage,
            supportsVisionToggle: false,
            visionToggleTitleKey: "enable_vision",
            audioToggleTitleKey: nil,
            visionAvailableCheck: nil,
            writingMode: nil,
            modelFilter: isAgentModel,
            onLoad: loadAgentModel,
            onUnload: { LLMBackend.shared.unloadModel() },
            extraModelConfigsContent: AnyView(MCPConfigurationSection(viewModel: vm))
        )
        .environmentObject(settings)
    }

    private func isAgentModel(_ model: AIModel) -> Bool {
        !model.isDependencyOnly &&
        model.category != .embedding &&
        model.category != .asr &&
        !model.name.lowercased().contains("vision projector") &&
        !model.name.lowercased().contains("mmproj") &&
        !model.name.lowercased().contains("projector")
    }

    private func loadAgentModel() async {
        isLoadingModel = true
        errorMessage = nil
        defer { isLoadingModel = false }
        guard let model = ModelData.allModels().first(where: { $0.name == agentModelName }) else {
            errorMessage = "Model not found"
            return
        }
        do {
            let modelContextCap = model.contextWindowSize > 0 ? model.contextWindowSize : 4096
            let effectiveContext = min(max(1, Int(agentMaxTokens)), modelContextCap)
            LLMBackend.shared.enableThinking = agentEnableThinking
            LLMBackend.shared.maxTokens = min(Int(agentMaxTokens), effectiveContext)
            LLMBackend.shared.contextWindow = effectiveContext
            try await LLMBackend.shared.loadModel(model)
            showSettings = false
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    private var composerPanel: some View {
        VStack(spacing: 8) {
            if copiedMessageId != nil {
                Text(settings.localized("message_copied"))
                    .font(.caption)
                    .foregroundColor(.white)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 6)
                    .background(.ultraThinMaterial)
                    .clipShape(Capsule())
                    .transition(.scale.combined(with: .opacity))
            }

            VStack(spacing: 0) {
                ZStack(alignment: .leading) {
                    if micTranscriber.isPreparing {
                        Text(settings.localized("preparing_mic"))
                            .foregroundColor(.white.opacity(0.45))
                            .font(.body)
                            .padding(.horizontal, 16)
                            .padding(.top, 14)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                    TextField(settings.localized("type_a_message"), text: $vm.inputText, axis: .vertical)
                        .lineLimit(1...6)
                        .padding(.horizontal, 16)
                        .padding(.top, 14)
                        .padding(.bottom, 8)
                        .focused($isComposerFocused)
                        .foregroundColor(.white)
                        .onSubmit { sendCurrentPrompt() }
                }

                HStack(spacing: 6) {
                    Button { vm.isWebSearchEnabled.toggle() } label: {
                        AgentWebSearchToggleLabel(
                            text: settings.localized("web_search"),
                            isEnabled: vm.isWebSearchEnabled
                        )
                    }
                    .disabled(vm.isGenerating)
                    .animation(.easeInOut(duration: 0.15), value: vm.isWebSearchEnabled)

                    Spacer()
                    AgentContextUsageRing()

                    Button {
                        if micTranscriber.isRecording {
                            Task { _ = await micTranscriber.stopLive() }
                        } else {
                            Task { await micTranscriber.startLive() }
                        }
                    } label: {
                        AgentMicButtonLabel(
                            isPreparing: micTranscriber.isPreparing,
                            isRecording: micTranscriber.isRecording
                        )
                    }
                    .disabled(vm.isGenerating)

                    Button {
                        isComposerFocused = false
                        if micTranscriber.isRecording {
                            Task { _ = await micTranscriber.stopLive() }
                        }
                        if !vm.isGenerating { sendCurrentPrompt() }
                    } label: {
                        AgentSendButtonLabel(isGenerating: vm.isGenerating)
                    }
                    .disabled(!vm.isGenerating && vm.inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                }
                .padding(.horizontal, 10)
                .padding(.bottom, 10)
            }
            .background(.ultraThinMaterial)
            .clipShape(RoundedRectangle(cornerRadius: 24))
            .overlay(
                RoundedRectangle(cornerRadius: 24)
                    .stroke(Color.white.opacity(0.12), lineWidth: 1)
            )
        }
        .padding(.horizontal, 16)
        .padding(.bottom, 12)
        .padding(.top, 4)
        .animation(.easeOut(duration: 0.2), value: isComposerFocused)
    }

    private func sendCurrentPrompt() {
        let text = vm.inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        if !text.isEmpty {
            vm.inputText = ""
            micTranscriber.liveText = ""
            vm.sendMessage(text)
        }
    }

    // MARK: - Empty State (1:1 matching AI Chat)

    private var emptyState: some View {
        VStack(spacing: 20) {
            Spacer(minLength: 60)

            Text(settings.localized("welcome_to_llm_hub"))
                .font(.title2.bold())
                .foregroundColor(.white)
                .multilineTextAlignment(.center)

            Text(settings.localized("agent_no_model_ios"))
                .font(.subheadline)
                .foregroundColor(.white.opacity(0.68))
                .multilineTextAlignment(.center)

            if let onNavigateToModels = onNavigateToModels {
                Button {
                    onNavigateToModels()
                } label: {
                    Label(settings.localized("download_a_model"), systemImage: "arrow.down.circle")
                }
                .buttonStyle(ApolloIconButtonStyle())
            }

            Spacer()
        }
        .padding(.horizontal, 32)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    // MARK: - Message rendering (AI messages take FULL WIDTH without bubble, matching AI Chat)

    @ViewBuilder
    private func agentMessageBubble(for msg: AgentMessageItem) -> some View {
        switch msg {
        case .text(_, let sender, let content, _):
            if sender == .user {
                HStack {
                    Spacer(minLength: 40)
                    Text(content)
                        .font(.body)
                        .foregroundColor(.white)
                        .padding(.horizontal, 14)
                        .padding(.vertical, 10)
                        .background(
                            RoundedRectangle(cornerRadius: 18)
                                .fill(LinearGradient(colors: [Color(hex: "6f93cd"), Color(hex: "455c82")], startPoint: .topLeading, endPoint: .bottomTrailing))
                        )
                        .overlay(
                            RoundedRectangle(cornerRadius: 18)
                                .stroke(Color.white.opacity(0.16), lineWidth: 1)
                        )
                }
                .padding(.horizontal, 16)
            } else {
                // AI Side (Greetings, Agent responses): Markdown, Tables, LaTeX 1:1 matching AI Chat!
                VStack(alignment: .leading, spacing: 8) {
                    if content.contains("|") && content.contains("-|-") {
                        MarkdownTableView(rawTable: content)
                    } else {
                        let isCurrentMsgGenerating = vm.isGenerating && msg.id == vm.messages.last?.id
                        let selectedModel = ModelData.allModels().first(where: { $0.name == agentModelName })
                        let isLfm = agentModelName.lowercased().contains("lfm")
                        let modelSupportsThinking = isLfm || (selectedModel?.supportsThinking ?? false)
                        let preferThinking = modelSupportsThinking && agentEnableThinking
                        ThinkingAwareResultContent(
                            content: content,
                            isGenerating: isCurrentMsgGenerating,
                            preferThinkingWhileStreaming: preferThinking,
                            useChatRenderer: true
                        )
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.vertical, 4)
                .padding(.horizontal, 16)
            }

        case .toolCall(let id, let name, let args, let status, let result):
            AgentToolCallCell(
                name: name,
                args: args,
                status: status,
                result: result,
                onApprove: name.hasPrefix("mcp_") ? { vm.approveMCPTool(id: id) } : nil,
                onDeny: name.hasPrefix("mcp_") ? { vm.denyMCPTool(id: id) } : nil
            )
                .padding(.horizontal, 16)

        case .map(_, let label, let latitude, let longitude):
            AgentMapViewCell(label: label, latitude: latitude, longitude: longitude)
                .padding(.horizontal, 16)
        }
    }
}



// MARK: - Tool Call Cell

struct AgentToolCallCell: View {
    @EnvironmentObject var settings: AppSettings
    let name: String
    let args: String
    let status: AgentMessageItem.ToolStatus
    let result: String?
    let onApprove: (() -> Void)?
    let onDeny: (() -> Void)?

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Image(systemName: statusIcon)
                    .foregroundColor(statusColor)

                VStack(alignment: .leading, spacing: 2) {
                    Text("\(name)(\(args))")
                        .font(.caption)
                        .bold()
                        .foregroundColor(.white)
                    if let res = result {
                        Text(res)
                            .font(.caption2)
                            .foregroundColor(.white.opacity(0.7))
                            .lineLimit(2)
                    }
                }

                Spacer()

                Text(statusText)
                    .font(.system(size: 10))
                    .foregroundColor(.white.opacity(0.6))
            }
            if status == .pendingApproval {
                HStack(spacing: 8) {
                    Button(settings.localized("agent_mcp_allow")) { onApprove?() }
                        .buttonStyle(.borderedProminent)
                        .tint(Color(hex: "10B981"))
                    Button(settings.localized("agent_mcp_deny"), role: .cancel) { onDeny?() }
                        .buttonStyle(.bordered)
                }
                .font(.caption.bold())
            }
        }
        .padding(12)
        .background(Color.white.opacity(0.1))
        .cornerRadius(14)
        .overlay(
            RoundedRectangle(cornerRadius: 14)
                .stroke(Color(hex: "A78BFA").opacity(0.3), lineWidth: 1)
        )
    }

    private var statusIcon: String {
        switch status {
        case .pendingApproval: return "hand.raised.fill"
        case .running: return "gearshape.fill"
        case .success: return "checkmark.circle.fill"
        case .failed: return "xmark.circle.fill"
        }
    }

    private var statusColor: Color {
        switch status {
        case .pendingApproval: return Color(hex: "F59E0B")
        case .running: return Color(hex: "FBBF24")
        case .success: return Color(hex: "34D399")
        case .failed: return Color(hex: "F87171")
        }
    }

    private var statusText: String {
        switch status {
        case .pendingApproval: return settings.localized("agent_mcp_approval_required")
        case .running: return NSLocalizedString("agent_tool_running", comment: "")
        case .success: return NSLocalizedString("agent_tool_success", comment: "")
        case .failed: return NSLocalizedString("agent_tool_failed", comment: "")
        }
    }
}

private struct AgentWebSearchToggleLabel: View {
    let text: String
    let isEnabled: Bool

    var body: some View {
        HStack(spacing: 4) {
            Image(systemName: "globe")
                .font(.system(size: 12, weight: .semibold))
            Text(text)
                .font(.system(size: 12, weight: .semibold))
        }
        .foregroundColor(isEnabled ? .black : .white.opacity(0.75))
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
        .background(isEnabled ? Color.white : Color.white.opacity(0.09))
        .clipShape(Capsule())
        .overlay(
            Capsule().stroke(
                isEnabled ? Color.clear : Color.white.opacity(0.15),
                lineWidth: 1
            )
        )
    }
}

private struct AgentContextUsageRing: View {
    var body: some View {
        ZStack {
            Circle()
                .inset(by: 0.75)
                .stroke(Color.white.opacity(0.18), lineWidth: 1.5)
            Circle()
                .inset(by: 1.0)
                .trim(from: 0, to: 0.05)
                .stroke(
                    ApolloPalette.accentStrong,
                    style: StrokeStyle(lineWidth: 2.0, lineCap: .round)
                )
                .rotationEffect(.degrees(-90))
            Text("0%")
                .font(.system(size: 8, weight: .bold, design: .rounded))
        }
        .frame(width: 32, height: 32)
        .padding(.trailing, 4)
    }
}

private struct AgentMicButtonLabel: View {
    let isPreparing: Bool
    let isRecording: Bool

    var body: some View {
        ZStack {
            Circle()
                .fill(.ultraThinMaterial)
                .frame(width: 32, height: 32)
                .overlay(Circle().stroke(Color.white.opacity(0.18), lineWidth: 1))
            if isRecording {
                Circle()
                    .fill(Color.red.opacity(0.25))
                    .frame(width: 32, height: 32)
            }
            Image(systemName: isPreparing ? "ellipsis" : isRecording ? "stop.fill" : "mic.fill")
                .font(.system(size: 14, weight: .semibold))
                .foregroundColor(isRecording ? .red : .white)
        }
    }
}

private struct AgentSendButtonLabel: View {
    let isGenerating: Bool

    var body: some View {
        Image(systemName: isGenerating ? "stop.fill" : "arrow.up")
            .font(.system(size: 14, weight: .bold))
            .foregroundColor(isGenerating ? .white : .black)
            .frame(width: 32, height: 32)
            .background(isGenerating ? Color.red.opacity(0.8) : Color.white)
            .clipShape(Circle())
    }
}

private struct MCPConfigurationSection: View {
    @EnvironmentObject var settings: AppSettings
    @ObservedObject var viewModel: AgentViewModel
    @State private var enabled: Bool
    @State private var url: String
    @State private var token: String

    init(viewModel: AgentViewModel) {
        self.viewModel = viewModel
        _enabled = State(initialValue: viewModel.mcpSettings.enabled)
        _url = State(initialValue: viewModel.mcpSettings.url)
        _token = State(initialValue: viewModel.mcpSettings.token)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Toggle(isOn: Binding(
                get: { enabled },
                set: { value in
                    enabled = value
                    viewModel.setMCPEnabled(value)
                }
            )) { Text(settings.localized("agent_mcp_title")) }
            .tint(ApolloPalette.accentStrong)
            .foregroundStyle(.white)

            if enabled {
                Text(settings.localized("agent_mcp_description"))
                    .font(.caption)
                    .foregroundStyle(.white.opacity(0.65))
                TextField(settings.localized("agent_mcp_server_url"), text: $url)
                    .textInputAutocapitalization(.never)
                    .keyboardType(.URL)
                    .padding(10)
                    .background(Color.white.opacity(0.08))
                    .clipShape(RoundedRectangle(cornerRadius: 10))

                SecureField(settings.localized("agent_mcp_bearer_token"), text: $token)
                    .textInputAutocapitalization(.never)
                    .padding(10)
                    .background(Color.white.opacity(0.08))
                    .clipShape(RoundedRectangle(cornerRadius: 10))

                Text(settings.localized("agent_mcp_approval_notice"))
                    .font(.caption)
                    .foregroundStyle(.white.opacity(0.65))

                if !viewModel.mcpStatus.isEmpty {
                    if viewModel.mcpStatus == "connecting" {
                        HStack { ProgressView(); Text(settings.localized("agent_mcp_connecting")) }
                    } else if viewModel.mcpStatus == "connected" {
                        Text(String(format: settings.localized("agent_mcp_connected_tools"), viewModel.mcpTools.count))
                            .foregroundStyle(.green)
                    } else {
                        Text(String(format: settings.localized("agent_mcp_error"), viewModel.mcpStatus))
                            .foregroundStyle(.red)
                    }
                }

                if !viewModel.mcpTools.isEmpty {
                    ForEach(viewModel.mcpTools) { tool in
                        VStack(alignment: .leading, spacing: 2) {
                            Text("• \(tool.name)").font(.caption.weight(.semibold))
                            if !tool.description.isEmpty {
                                Text(tool.description).font(.caption2).foregroundStyle(.white.opacity(0.6))
                            }
                        }
                    }
                }

                HStack(spacing: 8) {
                    Button(settings.localized("agent_mcp_connect")) {
                        viewModel.connectMCP(MCPSettings(enabled: true, url: url, token: token))
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(ApolloPalette.accentStrong)
                    .disabled(url.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)

                    Button(settings.localized("agent_mcp_disconnect"), role: .destructive) {
                        viewModel.disconnectMCP()
                        enabled = false
                    }
                    .buttonStyle(.bordered)
                }
            }
        }
    }
}

// MARK: - Interactive Map View Cell (MapKit)

struct AgentMapViewCell: View {
    let label: String
    let latitude: Double
    let longitude: Double

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Image(systemName: "mappin.and.ellipse")
                    .foregroundColor(Color(hex: "A78BFA"))
                Text(label)
                    .font(.caption)
                    .bold()
                    .foregroundColor(.white)

                Spacer()

                Button {
                    openExternalMap()
                } label: {
                    HStack(spacing: 4) {
                        Text(NSLocalizedString("open_maps", comment: ""))
                            .font(.system(size: 11, weight: .semibold))
                        Image(systemName: "arrow.up.right")
                            .font(.system(size: 10, weight: .bold))
                    }
                    .foregroundColor(Color(hex: "A78BFA"))
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(Color.white.opacity(0.12))
                    .cornerRadius(12)
                }
            }
            .padding(.horizontal, 12)
            .padding(.top, 8)

            MapViewRepresentable(coordinate: CLLocationCoordinate2D(latitude: latitude, longitude: longitude), title: label)
                .frame(height: 200)
                .cornerRadius(12)
                .onTapGesture {
                    openExternalMap()
                }
        }
        .background(Color.white.opacity(0.1))
        .cornerRadius(16)
    }

    private func openExternalMap() {
        let query = label.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? ""
        if let url = URL(string: "http://maps.apple.com/?q=\(query)&ll=\(latitude),\(longitude)") {
            UIApplication.shared.open(url)
        }
    }
}

struct MapViewRepresentable: UIViewRepresentable {
    let coordinate: CLLocationCoordinate2D
    let title: String

    func makeUIView(context: Context) -> MKMapView {
        let mapView = MKMapView()
        mapView.isScrollEnabled = true
        mapView.isZoomEnabled = true
        let region = MKCoordinateRegion(center: coordinate, latitudinalMeters: 2000, longitudinalMeters: 2000)
        mapView.setRegion(region, animated: false)

        let annotation = MKPointAnnotation()
        annotation.coordinate = coordinate
        annotation.title = title
        mapView.addAnnotation(annotation)

        return mapView
    }

    func updateUIView(_ uiView: MKMapView, context: Context) {}
}
