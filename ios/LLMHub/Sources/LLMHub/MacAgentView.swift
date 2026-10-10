//
//  MacAgentView.swift
//  LLMHub
//
//  Native macOS Agent. Drives the same `AgentViewModel` (tools, MCP, web
//  search, permission flows), persisted settings and localized strings as
//  `AgentScreen` on iOS.
//

#if os(macOS)
import SwiftUI
import AppKit
import MapKit

struct MacAgentView: View {
    @EnvironmentObject var settings: AppSettings
    @StateObject private var vm = AgentViewModel()
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

    init(onNavigateToModels: (() -> Void)? = nil) {
        self.onNavigateToModels = onNavigateToModels
    }

    private var downloadedModels: [AIModel] {
        ModelData.allModels().filter { model in
            if model.isDependencyOnly { return false }
            if model.name.hasPrefix("Translate Gemma") { return false }
            if !model.isLanguageModel { return false }
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

    private var hasStreamingAIResponse: Bool {
        guard let last = vm.messages.last else { return false }
        if case .text(_, .agent, let content, _) = last { return !content.isEmpty }
        return false
    }

    var body: some View {
        VStack(spacing: 0) {
            ScrollViewReader { proxy in
                messageList(proxy: proxy)
            }
            Divider()
            composerPanel
        }
        .navigationTitle(settings.localized("agent_title"))
        .toolbar {
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
                extraModelConfigsContent: AnyView(MacMCPConfigurationSection(viewModel: vm))
            )
            .inspectorColumnWidth(min: 300, ideal: 340, max: 440)
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

    // MARK: - Messages

    private func messageList(proxy: ScrollViewProxy) -> some View {
        ScrollView {
            LazyVStack(spacing: 12) {
                if vm.messages.isEmpty {
                    emptyState
                } else {
                    ForEach(vm.messages) { msg in
                        agentMessageBubble(for: msg)
                    }

                    if vm.isGenerating && !hasStreamingAIResponse {
                        HStack(spacing: 10) {
                            ProgressView().controlSize(.small)
                            Text(settings.localized("agent_processing_tool"))
                                .font(.subheadline)
                                .foregroundStyle(.secondary)
                            Spacer()
                        }
                        .padding(.horizontal, 12)
                        .padding(.vertical, 8)
                        .background(Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: 10))
                        .padding(.horizontal, 20)
                    }
                }

                Color.clear.frame(height: 1).id("bottom_sentinel")
            }
            .frame(maxWidth: 860)
            .frame(maxWidth: .infinity)
            .padding(.vertical, 16)
        }
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

    private var emptyState: some View {
        ContentUnavailableView {
            Text(settings.localized("welcome_to_llm_hub"))
        } description: {
            Text(settings.localized("agent_no_model_ios"))
        } actions: {
            if let onNavigateToModels {
                Button {
                    onNavigateToModels()
                } label: {
                    Label(settings.localized("download_a_model"), systemImage: "arrow.down.circle")
                }
                .buttonStyle(.borderedProminent)
            }
        }
        .padding(.top, 60)
    }

    @ViewBuilder
    private func agentMessageBubble(for msg: AgentMessageItem) -> some View {
        switch msg {
        case .text(_, let sender, let content, _):
            if sender == .user {
                HStack {
                    Spacer(minLength: 80)
                    Text(content)
                        .font(.body)
                        .textSelection(.enabled)
                        .foregroundStyle(.white)
                        .padding(.horizontal, 12)
                        .padding(.vertical, 8)
                        .background(Color.accentColor, in: RoundedRectangle(cornerRadius: 14))
                }
                .padding(.horizontal, 20)
            } else {
                VStack(alignment: .leading, spacing: 8) {
                    if content.contains("|") && content.contains("-|-") {
                        MarkdownTableView(rawTable: content)
                    } else {
                        let isCurrentMsgGenerating = vm.isGenerating && msg.id == vm.messages.last?.id
                        let selectedModel = ModelData.allModels().first(where: { $0.name == agentModelName })
                        let isLfm = agentModelName.lowercased().contains("lfm")
                        let isGranite42 = agentModelName.lowercased().contains("granite-4.2") || agentModelName.lowercased().contains("granite 4.2")
                        let modelSupportsThinking = !isGranite42 && (isLfm || (selectedModel?.supportsThinking ?? false))
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
                .padding(.horizontal, 20)
            }

        case .toolCall(let id, let name, let args, let status, let result):
            MacAgentToolCallCell(
                name: name,
                args: args,
                status: status,
                result: result,
                onApprove: name.hasPrefix("mcp_") ? { vm.approveMCPTool(id: id) } : nil,
                onDeny: name.hasPrefix("mcp_") ? { vm.denyMCPTool(id: id) } : nil
            )
            .padding(.horizontal, 20)

        case .map(_, let label, let latitude, let longitude):
            MacAgentMapViewCell(label: label, latitude: latitude, longitude: longitude)
                .padding(.horizontal, 20)
        }
    }

    // MARK: - Settings / model loading

    private func isAgentModel(_ model: AIModel) -> Bool {
        !model.isDependencyOnly &&
        model.isLanguageModel &&
        !model.name.lowercased().contains("vision projector") &&
        !model.name.lowercased().contains("mmproj") &&
        !model.name.lowercased().contains("projector")
    }

    private func loadAgentModel() async {
        isLoadingModel = true
        errorMessage = nil
        defer { isLoadingModel = false }
        var candidates = ModelData.allModels()
        if let appleModel = appleFoundationModelIfAvailable() {
            candidates.append(appleModel)
        }
        guard let model = candidates.first(where: {
            $0.name == agentModelName && isAgentModel($0)
        }) else {
            errorMessage = "Model not found"
            return
        }
        do {
            let modelContextCap = LLMBackend.shared.modelMaxContextWindow(for: model)
            let effectiveContext = min(max(1, Int(agentMaxTokens)), modelContextCap)
            let isGranite42 = agentModelName.lowercased().contains("granite-4.2") || agentModelName.lowercased().contains("granite 4.2")
            LLMBackend.shared.enableThinking = isGranite42 ? false : agentEnableThinking
            LLMBackend.shared.maxTokens = min(Int(agentMaxTokens), effectiveContext)
            LLMBackend.shared.contextWindow = effectiveContext
            try await LLMBackend.shared.loadModel(model)
            showSettings = false
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    // MARK: - Composer

    private var composerPanel: some View {
        VStack(spacing: 8) {
            if copiedMessageId != nil {
                Text(settings.localized("message_copied"))
                    .font(.caption)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 4)
                    .background(.regularMaterial, in: Capsule())
                    .transition(.scale.combined(with: .opacity))
            }

            VStack(spacing: 8) {
                ZStack(alignment: .topLeading) {
                    TextField(
                        micTranscriber.isPreparing ? settings.localized("preparing_mic") : settings.localized("type_a_message"),
                        text: $vm.inputText,
                        axis: .vertical
                    )
                    .lineLimit(1...6)
                    .textFieldStyle(.plain)
                    .font(.body)
                    .focused($isComposerFocused)
                    .onSubmit { sendCurrentPrompt() }
                }
                .padding(.horizontal, 10)
                .padding(.top, 10)

                HStack(spacing: 8) {
                    Toggle(isOn: $vm.isWebSearchEnabled) {
                        Label(settings.localized("web_search"), systemImage: "globe")
                    }
                    .toggleStyle(.button)
                    .controlSize(.small)
                    .disabled(vm.isGenerating)

                    Spacer()

                    MacAgentContextUsageRing()

                    Button {
                        if micTranscriber.isRecording {
                            Task { _ = await micTranscriber.stopLive() }
                        } else {
                            Task { await micTranscriber.startLive() }
                        }
                    } label: {
                        Image(systemName: micTranscriber.isPreparing ? "ellipsis" : micTranscriber.isRecording ? "stop.fill" : "mic.fill")
                            .foregroundStyle(micTranscriber.isRecording ? Color.red : Color.primary)
                            .frame(width: 16, height: 16)
                    }
                    .buttonStyle(.bordered)
                    .disabled(vm.isGenerating)

                    Button {
                        isComposerFocused = false
                        if micTranscriber.isRecording {
                            Task { _ = await micTranscriber.stopLive() }
                        }
                        if !vm.isGenerating { sendCurrentPrompt() }
                    } label: {
                        Image(systemName: vm.isGenerating ? "stop.fill" : "arrow.up")
                            .frame(width: 16, height: 16)
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(vm.isGenerating ? .red : .accentColor)
                    .disabled(!vm.isGenerating && vm.inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                }
                .padding(.horizontal, 8)
                .padding(.bottom, 8)
            }
            .background(Color(nsColor: .textBackgroundColor).opacity(0.35), in: RoundedRectangle(cornerRadius: 12))
            .overlay(RoundedRectangle(cornerRadius: 12).stroke(Color(nsColor: .separatorColor)))
            .frame(maxWidth: 860)
        }
        .frame(maxWidth: .infinity)
        .padding(.horizontal, 20)
        .padding(.vertical, 12)
    }

    private func sendCurrentPrompt() {
        let text = vm.inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        if !text.isEmpty {
            vm.inputText = ""
            micTranscriber.liveText = ""
            vm.sendMessage(text)
        }
    }
}

// MARK: - Context ring (same static ring as iOS)

private struct MacAgentContextUsageRing: View {
    var body: some View {
        ZStack {
            Circle()
                .inset(by: 0.75)
                .stroke(Color.secondary.opacity(0.3), lineWidth: 1.5)
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
        .frame(width: 26, height: 26)
    }
}

// MARK: - Tool call cell

private struct MacAgentToolCallCell: View {
    @EnvironmentObject var settings: AppSettings
    let name: String
    let args: String
    let status: AgentMessageItem.ToolStatus
    let result: String?
    let onApprove: (() -> Void)?
    let onDeny: (() -> Void)?

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .top, spacing: 10) {
                Image(systemName: statusIcon)
                    .font(.system(size: 15, weight: .semibold))
                    .foregroundStyle(statusColor)
                    .frame(width: 20)

                VStack(alignment: .leading, spacing: 4) {
                    Text(name)
                        .font(.callout.weight(.semibold))
                        .lineLimit(1)

                    if !args.isEmpty {
                        Text(args)
                            .font(.caption.monospaced())
                            .foregroundStyle(.secondary)
                            .lineLimit(3)
                            .textSelection(.enabled)
                    }

                    if let res = result {
                        Text(res)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .lineLimit(3)
                            .textSelection(.enabled)
                    }
                }

                Spacer(minLength: 8)

                Text(statusText)
                    .font(.caption.weight(.medium))
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }

            if status == .pendingApproval {
                HStack(spacing: 8) {
                    Spacer()
                    Button(role: .cancel) {
                        onDeny?()
                    } label: {
                        Label(settings.localized("agent_mcp_deny"), systemImage: "xmark")
                    }
                    .buttonStyle(.bordered)
                    .tint(ApolloPalette.destructive)

                    Button {
                        onApprove?()
                    } label: {
                        Label(settings.localized("agent_mcp_allow"), systemImage: "checkmark")
                    }
                    .buttonStyle(.borderedProminent)
                    .keyboardShortcut(.defaultAction)
                }
            }
        }
        .padding(12)
        .background(Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: 10, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 10, style: .continuous).stroke(Color(nsColor: .separatorColor)))
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

// MARK: - Map cell

private struct MacAgentMapViewCell: View {
    let label: String
    let latitude: Double
    let longitude: Double

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Label(label, systemImage: "mappin.and.ellipse")
                    .font(.callout.weight(.semibold))
                Spacer()
                Button {
                    openExternalMap()
                } label: {
                    Label(NSLocalizedString("open_maps", comment: ""), systemImage: "arrow.up.right")
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
            }

            MapViewRepresentable(coordinate: CLLocationCoordinate2D(latitude: latitude, longitude: longitude), title: label)
                .frame(height: 240)
                .clipShape(RoundedRectangle(cornerRadius: 8))
                .onTapGesture(count: 2) {
                    openExternalMap()
                }
        }
        .padding(12)
        .background(Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: 10, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 10, style: .continuous).stroke(Color(nsColor: .separatorColor)))
    }

    private func openExternalMap() {
        let query = label.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? ""
        if let url = URL(string: "http://maps.apple.com/?q=\(query)&ll=\(latitude),\(longitude)") {
            NSWorkspace.shared.open(url)
        }
    }
}

// MARK: - MCP configuration (inspector section)

private struct MacMCPConfigurationSection: View {
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
            .toggleStyle(.switch)

            if enabled {
                Text(settings.localized("agent_mcp_description"))
                    .font(.caption)
                    .foregroundStyle(.secondary)

                TextField(settings.localized("agent_mcp_server_url"), text: $url)
                    .textFieldStyle(.roundedBorder)
                    .textContentType(.URL)
                    .autocorrectionDisabled()

                SecureField(settings.localized("agent_mcp_bearer_token"), text: $token)
                    .textFieldStyle(.roundedBorder)

                Text(settings.localized("agent_mcp_approval_notice"))
                    .font(.caption)
                    .foregroundStyle(.secondary)

                if !viewModel.mcpStatus.isEmpty {
                    if viewModel.mcpStatus == "connecting" {
                        HStack(spacing: 6) {
                            ProgressView().controlSize(.small)
                            Text(settings.localized("agent_mcp_connecting"))
                        }
                    } else if viewModel.mcpStatus == "connected" {
                        Text(String(format: settings.localized("agent_mcp_connected_tools"), viewModel.mcpTools.count))
                            .foregroundStyle(.green)
                    } else {
                        Text(String(format: settings.localized("agent_mcp_error"), viewModel.mcpStatus))
                            .foregroundStyle(.red)
                            .textSelection(.enabled)
                    }
                }

                if !viewModel.mcpTools.isEmpty {
                    ForEach(viewModel.mcpTools) { tool in
                        VStack(alignment: .leading, spacing: 2) {
                            Text("• \(tool.name)").font(.caption.weight(.semibold))
                            if !tool.description.isEmpty {
                                Text(tool.description).font(.caption2).foregroundStyle(.secondary)
                            }
                        }
                    }
                }

                HStack {
                    Button {
                        viewModel.connectMCP(MCPSettings(enabled: true, url: url, token: token))
                    } label: {
                        Text(settings.localized("agent_mcp_connect"))
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(url.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)

                    Button(role: .destructive) {
                        viewModel.disconnectMCP()
                        enabled = false
                    } label: {
                        Text(settings.localized("agent_mcp_disconnect"))
                    }
                    .buttonStyle(.bordered)
                }
            }
        }
    }
}
#endif
