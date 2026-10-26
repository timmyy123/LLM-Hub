import SwiftUI

/// The Agent tab (design: Agent.dc.html): the session transcript and composer, with the Diff / Terminal / Files /
/// Preview panel on the right. Before any session: pick a project folder, and install Claude Code if needed.
struct AgentView: View {
    @Environment(AppModel.self) private var app

    var body: some View {
        let vm = app.agent.ui!
        Group {
            if let session = vm.selected {
                HStack(spacing: 0) {
                    SessionPane(session: session)
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                    Rectangle().fill(Theme.divider).frame(width: 1)
                    AgentSidePanel(session: session)
                        .frame(width: 360)
                }
            } else {
                AgentEmptyState()
            }
        }
    }
}

// MARK: Empty state

private struct AgentEmptyState: View {
    @Environment(AppModel.self) private var app

    var body: some View {
        let vm = app.agent.ui!
        VStack(spacing: 16) {
            Text(L.s("AgentEmptyTitle"))
                .font(.system(size: 26, weight: .semibold))
                .multilineTextAlignment(.center)
                .accessibilityIdentifier("AgentEmptyTitle")
            Text(L.s("AgentEmptyBody"))
                .foregroundStyle(Theme.textMuted)
                .multilineTextAlignment(.center)
            Button(L.s("AgentPickFolder")) { vm.pickFolderAndStart() }
                .buttonStyle(.glassProminent)
                .controlSize(.large)
                .accessibilityIdentifier("PickFolderButton")
            if !vm.notice.isEmpty {
                Text(vm.notice).font(.system(size: 12)).foregroundStyle(Theme.textMuted)
            }
            EngineCard().padding(.top, 8)
        }
        .frame(maxWidth: 560)
        .padding(24)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

/// The engine for new sessions (Claude Code, or OpenClaw installed by the app), and its install button when it is missing.
private struct EngineCard: View {
    @Environment(AppModel.self) private var app

    var body: some View {
        @Bindable var vm = app.agent.ui!
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 12) {
                Text(L.s("AgentEngineTitle")).fontWeight(.semibold)
                Spacer(minLength: 0)
                Picker("", selection: $vm.engineChoice) {
                    ForEach(vm.engines, id: \.mode) { Text($0.label).tag($0.mode) }
                }
                .labelsHidden()
                .fixedSize()
                .accessibilityIdentifier("AgentEnginePicker")
            }
            if !vm.isEngineInstalled {
                Text(vm.installHelp).font(.system(size: 13)).foregroundStyle(Theme.textMuted)
                    .fixedSize(horizontal: false, vertical: true)
                HStack(spacing: 10) {
                    Button(vm.installLabel) { Task { await vm.installEngine() } }
                        .buttonStyle(.glass)
                        .disabled(vm.isEngineInstalling)
                        .accessibilityIdentifier("InstallClaudeButton")
                    if vm.isEngineInstalling { ProgressView().controlSize(.small) }
                    Text(vm.installStatus).font(.system(size: 12)).foregroundStyle(Theme.textMuted)
                        .fixedSize(horizontal: false, vertical: true)
                        .accessibilityIdentifier("AgentInstallStatus")
                }
            }
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .glassEffect(.regular, in: .rect(cornerRadius: 16))
    }
}

// MARK: Session

private struct SessionPane: View {
    @Environment(AppModel.self) private var app
    let session: AgentSession

    var body: some View {
        VStack(spacing: 0) {
            header
            Rectangle().fill(Theme.divider).frame(height: 1)
            Transcript(session: session)
            AgentComposer(session: session)
        }
    }

    private var header: some View {
        HStack(spacing: 10) {
            Text(session.title)
                .font(.system(size: 15, weight: .medium))
                .lineLimit(1)
                .accessibilityIdentifier("AgentSessionTitle")
            Chip(symbol: "folder", text: (session.folder as NSString).abbreviatingWithTildeInPath)
            if !session.branch.isEmpty {
                Chip(symbol: "arrow.triangle.branch", text: session.branch)
            }
            Spacer(minLength: 0)
            if session.isWorking {
                ProgressView().controlSize(.small)
                Text(session.statusText).font(.system(size: 12)).foregroundStyle(session.status == .needsYou ? Theme.warning : Theme.textMuted)
            }
        }
        .padding(.horizontal, 24)
        .frame(height: 52)
    }
}

private struct Chip: View {
    let symbol: String
    let text: String

    var body: some View {
        HStack(spacing: 6) {
            Image(systemName: symbol).font(.system(size: 10)).foregroundStyle(Theme.textMuted)
            Text(text).font(.system(size: 12, design: .monospaced)).foregroundStyle(Theme.textMuted).lineLimit(1).truncationMode(.middle)
        }
        .padding(.horizontal, 8)
        .frame(height: 24)
        .background(.fill.tertiary, in: RoundedRectangle(cornerRadius: 6))
    }
}

private struct Transcript: View {
    let session: AgentSession

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 16) {
                    ForEach(session.items) { item in
                        ItemRow(session: session, item: item).id(item.id)
                    }
                    if let pending = session.pending {
                        PermissionCard(session: session, request: pending).id(pending.id)
                    }
                    Color.clear.frame(height: 1).id("bottom")
                }
                .frame(maxWidth: 720)
                .padding(24)
                .frame(maxWidth: .infinity)
            }
            .onChange(of: session.items.count) { proxy.scrollTo("bottom", anchor: .bottom) }
            .onChange(of: session.pending?.id) { proxy.scrollTo("bottom", anchor: .bottom) }
            .onChange(of: session.id, initial: true) { proxy.scrollTo("bottom", anchor: .bottom) }
        }
    }
}

private struct ItemRow: View {
    let session: AgentSession
    let item: AgentItem

    var body: some View {
        switch item.kind {
        case .user:
            HStack {
                Spacer(minLength: 80)
                Text(item.title)
                    .font(.system(size: 15))
                    .lineSpacing(3)
                    .textSelection(.enabled)
                    .padding(.horizontal, 16)
                    .padding(.vertical, 12)
                    .glassEffect(.regular.tint(.accentColor.opacity(0.22)), in: .rect(cornerRadius: 16))
                    .accessibilityIdentifier("AgentUserMessage")
            }
        case .assistant:
            MarkdownView(text: item.title)
                .frame(maxWidth: .infinity, alignment: .leading)
                .accessibilityIdentifier("AgentMessage")
        case .error:
            Text(item.title).foregroundStyle(Theme.danger).textSelection(.enabled)
        case .tasks:
            TasksCard(tasks: session.tasks)
        default:
            StepCard(item: item)
        }
    }
}

private struct TasksCard: View {
    let tasks: [AgentTask]

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(L.s("AgentTasks")).font(.system(size: 12, weight: .medium)).foregroundStyle(Theme.textMuted)
            ForEach(tasks) { task in
                HStack(alignment: .firstTextBaseline, spacing: 10) {
                    icon(task.status)
                    Text(task.text)
                        .font(.system(size: 13))
                        .strikethrough(task.status == "completed")
                        .foregroundStyle(task.status == "completed" ? Theme.textMuted : .primary)
                }
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .glassEffect(.regular, in: .rect(cornerRadius: 12))
        .accessibilityIdentifier("AgentTasks")
    }

    @ViewBuilder private func icon(_ status: String) -> some View {
        switch status {
        case "completed": Image(systemName: "checkmark.circle.fill").foregroundStyle(Theme.success)
        case "in_progress": Image(systemName: "circle").fontWeight(.bold).foregroundStyle(Theme.accent)
        default: Image(systemName: "circle").foregroundStyle(Theme.textFaint)
        }
    }
}

/// A tool step: label, target and meta; the edit's diff or the command's output below.
private struct StepCard: View {
    let item: AgentItem

    var body: some View {
        let expanded = !item.diff.isEmpty || !item.output.isEmpty
        VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: 10) {
                Text(item.detail).font(.system(size: 13, weight: .medium)).foregroundStyle(Theme.textMuted)
                Text(item.title)
                    .font(.system(size: 13, design: .monospaced))
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .textSelection(.enabled)
                Spacer(minLength: 0)
                if item.failed { Image(systemName: "exclamationmark.triangle").font(.system(size: 11)).foregroundStyle(Theme.danger) }
                MetaText(meta: item.meta)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 10)
            .background(expanded ? AnyShapeStyle(.fill.quaternary) : AnyShapeStyle(.clear))
            if !item.diff.isEmpty {
                DiffLines(lines: item.diff, fontSize: 12.5, horizontalPadding: 14)
                    .padding(.vertical, 8)
            }
            if !item.output.isEmpty {
                Text(item.output.trimmingCharacters(in: .newlines))
                    .font(.system(size: 12, design: .monospaced))
                    .foregroundStyle(item.failed ? Theme.danger : Theme.textMuted)
                    .lineLimit(12)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 10)
            }
        }
        .clipShape(RoundedRectangle(cornerRadius: 10))
        .overlay(RoundedRectangle(cornerRadius: 10).strokeBorder(Theme.controlBorder.opacity(0.8)))
        .accessibilityElement(children: .combine)
        .accessibilityIdentifier("AgentStep")
    }
}

/// "+24 −2" in the design's green and red.
struct MetaText: View {
    let meta: String

    var body: some View {
        HStack(spacing: 4) {
            ForEach(Array(meta.split(separator: " ").enumerated()), id: \.offset) { _, part in
                Text(part)
                    .font(.system(size: 12))
                    .foregroundStyle(part.hasPrefix("+") ? Theme.success : part.hasPrefix("−") ? Theme.danger : Theme.textFaint)
            }
        }
        .fixedSize()
    }
}

/// Diff lines: green for added, red for removed, plain context, faint hunk headers.
struct DiffLines: View {
    let lines: [DiffLine]
    var fontSize: CGFloat = 12
    var horizontalPadding: CGFloat = 12

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            ForEach(Array(lines.enumerated()), id: \.offset) { _, line in
                Text(line.kind == "@" ? line.text : "\(line.kind) \(line.text)")
                    .font(.system(size: fontSize, design: .monospaced))
                    .foregroundStyle(foreground(line.kind))
                    .lineLimit(1)
                    .fixedSize(horizontal: true, vertical: false)
                    .padding(.horizontal, horizontalPadding)
                    .padding(.vertical, 2)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(background(line.kind))
            }
        }
        .textSelection(.enabled)
    }

    private func foreground(_ kind: Character) -> Color {
        switch kind {
        case "+": return Theme.diffAddedText
        case "-": return Theme.diffRemovedText
        case "@": return Theme.textFaint
        default: return Theme.textMuted
        }
    }

    private func background(_ kind: Character) -> Color {
        switch kind {
        case "+": return Theme.diffAddedBackground
        case "-": return Theme.diffRemovedBackground
        default: return .clear
        }
    }
}

private struct PermissionCard: View {
    @Environment(AppModel.self) private var app
    let session: AgentSession
    let request: PermissionRequest

    var body: some View {
        let vm = app.agent.ui!
        VStack(alignment: .leading, spacing: 10) {
            Text(request.question).fontWeight(.semibold).accessibilityIdentifier("PermissionQuestion")
            Text(request.summary)
                .font(.system(size: 13, design: .monospaced))
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.horizontal, 10)
                .padding(.vertical, 8)
                .background(.fill.tertiary, in: RoundedRectangle(cornerRadius: 8))
            HStack(spacing: 8) {
                Button(L.s("AgentAllow")) { vm.allow() }
                    .buttonStyle(.glassProminent)
                    .keyboardShortcut(.defaultAction)
                    .accessibilityIdentifier("AllowButton")
                Button(L.s("AgentAllowSession")) { vm.allowForSession() }
                    .buttonStyle(.glass)
                    .accessibilityIdentifier("AllowSessionButton")
                Button(L.s("AgentDeny")) { vm.deny() }
                    .buttonStyle(.glass)
                    .accessibilityIdentifier("DenyButton")
            }
            .controlSize(.large)
        }
        .padding(14)
        .glassEffect(.regular.tint(Theme.warning.opacity(0.12)), in: .rect(cornerRadius: 12))
        .overlay(RoundedRectangle(cornerRadius: 12).strokeBorder(Theme.warning.opacity(0.45)))
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("PermissionCard")
    }
}

// MARK: Composer

private struct AgentComposer: View {
    @Environment(AppModel.self) private var app
    let session: AgentSession
    @FocusState private var focused: Bool

    var body: some View {
        @Bindable var vm = app.agent.ui!
        VStack(spacing: 8) {
            if !vm.notice.isEmpty {
                Text(vm.notice)
                    .font(.system(size: 12))
                    .foregroundStyle(Theme.textMuted)
                    .frame(maxWidth: 720, alignment: .leading)
                    .accessibilityIdentifier("AgentNotice")
            }
            VStack(spacing: 10) {
                TextField(L.s("AgentPlaceholder"), text: $vm.draft, axis: .vertical)
                    .textFieldStyle(.plain)
                    .font(.system(size: 15))
                    .lineLimit(2...8)
                    .focused($focused)
                    .onKeyPress(.return, phases: .down) { press in
                        if press.modifiers.contains(.shift) { return .ignored }
                        vm.send()
                        return .handled
                    }
                    .padding(.horizontal, 4)
                    .accessibilityIdentifier("AgentComposer")
                HStack(spacing: 8) {
                    Picker("", selection: $vm.mode) {
                        ForEach(vm.modes, id: \.mode) { Text($0.label).tag($0.mode) }
                    }
                    .labelsHidden()
                    .fixedSize()
                    .disabled(session.isWorking)
                    .accessibilityIdentifier("AgentMode")
                    Spacer()
                    Picker("", selection: $vm.selectedModelId) {
                        if vm.selectedModelId == nil { Text(L.s("AgentModel")).tag(String?.none) }
                        ForEach(vm.models) { Text($0.displayName).tag(Optional($0.id)) }
                    }
                    .labelsHidden()
                    .frame(maxWidth: 260)
                    .disabled(session.isWorking)
                    .accessibilityIdentifier("AgentModelPicker")
                    if session.isWorking {
                        Button { vm.stop() } label: { Image(systemName: "stop.fill").frame(width: 22, height: 22) }
                            .buttonStyle(.glass)
                            .buttonBorderShape(.circle)
                            .help(L.s("Stop"))
                            .accessibilityLabel(L.s("Stop"))
                            .accessibilityIdentifier("AgentStop")
                    } else {
                        Button { vm.send() } label: { Image(systemName: "arrow.up").fontWeight(.semibold).frame(width: 22, height: 22) }
                            .buttonStyle(.glassProminent)
                            .buttonBorderShape(.circle)
                            .disabled(vm.draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || vm.selectedModelId == nil)
                            .help(L.s("Send"))
                            .accessibilityLabel(L.s("Send"))
                            .accessibilityIdentifier("AgentSend")
                    }
                }
            }
            .padding(EdgeInsets(top: 14, leading: 14, bottom: 10, trailing: 10))
            .glassEffect(.regular, in: .rect(cornerRadius: 20))
            .frame(maxWidth: 720)
        }
        .padding(EdgeInsets(top: 8, leading: 24, bottom: 20, trailing: 24))
        .onAppear { focused = true }
    }
}
