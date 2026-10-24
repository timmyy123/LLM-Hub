import SwiftUI

/// The right side of a session: Diff (changed files and their diff), Terminal, Files and Preview, then Revert and Commit.
struct AgentSidePanel: View {
    @Environment(AppModel.self) private var app
    let session: AgentSession

    var body: some View {
        @Bindable var vm = app.agent.ui!
        VStack(spacing: 0) {
            Picker("", selection: $vm.sideTab) {
                Text(L.s("AgentDiff")).tag(AgentViewModel.SideTab.diff)
                Text(L.s("AgentTerminal")).tag(AgentViewModel.SideTab.terminal)
                Text(L.s("AgentFiles")).tag(AgentViewModel.SideTab.files)
                Text(L.s("AgentPreview")).tag(AgentViewModel.SideTab.preview)
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .padding(.horizontal, 12)
            .frame(height: 52)
            .accessibilityIdentifier("AgentSideTabs")
            Rectangle().fill(Theme.divider).frame(height: 1)

            Group {
                switch vm.sideTab {
                case .diff: diff
                case .terminal: terminal
                case .files: files
                case .preview: preview
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)

            commitBar
        }
    }

    // MARK: Diff

    private var diff: some View {
        let vm = app.agent.ui!
        return VStack(alignment: .leading, spacing: 8) {
            if session.changes.isEmpty {
                Text(L.s("AgentNoChanges")).font(.system(size: 13)).foregroundStyle(Theme.textMuted).padding(4)
            } else {
                ScrollView {
                    VStack(spacing: 2) {
                        ForEach(session.changes) { change in
                            changeRow(change, selected: change.path == vm.selectedChange?.path)
                        }
                    }
                }
                .frame(maxHeight: min(CGFloat(session.changes.count) * 36, 200))
                .scrollBounceBehavior(.basedOnSize)
                if let change = vm.selectedChange {
                    ScrollView([.horizontal, .vertical]) {
                        DiffLines(lines: change.diff, fontSize: 12, horizontalPadding: 12)
                            .padding(.vertical, 10)
                            .frame(minWidth: 334, alignment: .topLeading)
                    }
                    .background(Theme.surfaceAlt, in: RoundedRectangle(cornerRadius: 10))
                    .overlay(RoundedRectangle(cornerRadius: 10).strokeBorder(Theme.divider))
                    .accessibilityIdentifier("AgentDiffView")
                }
            }
        }
        .padding(12)
    }

    private func changeRow(_ change: FileChange, selected: Bool) -> some View {
        Button {
            app.agent.ui.selectedChangePath = change.path
        } label: {
            HStack(spacing: 8) {
                Text(String(change.kind))
                    .foregroundStyle(change.kind == "A" ? Theme.success : change.kind == "D" ? Theme.danger : Theme.warning)
                Text(change.path).lineLimit(1).truncationMode(.middle)
                    .accessibilityIdentifier("ChangedFile")
                Spacer(minLength: 0)
                MetaText(meta: change.stats)
            }
            .font(.system(size: 12.5, design: .monospaced))
            .padding(.horizontal, 10)
            .frame(height: 34)
            .contentShape(Rectangle())
            .background(selected ? Color.primary.opacity(0.07) : .clear, in: RoundedRectangle(cornerRadius: 8))
        }
        .buttonStyle(.plain)
    }

    // MARK: Terminal, Files, Preview

    private var terminal: some View {
        ScrollView {
            Text(session.terminal.isEmpty ? " " : session.terminal)
                .font(.system(size: 12, design: .monospaced))
                .foregroundStyle(Theme.textMuted)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.horizontal, 14)
                .padding(.vertical, 10)
                .accessibilityIdentifier("AgentTerminalText")
        }
        .defaultScrollAnchor(.bottom)
    }

    private var files: some View {
        let vm = app.agent.ui!
        return List(vm.files, id: \.self) { file in
            Button { vm.open(file: file) } label: {
                Text(file).font(.system(size: 12, design: .monospaced)).lineLimit(1).truncationMode(.middle)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
        }
        .scrollContentBackground(.hidden)
        .onAppear { vm.loadFiles() }
    }

    private var preview: some View {
        let file = app.agent.ui.previewFile
        return VStack(alignment: .leading, spacing: 12) {
            Text(file.map { ($0 as NSString).abbreviatingWithTildeInPath } ?? L.s("AgentNoPreview"))
                .font(.system(size: 13))
                .foregroundStyle(Theme.textMuted)
            Button(L.s("AgentPreview")) {
                if let file { NSWorkspace.shared.open(URL(fileURLWithPath: file)) }
            }
            .buttonStyle(.glass)
            .disabled(file == nil)
        }
        .padding(16)
    }

    // MARK: Revert / Commit

    private var commitBar: some View {
        @Bindable var vm = app.agent.ui!
        let canChange = !session.changes.isEmpty && !session.isWorking
        return VStack(alignment: .leading, spacing: 10) {
            TextField(L.s("AgentCommitMessage"), text: $vm.commitMessage)
                .textFieldStyle(.roundedBorder)
                .font(.system(size: 13))
                .onSubmit { if canChange { vm.commit() } }
                .accessibilityIdentifier("CommitMessage")
            GlassEffectContainer(spacing: 8) {
                HStack(spacing: 8) {
                    Button { vm.revert() } label: { Text(L.s("AgentRevert")).frame(maxWidth: .infinity) }
                        .buttonStyle(.glass)
                        .accessibilityIdentifier("RevertButton")
                    Button { vm.commit() } label: { Text(L.s("AgentCommit")).fontWeight(.medium).frame(maxWidth: .infinity) }
                        .buttonStyle(.glassProminent)
                        .accessibilityIdentifier("CommitButton")
                }
                .controlSize(.large)
                .disabled(!canChange)
            }
            Text(L.s("AgentFooter"))
                .font(.system(size: 12))
                .foregroundStyle(Theme.textFaint)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(12)
        .overlay(alignment: .top) { Rectangle().fill(Theme.divider).frame(height: 1) }
    }
}

/// A session in the sidebar: status dot, title and "Working" / "Needs you".
struct AgentSessionRow: View {
    let session: AgentSession

    var body: some View {
        HStack(spacing: 8) {
            Circle().fill(dot).frame(width: 7, height: 7)
            Text(session.title).lineLimit(1).accessibilityIdentifier("AgentSessionTitle")
            Spacer(minLength: 4)
            if !session.statusText.isEmpty {
                Text(session.statusText).font(.system(size: 11)).foregroundStyle(dot).lineLimit(1)
            }
        }
    }

    private var dot: Color {
        switch session.status {
        case .working: return Theme.accent
        case .needsYou: return Theme.warning
        case .error: return Theme.danger
        case .idle: return Theme.textFaint
        }
    }
}
