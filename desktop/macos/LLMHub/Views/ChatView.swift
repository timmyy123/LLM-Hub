import AppKit
import SwiftUI
import UniformTypeIdentifiers

/// Chat, as in the design: messages with file chips, activity and reasoning pills, Markdown replies with sources and an
/// action row, and the Liquid Glass composer (attach, Web search, Extended thinking, context meter, model, dictate, send).
struct ChatView: View {
    @Environment(AppModel.self) private var app
    @State private var dropTargeted = false

    var body: some View {
        let chat = app.chat
        VStack(spacing: 0) {
            if chat.messages.isEmpty {
                EmptyChat()
            } else {
                MessageList()
            }
            Composer(dropTargeted: dropTargeted)
        }
        .background(Theme.background)
        // Files dropped anywhere on the chat are attached to the next message.
        .dropDestination(for: URL.self) { urls, _ in
            let files = urls.filter(\.isFileURL)
            guard !files.isEmpty else { return false }
            chat.attachFiles(files)
            return true
        } isTargeted: { dropTargeted = $0 }
        .toolbar { ChatToolbar() }
    }
}

/// Star and Share for the open chat (the design's chat header; the window title shows the chat's name).
private struct ChatToolbar: ToolbarContent {
    @Environment(AppModel.self) private var app

    var body: some ToolbarContent {
        let chat = app.chat
        ToolbarItemGroup(placement: .primaryAction) {
            if chat.hasOpenChat {
                Button {
                    chat.toggleStar()
                } label: {
                    Label(L.s(chat.isCurrentStarred ? "Unstar" : "Star"), systemImage: chat.isCurrentStarred ? "star.fill" : "star")
                }
                .help(L.s(chat.isCurrentStarred ? "Unstar" : "Star"))
                .accessibilityIdentifier("StarButton")
            }
            if !chat.messages.isEmpty {
                Button {
                    share()
                } label: {
                    Label(L.s("Share"), systemImage: "square.and.arrow.up")
                }
                .help(L.s("Share"))
                .accessibilityIdentifier("ShareButton")
            }
        }
    }

    /// Saves the chat as Markdown.
    private func share() {
        let chat = app.chat
        let panel = NSSavePanel()
        panel.nameFieldStringValue = chat.exportFileName()
        panel.allowedContentTypes = [UTType(filenameExtension: "md") ?? .plainText]
        panel.canCreateDirectories = true
        let markdown = chat.exportMarkdown()
        panel.begin { response in
            guard response == .OK, let url = panel.url else { return }
            do {
                try markdown.write(to: url, atomically: true, encoding: .utf8)
            } catch {
                Log.error("Share failed", error)
            }
        }
    }
}

private struct EmptyChat: View {
    @Environment(AppModel.self) private var app

    var body: some View {
        VStack(spacing: 14) {
            Spacer()
            Text(L.s("EmptyChatTitle")).font(.system(size: 28, weight: .semibold))
                .accessibilityIdentifier("EmptyTitle")
            if app.chat.models.isEmpty {
                Text(L.s("EmptyNoModel")).foregroundStyle(Theme.textMuted).multilineTextAlignment(.center)
                Button(L.s("GetModel")) { app.section = .settings(.models) }
                    .buttonStyle(.glassProminent)
                    .controlSize(.large)
                    .accessibilityIdentifier("GetModelButton")
            }
            Spacer()
        }
        .frame(maxWidth: 520)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("EmptyState")
    }
}

private struct MessageList: View {
    @Environment(AppModel.self) private var app

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 28) {
                    ForEach(app.chat.messages) { message in
                        MessageRow(message: message, isLast: message === app.chat.messages.last).id(message.id)
                    }
                    Color.clear.frame(height: 1).id("bottom")
                }
                .frame(maxWidth: 760)
                .padding(EdgeInsets(top: 32, leading: 24, bottom: 16, trailing: 24))
                .frame(maxWidth: .infinity)
            }
            .onChange(of: app.chat.messages.last?.content) { proxy.scrollTo("bottom", anchor: .bottom) }
            .onChange(of: app.chat.messages.last?.reasoning) { proxy.scrollTo("bottom", anchor: .bottom) }
            .onChange(of: app.chat.messages.count) { proxy.scrollTo("bottom", anchor: .bottom) }
            .onAppear { proxy.scrollTo("bottom", anchor: .bottom) }
        }
    }
}

private struct MessageRow: View {
    @Environment(AppModel.self) private var app
    let message: MessageItem
    let isLast: Bool
    @State private var copied = false

    var body: some View {
        if message.isUser {
            VStack(alignment: .trailing, spacing: 8) {
                if !message.attachments.isEmpty {
                    ChipFlow(alignment: .trailing) {
                        ForEach(message.attachments) { AttachmentChip(attachment: $0) }
                    }
                }
                Text(message.content)
                    .font(.system(size: 15))
                    .lineSpacing(4)
                    .textSelection(.enabled)
                    .padding(.horizontal, 16)
                    .padding(.vertical, 12)
                    .background(Theme.userBubble, in: .rect(cornerRadius: 16))
                    .accessibilityIdentifier("UserMessage")
            }
            .frame(maxWidth: 608, alignment: .trailing)
            .frame(maxWidth: .infinity, alignment: .trailing)
        } else {
            VStack(alignment: .leading, spacing: 14) {
                pills
                if message.reasoningOpen && !message.reasoning.isEmpty {
                    Text(message.reasoning.trimmingCharacters(in: .whitespacesAndNewlines))
                        .font(.system(size: 13))
                        .lineSpacing(3)
                        .foregroundStyle(Theme.textMuted)
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(.leading, 14)
                        .overlay(alignment: .leading) { Rectangle().fill(Theme.controlBorder).frame(width: 2) }
                        .accessibilityIdentifier("ReasoningText")
                }
                if !message.status.isEmpty {
                    HStack(spacing: 10) {
                        if !message.isDone { ProgressView().controlSize(.small) }
                        Text(message.status).foregroundStyle(message.isDone ? Theme.danger : Theme.textMuted).textSelection(.enabled)
                            .accessibilityIdentifier("MessageStatus")
                    }
                }
                MarkdownView(text: message.content)
                    .frame(minHeight: 1)
                    .accessibilityElement(children: .ignore)
                    .accessibilityLabel(message.content)
                    .accessibilityValue(message.content)
                    .accessibilityIdentifier("AssistantMessage")
                if !message.sources.isEmpty { sources }
                actions
            }
        }
    }

    /// "Searched the web · 5 results" and "Thought for 6s" (click to show the reasoning).
    @ViewBuilder private var pills: some View {
        if !message.activity.isEmpty || !message.reasoning.isEmpty {
            HStack(spacing: 8) {
                if !message.activity.isEmpty {
                    HStack(spacing: 8) {
                        Image(systemName: "globe").font(.system(size: 12))
                        Text(message.activity).font(.system(size: 13))
                    }
                    .foregroundStyle(Theme.textMuted)
                    .padding(.horizontal, 12)
                    .frame(height: 30)
                    .overlay(Capsule().stroke(Theme.controlBorder))
                    .accessibilityElement(children: .combine)
                    .accessibilityIdentifier("ActivityPill")
                }
                if !message.reasoning.isEmpty {
                    Button {
                        message.reasoningOpen.toggle()
                    } label: {
                        HStack(spacing: 8) {
                            if !message.isDone && message.content.isEmpty { ProgressView().controlSize(.mini) }
                            Text(message.thinkingLabel).font(.system(size: 13))
                            Image(systemName: message.reasoningOpen ? "chevron.up" : "chevron.down").font(.system(size: 10, weight: .semibold))
                        }
                        .foregroundStyle(Theme.textMuted)
                        .padding(.horizontal, 12)
                        .frame(height: 30)
                        .contentShape(Capsule())
                        .overlay(Capsule().stroke(Theme.controlBorder))
                    }
                    .buttonStyle(.plain)
                    .accessibilityIdentifier("ThinkingExpander")
                }
            }
        }
    }

    /// Numbered sources with a host chip that opens the page.
    private var sources: some View {
        VStack(alignment: .leading, spacing: 6) {
            ForEach(message.sources) { source in
                HStack(spacing: 10) {
                    Text("\(source.number).").foregroundStyle(Theme.textMuted).monospacedDigit()
                    Text(source.title).lineLimit(1).truncationMode(.tail).frame(maxWidth: 520, alignment: .leading).fixedSize(horizontal: false, vertical: true)
                    if let link = source.link {
                        Link(destination: link) {
                            Text(source.host).font(.system(size: 12)).foregroundStyle(Theme.accent)
                                .padding(.horizontal, 6).padding(.vertical, 1)
                                .background(Theme.tag, in: .rect(cornerRadius: 6))
                        }
                        .help(source.url)
                        .accessibilityIdentifier("SourceLink")
                    }
                }
                .font(.system(size: 13))
            }
        }
    }

    /// Copy, Read aloud and Regenerate under a finished reply, then the model and speed.
    @ViewBuilder private var actions: some View {
        let speaking = app.speech.speakingId == message.id
        HStack(spacing: 4) {
            if message.showActions {
                IconButton(symbol: copied ? "checkmark" : "doc.on.doc", label: L.s(copied ? "Copied" : "Copy"), id: "CopyMessageButton") {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(message.content, forType: .string)
                    copied = true
                    Task {
                        try? await Task.sleep(for: .seconds(1.5))
                        copied = false
                    }
                }
                IconButton(symbol: speaking ? "stop.fill" : "speaker.wave.2", label: L.s(speaking ? "StopReading" : "ReadAloud"), id: "ReadAloudButton") {
                    app.speech.toggleSpeak(id: message.id, text: message.content)
                }
                if isLast && !app.chat.isGenerating {
                    IconButton(symbol: "arrow.clockwise", label: L.s("Regenerate"), id: "RegenerateButton") {
                        app.chat.regenerate(message)
                    }
                }
            }
            if !message.footer.isEmpty {
                Text(message.footer).font(.system(size: 12)).foregroundStyle(Theme.textFaint)
                    .padding(.leading, message.showActions ? 8 : 0)
                    .accessibilityIdentifier("MessageFooter")
            }
            if speaking && !app.speech.status.isEmpty {
                Text(app.speech.status).font(.system(size: 12)).foregroundStyle(Theme.textFaint).padding(.leading, 8)
            }
        }
    }
}

/// Chat-only colours from the design (file chip badges), in light and dark.
private extension Theme {
    static let imageBadge = chatColor(light: 0xDCE7FA, dark: 0x263040)
    static let imageBadgeText = chatColor(light: 0x1D4FAE, dark: 0x9CC2FF)
    static let docBadge = chatColor(light: 0xF8E3D9, dark: 0x3A2F2A)
    static let docBadgeText = chatColor(light: 0xA3431C, dark: 0xF0A98A)
    static let attachmentChip = chatColor(light: 0xF3F2EE, dark: 0x222220)

    static func chatColor(light: UInt32, dark: UInt32) -> Color {
        Color(nsColor: NSColor(name: nil) { appearance in
            let hex = appearance.bestMatch(from: [.darkAqua, .aqua]) == .darkAqua ? dark : light
            return NSColor(srgbRed: CGFloat((hex >> 16) & 0xFF) / 255, green: CGFloat((hex >> 8) & 0xFF) / 255, blue: CGFloat(hex & 0xFF) / 255, alpha: 1)
        })
    }
}

/// A 32pt borderless icon button with a hover highlight (the design's message actions).
private struct IconButton: View {
    let symbol: String
    let label: String
    let id: String
    let action: () -> Void
    @State private var hover = false

    var body: some View {
        Button(action: action) {
            Image(systemName: symbol)
                .font(.system(size: 14))
                .frame(width: 32, height: 32)
                .contentShape(.rect(cornerRadius: 8))
                .background(hover ? Theme.surfaceAlt : .clear, in: .rect(cornerRadius: 8))
        }
        .buttonStyle(.plain)
        .foregroundStyle(Theme.textMuted)
        .onHover { hover = $0 }
        .help(label)
        .accessibilityLabel(label)
        .accessibilityIdentifier(id)
    }
}

/// File chip: coloured badge (PDF, PNG, …), name and detail. Used in sent messages and in the composer.
private struct AttachmentChip: View {
    let attachment: Attachment
    var onRemove: (() -> Void)?

    var body: some View {
        HStack(spacing: 10) {
            Text(attachment.badge)
                .font(.system(size: 10, weight: .semibold))
                .foregroundStyle(attachment.isImage ? Theme.imageBadgeText : Theme.docBadgeText)
                .frame(width: 32, height: 36)
                .background(attachment.isImage ? Theme.imageBadge : Theme.docBadge, in: .rect(cornerRadius: 6))
            VStack(alignment: .leading, spacing: 0) {
                Text(attachment.name).font(.system(size: 13)).lineLimit(1).truncationMode(.middle)
                Text(attachment.detail).font(.system(size: 12)).foregroundStyle(Theme.textMuted)
            }
            .frame(maxWidth: 200, alignment: .leading)
            .fixedSize(horizontal: true, vertical: false)
        }
        .padding(EdgeInsets(top: 8, leading: 8, bottom: 8, trailing: 12))
        .background(Theme.attachmentChip, in: .rect(cornerRadius: 10))
        .overlay(RoundedRectangle(cornerRadius: 10).stroke(Theme.controlBorder.opacity(0.8)))
        .overlay(alignment: .topTrailing) {
            if let onRemove {
                Button(action: onRemove) {
                    Image(systemName: "xmark").font(.system(size: 8, weight: .bold)).frame(width: 20, height: 20)
                }
                .buttonStyle(.glass)
                .buttonBorderShape(.circle)
                .controlSize(.mini)
                .offset(x: 7, y: -7)
                .help(L.s("Delete"))
                .accessibilityLabel(L.s("Delete"))
                .accessibilityIdentifier("RemoveAttachmentButton")
            }
        }
        .help(attachment.name)
        .accessibilityElement(children: .contain)
        .accessibilityLabel(attachment.name)
        .accessibilityIdentifier("AttachmentChip")
    }
}

/// Lays chips out in rows, wrapping when a row is full.
private struct ChipFlow: Layout {
    var alignment: HorizontalAlignment = .leading
    var spacing: CGFloat = 8

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let rows = rows(subviews, width: proposal.width ?? .infinity)
        let width = rows.map(\.width).max() ?? 0
        let height = rows.map(\.height).reduce(0, +) + spacing * CGFloat(max(0, rows.count - 1))
        return CGSize(width: proposal.width.map { min($0, width) } ?? width, height: height)
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        var y = bounds.minY
        for row in rows(subviews, width: bounds.width) {
            var x = alignment == .trailing ? bounds.maxX - row.width : bounds.minX
            for index in row.items {
                let size = subviews[index].sizeThatFits(.unspecified)
                subviews[index].place(at: CGPoint(x: x, y: y), proposal: ProposedViewSize(size))
                x += size.width + spacing
            }
            y += row.height + spacing
        }
    }

    private func rows(_ subviews: Subviews, width: CGFloat) -> [(items: [Int], width: CGFloat, height: CGFloat)] {
        var rows: [(items: [Int], width: CGFloat, height: CGFloat)] = []
        var current: (items: [Int], width: CGFloat, height: CGFloat) = ([], 0, 0)
        for (i, view) in subviews.enumerated() {
            let size = view.sizeThatFits(.unspecified)
            let needed = current.items.isEmpty ? size.width : current.width + spacing + size.width
            if !current.items.isEmpty && needed > width {
                rows.append(current)
                current = ([i], size.width, size.height)
            } else {
                current = (current.items + [i], needed, max(current.height, size.height))
            }
        }
        if !current.items.isEmpty { rows.append(current) }
        return rows
    }
}

/// A composer switch in Liquid Glass: accent-tinted glass and text when on.
private struct GlassSwitchStyle: ToggleStyle {
    func makeBody(configuration: Configuration) -> some View {
        Button {
            configuration.isOn.toggle()
        } label: {
            configuration.label
                .font(.system(size: 13))
                .foregroundStyle(configuration.isOn ? Theme.accent : Theme.textMuted)
                .padding(.horizontal, 12)
                .frame(height: 36)
                .contentShape(.rect(cornerRadius: 10))
        }
        .buttonStyle(.plain)
        .glassEffect(configuration.isOn ? .regular.tint(Theme.accent.opacity(0.22)).interactive() : .regular.interactive(), in: .rect(cornerRadius: 10))
        .accessibilityValue(configuration.isOn ? "1" : "0")
        .accessibilityAddTraits(configuration.isOn ? .isSelected : [])
    }
}

private struct Composer: View {
    @Environment(AppModel.self) private var app
    let dropTargeted: Bool
    @FocusState private var focused: Bool

    var body: some View {
        @Bindable var chat = app.chat
        let speech = app.speech
        VStack(spacing: 8) {
            VStack(alignment: .leading, spacing: 10) {
                if !chat.pendingAttachments.isEmpty {
                    ChipFlow {
                        ForEach(chat.pendingAttachments) { a in
                            AttachmentChip(attachment: a) { chat.removeAttachment(a) }
                        }
                    }
                    .padding(.top, 4)
                    .padding(.trailing, 8)
                }
                if !chat.attachError.isEmpty {
                    Text(chat.attachError).font(.system(size: 12)).foregroundStyle(Theme.danger).fixedSize(horizontal: false, vertical: true)
                        .accessibilityIdentifier("AttachError")
                } else if !speech.status.isEmpty && (speech.isRecording || speech.isTranscribing) {
                    Text(speech.status).font(.system(size: 12)).foregroundStyle(Theme.textMuted)
                }
                TextField(L.s("ComposerPlaceholder"), text: $chat.draft, axis: .vertical)
                    .textFieldStyle(.plain)
                    .font(.system(size: 15))
                    .lineLimit(1...10)
                    .padding(.horizontal, 4)
                    .padding(.vertical, 2)
                    .focused($focused)
                    .onKeyPress(.return, phases: .down) { press in
                        if press.modifiers.contains(.shift) { return .ignored }
                        chat.send()
                        return .handled
                    }
                    .accessibilityIdentifier("Composer")
                controls
            }
            .padding(EdgeInsets(top: 14, leading: 14, bottom: 10, trailing: 14))
            .glassEffect(.regular, in: .rect(cornerRadius: 18))
            .overlay {
                if dropTargeted {
                    RoundedRectangle(cornerRadius: 18).stroke(Theme.accent, style: StrokeStyle(lineWidth: 2, dash: [6, 4]))
                }
            }
            .frame(maxWidth: 760)

            Text(L.s("PrivacyNote"))
                .font(.system(size: 12))
                .foregroundStyle(Theme.textFaint)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 760)
                .accessibilityIdentifier("PrivacyNote")
        }
        .padding(EdgeInsets(top: 8, leading: 24, bottom: 16, trailing: 24))
        .onAppear { focused = true }
    }

    private var controls: some View {
        @Bindable var chat = app.chat
        let speech = app.speech
        return HStack(spacing: 6) {
            Button(action: pickFiles) {
                Image(systemName: "paperclip").font(.system(size: 15)).frame(width: 36, height: 36).contentShape(.rect(cornerRadius: 10))
            }
            .buttonStyle(.plain)
            .foregroundStyle(Theme.textMuted)
            .glassEffect(.regular.interactive(), in: .rect(cornerRadius: 10))
            .help(L.s("Attach"))
            .accessibilityLabel(L.s("Attach"))
            .accessibilityIdentifier("AttachButton")

            Toggle(isOn: $chat.webSearchEnabled) {
                Label(L.s("WebSearch"), systemImage: "globe").labelStyle(.titleAndIcon)
            }
            .toggleStyle(GlassSwitchStyle())
            .accessibilityIdentifier("WebSearchToggle")

            Toggle(isOn: $chat.thinkingEnabled) { Text(L.s("ExtendedThinking")) }
                .toggleStyle(GlassSwitchStyle())
                .accessibilityIdentifier("ThinkingToggle")

            if chat.mtpVisible {
                Toggle(isOn: $chat.mtpEnabled) { Text(L.s("Mtp")) }
                    .toggleStyle(GlassSwitchStyle())
                    .help(L.s("MtpHelp"))
                    .accessibilityIdentifier("MtpToggle")
            }

            Spacer(minLength: 4)

            if !chat.contextText.isEmpty {
                Text(chat.contextText)
                    .font(.system(size: 12))
                    .foregroundStyle(Theme.textFaint)
                    .lineLimit(1)
                    .fixedSize()
                    .padding(.trailing, 2)
                    .accessibilityIdentifier("ContextMeter")
            }

            Picker("", selection: $chat.selectedModelId) {
                if chat.selectedModelId == nil { Text(L.s("ChooseModel")).tag(String?.none) }
                ForEach(chat.models) { model in
                    Text(model.displayName).tag(Optional(model.id))
                }
            }
            .labelsHidden()
            .pickerStyle(.menu)
            .buttonStyle(.borderless)
            .fixedSize()
            .frame(maxWidth: 260, alignment: .trailing)
            .accessibilityIdentifier("ModelPicker")

            Button(action: dictate) {
                Group {
                    if speech.isTranscribing {
                        ProgressView().controlSize(.small)
                    } else {
                        Image(systemName: speech.isRecording ? "stop.circle.fill" : "mic").font(.system(size: 15))
                    }
                }
                .frame(width: 36, height: 36)
                .contentShape(.rect(cornerRadius: 10))
            }
            .buttonStyle(.plain)
            .foregroundStyle(speech.isRecording ? Theme.danger : Theme.textMuted)
            .glassEffect(speech.isRecording ? .regular.tint(Theme.danger.opacity(0.25)).interactive() : .regular.interactive(), in: .rect(cornerRadius: 10))
            .disabled(speech.isTranscribing)
            .help(speech.isRecording ? L.s("Listening") : speech.isTranscribing ? L.s("Transcribing") : L.s("Dictate"))
            .accessibilityLabel(speech.isRecording ? L.s("Listening") : speech.isTranscribing ? L.s("Transcribing") : L.s("Dictate"))
            .accessibilityIdentifier("DictateButton")

            if chat.isGenerating {
                Button { chat.stop() } label: { Image(systemName: "stop.fill").frame(width: 22, height: 22) }
                    .buttonStyle(.glass)
                    .buttonBorderShape(.circle)
                    .help(L.s("Stop"))
                    .accessibilityLabel(L.s("Stop"))
                    .accessibilityIdentifier("StopButton")
            } else {
                Button { chat.send() } label: { Image(systemName: "arrow.up").fontWeight(.semibold).frame(width: 22, height: 22) }
                    .buttonStyle(.glassProminent)
                    .buttonBorderShape(.circle)
                    .tint(Theme.accent)
                    .disabled(!chat.canSend)
                    .help(L.s("Send"))
                    .accessibilityLabel(L.s("Send"))
                    .accessibilityIdentifier("SendButton")
            }
        }
    }

    private func pickFiles() {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = true
        panel.canChooseDirectories = false
        panel.allowedContentTypes = Attachments.extensions.compactMap { UTType(filenameExtension: $0) }
        panel.begin { response in
            guard response == .OK else { return }
            app.chat.attachFiles(panel.urls)
        }
    }

    /// First click records; the second stops, transcribes with Whisper and adds the text to the draft.
    private func dictate() {
        Task {
            let stopping = app.speech.isRecording
            await app.chat.toggleDictation()
            if stopping { focused = true }
        }
    }
}
