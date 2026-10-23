import SwiftUI

struct ChatView: View {
    @Environment(AppModel.self) private var app

    var body: some View {
        let chat = app.chat
        VStack(spacing: 0) {
            if chat.messages.isEmpty {
                EmptyChat()
            } else {
                MessageList()
            }
            Composer()
        }
    }
}

private struct EmptyChat: View {
    @Environment(AppModel.self) private var app

    var body: some View {
        VStack(spacing: 14) {
            Spacer()
            Text(L.s("EmptyChatTitle")).font(.system(size: 28, weight: .semibold))
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
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("EmptyState")
    }
}

private struct MessageList: View {
    @Environment(AppModel.self) private var app

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 24) {
                    ForEach(app.chat.messages) { message in
                        MessageRow(message: message).id(message.id)
                    }
                    Color.clear.frame(height: 1).id("bottom")
                }
                .frame(maxWidth: 760)
                .padding(.horizontal, 24)
                .padding(.vertical, 28)
                .frame(maxWidth: .infinity)
            }
            .onChange(of: app.chat.messages.last?.content) { proxy.scrollTo("bottom", anchor: .bottom) }
            .onChange(of: app.chat.messages.count) { proxy.scrollTo("bottom", anchor: .bottom) }
            .onAppear { proxy.scrollTo("bottom", anchor: .bottom) }
        }
    }
}

private struct MessageRow: View {
    let message: MessageItem

    var body: some View {
        if message.isUser {
            HStack {
                Spacer(minLength: 120)
                Text(message.content)
                    .font(.system(size: 15))
                    .textSelection(.enabled)
                    .padding(.horizontal, 16)
                    .padding(.vertical, 12)
                    .glassEffect(.regular.tint(.accentColor.opacity(0.22)), in: .rect(cornerRadius: 18))
                    .accessibilityIdentifier("UserMessage")
            }
        } else {
            VStack(alignment: .leading, spacing: 10) {
                if !message.reasoning.isEmpty {
                    DisclosureGroup(L.s("Thinking")) {
                        Text(message.reasoning)
                            .font(.system(size: 13))
                            .foregroundStyle(Theme.textMuted)
                            .textSelection(.enabled)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                    .accessibilityIdentifier("ThinkingExpander")
                }
                if !message.status.isEmpty {
                    HStack(spacing: 10) {
                        ProgressView().controlSize(.small)
                        Text(message.status).foregroundStyle(Theme.textMuted).accessibilityIdentifier("MessageStatus")
                    }
                }
                Text(Self.markdown(message.content))
                    .font(.system(size: 15))
                    .lineSpacing(4)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .accessibilityIdentifier("AssistantMessage")
                    .accessibilityValue(message.content)
                if !message.footer.isEmpty {
                    Text(message.footer).font(.system(size: 12)).foregroundStyle(.secondary)
                        .accessibilityIdentifier("MessageFooter")
                }
            }
        }
    }

    private static func markdown(_ text: String) -> AttributedString {
        (try? AttributedString(markdown: text, options: .init(interpretedSyntax: .inlineOnlyPreservingWhitespace))) ?? AttributedString(text)
    }
}

private struct Composer: View {
    @Environment(AppModel.self) private var app
    @FocusState private var focused: Bool

    var body: some View {
        @Bindable var chat = app.chat
        VStack(spacing: 8) {
            TextField(L.s("ComposerPlaceholder"), text: $chat.draft, axis: .vertical)
                .textFieldStyle(.plain)
                .font(.system(size: 15))
                .lineLimit(1...8)
                .focused($focused)
                .onKeyPress(.return, phases: .down) { press in
                    if press.modifiers.contains(.shift) { return .ignored }
                    chat.send()
                    return .handled
                }
                .accessibilityIdentifier("Composer")
            HStack(spacing: 8) {
                Spacer()
                if chat.mtpVisible {
                    Toggle(L.s("Mtp"), isOn: $chat.mtpEnabled)
                        .toggleStyle(.button)
                        .buttonStyle(.glass)
                        .help(L.s("MtpHelp"))
                        .accessibilityIdentifier("MtpToggle")
                }
                Picker("", selection: $chat.selectedModelId) {
                    if chat.selectedModelId == nil { Text(L.s("ChooseModel")).tag(String?.none) }
                    ForEach(chat.models) { model in
                        Text(model.displayName).tag(Optional(model.id))
                    }
                }
                .labelsHidden()
                .frame(maxWidth: 280)
                .accessibilityIdentifier("ModelPicker")
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
                        .disabled(chat.draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || chat.selectedModel == nil)
                        .help(L.s("Send"))
                        .accessibilityLabel(L.s("Send"))
                        .accessibilityIdentifier("SendButton")
                }
            }
        }
        .padding(EdgeInsets(top: 14, leading: 16, bottom: 10, trailing: 10))
        .glassEffect(.regular, in: .rect(cornerRadius: 24))
        .frame(maxWidth: 760)
        .padding(EdgeInsets(top: 8, leading: 24, bottom: 20, trailing: 24))
        .onAppear { focused = true }
    }
}
