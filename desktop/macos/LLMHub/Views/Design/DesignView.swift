import AppKit
import SwiftUI
import UniformTypeIdentifiers

/// Design, from the design canvas: the conversation with kind tabs and the Projects menu on the left, the live WebKit preview
/// with sizes, versions, Comment / Hand off / Export in the middle, and the design-system panel on the right.
struct DesignView: View {
    @Environment(AppModel.self) private var app
    @FocusState private var composerFocused: Bool

    var body: some View {
        HStack(spacing: 0) {
            conversation
                .frame(width: 340)
            VerticalRule()
            previewColumn
                .frame(minWidth: 0, maxWidth: .infinity)
            VerticalRule()
            DesignSystemPanel()
                .frame(width: 260)
        }
    }

    // MARK: - Conversation

    private var conversation: some View {
        VStack(spacing: 0) {
            kindBar
            messageList
            composer
        }
    }

    private var kindBar: some View {
        let design = app.design
        return HStack(spacing: 4) {
            HStack(spacing: 2) {
                ForEach(DesignKind.allCases) { kind in
                    let on = design.kind == kind
                    Button {
                        design.chooseKind(kind)
                    } label: {
                        Text(kind.label)
                            .font(.system(size: 13))
                            .lineLimit(1)
                            .minimumScaleFactor(0.8)
                            .padding(.horizontal, 10)
                            .frame(height: 30)
                            .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                    .foregroundStyle(on ? Color.primary : Theme.textMuted)
                    .glassEffect(on ? .regular.interactive() : .identity, in: .rect(cornerRadius: 8))
                    .disabled(design.isGenerating && !on)
                    .accessibilityAddTraits(on ? .isSelected : [])
                    .accessibilityIdentifier("Kind\(kind.rawValue)")
                }
            }
            Spacer(minLength: 0)
            ProjectsMenu()
        }
        .padding(EdgeInsets(top: 10, leading: 12, bottom: 10, trailing: 8))
        .overlay(alignment: .bottom) { HorizontalRule() }
    }

    private var messageList: some View {
        let design = app.design
        return ScrollViewReader { proxy in
            ScrollView {
                VStack(alignment: .leading, spacing: 14) {
                    if design.messages.isEmpty {
                        VStack(alignment: .leading, spacing: 8) {
                            Text(L.s("DesignEmptyTitle"))
                                .font(.system(size: 17, weight: .semibold))
                                .accessibilityIdentifier("DesignEmptyTitle")
                            Text(L.s("DesignEmptyBody"))
                                .font(.system(size: 13))
                                .foregroundStyle(Theme.textMuted)
                        }
                        .fixedSize(horizontal: false, vertical: true)
                    }
                    ForEach(design.messages) { message in
                        MessageRow(message: message)
                    }
                    if design.isGenerating {
                        HStack(alignment: .firstTextBaseline, spacing: 8) {
                            Circle().fill(Theme.accent).frame(width: 7, height: 7)
                            Text(design.progressText)
                                .font(.system(size: 13))
                                .foregroundStyle(Theme.textMuted)
                                .accessibilityIdentifier("DesignProgress")
                        }
                    }
                    if !design.status.isEmpty {
                        Text(design.status)
                            .font(.system(size: 12))
                            .foregroundStyle(Theme.textMuted)
                            .textSelection(.enabled)
                            .accessibilityIdentifier("DesignStatus")
                    }
                    Color.clear.frame(height: 1).id("end")
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(16)
            }
            .onChange(of: design.liveCount / 400) { proxy.scrollTo("end", anchor: .bottom) }
            .onChange(of: design.messages.count) { proxy.scrollTo("end", anchor: .bottom) }
            .onChange(of: design.selectedId) { proxy.scrollTo("end", anchor: .bottom) }
            .onAppear { proxy.scrollTo("end", anchor: .bottom) }
        }
    }

    private var composer: some View {
        @Bindable var design = app.design
        return VStack(spacing: 8) {
            TextField(L.s(design.hasPreview ? "DesignPlaceholder" : "DesignStartPlaceholder"), text: $design.draft, axis: .vertical)
                .textFieldStyle(.plain)
                .font(.system(size: 14))
                .lineLimit(2...7)
                .focused($composerFocused)
                .onKeyPress(.return, phases: .down) { press in
                    if press.modifiers.contains(.shift) { return .ignored }
                    design.send()
                    return .handled
                }
                .accessibilityIdentifier("DesignComposer")
            HStack(spacing: 6) {
                Picker("", selection: $design.selectedModelId) {
                    if design.selectedModelId == nil { Text(L.s("ChooseModel")).tag(String?.none) }
                    ForEach(design.models) { model in
                        Text(model.displayName).tag(Optional(model.id))
                    }
                }
                .labelsHidden()
                .controlSize(.small)
                .frame(maxWidth: 230, alignment: .leading)
                .accessibilityIdentifier("DesignModelPicker")
                Spacer(minLength: 0)
                if design.isGenerating {
                    Button { design.stop() } label: { Image(systemName: "stop.fill").frame(width: 20, height: 20) }
                        .buttonStyle(.glass)
                        .buttonBorderShape(.circle)
                        .help(L.s("Stop"))
                        .accessibilityLabel(L.s("Stop"))
                        .accessibilityIdentifier("DesignStop")
                } else {
                    Button { design.send() } label: { Image(systemName: "arrow.up").fontWeight(.semibold).frame(width: 20, height: 20) }
                        .buttonStyle(.glassProminent)
                        .buttonBorderShape(.circle)
                        .disabled(design.draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || design.selectedModel == nil)
                        .help(L.s("Send"))
                        .accessibilityLabel(L.s("Send"))
                        .accessibilityIdentifier("DesignSend")
                }
            }
        }
        .padding(EdgeInsets(top: 12, leading: 12, bottom: 8, trailing: 8))
        .glassEffect(.regular, in: .rect(cornerRadius: 18))
        .padding(12)
    }

    // MARK: - Preview

    private var previewColumn: some View {
        let design = app.design
        return VStack(spacing: 0) {
            ViewThatFits(in: .horizontal) {
                toolbar(labels: true, versionText: true, sizeIcons: false)
                toolbar(labels: false, versionText: true, sizeIcons: false)
                toolbar(labels: false, versionText: false, sizeIcons: true)
            }
            .frame(minHeight: 52)
            .overlay(alignment: .bottom) { HorizontalRule() }

            ZStack {
                PreviewWebView(preview: design.preview, file: design.previewFile)
                    .background(Color.white)
                    .clipShape(RoundedRectangle(cornerRadius: 10))
                    .overlay(RoundedRectangle(cornerRadius: 10).strokeBorder(Theme.divider))
                    .frame(maxWidth: design.previewWidth ?? .infinity, maxHeight: .infinity)
                    .accessibilityIdentifier("DesignPreview")
                if !design.hasPreview {
                    Text(L.s("DesignNothingYet"))
                        .foregroundStyle(Color(hex: "#8F8E88"))
                        .allowsHitTesting(false)
                }
            }
            .animation(.smooth(duration: 0.2), value: design.previewSize)
            .padding(24)
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .background(DesignView.canvas)
    }

    private func toolbar(labels: Bool, versionText: Bool, sizeIcons: Bool) -> some View {
        @Bindable var design = app.design
        return HStack(spacing: 8) {
            Picker("", selection: $design.previewSize) {
                ForEach([("desktop", "DesignDesktop", "desktopcomputer"), ("tablet", "DesignTablet", "ipad.landscape"), ("phone", "DesignPhone", "iphone")], id: \.0) { size in
                    Group {
                        if sizeIcons { Image(systemName: size.2).help(L.s(size.1)) } else { Text(L.s(size.1)) }
                    }
                    .tag(size.0)
                }
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .controlSize(.small)
            .fixedSize()
            .accessibilityIdentifier("SizeTabs")

            if design.hasPreview {
                HStack(spacing: 2) {
                    Button { design.previousVersion() } label: { Image(systemName: "chevron.left").font(.system(size: 11, weight: .semibold)).frame(width: 22, height: 22) }
                        .buttonStyle(.borderless)
                        .disabled(design.version <= 1)
                        .accessibilityLabel("Previous version")
                    if versionText {
                        Text(design.versionText)
                            .font(.system(size: 12))
                            .foregroundStyle(Theme.textFaint)
                            .fixedSize()
                            .accessibilityIdentifier("DesignVersion")
                    }
                    Button { design.nextVersion() } label: { Image(systemName: "chevron.right").font(.system(size: 11, weight: .semibold)).frame(width: 22, height: 22) }
                        .buttonStyle(.borderless)
                        .disabled(design.version >= (design.selected?.versions ?? 0))
                        .accessibilityLabel("Next version")
                }
                .help(design.versionText)
            }

            Spacer(minLength: 8)

            HStack(spacing: 8) {
                Button {
                    composerFocused = true
                } label: {
                    toolbarLabel(L.s("DesignComment"), "text.bubble", labels)
                }
                .buttonStyle(.glass)
                .help(L.s("DesignComment"))
                .accessibilityIdentifier("CommentButton")

                Button {
                    design.handOffToAgent()
                } label: {
                    toolbarLabel(L.s("DesignHandOff"), "arrow.up.forward.app", labels)
                }
                .buttonStyle(.glass)
                .disabled(!design.hasPreview || design.isGenerating)
                .help(L.s("DesignHandOff"))
                .accessibilityIdentifier("HandOffButton")

                Menu {
                    Button("HTML") { export("html") }
                    Button("PDF") { export("pdf") }
                    Button("ZIP") { export("zip") }
                    Button("Markdown") { export("md") }
                } label: {
                    Text(L.s("DesignExport")).fontWeight(.medium)
                }
                .menuStyle(.button)
                .buttonStyle(.glassProminent)
                .fixedSize()
                .disabled(!design.hasPreview)
                .accessibilityIdentifier("ExportButton")
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
    }

    @ViewBuilder
    private func toolbarLabel(_ title: String, _ symbol: String, _ showTitle: Bool) -> some View {
        if showTitle {
            Label(title, systemImage: symbol).labelStyle(.titleAndIcon).fixedSize()
        } else {
            Image(systemName: symbol).accessibilityLabel(title)
        }
    }

    /// Asks where to save, then writes HTML, PDF, ZIP or Markdown; the result goes to the status line.
    private func export(_ format: String) {
        let design = app.design
        guard let project = design.selected, project.currentFile != nil else { return }
        let ext = ["md": "md", "zip": "zip", "pdf": "pdf"][format] ?? "html"
        let panel = NSSavePanel()
        panel.nameFieldStringValue = DesignService.safe(project.name) + "." + ext
        panel.allowedContentTypes = [UTType(filenameExtension: ext) ?? .data]
        panel.canCreateDirectories = true
        guard panel.runModal() == .OK, let url = panel.url else { return }
        Task {
            do {
                try await design.export(format, to: url)
                design.status = L.f("StudioSavedTo", url.path)
            } catch {
                Log.error("Design export failed", error)
                design.status = L.f("ErrorPrefix", error.localizedDescription)
            }
        }
    }

    /// The darker canvas behind the page, as in the design (#121211 in dark).
    static let canvas = Color(nsColor: NSColor(name: nil) { appearance in
        appearance.bestMatch(from: [.darkAqua, .aqua]) == .darkAqua
            ? NSColor(srgbRed: 0x12 / 255, green: 0x12 / 255, blue: 0x11 / 255, alpha: 1)
            : NSColor(srgbRed: 0xEE / 255, green: 0xED / 255, blue: 0xE8 / 255, alpha: 1)
    })
}

// MARK: - Pieces

private struct MessageRow: View {
    let message: DesignMessage

    var body: some View {
        if message.role == "user" {
            HStack {
                Spacer(minLength: 30)
                Text(message.text)
                    .font(.system(size: 14))
                    .textSelection(.enabled)
                    .padding(.horizontal, 14)
                    .padding(.vertical, 10)
                    .background(Theme.userBubble, in: RoundedRectangle(cornerRadius: 14))
                    .frame(maxWidth: 290, alignment: .trailing)
                    .accessibilityIdentifier("DesignUserMessage")
            }
        } else {
            Text(Self.markdown(message.text))
                .font(.system(size: 14))
                .lineSpacing(3)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
                .accessibilityIdentifier("DesignAssistantMessage")
        }
    }

    private static func markdown(_ text: String) -> AttributedString {
        (try? AttributedString(markdown: text, options: .init(interpretedSyntax: .inlineOnlyPreservingWhitespace))) ?? AttributedString(text)
    }
}

/// Switch, start or delete a design (the sidebar is a rail on this screen).
private struct ProjectsMenu: View {
    @Environment(AppModel.self) private var app

    var body: some View {
        let design = app.design
        Menu {
            ForEach(design.projects) { project in
                Toggle(project.displayName, isOn: Binding(get: { project.id == design.selectedId }, set: { _ in design.select(project.id) }))
            }
            if !design.projects.isEmpty { Divider() }
            Button { design.newProject() } label: { Label(L.s("DesignNewProject"), systemImage: "plus") }
            if let current = design.selected {
                Button(role: .destructive) { design.delete(current.id) } label: { Label(L.s("Delete"), systemImage: "trash") }
            }
        } label: {
            Image(systemName: "list.bullet")
        }
        .menuStyle(.button)
        .menuIndicator(.hidden)
        .buttonStyle(.glass)
        .fixedSize()
        .disabled(design.isGenerating)
        .help(L.s("DesignProjects"))
        .accessibilityLabel(L.s("DesignProjects"))
        .accessibilityIdentifier("ProjectsButton")
    }
}

/// The design-system panel: system picker, swatches, type sample, the selected element, and the export formats.
private struct DesignSystemPanel: View {
    @Environment(AppModel.self) private var app

    var body: some View {
        @Bindable var design = app.design
        VStack(alignment: .leading, spacing: 18) {
            field(L.s("DesignSystemLabel")) {
                Menu {
                    Picker("", selection: $design.system) {
                        ForEach(DesignSystem.all) { system in
                            Text(system.name).tag(system)
                        }
                    }
                    .pickerStyle(.inline)
                    .labelsHidden()
                } label: {
                    HStack {
                        Text(design.system.name)
                        Spacer(minLength: 4)
                        Image(systemName: "chevron.up.chevron.down").font(.system(size: 10, weight: .semibold)).foregroundStyle(Theme.textMuted)
                    }
                    .padding(.horizontal, 4)
                    .frame(maxWidth: .infinity, minHeight: 26)
                    .contentShape(Rectangle())
                }
                .menuStyle(.button)
                .menuIndicator(.hidden)
                .buttonStyle(.glass)
                .buttonSizing(.flexible)
                .frame(maxWidth: .infinity)
                .accessibilityIdentifier("SystemPicker")
            }
            field(L.s("DesignColors")) {
                HStack(spacing: 8) {
                    ForEach(design.system.colors, id: \.self) { hex in
                        RoundedRectangle(cornerRadius: 8)
                            .fill(Color(hex: hex))
                            .frame(height: 32)
                            .overlay(RoundedRectangle(cornerRadius: 8).strokeBorder(Theme.controlBorder))
                            .help(hex)
                            .accessibilityLabel(hex)
                    }
                }
            }
            field(L.s("DesignType")) {
                VStack(alignment: .leading, spacing: 6) {
                    Text(design.system.headingFont)
                        .font(Self.sample(design.system.headingFont, size: 22, weight: .semibold))
                    Text(design.system.bodyFont)
                        .font(Self.sample(design.system.bodyFont, size: 13, weight: .regular))
                        .foregroundStyle(Theme.textMuted)
                }
                .lineLimit(1)
                .padding(12)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(Color.primary.opacity(0.05), in: RoundedRectangle(cornerRadius: 10))
            }
            field(L.s("DesignSelected")) {
                Text(design.selectedElement.isEmpty ? L.s("DesignSelectHint") : design.selectedElement)
                    .font(.system(size: 13))
                    .foregroundStyle(design.selectedElement.isEmpty ? Theme.textMuted : Color.primary)
                    .textSelection(.enabled)
                    .fixedSize(horizontal: false, vertical: true)
                    .padding(12)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(Color.primary.opacity(0.05), in: RoundedRectangle(cornerRadius: 10))
                    .accessibilityIdentifier("DesignSelectedElement")
            }
            Spacer(minLength: 0)
            Text(L.s("DesignExports"))
                .font(.system(size: 12))
                .foregroundStyle(Theme.textFaint)
        }
        .padding(16)
        .frame(maxHeight: .infinity, alignment: .top)
    }

    /// The family itself when this Mac has it (Google Fonts are often installed), else the closest system face.
    private static func sample(_ family: String, size: CGFloat, weight: Font.Weight) -> Font {
        if NSFontManager.shared.availableFontFamilies.contains(family) { return .custom(family, size: size).weight(weight) }
        let serif = family.contains("Serif") || family == "Fraunces"
        return .system(size: size, weight: weight, design: serif ? .serif : .default)
    }

    private func field<Content: View>(_ title: String, @ViewBuilder _ content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title).font(.system(size: 12, weight: .medium)).foregroundStyle(Theme.textFaint)
            content()
        }
    }
}

/// Hosts the app-lifetime WKWebView in a fresh container, so moving between tabs never tears the page down.
private struct PreviewWebView: NSViewRepresentable {
    let preview: DesignPreview
    let file: URL?

    func makeNSView(context: Context) -> NSView {
        let container = NSView()
        let web = preview.webView
        web.removeFromSuperview()
        web.frame = container.bounds
        web.autoresizingMask = [.width, .height]
        container.addSubview(web)
        return container
    }

    func updateNSView(_ container: NSView, context: Context) {
        if preview.webView.superview !== container {
            preview.webView.removeFromSuperview()
            preview.webView.frame = container.bounds
            container.addSubview(preview.webView)
        }
        preview.show(file)
    }
}

private struct VerticalRule: View {
    var body: some View { Rectangle().fill(Theme.divider).frame(width: 1) }
}

private struct HorizontalRule: View {
    var body: some View { Rectangle().fill(Theme.divider).frame(height: 1) }
}

private extension Color {
    /// "#b4532a" → Color (sRGB).
    init(hex: String) {
        let value = UInt32(hex.trimmingCharacters(in: CharacterSet(charactersIn: "#")), radix: 16) ?? 0
        self.init(.sRGB, red: Double((value >> 16) & 0xFF) / 255, green: Double((value >> 8) & 0xFF) / 255, blue: Double(value & 0xFF) / 255)
    }
}
