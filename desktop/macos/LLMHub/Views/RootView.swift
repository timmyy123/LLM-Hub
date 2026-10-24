import SwiftUI

/// One sidebar for everything: app sections and chats, or, in Settings, a Back button and the settings sections.
struct RootView: View {
    @Environment(AppModel.self) private var app

    var body: some View {
        Group {
            if app.isCompact {
                // Design and Studio: the sidebar becomes the design's 64pt icon rail.
                HStack(spacing: 0) {
                    Rail()
                    content
                }
            } else {
                NavigationSplitView {
                    Sidebar()
                        .navigationSplitViewColumnWidth(min: 240, ideal: 260, max: 320)
                } detail: {
                    content
                }
            }
        }
        .navigationTitle(title)
    }

    private var content: some View {
        Group {
            switch app.section {
            case .chat: ChatView()
            case .agent: AgentView()
            case .design: DesignView()
            case .studio: StudioView()
            case .settings(let section): SettingsView(section: section)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var title: String {
        switch app.section {
        case .chat: return app.chat.currentTitle.isEmpty ? L.s("NavChat") : app.chat.currentTitle
        case .agent: return L.s("NavAgent")
        case .design: return app.design.currentName.isEmpty ? L.s("NavDesign") : "\(L.s("NavDesign")) · \(app.design.currentName)"
        case .studio: return L.s("NavStudio")
        case .settings: return L.s("NavSettings")
        }
    }
}

private struct Sidebar: View {
    @Environment(AppModel.self) private var app

    var body: some View {
        VStack(spacing: 0) {
            if app.isSettings { settingsList } else { appList }
            EngineCard()
                .padding(10)
        }
    }

    private var appList: some View {
        @Bindable var chat = app.chat
        return VStack(spacing: 8) {
            Button {
                app.chat.newChat()
                app.section = .chat
            } label: {
                Label(L.s("NewChat"), systemImage: "plus")
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .fontWeight(.semibold)
            }
            .buttonStyle(.glass)
            .controlSize(.large)
            .accessibilityIdentifier("NewChatButton")
            .padding(.horizontal, 10)
            .padding(.top, 8)

            List(selection: Binding(get: { app.section == .chat ? chat.selectedChatId.map { "chat:\($0)" } ?? "nav:chat" : navKey },
                                    set: { select($0) })) {
                Section {
                    navRow("chat", L.s("NavChat"), "bubble.left", "NavChat")
                    navRow("agent", L.s("NavAgent"), "terminal", "NavAgent")
                    navRow("design", L.s("NavDesign"), "paintpalette", "NavDesign")
                    navRow("studio", L.s("NavStudio"), "photo", "NavStudio")
                }
                if app.section == .chat {
                    ForEach(chat.groups, id: \.title) { group in
                        Section(group.title) {
                            ForEach(group.chats) { item in
                                Text(item.title)
                                    .lineLimit(1)
                                    .tag("chat:\(item.id)")
                                    .accessibilityIdentifier("ChatTitle")
                                    .contextMenu {
                                        Button(L.s(item.starred ? "Unstar" : "Star"), systemImage: item.starred ? "star.slash" : "star") {
                                            app.chat.toggleStar(item)
                                        }
                                        .accessibilityIdentifier("StarChatMenuItem")
                                        Button(L.s("DeleteChat"), role: .destructive) { app.chat.delete(item) }
                                            .accessibilityIdentifier("DeleteChatMenuItem")
                                    }
                            }
                        }
                    }
                }
            }
            .listStyle(.sidebar)
            .searchable(text: $chat.search, placement: .sidebar, prompt: L.s("SearchChats"))
        }
    }

    private var settingsList: some View {
        VStack(alignment: .leading, spacing: 4) {
            Button {
                app.section = .chat
            } label: {
                Label(L.s("Back"), systemImage: "chevron.left").fontWeight(.semibold)
            }
            .buttonStyle(.plain)
            .padding(.horizontal, 18)
            .padding(.vertical, 12)
            .accessibilityIdentifier("BackButton")

            List(selection: Binding(get: { app.section }, set: { if let s = $0 { app.section = s } })) {
                Section(L.s("NavSettings")) {
                    ForEach(SettingsSection.allCases, id: \.self) { section in
                        Label(section.title, systemImage: section.symbol).tag(AppSection.settings(section)).accessibilityIdentifier(section.identifier)
                    }
                }
            }
            .listStyle(.sidebar)
        }
    }

    private var navKey: String {
        switch app.section {
        case .agent: return "nav:agent"
        case .design: return "nav:design"
        case .studio: return "nav:studio"
        default: return "nav:chat"
        }
    }

    private func navRow(_ key: String, _ title: String, _ symbol: String, _ id: String) -> some View {
        Label(title, systemImage: symbol).tag("nav:\(key)").accessibilityIdentifier(id)
    }

    private func select(_ tag: String?) {
        guard let tag else { return }
        if tag.hasPrefix("chat:") {
            app.section = .chat
            app.chat.selectedChatId = String(tag.dropFirst(5))
            return
        }
        switch tag {
        case "nav:agent": app.section = .agent
        case "nav:design": app.section = .design
        case "nav:studio": app.section = .studio
        default: app.section = .chat
        }
    }
}

private struct EngineCard: View {
    @Environment(AppModel.self) private var app

    var body: some View {
        Button {
            app.section = .settings(.models)
        } label: {
            HStack(spacing: 10) {
                Circle().fill(dotColor).frame(width: 8, height: 8)
                VStack(alignment: .leading, spacing: 1) {
                    Text(app.engines.statusText).font(.system(size: 13, weight: .semibold)).lineLimit(1)
                        .accessibilityIdentifier("EngineStatus")
                    Text(app.hardwareLine).font(.system(size: 12)).foregroundStyle(Theme.textMuted).lineLimit(1)
                }
                Spacer(minLength: 0)
                Image(systemName: "gearshape").foregroundStyle(Theme.textMuted)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 10)
            .contentShape(Rectangle())
            .glassEffect(.regular.interactive(), in: .rect(cornerRadius: 14))
        }
        .buttonStyle(.plain)
        .accessibilityIdentifier("SettingsButton")
    }

    private var dotColor: Color {
        switch app.engines.state {
        case .ready: return Theme.success
        case .installing, .loading: return Theme.warning
        case .error: return Theme.danger
        case .idle: return Theme.textMuted
        }
    }
}

extension SettingsSection {
    var title: String {
        switch self {
        case .general: return L.s("SettingsGeneral")
        case .models: return L.s("SettingsModels")
        case .engines: return L.s("SettingsEngines")
        case .api: return L.s("SettingsLocalApi")
        case .speech: return L.s("SettingsSpeech")
        case .studio: return L.s("SettingsStudio")
        case .agent: return L.s("SettingsAgent")
        case .websearch: return L.s("SettingsWebSearch")
        case .storage: return L.s("SettingsStorage")
        }
    }

    var symbol: String {
        switch self {
        case .general: return "gearshape"
        case .models: return "cube"
        case .engines: return "bolt"
        case .api: return "network"
        case .speech: return "mic"
        case .studio: return "photo.on.rectangle"
        case .agent: return "terminal"
        case .websearch: return "globe"
        case .storage: return "internaldrive"
        }
    }

    var identifier: String {
        switch self {
        case .api: return "SectionApi"
        case .websearch: return "SectionWebSearch"
        default: return "Section" + rawValue.prefix(1).uppercased() + rawValue.dropFirst()
        }
    }
}

/// The 64pt icon rail from the design (Design and Studio): section icons and the engine dot, in Liquid Glass.
private struct Rail: View {
    @Environment(AppModel.self) private var app

    var body: some View {
        VStack(spacing: 6) {
            item(.chat, "bubble.left", L.s("NavChat"), "NavChat")
            item(.agent, "terminal", L.s("NavAgent"), "NavAgent")
            item(.design, "paintpalette", L.s("NavDesign"), "NavDesign")
            item(.studio, "photo", L.s("NavStudio"), "NavStudio")
            Spacer()
            Button {
                app.section = .settings(.models)
            } label: {
                Circle().fill(engineColor).frame(width: 8, height: 8).frame(width: 44, height: 44)
            }
            .buttonStyle(.plain)
            .glassEffect(.regular.interactive(), in: .rect(cornerRadius: 12))
            .help(app.engines.statusText)
            .accessibilityIdentifier("SettingsButton")
        }
        .padding(.vertical, 12)
        .frame(width: 64)
        .frame(maxHeight: .infinity)
        .overlay(alignment: .trailing) { Rectangle().fill(Theme.divider).frame(width: 1) }
    }

    private func item(_ section: AppSection, _ symbol: String, _ title: String, _ id: String) -> some View {
        Button {
            app.section = section
        } label: {
            Image(systemName: symbol).font(.system(size: 16)).frame(width: 44, height: 44).contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .foregroundStyle(app.section == section ? .primary : Theme.textMuted)
        .glassEffect(app.section == section ? .regular.interactive() : .identity, in: .rect(cornerRadius: 12))
        .help(title)
        .accessibilityLabel(title)
        .accessibilityIdentifier(id)
    }

    private var engineColor: Color {
        switch app.engines.state {
        case .ready: return Theme.success
        case .installing, .loading: return Theme.warning
        case .error: return Theme.danger
        case .idle: return Theme.textMuted
        }
    }
}

struct PlaceholderView: View {
    let section: String
    let symbol: String

    var body: some View {
        VStack(spacing: 10) {
            Image(systemName: symbol).font(.system(size: 32)).foregroundStyle(Theme.textMuted)
            Text(L.f("ComingSoonTitle", L.s("Nav\(section)"))).font(.system(size: 22, weight: .semibold))
                .accessibilityIdentifier("PlaceholderHeading")
            Text(L.s("\(section)Description")).foregroundStyle(Theme.textMuted).multilineTextAlignment(.center)
        }
        .frame(maxWidth: 520)
    }
}
