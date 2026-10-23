import SwiftUI

/// One sidebar for everything: app sections and chats, or, in Settings, a Back button and the settings sections.
struct RootView: View {
    @Environment(AppModel.self) private var app

    var body: some View {
        NavigationSplitView {
            Sidebar()
                .navigationSplitViewColumnWidth(min: 240, ideal: 260, max: 320)
        } detail: {
            Group {
                switch app.section {
                case .chat: ChatView()
                case .agent: PlaceholderView(section: "Agent", symbol: "terminal")
                case .design: PlaceholderView(section: "Design", symbol: "paintpalette")
                case .studio: PlaceholderView(section: "Studio", symbol: "photo")
                case .settings(let section): SettingsView(section: section)
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .navigationTitle(title)
    }

    private var title: String {
        switch app.section {
        case .chat: return app.chat.currentTitle.isEmpty ? L.s("NavChat") : app.chat.currentTitle
        case .agent: return L.s("NavAgent")
        case .design: return L.s("NavDesign")
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
                    Label(L.s("SettingsGeneral"), systemImage: "gearshape").tag(AppSection.settings(.general)).accessibilityIdentifier("SectionGeneral")
                    Label(L.s("SettingsModels"), systemImage: "cube").tag(AppSection.settings(.models)).accessibilityIdentifier("SectionModels")
                    Label(L.s("SettingsLocalApi"), systemImage: "network").tag(AppSection.settings(.api)).accessibilityIdentifier("SectionApi")
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
