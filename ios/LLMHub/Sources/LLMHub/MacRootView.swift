//
//  MacRootView.swift
//  LLMHub
//
//  macOS window layout: a sidebar with an AI Chat / Tools switch. The chat tab
//  lists recent chats, the tools tab lists every tool (same names, icons,
//  order and premium locks as the iOS Home screen); Models and Settings sit at
//  the bottom of both. The selected item fills the detail column.
//

#if os(macOS)
import SwiftUI

enum MacSidebarItem: Hashable {
    case chat
    case chatSession(UUID)
    case feature(String)
    case models
    case settings
}

struct MacRootView: View {
    @EnvironmentObject var settings: AppSettings
    @ObservedObject private var purchases = PurchaseManager.shared
    @State private var selection: MacSidebarItem?
    @State private var showPremium = false
    @StateObject private var chatVM = ChatViewModel()
    @AppStorage("mac_sidebar_tab") private var sidebarTabRaw = MacSidebarTab.chat.rawValue
    @AppStorage("mac_last_tool_route") private var lastToolRoute = "writing_aid"
    @State private var showDeleteAllChatsAlert = false

    enum MacSidebarTab: String {
        case chat, tools
    }

    init(initialSelection: MacSidebarItem = .chat) {
        _selection = State(initialValue: initialSelection)
    }

    private var sidebarTab: Binding<MacSidebarTab> {
        Binding(
            get: { MacSidebarTab(rawValue: sidebarTabRaw) ?? .chat },
            set: { tab in
                sidebarTabRaw = tab.rawValue
                switch tab {
                case .chat:
                    selection = .chatSession(chatVM.currentSessionId)
                case .tools:
                    selection = .feature(isLocked(lastToolRoute) ? "writing_aid" : lastToolRoute)
                }
            }
        )
    }

    /// Same routes the iOS Home screen gates behind Premium.
    private static let lockedRoutes: Set<String> = [
        "agent", "vibe_voice", "vibe_coder", "image_generator", "video_generator", "music_generator"
    ]

    /// Feature definitions come straight from the iOS Home screen.
    private var home: HomeScreen {
        HomeScreen(onNavigateToChat: {}, onNavigateToModels: {}, onNavigateToSettings: {}, onNavigateToRoute: { _ in })
    }

    private func isLocked(_ route: String) -> Bool {
        !purchases.isPremium && Self.lockedRoutes.contains(route)
    }

    var body: some View {
        NavigationSplitView {
            sidebar
                .navigationSplitViewColumnWidth(min: 200, ideal: 240, max: 320)
        } detail: {
            NavigationStack {
                detail
            }
            .id(detailIdentity)
        }
        .onChange(of: selection) { oldValue, newValue in
            if case .feature(let route)? = newValue {
                if isLocked(route) {
                    selection = oldValue
                    showPremium = true
                    return
                }
                lastToolRoute = route
                sidebarTabRaw = MacSidebarTab.tools.rawValue
            }
            if case .chatSession(let id)? = newValue {
                sidebarTabRaw = MacSidebarTab.chat.rawValue
                if chatVM.currentSessionId != id {
                    chatVM.stopAutoReadout()
                    chatVM.currentSessionId = id
                }
            }
            if case .chat? = newValue {
                sidebarTabRaw = MacSidebarTab.chat.rawValue
            }
        }
        .onChange(of: chatVM.currentSessionId) { _, newId in
            switch selection {
            case .chat?, .chatSession?: selection = .chatSession(newId)
            default: break
            }
        }
        .alert(settings.localized("dialog_delete_all_chats_title"), isPresented: $showDeleteAllChatsAlert) {
            Button(settings.localized("action_delete_all"), role: .destructive) {
                ChatStore.shared.clearAll()
                chatVM.newChat()
            }
            Button(settings.localized("action_cancel"), role: .cancel) {}
        } message: {
            Text(settings.localized("dialog_delete_all_chats_message"))
        }
        .onAppear {
            switch selection ?? .chat {
            case .chat:
                sidebarTabRaw = MacSidebarTab.chat.rawValue
                selection = .chatSession(chatVM.currentSessionId)
            case .feature:
                sidebarTabRaw = MacSidebarTab.tools.rawValue
            default:
                break
            }
        }
        .onOpenURL { url in
            guard AudioSearchModel.isAudioURL(url) else { return }
            AudioSearchModel.importShared([url])
            selection = .feature("audio_search")
        }
        .sheet(isPresented: $showPremium) {
            MacPremiumView()
                .environmentObject(settings)
        }
    }

    private var detailIdentity: String {
        switch selection ?? .chat {
        case .chat, .chatSession: return "chat"
        case .models: return "models"
        case .settings: return "settings"
        case .feature(let route): return route
        }
    }

    private var sidebar: some View {
        let hero = home.heroFeature
        return List(selection: $selection) {
            switch sidebarTab.wrappedValue {
            case .chat:
                Section {
                    Button {
                        chatVM.newChat()
                        selection = .chatSession(chatVM.currentSessionId)
                    } label: {
                        Label(settings.localized("drawer_new_chat"), systemImage: "plus.bubble")
                    }
                    .buttonStyle(.plain)
                }

                Section(settings.localized("drawer_recent_chats")) {
                    if chatVM.chatSessions.isEmpty {
                        Text(settings.localized("drawer_no_chats"))
                            .foregroundStyle(.secondary)
                    } else {
                        ForEach(chatVM.chatSessions) { session in
                            VStack(alignment: .leading, spacing: 1) {
                                Text(session.title).lineLimit(1)
                                Text(session.createdAt, style: .date)
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                            .tag(MacSidebarItem.chatSession(session.id))
                            .contextMenu {
                                Button(settings.localized("action_delete"), role: .destructive) {
                                    chatVM.deleteSession(session.id)
                                }
                                Button(settings.localized("drawer_clear_all_chats"), role: .destructive) {
                                    showDeleteAllChatsAlert = true
                                }
                            }
                        }
                    }
                }
            case .tools:
                Section {
                    ForEach(home.toolsFeatures + home.utilityFeatures, id: \.route) { feature in
                        HStack {
                            Label(settings.localized(feature.titleKey), systemImage: feature.iconSystemName)
                            if isLocked(feature.route) {
                                Spacer()
                                Image(systemName: "lock.fill")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                        }
                        .tag(MacSidebarItem.feature(feature.route))
                    }
                }
            }
        }
        .listStyle(.sidebar)
        .safeAreaInset(edge: .top) {
            Picker("", selection: sidebarTab) {
                Label(settings.localized(hero.titleKey), systemImage: hero.iconSystemName)
                    .tag(MacSidebarTab.chat)
                Label(settings.localized("home_section_tools"), systemImage: "square.grid.2x2")
                    .tag(MacSidebarTab.tools)
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .controlSize(.large)
            .frame(maxWidth: .infinity)
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
        }
        .safeAreaInset(edge: .bottom) {
            VStack(spacing: 2) {
                Divider()
                sidebarFooterButton(.models, title: settings.localized("models"), icon: "square.and.arrow.down")
                sidebarFooterButton(.settings, title: settings.localized("settings"), icon: "gearshape")
            }
            .padding(.horizontal, 8)
            .padding(.bottom, 8)
            .background(.bar)
        }
        .navigationTitle(settings.localized("app_name"))
    }

    private func sidebarFooterButton(_ item: MacSidebarItem, title: String, icon: String) -> some View {
        Button {
            selection = item
        } label: {
            Label(title, systemImage: icon)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.horizontal, 8)
                .padding(.vertical, 5)
                .contentShape(Rectangle())
                .background(
                    RoundedRectangle(cornerRadius: 6)
                        .fill(selection == item ? Color.accentColor.opacity(0.25) : Color.clear)
                )
        }
        .buttonStyle(.plain)
    }

    @ViewBuilder
    private var detail: some View {
        let toModels = { selection = .models }
        switch selection ?? .chat {
        case .chat, .chatSession:
            MacChatView(vm: chatVM, onNavigateToModels: toModels)
        case .models:
            MacModelsView(onShowPremium: { showPremium = true })
        case .settings:
            MacSettingsView(onNavigateToModels: toModels, onShowPremium: { showPremium = true })
        case .feature(let route):
            switch route {
            case "writing_aid": MacWritingAidView()
            case "translator": MacTranslatorView(onNavigateToModels: toModels)
            case "transcriber": MacTranscriberView()
            case "scam_detector": MacScamDetectorView()
            case "vibe_coder": MacVibeCoderView()
            case "vibe_voice": MacVibeVoiceView()
            case "image_generator": MacImageGeneratorView(onNavigateToModels: toModels)
            case "video_generator": MacVideoGeneratorView(onNavigateToModels: toModels)
            case "image_upscaler": MacImageUpscalerView(onNavigateToModels: toModels)
            case "music_generator": MacMusicGeneratorView(onNavigateToModels: toModels)
            case "photo_search": MacPhotoSearchView(onNavigateToModels: toModels)
            case "audio_search": MacAudioSearchView(onNavigateToModels: toModels)
            case "video_moment": MacVideoMomentView(onNavigateToModels: toModels)
            case "agent": MacAgentView(onNavigateToModels: toModels)
            default: EmptyView()
            }
        }
    }
}

#endif
