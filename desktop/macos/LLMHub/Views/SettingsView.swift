import SwiftUI

struct SettingsView: View {
    let section: SettingsSection

    var body: some View {
        switch section {
        case .general: GeneralSettings()
        case .models: ModelsSettings()
        case .api: LocalApiSettings()
        }
    }
}

private struct GeneralSettings: View {
    @State private var language = L.override
    @State private var languageChanged = false
    @State private var context = AppSettings.current.contextLength
    @State private var token = AppSettings.current.huggingFaceToken

    var body: some View {
        Form {
            Picker(L.s("Language"), selection: $language) {
                Text(L.s("LanguageSystem")).tag("")
                ForEach(L.languages, id: \.tag) { Text($0.name).tag($0.tag) }
            }
            .accessibilityIdentifier("LanguageBox")
            .onChange(of: language) {
                L.override = language
                languageChanged = true
            }
            if languageChanged {
                Text(L.s("LanguageRestart")).foregroundStyle(Theme.textMuted).accessibilityIdentifier("RestartHint")
            }
            Picker(L.s("ContextLength"), selection: $context) {
                ForEach([4096, 8192, 16384, 32768, 65536, 131072], id: \.self) { Text("\($0 / 1024)K").tag($0) }
            }
            .accessibilityIdentifier("ContextBox")
            .onChange(of: context) {
                AppSettings.current.contextLength = context
                AppSettings.save()
            }
            VStack(alignment: .leading, spacing: 6) {
                SecureField(L.s("HfToken"), text: $token)
                    .accessibilityIdentifier("TokenBox")
                    .onChange(of: token) {
                        AppSettings.current.huggingFaceToken = token
                        AppSettings.save()
                    }
                Text(L.s("HfTokenHelp")).font(.caption).foregroundStyle(Theme.textMuted)
            }
        }
        .formStyle(.grouped)
        .frame(maxWidth: 640)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .padding(.top, 12)
    }
}

private struct ModelsSettings: View {
    @Environment(AppModel.self) private var app

    var body: some View {
        @Bindable var vm = app.models
        VStack(alignment: .leading, spacing: 14) {
            VStack(alignment: .leading, spacing: 4) {
                Text(L.s("SettingsModels")).font(.system(size: 20, weight: .semibold))
                Text(vm.hardwareText).font(.system(size: 12)).foregroundStyle(Theme.textMuted)
            }
            Picker("", selection: $vm.tab) {
                Text(L.s("TabCatalog")).tag(ModelsTab.catalog)
                Text(L.s("TabHuggingFace")).tag(ModelsTab.huggingFace)
                Text(L.s("TabInstalled")).tag(ModelsTab.installed)
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .fixedSize()
            .accessibilityIdentifier("ModelTabs")

            switch vm.tab {
            case .catalog: CatalogList()
            case .huggingFace: HuggingFacePanel()
            case .installed: InstalledList()
            }
        }
        .padding(EdgeInsets(top: 24, leading: 28, bottom: 0, trailing: 28))
        .task { await app.models.loadCatalog() }
    }
}

private struct CatalogList: View {
    @Environment(AppModel.self) private var app

    var body: some View {
        List(app.models.catalog) { item in
            HStack(spacing: 14) {
                VStack(alignment: .leading, spacing: 2) {
                    HStack(spacing: 8) {
                        Text(item.model.name).fontWeight(.semibold).accessibilityIdentifier("CatalogName")
                        Tag(text: item.engineLabel)
                    }
                    Text(item.subtitle).font(.system(size: 12)).foregroundStyle(Theme.textMuted).lineLimit(1)
                    if !item.status.isEmpty {
                        Text(item.status).font(.system(size: 12)).foregroundStyle(Theme.textMuted)
                    }
                }
                Spacer(minLength: 12)
                VariantControls(item: item)
            }
            .padding(.vertical, 6)
            .accessibilityElement(children: .contain)
            .accessibilityIdentifier("CatalogRow-\(item.model.id)")
        }
        .listStyle(.inset)
        .accessibilityIdentifier("CatalogList")
    }
}

/// Variant picker plus download / progress / installed state. Shared by the catalog and Hugging Face tabs.
struct VariantControls: View {
    @Environment(AppModel.self) private var app
    let item: DownloadableItem

    var body: some View {
        VStack(alignment: .trailing, spacing: 4) {
            HStack(spacing: 12) {
                if !item.variants.isEmpty {
                    Picker("", selection: Binding(get: { item.selectedId }, set: { item.selectedId = $0 })) {
                        ForEach(item.variants) { option in
                            Text("\(option.label) · \(option.fitText)").tag(Optional(option.id))
                        }
                    }
                    .labelsHidden()
                    .frame(width: 280)
                    .accessibilityIdentifier("VariantPicker")
                }
                Group {
                    if let job = item.job {
                        HStack(spacing: 8) {
                            VStack(alignment: .leading, spacing: 3) {
                                ProgressView(value: job.percent, total: 100).accessibilityIdentifier("DownloadProgress")
                                    .accessibilityValue(String(format: "%.1f", job.percent))
                                Text(job.status).font(.system(size: 11)).foregroundStyle(Theme.textMuted).lineLimit(1)
                            }
                            Button { item.cancel() } label: { Image(systemName: "xmark") }
                                .buttonStyle(.borderless)
                                .help(L.s("Cancel"))
                                .accessibilityLabel(L.s("Cancel"))
                                .accessibilityIdentifier("CancelDownloadButton")
                        }
                    } else if item.installed != nil {
                        HStack(spacing: 8) {
                            Label(L.s("Installed"), systemImage: "checkmark").foregroundStyle(Theme.success)
                                .accessibilityIdentifier("InstalledLabel")
                            Spacer(minLength: 0)
                            Button { item.delete() } label: { Image(systemName: "trash") }
                                .buttonStyle(.borderless)
                                .help(L.s("Delete"))
                                .accessibilityLabel(L.s("Delete"))
                                .accessibilityIdentifier("DeleteModelButton")
                        }
                    } else if item.selected != nil {
                        Button(L.s("Download")) { item.download() }
                            .buttonStyle(.glassProminent)
                            .frame(maxWidth: .infinity)
                            .accessibilityIdentifier("DownloadButton")
                    }
                }
                .frame(width: 170)
            }
            if let error = item.error {
                Text(error).font(.system(size: 12)).foregroundStyle(Theme.danger).frame(maxWidth: 470, alignment: .trailing)
                    .accessibilityIdentifier("DownloadError")
            }
        }
    }
}

private struct HuggingFacePanel: View {
    @Environment(AppModel.self) private var app

    var body: some View {
        @Bindable var vm = app.models
        HStack(alignment: .top, spacing: 16) {
            VStack(spacing: 10) {
                HStack {
                    TextField(L.s("SearchHf"), text: $vm.query)
                        .textFieldStyle(.roundedBorder)
                        .onSubmit { Task { await vm.search() } }
                        .accessibilityIdentifier("HfQuery")
                    Button { Task { await vm.search() } } label: { Image(systemName: "magnifyingglass") }
                        .buttonStyle(.glass)
                        .accessibilityLabel(L.s("SearchHf"))
                        .accessibilityIdentifier("HfSearchButton")
                }
                List(vm.results, selection: $vm.selectedRepoId) { repo in
                    VStack(alignment: .leading, spacing: 2) {
                        Text(repo.repo.id).font(.system(size: 13, design: .monospaced)).lineLimit(1)
                        Text(repo.subtitle).font(.system(size: 12)).foregroundStyle(Theme.textMuted)
                    }
                    .padding(.vertical, 4)
                    .tag(repo.id)
                }
                .accessibilityIdentifier("HfResults")
            }
            .frame(width: 360)

            if let repo = vm.selectedRepo {
                VStack(alignment: .leading, spacing: 12) {
                    Text(repo.repo.id).font(.system(size: 15, design: .monospaced))
                    if !repo.status.isEmpty { Text(repo.status).font(.system(size: 12)).foregroundStyle(Theme.textMuted) }
                    VariantControls(item: repo)
                    Text(L.s("HfHint")).font(.system(size: 12)).foregroundStyle(Theme.textMuted)
                }
                .padding(18)
                .glassEffect(.regular, in: .rect(cornerRadius: 20))
                .accessibilityElement(children: .contain)
                .accessibilityIdentifier("HfDetail")
            }
            Spacer(minLength: 0)
        }
        .padding(.bottom, 24)
    }
}

private struct InstalledList: View {
    @Environment(AppModel.self) private var app

    var body: some View {
        if app.modelStore.installed.isEmpty {
            Text(L.s("NoInstalled")).foregroundStyle(Theme.textMuted)
            Spacer()
        } else {
            List(app.modelStore.installed) { model in
                HStack(spacing: 14) {
                    VStack(alignment: .leading, spacing: 2) {
                        Text(model.displayName).fontWeight(.semibold).accessibilityIdentifier("InstalledName")
                        Text(model.repo).font(.system(size: 12, design: .monospaced)).foregroundStyle(Theme.textMuted)
                    }
                    Spacer()
                    Text(Fmt.bytes(model.sizeBytes)).foregroundStyle(Theme.textMuted)
                    Button { app.modelStore.delete(model, engine: app.engines) } label: { Image(systemName: "trash") }
                        .buttonStyle(.borderless)
                        .help(L.s("Delete"))
                        .accessibilityLabel(L.s("Delete"))
                        .accessibilityIdentifier("DeleteInstalledButton")
                }
                .padding(.vertical, 6)
                .accessibilityElement(children: .contain)
                .accessibilityIdentifier("InstalledRow")
            }
            .listStyle(.inset)
            .accessibilityIdentifier("InstalledList")
        }
    }
}

private struct LocalApiSettings: View {
    @Environment(AppModel.self) private var app
    @State private var enabled = AppSettings.current.apiEnabled
    @State private var port = AppSettings.current.apiPort
    @State private var lan = AppSettings.current.apiAllowLan
    @State private var requireKey = AppSettings.current.apiRequireKey
    @State private var key = AppSettings.current.apiKey

    var body: some View {
        Form {
            Section {
                Toggle(L.s("ApiServing"), isOn: $enabled).accessibilityIdentifier("ApiToggle")
                    .onChange(of: enabled) { save { $0.apiEnabled = enabled } }
                if !app.api.error.isEmpty {
                    Text(app.api.error).foregroundStyle(Theme.danger).accessibilityIdentifier("ApiError")
                }
            } header: {
                Text(L.s("ApiIntro")).font(.callout).foregroundStyle(.secondary).textCase(nil)
            }
            Section {
                LabeledContent(L.s("ApiBaseUrl")) {
                    HStack {
                        Text(app.api.baseURL).font(.system(.body, design: .monospaced)).textSelection(.enabled)
                        Button(L.s("Copy")) { copy(app.api.baseURL) }.buttonStyle(.glass)
                    }
                }
                TextField(L.s("ApiPort"), value: $port, format: .number.grouping(.never)).accessibilityIdentifier("PortBox")
                    .onSubmit { save { $0.apiPort = min(max(port, 1024), 65535) } }
                Toggle(L.s("ApiAllowLan"), isOn: $lan).onChange(of: lan) { save { $0.apiAllowLan = lan } }
                Text(L.s("ApiAllowLanHelp")).font(.caption).foregroundStyle(.secondary)
                Toggle(L.s("ApiRequireKey"), isOn: $requireKey).onChange(of: requireKey) { save(restart: false) { $0.apiRequireKey = requireKey } }
                LabeledContent(L.s("ApiKey")) {
                    HStack {
                        Text(key.isEmpty ? "-" : String(key.prefix(10)) + "…").font(.system(.body, design: .monospaced))
                        Button(L.s("Copy")) { copy(key) }.buttonStyle(.glass)
                        Button(L.s("ApiNewKey")) {
                            key = LocalApiServer.newKey()
                            save(restart: false) { $0.apiKey = key }
                        }.buttonStyle(.glass)
                    }
                }
            } header: { Text(L.s("ApiAccess")) }
            Section {
                Text("POST /v1/chat/completions   (OpenAI)\nPOST /v1/messages           (Anthropic)\nPOST /v1/messages/count_tokens\nGET  /v1/models")
                    .font(.system(size: 12, design: .monospaced)).textSelection(.enabled)
                Text(L.s("ApiClaudeCode")).font(.caption).foregroundStyle(.secondary)
                Text("export ANTHROPIC_BASE_URL=\(app.api.baseURL)\nexport ANTHROPIC_AUTH_TOKEN=\(requireKey ? key : "none")\nclaude --model \(app.chat.models.first { $0.format == "gguf" }.map(app.api.apiName) ?? "gemma-4-12b")")
                    .font(.system(size: 12, design: .monospaced)).textSelection(.enabled)
            } header: { Text(L.s("ApiEndpoints")) }
            Section {
                if app.api.requests.isEmpty {
                    Text(L.s("ApiNoRequests")).foregroundStyle(.secondary)
                } else {
                    ForEach(app.api.requests) { r in
                        HStack(spacing: 12) {
                            Text(r.time).foregroundStyle(.secondary)
                            Text(r.client).frame(width: 110, alignment: .leading).lineLimit(1)
                            Text(r.path).font(.system(size: 12, design: .monospaced))
                            Text(r.model).font(.system(size: 12, design: .monospaced)).lineLimit(1)
                            Spacer()
                            Text(r.status).accessibilityIdentifier("ApiRequestStatus")
                        }
                    }
                }
            } header: { Text(L.s("ApiRecent")) }
        }
        .formStyle(.grouped)
        .frame(maxWidth: 860)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .onAppear {
            if key.isEmpty {
                key = LocalApiServer.newKey()
                save(restart: false) { $0.apiKey = key }
            }
        }
    }

    private func save(restart: Bool = true, _ change: (inout AppSettings) -> Void) {
        change(&AppSettings.current)
        AppSettings.save()
        if restart { app.api.restart() }
    }

    private func copy(_ text: String) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
    }
}
