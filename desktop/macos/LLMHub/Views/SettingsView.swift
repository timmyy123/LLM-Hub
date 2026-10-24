import SwiftUI

struct SettingsView: View {
    let section: SettingsSection

    var body: some View {
        switch section {
        case .general: GeneralSettings()
        case .models: ModelsSettings()
        case .api: LocalApiSettings()
        case .engines: EnginesSettings()
        case .speech: SpeechSettings()
        case .studio: StudioSettings()
        case .agent: AgentSettings()
        case .websearch: WebSearchSettings()
        case .storage: StorageSettings()
        }
    }
}

private struct GeneralSettings: View {
    @State private var language = L.override
    @State private var languageChanged = false
    @State private var context = AppSettings.current.contextLength
    @State private var token = AppSettings.current.huggingFaceToken
    @State private var appearance = AppSettings.current.appearance

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
            Picker(L.s("Appearance"), selection: $appearance) {
                Text(L.s("AppearanceDark")).tag("dark")
                Text(L.s("AppearanceLight")).tag("light")
                Text(L.s("AppearanceSystem")).tag("system")
            }
            .accessibilityIdentifier("AppearanceBox")
            .onChange(of: appearance) {
                AppSettings.current.appearance = appearance
                AppSettings.save()
                AppDelegate.applyAppearance()
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

private struct LocalApiSettings: View {
    @Environment(AppModel.self) private var app
    @State private var enabled = AppSettings.current.apiEnabled
    @State private var port = AppSettings.current.apiPort
    @State private var lan = AppSettings.current.apiAllowLan
    @State private var requireKey = AppSettings.current.apiRequireKey
    @State private var key = AppSettings.current.apiKey
    @State private var unloadAfter = AppSettings.current.apiUnloadAfterMinutes

    /// "llama.cpp", "LiteRT-LM" or "Apple Intelligence", plus Embedding for embedding models.
    static func engineLabel(_ model: InstalledModel) -> String {
        let engine = model.format == "litertlm" ? "LiteRT-LM" : model.format == "apple" ? "Apple Intelligence" : "llama.cpp"
        return model.isEmbedding ? "\(engine) · \(L.s("TagEmbedding"))" : engine
    }

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
                VStack(alignment: .leading, spacing: 6) {
                    Text(L.s("ApiOpenAiCompatible")).font(.system(size: 13, weight: .semibold))
                    ForEach(LocalApiServer.endpoints.filter { !$0.anthropic }) { EndpointRow(endpoint: $0) }
                    Text(L.s("ApiAnthropicCompatible")).font(.system(size: 13, weight: .semibold)).padding(.top, 8)
                    ForEach(LocalApiServer.endpoints.filter(\.anthropic)) { EndpointRow(endpoint: $0) }
                }
                .accessibilityElement(children: .contain)
                .accessibilityIdentifier("ApiEndpointList")
                Text(L.s("ApiClaudeCode")).font(.caption).foregroundStyle(.secondary)
                Text("export ANTHROPIC_BASE_URL=\(app.api.baseURL)\nexport ANTHROPIC_AUTH_TOKEN=\(requireKey ? key : "none")\nclaude --model \(app.chat.models.first { $0.format == "gguf" }.map(app.api.apiName) ?? "gemma-4-12b")")
                    .font(.system(size: 12, design: .monospaced)).textSelection(.enabled)
            } header: { Text(L.s("ApiEndpoints")) }
            Section {
                HStack(alignment: .top) {
                    VStack(alignment: .leading, spacing: 2) {
                        Text(L.s("ApiModelsOffered")).fontWeight(.semibold)
                        Text(L.s("ApiModelsOfferedHelp")).font(.caption).foregroundStyle(.secondary)
                    }
                    Spacer()
                    Picker(L.s("ApiUnloadAfter"), selection: $unloadAfter) {
                        ForEach(LocalApiServer.unloadChoices, id: \.self) { Text(LocalApiServer.unloadLabel($0)).tag($0) }
                    }
                    .fixedSize()
                    .accessibilityIdentifier("ApiUnloadPicker")
                    .onChange(of: unloadAfter) { save(restart: false) { $0.apiUnloadAfterMinutes = unloadAfter } }
                }
                ForEach(app.api.offeredModels) { model in
                    HStack(spacing: 12) {
                        Text(app.api.apiName(model)).font(.system(size: 12, design: .monospaced)).textSelection(.enabled).lineLimit(1).truncationMode(.middle)
                        Spacer()
                        Text(Self.engineLabel(model)).font(.system(size: 12)).foregroundStyle(.secondary)
                    }
                    .accessibilityIdentifier("ApiOfferedModel")
                }
            }
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

/// One endpoint in Settings › Local API: method badge, path, what serves it.
private struct EndpointRow: View {
    let endpoint: ApiEndpoint

    var body: some View {
        HStack(spacing: 10) {
            Text(endpoint.method)
                .font(.system(size: 11, design: .monospaced))
                .foregroundStyle(Color.accentColor)
                .padding(.horizontal, 6)
                .padding(.vertical, 1)
                .background(Color.accentColor.opacity(0.12), in: .rect(cornerRadius: 5))
                .frame(width: 46, alignment: .leading)
            Text(endpoint.path).font(.system(size: 13, design: .monospaced)).textSelection(.enabled)
            Text(endpoint.servedBy).font(.system(size: 12)).foregroundStyle(.secondary)
            Spacer(minLength: 0)
        }
        .padding(.vertical, 1)
    }
}
