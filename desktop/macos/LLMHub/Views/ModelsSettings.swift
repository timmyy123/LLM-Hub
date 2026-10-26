import AppKit
import SwiftUI
import UniformTypeIdentifiers

/// Settings › Models: the LLM Hub catalog, Hugging Face search and installed models, as on the design canvas.
struct ModelsSettings: View {
    @Environment(AppModel.self) private var app

    var body: some View {
        @Bindable var vm = app.models
        VStack(alignment: .leading, spacing: 14) {
            HStack(alignment: .top, spacing: 12) {
                VStack(alignment: .leading, spacing: 4) {
                    Text(L.s("SettingsModels")).font(.system(size: 20, weight: .semibold))
                    Text(vm.hardwareText).font(.system(size: 12)).foregroundStyle(Theme.textMuted)
                }
                Spacer()
                Button(L.s("ModelsAddLocalFile"), action: pickLocalFile)
                    .buttonStyle(.glass)
                    .accessibilityIdentifier("AddLocalFileButton")
            }
            if !vm.localFileMessage.isEmpty {
                Text(vm.localFileMessage)
                    .font(.system(size: 12))
                    .foregroundStyle(vm.localFileFailed ? Theme.danger : Theme.success)
                    .textSelection(.enabled)
                    .accessibilityIdentifier("AddLocalFileMessage")
            }
            Picker("", selection: $vm.tab) {
                Text(L.s("TabCatalog")).tag(ModelsTab.catalog)
                Text(L.s("TabHuggingFace")).tag(ModelsTab.huggingFace)
                let count = app.modelStore.installed.count
                Text(count > 0 ? "\(L.s("TabInstalled")) · \(count)" : L.s("TabInstalled")).tag(ModelsTab.installed)
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .fixedSize()
            .accessibilityIdentifier("ModelTabs")

            switch vm.tab {
            case .catalog: CatalogTab()
            case .huggingFace: HuggingFaceTab()
            case .installed: InstalledList()
            }
        }
        .padding(EdgeInsets(top: 24, leading: 28, bottom: 0, trailing: 28))
        .task { await app.models.loadCatalog() }
    }

    /// A .gguf or .litertlm stays where it is; LLM Hub only records its path.
    private func pickLocalFile() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = ["gguf", "litertlm"].compactMap { UTType(filenameExtension: $0) }
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        panel.prompt = L.s("ModelsAddLocalFile")
        panel.begin { response in
            guard response == .OK, let url = panel.url else { return }
            app.models.addLocalFile(url)
        }
    }
}

private extension ModelFit {
    /// The design's green for a fit, amber for partly on the CPU.
    var color: Color {
        switch self {
        case .gpu: Theme.success
        case .partial: Theme.warning
        case .tooLarge: Theme.danger
        }
    }
}

// MARK: - Catalog

private struct CatalogTab: View {
    @Environment(AppModel.self) private var app

    var body: some View {
        @Bindable var vm = app.models
        VStack(alignment: .leading, spacing: 14) {
            GlassEffectContainer(spacing: 12) {
                HStack(spacing: 12) {
                    EngineCard(name: "llama.cpp", line: vm.llamaEngineLine, active: vm.llamaInstalled)
                        .accessibilityIdentifier("EngineCardLlama")
                    EngineCard(name: "LiteRT-LM", line: vm.liteRtEngineLine, active: true)
                        .accessibilityIdentifier("EngineCardLiteRt")
                }
            }
            GlassEffectContainer(spacing: 8) {
                HStack(spacing: 8) {
                    ForEach(ModelsViewModel.filters, id: \.key) { filter in
                        FilterChip(label: filter.label, selected: vm.filter == filter.key) { vm.filter = filter.key }
                            .accessibilityIdentifier("Filter-\(filter.key)")
                    }
                }
            }
            List(vm.visibleCatalog) { item in
                CatalogRow(item: item)
            }
            .listStyle(.inset)
            .accessibilityIdentifier("CatalogList")
        }
    }
}

/// "llama.cpp / GGUF · Metal · running" above the catalog. The dot is green once the engine can run without a download.
private struct EngineCard: View {
    let name: String
    let line: String
    let active: Bool

    var body: some View {
        HStack(spacing: 12) {
            Circle().fill(active ? Theme.success : Theme.textFaint).frame(width: 8, height: 8)
            VStack(alignment: .leading, spacing: 1) {
                Text(name).fontWeight(.medium)
                Text(line).font(.system(size: 12)).foregroundStyle(Theme.textMuted)
            }
            Spacer(minLength: 0)
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
        .frame(maxWidth: .infinity)
        .glassEffect(.regular, in: .rect(cornerRadius: 12))
        .accessibilityElement(children: .combine)
    }
}

private struct FilterChip: View {
    let label: String
    let selected: Bool
    let action: () -> Void

    var body: some View {
        if selected {
            Button(label, action: action).buttonStyle(.glassProminent).accessibilityAddTraits(.isSelected)
        } else {
            Button(label, action: action).buttonStyle(.glass)
        }
    }
}

private struct CatalogRow: View {
    let item: CatalogItem

    var body: some View {
        HStack(spacing: 14) {
            VStack(alignment: .leading, spacing: 3) {
                HStack(spacing: 8) {
                    Text(item.model.name).fontWeight(.semibold).accessibilityIdentifier("CatalogName")
                    Tag(text: item.engineLabel)
                    ForEach(item.model.tags, id: \.self) { Tag(text: CatalogItem.tagLabel($0), neutral: true) }
                }
                Text(item.model.maker).font(.system(size: 12)).foregroundStyle(Theme.textMuted).lineLimit(1)
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
}

/// Variant picker, how the selected one fits this Mac, then download / progress / installed.
struct VariantControls: View {
    let item: DownloadableItem

    var body: some View {
        VStack(alignment: .trailing, spacing: 4) {
            HStack(spacing: 12) {
                if !item.variants.isEmpty {
                    Picker("", selection: Binding(get: { item.selectedId }, set: { item.selectedId = $0 })) {
                        ForEach(item.variants) { option in
                            Text(option.label).tag(Optional(option.id))
                        }
                    }
                    .labelsHidden()
                    .frame(width: 230)
                    .accessibilityIdentifier("VariantPicker")
                }
                Text(item.selected?.fitText ?? "")
                    .font(.system(size: 12))
                    .foregroundStyle(item.selected?.fit.color ?? Theme.textMuted)
                    .frame(width: 130, alignment: .leading)
                    .accessibilityIdentifier("CatalogFit")
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
                .frame(width: 150)
            }
            if let error = item.error {
                Text(error).font(.system(size: 12)).foregroundStyle(Theme.danger).frame(maxWidth: 470, alignment: .trailing)
                    .accessibilityIdentifier("DownloadError")
            }
        }
    }
}

// MARK: - Hugging Face

private struct HuggingFaceTab: View {
    @Environment(AppModel.self) private var app

    var body: some View {
        @Bindable var vm = app.models
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 10) {
                TextField(L.s("SearchHf"), text: $vm.query)
                    .textFieldStyle(.roundedBorder)
                    .onSubmit { vm.search() }
                    .accessibilityIdentifier("HfQuery")
                Button { vm.search() } label: { Image(systemName: "magnifyingglass") }
                    .buttonStyle(.glass)
                    .accessibilityLabel(L.s("SearchHf"))
                    .accessibilityIdentifier("HfSearchButton")
                if vm.isSearching { ProgressView().controlSize(.small) }
                Picker("", selection: $vm.hfFormat) {
                    ForEach(ModelsViewModel.hfFormats, id: \.key) { Text($0.label).tag($0.key) }
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                .fixedSize()
                .accessibilityIdentifier("HfFormat")
                Picker(L.s("HfSort"), selection: $vm.hfSort) {
                    ForEach(ModelsViewModel.hfSorts, id: \.key) { Text($0.label).tag($0.key) }
                }
                .pickerStyle(.menu)
                .fixedSize()
                .accessibilityIdentifier("HfSort")
            }
            if let error = vm.searchError {
                Text(error).font(.system(size: 12)).foregroundStyle(Theme.warning).accessibilityIdentifier("HfSearchError")
            }
            HStack(alignment: .top, spacing: 16) {
                List(vm.results, selection: $vm.selectedRepoId) { repo in
                    VStack(alignment: .leading, spacing: 3) {
                        Text(repo.repo.id).font(.system(size: 13, design: .monospaced)).lineLimit(1).truncationMode(.middle)
                        HStack(spacing: 6) {
                            Tag(text: repo.formatLabel)
                            Text(repo.meta).font(.system(size: 12)).foregroundStyle(Theme.textMuted)
                        }
                    }
                    .padding(.vertical, 4)
                    .tag(repo.id)
                    .accessibilityElement(children: .combine)
                    .accessibilityIdentifier("HfResult")
                }
                .accessibilityIdentifier("HfResults")
                .frame(width: 340)

                if let repo = vm.selectedRepo {
                    RepoDetail(repo: repo).id(repo.id)
                }
                Spacer(minLength: 0)
            }
        }
        .padding(.bottom, 24)
        // The tab opens on what is trending, like the design.
        .onAppear { if vm.results.isEmpty && !vm.isSearching { vm.search() } }
    }
}

/// The selected repo: name, what it is, gated notice, and every file with how it fits this Mac.
private struct RepoDetail: View {
    @Environment(AppModel.self) private var app
    @Environment(\.openURL) private var openURL
    let repo: RepoItem
    @State private var token = AppSettings.current.huggingFaceToken
    @State private var showTokenField = false

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                HStack(alignment: .top, spacing: 12) {
                    VStack(alignment: .leading, spacing: 2) {
                        Text(repo.repo.id).font(.system(size: 15, design: .monospaced)).textSelection(.enabled)
                        Text(repo.detail).font(.system(size: 12)).foregroundStyle(Theme.textMuted).accessibilityIdentifier("HfDetailLine")
                    }
                    Spacer(minLength: 0)
                    Link(L.s("HfOpenOnHub"), destination: repo.url).font(.system(size: 13)).accessibilityIdentifier("HfOpenOnHub")
                }
                if repo.repo.gated { gatedBanner }
                if !repo.status.isEmpty {
                    Text(repo.status).font(.system(size: 12)).foregroundStyle(Theme.textMuted)
                }
                if !repo.rows.isEmpty { FileTable(repo: repo) }
                if let error = repo.error {
                    Text(error).font(.system(size: 12)).foregroundStyle(Theme.danger).accessibilityIdentifier("DownloadError")
                }
                Text(L.s("HfMmprojNote")).font(.system(size: 12)).foregroundStyle(Theme.textFaint)
            }
            .padding(18)
        }
        .frame(maxWidth: 820, maxHeight: .infinity)
        .glassEffect(.regular, in: .rect(cornerRadius: 20))
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("HfDetail")
    }

    /// Gated repos need the license accepted on the Hub. Without a saved token the banner offers the token page and a
    /// field that saves to the same setting as Settings › General.
    private var gatedBanner: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 10) {
                Text(L.s("HfGated")).font(.system(size: 13))
                Spacer(minLength: 0)
                if token.trimmingCharacters(in: .whitespaces).isEmpty && !showTokenField {
                    Button(L.s("HfSignIn")) {
                        openURL(URL(string: "\(HuggingFaceClient.host)/settings/tokens")!)
                        showTokenField = true
                    }
                    .buttonStyle(.glass)
                    .accessibilityIdentifier("HfSignInButton")
                }
            }
            if showTokenField {
                SecureField(L.s("HfToken"), text: $token)
                    .textFieldStyle(.roundedBorder)
                    .accessibilityIdentifier("HfTokenField")
                    .onChange(of: token) {
                        AppSettings.current.huggingFaceToken = token
                        AppSettings.save()
                    }
                Text(L.s("HfSignInHelp")).font(.system(size: 12)).foregroundStyle(Theme.textMuted)
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
        .background(Theme.warning.opacity(0.12), in: .rect(cornerRadius: 10))
        .overlay(RoundedRectangle(cornerRadius: 10).strokeBorder(Theme.warning.opacity(0.35)))
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("HfGatedBanner")
    }
}

/// Every file smallest first, with how it fits and its own Download button; the best fit is highlighted. A download in
/// progress shows its bar in place of the button.
private struct FileTable: View {
    @Environment(AppModel.self) private var app
    let repo: RepoItem

    var body: some View {
        VStack(spacing: 0) {
            columns(Text(L.s("HfColFile")), Text(L.s("HfColFit")), Text(L.s("HfColAction")).hidden())
                .font(.system(size: 12, weight: .medium))
                .foregroundStyle(Theme.textFaint)
                .padding(.vertical, 6)
            ForEach(repo.rows) { row in
                Divider()
                columns(
                    VStack(alignment: .leading, spacing: 1) {
                        Text(row.quantAndSize).font(.system(size: 13)).lineLimit(1)
                        Text(row.file).font(.system(size: 11, design: .monospaced)).foregroundStyle(Theme.textMuted)
                            .lineLimit(1).truncationMode(.middle).help(row.file)
                    },
                    Text(row.fitText).font(.system(size: 13)).foregroundStyle(row.level == .bad ? Theme.danger : row.level == .warn ? Theme.warning : Theme.success),
                    action(row))
                    .padding(.vertical, 7)
                    .background(row.isBest ? Theme.accent.opacity(0.08) : .clear, in: .rect(cornerRadius: 6))
                    .accessibilityElement(children: .contain)
                    .accessibilityIdentifier("HfFileRow-\(row.option.variant.quant)")
            }
        }
        .accessibilityIdentifier("HfFileTable")
    }

    /// File / This computer / Action: three columns keep the table readable in a narrow window.
    private func columns(_ file: some View, _ fit: some View, _ action: some View) -> some View {
        HStack(spacing: 12) {
            file.frame(minWidth: 140, maxWidth: .infinity, alignment: .leading)
            fit.frame(width: 140, alignment: .leading)
            action.frame(width: 120, alignment: .trailing)
        }
        .padding(.horizontal, 10)
    }

    @ViewBuilder private func action(_ row: HfFileRow) -> some View {
        if let job = app.downloads.find(row.id) {
            HStack(spacing: 6) {
                ProgressView(value: job.percent, total: 100)
                    .accessibilityIdentifier("DownloadProgress")
                    .accessibilityValue(String(format: "%.1f", job.percent))
                Button { app.downloads.cancel(job) } label: { Image(systemName: "xmark") }
                    .buttonStyle(.borderless)
                    .help(L.s("Cancel"))
                    .accessibilityLabel(L.s("Cancel"))
                    .accessibilityIdentifier("CancelDownloadButton")
            }
        } else if app.modelStore.find(row.id) != nil {
            Label(L.s("Installed"), systemImage: "checkmark").font(.system(size: 13)).foregroundStyle(Theme.success)
                .accessibilityIdentifier("InstalledLabel")
        } else if row.isBest {
            Button(L.s("Download")) { repo.downloadRow(row) }
                .buttonStyle(.glassProminent)
                .accessibilityIdentifier("DownloadButton")
        } else {
            Button(L.s("Download")) { repo.downloadRow(row) }
                .buttonStyle(.glass)
                .disabled(row.level == .bad)
                .accessibilityIdentifier("DownloadButton")
        }
    }
}

// MARK: - Installed

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
                        // A local file shows where it lives; deleting it here only forgets it.
                        Text(model.isLocalFile ? (model.primaryPath.path as NSString).abbreviatingWithTildeInPath : model.repo)
                            .font(.system(size: 12, design: .monospaced)).foregroundStyle(Theme.textMuted)
                            .lineLimit(1).truncationMode(.middle)
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
