import Foundation
import Observation

/// A downloadable quantization with how well it fits this Mac.
struct VariantOption: Identifiable, Hashable {
    let variant: GgufVariant
    let fit: ModelFit
    var id: String { variant.modelId }
    var label: String { "\(variant.quant) · \(Fmt.bytes(variant.totalBytes))" }
    var fitText: String {
        switch fit {
        case .gpu: return L.s("FitGpu")
        case .partial: return L.s("FitPartial")
        case .tooLarge: return L.s("FitTooLarge")
        }
    }
}

/// Download/installed state for one repo, shared by the catalog and the Hugging Face tab.
@MainActor @Observable
class DownloadableItem: Identifiable {
    let id: String
    let displayName: String
    let catalogId: String?
    let tags: [String]
    let contextLength: Int
    var variants: [VariantOption] = []
    var selectedId: String?
    var isLoading = false
    var status = ""
    var nothingFits = false
    unowned let app: AppModel

    init(id: String, displayName: String, catalogId: String?, tags: [String], contextLength: Int, app: AppModel) {
        self.id = id
        self.displayName = displayName
        self.catalogId = catalogId
        self.tags = tags
        self.contextLength = contextLength
        self.app = app
    }

    var selected: VariantOption? { variants.first { $0.id == selectedId } }
    var job: DownloadJob? { selectedId.flatMap { app.downloads.find($0) } }
    var installed: InstalledModel? { selectedId.flatMap { app.modelStore.find($0) } }
    var error: String? { selectedId.flatMap { app.downloads.errors[$0] } }

    func setVariants(_ list: [GgufVariant], preferred: String?) {
        variants = list.map { VariantOption(variant: $0, fit: app.hardware.fit($0.weightBytes)) }
            .filter { app.modelStore.find($0.id) != nil || app.hardware.shouldOffer($0.fit) }
        selectedId = (variants.first { app.modelStore.find($0.id) != nil }
            ?? variants.first { $0.variant.quant.caseInsensitiveCompare(preferred ?? "") == .orderedSame }
            ?? variants.first { $0.variant.quant == "Q4_K_M" }
            ?? variants.last { $0.variant.quant.contains("Q4") }
            ?? variants.last)?.id
        nothingFits = variants.isEmpty && !list.isEmpty
    }

    func download() {
        guard let selected else { return }
        app.downloads.start(selected.variant, name: displayName, catalogId: catalogId, tags: tags, contextLength: contextLength)
    }

    func cancel() { job.map { app.downloads.cancel($0) } }
    func delete() { installed.map { app.modelStore.delete($0, engine: app.engines) } }
}

@MainActor @Observable
final class CatalogItem: DownloadableItem {
    let model: CatalogModel

    init(_ model: CatalogModel, app: AppModel) {
        self.model = model
        super.init(id: model.id, displayName: model.name, catalogId: model.id, tags: model.tags, contextLength: model.contextLength, app: app)
    }

    var ggufSource: CatalogSource? { model.sources.first { $0.format == "gguf" } }
    var liteRtSource: CatalogSource? { model.sources.first { $0.format == "litertlm" } }
    var engineLabel: String { ggufSource != nil && liteRtSource != nil ? "llama.cpp · LiteRT-LM" : ggufSource != nil ? "llama.cpp" : "LiteRT-LM" }
    var subtitle: String { ([model.maker] + model.tags.map(Self.tagLabel)).joined(separator: " · ") }

    static func tagLabel(_ tag: String) -> String {
        switch tag {
        case "agent": return L.s("TagAgent")
        case "vision": return L.s("TagVision")
        case "audio": return L.s("TagAudio")
        case "thinking": return L.s("TagThinking")
        case "embedding": return L.s("TagEmbedding")
        default: return tag
        }
    }

    /// The filter chips: all, chat, vision, agent, embedding, or fits (some quant runs fully in GPU memory; on Apple
    /// silicon that is the share of unified memory Metal may use).
    func matches(_ filter: String) -> Bool {
        switch filter {
        case "chat": return !model.tags.contains("embedding")
        case "vision": return model.tags.contains("vision")
        case "agent": return model.tags.contains("agent")
        case "embedding": return model.tags.contains("embedding")
        case "fits": return variants.contains { $0.fit == .gpu }
        default: return true
        }
    }

    func load() async {
        isLoading = true
        status = L.s("LoadingVariants")
        do {
            var list: [GgufVariant] = []
            if let gguf = ggufSource { list += try await app.hub.ggufVariants(gguf.repo) }
            if let litert = liteRtSource { list += try await app.hub.liteRtVariants(litert.repo) }
            // /v1/embeddings is text only: EmbeddingGemma 2's vision projector (555 MB) is not downloaded.
            if model.tags.contains("embedding") {
                list = list.map { GgufVariant(repo: $0.repo, quant: $0.quant, files: $0.files, mmproj: nil, format: $0.format) }
            }
            setVariants(list, preferred: ggufSource?.preferredQuant)
            status = nothingFits ? L.s("NothingFits") : ""
        } catch {
            Log.error("Variants for \(model.id)", error)
            status = L.s("VariantsError")
        }
        isLoading = false
    }
}

/// One file (or split set) in a Hugging Face repo's table, with how it fits this Mac.
struct HfFileRow: Identifiable {
    enum Level { case ok, warn, bad }
    let option: VariantOption
    /// "Best fit", "Fits", "Fits, lower quality", "Fits, slower" or "Too large".
    let fitText: String
    let level: Level
    let isBest: Bool
    var id: String { option.id }
    var file: String { option.variant.files.first?.fileName ?? "" }
    var size: String { Fmt.bytes(option.variant.totalBytes) }
    /// "Q4_K_M · 7.1 GB", the row's title above the file name.
    var quantAndSize: String { "\(option.variant.quant) · \(size)" }
}

@MainActor @Observable
final class RepoItem: DownloadableItem {
    let repo: HfRepo
    /// The catalog model that downloads from this repo, if any.
    let catalogModel: CatalogModel?
    /// Every variant in the repo, including the ones too large to offer.
    private var all: [GgufVariant] = []

    init(_ repo: HfRepo, app: AppModel) {
        self.repo = repo
        catalogModel = app.catalog.models.first { $0.sources.contains { $0.repo.caseInsensitiveCompare(repo.id) == .orderedSame } }
        let name = repo.id.split(separator: "/").last.map(String.init)?.replacingOccurrences(of: "-GGUF", with: "", options: .caseInsensitive) ?? repo.id
        super.init(id: repo.id, displayName: name, catalogId: nil, tags: [], contextLength: 0, app: app)
    }

    var subtitle: String { L.f("RepoStats", repo.downloads.formatted(), repo.likes.formatted()) }
    var formatLabel: String { repo.format == "litertlm" ? "LiteRT-LM" : "GGUF" }
    var url: URL { URL(string: "\(HuggingFaceClient.host)/\(repo.id)")! }
    /// "In LLM Hub catalog" or "Community quants" under the repo name.
    var meta: String { L.s(catalogModel == nil ? "HfCommunity" : "HfInCatalog") }
    /// "Google Gemma 4 12B · Vision · runs on llama.cpp" above the file table.
    var detail: String {
        let head = catalogModel.map { m in ["\(m.maker) \(m.name)"] + m.tags.map(CatalogItem.tagLabel) } ?? [subtitle]
        return (head + [L.f("HfRunsOn", repo.format == "litertlm" ? "LiteRT-LM" : "llama.cpp")]).joined(separator: " · ")
    }

    /// The file table, smallest first. The selected quant is the best fit; smaller ones that fit trade quality.
    var rows: [HfFileRow] {
        let best = selected?.variant
        let bestSize = best?.weightBytes ?? 0
        return all.sorted { $0.weightBytes < $1.weightBytes }.map { v in
            let option = variants.first { $0.variant == v } ?? VariantOption(variant: v, fit: app.hardware.fit(v.weightBytes))
            let (text, level): (String, HfFileRow.Level) = switch option.fit {
            case .tooLarge: (L.s("FitTooLarge"), .bad)
            case .partial: (L.s("FitSlower"), .warn)
            case .gpu where v == best: (L.s("FitBest"), .ok)
            case .gpu where v.weightBytes < bestSize: (L.s("FitLower"), .ok)
            case .gpu: (L.s("FitOk"), .ok)
            }
            return HfFileRow(option: option, fitText: text, level: level, isBest: v == best)
        }
    }

    func downloadRow(_ row: HfFileRow) {
        if !variants.contains(row.option) { variants.append(row.option) }
        selectedId = row.option.id
        download()
    }

    func load() async {
        isLoading = true
        status = L.s("LoadingVariants")
        do {
            let list = repo.format == "litertlm" ? try await app.hub.liteRtVariants(repo.id) : try await app.hub.ggufVariants(repo.id)
            all = list
            setVariants(list, preferred: "Q4_K_M")
            status = list.isEmpty ? L.s("HfNoGguf") : nothingFits ? L.s("NothingFits") : ""
        } catch {
            Log.error("Variants for \(repo.id)", error)
            status = L.s("VariantsError")
        }
        isLoading = false
    }
}

enum ModelsTab: String, CaseIterable { case catalog, huggingFace, installed }

@MainActor @Observable
final class ModelsViewModel {
    private weak var app: AppModel?
    private var catalogLoaded = false

    var catalog: [CatalogItem] = []
    var results: [RepoItem] = []
    var selectedRepoId: String? {
        didSet { if let repo = selectedRepo, repo.variants.isEmpty, !repo.isLoading { Task { await repo.load() } } }
    }
    var query = ""
    var isSearching = false
    var tab = ModelsTab.catalog
    var filter = "all"
    var hfFormat = "all" { didSet { if hfFormat != oldValue { search() } } }
    var hfSort = "trendingScore" { didSet { if hfSort != oldValue { search() } } }
    /// The running search; a new one cancels it.
    private(set) var searchTask: Task<Void, Never>?
    /// What "Add local file" did ("Added …" or why not), under the page title.
    var localFileMessage = ""
    var localFileFailed = false

    static var filters: [(key: String, label: String)] {
        [("all", L.s("FilterAll")), ("chat", L.s("FilterChat")), ("vision", L.s("TagVision")), ("agent", L.s("TagAgent")),
         ("embedding", L.s("TagEmbedding")), ("fits", L.s("FilterFits"))]
    }

    static var hfFormats: [(key: String, label: String)] {
        [("all", L.s("HfAllRunnable")), ("gguf", "GGUF"), ("litertlm", "LiteRT-LM")]
    }

    static var hfSorts: [(key: String, label: String)] {
        [("trendingScore", L.s("HfSortTrending")), ("downloads", L.s("HfSortDownloads")), ("lastModified", L.s("HfSortUpdated"))]
    }

    func attach(_ app: AppModel) {
        self.app = app
        catalog = app.catalog.models.map { CatalogItem($0, app: app) }
    }

    var selectedRepo: RepoItem? { results.first { $0.id == selectedRepoId } }
    /// The catalog cards that pass the selected filter chip.
    var visibleCatalog: [CatalogItem] { catalog.filter { $0.matches(filter) } }

    var hardwareText: String {
        guard let hw = app?.hardware else { return "" }
        // Apple Silicon GPUs use unified memory, so show it as shared rather than VRAM.
        return hw.gpuBytes > 0
            ? L.f("HardwareSummaryShared", hw.gpuName, "\(hw.ramGB)")
            : L.f("HardwareSummaryCpu", "\(hw.ramGB)")
    }

    // Engine cards: "GGUF · Metal · running" and ".litertlm · GPU · ready".
    var llamaRunning: Bool { app?.engines.loadedModel?.format == "gguf" }
    var liteRtRunning: Bool { app?.engines.loadedModel?.format == "litertlm" }
    /// llama.cpp downloads on first use; until then its card says so.
    var llamaInstalled: Bool { llamaRunning || app?.engine.isInstalled == true }

    var llamaEngineLine: String {
        "GGUF · Metal · " + L.s(llamaRunning ? "EngineCardRunning" : llamaInstalled ? "EngineCardReady" : "EngineCardOnDemand")
    }

    var liteRtEngineLine: String {
        ".litertlm · \(app?.engines.liteRtBackend?.uppercased() ?? "GPU") · " + L.s(liteRtRunning ? "EngineCardRunning" : "EngineCardReady")
    }

    /// Adds a .gguf or .litertlm from disk (see `ModelStore.addLocalFile`) and says how it went.
    func addLocalFile(_ url: URL) {
        guard let app else { return }
        do {
            let model = try app.modelStore.addLocalFile(url)
            localFileMessage = L.f("ModelsAdded", model.displayName)
            localFileFailed = false
        } catch {
            Log.error("Add local model \(url.path)", error)
            localFileMessage = L.f("ModelsAddFailed", url.lastPathComponent, error.localizedDescription)
            localFileFailed = true
        }
    }

    func loadCatalog() async {
        guard !catalogLoaded else { return }
        catalogLoaded = true
        await withTaskGroup(of: Void.self) { group in
            var running = 0
            for item in catalog {
                if running >= 4 { await group.next(); running -= 1 }
                group.addTask { await item.load() }
                running += 1
            }
        }
        // Models with no quant that fits this Mac are not offered at all.
        catalog.removeAll { $0.nothingFits }
    }

    /// Searches with the format and sort pickers. "All runnable" asks for both library filters and merges them by the
    /// sort key.
    func search() {
        guard let app else { return }
        searchTask?.cancel()
        let query = query.trimmingCharacters(in: .whitespaces)
        let sort = hfSort
        let hub = app.hub
        let filters = hfFormat == "all" ? ["gguf", "litert-lm"] : [hfFormat == "litertlm" ? "litert-lm" : "gguf"]
        isSearching = true
        searchTask = Task {
            do {
                var merged: [HfRepo] = []
                try await withThrowingTaskGroup(of: [HfRepo].self) { group in
                    for filter in filters { group.addTask { try await hub.search(query, filter: filter, sort: sort) } }
                    for try await list in group { merged += list }
                }
                if filters.count > 1 {
                    switch sort {
                    case "downloads": merged.sort { $0.downloads > $1.downloads }
                    case "lastModified": merged.sort { ($0.lastModified ?? .distantPast) > ($1.lastModified ?? .distantPast) }
                    default: merged.sort { ($0.trending, $0.downloads) > ($1.trending, $1.downloads) }
                    }
                }
                try Task.checkCancellation()
                results = merged.prefix(40).map { RepoItem($0, app: app) }
                selectedRepoId = results.first?.id
            } catch {
                // A newer search cancelled this one (URLSession reports that as URLError.cancelled); it owns the spinner.
                if Task.isCancelled { return }
                Log.error("HF search failed", error)
            }
            isSearching = false
        }
    }
}
