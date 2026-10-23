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
    var subtitle: String {
        ([model.maker] + model.tags.map { tag in
            switch tag {
            case "vision": return L.s("TagVision")
            case "audio": return L.s("TagAudio")
            case "thinking": return L.s("TagThinking")
            default: return tag
            }
        }).joined(separator: " · ")
    }

    func load() async {
        isLoading = true
        status = L.s("LoadingVariants")
        do {
            var list: [GgufVariant] = []
            if let gguf = ggufSource { list += try await app.hub.ggufVariants(gguf.repo) }
            if let litert = liteRtSource { list += try await app.hub.liteRtVariants(litert.repo) }
            setVariants(list, preferred: ggufSource?.preferredQuant)
            status = nothingFits ? L.s("NothingFits") : ""
        } catch {
            Log.error("Variants for \(model.id)", error)
            status = L.s("VariantsError")
        }
        isLoading = false
    }
}

@MainActor @Observable
final class RepoItem: DownloadableItem {
    let repo: HfRepo

    init(_ repo: HfRepo, app: AppModel) {
        self.repo = repo
        let name = repo.id.split(separator: "/").last.map(String.init)?.replacingOccurrences(of: "-GGUF", with: "", options: .caseInsensitive) ?? repo.id
        super.init(id: repo.id, displayName: name, catalogId: nil, tags: [], contextLength: 0, app: app)
    }

    var subtitle: String { L.f("RepoStats", repo.downloads.formatted(), repo.likes.formatted()) }

    func load() async {
        isLoading = true
        status = L.s("LoadingVariants")
        do {
            let list = try await app.hub.ggufVariants(repo.id)
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
        didSet { if let repo = selectedRepo, repo.variants.isEmpty { Task { await repo.load() } } }
    }
    var query = ""
    var isSearching = false
    var tab = ModelsTab.catalog

    func attach(_ app: AppModel) {
        self.app = app
        catalog = app.catalog.models.map { CatalogItem($0, app: app) }
    }

    var selectedRepo: RepoItem? { results.first { $0.id == selectedRepoId } }

    var hardwareText: String {
        guard let hw = app?.hardware else { return "" }
        // Apple Silicon GPUs use unified memory, so show it as shared rather than VRAM.
        return hw.gpuBytes > 0
            ? L.f("HardwareSummaryShared", hw.gpuName, "\(hw.ramGB)")
            : L.f("HardwareSummaryCpu", "\(hw.ramGB)")
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

    func search() async {
        guard let app else { return }
        isSearching = true
        do {
            results = try await app.hub.search(query.trimmingCharacters(in: .whitespaces), filter: "gguf").map { RepoItem($0, app: app) }
            selectedRepoId = results.first?.id
        } catch {
            Log.error("HF search failed", error)
        }
        isSearching = false
    }
}
